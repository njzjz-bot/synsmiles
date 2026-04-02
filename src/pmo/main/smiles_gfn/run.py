import logging

# save original method so we don't lose other warnings
_original_warning = logging.Logger.warning

def _filter_fast_tfmr(self, msg, *args, **kwargs):
    """Swallow ONLY the MoLFormer CUDA-kernel fallback line."""
    if "Falling back to (slow) pytorch implementation" in str(msg):
        return
    _original_warning(self, msg, *args, **kwargs)

logging.Logger.warning = _filter_fast_tfmr

import os
import sys
import numpy as np
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('/'.join(path_here.rstrip('/').split('/')[:-2]))
from main.optimizer import BaseOptimizer
from utils import Variable, seq_to_smiles, unique
from model import RNN
from data_structs import Vocabulary, Experience, MolData
from priority_queue import MaxRewardPriorityQueue
import math
import torch
from rdkit import Chem
from tdc import Evaluator, Oracle
# from polyleven import levenshtein

from synth_utils import mutate, diff_mask_molformer
from replay_buffer import ReplayBuffer

from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


from rxnflow.envs.action import Protocol, RxnAction, RxnActionType
from rxnflow.envs.reaction import BiReaction, Reaction, UniReaction
from rxnflow.envs.retrosynthesis import MultiRetroSyntheticAnalyzer, RetroSynthesisTree
from pathlib import Path
from numpy.typing import NDArray

os.environ["TOKENIZERS_PARALLELISM"] = "false"


import itertools
import pickle
import pandas as pd
import wandb
import math
import torch.nn.functional as F
from time import perf_counter

from joblib import Parallel



def sanitize(smiles):
    canonicalized = []
    for s in smiles:
        try:
            canonicalized.append(Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True))
        except:
            pass
    return canonicalized


def compute_sequence_logprobs(model, seqs, pad_token_id):
    outputs = model(
        input_ids=seqs[:, :-1],
        attention_mask=(seqs[:, :-1] != pad_token_id).long(),
        labels=seqs[:, 1:],
    )
    shift_labels = seqs[:, 1:]
    log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
    seq_token_logprobs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
    seq_token_logprobs = seq_token_logprobs * (shift_labels != pad_token_id)
    return seq_token_logprobs.sum(dim=1)


def refresh_negative_difficulty(model, negative_replay, tokenizer, device, batch_size):
    if len(negative_replay.heap) == 0:
        return

    model.eval()
    with torch.no_grad():
        for start in range(0, len(negative_replay.heap), batch_size):
            batch = negative_replay.heap[start:start + batch_size]
            ids = [traj.ids for traj in batch]
            seqs = pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
            difficulties = compute_sequence_logprobs(model, seqs, tokenizer.pad_token_id)
            for traj, difficulty in zip(batch, difficulties.tolist()):
                traj.difficulty = float(difficulty)


class SynthesizabilityEvaluator:
    def __init__(self, num_workers: int = 4, invalid: float = 0.0, max_size: int = 50_000, use_retrosynthesis: bool = False, sa_threshold: float = 4.0, env: str = 'stock', max_steps: int = 2):
        if use_retrosynthesis:
            env_dir = Path('../../data/envs/' + env)
            reaction_template_path = env_dir / "template.txt"
            building_block_path = env_dir / "building_block.smi"
            pre_computed_building_block_mask_path = env_dir / "bb_mask.npy"
            pre_computed_building_block_fp_path = env_dir / "bb_fp_2_1024.npy"
            pre_computed_building_block_desc_path = env_dir / "bb_desc.npy"

            # set protocol
            protocols: list[Protocol] = []
            protocols.append(Protocol("stop", RxnActionType.Stop))
            protocols.append(Protocol("firstblock", RxnActionType.FirstBlock))
            with reaction_template_path.open() as file:
                reaction_templates = [ln.strip() for ln in file.readlines()]
            for i, template in enumerate(reaction_templates):
                _rxn = Reaction(template)
                if _rxn.num_reactants == 1:
                    rxn = UniReaction(template)
                    protocols.append(Protocol(f"unirxn{i}", RxnActionType.UniRxn, _rxn))
                elif _rxn.num_reactants == 2:
                    for block_is_first in [True, False]:  # this order is important
                        rxn = BiReaction(template, block_is_first)
                        protocols.append(Protocol(f"birxn{i}_{block_is_first}", RxnActionType.BiRxn, rxn))
            protocol_dict: dict[str, Protocol] = {protocol.name: protocol for protocol in protocols}
            stop_list: list[Protocol] = [p for p in protocols if p.action is RxnActionType.Stop]
            firstblock_list: list[Protocol] = [p for p in protocols if p.action is RxnActionType.FirstBlock]
            unirxn_list: list[Protocol] = [p for p in protocols if p.action is RxnActionType.UniRxn]
            birxn_list: list[Protocol] = [p for p in protocols if p.action is RxnActionType.BiRxn]

            # set building blocks
            with building_block_path.open() as file:
                lines = file.readlines()
                building_blocks = [ln.split()[0] for ln in lines]
                building_block_ids = [ln.strip().split()[1] for ln in lines]
            blocks: list[str] = building_blocks

            # set precomputed building block feature
            block_fp = np.load(pre_computed_building_block_fp_path)
            block_prop = np.load(pre_computed_building_block_desc_path)

            # set block mask
            block_mask: NDArray[np.bool_] = np.load(pre_computed_building_block_mask_path)
            birxn_block_indices: dict[str, np.ndarray] = {}
            for i, protocol in enumerate(birxn_list):
                birxn_block_indices[protocol.name] = np.where(block_mask[i])[0]
            num_total_actions = (
                1 + len(unirxn_list) + sum(indices.shape[0] for indices in birxn_block_indices.values())
            )

            self.retrosynthesis_analyzer = MultiRetroSyntheticAnalyzer.create(protocols, blocks, num_workers=num_workers)
        else:
            self.retrosynthesis_analyzer = None
            
        self.sa_threshold = sa_threshold
        self.invalid = invalid
        self._seen  = {}  # cache to avoid recomputing (using canonical SMILES)
        self._max   = max_size
        self._max_steps = max_steps

    def get_synthesis(self, smiles: str) -> RetroSynthesisTree | None:

        if self.retrosynthesis_analyzer:
            try:
                # Canonicalize the SMILES string before scoring
                mol = Chem.MolFromSmiles(smiles)
                canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            except:
                return None
                
            if canonical_smiles in self._seen:
                return self._seen[canonical_smiles]
            self.retrosynthesis_analyzer.submit(0, smiles, self._max_steps, [])
            _, retro_tree = self.retrosynthesis_analyzer.result()[0]
            if len(self._seen) < self._max:   # cheap cap to avoid runaway RAM
                self._seen[canonical_smiles] = retro_tree
            return retro_tree
        else:
            return None

    def score(self, smiles: str) -> float:
        try:
            # Canonicalize the SMILES string before scoring
            mol = Chem.MolFromSmiles(smiles)
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        except:
            return 0.0

        if canonical_smiles in self._seen:
            if self.retrosynthesis_analyzer:
                retro_tree = self._seen[canonical_smiles]
                return 1.0 if retro_tree else 0.0
            else:
                return float(self._seen[canonical_smiles] < self.sa_threshold)

        if self.retrosynthesis_analyzer:
            self.retrosynthesis_analyzer.submit(0, canonical_smiles, self._max_steps, [])
            _, retro_tree = self.retrosynthesis_analyzer.result()[0]
            score = 1.0 if retro_tree else 0.0
        else:
            try:
                sa = sascore.calculateScore(mol)  # sometimes, it raises an error: devided by zero (number of fingerprints is zero)
            except:
                sa = 10.0  #self.invalid
            score = float(sa < self.sa_threshold)

        if len(self._seen) < self._max:   # cheap cap to avoid runaway RAM
            self._seen[canonical_smiles] = retro_tree if self.retrosynthesis_analyzer else sa
        return score
    
    def score_batch(self, smiles_list: list[str]) -> list[float]:
        return [self.score(s) for s in smiles_list]



class SMILES_GFN_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "smiles_gfn"

    def _optimize(self, oracle, config):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.oracle.assign_evaluator(oracle)
        self.oracle.assign_synth_evaluator(SynthesizabilityEvaluator(use_retrosynthesis=config['use_retrosynthesis'], 
                                                                     sa_threshold=config['sa_threshold'], 
                                                                     env=config['retro_env'], 
                                                                     max_steps=config['max_retro_steps']))

        print(config)

        # path_here = os.path.dirname(os.path.realpath(__file__))
        # voc = Vocabulary(init_from_file=os.path.join(path_here, "data/Voc"))

        tokenizer = AutoTokenizer.from_pretrained("ibm-research/MoLFormer-XL-both-10pct", trust_remote_code=True)
        prior = AutoModelForCausalLM.from_pretrained("ibm-research/GP-MoLFormer-Uniq", trust_remote_code=True).to(device)
        model = AutoModelForCausalLM.from_pretrained("ibm-research/GP-MoLFormer-Uniq", trust_remote_code=True).to(device)
        prior.eval()
        

        # optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=config['learning_rate'])
        log_z = torch.nn.Parameter(torch.tensor([0.]).cuda())
        optimizer = torch.optim.Adam([{'params': model.parameters(), 
                                       'lr': config['learning_rate']},
                                      {'params': log_z, 
                                       'lr': config['lr_z']}])


        # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
        # occur more often (which means the agent can get biased towards them). Using experience replay is
        # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
        # experience = Experience(voc, max_size=config['num_keep'])
        # neg_experience = Experience(voc, max_size=config['num_keep'], fifo=True)
        replay = ReplayBuffer(eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else 1,
                                   pad_token_id = tokenizer.pad_token_id,
                                   max_size=config['num_keep'],
                                   evict_by='reward'
                                   )
        negative_replay = ReplayBuffer(eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else 1,
                                pad_token_id = tokenizer.pad_token_id,
                                max_size=config['num_keep'],
                                evict_by='oldest'
                                )

        if config['use_ga']:
            from ga_expert import GeneticOperatorHandler
            ga_handler = GeneticOperatorHandler(mutation_rate=0.01, population_size=64)
        
        print("Model initialized, starting training...")

        step = 0
        patience = 0
        prev_n_oracles = 0
        stuck_cnt = 0

        prev_best = 0.
        
        # eval_times = []
        # training_times = []
        # total_start = perf_counter()

        synth_history = []

        while True:

            if len(self.oracle) > 100:
                self.sort_buffer()
                old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
            else:
                old_scores = 0

            if step % config['experience_loop'] == 0 or len(replay.heap) < config['experience_replay']:
                # model.eval()
                training_mode = 'onpolicy'
                with torch.no_grad():
                    seqs = model.generate(
                        # input_ids,
                        do_sample=True,
                        max_length=config['max_length'],
                        num_return_sequences=config['batch_size'],
                        # temperature=self.sampling_temp,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True
                    )
                
                # Remove duplicates, ie only consider unique seqs
                # print(seqs.shape, len(unique(seqs)))
                unique_idxs = unique(seqs)
                seqs = seqs[unique_idxs]
                # agent_likelihood = agent_likelihood[unique_idxs]
                # entropy = entropy[unique_idxs]

                # Get prior likelihood and score
                smiles = tokenizer.batch_decode(seqs, skip_special_tokens=True)
                # if config['valid_only']:
                #     smiles = sanitize(smiles)

                synthesizability = torch.tensor(self.oracle.synth_evaluator.score_batch(smiles)).to(device)
                synth_indices = (synthesizability == 1.0)
                synth_history.append(synthesizability.mean().item())
                # print(f"step {step}: unique {len(unique_idxs)}, synthesizability {synthesizability.mean().item()}")

                if config['reshape_reward'] or config['filter_unsynthesizable']:
                    positive_indices = (synthesizability == 1).nonzero(as_tuple=True)[0]
                    positive_smiles = [smiles[i] for i in positive_indices.tolist()]
                    positive_scores = torch.tensor(self.oracle(positive_smiles)).to(device)  # np.array(self.oracle(positive_smiles))
                    replay.add_batch(seqs[positive_indices], positive_smiles, positive_scores, [1] * len(positive_smiles), masks=None, use_reshaped_reward=config['reshape_reward'])
                    
                    # not to count oracle calls for negative samples
                    negative_indices = (synthesizability == 0).nonzero(as_tuple=True)[0]
                    negative_smiles = [smiles[i] for i in negative_indices.tolist()]
                    negative_scores = torch.zeros(len(negative_smiles)).to(device)  # np.array([0 for _ in range(len(negative_smiles))])
                    negative_seqs = seqs[negative_indices]

                    if config['filter_unsynthesizable']:  # cannot be used with reward shaping
                        negative_replay.add_batch(negative_seqs, negative_smiles, negative_scores, [0] * len(negative_smiles), masks=None, use_reshaped_reward=config['reshape_reward'])
                        valid_seqs = seqs[positive_indices]
                        valid_smiles = positive_smiles
                        valid_scores = positive_scores
                        valid_synth = synthesizability[positive_indices]
                    else:
                        negative_replay.add_batch(negative_seqs, negative_smiles, negative_scores, [0] * len(negative_smiles), masks=None, use_reshaped_reward=config['reshape_reward'])
                        replay.add_batch(negative_seqs, negative_smiles, negative_scores, [0] * len(negative_smiles), masks=None, use_reshaped_reward=config['reshape_reward'])
                        valid_seqs = seqs
                        valid_smiles = smiles
                        valid_scores = torch.zeros(len(valid_smiles)).to(device)
                        valid_scores[positive_indices] = positive_scores
                        valid_synth = synthesizability
                else:
                    scores = torch.tensor(self.oracle(smiles)).to(device)  # np.array(self.oracle(smiles))
                    replay.add_batch(seqs, smiles, scores, synthesizability.tolist(), masks=None, use_reshaped_reward=config['reshape_reward'])
                    valid_seqs = seqs
                    valid_smiles = smiles
                    valid_scores = scores
                    valid_synth = synthesizability
                try:
                    print(f"step {step}: unique {len(unique_idxs)}, synthesizability {synthesizability.mean().item()}, max score: {valid_scores.max().item()}, avg score: {valid_scores.mean().item()},  pos replay {len(replay.heap)}, neg replay {len(negative_replay.heap)}")
                except:
                    print(f"step {step}: unique {len(unique_idxs)}, synthesizability {synthesizability.mean().item()},")


                if self.finish:
                    print('max oracle hit')
                    break
                
                if config['use_ga'] and len(self.oracle) >= 64:
                    # mutation_rate: 0.01
                    # population_size: 64
                    # offspring_size: 8
                    # ga_generations: 2
                    self.oracle.sort_buffer()
                    pop_smis, pop_scores = tuple(map(list, zip(*[(smi, elem[0]) for (smi, elem) in self.oracle.mol_buffer.items()])))
                    mating_pool = (pop_smis[:config['num_keep']], pop_scores[:config['num_keep']])
                    for g in range(2):
                        child_smis, _, pop_smis, pop_scores = ga_handler.query(
                            query_size=32, mating_pool=mating_pool, pool=None, 
                            rank_coefficient=0.01,
                        )
                        child_synth = torch.tensor(self.oracle.synth_evaluator.score_batch(child_smis)).to(device)
                        child_seqs = tokenizer.batch_encode_plus(child_smis, add_special_tokens=True, padding=True, max_length=config['max_length'], return_tensors='pt')["input_ids"].to(device)
                        child_scores= torch.zeros(len(child_smis)).to(device)

                        if child_synth.sum() > 0:
                            child_pos_indices = (child_synth == 1).nonzero(as_tuple=True)[0]
                            child_pos_smiles = [child_smis[i] for i in child_pos_indices.tolist()]
                            child_pos_scores = torch.tensor(self.oracle(child_pos_smiles)).to(device)
                            child_pos_seqs = child_seqs[child_pos_indices]
                            child_scores[child_pos_indices] = child_pos_scores
                            replay.add_batch(child_pos_seqs, child_pos_smiles, child_pos_scores, [1] * len(child_pos_smiles), masks=None, use_reshaped_reward=config['reshape_reward'])
                        else:
                            continue

                        try:
                            print(f"step {step}: GA {g}: synth {child_synth.sum().item()}, max {child_scores.max().item()}, mean {child_scores[child_synth.bool()].mean().item()}")
                        except:
                            print(f"step {step}: GA {g}: synth {child_synth.sum().item()}, max {child_scores.max().item()}, mean {child_scores.mean().item()}")

                        negative_indices = (child_synth == 0).nonzero(as_tuple=True)[0]
                        negative_smiles = [child_smis[i] for i in negative_indices.tolist()]
                        negative_seqs = child_seqs[negative_indices]
                        negative_scores = torch.zeros(len(negative_smiles)).to(device)
                        if config['reshape_reward']:
                            replay.add_batch(negative_seqs, negative_smiles, negative_scores, [0] * len(negative_smiles), masks=None, use_reshaped_reward=config['reshape_reward'])
                        elif negative_replay:
                            negative_replay.add_batch(negative_seqs, negative_smiles, negative_scores, [0] * len(negative_smiles), masks=None, use_reshaped_reward=config['reshape_reward'])

                        # mating_pool = (pop_smis+child_smis, pop_scores+child_scores.tolist())  # or filtered?
                        if child_synth.sum() > 0:
                            mating_pool = (pop_smis+child_pos_smiles, pop_scores+child_pos_scores.tolist())  # or filtered?

            else:  # replay training
                training_mode = 'replay'
                # if len(replay.heap) < config['experience_replay'] or len(negative_replay.heap) < config['experience_replay']:
                #     continue
                valid_inputs, valid_scores = replay.sample(config['experience_replay'], device, reward_prioritized=True, rank_based=config['rank_based'], replace=config['replace'])
                valid_seqs = valid_inputs["input_ids"]
                valid_smiles = [tokenizer.decode(seq, skip_special_tokens=True) for seq in valid_seqs]
                valid_synth = torch.tensor(self.oracle.synth_evaluator.score_batch(valid_smiles)).to(device)  # won't be slow (cached)

                if config['aux_loss'] != "none":
                    if config.get('neg_sampling_strategy', 'uniform') == 'difficulty_rank':
                        refresh_interval = config.get('neg_difficulty_refresh', 10)
                        if step % refresh_interval == 0:
                            refresh_negative_difficulty(
                                model,
                                negative_replay,
                                tokenizer,
                                device,
                                config.get('neg_difficulty_batch_size', 128),
                            )
                        neg_inputs, negative_scores = negative_replay.sample(
                            config['experience_replay'],
                            device,
                            reward_prioritized=True,
                            rank_based=True,
                            replace=True,
                            score_attr="difficulty",
                        )
                    else:
                        neg_inputs, negative_scores = negative_replay.sample(config['experience_replay'], device)
                    negative_seqs = neg_inputs["input_ids"]
                    negative_smiles = [tokenizer.decode(seq, skip_special_tokens=True) for seq in negative_seqs]

            if self.finish:
                print('max oracle hit')
                break

            
            if (config['filter_unsynthesizable'] or config['reshape_reward']) and (valid_synth.sum() < 4 or negative_seqs.shape[0] < 4):
                # if len(replay.heap) > config['batch_size']:
                #     valid_inputs, valid_scores = replay.sample(config['batch_size'], device)
                #     valid_seqs = valid_inputs["input_ids"]
                #     valid_smiles = [tokenizer.decode(seq, skip_special_tokens=True) for seq in valid_seqs]
                #     valid_synth = torch.tensor(self.oracle.synth_evaluator.score_batch(valid_smiles)).to(device)  # won't be slow (cached)
                # else:
                step += 1
                continue
            else:
                aux_loss = torch.zeros((), device=device)

            # if negative_seqs.shape[0] < 4:
            #     if len(negative_replay.heap) > config['batch_size']:
            #         neg_inputs, negative_scores = negative_replay.sample(config['batch_size'], device)
            #         negative_seqs = neg_inputs["input_ids"]
            #         negative_smiles = [tokenizer.decode(seq, skip_special_tokens=True) for seq in negative_seqs]
            #         negative_synth = torch.zeros(len(negative_smiles)).to(device)
            #     else:
            #         step += 1
            #         continue

            # early stopping
            if len(self.oracle) > 1000:
                self.sort_buffer()
                new_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
                if new_scores == old_scores:
                    patience += 1
                    if patience >= self.args.patience * config['experience_loop']:
                        self.log_intermediate(finish=True)
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0
            
            # early stopping2
            if prev_n_oracles < len(self.oracle):
                stuck_cnt = 0
            else:
                stuck_cnt += 1
                if stuck_cnt >= 10 * config['experience_loop']:
                    self.log_intermediate(finish=True)
                    print('cannot find new molecules, abort ...... ')
                    break
            
            prev_n_oracles = len(self.oracle)

            # onpolicy training
            model.train()

            outputs = model(
                input_ids=valid_seqs[:, :-1],
                attention_mask=(valid_seqs[:, :-1] != tokenizer.pad_token_id).long(),
                labels=valid_seqs[:, 1:],
            )

            shift_labels = valid_seqs[:, 1:]
            logits = outputs.logits  # (batch, seq_len, vocab)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            seq_token_logprobs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
            seq_token_logprobs = seq_token_logprobs * (shift_labels != tokenizer.pad_token_id)
            seq_logprobs = seq_token_logprobs.sum(dim=1)

            with torch.no_grad():
                prior_logits = prior(
                    input_ids=valid_seqs[:, :-1],
                    attention_mask=(valid_seqs[:, :-1] != tokenizer.pad_token_id).long(),
                    labels=valid_seqs[:, 1:],
                ).logits
                prior_log_probs = torch.nn.functional.log_softmax(prior_logits, dim=-1)
                prior_seq_token_logprobs = torch.gather(prior_log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
                prior_seq_token_logprobs = prior_seq_token_logprobs * (shift_labels != tokenizer.pad_token_id)
                prior_seq_logprobs = prior_seq_token_logprobs.sum(dim=1).detach()

            forward_flow = seq_logprobs + log_z
            backward_flow = prior_seq_logprobs + config['beta'] * valid_scores
            loss = torch.pow(forward_flow - backward_flow, 2).mean()

            if config['separate_update'] or training_mode == 'onpolicy':
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_norm'])
                optimizer.step()
                
            if config['aux_loss'] != "none" and len(valid_smiles) > 0 and training_mode == 'replay':
                if config['without_mutation']:
                    aux_loss = torch.zeros((), device=device)
                    pos_seq_logprobs = seq_logprobs
                else:
                    mutated_neg_smiles, mutated_seqs= [], []
                    paired = []
                    for s, f in zip(valid_smiles, valid_synth):  # TODO: check when using RS + Aux
                        if not f:
                            continue
                        mutated = mutate(s, self.oracle.synth_evaluator)  # TODO: check if this is correct
                        if mutated:
                            try:
                                mutated_info = diff_mask_molformer(s, mutated, tokenizer)
                            except:
                                paired.append(False)
                                continue
                            paired.append(True)
                            mutated_neg_smiles.append(mutated)
                            mutated_seqs.append(torch.tensor(mutated_info['input_ids']))
                        else:
                            paired.append(False)
                    # mutated_neg_seqs = self.tokenizer.batch_encode_plus(mutated_neg_smiles, add_special_tokens=True, padding=True, max_length=self.max_length, return_tensors='pt')["input_ids"].to(self.device)
                    if len(mutated_seqs) > 0:
                        mutated_neg_seqs = pad_sequence(mutated_seqs, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)

                        pos_logits = model(
                            input_ids=valid_seqs[:, :-1],
                            attention_mask=(valid_seqs[:, :-1] != tokenizer.pad_token_id).long(),
                            labels=valid_seqs[:, 1:],
                        ).logits

                        pos_log_probs = torch.nn.functional.log_softmax(pos_logits, dim=-1)
                        pos_seq_token_logprobs = torch.gather(pos_log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
                        pos_seq_token_logprobs = pos_seq_token_logprobs * (shift_labels != tokenizer.pad_token_id)
                        pos_seq_logprobs = pos_seq_token_logprobs.sum(dim=1)
                        if config['reshape_reward']:
                            pos_seq_logprobs = pos_seq_logprobs[valid_synth.bool()]

                        mut_logits = model(
                            input_ids=mutated_neg_seqs[:, :-1],
                            attention_mask=(mutated_neg_seqs[:, :-1] != tokenizer.pad_token_id).long(),
                            labels=mutated_neg_seqs[:, 1:],
                        ).logits

                        mut_shift_labels = mutated_neg_seqs[:, 1:]
                        mut_log_probs = torch.nn.functional.log_softmax(mut_logits, dim=-1)
                        mut_seq_token_logprobs = torch.gather(mut_log_probs, 2, mut_shift_labels.unsqueeze(-1)).squeeze(-1)
                        mut_seq_token_logprobs = mut_seq_token_logprobs * (mut_shift_labels != tokenizer.pad_token_id)
                        mut_seq_logprobs = mut_seq_token_logprobs.sum(dim=1)
                        
                        paired_mask = torch.tensor(paired).to(device)
                        
                        if config['pairwise_mutated']:
                            aux_loss = -(pos_seq_logprobs[paired_mask] - torch.logaddexp(pos_seq_logprobs[paired_mask], mut_seq_logprobs)).mean()
                        else:
                            mutated_log_sum = torch.logsumexp(mut_seq_logprobs, dim=0) - math.log(max(mut_seq_logprobs.numel(), 1.0))
                            aux_loss = -(pos_seq_logprobs[paired_mask] - torch.logaddexp(pos_seq_logprobs[paired_mask], mutated_log_sum)).mean()

                    else:
                        aux_loss = torch.zeros((), device=device)
                # if len(negative_seqs) > 8:
                neg_logits = model(
                    input_ids=negative_seqs[:, :-1],
                    attention_mask=(negative_seqs[:, :-1] != tokenizer.pad_token_id).long(),
                    labels=negative_seqs[:, 1:],
                ).logits

                neg_shift_labels = negative_seqs[:, 1:]
                neg_log_probs = torch.nn.functional.log_softmax(neg_logits, dim=-1)
                neg_seq_token_logprobs = torch.gather(neg_log_probs, 2, neg_shift_labels.unsqueeze(-1)).squeeze(-1)
                neg_seq_token_logprobs = neg_seq_token_logprobs * (neg_shift_labels != tokenizer.pad_token_id)
                neg_seq_logprobs = neg_seq_token_logprobs.sum(dim=1)

                neg_log_sum = torch.logsumexp(neg_seq_logprobs, dim=0) - math.log(max(neg_seq_logprobs.numel(), 1.0))
                aux_loss += -(pos_seq_logprobs - torch.logaddexp(pos_seq_logprobs, neg_log_sum)).mean()

                if config['separate_update']:
                    optimizer.zero_grad()
                    (config['aux_coefficient'] * aux_loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_norm'])
                    optimizer.step()
                else:
                    optimizer.zero_grad()
                    (loss + config['aux_coefficient'] * aux_loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_norm'])
                    optimizer.step()

            # print(f"Step {step}: {loss.item()}, {aux_loss.item()}")
            step += 1

        try:
            wandb.log({
                    'final/synth_history_last5_mean': np.mean(synth_history[-5:]),
                    'final/synth_history_mean': np.mean(synth_history),
                    })
        except:
            pass
