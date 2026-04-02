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



class REINVENT_RS_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "reinvent_rs"

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

                # model.eval()
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
            

            if config['reshape_reward']:
                positive_indices = (synthesizability == 1).nonzero(as_tuple=True)[0]
                positive_smiles = [smiles[i] for i in positive_indices.tolist()]
                positive_scores = torch.tensor(self.oracle(positive_smiles)).to(device)  # np.array(self.oracle(positive_smiles))
                replay.add_batch(seqs[positive_indices], positive_smiles, positive_scores, [1] * len(positive_smiles), masks=None, use_reshaped_reward=config['reshape_reward'])
                
                # not to count oracle calls for negative samples
                negative_indices = (synthesizability == 0).nonzero(as_tuple=True)[0]
                negative_smiles = [smiles[i] for i in negative_indices.tolist()]
                negative_scores = torch.zeros(len(negative_smiles)).to(device)  # np.array([0 for _ in range(len(negative_smiles))])
                negative_seqs = seqs[negative_indices]

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
            

            # early stopping
            if len(self.oracle) > 1000:
                self.sort_buffer()
                new_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
                if new_scores == old_scores:
                    patience += 1
                    if patience >= self.args.patience:
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
                if stuck_cnt >= 10:
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

            augmented_likelihood = prior_seq_logprobs + 500 * valid_scores  # set sigma to 500 according to REINVENT implemented in PMO
            loss = torch.pow((augmented_likelihood - seq_logprobs), 2)

            if len(replay.heap) >= config['experience_replay']:
                buf_inputs, buf_reward = replay.sample(config['experience_replay'], device, reward_prioritized=True, replace=True)
                buf_seqs = buf_inputs["input_ids"]
                buf_smis = [tokenizer.decode(seq, skip_special_tokens=True) for seq in buf_seqs]

                buf_logits = model(
                    input_ids=buf_seqs[:, :-1],
                    attention_mask=(buf_seqs[:, :-1] != tokenizer.pad_token_id).long(),
                    labels=buf_seqs[:, 1:],
                ).logits

                # Fix shape mismatch for torch.gather by aligning shift_logits and shift_labels
                buf_shift_labels = buf_seqs[:, 1:]
                buf_log_probs = torch.nn.functional.log_softmax(buf_logits, dim=-1)
                buf_seq_token_logprobs = torch.gather(buf_log_probs, 2, buf_shift_labels.unsqueeze(-1)).squeeze(-1)
                buf_seq_token_logprobs = buf_seq_token_logprobs * (buf_shift_labels != tokenizer.pad_token_id)
                buf_seq_logprobs = buf_seq_token_logprobs.sum(dim=1)

                # prior likelihood
                with torch.no_grad():
                    prior_logits = prior(input_ids=buf_seqs[:, :-1], attention_mask=(buf_seqs[:, :-1] != tokenizer.pad_token_id).long(), labels=buf_seqs[:, 1:]).logits
                    prior_log_probs = torch.nn.functional.log_softmax(prior_logits, dim=-1)

                    prior_seq_token_logprobs = torch.gather(prior_log_probs, 2, buf_shift_labels.unsqueeze(-1)).squeeze(-1)
                    prior_seq_token_logprobs = prior_seq_token_logprobs * (buf_shift_labels != tokenizer.pad_token_id)
                    prior_seq_logprobs = prior_seq_token_logprobs.sum(dim=1).detach()

                if config['reshape_reward']:
                    buf_synth = torch.tensor(self.oracle.synth_evaluator.score_batch(buf_smis)).to(device)
                    buf_score = buf_reward * buf_synth
                else:
                    buf_score = buf_reward

                buf_augmented_likelihood = prior_seq_logprobs + 500 * buf_score  # set sigma to 500 according to REINVENT implemented in PMO
                replay_loss = torch.pow((buf_augmented_likelihood - buf_seq_logprobs), 2)

                loss = torch.cat([loss, replay_loss], dim=0).mean()
            else:
                loss = loss.mean()

            # Add regularizer that penalizes high likelihood for the entire sequence
            loss_p = - (1 / seq_logprobs).mean()
            loss = loss + 5 * 1e3 * loss_p
            
            optimizer.zero_grad()
            loss.backward()
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