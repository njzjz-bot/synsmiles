import copy
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append("/".join(path_here.rstrip("/").split("/")[:-2]))

from main.optimizer import BaseOptimizer
from replay_buffer import ReplayBuffer
from synth_utils import diff_mask_molformer, mutate
from utils import unique

from main.smiles_gfn.run import compute_sequence_logprobs, refresh_negative_difficulty, SynthesizabilityEvaluator
from main.smiles_gfn.run_sac_base import (
    SharedSACModel,
    _compute_sac_loss,
    _generate_sac_sequences,
    _soft_update,
)
from main.smiles_gfn.run_sql_base import _build_seq_mask, _get_special_token_ids


class SAC_S3_Optimizer(BaseOptimizer):
    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "sac_s3"

    def _optimize(self, oracle, config):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.oracle.assign_evaluator(oracle)
        self.oracle.assign_synth_evaluator(
            SynthesizabilityEvaluator(
                use_retrosynthesis=config["use_retrosynthesis"],
                sa_threshold=config["sa_threshold"],
                env=config["retro_env"],
                max_steps=config["max_retro_steps"],
            )
        )

        print(config)

        tokenizer = AutoTokenizer.from_pretrained("ibm-research/MoLFormer-XL-both-10pct", trust_remote_code=True)
        prior = AutoModelForCausalLM.from_pretrained("ibm-research/GP-MoLFormer-Uniq", trust_remote_code=True).to(device)
        model = SharedSACModel("ibm-research/GP-MoLFormer-Uniq").to(device)
        target_model = copy.deepcopy(model).to(device)
        prior.eval()
        target_model.eval()
        for param in prior.parameters():
            param.requires_grad = False
        for param in target_model.parameters():
            param.requires_grad = False

        bos_token_id, eos_token_id, pad_token_id = _get_special_token_ids(tokenizer)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

        replay = ReplayBuffer(
            eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else 1,
            pad_token_id=tokenizer.pad_token_id,
            max_size=config["num_keep"],
            evict_by="reward",
        )
        negative_replay = ReplayBuffer(
            eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else 1,
            pad_token_id=tokenizer.pad_token_id,
            max_size=config["num_keep"],
            evict_by="oldest",
        )

        if config["use_ga"]:
            from ga_expert import GeneticOperatorHandler

            ga_handler = GeneticOperatorHandler(mutation_rate=0.01, population_size=64)

        print("Model initialized, starting SAC-S3 training...")

        step = 0
        patience = 0
        prev_n_oracles = 0
        stuck_cnt = 0
        synth_history = []

        while True:
            if len(self.oracle) > 100:
                self.sort_buffer()
                old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
            else:
                old_scores = 0

            negative_seqs = torch.empty((0, 1), dtype=torch.long, device=device)

            if step % config["experience_loop"] == 0 or len(replay.heap) < config["experience_replay"]:
                training_mode = "onpolicy"
                seqs, seq_mask = _generate_sac_sequences(
                    model,
                    prior,
                    tokenizer,
                    batch_size=config["batch_size"],
                    max_length=config["max_length"],
                    device=device,
                )

                unique_idxs = unique(seqs)
                seqs = seqs[unique_idxs]
                seq_mask = seq_mask[unique_idxs]
                smiles = tokenizer.batch_decode(seqs, skip_special_tokens=True)

                synthesizability = torch.tensor(self.oracle.synth_evaluator.score_batch(smiles), device=device)
                synth_history.append(float(synthesizability.mean().item()))

                if config["reshape_reward"] or config["filter_unsynthesizable"]:
                    positive_indices = (synthesizability == 1).nonzero(as_tuple=True)[0]
                    positive_smiles = [smiles[i] for i in positive_indices.tolist()]
                    if len(positive_smiles) > 0:
                        positive_scores = torch.tensor(self.oracle(positive_smiles), device=device)
                    else:
                        positive_scores = torch.empty(0, device=device)
                    positive_masks = [m.clone().cpu() for m in seq_mask[positive_indices]]

                    replay.add_batch(
                        seqs[positive_indices],
                        positive_smiles,
                        positive_scores,
                        [1] * len(positive_smiles),
                        masks=positive_masks,
                        use_reshaped_reward=config["reshape_reward"],
                    )

                    negative_indices = (synthesizability == 0).nonzero(as_tuple=True)[0]
                    negative_smiles = [smiles[i] for i in negative_indices.tolist()]
                    negative_scores = torch.zeros(len(negative_smiles), device=device)
                    negative_seqs = seqs[negative_indices]
                    negative_masks = [m.clone().cpu() for m in seq_mask[negative_indices]]

                    negative_replay.add_batch(
                        negative_seqs,
                        negative_smiles,
                        negative_scores,
                        [0] * len(negative_smiles),
                        masks=negative_masks,
                        use_reshaped_reward=config["reshape_reward"],
                    )

                    if config["filter_unsynthesizable"]:
                        valid_seqs = seqs[positive_indices]
                        valid_seq_mask = seq_mask[positive_indices]
                        valid_smiles = positive_smiles
                        valid_scores = positive_scores
                        valid_synth = synthesizability[positive_indices]
                    else:
                        replay.add_batch(
                            negative_seqs,
                            negative_smiles,
                            negative_scores,
                            [0] * len(negative_smiles),
                            masks=negative_masks,
                            use_reshaped_reward=config["reshape_reward"],
                        )
                        valid_seqs = seqs
                        valid_seq_mask = seq_mask
                        valid_smiles = smiles
                        valid_scores = torch.zeros(len(valid_smiles), device=device)
                        valid_scores[positive_indices] = positive_scores
                        valid_synth = synthesizability
                else:
                    scores = torch.tensor(self.oracle(smiles), device=device)
                    replay.add_batch(
                        seqs,
                        smiles,
                        scores,
                        synthesizability.tolist(),
                        masks=[m.clone().cpu() for m in seq_mask],
                        use_reshaped_reward=config["reshape_reward"],
                    )
                    valid_seqs = seqs
                    valid_seq_mask = seq_mask
                    valid_smiles = smiles
                    valid_scores = scores
                    valid_synth = synthesizability

                try:
                    print(
                        f"step {step}: unique {len(unique_idxs)}, "
                        f"synthesizability {synthesizability.mean().item()}, "
                        f"max score: {valid_scores.max().item()}, avg score: {valid_scores.mean().item()}, "
                        f"pos replay {len(replay.heap)}, neg replay {len(negative_replay.heap)}"
                    )
                except Exception:
                    print(f"step {step}: unique {len(unique_idxs)}, synthesizability {synthesizability.mean().item()},")

                if self.finish:
                    print("max oracle hit")
                    break

                if config["use_ga"] and len(self.oracle) >= 64:
                    self.oracle.sort_buffer()
                    pop_smis, pop_scores = tuple(
                        map(list, zip(*[(smi, elem[0]) for (smi, elem) in self.oracle.mol_buffer.items()]))
                    )
                    mating_pool = (pop_smis[: config["num_keep"]], pop_scores[: config["num_keep"]])
                    for g in range(2):
                        child_smis, _, pop_smis, pop_scores = ga_handler.query(
                            query_size=32,
                            mating_pool=mating_pool,
                            pool=None,
                            rank_coefficient=0.01,
                        )
                        child_synth = torch.tensor(self.oracle.synth_evaluator.score_batch(child_smis), device=device)
                        child_seqs = tokenizer.batch_encode_plus(
                            child_smis,
                            add_special_tokens=True,
                            padding=True,
                            max_length=config["max_length"],
                            return_tensors="pt",
                        )["input_ids"].to(device)
                        child_seq_mask = _build_seq_mask(child_seqs, eos_token_id)
                        child_scores = torch.zeros(len(child_smis), device=device)

                        if child_synth.sum() > 0:
                            child_pos_indices = (child_synth == 1).nonzero(as_tuple=True)[0]
                            child_pos_smiles = [child_smis[i] for i in child_pos_indices.tolist()]
                            child_pos_scores = torch.tensor(self.oracle(child_pos_smiles), device=device)
                            child_pos_seqs = child_seqs[child_pos_indices]
                            child_pos_masks = [m.clone().cpu() for m in child_seq_mask[child_pos_indices]]
                            child_scores[child_pos_indices] = child_pos_scores
                            replay.add_batch(
                                child_pos_seqs,
                                child_pos_smiles,
                                child_pos_scores,
                                [1] * len(child_pos_smiles),
                                masks=child_pos_masks,
                                use_reshaped_reward=config["reshape_reward"],
                            )
                        else:
                            continue

                        try:
                            print(
                                f"step {step}: GA {g}: synth {child_synth.sum().item()}, "
                                f"max {child_scores.max().item()}, "
                                f"mean {child_scores[child_synth.bool()].mean().item()}"
                            )
                        except Exception:
                            print(
                                f"step {step}: GA {g}: synth {child_synth.sum().item()}, "
                                f"max {child_scores.max().item()}, mean {child_scores.mean().item()}"
                            )

                        negative_indices = (child_synth == 0).nonzero(as_tuple=True)[0]
                        negative_smiles = [child_smis[i] for i in negative_indices.tolist()]
                        negative_seqs = child_seqs[negative_indices]
                        negative_masks = [m.clone().cpu() for m in child_seq_mask[negative_indices]]
                        negative_scores = torch.zeros(len(negative_smiles), device=device)
                        if config["reshape_reward"]:
                            replay.add_batch(
                                negative_seqs,
                                negative_smiles,
                                negative_scores,
                                [0] * len(negative_smiles),
                                masks=negative_masks,
                                use_reshaped_reward=config["reshape_reward"],
                            )
                        else:
                            negative_replay.add_batch(
                                negative_seqs,
                                negative_smiles,
                                negative_scores,
                                [0] * len(negative_smiles),
                                masks=negative_masks,
                                use_reshaped_reward=config["reshape_reward"],
                            )

                        if child_synth.sum() > 0:
                            mating_pool = (pop_smis + child_pos_smiles, pop_scores + child_pos_scores.tolist())

            else:
                training_mode = "replay"
                valid_inputs, valid_scores = replay.sample(
                    config["experience_replay"],
                    device,
                    reward_prioritized=True,
                    rank_based=config["rank_based"],
                    replace=config["replace"],
                )
                valid_seqs = valid_inputs["input_ids"]
                valid_seq_mask = valid_inputs["mutation_mask"].bool()
                valid_smiles = [tokenizer.decode(seq, skip_special_tokens=True) for seq in valid_seqs]
                valid_synth = torch.tensor(self.oracle.synth_evaluator.score_batch(valid_smiles), device=device)

                if config["aux_loss"] != "none":
                    if config.get("neg_sampling_strategy", "uniform") == "difficulty_rank":
                        refresh_interval = config.get("neg_difficulty_refresh", 10)
                        if step % refresh_interval == 0:
                            refresh_negative_difficulty(
                                model.actor_model,
                                negative_replay,
                                tokenizer,
                                device,
                                config.get("neg_difficulty_batch_size", 128),
                            )
                        neg_inputs, _ = negative_replay.sample(
                            config["experience_replay"],
                            device,
                            reward_prioritized=True,
                            rank_based=True,
                            replace=True,
                            score_attr="difficulty",
                        )
                    else:
                        neg_inputs, _ = negative_replay.sample(config["experience_replay"], device)
                    negative_seqs = neg_inputs["input_ids"]
                    negative_smiles = [tokenizer.decode(seq, skip_special_tokens=True) for seq in negative_seqs]

            if self.finish:
                print("max oracle hit")
                break

            if (config["filter_unsynthesizable"] or config["reshape_reward"]) and (
                valid_synth.sum() < 4 or negative_seqs.shape[0] < 4
            ):
                step += 1
                continue
            else:
                aux_loss = torch.zeros((), device=device)

            if len(self.oracle) > 1000:
                self.sort_buffer()
                new_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
                if new_scores == old_scores:
                    patience += 1
                    if patience >= self.args.patience * config["experience_loop"]:
                        self.log_intermediate(finish=True)
                        print("convergence criteria met, abort ...... ")
                        break
                else:
                    patience = 0

            if prev_n_oracles < len(self.oracle):
                stuck_cnt = 0
            else:
                stuck_cnt += 1
                if stuck_cnt >= 10 * config["experience_loop"]:
                    self.log_intermediate(finish=True)
                    print("cannot find new molecules, abort ...... ")
                    break

            prev_n_oracles = len(self.oracle)

            model.train()
            actor_loss, critic_loss, sac_stats = _compute_sac_loss(
                model,
                target_model,
                prior,
                valid_seqs,
                valid_seq_mask,
                valid_scores,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                beta=float(config["beta"]),
            )
            loss = actor_loss + critic_loss
            stepped = False

            if config["separate_update"] or training_mode == "onpolicy":
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["max_norm"])
                optimizer.step()
                _soft_update(target_model, model, tau=float(config["target_tau"]))
                stepped = True

            valid_seq_logprobs = compute_sequence_logprobs(model.actor_model, valid_seqs, tokenizer.pad_token_id)

            if config["aux_loss"] != "none" and len(valid_smiles) > 0 and training_mode == "replay":
                if config["without_mutation"]:
                    aux_loss = torch.zeros((), device=device)
                    pos_seq_logprobs = valid_seq_logprobs
                else:
                    mutated_neg_smiles, mutated_seqs = [], []
                    paired = []
                    for s, f in zip(valid_smiles, valid_synth):
                        if not f:
                            continue
                        mutated = mutate(s, self.oracle.synth_evaluator)
                        if mutated:
                            try:
                                mutated_info = diff_mask_molformer(s, mutated, tokenizer)
                            except Exception:
                                paired.append(False)
                                continue
                            paired.append(True)
                            mutated_neg_smiles.append(mutated)
                            mutated_seqs.append(torch.tensor(mutated_info["input_ids"]))
                        else:
                            paired.append(False)
                    if len(mutated_seqs) > 0:
                        mutated_neg_seqs = pad_sequence(
                            mutated_seqs,
                            batch_first=True,
                            padding_value=tokenizer.pad_token_id,
                        ).to(device)

                        pos_logits = model.actor_model(
                            input_ids=valid_seqs[:, :-1],
                            attention_mask=(valid_seqs[:, :-1] != tokenizer.pad_token_id).long(),
                            labels=valid_seqs[:, 1:],
                        ).logits
                        shift_labels = valid_seqs[:, 1:]
                        pos_log_probs = F.log_softmax(pos_logits, dim=-1)
                        pos_seq_token_logprobs = torch.gather(pos_log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
                        pos_seq_token_logprobs = pos_seq_token_logprobs * (shift_labels != tokenizer.pad_token_id)
                        pos_seq_logprobs = pos_seq_token_logprobs.sum(dim=1)
                        if config["reshape_reward"]:
                            pos_seq_logprobs = pos_seq_logprobs[valid_synth.bool()]

                        mut_logits = model.actor_model(
                            input_ids=mutated_neg_seqs[:, :-1],
                            attention_mask=(mutated_neg_seqs[:, :-1] != tokenizer.pad_token_id).long(),
                            labels=mutated_neg_seqs[:, 1:],
                        ).logits

                        mut_shift_labels = mutated_neg_seqs[:, 1:]
                        mut_log_probs = F.log_softmax(mut_logits, dim=-1)
                        mut_seq_token_logprobs = torch.gather(mut_log_probs, 2, mut_shift_labels.unsqueeze(-1)).squeeze(-1)
                        mut_seq_token_logprobs = mut_seq_token_logprobs * (mut_shift_labels != tokenizer.pad_token_id)
                        mut_seq_logprobs = mut_seq_token_logprobs.sum(dim=1)

                        paired_mask = torch.tensor(paired, device=device)

                        if config["pairwise_mutated"]:
                            aux_loss = -(
                                pos_seq_logprobs[paired_mask]
                                - torch.logaddexp(pos_seq_logprobs[paired_mask], mut_seq_logprobs)
                            ).mean()
                        else:
                            mutated_log_sum = torch.logsumexp(mut_seq_logprobs, dim=0) - math.log(
                                max(mut_seq_logprobs.numel(), 1.0)
                            )
                            aux_loss = -(
                                pos_seq_logprobs[paired_mask]
                                - torch.logaddexp(pos_seq_logprobs[paired_mask], mutated_log_sum)
                            ).mean()
                    else:
                        aux_loss = torch.zeros((), device=device)

                neg_logits = model.actor_model(
                    input_ids=negative_seqs[:, :-1],
                    attention_mask=(negative_seqs[:, :-1] != tokenizer.pad_token_id).long(),
                    labels=negative_seqs[:, 1:],
                ).logits

                neg_shift_labels = negative_seqs[:, 1:]
                neg_log_probs = F.log_softmax(neg_logits, dim=-1)
                neg_seq_token_logprobs = torch.gather(neg_log_probs, 2, neg_shift_labels.unsqueeze(-1)).squeeze(-1)
                neg_seq_token_logprobs = neg_seq_token_logprobs * (neg_shift_labels != tokenizer.pad_token_id)
                neg_seq_logprobs = neg_seq_token_logprobs.sum(dim=1)

                neg_log_sum = torch.logsumexp(neg_seq_logprobs, dim=0) - math.log(max(neg_seq_logprobs.numel(), 1.0))
                aux_loss += -(pos_seq_logprobs - torch.logaddexp(pos_seq_logprobs, neg_log_sum)).mean()

                if config["separate_update"]:
                    optimizer.zero_grad()
                    (config["aux_coefficient"] * aux_loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["max_norm"])
                    optimizer.step()
                    _soft_update(target_model, model, tau=float(config["target_tau"]))
                    stepped = True
                else:
                    optimizer.zero_grad()
                    (loss + config["aux_coefficient"] * aux_loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["max_norm"])
                    optimizer.step()
                    _soft_update(target_model, model, tau=float(config["target_tau"]))
                    stepped = True

            try:
                print(
                    f"sac_s3 {training_mode} actor {actor_loss.item():.4f}, critic {critic_loss.item():.4f}, "
                    f"kl {sac_stats['kl_mean']:.4f}, log_ratio {sac_stats['sample_log_ratio_mean']:.4f}, "
                    f"prior_lp {sac_stats['prior_action_logprob_mean']:.4f}"
                )
            except Exception:
                pass

            try:
                wandb.log(
                    {
                        "sac_s3/actor_loss": float(actor_loss.item()),
                        "sac_s3/critic_loss": float(critic_loss.item()),
                        "sac_s3/loss": float(loss.item()),
                        "sac_s3/aux_loss": float(aux_loss.item()),
                        "sac_s3/training_mode_is_replay": float(training_mode == "replay"),
                        "sac_s3/stepped": float(stepped),
                        "sac_s3/pos_replay_size": len(replay.heap),
                        "sac_s3/neg_replay_size": len(negative_replay.heap),
                        "sac_s3/synth_rate": float(synth_history[-1]) if synth_history else 0.0,
                        "sac_s3/avg_score": float(valid_scores.mean().item()) if valid_scores.numel() > 0 else 0.0,
                        "sac_s3/kl_mean": sac_stats["kl_mean"],
                        "sac_s3/sample_log_ratio_mean": sac_stats["sample_log_ratio_mean"],
                        "sac_s3/policy_action_logprob_mean": sac_stats["policy_action_logprob_mean"],
                        "sac_s3/prior_action_logprob_mean": sac_stats["prior_action_logprob_mean"],
                        "sac_s3/q_chosen_mean": sac_stats["q_chosen_mean"],
                        "sac_s3/soft_value_mean": sac_stats["soft_value_mean"],
                        "sac_s3/td_target_mean": sac_stats["td_target_mean"],
                    }
                )
            except Exception:
                pass

            step += 1

        try:
            wandb.log(
                {
                    "final/synth_history_last5_mean": np.mean(synth_history[-5:]),
                    "final/synth_history_mean": np.mean(synth_history),
                }
            )
        except Exception:
            pass
