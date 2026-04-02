import copy
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append("/".join(path_here.rstrip("/").split("/")[:-2]))

from main.optimizer import BaseOptimizer
from main.smiles_gfn.run_reinvent import SynthesizabilityEvaluator
from main.smiles_gfn.run_sql_base import (
    _build_seq_mask,
    _get_special_token_ids,
    _masked_prior_log_probs,
)
from utils import unique


_original_warning = logging.Logger.warning


def _filter_fast_tfmr(self, msg, *args, **kwargs):
    """Swallow only the slow MoLFormer fallback warning."""
    if "Falling back to (slow) pytorch implementation" in str(msg):
        return
    _original_warning(self, msg, *args, **kwargs)


logging.Logger.warning = _filter_fast_tfmr

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SharedSACModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.actor_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = self.actor_model.config.hidden_size
        vocab_size = self.actor_model.config.vocab_size
        self.critic_head = nn.Linear(hidden_size, vocab_size, bias=False)

        with torch.no_grad():
            self.actor_model.lm_head.decoder.weight.zero_()
            self.critic_head.weight.zero_()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.actor_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]
        q_values = self.critic_head(hidden_states)
        return {
            "policy_logits": outputs.logits,
            "q_values": q_values,
        }


@torch.no_grad()
def _soft_update(target_model: nn.Module, source_model: nn.Module, tau: float):
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)


@torch.no_grad()
def _generate_sac_sequences(model, prior, tokenizer, batch_size: int, max_length: int, device: str):
    bos_token_id, eos_token_id, pad_token_id = _get_special_token_ids(tokenizer)
    seqs = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_length - 1):
        attention_mask = seqs.ne(pad_token_id).long()
        outputs = model(input_ids=seqs, attention_mask=attention_mask)
        policy_correction_logits = outputs["policy_logits"][:, -1, :]
        prior_logits = prior(input_ids=seqs, attention_mask=attention_mask).logits[:, -1, :]

        prior_log_probs = _masked_prior_log_probs(prior_logits, bos_token_id, pad_token_id)
        combined_logits = prior_log_probs + policy_correction_logits
        probs = F.softmax(combined_logits, dim=-1)

        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        next_tokens = torch.where(finished, torch.full_like(next_tokens, eos_token_id), next_tokens)

        seqs = torch.cat([seqs, next_tokens.unsqueeze(1)], dim=1)
        finished |= next_tokens.eq(eos_token_id)
        if finished.all():
            break

    seq_mask = _build_seq_mask(seqs, eos_token_id)
    return seqs, seq_mask


def _compute_sac_loss(
    model,
    target_model,
    prior,
    seqs: torch.Tensor,
    seq_mask: torch.Tensor,
    terminal_scores: torch.Tensor,
    bos_token_id: int,
    pad_token_id: int,
    beta: float,
):
    input_ids = seqs[:, :-1]
    labels = seqs[:, 1:]
    input_mask = seq_mask[:, :-1].long()
    action_mask = seq_mask[:, 1:].float()

    outputs = model(input_ids=input_ids, attention_mask=input_mask)
    policy_correction_logits = outputs["policy_logits"]
    q_values = outputs["q_values"]

    with torch.no_grad():
        prior_logits = prior(input_ids=input_ids, attention_mask=input_mask).logits
        target_outputs = target_model(input_ids=input_ids, attention_mask=input_mask)
        target_q_values = target_outputs["q_values"]

    prior_log_probs = _masked_prior_log_probs(prior_logits, bos_token_id, pad_token_id)
    combined_logits = prior_log_probs + policy_correction_logits
    policy_log_probs = F.log_softmax(combined_logits, dim=-1)
    policy_probs = policy_log_probs.exp()
    chosen_q_values = torch.gather(q_values, 2, labels.unsqueeze(-1)).squeeze(-1)
    chosen_policy_log_probs = torch.gather(policy_log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
    chosen_prior_log_probs = torch.gather(prior_log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)

    valid_action_support = torch.isfinite(prior_log_probs)
    safe_log_ratio = torch.where(
        valid_action_support,
        policy_log_probs - prior_log_probs,
        torch.zeros_like(policy_log_probs),
    )

    with torch.no_grad():
        detached_policy_probs = policy_probs.detach()
        detached_log_ratio = safe_log_ratio.detach()
        next_state_values = torch.where(
            valid_action_support,
            detached_policy_probs * (target_q_values - detached_log_ratio),
            torch.zeros_like(target_q_values),
        ).sum(dim=-1)

        td_target = torch.zeros_like(chosen_q_values)
        td_target[:, :-1] = next_state_values[:, 1:]

        lengths = action_mask.sum(dim=1).long()
        terminal_indices = torch.clamp(lengths - 1, min=0)
        batch_indices = torch.arange(seqs.shape[0], device=seqs.device)
        td_target[batch_indices, terminal_indices] = beta * terminal_scores

    critic_loss = ((chosen_q_values - td_target) ** 2 * action_mask).sum() / action_mask.sum().clamp_min(1.0)

    detached_q_values = q_values.detach()
    actor_state_loss = torch.where(
        valid_action_support,
        policy_probs * (safe_log_ratio - detached_q_values),
        torch.zeros_like(policy_probs),
    ).sum(dim=-1)
    actor_loss = (actor_state_loss * action_mask).sum() / action_mask.sum().clamp_min(1.0)

    with torch.no_grad():
        statewise_kl = torch.where(
            valid_action_support,
            policy_probs * safe_log_ratio,
            torch.zeros_like(policy_probs),
        ).sum(dim=-1)
        denom = action_mask.sum().clamp_min(1.0)
        safe_chosen_policy_log_probs = chosen_policy_log_probs.masked_fill(action_mask.eq(0), 0.0)
        safe_chosen_prior_log_probs = chosen_prior_log_probs.masked_fill(action_mask.eq(0), 0.0)
        safe_chosen_q_values = chosen_q_values.masked_fill(action_mask.eq(0), 0.0)
        safe_td_target = td_target.masked_fill(action_mask.eq(0), 0.0)
        safe_next_state_values = next_state_values.masked_fill(action_mask.eq(0), 0.0)
        stats = {
            "kl_mean": float((statewise_kl * action_mask).sum().item() / denom.item()),
            "sample_log_ratio_mean": float(
                ((safe_chosen_policy_log_probs - safe_chosen_prior_log_probs) * action_mask).sum().item()
                / denom.item()
            ),
            "policy_action_logprob_mean": float((safe_chosen_policy_log_probs * action_mask).sum().item() / denom.item()),
            "prior_action_logprob_mean": float((safe_chosen_prior_log_probs * action_mask).sum().item() / denom.item()),
            "q_chosen_mean": float((safe_chosen_q_values * action_mask).sum().item() / denom.item()),
            "soft_value_mean": float((safe_next_state_values * action_mask).sum().item() / denom.item()),
            "td_target_mean": float((safe_td_target * action_mask).sum().item() / denom.item()),
        }

    return actor_loss, critic_loss, stats


class SAC_BASE_Optimizer(BaseOptimizer):
    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "sac_base"

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

        bos_token_id, _, pad_token_id = _get_special_token_ids(tokenizer)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

        print("Model initialized, starting SAC-base training...")

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

            if config["reshape_reward"]:
                positive_indices = (synthesizability == 1).nonzero(as_tuple=True)[0]
                positive_smiles = [smiles[i] for i in positive_indices.tolist()]
                if len(positive_smiles) > 0:
                    positive_scores = torch.tensor(self.oracle(positive_smiles), device=device)
                else:
                    positive_scores = torch.empty(0, device=device)

                valid_scores = torch.zeros(len(smiles), device=device)
                valid_scores[positive_indices] = positive_scores
            else:
                valid_scores = torch.tensor(self.oracle(smiles), device=device)

            try:
                print(
                    f"step {step}: unique {len(unique_idxs)}, "
                    f"synthesizability {synthesizability.mean().item():.3f}, "
                    f"max score {valid_scores.max().item():.3f}, "
                    f"avg score {valid_scores.mean().item():.3f}"
                )
            except Exception:
                print(f"step {step}: unique {len(unique_idxs)}, synthesizability {synthesizability.mean().item():.3f}")

            if self.finish:
                print("max oracle hit")
                break

            if len(self.oracle) > 1000:
                self.sort_buffer()
                new_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
                if new_scores == old_scores:
                    patience += 1
                    if patience >= self.args.patience:
                        self.log_intermediate(finish=True)
                        print("convergence criteria met, abort ...... ")
                        break
                else:
                    patience = 0

            if prev_n_oracles < len(self.oracle):
                stuck_cnt = 0
            else:
                stuck_cnt += 1
                if stuck_cnt >= 10:
                    self.log_intermediate(finish=True)
                    print("cannot find new molecules, abort ...... ")
                    break

            prev_n_oracles = len(self.oracle)

            model.train()
            actor_loss, critic_loss, sac_stats = _compute_sac_loss(
                model,
                target_model,
                prior,
                seqs,
                seq_mask,
                valid_scores,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                beta=float(config["beta"]),
            )
            loss = actor_loss + critic_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["max_norm"])
            optimizer.step()
            _soft_update(target_model, model, tau=float(config["target_tau"]))

            try:
                print(
                    f"sac actor {actor_loss.item():.4f}, critic {critic_loss.item():.4f}, "
                    f"kl {sac_stats['kl_mean']:.4f}, log_ratio {sac_stats['sample_log_ratio_mean']:.4f}, "
                    f"prior_lp {sac_stats['prior_action_logprob_mean']:.4f}"
                )
            except Exception:
                pass

            try:
                wandb.log(
                    {
                        "sac/actor_loss": float(actor_loss.item()),
                        "sac/critic_loss": float(critic_loss.item()),
                        "sac/loss": float(loss.item()),
                        "sac/synth_rate": float(synthesizability.mean().item()),
                        "sac/avg_score": float(valid_scores.mean().item()),
                        "sac/kl_mean": sac_stats["kl_mean"],
                        "sac/sample_log_ratio_mean": sac_stats["sample_log_ratio_mean"],
                        "sac/policy_action_logprob_mean": sac_stats["policy_action_logprob_mean"],
                        "sac/prior_action_logprob_mean": sac_stats["prior_action_logprob_mean"],
                        "sac/q_chosen_mean": sac_stats["q_chosen_mean"],
                        "sac/soft_value_mean": sac_stats["soft_value_mean"],
                        "sac/td_target_mean": sac_stats["td_target_mean"],
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
