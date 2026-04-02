from dataclasses import dataclass, field
import heapq, random, difflib
from typing import Optional, List, Literal
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

# RDKit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


# ---------- RDKit helpers ----------
def mol_from_smiles(smi: str) -> Optional[Chem.Mol]:
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def ecfp4(mol: Chem.Mol, nBits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits)


def tanimoto_bulk(fp, fps: List):
    return DataStructs.BulkTanimotoSimilarity(fp, fps) if fps else []


def _dense_rank_desc(vals):
    # rank = 1 + count of strictly greater values (dense, ties share the same rank)
    return [1 + sum(vj > vi for vj in vals) for vi in vals]


def _dense_rank_desc_for_candidate(v_new, vals):
    return 1 + sum(vj > v_new for vj in vals)


# ---------- Buffer ----------
@dataclass(order=True)
class Trajectory:
    sort_idx: float = field(init=False, repr=False)
    reward: float
    synthesizability: float
    smiles: str
    ids: torch.Tensor
    mask: torch.Tensor
    fp: object = field(compare=False, repr=False, default=None)      # RDKit ExplicitBitVect
    novelty: float = field(compare=False, repr=False, default=1.0)   # 1 - max(sim to others)
    difficulty: float = field(compare=False, repr=False, default=0.0)

    def __post_init__(self):
        self.sort_idx = self.reward  # min-heap on reward


class ReplayBuffer:
    """
    Diversity-aware replay buffer with switchable eviction policy.

    - Near-duplicate: Tanimoto >= (1 - sim)  -> treat as near-dup (replace only if higher reward)
    - Novelty: 1 - max(similarity to others)
    - Eviction when full:
        * evict_by="novelty": evict item with smallest novelty (tie -> lower reward)
        * evict_by="reward" : evict global min-reward item (heap root) if new has higher reward
        * evict_by="hybrid" : evict by average rank = alpha * rank_reward + (1-alpha) * rank_novelty
    """
    def __init__(
        self,
        eos_token_id: int,
        pad_token_id: int,
        max_size: int = 4096,
        sim: float = 0.25,
        evict_by: Literal["novelty", "reward", "hybrid", "oldest"] = "reward",
    ):
        self.eos  = eos_token_id
        self.pad  = pad_token_id
        self.max  = max_size
        self.sim  = sim
        self.heap: List[Trajectory] = []   # min-heap by reward
        self.pool = set()                  # SMILES strings
        self.evict_by = evict_by
        self.replaced_cnt = 0

    # ---- similarity helpers ----
    def _best_match(self, fp_new):
        if not self.heap:
            return -1, 0.0
        sims = tanimoto_bulk(fp_new, [it.fp for it in self.heap])
        if not sims:
            return -1, 0.0
        j = int(np.argmax(sims))
        return j, float(sims[j])

    def _novelty_vs_all(self, fp_new) -> float:
        if not self.heap:
            return 1.0
        sims = tanimoto_bulk(fp_new, [it.fp for it in self.heap])
        best_sim = max(sims) if sims else 0.0
        return 1.0 - float(best_sim)

    def _victim_index_by_least_diverse(self) -> int:
        victim_idx, victim_nov, victim_rew = -1, float('inf'), float('inf')
        for j, it in enumerate(self.heap):
            nv = it.novelty
            if (nv < victim_nov) or (nv == victim_nov and it.reward < victim_rew):
                victim_idx, victim_nov, victim_rew = j, nv, it.reward
        return victim_idx

    def reinitialize(self, trajectories: List[Trajectory]):
        self.heap = trajectories
        self.pool = set([trj.smiles for trj in trajectories])
        heapq.heapify(self.heap)
        self.replaced_cnt = 0

    # ---- batch add with switchable eviction ----
    def add_batch(
        self,
        ids: torch.Tensor,
        decoded: List[str],
        rewards: torch.Tensor,
        synthesizability: torch.Tensor,
        masks: List[torch.Tensor] | None = None,
        alpha: float = 0.5,
        use_reshaped_reward: bool = False,
    ):
        ids, rewards = ids.cpu(), rewards.cpu()
        near_dup_sim_thresh = 1.0 - self.sim

        for i, smi in enumerate(decoded):
            if smi in self.pool:
                continue

            mol = mol_from_smiles(smi)
            if mol is None:
                continue
            fp = ecfp4(mol)
            rew_i = float(rewards[i])

            # near-duplicate search
            best_idx, best_sim = self._best_match(fp)
            near_dup = self.heap[best_idx] if (best_idx >= 0 and best_sim >= near_dup_sim_thresh) else None

            # If similar and not better -> skip
            if near_dup and near_dup.reward >= rew_i and near_dup.synthesizability >= synthesizability[i]:
                continue

            ids_i = ids[i].clone()
            ids_i = ids_i[ids_i != self.pad]

            # Candidate
            if masks:
                traj = Trajectory(rew_i, synthesizability[i], smi, ids[i].clone(), masks[i].clone(), fp=fp)
            else:
                traj = Trajectory(rew_i, synthesizability[i], smi, ids[i].clone(), torch.ones_like(ids[i]), fp=fp)

            # If similar but better -> replace that slot directly
            if near_dup:
                victim_idx = best_idx
                self.pool.discard(self.heap[victim_idx].smiles)
                # Set novelty for the new item (optional for reward policy; harmless)
                traj.novelty = self._novelty_vs_all(traj.fp)
                self.heap[victim_idx] = traj
                heapq.heapify(self.heap)  # maintain min-heap by reward
                self.pool.add(smi)
                continue

            # Not similar: insert or evict depending on capacity & policy
            if len(self.heap) < self.max:
                # compute novelty (useful later; minor cost)
                traj.novelty = self._novelty_vs_all(traj.fp)
                heapq.heappush(self.heap, traj)
                self.pool.add(smi)
                continue

            # Buffer full -> decide by policy
            if self.evict_by == "hybrid":
                # Use stored novelties (computed on insert). If any are missing, treat as 1.0.
                rewards_list  = [it.reward  for it in self.heap]
                novelty_list  = [getattr(it, "novelty", 1.0) for it in self.heap]

                # Ranks for existing items (smaller is better)
                r_ranks = _dense_rank_desc(rewards_list)
                n_ranks = _dense_rank_desc(novelty_list)
                avg_ranks = [alpha * rr + (1 - alpha) * nr for rr, nr in zip(r_ranks, n_ranks)]

                # Candidate ranks computed against current lists + candidate value
                cand_r_rank = _dense_rank_desc_for_candidate(traj.reward, rewards_list)
                cand_n_rank = _dense_rank_desc_for_candidate(traj.novelty, novelty_list)
                cand_avg    = alpha * cand_r_rank + (1 - alpha) * cand_n_rank

                # Pick current worst (largest average rank); tie-break by lower reward
                worst_idx, worst_val, worst_rew = 0, -1.0, float("inf")
                for j, ar in enumerate(avg_ranks):
                    if (ar > worst_val) or (ar == worst_val and self.heap[j].reward < worst_rew):
                        worst_idx, worst_val, worst_rew = j, ar, self.heap[j].reward

                if (cand_avg < worst_val) or (cand_avg == worst_val and traj.reward > self.heap[worst_idx].reward):
                    self.pool.discard(self.heap[worst_idx].smiles)
                    self.heap[worst_idx] = traj
                    heapq.heapify(self.heap)
                    self.pool.add(smi)
                    self.replaced_cnt += 1
                # else: drop

            elif self.evict_by == "novelty":
                traj.novelty = self._novelty_vs_all(traj.fp)
                victim_idx = self._victim_index_by_least_diverse()
                victim = self.heap[victim_idx]
                if (traj.novelty > victim.novelty) or (traj.novelty == victim.novelty and traj.reward > victim.reward):
                    self.pool.discard(victim.smiles)
                    self.heap[victim_idx] = traj
                    heapq.heapify(self.heap)
                    self.pool.add(smi)
                    self.replaced_cnt += 1
            elif self.evict_by == "reward":
                curr = rew_i * synthesizability[i] if use_reshaped_reward else rew_i
                target = self.heap[0].reward * self.heap[0].synthesizability if use_reshaped_reward else self.heap[0].reward
                if curr > target:
                    worst = heapq.heapreplace(self.heap, traj)
                    self.pool.discard(worst.smiles)
                    self.pool.add(smi)
                    self.replaced_cnt += 1
            elif self.evict_by == "oldest":
                # Replace the oldest trajectory with the new one
                oldest = self.heap.pop(0)
                self.pool.discard(oldest.smiles)
                self.heap.append(traj)
                self.pool.add(smi)
                self.replaced_cnt += 1

    def periodic_refresh(self):
        """O(N^2) novelty recompute; call occasionally if you need tight novelty values."""
        if not self.heap:
            return
        fps = [it.fp for it in self.heap]
        for i, it in enumerate(self.heap):
            sims = tanimoto_bulk(it.fp, fps)
            sims_i = sims[:i] + sims[i+1:] if sims else []
            best_sim = max(sims_i) if sims_i else 0.0
            it.novelty = 1.0 - float(best_sim)

    def sample(self, n: int, device: str, reward_prioritized: bool = False, rank_based: bool = False, replace: bool = True, score_attr: str = "reward"):
        n = min(n, len(self.heap))
        if reward_prioritized:
            rewards = torch.tensor([float(getattr(t, score_attr, 0.0)) for t in self.heap], dtype=torch.float32)
            # rewards = torch.tensor([t.reward * t.synthesizability for t in self.heap], dtype=torch.float32)
            # Avoid negative or zero rewards for samplings
            min_reward = rewards.min().item()
            if min_reward <= 0:
                rewards = rewards - min_reward + 1e-6
            if rank_based:
                ## rank-based sampling
                scores_np = np.array([float(getattr(t, score_attr, 0.0)) for t in self.heap])
                ranks = np.argsort(np.argsort(-1 * scores_np))
                weights = 1.0 / (0.01 * len(scores_np) + ranks)
                indices = list(torch.utils.data.WeightedRandomSampler(
                    weights=weights, num_samples=n, replacement=True
                    ))
            else:
                probs = rewards / rewards.sum()
                indices = torch.multinomial(probs, n, replacement=replace).tolist()  # changed replacement to True
            ############################################################
            batch = [self.heap[i] for i in indices]
        else:
            batch = random.sample(self.heap, n)
        ids  = [t.ids for t in batch]
        # mask = [torch.ones_like(t.ids) for t in batch]
        mask = [t.mask for t in batch]
        ids  = pad_sequence(ids,  batch_first=True, padding_value=self.pad).to(device)
        mask = pad_sequence(mask, batch_first=True, padding_value=0).to(device)
        rewards = torch.tensor([t.reward for t in batch], device=device)
        synthesizabilities = torch.tensor([t.synthesizability for t in batch], device=device)
        return {"input_ids": ids, "mutation_mask": mask, "synthesizability": synthesizabilities, "smiles": [t.smiles for t in batch]}, rewards
