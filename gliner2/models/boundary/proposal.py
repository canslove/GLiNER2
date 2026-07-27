"""Sparse boundary-pair proposal generation.

Selects a small set of start/end boundaries per query, then scores conditional
end (and, bidirectionally, start) boundaries in streaming blocks. Work is linear
in sequence length for fixed schema and budgets: the largest pair-score tensor
materialized at once is ``[B, Q, Ks, end_block_size]``. There is deliberately no
condition on ``end - start`` — a start at ``0`` may pair with an end at ``L``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from gliner2.processing.targets import TargetCapacityError


# =============================================================================
# Settings and outputs
# =============================================================================

@dataclass(frozen=True)
class ProposalSettings:
    start_top_k: int
    end_top_k: int
    ends_per_start: int
    starts_per_end: int
    candidate_budget: int
    training_candidate_budget: int
    max_gold_per_query: int
    end_block_size: int
    bidirectional: bool = True
    export_mode: str = "streaming"  # "streaming" (blockwise) | "vectorized" (single [B,Q,Ks,L+1] block)


@dataclass(frozen=True)
class ProposalStats:
    boundary_score_elements: int
    conditional_pair_score_elements: int
    max_materialized_pair_elements: int
    retained_candidate_count: int


@dataclass
class BoundaryProposals:
    indices: torch.LongTensor          # [B, Q, C, 2] half-open [start, end)
    logits: torch.Tensor               # [B, Q, C]
    valid_mask: torch.BoolTensor       # [B, Q, C]
    gold_mask: Optional[torch.BoolTensor] = None  # [B, Q, C]
    stats: Optional[ProposalStats] = None


# =============================================================================
# Boundary selection
# =============================================================================

def select_top_boundaries(
    logits: torch.Tensor,
    valid_mask: torch.BoolTensor,
    k: int,
) -> Tuple[torch.Tensor, torch.LongTensor, torch.BoolTensor]:
    """Select the top-``k`` boundaries by logit (stable, index tie-break).

    Args:
        logits: [B, Q, N]
        valid_mask: [B, Q, N] True where the boundary/query is valid.
        k: number to select.
    Returns:
        scores [B, Q, k], indices [B, Q, k], valid [B, Q, k].
    """
    n = logits.shape[-1]
    k = min(k, n)
    masked = logits.masked_fill(~valid_mask, float("-inf"))
    order = torch.argsort(masked, dim=-1, descending=True, stable=True)
    idx = order[..., :k]
    scores = torch.gather(masked, -1, idx)
    valid = torch.isfinite(scores)
    # Replace -inf sentinel scores with 0 for downstream arithmetic; validity is
    # carried separately.
    scores = torch.where(valid, scores, torch.zeros_like(scores))
    idx = torch.where(valid, idx, torch.zeros_like(idx))
    return scores, idx, valid


def merge_running_topk(
    current_scores: torch.Tensor,
    current_indices: torch.LongTensor,
    block_scores: torch.Tensor,
    block_indices: torch.LongTensor,
    k: int,
) -> Tuple[torch.Tensor, torch.LongTensor]:
    """Merge a running top-``k`` with a new block's candidates (stable)."""
    scores = torch.cat([current_scores, block_scores], dim=-1)
    indices = torch.cat([current_indices, block_indices], dim=-1)
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    order = order[..., :k]
    top_scores = torch.gather(scores, -1, order)
    top_indices = torch.gather(indices, -1, order)
    return top_scores, top_indices


def _score_ends_blockwise(
    sq: torch.Tensor,                  # [B, Q, K, d] projected+gated start states
    end_proj_all: torch.Tensor,        # [B, L+1, d] projected end states
    start_indices: torch.LongTensor,   # [B, Q, K]
    start_scores: torch.Tensor,        # [B, Q, K]
    boundary_mask: torch.BoolTensor,   # [B, L+1]
    query_mask: torch.BoolTensor,      # [B, Q]
    end_marginals: torch.Tensor,       # [B, Q, L+1]
    block_size: int,
    top_k: int,
    scale: float,
) -> Tuple[torch.Tensor, torch.LongTensor, int, int]:
    """Stream over end blocks, keeping running top-``top_k`` ends per start.

    Returns ``(top_scores, top_end_idx, conditional_elems, max_block_elems)``
    where scores/idx are ``[B, Q, K, top_k]``.
    """
    b, q, k, d = sq.shape
    n = end_proj_all.shape[1]
    device = sq.device
    neg_inf = float("-inf")

    top_scores = torch.full((b, q, k, top_k), neg_inf, device=device, dtype=sq.dtype)
    top_idx = torch.zeros((b, q, k, top_k), device=device, dtype=torch.long)

    conditional_elems = 0
    max_block_elems = 0

    for j0 in range(0, n, block_size):
        j1 = min(j0 + block_size, n)
        e = j1 - j0
        ej = end_proj_all[:, j0:j1]                                    # [B, e, d]
        block_compat = torch.einsum("bqkd,bed->bqke", sq, ej) * scale  # [B,Q,K,e]
        end_marg_block = end_marginals[:, :, j0:j1].unsqueeze(2)        # [B,Q,1,e]
        block = block_compat + end_marg_block + start_scores.unsqueeze(-1)

        end_index = torch.arange(j0, j1, device=device)               # [e]
        # valid end iff boundary valid, query valid, and end > start.
        end_valid = boundary_mask[:, j0:j1].view(b, 1, 1, e)          # [B,1,1,e]
        q_valid = query_mask.view(b, q, 1, 1)
        after_start = end_index.view(1, 1, 1, e) > start_indices.unsqueeze(-1)
        keep = end_valid & q_valid & after_start
        block = block.masked_fill(~keep, neg_inf)

        block_idx = end_index.view(1, 1, 1, e).expand(b, q, k, e)
        top_scores, top_idx = merge_running_topk(top_scores, top_idx, block, block_idx, top_k)

        conditional_elems += b * q * k * e
        max_block_elems = max(max_block_elems, b * q * k * e)

    return top_scores, top_idx, conditional_elems, max_block_elems


def _score_starts_blockwise(
    eq: torch.Tensor,                  # [B, Q, K, d] projected+gated end states
    start_proj_all: torch.Tensor,      # [B, L+1, d]
    end_indices: torch.LongTensor,     # [B, Q, K]
    end_scores: torch.Tensor,          # [B, Q, K]
    boundary_mask: torch.BoolTensor,
    query_mask: torch.BoolTensor,
    start_marginals: torch.Tensor,     # [B, Q, L+1]
    block_size: int,
    top_k: int,
    scale: float,
) -> Tuple[torch.Tensor, torch.LongTensor, int, int]:
    """Bidirectional counterpart: top starts strictly before each selected end."""
    b, q, k, d = eq.shape
    n = start_proj_all.shape[1]
    device = eq.device
    neg_inf = float("-inf")

    top_scores = torch.full((b, q, k, top_k), neg_inf, device=device, dtype=eq.dtype)
    top_idx = torch.zeros((b, q, k, top_k), device=device, dtype=torch.long)

    conditional_elems = 0
    max_block_elems = 0

    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        e = i1 - i0
        sj = start_proj_all[:, i0:i1]
        block_compat = torch.einsum("bqkd,bed->bqke", eq, sj) * scale
        start_marg_block = start_marginals[:, :, i0:i1].unsqueeze(2)
        block = block_compat + start_marg_block + end_scores.unsqueeze(-1)

        start_index = torch.arange(i0, i1, device=device)
        b_valid = boundary_mask[:, i0:i1].view(b, 1, 1, e)
        q_valid = query_mask.view(b, q, 1, 1)
        before_end = start_index.view(1, 1, 1, e) < end_indices.unsqueeze(-1)
        keep = b_valid & q_valid & before_end
        block = block.masked_fill(~keep, neg_inf)

        block_idx = start_index.view(1, 1, 1, e).expand(b, q, k, e)
        top_scores, top_idx = merge_running_topk(top_scores, top_idx, block, block_idx, top_k)

        conditional_elems += b * q * k * e
        max_block_elems = max(max_block_elems, b * q * k * e)

    return top_scores, top_idx, conditional_elems, max_block_elems


# =============================================================================
# Deduplication, gold injection, padding (per (b, q))
# =============================================================================

def _stable_desc_order(scores: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    """Order indices by descending score, ties broken by (start, end) ascending."""
    n = scores.shape[0]
    device = scores.device
    # Composite sort: primary score desc, then start asc, then end asc.
    # Sort ascending by end, then start, then by -score using stable sorts.
    order = torch.arange(n, device=device)
    # end asc
    order = order[torch.argsort(ends[order], stable=True)]
    # start asc
    order = order[torch.argsort(starts[order], stable=True)]
    # score desc (stable keeps prior tie-break)
    order = order[torch.argsort(-scores[order], stable=True)]
    return order


def deduplicate_boundary_pairs(
    starts: torch.Tensor,
    ends: torch.Tensor,
    scores: torch.Tensor,
    valid: torch.BoolTensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collapse duplicate ``(start, end)`` pairs keeping the highest score.

    Operates on 1-D tensors for a single (sample, query). Returns
    ``(starts, ends, scores)`` of unique valid pairs, ordered by descending
    score with deterministic tie-break.
    """
    keep_s: List[int] = []
    keep_e: List[int] = []
    keep_sc: List[float] = []
    best = {}
    for i in range(starts.shape[0]):
        if not bool(valid[i]):
            continue
        key = (int(starts[i]), int(ends[i]))
        sc = float(scores[i])
        if key not in best or sc > best[key]:
            best[key] = sc
    if not best:
        empty_l = torch.zeros(0, dtype=torch.long, device=starts.device)
        empty_f = torch.zeros(0, dtype=scores.dtype, device=starts.device)
        return empty_l, empty_l, empty_f
    for (s, e), sc in best.items():
        keep_s.append(s)
        keep_e.append(e)
        keep_sc.append(sc)
    s_t = torch.tensor(keep_s, dtype=torch.long, device=starts.device)
    e_t = torch.tensor(keep_e, dtype=torch.long, device=starts.device)
    sc_t = torch.tensor(keep_sc, dtype=scores.dtype, device=starts.device)
    order = _stable_desc_order(sc_t, s_t, e_t)
    return s_t[order], e_t[order], sc_t[order]


class SparseBoundaryProposer(nn.Module):
    def __init__(self, boundary_dim: int, query_dim: int, settings: ProposalSettings):
        super().__init__()
        self.boundary_dim = boundary_dim
        self.settings = settings
        self.start_pair_projection = nn.Linear(boundary_dim, boundary_dim)
        self.end_key_projection = nn.Linear(boundary_dim, boundary_dim)
        self.start_query_projection = nn.Linear(query_dim, boundary_dim)

    def forward(
        self,
        boundary_states: torch.Tensor,   # [B, L+1, d]
        boundary_mask: torch.BoolTensor,  # [B, L+1]
        query_states: torch.Tensor,       # [B, Q, Hq]
        query_mask: torch.BoolTensor,     # [B, Q]
        start_logits: torch.Tensor,       # [B, Q, L+1]
        end_logits: torch.Tensor,         # [B, Q, L+1]
        *,
        gold_pairs: Optional[torch.LongTensor] = None,   # [B, Q, G, 2]
        gold_mask: Optional[torch.BoolTensor] = None,     # [B, Q, G]
        return_stats: bool = False,
    ) -> BoundaryProposals:
        s = self.settings
        b, n, d = boundary_states.shape
        q = query_states.shape[1]
        device = boundary_states.device
        scale = 1.0 / math.sqrt(self.boundary_dim)
        training = self.training and gold_pairs is not None
        capacity = s.training_candidate_budget if training else s.candidate_budget

        # Export mode materializes a single full-width [B,Q,Ks,L+1] block
        # (still linear in L for fixed Ks) instead of streaming blocks — this
        # avoids the block loop for graph exporters while keeping identical
        # results and never building an [L, L] tensor.
        end_block = n if s.export_mode == "vectorized" else s.end_block_size

        b_valid = boundary_mask.unsqueeze(1) & query_mask.unsqueeze(-1)  # [B,Q,L+1]

        # Projected boundary states (shared across queries).
        start_proj_all = self.start_pair_projection(boundary_states)  # [B,L+1,d]
        end_proj_all = self.end_key_projection(boundary_states)       # [B,L+1,d]
        gate = torch.sigmoid(self.start_query_projection(query_states))  # [B,Q,d]

        # ---- forward direction: top starts -> conditional ends --------------
        st_scores, st_idx, st_valid = select_top_boundaries(start_logits, b_valid, s.start_top_k)
        # gather start states and gate by query
        st_states = torch.gather(
            start_proj_all.unsqueeze(1).expand(b, q, n, d), 2,
            st_idx.unsqueeze(-1).expand(-1, -1, -1, d),
        )  # [B,Q,Ks,d]
        sq = st_states * gate.unsqueeze(2)
        fwd_scores, fwd_end_idx, cond_e1, maxe1 = _score_ends_blockwise(
            sq, end_proj_all, st_idx, st_scores, boundary_mask, query_mask,
            end_logits, end_block, s.ends_per_start, scale,
        )
        # forward pairs: (start = st_idx[k], end = fwd_end_idx[k, :])
        fwd_start = st_idx.unsqueeze(-1).expand(-1, -1, -1, s.ends_per_start)  # [B,Q,Ks,eps]
        fwd_pairs_s = fwd_start.reshape(b, q, -1)
        fwd_pairs_e = fwd_end_idx.reshape(b, q, -1)
        fwd_pairs_sc = fwd_scores.reshape(b, q, -1)

        pair_starts = [fwd_pairs_s]
        pair_ends = [fwd_pairs_e]
        pair_scores = [fwd_pairs_sc]

        cond_e2 = 0
        maxe2 = 0
        if s.bidirectional:
            en_scores, en_idx, en_valid = select_top_boundaries(end_logits, b_valid, s.end_top_k)
            en_states = torch.gather(
                end_proj_all.unsqueeze(1).expand(b, q, n, d), 2,
                en_idx.unsqueeze(-1).expand(-1, -1, -1, d),
            )
            eq = en_states * gate.unsqueeze(2)
            bwd_scores, bwd_start_idx, cond_e2, maxe2 = _score_starts_blockwise(
                eq, start_proj_all, en_idx, en_scores, boundary_mask, query_mask,
                start_logits, end_block, s.starts_per_end, scale,
            )
            bwd_end = en_idx.unsqueeze(-1).expand(-1, -1, -1, s.starts_per_end)
            pair_starts.append(bwd_start_idx.reshape(b, q, -1))
            pair_ends.append(bwd_end.reshape(b, q, -1))
            pair_scores.append(bwd_scores.reshape(b, q, -1))

        all_s = torch.cat(pair_starts, dim=-1)   # [B,Q,P]
        all_e = torch.cat(pair_ends, dim=-1)
        # Ordering/selection scores only; differentiable logits are recomputed
        # from the chosen indices below, so detach to avoid grad-to-scalar noise.
        all_sc = torch.cat(pair_scores, dim=-1).detach()
        all_valid = torch.isfinite(all_sc) & (all_e > all_s)

        # ---- per (b, q): dedup, gold-inject, cap, pad -----------------------
        out_idx = torch.zeros(b, q, capacity, 2, dtype=torch.long, device=device)
        out_logits = torch.full((b, q, capacity), float("-inf"), device=device, dtype=all_sc.dtype)
        out_valid = torch.zeros(b, q, capacity, dtype=torch.bool, device=device)
        out_gold = torch.zeros(b, q, capacity, dtype=torch.bool, device=device)

        for bi in range(b):
            for qi in range(q):
                if not bool(query_mask[bi, qi]):
                    continue
                s_t, e_t, sc_t = deduplicate_boundary_pairs(
                    all_s[bi, qi], all_e[bi, qi], all_sc[bi, qi], all_valid[bi, qi]
                )
                gold_set = set()
                if training:
                    for gi in range(gold_pairs.shape[2]):
                        if not bool(gold_mask[bi, qi, gi]):
                            continue
                        gs = int(gold_pairs[bi, qi, gi, 0])
                        ge = int(gold_pairs[bi, qi, gi, 1])
                        gold_set.add((gs, ge))
                    if len(gold_set) > capacity:
                        raise TargetCapacityError(
                            f"sample={bi} query={qi} has {len(gold_set)} unique gold "
                            f"pairs but candidate capacity is {capacity}. Increase "
                            "boundary_head.training_candidate_budget."
                        )

                # Build ordered candidate list (dedup already score-sorted).
                cand_s = s_t.tolist()
                cand_e = e_t.tolist()
                cand_sc = sc_t.tolist()
                present = {(cand_s[i], cand_e[i]): i for i in range(len(cand_s))}
                is_gold = [(cand_s[i], cand_e[i]) in gold_set for i in range(len(cand_s))]

                # Force-include any missing gold pairs (score = +inf sentinel so
                # they are never dropped by capacity trimming).
                for (gs, ge) in gold_set:
                    if (gs, ge) not in present:
                        cand_s.append(gs)
                        cand_e.append(ge)
                        cand_sc.append(float("inf"))
                        is_gold.append(True)

                # If over capacity, drop lowest-scoring NON-gold candidates.
                if len(cand_s) > capacity:
                    order = sorted(
                        range(len(cand_s)),
                        key=lambda i: (not is_gold[i], -cand_sc[i], cand_s[i], cand_e[i]),
                    )
                    # Keep gold + highest non-gold up to capacity.
                    keep = order[:capacity]
                    # Preserve score order among kept.
                    keep = sorted(keep, key=lambda i: (-cand_sc[i], cand_s[i], cand_e[i]))
                    cand_s = [cand_s[i] for i in keep]
                    cand_e = [cand_e[i] for i in keep]
                    cand_sc = [cand_sc[i] for i in keep]
                    is_gold = [is_gold[i] for i in keep]

                c = min(len(cand_s), capacity)
                for i in range(c):
                    out_idx[bi, qi, i, 0] = cand_s[i]
                    out_idx[bi, qi, i, 1] = cand_e[i]
                    out_valid[bi, qi, i] = True
                    out_gold[bi, qi, i] = is_gold[i]

        # Differentiable proposal logits: recompute from the (detached) selected
        # indices so gradients flow to the proposer projections and marginals.
        si = out_idx[..., 0]                                            # [B,Q,C]
        ej = out_idx[..., 1]
        sp = start_proj_all.unsqueeze(1).expand(b, q, n, d)
        ep = end_proj_all.unsqueeze(1).expand(b, q, n, d)
        g_s = torch.gather(sp, 2, si.unsqueeze(-1).expand(-1, -1, -1, d)) * gate.unsqueeze(2)
        g_e = torch.gather(ep, 2, ej.unsqueeze(-1).expand(-1, -1, -1, d))
        compat = (g_s * g_e).sum(-1) * scale                            # [B,Q,C]
        sm = torch.gather(start_logits, 2, si)
        em = torch.gather(end_logits, 2, ej)
        logits_diff = compat + sm + em
        out_logits = torch.where(
            out_valid, logits_diff, torch.full_like(logits_diff, float("-inf"))
        )

        stats = None
        if return_stats:
            retained = int(out_valid.sum())
            stats = ProposalStats(
                boundary_score_elements=b * q * n * 2,
                conditional_pair_score_elements=cond_e1 + cond_e2,
                max_materialized_pair_elements=max(maxe1, maxe2),
                retained_candidate_count=retained,
            )

        return BoundaryProposals(
            indices=out_idx,
            logits=out_logits,
            valid_mask=out_valid,
            gold_mask=out_gold if training else None,
            stats=stats,
        )
