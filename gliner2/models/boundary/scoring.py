"""Sparse pair (candidate) reranking.

Produces one scalar logit per proposed candidate. The score is a sum of scalar
factors: start marginal, end marginal, endpoint compatibility, optional inside
evidence, continuous length features, and the proposal prior. No width
embedding table (which would reintroduce a maximum length) and no persistent
per-candidate vector is materialized beyond transient gathers.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from gliner2.models.boundary.proposal import BoundaryProposals


def gather_boundary_states(boundary_states: torch.Tensor, indices: torch.LongTensor) -> torch.Tensor:
    """Gather ``[B, L+1, d]`` states at ``[B, Q, C]`` boundary indices → ``[B, Q, C, d]``."""
    b, n, d = boundary_states.shape
    q, c = indices.shape[1], indices.shape[2]
    exp = boundary_states.unsqueeze(1).expand(b, q, n, d)
    return torch.gather(exp, 2, indices.unsqueeze(-1).expand(-1, -1, -1, d))


def interval_prefix_score(prefix: torch.Tensor, starts: torch.LongTensor, ends: torch.LongTensor) -> torch.Tensor:
    """``prefix[..., end] - prefix[..., start]`` for ``[B, Q, C]`` indices.

    ``prefix`` is ``[B, Q, L+1]``. Returns ``[B, Q, C]``.
    """
    p_end = torch.gather(prefix, 2, ends)
    p_start = torch.gather(prefix, 2, starts)
    return p_end - p_start


def continuous_length_features(
    starts: torch.LongTensor,
    ends: torch.LongTensor,
    text_lengths: torch.LongTensor,
) -> torch.Tensor:
    """Length features ``[B, Q, C, 3]`` with no maximum-length lookup."""
    length = (ends - starts).clamp(min=1).float()
    b = starts.shape[0]
    tl = text_lengths.view(b, 1, 1).float().clamp(min=1)
    feats = torch.stack(
        [
            torch.log1p(length),
            length / tl,
            torch.rsqrt(length),
        ],
        dim=-1,
    )
    return feats


def mask_invalid_candidate_logits(logits: torch.Tensor, valid_mask: torch.BoolTensor) -> torch.Tensor:
    return logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)


class SparseBoundaryPairScorer(nn.Module):
    def __init__(
        self,
        boundary_dim: int,
        query_dim: int,
        pair_dim: int,
        use_inside_evidence: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.boundary_dim = boundary_dim
        self.pair_dim = pair_dim
        self.use_inside_evidence = use_inside_evidence
        self.start_endpoint_projection = nn.Linear(boundary_dim, pair_dim)
        self.end_endpoint_projection = nn.Linear(boundary_dim, pair_dim)
        self.query_gate = nn.Linear(query_dim, pair_dim)
        self.length_query_projection = nn.Linear(query_dim, 3)
        self.inside_weight = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        boundary_states: torch.Tensor,      # [B, L+1, d]
        query_states: torch.Tensor,         # [B, Q, Hq]
        proposals: BoundaryProposals,
        start_logits: torch.Tensor,         # [B, Q, L+1]
        end_logits: torch.Tensor,           # [B, Q, L+1]
        inside_prefix: Optional[torch.Tensor],  # [B, Q, L+1] or None
        text_lengths: torch.LongTensor,     # [B]
    ) -> torch.Tensor:
        starts = proposals.indices[..., 0]   # [B,Q,C]
        ends = proposals.indices[..., 1]
        valid = proposals.valid_mask
        scale = 1.0 / math.sqrt(self.pair_dim)

        # Endpoint compatibility.
        g_start = gather_boundary_states(boundary_states, starts)   # [B,Q,C,d]
        g_end = gather_boundary_states(boundary_states, ends)
        s_proj = self.dropout(self.start_endpoint_projection(g_start))
        e_proj = self.dropout(self.end_endpoint_projection(g_end))
        gate = torch.sigmoid(self.query_gate(query_states)).unsqueeze(2)  # [B,Q,1,pair]
        compat = (s_proj * gate * e_proj).sum(-1) * scale                 # [B,Q,C]

        # Start/end marginals gathered at the candidate boundaries.
        a = torch.gather(start_logits, 2, starts)
        bmarg = torch.gather(end_logits, 2, ends)

        prior = torch.where(valid, proposals.logits, torch.zeros_like(proposals.logits))
        score = compat + a + bmarg + prior

        # Inside evidence.
        if self.use_inside_evidence and inside_prefix is not None:
            interval = interval_prefix_score(inside_prefix, starts, ends)  # [B,Q,C]
            denom = torch.sqrt((ends - starts).clamp(min=1).float())
            score = score + self.inside_weight * (interval / denom)

        # Length features.
        feats = continuous_length_features(starts, ends, text_lengths)     # [B,Q,C,3]
        length_coeff = self.length_query_projection(query_states).unsqueeze(2)  # [B,Q,1,3]
        length_score = (feats * length_coeff).sum(-1)                      # [B,Q,C]
        score = score + length_score

        return mask_invalid_candidate_logits(score, valid)
