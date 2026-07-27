"""Masked boundary/pair/inside losses.

Start and end objectives are multi-label BCE (nested spans may share a
boundary — never a softmax over positions). All losses are masking-aware and
empty-query safe: denominators use ``clamp_min(1)`` so a query with no positive
span still contributes finite negative supervision without NaNs.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _safe_bce(logits: torch.Tensor, targets: torch.Tensor, keep: torch.BoolTensor) -> torch.Tensor:
    """Elementwise BCE-with-logits with extreme masked logits neutralized."""
    safe_logits = torch.where(keep, logits, torch.zeros_like(logits))
    safe_targets = torch.where(keep, targets, torch.zeros_like(targets))
    return F.binary_cross_entropy_with_logits(safe_logits, safe_targets, reduction="none")


def balanced_multilabel_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.BoolTensor,
    *,
    negative_weight: float = 1.0,
) -> torch.Tensor:
    """Mean multi-label BCE over valid positions, down-weighting negatives."""
    bce = _safe_bce(logits, targets, valid_mask)
    weight = torch.where(targets > 0.5, torch.ones_like(targets), torch.full_like(targets, negative_weight))
    masked = bce * weight * valid_mask.to(bce.dtype)
    denom = valid_mask.sum().clamp_min(1).to(bce.dtype)
    return masked.sum() / denom


def asymmetric_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.BoolTensor,
    *,
    gamma_positive: float = 0.0,
    gamma_negative: float = 2.0,
    clip: float = 0.05,
) -> torch.Tensor:
    """Asymmetric focal loss for imbalanced multi-label boundary targets."""
    keep = valid_mask
    safe_logits = torch.where(keep, logits, torch.zeros_like(logits))
    p = torch.sigmoid(safe_logits)
    if clip > 0:
        p_neg = (1 - p + clip).clamp(max=1.0)
    else:
        p_neg = 1 - p
    los_pos = targets * torch.log(p.clamp_min(1e-8)) * ((1 - p) ** gamma_positive)
    los_neg = (1 - targets) * torch.log(p_neg.clamp_min(1e-8)) * (p ** gamma_negative)
    loss = -(los_pos + los_neg) * valid_mask.to(logits.dtype)
    denom = valid_mask.sum().clamp_min(1).to(logits.dtype)
    return loss.sum() / denom


def build_candidate_labels(
    candidate_indices: torch.LongTensor,   # [B, Q, C, 2]
    candidate_mask: torch.BoolTensor,      # [B, Q, C]
    gold_pairs: torch.LongTensor,          # [B, Q, G, 2]
    gold_mask: torch.BoolTensor,           # [B, Q, G]
) -> torch.Tensor:
    """Label each candidate 1.0 iff it matches a gold pair for its query."""
    cand = candidate_indices.unsqueeze(3)          # [B,Q,C,1,2]
    gold = gold_pairs.unsqueeze(2)                 # [B,Q,1,G,2]
    same = (cand == gold).all(dim=-1)              # [B,Q,C,G]
    same = same & gold_mask.unsqueeze(2)           # respect gold validity
    labels = same.any(dim=-1).to(torch.float)      # [B,Q,C]
    return labels * candidate_mask.to(labels.dtype)


def select_hard_negative_candidates(
    pair_logits: torch.Tensor,        # [B, Q, C]
    labels: torch.Tensor,             # [B, Q, C]
    valid_mask: torch.BoolTensor,     # [B, Q, C]
    *,
    negatives_per_positive: int,
    minimum_negatives: int,
) -> torch.BoolTensor:
    """Select the highest-scoring negatives per query (positives always kept)."""
    b, q, c = pair_logits.shape
    keep = torch.zeros(b, q, c, dtype=torch.bool, device=pair_logits.device)
    for bi in range(b):
        for qi in range(q):
            valid = valid_mask[bi, qi]
            pos = (labels[bi, qi] > 0.5) & valid
            neg = (labels[bi, qi] <= 0.5) & valid
            n_pos = int(pos.sum())
            n_keep = max(minimum_negatives, negatives_per_positive * n_pos)
            neg_idx = torch.nonzero(neg, as_tuple=False).flatten()
            if neg_idx.numel() > 0:
                neg_scores = pair_logits[bi, qi, neg_idx]
                order = torch.argsort(neg_scores, descending=True, stable=True)
                chosen = neg_idx[order[:n_keep]]
                keep[bi, qi, chosen] = True
            keep[bi, qi][pos] = True
    return keep


def candidate_pair_loss(
    logits: torch.Tensor,             # [B, Q, C]
    labels: torch.Tensor,             # [B, Q, C]
    valid_mask: torch.BoolTensor,     # [B, Q, C]
    hard_negative_mask: Optional[torch.BoolTensor] = None,
) -> torch.Tensor:
    """BCE over candidates; optionally restrict negatives to a hard subset."""
    if hard_negative_mask is not None:
        effective = valid_mask & ((labels > 0.5) | hard_negative_mask)
    else:
        effective = valid_mask
    bce = _safe_bce(logits, labels, effective)
    masked = bce * effective.to(bce.dtype)
    denom = effective.sum().clamp_min(1).to(bce.dtype)
    return masked.sum() / denom


def inside_consistency_loss(
    inside_logits: torch.Tensor,      # [B, Q, L]
    inside_targets: torch.Tensor,     # [B, Q, L]
    text_mask: torch.BoolTensor,      # [B, L]
    query_mask: torch.BoolTensor,     # [B, Q]
) -> torch.Tensor:
    keep = text_mask.unsqueeze(1) & query_mask.unsqueeze(-1)
    return balanced_multilabel_bce(inside_logits, inside_targets, keep)
