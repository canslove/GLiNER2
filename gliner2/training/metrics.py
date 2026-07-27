"""Boundary training metrics.

Proposal oracle recall is reported separately from final extraction F1 so a
failure can be attributed to proposal generation vs. reranking. All coordinates
are half-open ``[start, end)``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from gliner2.models.outputs import CandidateTensorBatch
from gliner2.processing.targets import PaddedTargetBatch, TargetGraph


@dataclass
class BoundaryTrainingMetrics:
    total_loss: float = 0.0
    start_loss: float = 0.0
    end_loss: float = 0.0
    pair_loss: float = 0.0
    inside_loss: float = 0.0
    start_recall: float = 0.0
    end_recall: float = 0.0
    proposal_oracle_recall: float = 0.0
    exact_precision: float = 0.0
    exact_recall: float = 0.0
    exact_f1: float = 0.0
    candidate_count_per_query: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _pairs_from_padded(targets: PaddedTargetBatch) -> List[List[set]]:
    """Gold ``{(start, end)}`` sets indexed ``[b][q]`` from padded targets."""
    b, q, g, _ = targets.mention_pairs.shape
    out: List[List[set]] = []
    for bi in range(b):
        per_q = []
        for qi in range(q):
            s = set()
            for gi in range(g):
                if bool(targets.mention_mask[bi, qi, gi]):
                    st = int(targets.mention_pairs[bi, qi, gi, 0])
                    en = int(targets.mention_pairs[bi, qi, gi, 1])
                    s.add((st, en))
            per_q.append(s)
        out.append(per_q)
    return out


def candidate_oracle_recall(
    candidates: CandidateTensorBatch,
    targets: PaddedTargetBatch,
) -> float:
    """Fraction of gold mentions present among the proposed candidates."""
    gold = _pairs_from_padded(targets)
    b, q, c, _ = candidates.indices.shape
    total = 0
    hit = 0
    for bi in range(b):
        for qi in range(q):
            g = gold[bi][qi]
            if not g:
                continue
            present = set()
            for ci in range(c):
                if bool(candidates.valid_mask[bi, qi, ci]):
                    present.add(
                        (int(candidates.indices[bi, qi, ci, 0]), int(candidates.indices[bi, qi, ci, 1]))
                    )
            for pair in g:
                total += 1
                if pair in present:
                    hit += 1
    return hit / total if total else 1.0


def boundary_recall(
    marginal_logits: torch.Tensor,   # [B, Q, N]
    boundary_targets: torch.Tensor,  # [B, Q, N] (1.0 at gold boundaries)
    keep_mask: torch.BoolTensor,     # [B, Q, N]
    *,
    threshold: float = 0.0,
) -> float:
    """Recall of gold boundaries whose logit exceeds ``threshold``."""
    gold = (boundary_targets > 0.5) & keep_mask
    pred = (marginal_logits > threshold) & keep_mask
    tp = int((gold & pred).sum())
    total = int(gold.sum())
    return tp / total if total else 1.0


def f1_from_counts(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> Tuple[float, float, float]:
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 1.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 1.0
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def exact_span_counts(
    predicted: Sequence[List[List[Tuple[int, int]]]],
    gold: Sequence[List[set]],
) -> Tuple[int, int, int]:
    """``(tp, fp, fn)`` over exact ``(query, start, end)`` matches."""
    tp = fp = fn = 0
    for bi in range(len(predicted)):
        for qi in range(len(predicted[bi])):
            pred = set(predicted[bi][qi])
            g = gold[bi][qi] if bi < len(gold) and qi < len(gold[bi]) else set()
            tp += len(pred & g)
            fp += len(pred - g)
            fn += len(g - pred)
    return tp, fp, fn


def gold_from_target_graphs(
    graphs: Sequence[TargetGraph],
    query_count: int,
) -> List[List[set]]:
    """Build ``gold[b][q] = {(start, end)}`` from canonical target graphs."""
    out: List[List[set]] = []
    for graph in graphs:
        per_q: List[set] = [set() for _ in range(query_count)]
        for m in graph.mentions:
            if 0 <= m.query_id < query_count:
                per_q[m.query_id].add((m.start, m.end))
        out.append(per_q)
    return out


__all__ = [
    "BoundaryTrainingMetrics",
    "candidate_oracle_recall",
    "boundary_recall",
    "f1_from_counts",
    "exact_span_counts",
    "gold_from_target_graphs",
]
