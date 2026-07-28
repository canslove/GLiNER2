"""Boundary training metrics: oracle recall, boundary recall, F1 counts."""

from __future__ import annotations

import torch

from gliner2.models.outputs import CandidateTensorBatch
from gliner2.processing.targets import MentionTarget, TargetGraph, pad_target_graphs
from gliner2.training.metrics import (
    boundary_recall,
    candidate_oracle_recall,
    exact_span_counts,
    f1_from_counts,
    gold_from_target_graphs,
)


def _candidates(indices, valid, query_mask):
    b, q, c, _ = indices.shape
    zeros = torch.zeros(b, q, c)
    return CandidateTensorBatch(
        indices=indices, proposal_logits=zeros, pair_logits=zeros,
        valid_mask=valid, query_mask=query_mask,
    )


def test_candidate_oracle_recall_full_and_partial():
    # One query, gold {(0,2),(1,4)}; candidates include both -> recall 1.0
    targets = pad_target_graphs(
        [TargetGraph(mentions=(MentionTarget(0, 0, 2), MentionTarget(0, 1, 4)))],
        [1], [5], max_gold_per_query=4,
    )
    idx = torch.tensor([[[[0, 2], [1, 4], [0, 0]]]])
    valid = torch.tensor([[[True, True, False]]])
    qm = torch.ones(1, 1, dtype=torch.bool)
    assert candidate_oracle_recall(_candidates(idx, valid, qm), targets) == 1.0

    # Drop (1,4) from candidates -> recall 0.5
    idx2 = torch.tensor([[[[0, 2], [3, 4], [0, 0]]]])
    assert candidate_oracle_recall(_candidates(idx2, valid, qm), targets) == 0.5


def test_boundary_recall_thresholding():
    logits = torch.tensor([[[2.0, -1.0, 3.0]]])
    targets = torch.tensor([[[1.0, 1.0, 0.0]]])
    keep = torch.ones(1, 1, 3, dtype=torch.bool)
    # gold positions 0 and 1; only position 0 exceeds threshold 0 -> recall 0.5
    assert boundary_recall(logits, targets, keep, threshold=0.0) == 0.5


def test_f1_from_counts():
    p, r, f = f1_from_counts(2, 1, 1)
    assert round(p, 3) == 0.667
    assert round(r, 3) == 0.667
    assert round(f, 3) == 0.667
    # No predictions and no gold -> zero by sklearn convention (a model that
    # produces nothing must not score perfectly).
    assert f1_from_counts(0, 0, 0) == (0.0, 0.0, 0.0)
    # zero_division is configurable for callers that want the old behavior.
    assert f1_from_counts(0, 0, 0, zero_division=1.0) == (1.0, 1.0, 1.0)


def test_recall_helpers_empty_denominator_report_zero():
    # No gold mentions -> oracle recall 0.0 (not 1.0).
    targets = pad_target_graphs(
        [TargetGraph(mentions=())], [1], [5], max_gold_per_query=4,
    )
    idx = torch.tensor([[[[0, 2], [1, 4], [0, 0]]]])
    valid = torch.tensor([[[True, True, False]]])
    qm = torch.ones(1, 1, dtype=torch.bool)
    assert candidate_oracle_recall(_candidates(idx, valid, qm), targets) == 0.0

    # No gold boundaries -> boundary recall 0.0.
    logits = torch.tensor([[[2.0, -1.0, 3.0]]])
    zero_targets = torch.zeros(1, 1, 3)
    keep = torch.ones(1, 1, 3, dtype=torch.bool)
    assert boundary_recall(logits, zero_targets, keep, threshold=0.0) == 0.0


def test_exact_span_counts_and_gold_helper():
    graphs = [TargetGraph(mentions=(MentionTarget(0, 0, 2), MentionTarget(1, 3, 5)))]
    gold = gold_from_target_graphs(graphs, query_count=2)
    assert gold == [[{(0, 2)}, {(3, 5)}]]
    preds = [[[(0, 2)], [(3, 5), (0, 1)]]]  # one false positive in query 1
    tp, fp, fn = exact_span_counts(preds, gold)
    assert (tp, fp, fn) == (2, 1, 0)
