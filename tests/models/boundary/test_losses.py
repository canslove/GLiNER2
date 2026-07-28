"""Boundary losses: labels, masking, empty-query safety, hard negatives, grads."""

from __future__ import annotations

import pytest
import torch

from gliner2.models.boundary.losses import (
    balanced_multilabel_bce,
    build_candidate_labels,
    candidate_pair_loss,
    inside_consistency_loss,
    select_hard_negative_candidates,
)


def test_build_candidate_labels_matches_gold_pairs():
    # B=1, Q=1, C=3 candidates, G=2 gold
    cand = torch.tensor([[[[0, 2], [1, 3], [4, 5]]]])       # [1,1,3,2]
    cand_mask = torch.ones(1, 1, 3, dtype=torch.bool)
    gold = torch.tensor([[[[1, 3], [9, 9]]]])               # [1,1,2,2]
    gold_mask = torch.tensor([[[True, False]]])             # second gold invalid
    labels = build_candidate_labels(cand, cand_mask, gold, gold_mask)
    assert labels.shape == (1, 1, 3)
    assert labels[0, 0].tolist() == [0.0, 1.0, 0.0]


def test_balanced_multilabel_bce_empty_query_is_finite_and_zeroish():
    logits = torch.randn(2, 3, 5)
    targets = torch.zeros(2, 3, 5)
    valid = torch.zeros(2, 3, 5, dtype=torch.bool)  # no valid positions
    loss = balanced_multilabel_bce(logits, targets, valid)
    assert torch.isfinite(loss)
    assert float(loss) == 0.0


def test_balanced_multilabel_bce_downweights_negatives():
    # One positive, three negatives; all valid. With negative_weight < 1 the
    # negative contribution shrinks, so the loss must strictly decrease.
    logits = torch.zeros(1, 1, 4)          # p = 0.5 everywhere
    targets = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    valid = torch.ones(1, 1, 4, dtype=torch.bool)

    full = float(balanced_multilabel_bce(logits, targets, valid, negative_weight=1.0))
    down = float(balanced_multilabel_bce(logits, targets, valid, negative_weight=0.5))

    ln2 = 0.6931471805599453
    # bce = ln2 everywhere (p=0.5). Denominator = valid count = 4.
    # full = (1*ln2 + 3*ln2)/4 = ln2
    assert full == pytest.approx(ln2, rel=1e-5)
    # down = (1*ln2 + 3*0.5*ln2)/4 = (ln2 + 1.5 ln2)/4 = 2.5 ln2 / 4
    assert down == pytest.approx(2.5 * ln2 / 4, rel=1e-5)
    assert down < full


def test_balanced_multilabel_bce_handles_extreme_masked_logits():
    logits = torch.randn(1, 2, 4)
    logits[0, 0, 3] = torch.finfo(logits.dtype).min  # masked-out marginal sentinel
    targets = torch.zeros(1, 2, 4)
    targets[0, 0, 0] = 1.0
    valid = torch.ones(1, 2, 4, dtype=torch.bool)
    valid[0, 0, 3] = False
    loss = balanced_multilabel_bce(logits, targets, valid)
    assert torch.isfinite(loss)


def test_candidate_pair_loss_with_hard_negatives():
    torch.manual_seed(0)
    logits = torch.randn(1, 1, 6, requires_grad=True)
    labels = torch.tensor([[[1.0, 0, 0, 0, 0, 0]]])
    valid = torch.ones(1, 1, 6, dtype=torch.bool)
    hard = select_hard_negative_candidates(
        logits.detach(), labels, valid, negatives_per_positive=2, minimum_negatives=1,
    )
    # positive always kept
    assert bool(hard[0, 0, 0])
    # exactly 2 negatives kept (2 per 1 positive)
    assert int((hard[0, 0] & (labels[0, 0] <= 0.5)).sum()) == 2
    loss = candidate_pair_loss(logits, labels, valid, hard_negative_mask=hard)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_inside_consistency_loss_masks_padding():
    logits = torch.randn(2, 2, 5)
    targets = torch.zeros(2, 2, 5)
    targets[0, 0, 1:3] = 1.0
    text_mask = torch.ones(2, 5, dtype=torch.bool)
    text_mask[1, 3:] = False
    query_mask = torch.ones(2, 2, dtype=torch.bool)
    query_mask[1, 1] = False
    loss = inside_consistency_loss(logits, targets, text_mask, query_mask)
    assert torch.isfinite(loss)
    assert float(loss) > 0.0
