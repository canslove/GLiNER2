"""Reference equivalence tests for boundary hot-path vectorization."""

from __future__ import annotations

import pytest
import torch

from gliner2.models.boundary.losses import select_hard_negative_candidates
from gliner2.models.boundary.model import decode_candidates
from gliner2.models.boundary.proposal import assemble_candidates, select_top_boundaries
from gliner2.models.outputs import CandidateTensorBatch


def _assemble_reference(starts, ends, scores, valid, query_mask, capacity, n, gold, gold_mask):
    b, q, _ = starts.shape
    out = torch.zeros(b, q, capacity, 2, dtype=torch.long)
    out_valid = torch.zeros(b, q, capacity, dtype=torch.bool)
    out_gold = torch.zeros(b, q, capacity, dtype=torch.bool)
    ceiling = -torch.finfo(scores.dtype).min / 4.0
    for bi in range(b):
        for qi in range(q):
            if not bool(query_mask[bi, qi]):
                continue
            best = {}
            for pi in range(starts.shape[-1]):
                if bool(valid[bi, qi, pi]):
                    key = (int(starts[bi, qi, pi]), int(ends[bi, qi, pi]))
                    score = float(scores[bi, qi, pi])
                    best[key] = max(best.get(key, -float("inf")), score)
            gold_set = set()
            if gold is not None:
                for gi in range(gold.shape[2]):
                    if bool(gold_mask[bi, qi, gi]):
                        key = tuple(int(v) for v in gold[bi, qi, gi])
                        gold_set.add(key)
                        best[key] = ceiling
            ordered = sorted(best.items(), key=lambda item: (-item[1], item[0]))
            for ci, (key, _) in enumerate(ordered[:capacity]):
                out[bi, qi, ci] = torch.tensor(key)
                out_valid[bi, qi, ci] = True
                out_gold[bi, qi, ci] = key in gold_set
    return out, out_valid, out_gold


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("with_gold", [False, True])
def test_assemble_candidates_matches_reference(seed, with_gold):
    torch.manual_seed(seed)
    b, q, p, n, capacity = 2, 3, 23, 12, 13
    starts = torch.randint(0, n - 1, (b, q, p))
    ends = torch.maximum(starts + 1, torch.randint(1, n, (b, q, p))).clamp_max(n - 1)
    scores = torch.randint(-2, 4, (b, q, p)).float()
    valid = torch.rand(b, q, p) > 0.2
    query_mask = torch.tensor([[True, True, False], [True, False, True]])
    gold = torch.randint(0, n - 1, (b, q, 5, 2))
    gold[..., 1] = torch.maximum(gold[..., 0] + 1, gold[..., 1]).clamp_max(n - 1)
    gold_mask = torch.rand(b, q, 5) > 0.35
    if not with_gold:
        gold = gold_mask = None

    actual = assemble_candidates(
        starts, ends, scores, valid, query_mask,
        capacity=capacity, n_boundaries=n,
        gold_pairs=gold, gold_mask=gold_mask,
    )[:3]
    expected = _assemble_reference(
        starts, ends, scores, valid & (ends > starts), query_mask,
        capacity, n, gold, gold_mask,
    )
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))


@pytest.mark.parametrize("keep_absent", [False, True])
def test_hard_negatives_matches_reference(keep_absent):
    torch.manual_seed(7)
    logits = torch.randn(2, 3, 11)
    labels = (torch.rand(2, 3, 11) > 0.8).float()
    labels[0, 0] = 0
    valid = torch.rand(2, 3, 11) > 0.15
    actual = select_hard_negative_candidates(
        logits, labels, valid,
        negatives_per_positive=3,
        minimum_negatives=2,
        keep_all_when_no_positive=keep_absent,
    )
    expected = torch.zeros_like(valid)
    for bi in range(2):
        for qi in range(3):
            positive = (labels[bi, qi] > 0.5) & valid[bi, qi]
            negative = (labels[bi, qi] <= 0.5) & valid[bi, qi]
            count = int(positive.sum())
            n_keep = logits.shape[-1] if keep_absent and count == 0 else max(2, 3 * count)
            indices = negative.nonzero().flatten()
            order = torch.argsort(logits[bi, qi, indices], descending=True, stable=True)
            expected[bi, qi, indices[order[:n_keep]]] = True
            expected[bi, qi] |= positive
    assert torch.equal(actual, expected)


def test_select_top_boundaries_matches_reference_with_ties_and_invalid():
    logits = torch.tensor([[[2.0, 2.0, -1.0, 2.0], [3.0, 1.0, 0.0, -2.0]]])
    valid = torch.tensor([[[True, True, False, True], [False, False, False, False]]])
    scores, indices, selected_valid = select_top_boundaries(logits, valid, 8)
    assert indices[0, 0].tolist() == [0, 1, 3, 0]
    assert selected_valid[0, 0].tolist() == [True, True, True, False]
    assert not selected_valid[0, 1].any()
    assert torch.equal(scores[~selected_valid], torch.zeros_like(scores[~selected_valid]))


def test_decode_candidates_matches_reference_order():
    candidates = CandidateTensorBatch(
        indices=torch.tensor([[[[2, 4], [0, 1], [1, 3], [3, 4]]]]),
        proposal_logits=torch.zeros(1, 1, 4),
        pair_logits=torch.tensor([[[0.0, 2.0, 2.0, -3.0]]]),
        valid_mask=torch.tensor([[[True, True, True, False]]]),
        query_mask=torch.tensor([[True]]),
    )
    assert decode_candidates(candidates, threshold=0.5) == [[[(0, 1), (1, 3), (2, 4)]]]
