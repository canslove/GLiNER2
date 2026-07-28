"""Boundary marginal heads: shapes, masking, inside prefix identity."""

from __future__ import annotations

import torch

from gliner2.models.boundary.constants import MASK_LOGIT
from gliner2.models.boundary.heads import BoundaryQueryHead


def _inputs(b=2, l=5, h=12, d=8, q=3):
    torch.manual_seed(0)
    boundary_states = torch.randn(b, l + 1, d)
    text_states = torch.randn(b, l, h)
    query_states = torch.randn(b, q, h)
    boundary_mask = torch.ones(b, l + 1, dtype=torch.bool)
    text_mask = torch.ones(b, l, dtype=torch.bool)
    query_mask = torch.ones(b, q, dtype=torch.bool)
    return boundary_states, text_states, query_states, boundary_mask, text_mask, query_mask


def test_head_output_shapes_and_finite():
    b, l, h, d, q = 2, 5, 12, 8, 3
    bs, ts, qs, bm, tm, qm = _inputs(b, l, h, d, q)
    head = BoundaryQueryHead(hidden_size=h, boundary_dim=d, query_dim=h)
    out = head(bs, bm, ts, tm, qs, qm)
    assert out.start_logits.shape == (b, q, l + 1)
    assert out.end_logits.shape == (b, q, l + 1)
    assert out.inside_logits.shape == (b, q, l)
    assert out.inside_prefix.shape == (b, q, l + 1)
    assert torch.isfinite(out.start_logits).all()


def test_head_masks_invalid_boundaries_and_queries():
    b, l, h, d, q = 2, 5, 12, 8, 3
    bs, ts, qs, bm, tm, qm = _inputs(b, l, h, d, q)
    bm[0, -1] = False            # invalid final boundary for sample 0
    qm[1, 2] = False             # invalid query 2 for sample 1
    head = BoundaryQueryHead(hidden_size=h, boundary_dim=d, query_dim=h)
    out = head(bs, bm, ts, tm, qs, qm)
    assert (out.start_logits[0, :, -1] == MASK_LOGIT).all()
    assert (out.start_logits[1, 2] == MASK_LOGIT).all()
    assert (out.inside_logits[1, 2] == MASK_LOGIT).all()


def test_inside_prefix_difference_equals_interval_sum():
    b, l, h, d, q = 1, 6, 12, 8, 2
    bs, ts, qs, bm, tm, qm = _inputs(b, l, h, d, q)
    head = BoundaryQueryHead(hidden_size=h, boundary_dim=d, query_dim=h)
    out = head(bs, bm, ts, tm, qs, qm)
    # For fully valid inputs, prefix[j] - prefix[i] == sum inside_logits[i:j]
    i, j = 1, 4
    expected = out.inside_logits[0, 0, i:j].sum()
    got = (
        out.inside_prefix[0, 0, j]
        - out.inside_prefix[0, 0, i]
        + out.inside_prefix_mean[0, 0, 0] * (j - i)
    )
    assert torch.allclose(got, expected, atol=1e-6)
    # prefix starts at zero
    assert torch.allclose(out.inside_prefix[..., 0], torch.zeros_like(out.inside_prefix[..., 0]))
