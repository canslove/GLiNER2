"""Span-content prefix pooling tests."""

import torch

from gliner2.models.boundary.content import SpanContentPooler


def test_prefix_pooling_matches_naive_mean():
    torch.manual_seed(9)
    pooler = SpanContentPooler(6, 4, dropout=0.0).eval()
    text = torch.randn(2, 9, 6)
    mask = torch.tensor(
        [[True] * 9, [True] * 7 + [False, False]], dtype=torch.bool
    )
    starts = torch.tensor([[[0, 2, 8]], [[0, 3, 6]]])
    ends = torch.tensor([[[1, 7, 9]], [[2, 7, 7]]])
    mean_prefix, lse_prefix = pooler.build_prefix(text, mask)
    actual = pooler.pool(mean_prefix, lse_prefix, starts, ends, text.dtype)
    values = pooler.value_projection(text)
    expected_rows = []
    for bi in range(2):
        row = []
        for start, end in zip(starts[bi, 0], ends[bi, 0]):
            row.append(values[bi, int(start):int(end)].float().mean(0))
        expected_rows.append(torch.stack(row))
    expected = pooler.layer_norm(torch.stack(expected_rows)).unsqueeze(1)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_soft_max_pool_has_declared_shape_and_is_finite():
    pooler = SpanContentPooler(
        8, 5, dropout=0.0, use_soft_max_pool=True
    ).eval()
    text = torch.randn(1, 6, 8)
    mask = torch.ones(1, 6, dtype=torch.bool)
    starts = torch.tensor([[[0, 1, 5]]])
    ends = torch.tensor([[[6, 4, 6]]])
    prefixes = pooler.build_prefix(text, mask)
    output = pooler.pool(*prefixes, starts, ends, text.dtype)
    assert output.shape == (1, 1, 3, 10)
    assert torch.isfinite(output).all()
