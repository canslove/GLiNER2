"""Long-sequence half-precision numerical guards."""

import torch

from gliner2.models.boundary.heads import BoundaryQueryHead
from gliner2.models.boundary.scoring import interval_prefix_score


def test_inside_prefix_half_model_accumulates_in_fp32():
    torch.manual_seed(3)
    length = 4096
    head = BoundaryQueryHead(8, 8, query_dim=8, dropout=0.0).half()
    boundary_states = torch.randn(1, length + 1, 8).half()
    text_states = torch.randn(1, length, 8).half()
    query_states = torch.randn(1, 1, 8).half()
    boundary_mask = torch.ones(1, length + 1, dtype=torch.bool)
    text_mask = torch.ones(1, length, dtype=torch.bool)
    query_mask = torch.ones(1, 1, dtype=torch.bool)
    output = head(
        boundary_states,
        boundary_mask,
        text_states,
        text_mask,
        query_states,
        query_mask,
    )
    assert output.inside_prefix.dtype == torch.float32
    assert torch.isfinite(output.inside_prefix).all()
    starts = torch.tensor([[[17, 1024, 2048]]])
    ends = torch.tensor([[[4011, 2048, 4096]]])
    actual = interval_prefix_score(
        output.inside_prefix, starts, ends, output.inside_prefix_mean
    )
    expected = torch.stack(
        [
            output.inside_logits[0, 0, 17:4011].float().sum(),
            output.inside_logits[0, 0, 1024:2048].float().sum(),
            output.inside_logits[0, 0, 2048:4096].float().sum(),
        ]
    ).view(1, 1, 3)
    assert torch.allclose(actual, expected, atol=1e-3, rtol=1e-4)
