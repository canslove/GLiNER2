"""Boundary self-attention masking and gradient tests."""

import torch

from gliner2.models.boundary.encoding import BoundaryAttentionBlock, BoundaryEncoder


def test_attention_masks_padding_without_nan():
    torch.manual_seed(4)
    block = BoundaryAttentionBlock(16, num_heads=4, window=2, dropout=0.0)
    states = torch.randn(2, 9, 16, requires_grad=True)
    mask = torch.tensor(
        [[True] * 9, [True] * 5 + [False] * 4], dtype=torch.bool
    )
    output = block(states, mask)
    assert torch.isfinite(output).all()
    assert torch.equal(output[1, 5:], torch.zeros_like(output[1, 5:]))
    output.sum().backward()
    assert torch.isfinite(states.grad).all()


def test_encoder_attention_layers_are_opt_in():
    legacy = BoundaryEncoder(12, 16, attention_layers=0)
    enabled = BoundaryEncoder(
        12, 16, attention_layers=2, attention_heads=4, attention_window=0
    )
    assert len(legacy.attention_blocks) == 0
    assert len(enabled.attention_blocks) == 2
    text = torch.randn(2, 7, 12)
    mask = torch.tensor([[True] * 7, [True] * 4 + [False] * 3])
    output = enabled(text, mask)
    assert output.states.shape == (2, 8, 16)
    assert torch.isfinite(output.states).all()
