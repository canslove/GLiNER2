"""Boundary state encoding: shapes, masks, BOS/EOS placement, padding."""

from __future__ import annotations

import torch

from gliner2.models.boundary.encoding import (
    BoundaryEncoder,
    build_boundary_mask,
    shift_left_with_bos,
    shift_right_with_eos,
)


def test_build_boundary_mask_counts_n_plus_one():
    lengths = torch.tensor([3, 5, 0])
    mask = build_boundary_mask(lengths, max_text_length=5)
    assert mask.shape == (3, 6)
    # boundary i valid iff i <= n
    assert mask.sum(dim=1).tolist() == [4, 6, 1]
    assert bool(mask[0, 0]) and bool(mask[0, 3]) and not bool(mask[0, 4])


def test_shift_left_places_bos_at_boundary_zero():
    states = torch.arange(2 * 3 * 4, dtype=torch.float).view(2, 3, 4)
    bos = torch.full((4,), -1.0)
    out = shift_left_with_bos(states, bos)
    assert out.shape == (2, 4, 4)
    assert torch.equal(out[:, 0], bos.expand(2, 4))
    assert torch.equal(out[:, 1:], states)


def test_shift_right_places_eos_at_each_final_boundary():
    states = torch.arange(2 * 3 * 4, dtype=torch.float).view(2, 3, 4)
    lengths = torch.tensor([3, 2])
    eos = torch.full((4,), 7.0)
    out = shift_right_with_eos(states, lengths, eos)
    assert out.shape == (2, 4, 4)
    # right(i) = token i for i < n; right(n) = EOS
    assert torch.equal(out[0, :3], states[0])
    assert torch.equal(out[0, 3], eos)
    assert torch.equal(out[1, :2], states[1, :2])
    assert torch.equal(out[1, 2], eos)


def test_boundary_encoder_shapes_and_padding_zeroed():
    torch.manual_seed(0)
    b, l, h, d = 2, 6, 12, 8
    enc = BoundaryEncoder(hidden_size=h, boundary_dim=d)
    text_states = torch.randn(b, l, h)
    text_mask = torch.zeros(b, l, dtype=torch.bool)
    text_mask[0, :4] = True   # n=4
    text_mask[1, :6] = True   # n=6

    out = enc(text_states, text_mask)
    assert out.states.shape == (b, l + 1, d)
    assert out.mask.shape == (b, l + 1)
    # boundary 0..n valid
    assert out.mask.sum(dim=1).tolist() == [5, 7]
    # padding boundary states are exactly zero
    pad = ~out.mask
    assert torch.count_nonzero(out.states[pad]) == 0
    assert torch.isfinite(out.states).all()



def test_boundary_encoder_refinement_parameters_receive_gradients():
    torch.manual_seed(1)
    enc = BoundaryEncoder(
        hidden_size=12,
        boundary_dim=8,
        dropout=0.0,
        refinement_layers=2,
        ffn_multiplier=1.5,
    )
    text_states = torch.randn(2, 5, 12, requires_grad=True)
    text_mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )

    out = enc(text_states, text_mask)
    weights = torch.randn_like(out.states)
    (out.states * weights).sum().backward()

    assert len(enc.refinement_blocks) == 2
    for block in enc.refinement_blocks:
        for parameter in block.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert torch.count_nonzero(parameter.grad) > 0


def test_boundary_encoder_zero_refinement_layers_preserves_masking():
    enc = BoundaryEncoder(
        hidden_size=12,
        boundary_dim=8,
        dropout=0.0,
        refinement_layers=0,
    )
    text_states = torch.randn(2, 4, 12)
    text_mask = torch.tensor(
        [[True, True, False, False], [True, True, True, True]]
    )

    out = enc(text_states, text_mask)

    assert len(enc.refinement_blocks) == 0
    assert out.states.shape == (2, 5, 8)
    assert out.mask.sum(dim=1).tolist() == [3, 5]
    assert torch.count_nonzero(out.states[~out.mask]) == 0
