"""Relative-position and gated-proposer rotary tests."""

import torch

from gliner2.models.boundary.proposal import ProposalSettings, SparseBoundaryProposer
from gliner2.models.boundary.rotary import RotaryBoundaryEmbedding


def test_relative_position_property():
    torch.manual_seed(5)
    rotary = RotaryBoundaryEmbedding(64)
    start = torch.randn(64)
    end = torch.randn(64)
    values = []
    for offset in range(0, 40, 7):
        left = rotary(start, torch.tensor(offset))
        right = rotary(end, torch.tensor(offset + 5))
        values.append(left @ right)
    assert torch.allclose(torch.stack(values), values[0].expand(len(values)), atol=1e-4)


def test_rotary_proposer_runs_with_paired_gate():
    settings = ProposalSettings(
        start_top_k=4,
        end_top_k=4,
        ends_per_start=3,
        starts_per_end=3,
        candidate_budget=8,
        training_candidate_budget=12,
        max_gold_per_query=4,
        end_block_size=4,
        enable_rotary_endpoints=True,
    )
    proposer = SparseBoundaryProposer(8, 12, settings).eval()
    assert proposer.start_query_projection.out_features == 4
    output = proposer(
        torch.randn(2, 10, 8),
        torch.ones(2, 10, dtype=torch.bool),
        torch.randn(2, 3, 12),
        torch.ones(2, 3, dtype=torch.bool),
        torch.randn(2, 3, 10),
        torch.randn(2, 3, 10),
    )
    assert output.indices.shape == (2, 3, 8, 2)
