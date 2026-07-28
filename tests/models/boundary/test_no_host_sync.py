"""CUDA guard against scalar host synchronization in the boundary hot path."""

import pytest
import torch

from gliner2.models.boundary.proposal import ProposalSettings, SparseBoundaryProposer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_proposer_forward_has_no_cuda_host_sync():
    device = torch.device("cuda")
    settings = ProposalSettings(
        start_top_k=8,
        end_top_k=8,
        ends_per_start=4,
        starts_per_end=4,
        candidate_budget=32,
        training_candidate_budget=40,
        max_gold_per_query=8,
        end_block_size=32,
        bidirectional=True,
    )
    proposer = SparseBoundaryProposer(32, 32, settings).to(device).eval()
    inputs = (
        torch.randn(2, 65, 32, device=device),
        torch.ones(2, 65, dtype=torch.bool, device=device),
        torch.randn(2, 4, 32, device=device),
        torch.ones(2, 4, dtype=torch.bool, device=device),
        torch.randn(2, 4, 65, device=device),
        torch.randn(2, 4, 65, device=device),
    )
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        with torch.no_grad():
            proposer(*inputs)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
