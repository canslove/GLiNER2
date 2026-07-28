"""Mechanical-efficiency contracts for boundary PR-08 through PR-14."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from gliner2.configuration import BoundaryHeadSettings
from gliner2.models.boundary.indexing import gather_states
from gliner2.models.boundary.model import BoundaryHead
from gliner2.models.boundary.proposal import (
    ProposalSettings,
    SparseBoundaryProposer,
    merge_running_topk,
)
from gliner2.models.boundary.targets_device import dense_targets_from_pairs
from gliner2.processing.targets import (
    MentionTarget,
    TargetGraph,
    pad_target_graphs,
)
from gliner2.training import ExtractorCollator
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model


def _old_gather(states, indices):
    b, n, d = states.shape
    q = indices.shape[1]
    expanded = states.unsqueeze(1).expand(b, q, n, d)
    return expanded.gather(
        2, indices.unsqueeze(-1).expand(-1, -1, -1, d)
    )


@pytest.mark.parametrize("shape", [(2, 9, 5, 3, 4), (1, 3, 2, 7, 1)])
def test_gather_states_exact_forward_and_backward(shape):
    b, n, d, q, c = shape
    torch.manual_seed(8)
    indices = torch.randint(n, (b, q, c))
    left = torch.randn(b, n, d, requires_grad=True)
    right = left.detach().clone().requires_grad_(True)
    actual = gather_states(left, indices)
    expected = _old_gather(right, indices)
    assert torch.equal(actual, expected)
    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    # Scatter-add accumulation order differs after flattening repeated indices;
    # values are equivalent to normal fp32 reduction tolerance.
    torch.testing.assert_close(left.grad, right.grad, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gather_states_backward_memory_is_not_query_expanded():
    b, n, d, q, c = 2, 257, 32, 32, 8
    device = torch.device("cuda")
    states = torch.randn(b, n, d, device=device, requires_grad=True)
    indices = torch.randint(n, (b, q, c), device=device)
    output = gather_states(states, indices)
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    output.sum().backward()
    peak = torch.cuda.max_memory_allocated() - before
    input_bytes = b * n * d * states.element_size()
    assert peak < 3 * input_bytes


def test_merge_running_topk_is_deterministic_with_ties():
    current_scores = torch.tensor([[[1.0, 1.0, 0.0]]])
    current_indices = torch.tensor([[[7, 2, 5]]])
    block_scores = torch.tensor([[[1.0, 0.5]]])
    block_indices = torch.tensor([[[1, 3]]])
    outputs = [
        merge_running_topk(
            current_scores, current_indices, block_scores, block_indices, 4
        )
        for _ in range(20)
    ]
    assert all(torch.equal(outputs[0][0], out[0]) for out in outputs[1:])
    assert all(torch.equal(outputs[0][1], out[1]) for out in outputs[1:])


def _proposal_settings(mode, block, threshold=16_777_216):
    return ProposalSettings(
        start_top_k=4,
        end_top_k=4,
        ends_per_start=3,
        starts_per_end=3,
        candidate_budget=12,
        training_candidate_budget=12,
        max_gold_per_query=4,
        end_block_size=block,
        export_mode=mode,
        vectorized_pair_elements=threshold,
    )


@pytest.mark.parametrize("length", [5, 63, 64, 65, 128])
@pytest.mark.parametrize("block", [16, 256])
def test_streaming_vectorized_and_auto_pairing_parity(length, block):
    torch.manual_seed(11)
    inputs = (
        torch.randn(1, length + 1, 8),
        torch.ones(1, length + 1, dtype=torch.bool),
        torch.randn(1, 2, 8),
        torch.ones(1, 2, dtype=torch.bool),
        torch.randn(1, 2, length + 1),
        torch.randn(1, 2, length + 1),
    )
    streaming = SparseBoundaryProposer(8, 8, _proposal_settings("streaming", block))
    vectorized = SparseBoundaryProposer(8, 8, _proposal_settings("vectorized", block))
    vectorized.load_state_dict(streaming.state_dict())
    auto = SparseBoundaryProposer(
        8, 8, _proposal_settings("auto", block, 10**9)
    )
    auto.load_state_dict(streaming.state_dict())
    outputs = [module.eval()(*inputs) for module in (streaming, vectorized, auto)]
    for other in outputs[1:]:
        assert torch.equal(outputs[0].indices, other.indices)
        assert torch.equal(outputs[0].valid_mask, other.valid_mask)
        assert torch.equal(outputs[0].logits, other.logits)


def test_proposer_projection_gradient_path_survives_selection_detach():
    settings = BoundaryHeadSettings(
        boundary_dim=8,
        pair_dim=8,
        start_top_k=4,
        end_top_k=4,
        ends_per_start=3,
        starts_per_end=3,
        candidate_budget=12,
        training_candidate_budget=12,
        max_gold_per_query=4,
        proposal_loss_weight=0.0,
        dropout=0.0,
    )
    head = BoundaryHead(8, settings).train()
    graph = TargetGraph(mentions=(MentionTarget(0, 1, 3),))
    targets = pad_target_graphs([graph], [2], [6], 4)
    output = head(
        torch.randn(1, 6, 8),
        torch.ones(1, 6, dtype=torch.bool),
        torch.randn(1, 2, 8),
        torch.ones(1, 2, dtype=torch.bool),
        targets,
    )
    output.total_loss.backward()
    proposer = head.boundary_proposer
    for projection in (
        proposer.start_pair_projection,
        proposer.end_key_projection,
        proposer.start_query_projection,
    ):
        assert projection.weight.grad is not None
        assert torch.count_nonzero(projection.weight.grad) > 0


def test_sparse_targets_reconstruct_dense_exactly():
    graphs = [
        TargetGraph(mentions=(
            MentionTarget(0, 0, 2),
            MentionTarget(0, 0, 2),
            MentionTarget(0, 3, 7),
        )),
        TargetGraph(),
    ]
    dense = pad_target_graphs(graphs, [2, 2], [7, 4], 5, build_dense=True)
    sparse = pad_target_graphs(graphs, [2, 2], [7, 4], 5, build_dense=False)
    actual = dense_targets_from_pairs(
        sparse.mention_pairs, sparse.mention_mask, 7
    )
    assert sparse.start_targets is None
    assert torch.equal(actual[0], dense.start_targets)
    assert torch.equal(actual[1], dense.end_targets)
    assert torch.equal(actual[2], dense.inside_targets)
    duplicate_pairs = torch.tensor([[[[0, 2], [0, 2], [3, 7]]]])
    duplicate_mask = torch.ones(1, 1, 3, dtype=torch.bool)
    start, end, inside = dense_targets_from_pairs(
        duplicate_pairs, duplicate_mask, 7
    )
    assert start[0, 0, 0] == 1 and end[0, 0, 2] == 1
    assert torch.equal(
        inside[0, 0], torch.tensor([1, 1, 0, 1, 1, 1, 1.0])
    )


def test_batched_encode_routing_matches_loop_and_roundtrips():
    model = build_tiny_boundary_model().eval()
    collator = ExtractorCollator(
        model.processor, is_training=True, architecture="boundary"
    )
    batch = collator([
        ("john works at acme", {"entities": {"person": ["john"], "company": ["acme"]}}),
        ("good product", {"classifications": [{
            "task": "sentiment",
            "labels": ["positive", "negative"],
            "true_label": ["positive"],
        }]}),
    ])
    with torch.no_grad():
        fast = model._encode_core(batch)
    fallback = replace(
        batch,
        text_word_mask=None,
        query_marker_indices=None,
        query_marker_mask=None,
    )
    with torch.no_grad():
        loop = model._encode_core(fallback)
    for key in ("text_states", "text_mask", "query_states", "query_mask"):
        assert torch.equal(fast[key], loop[key])
    for sample_index, layout in enumerate(batch.query_layouts):
        expected_groups = torch.tensor(
            [query.task_index for query in layout.queries], dtype=torch.long
        )
        count = expected_groups.numel()
        assert torch.equal(
            batch.query_group_index[sample_index, :count], expected_groups
        )
    moved = batch.to(torch.device("cpu"))
    for name in (
        "text_word_mask", "query_marker_indices", "query_marker_mask",
        "query_group_index", "cls_marker_indices", "cls_marker_mask",
        "cls_group_index",
    ):
        assert torch.equal(getattr(batch, name), getattr(moved, name))
    if torch.cuda.is_available():
        pinned = batch.pin_memory()
        assert pinned.query_marker_indices.is_pinned()


def test_boundary_config_defaults_to_auto_pairing():
    settings = BoundaryHeadSettings()
    assert settings.export_mode == "auto"
    assert settings.vectorized_pair_elements == 16_777_216


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_boundary_training_hot_path_has_no_cuda_host_sync():
    device = torch.device("cuda")
    settings = BoundaryHeadSettings(
        boundary_dim=8,
        pair_dim=8,
        start_top_k=4,
        end_top_k=4,
        ends_per_start=3,
        starts_per_end=3,
        candidate_budget=12,
        training_candidate_budget=12,
        max_gold_per_query=4,
        negative_query_ratio=1.0,
        dropout=0.0,
    )
    head = BoundaryHead(8, settings).to(device).train()
    targets = pad_target_graphs(
        [TargetGraph(mentions=(MentionTarget(0, 1, 3),))],
        [2],
        [6],
        4,
        build_dense=False,
    ).to(device)
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        output = head(
            torch.randn(1, 6, 8, device=device),
            torch.ones(1, 6, dtype=torch.bool, device=device),
            torch.randn(1, 2, 8, device=device),
            torch.ones(1, 2, dtype=torch.bool, device=device),
            targets,
        )
        output.total_loss.backward()
    finally:
        torch.cuda.set_sync_debug_mode(previous)
