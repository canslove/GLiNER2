"""Head-only synthetic overfit (blueprint 16.1).

Proves gradients reach every boundary component, long spans are learnable, and
proposal oracle recall + exact decode reach 1.0 — isolated from the encoder and
tokenizer, fast on CPU.
"""

from __future__ import annotations

import pytest
import torch

from gliner2.configuration import BoundaryHeadSettings
from gliner2.models.boundary.model import BoundaryHead, decode_candidates
from gliner2.training.metrics import (
    candidate_oracle_recall,
    exact_span_counts,
    f1_from_counts,
    gold_from_target_graphs,
)
from gliner2.processing.targets import MentionTarget, TargetGraph, pad_target_graphs

GOLD = {
    (0, 0): [(0, 1), (3, 41)],       # one-token and 38-token span
    (0, 1): [(10, 20), (14, 18)],    # nested
    (0, 2): [(44, 48)],              # ends at final boundary
    (1, 0): [(2, 46)],               # long span
    (1, 1): [],                      # empty query
    (1, 2): [(7, 8), (20, 35)],
}


def _build_targets(B, Q, L):
    graphs = []
    for bi in range(B):
        mentions = []
        for qi in range(Q):
            for (s, e) in GOLD[(bi, qi)]:
                mentions.append(MentionTarget(query_id=qi, start=s, end=e))
        graphs.append(TargetGraph(mentions=tuple(mentions)))
    targets = pad_target_graphs(graphs, [Q] * B, [L] * B, max_gold_per_query=32)
    return graphs, targets


def test_head_only_overfit_recovers_long_and_nested_spans():
    torch.manual_seed(13)
    B, L, Q, H = 2, 48, 3, 32
    settings = BoundaryHeadSettings(
        boundary_dim=24, pair_dim=24, start_top_k=16, end_top_k=16,
        ends_per_start=8, starts_per_end=8, candidate_budget=128,
        training_candidate_budget=160, max_gold_per_query=32,
        end_block_size=16, dropout=0.0,
        # This is the primary-objective overfit gate. Auxiliary objectives have
        # separate contracts and schedules in test_pr18_pr22_contracts.py.
        proposal_loss_weight=0.0,
        consistency_loss_weight=0.0,
        rerank_listwise_weight=0.0,
        soft_iou_aux_weight=0.0,
        enable_abstention=False,
        abstention_loss_weight=0.0,
        enable_count_head=False,
        count_loss_weight=0.0,
        negative_query_ratio=0.0,
    )
    model = BoundaryHead(H, settings, query_dim=H)

    token_states = torch.randn(B, L, H)
    text_mask = torch.ones(B, L, dtype=torch.bool)
    query_states = torch.randn(B, Q, H)
    query_mask = torch.ones(B, Q, dtype=torch.bool)
    graphs, targets = _build_targets(B, Q, L)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    initial_loss = None
    model.train()
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        output = model(token_states, text_mask, query_states, query_mask, targets)
        loss = output.total_loss
        assert torch.isfinite(loss)
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all()
        optimizer.step()

    final_loss = float(loss.detach())
    assert final_loss < 0.02
    assert final_loss < initial_loss * 0.05

    model.eval()
    with torch.no_grad():
        eval_out = model(token_states, text_mask, query_states, query_mask)

    assert candidate_oracle_recall(eval_out.candidates, targets) == 1.0

    predictions = decode_candidates(eval_out.candidates, threshold=0.5)
    gold_sets = gold_from_target_graphs(graphs, Q)
    tp, fp, fn = exact_span_counts(predictions, gold_sets)
    precision, recall, f1 = f1_from_counts(tp, fp, fn)
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0

    # Long / boundary-touching spans that legacy max_width=8 could not represent.
    flat = {span for per_q in predictions[0] for span in per_q}
    flat |= {span for per_q in predictions[1] for span in per_q}
    assert (3, 41) in flat
    assert (2, 46) in flat
    assert (44, 48) in flat


def test_head_components_all_receive_gradients():
    torch.manual_seed(13)
    B, L, Q, H = 2, 48, 3, 32
    settings = BoundaryHeadSettings(
        boundary_dim=24, pair_dim=24, start_top_k=16, end_top_k=16,
        ends_per_start=8, starts_per_end=8, candidate_budget=128,
        training_candidate_budget=160, max_gold_per_query=32,
        end_block_size=16, dropout=0.0,
    )
    model = BoundaryHead(H, settings, query_dim=H)
    token_states = torch.randn(B, L, H)
    text_mask = torch.ones(B, L, dtype=torch.bool)
    query_states = torch.randn(B, Q, H)
    query_mask = torch.ones(B, Q, dtype=torch.bool)
    _, targets = _build_targets(B, Q, L)

    model.train()
    out = model(token_states, text_mask, query_states, query_mask, targets)
    out.total_loss.backward()

    components = [
        "boundary_encoder",
        "boundary_query_head.start_boundary_projection",
        "boundary_query_head.end_boundary_projection",
        "boundary_query_head.inside_text_projection",
        "boundary_proposer.start_query_projection",
        "boundary_proposer.end_key_projection",
        "pair_scorer.start_endpoint_projection",
        "pair_scorer.end_endpoint_projection",
    ]
    for prefix in components:
        grads = [
            p.grad for n, p in model.named_parameters()
            if n.startswith(prefix) and p.grad is not None
        ]
        assert grads, f"no grad tensors for {prefix}"
        assert any(float(g.abs().sum()) > 0 for g in grads), f"zero gradient for {prefix}"


@pytest.mark.parametrize(
    "features",
    [
        {"enable_span_content": True},
        {"enable_rotary_endpoints": True},
        {"boundary_attention_layers": 1},
        {
            "enable_span_content": True,
            "enable_rotary_endpoints": True,
            "boundary_attention_layers": 1,
            "query_conditioned_inside_weight": True,
            "endpoint_difference_features": True,
        },
    ],
)
def test_optional_representation_features_overfit_small_batch(features):
    torch.manual_seed(21)
    settings = BoundaryHeadSettings(
        boundary_dim=16,
        pair_dim=16,
        start_top_k=8,
        end_top_k=8,
        ends_per_start=4,
        starts_per_end=4,
        candidate_budget=32,
        training_candidate_budget=40,
        max_gold_per_query=4,
        end_block_size=16,
        content_dim=8,
        boundary_attention_heads=4,
        dropout=0.0,
        **features,
    )
    model = BoundaryHead(16, settings, query_dim=16)
    tokens = torch.randn(1, 20, 16)
    text_mask = torch.ones(1, 20, dtype=torch.bool)
    queries = torch.randn(1, 2, 16)
    query_mask = torch.ones(1, 2, dtype=torch.bool)
    targets = pad_target_graphs(
        [
            TargetGraph(
                mentions=(
                    MentionTarget(0, 2, 8),
                    MentionTarget(1, 11, 19),
                )
            )
        ],
        [2],
        [20],
        max_gold_per_query=4,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
    losses = []
    model.train()
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        loss = model(tokens, text_mask, queries, query_mask, targets).total_loss
        assert torch.isfinite(loss)
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0] * 0.5
