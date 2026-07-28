"""Cheap-expressiveness contracts for boundary PR-18 through PR-22."""

from __future__ import annotations

import math

import pytest
import torch

from gliner2.configuration import BoundaryHeadSettings, validate_boundary_head
from gliner2.models.boundary.losses import (
    build_candidate_labels,
    count_log_rate_loss,
    reranker_listwise_loss,
)
from gliner2.models.boundary.model import BoundaryHead, _group_scored_candidates
from gliner2.models.boundary.proposal import BoundaryProposals
from gliner2.models.boundary.scoring import SparseBoundaryPairScorer
from gliner2.models.outputs import CandidateTensorBatch
from gliner2.processing.targets import MentionTarget, TargetGraph, pad_target_graphs
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig


def test_multihead_one_reduces_to_original_gated_dot_product():
    torch.manual_seed(18)
    pair_dim = 8
    scorer = SparseBoundaryPairScorer(
        boundary_dim=pair_dim,
        query_dim=pair_dim,
        pair_dim=pair_dim,
        use_inside_evidence=False,
        dropout=0.0,
        multihead_pair_compat_heads=1,
    ).eval()
    with torch.no_grad():
        scorer.length_query_projection.weight.zero_()
        scorer.length_query_projection.bias.zero_()
    start_states = torch.randn(1, 2, 3, pair_dim)
    end_states = torch.randn(1, 2, 3, pair_dim)
    query_states = torch.randn(1, 2, pair_dim)
    indices = torch.tensor(
        [[[[0, 1], [1, 3], [2, 4]], [[0, 2], [1, 2], [2, 4]]]]
    )
    valid = torch.ones(1, 2, 3, dtype=torch.bool)
    zero = torch.zeros(1, 2, 3)
    proposals = BoundaryProposals(
        indices=indices,
        logits=zero,
        valid_mask=valid,
        compat_logits=zero,
        score_start_states=start_states,
        score_end_states=end_states,
    )
    actual = scorer(
        torch.randn(1, 5, pair_dim),
        query_states,
        proposals,
        torch.zeros(1, 2, 5),
        torch.zeros(1, 2, 5),
        None,
        torch.tensor([4]),
    )
    gate = torch.sigmoid(scorer.query_gate(query_states)).unsqueeze(2)
    expected = (start_states * gate * end_states).sum(-1) / math.sqrt(pair_dim)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    assert torch.equal(scorer.compat_mix.weight, torch.ones(1, 1))
    assert torch.equal(scorer.compat_mix.bias, torch.zeros(1))


def test_multihead_divisibility_validation_and_state_dict():
    with pytest.raises(ValueError, match="must be divisible"):
        validate_boundary_head(
            {"pair_dim": 10, "multihead_pair_compat_heads": 8}
        )
    head = BoundaryHead(
        16,
        BoundaryHeadSettings(
            boundary_dim=16,
            pair_dim=16,
            multihead_pair_compat_heads=8,
            enable_count_head=True,
            enable_abstention=True,
        ),
    )
    keys = set(head.state_dict())
    assert "pair_scorer.compat_mix.weight" in keys
    assert "count_head.weight" in keys
    assert "null_projection.weight" in keys


def test_candidate_soft_iou_targets_are_analytic():
    candidates = torch.tensor([[[[0, 4], [2, 6], [7, 9]]]])
    candidate_mask = torch.ones(1, 1, 3, dtype=torch.bool)
    gold = torch.tensor([[[[2, 6]]]])
    gold_mask = torch.ones(1, 1, 1, dtype=torch.bool)
    exact, soft = build_candidate_labels(
        candidates, candidate_mask, gold, gold_mask, return_iou=True
    )
    torch.testing.assert_close(
        soft, torch.tensor([[[1.0 / 3.0, 1.0, 0.0]]])
    )
    assert torch.equal(exact, torch.tensor([[[0.0, 1.0, 0.0]]]))


def test_soft_iou_weight_zero_has_exact_objective_parity():
    common = dict(
        boundary_dim=8,
        pair_dim=8,
        multihead_pair_compat_heads=1,
        start_top_k=4,
        end_top_k=4,
        ends_per_start=3,
        starts_per_end=3,
        candidate_budget=12,
        training_candidate_budget=12,
        max_gold_per_query=4,
        proposal_loss_weight=0.0,
        consistency_loss_weight=0.0,
        rerank_listwise_weight=0.0,
        enable_abstention=False,
        abstention_loss_weight=0.0,
        enable_count_head=False,
        count_loss_weight=0.0,
        negative_query_ratio=0.0,
        dropout=0.0,
    )
    disabled = BoundaryHead(
        8, BoundaryHeadSettings(**common, soft_iou_aux_weight=0.0)
    ).train()
    annealed = BoundaryHead(
        8, BoundaryHeadSettings(**common, soft_iou_aux_weight=0.2)
    ).train()
    annealed.load_state_dict(disabled.state_dict())
    annealed.set_soft_iou_scale(0.0)
    targets = pad_target_graphs(
        [TargetGraph(mentions=(MentionTarget(0, 1, 3),))],
        [1],
        [5],
        4,
    )
    inputs = (
        torch.randn(1, 5, 8),
        torch.ones(1, 5, dtype=torch.bool),
        torch.randn(1, 1, 8),
        torch.ones(1, 1, dtype=torch.bool),
        targets,
    )
    left = disabled(*inputs)
    right = annealed(*inputs)
    assert torch.equal(left.total_loss, right.total_loss)
    assert torch.equal(left.candidates.pair_logits, right.candidates.pair_logits)


def test_reranker_listwise_empty_and_monotonic_contracts():
    valid = torch.tensor([[[True, True], [True, False], [False, False]]])
    labels = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]])
    query_mask = torch.ones(1, 3, dtype=torch.bool)
    low = reranker_listwise_loss(
        torch.tensor([[[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]]),
        labels,
        valid,
        query_mask,
    )
    high = reranker_listwise_loss(
        torch.tensor([[[4.0, 2.0], [3.0, 0.0], [0.0, 0.0]]]),
        labels,
        valid,
        query_mask,
    )
    assert torch.isfinite(low) and torch.isfinite(high)
    assert high < low
    only_gold = reranker_listwise_loss(
        torch.tensor([[[3.0]]]),
        torch.ones(1, 1, 1),
        torch.ones(1, 1, 1, dtype=torch.bool),
        torch.ones(1, 1, dtype=torch.bool),
    )
    assert only_gold == 0
    no_gold = reranker_listwise_loss(
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
        torch.ones(1, 1, 2, dtype=torch.bool),
        torch.ones(1, 1, dtype=torch.bool),
    )
    assert torch.isfinite(no_gold) and no_gold == 0


def test_count_log_rate_poisson_semantics():
    mention_mask = torch.ones(1, 1, 3, dtype=torch.bool)
    query_mask = torch.ones(1, 1, dtype=torch.bool)
    optimum = count_log_rate_loss(
        torch.tensor([[math.log(3.0)]]), mention_mask, query_mask
    )
    low = count_log_rate_loss(torch.zeros(1, 1), mention_mask, query_mask)
    high = count_log_rate_loss(
        torch.tensor([[math.log(8.0)]]), mention_mask, query_mask
    )
    assert optimum < low
    assert optimum < high


def test_adaptive_decode_adds_to_threshold_hits_without_capping():
    candidates = CandidateTensorBatch(
        indices=torch.tensor([[[[0, 1], [1, 2], [2, 3], [3, 4]]]]),
        proposal_logits=None,
        pair_logits=torch.zeros(1, 1, 4),
        valid_mask=torch.ones(1, 1, 4, dtype=torch.bool),
        query_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    probabilities = torch.tensor([[[0.9, 0.4, 0.3, 0.2]]])
    fixed = _group_scored_candidates(
        candidates, threshold=0.5, probabilities=probabilities
    )
    adaptive = _group_scored_candidates(
        candidates,
        threshold=0.5,
        probabilities=probabilities,
        count_log_rates=torch.tensor([[math.log(3.0)]]),
        adaptive_threshold=True,
    )
    assert len(fixed[0][0]) == 1
    assert len(adaptive[0][0]) == 3

    threshold_hits = _group_scored_candidates(
        candidates,
        threshold=0.25,
        probabilities=probabilities,
        count_log_rates=torch.tensor([[math.log(1.0)]]),
        adaptive_threshold=True,
    )
    assert len(threshold_hits[0][0]) == 3


def test_pr18_pr22_defaults_and_schedules():
    settings = BoundaryHeadSettings()
    assert settings.multihead_pair_compat_heads == 8
    assert settings.proposal_loss_weight == 0.3
    assert settings.consistency_loss_weight == 0.1
    assert settings.rerank_listwise_weight == 0.3
    assert settings.soft_iou_aux_weight == 0.2
    assert settings.soft_iou_anneal_steps == 20_000
    assert settings.enable_count_head and settings.count_loss_weight == 0.2
    assert not settings.adaptive_threshold
    assert settings.enable_abstention and settings.abstention_loss_weight == 0.2
    assert settings.negative_query_ratio == 0.5

    config = TrainingConfig(fp16=False, bf16=False)
    assert config.gold_injection_hold_frac == 0.15
    assert config.gold_injection_end == 0.25
    schedule = ExtractorTrainer._gold_injection_probability
    assert schedule(0.0, 1.0, 0.25, 0.15) == 1.0
    assert schedule(0.15, 1.0, 0.25, 0.15) == 1.0
    assert schedule(1.0, 1.0, 0.25, 0.15) == 0.25
    soft_schedule = ExtractorTrainer._soft_iou_anneal_scale
    assert soft_schedule(0, 20_000) == 1.0
    assert soft_schedule(10_000, 20_000) == 0.5
    assert soft_schedule(20_000, 20_000) == 0.0
