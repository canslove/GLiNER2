"""Recall-objective, consistency, abstention, and injection tests."""

import torch

from gliner2.models.boundary.losses import (
    abstention_loss,
    marginal_pair_consistency_loss,
    proposal_listwise_loss,
)
from gliner2.models.boundary.proposal import ProposalSettings, SparseBoundaryProposer


def test_proposal_listwise_loss_improves_when_gold_rises():
    valid = torch.ones(1, 1, 3, dtype=torch.bool)
    query = torch.ones(1, 1, dtype=torch.bool)
    gold = torch.tensor([[[True, False, False]]])
    low = proposal_listwise_loss(
        torch.tensor([[[-3.0, 2.0, 1.0]]]), gold, valid, query
    )
    high = proposal_listwise_loss(
        torch.tensor([[[3.0, 2.0, 1.0]]]), gold, valid, query
    )
    assert high < low
    assert proposal_listwise_loss(
        torch.zeros(1, 1, 3), torch.zeros_like(gold), valid, query
    ) == 0


def test_consistency_loss_is_zero_at_exact_agreement():
    probability = torch.tensor(0.4)
    logit = torch.logit(probability)
    pair_logits = logit.view(1, 1, 1)
    indices = torch.tensor([[[[0, 1]]]])
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    start = torch.tensor([[[logit, -8.0]]])
    end = torch.tensor([[[-8.0, logit]]])
    keep = torch.ones(1, 1, 2, dtype=torch.bool)
    loss = marginal_pair_consistency_loss(
        pair_logits, indices, valid, start, end, keep
    )
    assert float(loss) < 1e-10


def test_abstention_targets_absent_queries():
    logits = torch.tensor([[4.0, -4.0]], requires_grad=True)
    mentions = torch.tensor([[[False], [True]]])
    query = torch.ones(1, 2, dtype=torch.bool)
    loss = abstention_loss(logits, mentions, query)
    assert float(loss.detach()) < 0.05
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_zero_injection_probability_does_not_force_gold():
    settings = ProposalSettings(
        start_top_k=1,
        end_top_k=1,
        ends_per_start=1,
        starts_per_end=1,
        candidate_budget=2,
        training_candidate_budget=2,
        max_gold_per_query=1,
        end_block_size=8,
        bidirectional=False,
    )
    proposer = SparseBoundaryProposer(8, 8, settings).train()
    boundary = torch.randn(1, 9, 8)
    query_state = torch.randn(1, 1, 8)
    start_logits = torch.full((1, 1, 9), -10.0)
    end_logits = torch.full((1, 1, 9), -10.0)
    start_logits[..., 0] = 10
    end_logits[..., 1] = 10
    gold = torch.tensor([[[[6, 8]]]])
    output = proposer(
        boundary,
        torch.ones(1, 9, dtype=torch.bool),
        query_state,
        torch.ones(1, 1, dtype=torch.bool),
        start_logits,
        end_logits,
        gold_pairs=gold,
        gold_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        gold_injection_prob=0.0,
    )
    present = output.indices[output.valid_mask].view(-1, 2)
    assert not ((present == gold.view(1, 2)).all(-1)).any()
    assert not output.gold_mask.any()
