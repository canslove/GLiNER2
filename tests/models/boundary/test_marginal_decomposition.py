"""Marginal decomposition + dead-slot masking (Findings 7 & 6 / Phase 3.1-3.2).

Two properties:
  * The proposer exposes a marginal-free compatibility prior, so the reranker
    adds start/end marginals exactly once (no double-count).
  * Padded / dead top-k start slots cannot spawn candidates.
"""

from __future__ import annotations

import torch

from gliner2.models.boundary.proposal import ProposalSettings, SparseBoundaryProposer
from gliner2.models.boundary.scoring import SparseBoundaryPairScorer


def _settings(**overrides):
    base = dict(
        start_top_k=4, end_top_k=4, ends_per_start=3, starts_per_end=3,
        candidate_budget=8, training_candidate_budget=12, max_gold_per_query=6,
        end_block_size=4, bidirectional=True,
    )
    base.update(overrides)
    return ProposalSettings(**base)


def test_proposal_prior_is_marginal_free():
    torch.manual_seed(0)
    b, l, d, q = 1, 8, 8, 2
    proposer = SparseBoundaryProposer(boundary_dim=d, query_dim=d, settings=_settings()).eval()
    boundary_states = torch.randn(b, l + 1, d)
    query_states = torch.randn(b, q, d)
    boundary_mask = torch.ones(b, l + 1, dtype=torch.bool)
    query_mask = torch.ones(b, q, dtype=torch.bool)
    start_logits = torch.randn(b, q, l + 1)
    end_logits = torch.randn(b, q, l + 1)

    out = proposer(boundary_states, boundary_mask, query_states, query_mask, start_logits, end_logits)
    assert out.compat_logits is not None

    starts = out.indices[..., 0]
    ends = out.indices[..., 1]
    sm = torch.gather(start_logits, 2, starts)
    em = torch.gather(end_logits, 2, ends)
    valid = out.valid_mask
    # full prior == marginal-free compat + start marginal + end marginal.
    reconstructed = out.compat_logits + sm + em
    assert torch.allclose(out.logits[valid], reconstructed[valid], atol=1e-5)


def test_scorer_counts_start_marginal_exactly_once():
    torch.manual_seed(1)
    b, l, d, q = 1, 8, 8, 1
    proposer = SparseBoundaryProposer(boundary_dim=d, query_dim=d, settings=_settings()).eval()
    scorer = SparseBoundaryPairScorer(boundary_dim=d, query_dim=d, pair_dim=d).eval()

    boundary_states = torch.randn(b, l + 1, d)
    query_states = torch.randn(b, q, d)
    boundary_mask = torch.ones(b, l + 1, dtype=torch.bool)
    query_mask = torch.ones(b, q, dtype=torch.bool)
    start_logits = torch.randn(b, q, l + 1)
    end_logits = torch.randn(b, q, l + 1)
    text_lengths = torch.tensor([l])

    proposals = proposer(boundary_states, boundary_mask, query_states, query_mask, start_logits, end_logits)
    # pick a valid candidate and perturb the start marginal at its start boundary.
    vidx = proposals.valid_mask[0, 0].nonzero(as_tuple=False)
    assert vidx.numel() > 0
    ci = int(vidx[0])
    start_bd = int(proposals.indices[0, 0, ci, 0])

    delta = 3.0
    bumped = start_logits.clone()
    bumped[0, 0, start_bd] += delta
    with torch.no_grad():
        base = scorer(boundary_states, query_states, proposals, start_logits, end_logits, None, text_lengths)
        perturbed = scorer(boundary_states, query_states, proposals, bumped, end_logits, None, text_lengths)

    # The score for that candidate must change by exactly delta (marginal counted
    # once), not 2*delta (the old double-count).
    change = float(perturbed[0, 0, ci] - base[0, 0, ci])
    assert abs(change - delta) < 1e-4


def test_candidate_validity_is_independent_of_score_sentinel():
    # Validity comes from masks and span ordering, not a score threshold.
    torch.manual_seed(2)
    b, l, d, q = 1, 6, 8, 1
    proposer = SparseBoundaryProposer(
        boundary_dim=d, query_dim=d, settings=_settings(start_top_k=5)
    ).eval()
    boundary_states = torch.randn(b, l + 1, d)
    query_states = torch.randn(b, q, d)
    boundary_mask = torch.ones(b, l + 1, dtype=torch.bool)
    query_mask = torch.ones(b, q, dtype=torch.bool)

    neg_inf = float("-inf")
    start_logits = torch.full((b, q, l + 1), neg_inf)
    start_logits[0, 0, 2] = 5.0                       # only valid start
    end_logits = torch.full((b, q, l + 1), neg_inf)
    end_logits[0, 0, 4] = 5.0
    end_logits[0, 0, 5] = 5.0
    end_logits[0, 0, 6] = 5.0

    out = proposer(boundary_states, boundary_mask, query_states, query_mask, start_logits, end_logits)
    valid = out.valid_mask[0, 0]
    starts = out.indices[0, 0, :, 0][valid].tolist()
    assert starts, "expected at least one candidate"
    assert 2 in starts
    ends = out.indices[0, 0, :, 1][valid]
    starts_tensor = out.indices[0, 0, :, 0][valid]
    assert torch.all(ends > starts_tensor)
    assert torch.all(ends <= l)
