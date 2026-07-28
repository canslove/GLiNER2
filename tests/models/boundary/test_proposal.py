"""Sparse boundary proposal: half-open pairs, dedup, gold injection, capacity,
empty queries, and linear/bounded materialization counters."""

from __future__ import annotations

import pytest
import torch

from gliner2.models.boundary.proposal import (
    ProposalSettings,
    SparseBoundaryProposer,
    deduplicate_boundary_pairs,
)


def _settings(**overrides):
    base = dict(
        start_top_k=4,
        end_top_k=4,
        ends_per_start=3,
        starts_per_end=3,
        candidate_budget=8,
        training_candidate_budget=12,
        max_gold_per_query=6,
        end_block_size=4,
        bidirectional=True,
        export_mode="streaming",
    )
    base.update(overrides)
    return ProposalSettings(**base)


def _inputs(b=2, l=8, d=8, q=3, seed=0):
    torch.manual_seed(seed)
    boundary_states = torch.randn(b, l + 1, d)
    query_states = torch.randn(b, q, d)
    boundary_mask = torch.ones(b, l + 1, dtype=torch.bool)
    query_mask = torch.ones(b, q, dtype=torch.bool)
    start_logits = torch.randn(b, q, l + 1)
    end_logits = torch.randn(b, q, l + 1)
    return boundary_states, boundary_mask, query_states, query_mask, start_logits, end_logits


def test_dedup_keeps_highest_score_and_stable_order():
    starts = torch.tensor([0, 0, 1, 0])
    ends = torch.tensor([2, 2, 3, 2])
    scores = torch.tensor([0.1, 0.9, 0.5, 0.3])
    valid = torch.ones(4, dtype=torch.bool)
    s, e, sc = deduplicate_boundary_pairs(starts, ends, scores, valid)
    # unique pairs (0,2) and (1,3); (0,2) kept with best score 0.9
    assert s.shape[0] == 2
    # verify by pairing
    pairs = list(zip(s.tolist(), e.tolist(), sc.tolist()))
    assert (0, 2, pytest.approx(0.9)) in [(a, b, pytest.approx(c)) for a, b, c in pairs]
    assert (1, 3, pytest.approx(0.5)) in [(a, b, pytest.approx(c)) for a, b, c in pairs]
    # descending by score
    assert sc[0] >= sc[1]


def _dedup_reference(starts, ends, scores, valid):
    """Straightforward dict-based reference for the vectorized dedup path."""
    best = {}
    for i in range(starts.shape[0]):
        if not bool(valid[i]):
            continue
        key = (int(starts[i]), int(ends[i]))
        sc = float(scores[i])
        if key not in best or sc > best[key]:
            best[key] = sc
    if not best:
        empty_l = torch.zeros(0, dtype=torch.long)
        empty_f = torch.zeros(0, dtype=scores.dtype)
        return empty_l, empty_l, empty_f
    items = list(best.items())
    s_t = torch.tensor([k[0] for k, _ in items], dtype=torch.long)
    e_t = torch.tensor([k[1] for k, _ in items], dtype=torch.long)
    sc_t = torch.tensor([v for _, v in items], dtype=scores.dtype)
    # score desc, start asc, end asc
    order = sorted(range(len(items)),
                   key=lambda i: (-float(sc_t[i]), int(s_t[i]), int(e_t[i])))
    idx = torch.tensor(order, dtype=torch.long)
    return s_t[idx], e_t[idx], sc_t[idx]


@pytest.mark.parametrize("seed", range(12))
def test_dedup_matches_reference_across_random_inputs(seed):
    torch.manual_seed(seed)
    n = int(torch.randint(0, 40, (1,)).item())
    # Small coordinate range forces frequent duplicate keys and score ties.
    starts = torch.randint(0, 5, (n,))
    ends = torch.randint(0, 5, (n,)) + starts + 1
    scores = torch.randint(0, 4, (n,)).to(torch.float32)  # integer scores -> exact ties
    valid = torch.rand(n) > 0.3 if n else torch.zeros(0, dtype=torch.bool)

    s, e, sc = deduplicate_boundary_pairs(starts, ends, scores, valid)
    rs, re, rsc = _dedup_reference(starts, ends, scores, valid)

    assert torch.equal(s, rs)
    assert torch.equal(e, re)
    assert torch.equal(sc, rsc)
    # Contract: descending score and unique keys.
    if s.numel() > 1:
        assert torch.all(sc[:-1] >= sc[1:])
    keys = {(int(a), int(b)) for a, b in zip(s.tolist(), e.tolist())}
    assert len(keys) == s.shape[0]


def test_dedup_empty_and_all_invalid_return_empty():
    starts = torch.tensor([0, 1, 2])
    ends = torch.tensor([1, 2, 3])
    scores = torch.tensor([0.5, 0.6, 0.7])
    none_valid = torch.zeros(3, dtype=torch.bool)
    for args in ((starts, ends, scores, none_valid),
                 (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long),
                  torch.zeros(0), torch.zeros(0, dtype=torch.bool))):
        s, e, sc = deduplicate_boundary_pairs(*args)
        assert s.numel() == 0 and e.numel() == 0 and sc.numel() == 0
        assert s.dtype == torch.long and sc.dtype == args[2].dtype


def test_proposals_are_half_open_and_shaped():
    s = _settings()
    prop = SparseBoundaryProposer(boundary_dim=8, query_dim=8, settings=s).eval()
    inp = _inputs()
    out = prop(*inp)
    b, q = inp[2].shape[0], inp[2].shape[1]
    assert out.indices.shape == (b, q, s.candidate_budget, 2)
    assert out.logits.shape == (b, q, s.candidate_budget)
    assert out.valid_mask.shape == (b, q, s.candidate_budget)
    starts = out.indices[..., 0][out.valid_mask]
    ends = out.indices[..., 1][out.valid_mask]
    assert (ends > starts).all()  # strictly half-open [start, end)
    assert torch.isfinite(out.logits[out.valid_mask]).all()


def test_proposals_allow_full_span_zero_to_l():
    # Force marginals so boundary 0 (start) and L (end) dominate.
    s = _settings()
    prop = SparseBoundaryProposer(boundary_dim=8, query_dim=8, settings=s).eval()
    b, l, d, q = 1, 6, 8, 1
    torch.manual_seed(1)
    boundary_states = torch.randn(b, l + 1, d)
    query_states = torch.randn(b, q, d)
    boundary_mask = torch.ones(b, l + 1, dtype=torch.bool)
    query_mask = torch.ones(b, q, dtype=torch.bool)
    start_logits = torch.full((b, q, l + 1), -5.0)
    start_logits[0, 0, 0] = 10.0
    end_logits = torch.full((b, q, l + 1), -5.0)
    end_logits[0, 0, l] = 10.0
    out = prop(boundary_states, boundary_mask, query_states, query_mask, start_logits, end_logits)
    valid = out.valid_mask[0, 0]
    pairs = out.indices[0, 0][valid].tolist()
    assert [0, l] in pairs


def test_deterministic_across_runs():
    s = _settings()
    prop = SparseBoundaryProposer(boundary_dim=8, query_dim=8, settings=s).eval()
    inp = _inputs(seed=3)
    a = prop(*inp)
    b = prop(*inp)
    assert torch.equal(a.indices, b.indices)
    assert torch.equal(a.valid_mask, b.valid_mask)
    assert torch.allclose(a.logits[a.valid_mask], b.logits[b.valid_mask])


def test_empty_query_yields_no_candidates():
    s = _settings()
    prop = SparseBoundaryProposer(boundary_dim=8, query_dim=8, settings=s).eval()
    inp = list(_inputs())
    boundary_states, boundary_mask, query_states, query_mask, start_logits, end_logits = inp
    query_mask[0, 1] = False
    out = prop(boundary_states, boundary_mask, query_states, query_mask, start_logits, end_logits)
    assert out.valid_mask[0, 1].sum() == 0


def test_gold_injection_recall_is_one():
    s = _settings()
    prop = SparseBoundaryProposer(boundary_dim=8, query_dim=8, settings=s).train()
    b, l, d, q = 1, 8, 8, 2
    torch.manual_seed(5)
    boundary_states = torch.randn(b, l + 1, d)
    query_states = torch.randn(b, q, d)
    boundary_mask = torch.ones(b, l + 1, dtype=torch.bool)
    query_mask = torch.ones(b, q, dtype=torch.bool)
    start_logits = torch.randn(b, q, l + 1)
    end_logits = torch.randn(b, q, l + 1)
    # Golds unlikely to be proposed on their own (include a long span 0..L).
    gold_pairs = torch.zeros(b, q, 2, 2, dtype=torch.long)
    gold_pairs[0, 0, 0] = torch.tensor([0, l])
    gold_pairs[0, 0, 1] = torch.tensor([2, 3])
    gold_pairs[0, 1, 0] = torch.tensor([1, 7])
    gold_mask = torch.zeros(b, q, 2, dtype=torch.bool)
    gold_mask[0, 0, :] = True
    gold_mask[0, 1, 0] = True
    out = prop(
        boundary_states, boundary_mask, query_states, query_mask,
        start_logits, end_logits, gold_pairs=gold_pairs, gold_mask=gold_mask,
    )
    assert out.gold_mask is not None
    for qi, golds in {0: [(0, l), (2, 3)], 1: [(1, 7)]}.items():
        present = set(
            tuple(x) for x in out.indices[0, qi][out.valid_mask[0, qi]].tolist()
        )
        for g in golds:
            assert g in present
        # every injected gold flagged in gold_mask
        gold_pairs_out = set(
            tuple(x) for x in out.indices[0, qi][out.gold_mask[0, qi]].tolist()
        )
        for g in golds:
            assert g in gold_pairs_out


def test_direct_capacity_violation_is_safely_trimmed():
    # Public configuration rejects this shape. A direct low-level caller still
    # receives a bounded result without a device synchronization or overflow.
    s = _settings(training_candidate_budget=2)
    prop = SparseBoundaryProposer(boundary_dim=8, query_dim=8, settings=s).train()
    b, l, d, q = 1, 8, 8, 1
    torch.manual_seed(6)
    boundary_states = torch.randn(b, l + 1, d)
    query_states = torch.randn(b, q, d)
    boundary_mask = torch.ones(b, l + 1, dtype=torch.bool)
    query_mask = torch.ones(b, q, dtype=torch.bool)
    start_logits = torch.randn(b, q, l + 1)
    end_logits = torch.randn(b, q, l + 1)
    gold_pairs = torch.tensor([[[[0, 1], [1, 2], [2, 3]]]], dtype=torch.long)
    gold_mask = torch.ones(b, q, 3, dtype=torch.bool)
    out = prop(
        boundary_states, boundary_mask, query_states, query_mask,
        start_logits, end_logits, gold_pairs=gold_pairs, gold_mask=gold_mask,
    )
    assert out.valid_mask.sum() == 2
    assert out.gold_mask.sum() == 2


def test_materialization_counters_bounded_and_linear():
    s = _settings()
    prop = SparseBoundaryProposer(boundary_dim=8, query_dim=8, settings=s).eval()

    def stats_for(l):
        b, d, q = 1, 8, 2
        torch.manual_seed(7)
        boundary_states = torch.randn(b, l + 1, d)
        query_states = torch.randn(b, q, d)
        boundary_mask = torch.ones(b, l + 1, dtype=torch.bool)
        query_mask = torch.ones(b, q, dtype=torch.bool)
        start_logits = torch.randn(b, q, l + 1)
        end_logits = torch.randn(b, q, l + 1)
        out = prop(
            boundary_states, boundary_mask, query_states, query_mask,
            start_logits, end_logits, return_stats=True,
        )
        return out.stats

    s16 = stats_for(16)
    s32 = stats_for(32)
    # Peak materialized block is bounded by [B,Q,K,end_block_size] — independent of L.
    assert s16.max_materialized_pair_elements == s32.max_materialized_pair_elements
    # Boundary/conditional work scales ~linearly (never quadratically) with L.
    assert s32.boundary_score_elements < 3 * s16.boundary_score_elements
    assert s32.conditional_pair_score_elements < 3 * s16.conditional_pair_score_elements
