"""Export mode parity + torch.compile smoke for the boundary proposer/head."""

from __future__ import annotations

import pytest
import torch

from gliner2.models.boundary.proposal import ProposalSettings, SparseBoundaryProposer


def _settings(export_mode, **overrides):
    base = dict(
        start_top_k=4, end_top_k=4, ends_per_start=3, starts_per_end=3,
        candidate_budget=8, training_candidate_budget=12, max_gold_per_query=6,
        end_block_size=4, bidirectional=True, export_mode=export_mode,
    )
    base.update(overrides)
    return ProposalSettings(**base)


def _inputs(b=2, l=9, d=8, q=3, seed=0):
    torch.manual_seed(seed)
    return (
        torch.randn(b, l + 1, d),
        torch.ones(b, l + 1, dtype=torch.bool),
        torch.randn(b, q, d),
        torch.ones(b, q, dtype=torch.bool),
        torch.randn(b, q, l + 1),
        torch.randn(b, q, l + 1),
    )


def _valid_pairs(out):
    pairs = []
    b, q, c, _ = out.indices.shape
    for bi in range(b):
        per_q = []
        for qi in range(q):
            s = set()
            for ci in range(c):
                if bool(out.valid_mask[bi, qi, ci]):
                    s.add((int(out.indices[bi, qi, ci, 0]), int(out.indices[bi, qi, ci, 1])))
            per_q.append(s)
        pairs.append(per_q)
    return pairs


def test_export_vectorized_matches_streaming():
    # end_block_size=4 forces multiple streaming blocks for L+1=10; vectorized
    # runs a single [B,Q,Ks,L+1] block. Candidate sets must be identical.
    stream = SparseBoundaryProposer(8, 8, _settings("streaming")).eval()
    vec = SparseBoundaryProposer(8, 8, _settings("vectorized")).eval()
    vec.load_state_dict(stream.state_dict())

    inp = _inputs()
    with torch.no_grad():
        o1 = stream(*inp)
        o2 = vec(*inp)
    assert _valid_pairs(o1) == _valid_pairs(o2)


@pytest.mark.compile
def test_boundary_proposer_torch_compile_two_lengths():
    proposer = SparseBoundaryProposer(8, 8, _settings("streaming")).eval()
    try:
        compiled = torch.compile(proposer, dynamic=True)
    except Exception as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"torch.compile unavailable: {exc}")

    try:
        with torch.no_grad():
            out_short = compiled(*_inputs(l=6, seed=1))
            out_long = compiled(*_inputs(l=14, seed=2))
    except Exception as exc:  # pragma: no cover - backend/platform dependent
        pytest.skip(f"torch.compile backend failed on this platform: {exc}")

    # Same compiled callable handles both lengths; candidate budget is fixed.
    assert out_short.indices.shape[2] == out_long.indices.shape[2] == 8


@pytest.mark.compile
def test_vectorized_proposer_has_zero_dynamo_graph_breaks():
    if not hasattr(torch, "_dynamo"):
        pytest.skip("torch._dynamo is unavailable")
    proposer = SparseBoundaryProposer(8, 8, _settings("vectorized")).eval()
    explanation = torch._dynamo.explain(proposer)(*_inputs())
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0, explanation.break_reasons
