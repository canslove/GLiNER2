"""Tests for the span -> half-open boundary candidate adapter (PR 3)."""

from __future__ import annotations

import torch

from gliner2.models.span.adapter import (
    width_indices_to_boundaries,
    inclusive_tokens_to_boundary_pair,
    dense_span_scores_to_packed_candidates,
)
from gliner2.models.candidates import CandidateSet


def test_inclusive_scalar_conversion():
    assert inclusive_tokens_to_boundary_pair(3, 3) == (3, 4)
    assert inclusive_tokens_to_boundary_pair(0, 5) == (0, 6)


def test_width_indices_to_boundaries_adds_one_to_end():
    starts = torch.tensor([0, 2, 5])
    ends_inclusive = torch.tensor([0, 4, 5])
    out = width_indices_to_boundaries(starts, ends_inclusive)
    assert out.tolist() == [[0, 1], [2, 5], [5, 6]]


def test_dense_span_scores_to_packed_candidates_half_open():
    # B=1, Q=1, L=3, W=2. Mark spans (start=0,w=0)->[0,1) and (start=1,w=1)->[1,3)
    logits = torch.tensor([[[[0.9, 0.1], [0.2, 0.8], [0.3, 0.0]]]])
    valid = torch.zeros(1, 1, 3, 2, dtype=torch.bool)
    valid[0, 0, 0, 0] = True
    valid[0, 0, 1, 1] = True
    packed = dense_span_scores_to_packed_candidates(logits, valid, query_layouts=())
    assert packed.starts.tolist() == [0, 1]
    assert packed.ends.tolist() == [1, 3]
    # offsets delimit the single (sample, query) run.
    assert packed.offsets.tolist()[0] == 0
    assert packed.offsets.tolist()[-1] == 2
