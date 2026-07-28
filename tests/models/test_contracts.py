"""Tests for architecture-neutral candidate/output contracts (PR 3)."""

from __future__ import annotations

import pytest
import torch

from gliner2.models.candidates import CandidateSet, ScoredSpanCandidate
from gliner2.models.outputs import (
    CandidateTensorBatch,
    PackedCandidateBatch,
    ExtractorOutput,
)


def _tiny_candidate_batch():
    # B=1, Q=2, C=2
    indices = torch.tensor([[[[0, 2], [3, 5]], [[1, 4], [0, 0]]]])
    proposal = torch.tensor([[[0.5, 0.1], [0.2, 0.0]]])
    pair = torch.tensor([[[1.0, 0.3], [0.7, 0.0]]])
    valid = torch.tensor([[[True, True], [True, False]]])
    query_mask = torch.tensor([[True, True]])
    return CandidateTensorBatch(indices, proposal, pair, valid, query_mask)


def test_scored_candidate_requires_half_open():
    with pytest.raises(ValueError):
        ScoredSpanCandidate(query_id=0, start=3, end=3, logit=0.0, probability=0.5)


def test_candidate_tensor_batch_validate_ok():
    ctb = _tiny_candidate_batch()
    ctb.validate(text_lengths=torch.tensor([5]))


def test_candidate_tensor_batch_validate_detects_out_of_range():
    ctb = _tiny_candidate_batch()
    with pytest.raises(ValueError):
        ctb.validate(text_lengths=torch.tensor([4]))  # candidate end=5 > 4


def test_pack_and_split_round_trip():
    ctb = _tiny_candidate_batch()
    packed = ctb.pack()
    assert len(packed) == 3  # two in q0, one in q1
    sets = packed.split_by_sample()
    assert len(sets) == 1
    s = sets[0]
    assert isinstance(s, CandidateSet)
    assert set(s.unique_spans()) == {(0, 0, 2), (0, 3, 5), (1, 1, 4)}


def test_extractor_output_mapping_access():
    out = ExtractorOutput(
        total_loss=torch.tensor(1.0),
        losses={"start_loss": torch.tensor(0.4), "pair_loss": torch.tensor(0.6)},
        batch_size=2,
    )
    assert float(out["total_loss"]) == pytest.approx(1.0)
    assert float(out["start_loss"]) == pytest.approx(0.4)
    assert out["batch_size"] == 2
    assert "pair_loss" in out
    assert "nonexistent" not in out
    assert out.get("nonexistent", 123) == 123


def test_extractor_output_declared_but_none_field_is_absent():
    # A declared field left as None must behave as absent: `.get` returns the
    # default, `in` is False, and `__getitem__` raises (consistent semantics).
    out = ExtractorOutput(batch_size=1)
    assert out.total_loss is None
    assert "total_loss" not in out
    sentinel = object()
    assert out.get("total_loss", sentinel) is sentinel
    with pytest.raises(KeyError):
        _ = out["total_loss"]
