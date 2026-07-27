"""Deterministic candidate decoding: thresholds, overlap policies, offsets."""

from __future__ import annotations

import math

import pytest
import torch

from gliner2.inference.candidate_decoder import (
    apply_overlap_policy,
    decode_candidate_set,
    format_candidate,
    token_boundaries_to_character_offsets,
)
from gliner2.models.candidates import CandidateSet, ScoredSpanCandidate


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _candidate_set(rows):
    # rows: list of (query_id, start, end, probability)
    q = torch.tensor([r[0] for r in rows], dtype=torch.long)
    s = torch.tensor([r[1] for r in rows], dtype=torch.long)
    e = torch.tensor([r[2] for r in rows], dtype=torch.long)
    lg = torch.tensor([_logit(r[3]) for r in rows], dtype=torch.float)
    return CandidateSet(query_ids=q, starts=s, ends=e, logits=lg)


def test_token_to_char_half_open_conversion():
    # tokens: "Apple"(0,5) "released"(6,14) "iPhone"(15,21)
    start_map = [0, 6, 15]
    end_map = [5, 14, 21]
    # half-open [0, 2) covers tokens 0..1 -> chars [0, 14)
    assert token_boundaries_to_character_offsets(0, 2, start_map, end_map) == (0, 14)
    # [2, 3) -> "iPhone"
    assert token_boundaries_to_character_offsets(2, 3, start_map, end_map) == (15, 21)
    with pytest.raises(ValueError):
        token_boundaries_to_character_offsets(1, 1, start_map, end_map)


def test_threshold_filters_low_confidence():
    cs = _candidate_set([(0, 0, 1, 0.9), (0, 2, 3, 0.2)])
    out = decode_candidate_set(cs, None, thresholds={}, overlap_policy="allow", default_threshold=0.5)
    assert len(out) == 1
    assert (out[0].start, out[0].end) == (0, 1)


def test_decode_overlapping_spans_allow_keeps_all():
    cs = _candidate_set([(0, 0, 4, 0.9), (0, 2, 6, 0.8)])
    out = decode_candidate_set(cs, None, thresholds={}, overlap_policy="allow")
    assert len(out) == 2


def test_disallow_overlap_policy_greedy_by_confidence():
    cands = [
        ScoredSpanCandidate(0, 0, 4, _logit(0.9), 0.9),
        ScoredSpanCandidate(0, 2, 6, _logit(0.8), 0.8),   # overlaps -> dropped
        ScoredSpanCandidate(0, 6, 9, _logit(0.7), 0.7),   # disjoint -> kept
    ]
    kept = apply_overlap_policy(cands, "disallow")
    spans = {(c.start, c.end) for c in kept}
    assert spans == {(0, 4), (6, 9)}


def test_nested_overlap_policy_allows_containment_not_crossing():
    cands = [
        ScoredSpanCandidate(0, 0, 10, _logit(0.95), 0.95),
        ScoredSpanCandidate(0, 2, 5, _logit(0.9), 0.9),    # nested -> kept
        ScoredSpanCandidate(0, 8, 12, _logit(0.85), 0.85),  # crosses -> dropped
    ]
    kept = apply_overlap_policy(cands, "nested")
    spans = {(c.start, c.end) for c in kept}
    assert spans == {(0, 10), (2, 5)}


def test_stable_ranking_is_deterministic():
    cs = _candidate_set([(1, 5, 6, 0.8), (0, 0, 1, 0.8), (0, 0, 2, 0.8)])
    out = decode_candidate_set(cs, None, thresholds={}, overlap_policy="allow")
    # equal confidence -> ascending start, then end
    keys = [(c.start, c.end) for c in out]
    assert keys == [(0, 1), (0, 2), (5, 6)]


def test_format_candidate_exact_text_slice():
    text = "Apple released iPhone"
    start_map = [0, 6, 15]
    end_map = [5, 14, 21]
    cand = ScoredSpanCandidate(0, 2, 3, _logit(0.9), 0.9)
    result = format_candidate(
        cand, text, None, start_map, end_map,
        include_confidence=True, include_spans=True,
    )
    assert result["text"] == "iPhone"
    assert text[result["char_start"]:result["char_end"]] == "iPhone"
    assert result["confidence"] == pytest.approx(0.9)
