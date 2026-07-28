"""Deterministic decoding of sparse span candidates.

Architecture-independent: consumes a ``CandidateSet`` (produced by the span
adapter or the boundary model) and turns it into thresholded, overlap-resolved,
ranked ``ScoredSpanCandidate`` results with exact half-open token→character
conversion.

Half-open ``[start, end)`` conversion to characters is exactly::

    char_start = start_mappings[start]
    char_end   = end_mappings[end - 1]

Stable ranking (guarantees deterministic ties):
    descending confidence, ascending start, ascending end, ascending query name.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from gliner2.models.base import QueryLayout, QuerySpec
from gliner2.models.candidates import CandidateSet, ScoredSpanCandidate


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def token_boundaries_to_character_offsets(
    start: int,
    end: int,
    start_mappings: Sequence[int],
    end_mappings: Sequence[int],
) -> Tuple[int, int]:
    """Exact half-open token→character conversion.

    ``char_start = start_mappings[start]``; ``char_end = end_mappings[end - 1]``.
    """
    if end <= start:
        raise ValueError(f"half-open span requires end > start, got [{start}, {end})")
    if start < 0 or end > len(end_mappings):
        raise ValueError(
            f"token boundaries [{start}, {end}) out of range for {len(end_mappings)} tokens"
        )
    return int(start_mappings[start]), int(end_mappings[end - 1])


def stable_candidate_sort_key(
    candidate: ScoredSpanCandidate,
    query_name: str = "",
) -> tuple:
    """Deterministic ranking key: -conf, start, end, query name."""
    return (-candidate.probability, candidate.start, candidate.end, query_name)


def _spans_overlap(a: ScoredSpanCandidate, b: ScoredSpanCandidate) -> bool:
    return a.start < b.end and b.start < a.end


def _strictly_nested(inner: ScoredSpanCandidate, outer: ScoredSpanCandidate) -> bool:
    return outer.start <= inner.start and inner.end <= outer.end and (
        outer.start < inner.start or inner.end < outer.end or inner is outer
    )


def apply_overlap_policy(
    candidates: Sequence[ScoredSpanCandidate],
    policy: str,
) -> List[ScoredSpanCandidate]:
    """Resolve overlaps deterministically within a single query.

    Policies:
      * ``"allow"``    - keep everything.
      * ``"nested"``   - keep spans that are either disjoint or fully nested
        relative to every higher-ranked kept span (reject partial/crossing
        overlaps only).
      * ``"disallow"`` - greedy highest-confidence; drop any span overlapping a
        kept span.
    """
    if policy not in ("allow", "nested", "disallow"):
        raise ValueError(f"unknown overlap_policy {policy!r}")
    if policy == "allow":
        return list(candidates)

    ranked = sorted(candidates, key=lambda c: stable_candidate_sort_key(c))
    kept: List[ScoredSpanCandidate] = []
    for cand in ranked:
        ok = True
        for k in kept:
            if not _spans_overlap(cand, k):
                continue
            if policy == "disallow":
                ok = False
                break
            # nested: allow only if fully contained in / containing k
            nested = (k.start <= cand.start and cand.end <= k.end) or (
                cand.start <= k.start and k.end <= cand.end
            )
            if not nested:
                ok = False
                break
        if ok:
            kept.append(cand)
    return kept


def decode_candidate_set(
    candidates: CandidateSet,
    query_layout: Optional[QueryLayout],
    *,
    thresholds: Mapping[int, float],
    overlap_policy: str,
    default_threshold: float = 0.5,
) -> List[ScoredSpanCandidate]:
    """Threshold, overlap-resolve (per query), and rank candidates."""
    by_query: Dict[int, List[ScoredSpanCandidate]] = {}
    for i in range(len(candidates)):
        qid = int(candidates.query_ids[i])
        logit = float(candidates.logits[i])
        prob = _sigmoid(logit)
        thr = thresholds.get(qid, default_threshold)
        if prob < thr:
            continue
        cand = ScoredSpanCandidate(
            query_id=qid,
            start=int(candidates.starts[i]),
            end=int(candidates.ends[i]),
            logit=logit,
            probability=prob,
        )
        by_query.setdefault(qid, []).append(cand)

    resolved: List[ScoredSpanCandidate] = []
    for qid, cands in by_query.items():
        resolved.extend(apply_overlap_policy(cands, overlap_policy))

    def query_name(qid: int) -> str:
        if query_layout is None:
            return ""
        try:
            return query_layout.query(qid).task_name
        except KeyError:
            return ""

    resolved.sort(key=lambda c: stable_candidate_sort_key(c, query_name(c.query_id)))
    return resolved


def format_candidate(
    candidate: ScoredSpanCandidate,
    text: str,
    query: Optional[QuerySpec],
    start_mappings: Optional[Sequence[int]] = None,
    end_mappings: Optional[Sequence[int]] = None,
    *,
    include_confidence: bool = False,
    include_spans: bool = False,
) -> Dict[str, Any]:
    """Format a single candidate into a result dict.

    When character mappings are provided the surface is sliced exactly as
    ``text[char_start:char_end]``; otherwise ``text`` is returned verbatim.
    """
    result: Dict[str, Any] = {}
    if query is not None:
        result["label"] = query.task_name

    if start_mappings is not None and end_mappings is not None:
        char_start, char_end = token_boundaries_to_character_offsets(
            candidate.start, candidate.end, start_mappings, end_mappings
        )
        result["text"] = text[char_start:char_end]
        if include_spans:
            result["char_start"] = char_start
            result["char_end"] = char_end
    else:
        result["text"] = text

    if include_spans:
        result["token_start"] = candidate.start
        result["token_end"] = candidate.end
    if include_confidence:
        result["confidence"] = candidate.probability
    return result


__all__ = [
    "ScoredSpanCandidate",
    "decode_candidate_set",
    "token_boundaries_to_character_offsets",
    "apply_overlap_policy",
    "stable_candidate_sort_key",
    "format_candidate",
    "RawSpan",
    "finalize_spans",
]


# --- final span selection ---

RawSpan = Tuple[str, float, int, int]


def finalize_spans(
    raw_spans: Sequence[RawSpan],
    *,
    dtype: str = "list",
    gate_open: bool = True,
    suppress: bool = True,
) -> List[RawSpan]:
    """Decode thresholded entity spans using the production contract."""
    if not gate_open:
        return []
    spans = sorted(raw_spans, key=lambda span: (-span[1], span[2], span[3]))
    if suppress:
        kept: List[RawSpan] = []
        for span in spans:
            _, _, start, end = span
            if any(
                not (end <= kept_start or start >= kept_end)
                for _, _, kept_start, kept_end in kept
            ):
                continue
            kept.append(span)
        spans = kept
    return spans if dtype == "list" else spans[:1]
