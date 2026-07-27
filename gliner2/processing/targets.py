"""Canonical, architecture-neutral training targets and coordinate utilities.

All span coordinates are half-open ``[start, end)`` token boundaries. Character
offsets index the *original* (unmutated) text. These structures are consumed by
the boundary model and by target validation; they never silently drop or
truncate gold annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import torch


class TargetCapacityError(ValueError):
    """Raised when unique gold targets exceed the configured capacity."""


# =============================================================================
# Canonical target classes
# =============================================================================

@dataclass(frozen=True)
class MentionTarget:
    query_id: int
    start: int
    end: int
    annotation_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(
                f"MentionTarget requires half-open end > start, got [{self.start}, {self.end})"
            )


@dataclass(frozen=True)
class ChoiceTarget:
    query_id: int
    selected_choice_ids: Tuple[int, ...]


@dataclass(frozen=True)
class InstanceTarget:
    instance_id: str
    schema_node_id: int
    anchor_annotation_id: Optional[str] = None


@dataclass(frozen=True)
class EdgeTarget:
    query_id: int
    source_id: str
    target_id: str


@dataclass(frozen=True)
class LiteralTarget:
    target_id: str
    value: Any


# -- Record / instance identity targets --------------------------------------

@dataclass(frozen=True)
class RecordFieldTarget:
    """Gold values that fill one field of one record instance.

    ``values`` is a tuple of *distinct gold values*; each value is itself a tuple
    of alternative half-open ``(start, end)`` spans (occurrence alternatives).
    Scalar fields hold at most one value; list fields may hold several. An empty
    ``values`` tuple means the field is absent for this instance.
    """

    query_id: int
    values: Tuple[Tuple[Tuple[int, int], ...], ...] = ()


@dataclass(frozen=True)
class RecordTarget:
    """One gold record/event instance with stable identity and field bindings."""

    instance_id: str
    task_index: int
    fields: Tuple[RecordFieldTarget, ...] = ()
    anchor_query_id: Optional[int] = None

    def field_for_query(self, query_id: int) -> Optional[RecordFieldTarget]:
        for f in self.fields:
            if f.query_id == query_id:
                return f
        return None


@dataclass(frozen=True)
class TargetGraph:
    mentions: Tuple[MentionTarget, ...] = ()
    choices: Tuple[ChoiceTarget, ...] = ()
    instances: Tuple[InstanceTarget, ...] = ()
    edges: Tuple[EdgeTarget, ...] = ()
    literals: Tuple[LiteralTarget, ...] = ()
    records: Tuple[RecordTarget, ...] = ()

    def mentions_for_query(self, query_id: int) -> Tuple[MentionTarget, ...]:
        return tuple(m for m in self.mentions if m.query_id == query_id)

    def records_for_task(self, task_index: int) -> Tuple[RecordTarget, ...]:
        return tuple(r for r in self.records if r.task_index == task_index)


# =============================================================================
# Coordinate conversions
# =============================================================================

def inclusive_tokens_to_boundary_pair(start: int, end_inclusive: int) -> Tuple[int, int]:
    """``(start, end_inclusive) -> (start, end_inclusive + 1)`` (half-open)."""
    return start, end_inclusive + 1


def char_span_to_word_boundaries(
    char_start: int,
    char_end: int,
    start_mappings: Sequence[int],
    end_mappings: Sequence[int],
) -> Tuple[int, int]:
    """Convert a character span to half-open word-token boundaries.

    ``start_mappings[i]`` is the character start of word token ``i`` and
    ``end_mappings[i]`` is its exclusive character end. Returns
    ``(token_start, token_end)`` half-open such that the covered tokens are
    ``[token_start, token_end)``.

    Raises:
        ValueError: if the character span does not align to any token, or is
            empty/inverted.
    """
    if char_end <= char_start:
        raise ValueError(f"empty/inverted char span [{char_start}, {char_end})")
    n = len(start_mappings)
    if n == 0 or len(end_mappings) != n:
        raise ValueError("invalid token offset mappings")

    token_start = None
    for i in range(n):
        if start_mappings[i] <= char_start < end_mappings[i] or start_mappings[i] == char_start:
            token_start = i
            break
        if start_mappings[i] > char_start:
            token_start = i
            break
    if token_start is None:
        raise ValueError(f"char_start {char_start} does not map into any token")

    token_end_inclusive = None
    for i in range(n - 1, -1, -1):
        if start_mappings[i] < char_end <= end_mappings[i] or end_mappings[i] == char_end:
            token_end_inclusive = i
            break
        if end_mappings[i] < char_end:
            token_end_inclusive = i
            break
    if token_end_inclusive is None or token_end_inclusive < token_start:
        raise ValueError(f"char_end {char_end} does not map to a token >= start")

    return token_start, token_end_inclusive + 1


def word_boundaries_to_char_span(
    token_start: int,
    token_end: int,
    start_mappings: Sequence[int],
    end_mappings: Sequence[int],
) -> Tuple[int, int]:
    """Inverse of :func:`char_span_to_word_boundaries` for half-open boundaries.

    ``char_start = start_mappings[token_start]``,
    ``char_end = end_mappings[token_end - 1]``.
    """
    if token_end <= token_start:
        raise ValueError(f"empty/inverted token span [{token_start}, {token_end})")
    if token_start < 0 or token_end > len(end_mappings):
        raise ValueError("token boundaries out of range")
    return int(start_mappings[token_start]), int(end_mappings[token_end - 1])


# =============================================================================
# Surface occurrence resolution
# =============================================================================

def normalize_surface_occurrences(
    matches: Sequence[Tuple[int, int]],
    *,
    occurrence_policy: str,
    query_id: int = -1,
    surface: str = "",
) -> List[Tuple[int, int]]:
    """Resolve repeated surface matches per an explicit occurrence policy.

    ``matches`` are half-open ``(start, end)`` token spans. Policies:
      * ``"all"``    - keep every occurrence.
      * ``"first"``  - keep only the first occurrence.
      * ``"error_on_ambiguous"`` - raise if more than one occurrence exists.
    """
    if occurrence_policy not in ("all", "first", "error_on_ambiguous"):
        raise ValueError(f"unknown occurrence_policy {occurrence_policy!r}")
    if not matches:
        return []
    if occurrence_policy == "all":
        return list(matches)
    if occurrence_policy == "first":
        return [matches[0]]
    # error_on_ambiguous
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous surface match: query_id={query_id} surface={surface!r} "
            f"has {len(matches)} occurrences; provide explicit offsets or use "
            "occurrence_policy='all'/'first'."
        )
    return [matches[0]]


# =============================================================================
# Dense per-query targets
# =============================================================================

def build_boundary_targets(
    graph: TargetGraph,
    query_count: int,
    text_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build dense start/end/inside targets from a target graph.

    Returns ``(start_targets, end_targets, inside_targets)`` with shapes
    ``[Q, L + 1]``, ``[Q, L + 1]``, ``[Q, L]`` respectively. Start and end are
    multi-label (nested spans can share a boundary). Inside marks tokens covered
    by at least one mention of that query.
    """
    start = torch.zeros(query_count, text_length + 1)
    end = torch.zeros(query_count, text_length + 1)
    inside = torch.zeros(query_count, text_length)
    for m in graph.mentions:
        if not (0 <= m.query_id < query_count):
            raise ValueError(f"mention query_id {m.query_id} out of range")
        if not (0 <= m.start < m.end <= text_length):
            raise ValueError(
                f"mention [{m.start}, {m.end}) out of range for text_length {text_length}"
            )
        start[m.query_id, m.start] = 1.0
        end[m.query_id, m.end] = 1.0
        inside[m.query_id, m.start:m.end] = 1.0
    return start, end, inside


# =============================================================================
# Padded target batch
# =============================================================================

@dataclass
class PaddedTargetBatch:
    """Padded gold targets for a batch.

    Shapes (``B`` samples, ``Q`` queries, ``G`` max gold per query):
        mention_pairs:         [B, Q, G, 2]  half-open [start, end)
        mention_mask:          [B, Q, G]
        start_targets:         [B, Q, L + 1]
        end_targets:           [B, Q, L + 1]
        inside_targets:        [B, Q, L] or None
        classification_targets:[B, Q_c] or None
    """
    mention_pairs: torch.LongTensor
    mention_mask: torch.BoolTensor
    start_targets: torch.Tensor
    end_targets: torch.Tensor
    inside_targets: Optional[torch.Tensor] = None
    classification_targets: Optional[torch.Tensor] = None
    instance_targets: Any = None
    edge_targets: Any = None
    # Per-sample record instance targets: List[List[RecordTarget]] or None.
    # Kept as opaque Python objects (span-based, device-independent) so the
    # record loss can map gold spans to per-sample candidate indices.
    records: Any = None

    def to(self, device) -> "PaddedTargetBatch":
        def mv(t):
            return t.to(device) if t is not None else None
        return PaddedTargetBatch(
            mention_pairs=self.mention_pairs.to(device),
            mention_mask=self.mention_mask.to(device),
            start_targets=self.start_targets.to(device),
            end_targets=self.end_targets.to(device),
            inside_targets=mv(self.inside_targets),
            classification_targets=mv(self.classification_targets),
            instance_targets=self.instance_targets,
            edge_targets=self.edge_targets,
            records=self.records,
        )

    def pin_memory(self) -> "PaddedTargetBatch":
        def pin(t):
            return t.pin_memory() if t is not None else None
        return PaddedTargetBatch(
            mention_pairs=self.mention_pairs.pin_memory(),
            mention_mask=self.mention_mask.pin_memory(),
            start_targets=self.start_targets.pin_memory(),
            end_targets=self.end_targets.pin_memory(),
            inside_targets=pin(self.inside_targets),
            classification_targets=pin(self.classification_targets),
            instance_targets=self.instance_targets,
            edge_targets=self.edge_targets,
            records=self.records,
        )


def pad_target_graphs(
    graphs: Sequence[TargetGraph],
    query_counts: Sequence[int],
    text_lengths: Sequence[int],
    max_gold_per_query: int,
) -> PaddedTargetBatch:
    """Pad a list of target graphs into a ``PaddedTargetBatch``.

    Every unique gold mention is retained. If any (sample, query) has more
    unique mentions than ``max_gold_per_query`` a :class:`TargetCapacityError`
    is raised — gold is never silently truncated.
    """
    b = len(graphs)
    if not (len(query_counts) == b and len(text_lengths) == b):
        raise ValueError("graphs, query_counts, text_lengths must be equal length")
    q = max(query_counts) if query_counts else 0
    l = max(text_lengths) if text_lengths else 0
    g = max_gold_per_query

    mention_pairs = torch.zeros(b, q, g, 2, dtype=torch.long)
    mention_mask = torch.zeros(b, q, g, dtype=torch.bool)
    start_targets = torch.zeros(b, q, l + 1)
    end_targets = torch.zeros(b, q, l + 1)
    inside_targets = torch.zeros(b, q, l)

    for bi, (graph, qc, tl) in enumerate(zip(graphs, query_counts, text_lengths)):
        # Group unique mentions per query.
        per_query: dict = {}
        for m in graph.mentions:
            if not (0 <= m.query_id < qc):
                raise ValueError(
                    f"sample={bi} mention query_id {m.query_id} outside query_count {qc}"
                )
            if not (0 <= m.start < m.end <= tl):
                raise ValueError(
                    f"sample={bi} mention [{m.start}, {m.end}) out of range for length {tl}"
                )
            per_query.setdefault(m.query_id, [])
            pair = (m.start, m.end)
            if pair not in per_query[m.query_id]:
                per_query[m.query_id].append(pair)

        for qi, pairs in per_query.items():
            if len(pairs) > g:
                raise TargetCapacityError(
                    f"sample={bi} query_id={qi} contains {len(pairs)} gold spans, "
                    f"but max_gold_per_query={g}. Increase "
                    "boundary_head.max_gold_per_query and "
                    "boundary_head.training_candidate_budget."
                )
            for k, (s, e) in enumerate(pairs):
                mention_pairs[bi, qi, k, 0] = s
                mention_pairs[bi, qi, k, 1] = e
                mention_mask[bi, qi, k] = True
                start_targets[bi, qi, s] = 1.0
                end_targets[bi, qi, e] = 1.0
                inside_targets[bi, qi, s:e] = 1.0

    records = [list(graph.records) for graph in graphs]
    has_records = any(records)
    return PaddedTargetBatch(
        mention_pairs=mention_pairs,
        mention_mask=mention_mask,
        start_targets=start_targets,
        end_targets=end_targets,
        inside_targets=inside_targets,
        records=records if has_records else None,
    )


# =============================================================================
# Truncation policy
# =============================================================================

def apply_truncation_policy(
    graph: TargetGraph,
    truncated_length: int,
    policy: str,
) -> Optional[TargetGraph]:
    """Apply a truncation policy to mentions that cross ``truncated_length``.

    Policies:
      * ``"error"``        - raise if any mention ends beyond the truncation.
      * ``"drop_target"``  - drop only mentions that cross the boundary.
      * ``"drop_example"`` - return ``None`` if any mention crosses.
    """
    if policy not in ("error", "drop_target", "drop_example"):
        raise ValueError(f"unknown truncation policy {policy!r}")
    crossing = [m for m in graph.mentions if m.end > truncated_length]
    if not crossing:
        return graph
    if policy == "error":
        raise ValueError(
            f"{len(crossing)} gold mention(s) cross truncation length "
            f"{truncated_length}; span [{crossing[0].start}, {crossing[0].end})"
        )
    if policy == "drop_example":
        return None
    kept = tuple(m for m in graph.mentions if m.end <= truncated_length)
    return TargetGraph(
        mentions=kept,
        choices=graph.choices,
        instances=graph.instances,
        edges=graph.edges,
        literals=graph.literals,
        records=graph.records,
    )
