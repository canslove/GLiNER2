"""Build boundary query layouts and targets from transformed processor records.

Queries are enumerated in *transformed group order* — every extractive schema
child (``[E]``/``[C]``/``[R]`` marker) becomes one boundary query, in the same
order the boundary model's ``encode()`` builds its query states. Classification
groups contribute no extractive queries (they are scored by the shared
classifier). This ordering is deliberate so ``batch.targets`` aligns 1:1 with
``query_states`` even when the processor shuffles task order during training.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from gliner2.models.base import QueryLayout, QuerySpec
from gliner2.processing.records import (
    RecordFieldSpec,
    RecordSpec,
    compile_record_specs,
)
from gliner2.processing.targets import (
    MentionTarget,
    RecordFieldTarget,
    RecordTarget,
    TargetGraph,
    inclusive_tokens_to_boundary_pair,
    pad_target_graphs,
)
from gliner2.processing.validation import validate_target_graph

_EXTRACTIVE_MARKERS = ("[E]", "[C]", "[R]")


def _extractive_fields(schema_tokens: Sequence[str]) -> list[str]:
    return [
        str(schema_tokens[i + 1])
        for i, token in enumerate(schema_tokens[:-1])
        if token in _EXTRACTIVE_MARKERS
    ]


def _group_name(schema_tokens: Sequence[str], task_type: str) -> str:
    if task_type == "entities":
        return "entities"
    if len(schema_tokens) > 2:
        return str(schema_tokens[2]).split(" [DESCRIPTION] ")[0]
    return task_type


def _iter_inclusive_spans(value: Any) -> Iterator[tuple[int, int]]:
    """Yield inclusive ``(start, end)`` spans from a structure-label field.

    Field values are lists of ``(start, end)`` tuples (possibly nested for choice
    fields); ``(-1, -1)`` marks "not found" and is skipped.
    """
    if value is None:
        return
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(x, int) for x in value)
    ):
        if value != (-1, -1):
            yield value  # type: ignore[misc]
        return
    if isinstance(value, (list, tuple)):
        for sub in value:
            yield from _iter_inclusive_spans(sub)


def _resolve_field_spans(
    raw_value: Any, text_length: int
) -> List[Tuple[int, int]]:
    """Half-open in-range spans from a structure-label field value."""
    spans: List[Tuple[int, int]] = []
    for (s, e_inc) in _iter_inclusive_spans(raw_value):
        if 0 <= s <= e_inc < text_length:
            spans.append(inclusive_tokens_to_boundary_pair(s, e_inc))
    return spans


def _apply_occurrence_policy(
    spans: List[Tuple[int, int]],
    policy: str,
    fspec: RecordFieldSpec,
) -> List[Tuple[int, int]]:
    if not spans:
        return spans
    if not fspec.cardinality.is_scalar:
        # List fields keep every detected value; policy governs scalar surfaces.
        return spans
    if policy == "first":
        return [spans[0]]
    if policy == "error_on_ambiguous" and len(spans) > 1:
        raise ValueError(
            f"ambiguous scalar field {fspec.name!r}: {len(spans)} occurrences; "
            "provide explicit offsets or use occurrence_policy='latent_all'/'first'."
        )
    return spans


def _build_record_field_target(
    fspec: RecordFieldSpec,
    instance: Sequence[Any],
    text_length: int,
    policy: str,
) -> RecordFieldTarget:
    raw = instance[fspec.role_index] if fspec.role_index < len(instance) else None
    spans = _apply_occurrence_policy(
        _resolve_field_spans(raw, text_length), policy, fspec
    )
    if fspec.cardinality.is_scalar:
        values: Tuple[Tuple[Tuple[int, int], ...], ...] = (
            (tuple(spans),) if spans else ()
        )
    else:
        values = tuple((sp,) for sp in spans)
    return RecordFieldTarget(query_id=fspec.query_id, values=values)


def _build_sample_records(
    specs: Mapping[int, RecordSpec],
    sample_labels: Sequence[Any],
    text_length: int,
) -> List[RecordTarget]:
    records: List[RecordTarget] = []
    for task_index, spec in specs.items():
        if task_index >= len(sample_labels):
            continue
        labels = sample_labels[task_index]
        if not labels or labels[0] == 0:
            continue
        _, instances = labels
        for inst_idx, instance in enumerate(instances):
            fields_t = [
                _build_record_field_target(
                    fspec, instance, text_length, spec.occurrence_policy
                )
                for fspec in spec.fields
            ]
            records.append(
                RecordTarget(
                    instance_id=f"{task_index}:{inst_idx}",
                    task_index=task_index,
                    fields=tuple(fields_t),
                    anchor_query_id=spec.anchor_query_id,
                )
            )
    return records


def build_boundary_batch_metadata(
    *,
    schema_tokens_list: Sequence[Sequence[Sequence[str]]],
    task_types: Sequence[Sequence[str]],
    structure_labels: Sequence[Sequence[Any]],
    text_lengths: Sequence[int],
    is_training: bool,
    max_gold_per_query: int,
    record_metadata_list: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
    field_dtypes_list: Optional[Sequence[Optional[Mapping[str, Any]]]] = None,
) -> tuple:
    """Build layouts, optional padded targets, and compiled record specs.

    Supports every extractive task type (entities, JSON structures, relations);
    classification groups are skipped (no extractive queries). Missing surfaces
    (``(-1, -1)``) are treated as absent and skipped rather than raising, so
    optional fields and unlabeled types are handled gracefully. When a group
    declares record metadata, its gold instance grouping is preserved as
    :class:`RecordTarget` objects (identity by coordinate, never surface text).

    Returns ``(layouts, targets, record_specs)`` where ``record_specs`` is a
    per-sample tuple of ``{task_index: RecordSpec}`` mappings.
    """
    layouts = []
    graphs = []
    record_specs_out: List[Dict[int, RecordSpec]] = []

    batch_size = len(schema_tokens_list)
    if not (
        len(task_types) == len(structure_labels) == len(text_lengths) == batch_size
    ):
        raise ValueError("boundary metadata batch fields have inconsistent lengths")

    for sample_idx, (sample_schemas, sample_types, sample_labels, text_length) in enumerate(
        zip(schema_tokens_list, task_types, structure_labels, text_lengths)
    ):
        if not (len(sample_schemas) == len(sample_types) == len(sample_labels)):
            raise ValueError(
                f"boundary sample {sample_idx} schema/task/label counts do not match"
            )
        queries: list[QuerySpec] = []
        mentions: list[MentionTarget] = []
        query_id = 0

        for task_index, (schema_tokens, task_type, labels) in enumerate(
            zip(sample_schemas, sample_types, sample_labels)
        ):
            if task_type == "classifications":
                continue  # not an extractive query; handled by the classifier

            fields = _extractive_fields(schema_tokens)
            if not fields:
                continue
            name = _group_name(schema_tokens, task_type)

            field_query_ids = []
            for role_index, field in enumerate(fields):
                field_query_ids.append(query_id)
                queries.append(
                    QuerySpec(
                        query_id=query_id,
                        task_index=task_index,
                        task_type=task_type,
                        task_name=name,
                        role_index=role_index,
                        role_name=field,
                        field_path=(name, field) if name != "entities" else (field,),
                        extractive=True,
                    )
                )
                query_id += 1

            if not is_training or not labels or labels[0] == 0:
                continue

            _, instances = labels
            for instance in instances:
                for field_index, positions in enumerate(instance):
                    if field_index >= len(field_query_ids):
                        break
                    if task_type == "entities":
                        # Entities are always expected: a labeled surface that
                        # cannot be located is a genuine annotation error and
                        # must raise under strict training (never silently drop).
                        for start, end_inclusive in positions:
                            if (start, end_inclusive) == (-1, -1):
                                raise ValueError(
                                    f"entity {fields[field_index]!r} was not found "
                                    f"in sample {sample_idx}"
                                )
                            start, end = inclusive_tokens_to_boundary_pair(start, end_inclusive)
                            mentions.append(
                                MentionTarget(field_query_ids[field_index], start, end)
                            )
                    else:
                        # JSON/relation fields may legitimately be absent within
                        # an instance; treat "not found" as absent and skip.
                        for start, end_inclusive in _iter_inclusive_spans(positions):
                            start, end = inclusive_tokens_to_boundary_pair(start, end_inclusive)
                            mentions.append(
                                MentionTarget(field_query_ids[field_index], start, end)
                            )

        layout = QueryLayout(queries=tuple(queries))
        layouts.append(layout)

        sample_meta = (
            record_metadata_list[sample_idx]
            if record_metadata_list is not None and sample_idx < len(record_metadata_list)
            else None
        )
        sample_dtypes = (
            field_dtypes_list[sample_idx]
            if field_dtypes_list is not None and sample_idx < len(field_dtypes_list)
            else None
        )
        specs = compile_record_specs(
            query_layout=layout,
            record_metadata=sample_meta,
            field_dtypes=sample_dtypes,
        )
        record_specs_out.append(specs)

        if is_training:
            records = _build_sample_records(specs, sample_labels, text_length)
            graph = TargetGraph(mentions=tuple(mentions), records=tuple(records))
            validate_target_graph(graph, layout, text_length)
            graphs.append(graph)

    targets = None
    if is_training:
        targets = pad_target_graphs(
            graphs,
            [layout.extractive_count() for layout in layouts],
            text_lengths,
            max_gold_per_query=max_gold_per_query,
        )
    return tuple(layouts), targets, tuple(record_specs_out)


__all__ = ["build_boundary_batch_metadata"]
