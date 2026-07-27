"""Strict validation for architecture-neutral target graphs."""

from __future__ import annotations

from gliner2.models.base import QueryLayout
from gliner2.processing.targets import TargetGraph


def validate_target_graph(
    graph: TargetGraph,
    query_layout: QueryLayout,
    text_length: int,
) -> None:
    """Validate a target graph against a query layout and text length.

    Raises ``ValueError`` on any invalid annotation. This is the strict policy
    used during training: no annotation is silently dropped.
    """
    valid_ids = {q.query_id for q in query_layout.queries}
    for m in graph.mentions:
        if m.query_id not in valid_ids:
            raise ValueError(
                f"mention references unknown query_id {m.query_id}; "
                f"valid ids: {sorted(valid_ids)}"
            )
        if not (0 <= m.start < m.end <= text_length):
            raise ValueError(
                f"mention [{m.start}, {m.end}) invalid for text_length {text_length}"
            )
    for c in graph.choices:
        if c.query_id not in valid_ids:
            raise ValueError(f"choice references unknown query_id {c.query_id}")
    for e in graph.edges:
        if e.query_id not in valid_ids:
            raise ValueError(f"edge references unknown query_id {e.query_id}")

    seen_instance_ids = set()
    for record in graph.records:
        if record.instance_id in seen_instance_ids:
            raise ValueError(f"duplicate record instance_id {record.instance_id!r}")
        seen_instance_ids.add(record.instance_id)
        if record.anchor_query_id is not None and record.anchor_query_id not in valid_ids:
            raise ValueError(
                f"record {record.instance_id!r} anchor references unknown "
                f"query_id {record.anchor_query_id}"
            )
        for ft in record.fields:
            if ft.query_id not in valid_ids:
                raise ValueError(
                    f"record {record.instance_id!r} field references unknown "
                    f"query_id {ft.query_id}"
                )
            for value in ft.values:
                for (s, e) in value:
                    if not (0 <= s < e <= text_length):
                        raise ValueError(
                            f"record {record.instance_id!r} field {ft.query_id} span "
                            f"[{s}, {e}) invalid for text_length {text_length}"
                        )
