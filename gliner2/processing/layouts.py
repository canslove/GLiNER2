"""Build architecture-neutral query layouts from schemas and explicit spans.

A ``QueryLayout`` enumerates the extractive and classification queries of a
sample. This module constructs one from the existing schema dict form (the
same shape ``SchemaTransformer`` consumes) so both architectures can share a
single query addressing scheme.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional

from gliner2.models.base import QueryLayout, QuerySpec


@dataclass(frozen=True)
class SpanAnnotation:
    """Explicit-offset span annotation. Offsets index the original text."""
    label: str
    start: int
    end: int
    text: Optional[str] = None

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(
                f"SpanAnnotation requires end > start, got [{self.start}, {self.end})"
            )


def build_query_layout(schema: Mapping[str, Any]) -> QueryLayout:
    """Build a :class:`QueryLayout` from a schema dict.

    Supported keys:
      * ``entities``: mapping ``label -> ...`` → one extractive query per label.
      * ``classifications``: list of ``{"task": name, "labels": [...]}`` →
        one classification query per task.
      * ``json_structures``: list of ``{parent: {field: ...}}`` → one
        extractive query per (parent, field).

    Query ids are assigned in a stable order: entities, then json-structure
    fields, then classifications.
    """
    queries: List[QuerySpec] = []
    qid = 0
    task_index = 0

    entities = schema.get("entities") or {}
    if entities:
        for role_index, label in enumerate(entities.keys()):
            queries.append(
                QuerySpec(
                    query_id=qid,
                    task_index=task_index,
                    task_type="entities",
                    task_name="entities",
                    role_index=role_index,
                    role_name=str(label),
                    field_path=(str(label),),
                    extractive=True,
                )
            )
            qid += 1
        task_index += 1

    for struct in schema.get("json_structures", []) or []:
        for parent, fields in struct.items():
            for role_index, fname in enumerate(fields.keys()):
                queries.append(
                    QuerySpec(
                        query_id=qid,
                        task_index=task_index,
                        task_type="json_structures",
                        task_name=str(parent),
                        role_index=role_index,
                        role_name=str(fname),
                        field_path=(str(parent), str(fname)),
                        extractive=True,
                    )
                )
                qid += 1
            task_index += 1

    for item in schema.get("classifications", []) or []:
        name = str(item.get("task", f"classification_{task_index}"))
        queries.append(
            QuerySpec(
                query_id=qid,
                task_index=task_index,
                task_type="classifications",
                task_name=name,
                role_index=0,
                role_name=name,
                field_path=(name,),
                extractive=False,
            )
        )
        qid += 1
        task_index += 1

    return QueryLayout(queries=tuple(queries))
