"""Tests for query layout construction (PR 3)."""

from __future__ import annotations

import pytest

from gliner2.processing.layouts import build_query_layout, SpanAnnotation


def test_span_annotation_requires_half_open():
    with pytest.raises(ValueError):
        SpanAnnotation("company", 5, 5)


def test_entities_layout_assigns_extractive_queries():
    schema = {"entities": {"company": "companies", "person": "people"}}
    layout = build_query_layout(schema)
    assert len(layout) == 2
    assert layout.extractive_count() == 2
    assert layout.classification_count() == 0
    assert [q.role_name for q in layout.queries] == ["company", "person"]


def test_mixed_schema_layout_orders_and_classifies():
    schema = {
        "entities": {"company": "c"},
        "classifications": [{"task": "sentiment", "labels": ["pos", "neg"]}],
        "json_structures": [{"employment": {"person": None, "org": None}}],
    }
    layout = build_query_layout(schema)
    # entities(1) + json_structure fields(2) + classification(1) = 4
    assert len(layout) == 4
    assert layout.extractive_count() == 3
    assert layout.classification_count() == 1
    cls = [q for q in layout.queries if not q.extractive]
    assert cls[0].task_name == "sentiment"
    # query ids are unique and contiguous.
    assert sorted(q.query_id for q in layout.queries) == [0, 1, 2, 3]


def test_query_lookup_by_id():
    schema = {"entities": {"a": 1, "b": 2}}
    layout = build_query_layout(schema)
    assert layout.query(1).role_name == "b"
    with pytest.raises(KeyError):
        layout.query(99)
