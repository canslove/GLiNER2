"""Tests for strict target-graph validation (PR 3)."""

from __future__ import annotations

import pytest

from gliner2.processing.layouts import build_query_layout
from gliner2.processing.targets import MentionTarget, TargetGraph
from gliner2.processing.validation import validate_target_graph


def _layout():
    return build_query_layout({"entities": {"company": "c", "person": "p"}})


def test_valid_graph_passes():
    layout = _layout()
    graph = TargetGraph(mentions=(MentionTarget(0, 0, 2), MentionTarget(1, 2, 4)))
    validate_target_graph(graph, layout, text_length=5)


def test_unknown_query_id_raises():
    layout = _layout()
    graph = TargetGraph(mentions=(MentionTarget(9, 0, 2),))
    with pytest.raises(ValueError):
        validate_target_graph(graph, layout, text_length=5)


def test_out_of_range_mention_raises():
    layout = _layout()
    graph = TargetGraph(mentions=(MentionTarget(0, 3, 10),))
    with pytest.raises(ValueError):
        validate_target_graph(graph, layout, text_length=5)
