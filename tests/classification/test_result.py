"""Phase 7 gate: results are frozen, confidence follows the spec, and output
uses declaration order. No torch.
"""
from __future__ import annotations

import math

import pytest

from gliner2.classification.candidates import LocalAssignment
from gliner2.classification.compiler import compile_schema
from gliner2.classification.decoding.base import Solution
from gliner2.classification.errors import SchemaError
from gliner2.classification.result import ClassificationResult, ResultBuilder, Violation
from gliner2.classification.schema import ClassificationSchema
from gliner2.classification.scoring import ClassificationScores
from gliner2.joint_ie.result import _geometric_mean


def _scores(compiled, tasks):
    return ClassificationScores(
        text="doc", tasks=tasks, fingerprint=compiled.fingerprint,
        specs={s.name: s for s in compiled.task_specs})


def _solution(assignments, *, score=0.0, violations=(), decoder="exact", exact=True):
    return Solution(assignments=assignments, score=score, violations=violations,
                    exact=exact, decoder=decoder)


def _local(task, labels):
    return LocalAssignment(task, frozenset(labels), 0.0)


# ---- T-R3 : frozen -----------------------------------------------------

def test_result_and_maps_are_immutable():
    schema = ClassificationSchema().single("s", ["pos", "neg"])
    compiled = compile_schema(schema)
    scores = _scores(compiled, {"s": {"pos": 2.0, "neg": -1.0}})
    sol = _solution({"s": _local("s", ["pos"])})
    result = ResultBuilder().build(compiled, scores, sol)
    with pytest.raises(TypeError):
        result.tasks["s"] = None
    with pytest.raises(TypeError):
        result["s"].probabilities["pos"] = 0.0
    with pytest.raises(Exception):
        result.feasible = False


# ---- T-R4 : label / level ----------------------------------------------

def test_label_property_and_level():
    schema = (ClassificationSchema()
              .single("s", ["pos", "neg"])
              .multi("topics", ["a", "b", "c"])
              .ordinal("risk", ["safe", "low", "high"]))
    compiled = compile_schema(schema)
    scores = _scores(compiled, {
        "s": {"pos": 2.0, "neg": -1.0},
        "topics": {"a": 1.0, "b": 1.0, "c": -1.0},
        "risk": {"safe": -1.0, "low": 2.0, "high": 0.0},
    })
    sol = _solution({
        "s": _local("s", ["pos"]),
        "topics": _local("topics", ["a", "b"]),
        "risk": _local("risk", ["low"]),
    })
    result = ResultBuilder().build(compiled, scores, sol)
    assert result["s"].label == "pos"
    with pytest.raises(SchemaError):
        _ = result["topics"].label            # not single-label
    assert result["s"].level is None          # not ordered
    assert result["risk"].level == 1          # index of "low"


# ---- T-R5 : accessors --------------------------------------------------

def test_accessors():
    schema = (ClassificationSchema()
              .single("s", ["pos", "neg"])
              .multi("topics", ["a", "b", "c"]))
    compiled = compile_schema(schema)
    scores = _scores(compiled, {
        "s": {"pos": 2.0, "neg": -1.0},
        "topics": {"a": 1.0, "b": 1.0, "c": -1.0},
    })
    sol = _solution({"s": _local("s", ["pos"]),
                     "topics": _local("topics", ["b", "a"])})
    result = ResultBuilder().build(compiled, scores, sol)
    assert result.value("s") == "pos"
    assert result.value("topics") == ("a", "b")     # declaration order tuple
    assert result.selected("topics") == ("a", "b")
    assert isinstance(result.confidence("s"), float)
    assert result.utility("s", "pos") == pytest.approx(
        result["s"].utilities["pos"])
    with pytest.raises(SchemaError):
        result["missing"]


# ---- T-R8 : include_confidence toggles output --------------------------

def test_include_confidence_toggles_meta_and_confidence():
    schema = ClassificationSchema().single("s", ["pos", "neg"])
    compiled = compile_schema(schema)
    scores = _scores(compiled, {"s": {"pos": 2.0, "neg": -1.0}})
    sol = _solution({"s": _local("s", ["pos"])})

    with_conf = ResultBuilder().build(compiled, scores, sol, include_confidence=True)
    dict_conf = with_conf.to_dict()
    assert "_meta" in dict_conf
    assert dict_conf["s"]["confidence"] is not None

    without = ResultBuilder().build(compiled, scores, sol, include_confidence=False)
    dict_plain = without.to_dict()
    assert "_meta" not in dict_plain          # feasible + no confidence -> no meta
    assert dict_plain["s"] == "pos"


# ---- T-R9 : confidence semantics ---------------------------------------

def test_exclusive_confidence_is_softmax_prob():
    schema = ClassificationSchema().single("s", ["pos", "neg"])
    compiled = compile_schema(schema)
    scores = _scores(compiled, {"s": {"pos": 2.0, "neg": -1.0}})
    sol = _solution({"s": _local("s", ["pos"])})
    result = ResultBuilder().build(compiled, scores, sol)
    assert result.confidence("s") == pytest.approx(scores.probability("s", "pos"))


def test_multi_confidence_is_geometric_mean():
    schema = ClassificationSchema().multi("t", ["a", "b", "c"])
    compiled = compile_schema(schema)
    scores = _scores(compiled, {"t": {"a": 1.5, "b": -0.5, "c": 0.3}})
    sol = _solution({"t": _local("t", ["a"])})  # only a selected
    result = ResultBuilder().build(compiled, scores, sol)
    expected = _geometric_mean([
        scores.probability("t", "a"),
        1 - scores.probability("t", "b"),
        1 - scores.probability("t", "c"),
    ])
    assert result.confidence("t") == pytest.approx(expected)


def test_empty_selection_confidence():
    schema = (ClassificationSchema()
              .multi("with_def", ["a", "b"], default="a")
              .multi("no_def", ["x", "y"], min_labels=0))
    compiled = compile_schema(schema)
    scores = _scores(compiled, {"with_def": {"a": 0.0, "b": 0.0},
                                "no_def": {"x": 0.0, "y": 0.0}})
    sol = _solution({"with_def": _local("with_def", []),
                     "no_def": _local("no_def", [])})
    result = ResultBuilder().build(compiled, scores, sol)
    assert result.confidence("with_def") == 1.0     # empty with default
    assert result.confidence("no_def") is None       # empty without default


# ---- T-R10 : infeasible result -----------------------------------------

def test_infeasible_result_carries_violations_and_meta():
    schema = ClassificationSchema().single("s", ["pos", "neg"])
    compiled = compile_schema(schema)
    scores = _scores(compiled, {"s": {"pos": 2.0, "neg": -1.0}})

    class _FakeConstraint:
        def references(self):
            return frozenset({"s"})

    sol = _solution({"s": _local("s", ["pos"])},
                    violations=(_FakeConstraint(),), exact=True)
    result = ResultBuilder().build(compiled, scores, sol, include_confidence=False)
    assert not result.feasible
    assert len(result.violations) == 1
    assert isinstance(result.violations[0], Violation)
    assert result.violations[0].tasks == ("s",)
    # _meta present even without confidence, because the result is infeasible
    assert "_meta" in result.to_dict()


# ---- T-R11 : declaration order -----------------------------------------

def test_output_uses_declaration_order_not_selection_order():
    schema = ClassificationSchema().multi("t", ["first", "second", "third"])
    compiled = compile_schema(schema)
    scores = _scores(compiled, {"t": {"first": 1.0, "second": 1.0, "third": 1.0}})
    # selection built in reverse; output must still be declaration order.
    sol = _solution({"t": _local("t", ["third", "first"])})
    result = ResultBuilder().build(compiled, scores, sol)
    assert result.selected("t") == ("first", "third")
