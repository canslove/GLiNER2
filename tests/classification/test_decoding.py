"""Phase 6 gate: the decoders enforce hard constraints exactly, fall back
without ever fabricating a feasible-looking violating answer, and walk the
infeasibility ladder in order. No torch.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gliner2.classification import constraints as C
from gliner2.classification.compiler import compile_schema
from gliner2.classification.decoding import (
    ExactDecoder,
    IndependentDecoder,
    MinViolationsDecoder,
    build_problem,
    decode,
    select_decoder,
)
from gliner2.classification.errors import InfeasibleError
from gliner2.classification.schema import ClassificationSchema
from gliner2.classification.scoring import ClassificationScores


def _cfg(**kw):
    base = dict(decoder="auto", exact_node_budget=200_000, beam_size=16,
                candidate_threshold=0.0, max_candidates_per_task=64,
                on_infeasible="raise")
    base.update(kw)
    return SimpleNamespace(**base)


def _scores(compiled, tasks):
    return ClassificationScores(
        text="x", tasks=tasks, fingerprint=compiled.fingerprint,
        specs={s.name: s for s in compiled.task_specs})


def _problem(compiled, tasks, cfg, **kw):
    return build_problem(compiled, _scores(compiled, tasks), cfg, **kw)


# ---- T-D3 : exact enforces a hard implies at utility cost --------------

def test_exact_enforces_hard_implies():
    schema = (ClassificationSchema()
              .single("intent", ["read", "delete"])
              .multi("effects", ["read_only", "delete"], min_labels=1)
              .constrain(C.implies(("intent", "delete"), ("effects", "delete"))))
    compiled = compile_schema(schema)
    # intent strongly prefers delete; effects strongly disprefers delete.
    tasks = {"intent": {"read": -1.0, "delete": 3.0},
             "effects": {"read_only": 2.0, "delete": -2.0}}
    cfg = _cfg(decoder="exact")
    sol = decode(_problem(compiled, tasks, cfg), cfg)
    assert sol.feasible
    assert sol.assignments["intent"].labels == frozenset({"delete"})
    assert "delete" in sol.assignments["effects"].labels


# ---- T-D4 : exact == independent when the constraint does not bind ------

def test_exact_matches_independent_when_unbound():
    schema = (ClassificationSchema()
              .single("intent", ["read", "delete"])
              .multi("effects", ["read_only", "delete"], min_labels=1)
              .constrain(C.implies(("intent", "delete"), ("effects", "delete"))))
    compiled = compile_schema(schema)
    # argmax already satisfies the implication (intent=read).
    tasks = {"intent": {"read": 3.0, "delete": -1.0},
             "effects": {"read_only": 2.0, "delete": -2.0}}
    cfg = _cfg(decoder="exact")
    problem = _problem(compiled, tasks, cfg)
    exact = decode(problem, cfg)
    indep = IndependentDecoder().decode(problem)
    assert exact.assignments["intent"].labels == indep.assignments["intent"].labels
    assert exact.assignments["effects"].labels == indep.assignments["effects"].labels


# ---- T-D9 : auto decoder selection -------------------------------------

def test_auto_selects_independent_without_cross_task():
    schema = ClassificationSchema().single("a", ["x", "y"]).single("b", ["p", "q"])
    compiled = compile_schema(schema)
    tasks = {"a": {"x": 1.0, "y": 0.0}, "b": {"p": 1.0, "q": 0.0}}
    problem = _problem(compiled, tasks, _cfg())
    assert select_decoder(problem, "auto") == "independent"


def test_auto_selects_exact_with_cross_task():
    schema = (ClassificationSchema()
              .single("a", ["x", "y"]).single("b", ["p", "q"])
              .constrain(C.implies(("a", "x"), ("b", "p"))))
    compiled = compile_schema(schema)
    tasks = {"a": {"x": 1.0, "y": 0.0}, "b": {"p": 1.0, "q": 0.0}}
    problem = _problem(compiled, tasks, _cfg())
    assert select_decoder(problem, "auto") == "exact"


# ---- T-D6 : infeasibility ladder ---------------------------------------

def _retention_infeasible():
    # any_selected(effects) with all effects far below a high candidate floor and
    # nothing rescued (set predicate names no label) => retained empty => infeasible.
    schema = (ClassificationSchema()
              .multi("effects", ["p", "q", "r"], min_labels=0, threshold=0.5)
              .constrain(C.any_selected("effects")))
    compiled = compile_schema(schema)
    tasks = {"effects": {"p": -4.0, "q": -5.0, "r": -6.0}}
    return compiled, tasks


def test_relax_recovers_retention_infeasibility():
    compiled, tasks = _retention_infeasible()
    cfg = _cfg(decoder="exact", candidate_threshold=0.9, on_infeasible="relax")
    scores = _scores(compiled, tasks)
    problem = build_problem(compiled, scores, cfg)

    def widen():
        return build_problem(compiled, scores, cfg,
                             full_retention_tasks=compiled.task_order)

    sol = decode(problem, cfg, widen=widen)
    assert sol.feasible
    assert len(sol.assignments["effects"].labels) >= 1  # any_selected satisfied


def _hard_infeasible():
    # intent forced to x, which forces b to be both p and q (exclusive) -> impossible.
    schema = (ClassificationSchema()
              .single("a", ["x", "other"])
              .single("b", ["p", "q"])
              .constrain(C.label("a", "x"),
                         C.implies(("a", "x"), ("b", "p")),
                         C.implies(("a", "x"), ("b", "q"))))
    compiled = compile_schema(schema)
    tasks = {"a": {"x": 1.0, "other": 0.0}, "b": {"p": 1.0, "q": 0.5}}
    return compiled, tasks


def test_min_violations_returns_infeasible_with_violations():
    compiled, tasks = _hard_infeasible()
    cfg = _cfg(decoder="exact", on_infeasible="min_violations")
    sol = decode(_problem(compiled, tasks, cfg), cfg)
    assert not sol.feasible
    assert len(sol.violations) >= 1


def test_raise_raises_infeasible_error():
    compiled, tasks = _hard_infeasible()
    cfg = _cfg(decoder="exact", on_infeasible="raise")
    with pytest.raises(InfeasibleError) as excinfo:
        decode(_problem(compiled, tasks, cfg), cfg)
    assert excinfo.value.violations  # payload populated


# ---- T-D6b : relax that fails falls through to min_violations ----------

def test_relax_failure_falls_through_to_min_violations():
    compiled, tasks = _hard_infeasible()
    cfg = _cfg(decoder="exact", on_infeasible="relax")
    problem = _problem(compiled, tasks, cfg)
    sol = decode(problem, cfg, widen=lambda: problem)  # widen cannot help
    assert not sol.feasible
    assert len(sol.violations) >= 1


# ---- T-D10 : min_violations is lexicographic ---------------------------

def test_min_violations_minimizes_count_then_maximizes_utility():
    # a forced to x violates at most one implies at a time; b can satisfy exactly
    # one of the two consequents. min_violations must pick the single-violation
    # assignment, and among those the higher-utility b label.
    compiled, tasks = _hard_infeasible()
    cfg = _cfg(on_infeasible="min_violations")
    sol = MinViolationsDecoder().decode(_problem(compiled, tasks, cfg))
    assert len(sol.violations) == 1                    # not 2
    assert sol.assignments["b"].labels == frozenset({"p"})  # higher utility


# ---- B1 : beam fallback on budget exhaustion ---------------------------

def test_beam_fallback_matches_exact_on_small_problem():
    schema = (ClassificationSchema()
              .single("intent", ["read", "delete"])
              .multi("effects", ["read_only", "delete"], min_labels=1)
              .constrain(C.implies(("intent", "delete"), ("effects", "delete"))))
    compiled = compile_schema(schema)
    tasks = {"intent": {"read": -1.0, "delete": 3.0},
             "effects": {"read_only": 2.0, "delete": -2.0}}
    exact_sol = decode(_problem(compiled, tasks, _cfg(decoder="exact")),
                       _cfg(decoder="exact"))
    # Force the exact path to blow its budget so beam takes over.
    cfg_tiny = _cfg(decoder="exact", exact_node_budget=1)
    beam_sol = decode(_problem(compiled, tasks, cfg_tiny), cfg_tiny)
    assert beam_sol.feasible
    assert beam_sol.assignments["intent"].labels == exact_sol.assignments["intent"].labels
    assert beam_sol.assignments["effects"].labels == exact_sol.assignments["effects"].labels


# ---- T-D7 : active masking drops constraints ---------------------------

def test_active_masking_drops_cross_task_constraint():
    schema = (ClassificationSchema()
              .single("intent", ["read", "delete"])
              .multi("effects", ["read_only", "delete"], min_labels=1)
              .constrain(C.implies(("intent", "delete"), ("effects", "delete"))))
    compiled = compile_schema(schema)
    tasks = {"intent": {"read": -1.0, "delete": 3.0},
             "effects": {"read_only": 2.0, "delete": -2.0}}
    problem = _problem(compiled, tasks, _cfg(), active=["intent"])
    assert problem.task_order == ("intent",)
    assert problem.constraints == ()  # implies referenced a masked-out task
