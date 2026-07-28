"""Hard-constraint feasibility of joint-IE decoding.

Covers the three contract states: a feasible assignment is returned intact, an
infeasible one falls back to the empty solution with an explicit signal, and a
higher-scoring but infeasible greedy assignment never substitutes a feasible
beam result.
"""

import logging

import pytest

from gliner2.joint_ie.candidates import EdgeCandidate, JointProblem, NodeCandidate
from gliner2.joint_ie.optimizers import BeamOptimizer, GreedyOptimizer
from gliner2.joint_ie.optimizers.base import JointSolution
from gliner2.joint_ie.result import JointResult, ResultBuilder


def node(name, score):
    return NodeCandidate(name, 0, 1, score, candidate_id=name)


class AtMostOneRelation:
    """A whole-result constraint enforced only at validation time.

    Construction is unconstrained (``allow_edge`` defaults to True) so a greedy
    pass will happily pick every profitable edge; only the final assignment is
    rejected once it holds more than one relation.
    """

    def validate(self, relations, entities=()):
        return len(relations) <= 1


class RejectAnyRelation:
    def validate(self, relations, entities=()):
        return len(relations) == 0


class AlwaysInfeasible:
    """Rejects every assignment, including the empty one."""

    def validate(self, relations, entities=()):
        return False


@pytest.mark.parametrize("optimizer", [BeamOptimizer(beam_width=16), GreedyOptimizer()])
def test_feasible_solution_is_returned_intact(optimizer):
    a, b = node("a", 0.0), node("b", 0.0)
    edges = (EdgeCandidate("r", "a", "b", 5.0, slot=0, hypothesis="h"),)
    solution = optimizer.optimize(JointProblem((a, b), edges, (AtMostOneRelation(),)))
    assert solution.feasible is True
    assert len(solution.edges) == 1
    assert solution.score == pytest.approx(5.0)


def test_beam_prefers_feasible_over_higher_scoring_infeasible():
    a, b, c = node("a", 0.0), node("b", 0.0), node("c", 0.0)
    edges = (
        EdgeCandidate("r", "a", "b", 5.0, slot=0, hypothesis="h1"),
        EdgeCandidate("r", "a", "c", 4.0, slot=0, hypothesis="h2"),
    )
    problem = JointProblem((a, b, c), edges, (AtMostOneRelation(),))

    # Greedy alone would take both edges (score 9) but that violates the
    # whole-result constraint, so it must fall back rather than emit it.
    greedy = GreedyOptimizer().optimize(problem)
    assert greedy.feasible is False
    assert greedy.edges == ()

    # Beam keeps the best feasible single-edge assignment instead of the
    # infeasible pair, and never substitutes greedy's higher score.
    beam = BeamOptimizer(beam_width=16).optimize(problem)
    assert beam.feasible is True
    assert len(beam.edges) == 1
    assert beam.score == pytest.approx(5.0)


def test_greedy_signals_infeasible_final_assignment(caplog):
    # Nodes carry no free utility, so the only way greedy produces output is via
    # the edge; that single relation trips the whole-result constraint.
    a, b = node("a", 0.0), node("b", 0.0)
    edges = (EdgeCandidate("r", "a", "b", 5.0, slot=0, hypothesis="h"),)
    problem = JointProblem((a, b), edges, (RejectAnyRelation(),))
    with caplog.at_level(logging.WARNING):
        solution = GreedyOptimizer().optimize(problem)
    assert solution.feasible is False
    assert solution.edges == ()
    assert solution.nodes == ()
    assert solution.score == pytest.approx(0.0)
    assert any("constraint-violating" in rec.message for rec in caplog.records)


def test_beam_recovers_by_dropping_infeasible_relation():
    # A feasible subset exists (nodes without the relation), so the beam should
    # recover to it rather than signal infeasibility.
    a, b = node("a", 1.0), node("b", 1.0)
    edges = (EdgeCandidate("r", "a", "b", 5.0, slot=0, hypothesis="h"),)
    solution = BeamOptimizer(beam_width=8).optimize(
        JointProblem((a, b), edges, (RejectAnyRelation(),)))
    assert solution.feasible is True
    assert solution.edges == ()
    assert set(solution.node_ids) == {"a", "b"}


@pytest.mark.parametrize("optimizer", [BeamOptimizer(beam_width=8), GreedyOptimizer()])
def test_unsatisfiable_constraint_falls_back_to_empty_with_signal(optimizer, caplog):
    a, b = node("a", 1.0), node("b", 1.0)
    edges = (EdgeCandidate("r", "a", "b", 5.0, slot=0, hypothesis="h"),)
    problem = JointProblem((a, b), edges, (AlwaysInfeasible(),))
    with caplog.at_level(logging.WARNING):
        solution = optimizer.optimize(problem)
    assert solution.feasible is False
    assert solution.edges == ()
    assert solution.nodes == ()
    assert solution.score == pytest.approx(0.0)
    assert any("no constraint" in rec.message or "constraint-violating" in rec.message
               for rec in caplog.records)


def test_result_builder_propagates_infeasibility_signal():
    builder = ResultBuilder()
    problem = JointProblem((), ())
    feasible = builder.build(JointSolution((), (), 0.0), problem=problem, text="hi")
    infeasible = builder.build(JointSolution((), (), 0.0, feasible=False), problem=problem, text="hi")
    assert isinstance(feasible, JointResult) and feasible.feasible is True
    assert infeasible.feasible is False
