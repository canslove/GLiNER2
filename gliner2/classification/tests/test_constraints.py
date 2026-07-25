"""Phase 2 gate: the constraint AST is provable against a literal Assignment.

No import of scoring, decoding or torch anywhere in this module.
"""
from __future__ import annotations

import itertools
import random

import pytest

from gliner2.classification import constraints as C
from gliner2.classification.constraints import (
    And,
    AnyOtherSelected,
    AnySelected,
    AtLevel,
    Cardinality,
    Constraint,
    DictAssignment,
    Excludes,
    ExactlyOneOf,
    Iff,
    Implies,
    IsDefault,
    LabelRef,
    MaxLevel,
    MinLevel,
    Not,
    Or,
    constraint_from_dict,
    k_and,
    k_iff,
    k_implies,
    k_not,
    k_or,
)
from gliner2.classification.errors import SchemaError
from gliner2.classification.schema import ClassificationSchema

THREE = (True, False, None)


def _schema():
    return (ClassificationSchema()
            .single("intent", ["read", "write", "delete"])
            .multi("effects", ["read_only", "create", "modify", "delete"], min_labels=1)
            .ordinal("risk", ["safe", "low", "elevated", "dangerous"])
            .multi("tox", ["violence", "hate", "benign"], threshold=0.4, default="benign"))


def _decide(schema, **selected):
    """A complete assignment: every named task decided to its selected set."""
    return DictAssignment(schema, {t: set(v) for t, v in selected.items()},
                          decided=list(selected))


def _open(schema, domains=None, selected=None, decided=()):
    return DictAssignment(schema, selected or {}, decided=decided, domains=domains or {})


# ---- T-C2 : Kleene truth tables ---------------------------------------

@pytest.mark.parametrize("a,b", itertools.product(THREE, THREE))
def test_kleene_binary_tables(a, b):
    # not
    assert k_not(None) is None
    assert k_not(True) is False and k_not(False) is True
    # and
    if a is False or b is False:
        assert k_and([a, b]) is False
    elif a is None or b is None:
        assert k_and([a, b]) is None
    else:
        assert k_and([a, b]) is True
    # or
    if a is True or b is True:
        assert k_or([a, b]) is True
    elif a is None or b is None:
        assert k_or([a, b]) is None
    else:
        assert k_or([a, b]) is False
    # implies == or(not a, b)
    assert k_implies(a, b) == k_or([k_not(a), b])
    # iff
    if a is None or b is None:
        assert k_iff(a, b) is None
    else:
        assert k_iff(a, b) == (a == b)


def test_kleene_empty_identities():
    assert k_and([]) is True
    assert k_or([]) is False


# ---- T-C3 : satisfied == still_satisfiable on complete assignments -----

def test_derived_predicates_agree_on_complete_assignments():
    schema = _schema()
    rng = random.Random(0)
    nodes = [
        Implies(LabelRef("intent", "delete"), LabelRef("effects", "delete")),
        Iff(LabelRef("intent", "read"), LabelRef("effects", "read_only")),
        Excludes(LabelRef("effects", "read_only"), LabelRef("effects", "create")),
        MinLevel("risk", "elevated"),
        Cardinality("effects", 1, 2),
        AnyOtherSelected("tox"),
    ]
    for _ in range(200):
        a = _decide(
            schema,
            intent={rng.choice(["read", "write", "delete"])},
            effects=set(x for x in ["read_only", "create", "modify", "delete"]
                        if rng.random() < 0.5) or {"create"},
            risk={rng.choice(["safe", "low", "elevated", "dangerous"])},
            tox=set(x for x in ["violence", "hate", "benign"] if rng.random() < 0.5) or {"benign"},
        )
        for node in nodes:
            value = node.evaluate(a)
            assert value in (True, False)  # complete assignment => determined
            assert node.satisfied(a) == (value is True)
            assert node.still_satisfiable(a) == (value is not False)


# ---- T-C4 : ordinal decides from domains before the task is decided ----

def test_ordinal_pruning_all_three_outcomes():
    schema = _schema()
    labels = ("safe", "low", "elevated", "dangerous")
    # min_level(elevated) => floor index 2.
    node = MinLevel("risk", "elevated")
    # domain wholly >= elevated -> True even though undecided.
    assert node.evaluate(_open(schema, domains={"risk": {"elevated", "dangerous"}})) is True
    # domain wholly < elevated -> False.
    assert node.evaluate(_open(schema, domains={"risk": {"safe", "low"}})) is False
    # domain straddles -> None.
    assert node.evaluate(_open(schema, domains={"risk": set(labels)})) is None


def test_max_level_and_at_level_pruning():
    schema = _schema()
    assert MaxLevel("risk", "low").evaluate(_open(schema, domains={"risk": {"safe", "low"}})) is True
    assert MaxLevel("risk", "low").evaluate(_open(schema, domains={"risk": {"elevated"}})) is False
    assert AtLevel("risk", "low").evaluate(_open(schema, domains={"risk": {"low"}})) is True
    assert AtLevel("risk", "low").evaluate(_open(schema, domains={"risk": {"safe"}})) is False
    assert AtLevel("risk", "low").evaluate(_open(schema, domains={"risk": {"low", "safe"}})) is None


# ---- T-C4b : empty domain -> False -------------------------------------

def test_empty_domain_is_false_not_none():
    schema = _schema()
    empty = _open(schema, domains={"risk": set(), "effects": set(), "tox": set()})
    assert MinLevel("risk", "low").evaluate(empty) is False
    assert AnySelected("effects").evaluate(empty) is False
    assert AnyOtherSelected("tox").evaluate(empty) is False


# ---- T-C9 : the (level or -1) regression -------------------------------

def test_level_zero_is_not_conflated_with_undecided():
    schema = _schema()
    # decided at level 0 ("safe"): min_level(safe) [floor 0] must be True.
    decided0 = _decide(schema, risk={"safe"})
    assert MinLevel("risk", "safe").evaluate(decided0) is True
    # undecided task must not accidentally satisfy min_level for a positive floor.
    undecided = _open(schema, domains={"risk": {"safe", "low", "elevated", "dangerous"}})
    assert MinLevel("risk", "elevated").evaluate(undecided) is None
    # and None must never be coerced to a bool.
    assert MinLevel("risk", "elevated").evaluate(undecided) is not False
    assert MinLevel("risk", "elevated").evaluate(undecided) is not True


# ---- AnySelected / AnyOtherSelected semantics --------------------------

def test_any_selected_and_any_other_selected():
    schema = _schema()
    # tox default is benign.
    only_default = _decide(schema, tox={"benign"})
    assert AnySelected("tox").evaluate(only_default) is True
    assert AnyOtherSelected("tox").evaluate(only_default) is False
    with_other = _decide(schema, tox={"benign", "hate"})
    assert AnyOtherSelected("tox").evaluate(with_other) is True
    assert IsDefault("tox").evaluate(with_other) is True
    assert IsDefault("tox").evaluate(_decide(schema, tox={"hate"})) is False


# ---- T-C1 / T-C1b : serialization --------------------------------------

def _nested_tree():
    return And((
        Implies(LabelRef("intent", "delete"), LabelRef("effects", "delete")),
        Iff(AnyOtherSelected("tox"), Not(IsDefault("tox"))),
        Or((MinLevel("risk", "low"), MaxLevel("risk", "elevated"))),
        ExactlyOneOf((LabelRef("intent", "read"), LabelRef("intent", "write"))),
        Cardinality("effects", 1, 2),
        Excludes(LabelRef("effects", "read_only"), LabelRef("effects", "create")),
        AtLevel("risk", "safe"),
        AnySelected("effects"),
    ))


@pytest.mark.parametrize("node", [
    LabelRef("intent", "read"),
    Not(LabelRef("intent", "read")),
    AnySelected("tox"),
    AnyOtherSelected("tox"),
    IsDefault("tox"),
    Cardinality("effects", 0, 3),
    Cardinality("effects", 2, None),
    MinLevel("risk", "low"),
    MaxLevel("risk", "elevated"),
    AtLevel("risk", "safe"),
    _nested_tree(),
])
def test_round_trip_equal(node):
    assert constraint_from_dict(node.to_dict()) == node


def test_unknown_type_raises():
    with pytest.raises(SchemaError):
        constraint_from_dict({"type": "Nonsense", "task": "x"})


def test_tuple_field_survives_as_tuple():
    tree = And((LabelRef("intent", "read"), LabelRef("intent", "write")))
    restored = constraint_from_dict(tree.to_dict())
    assert isinstance(restored.children, tuple)


# ---- T-C7 : references / label_references union ------------------------

def test_references_are_union_of_children():
    tree = _nested_tree()
    assert tree.references() == {"intent", "effects", "tox", "risk"}
    assert ("intent", "delete") in tree.label_references()
    assert ("effects", "delete") in tree.label_references()
    # cardinality/ordinal/set nodes contribute no label_references.
    assert all(t in ("intent", "effects") for t, _ in tree.label_references())


def test_coupling_references_capture_set_and_count_nodes():
    tree = _nested_tree()
    coupled = tree.coupling_references()
    assert "tox" in coupled       # AnyOtherSelected / IsDefault
    assert "effects" in coupled   # Cardinality + AnySelected


# ---- T-C8 : check_schema -----------------------------------------------

def test_check_schema_errors():
    schema = _schema()
    with pytest.raises(SchemaError):
        LabelRef("nope", "x").check_schema(schema)
    with pytest.raises(SchemaError):
        LabelRef("intent", "nonexistent").check_schema(schema)
    with pytest.raises(SchemaError):
        MinLevel("intent", "read").check_schema(schema)   # unordered
    with pytest.raises(SchemaError):
        MinLevel("risk", "nonexistent").check_schema(schema)
    with pytest.raises(SchemaError):
        IsDefault("intent").check_schema(schema)           # no default
    with pytest.raises(SchemaError):
        AnyOtherSelected("intent").check_schema(schema)    # no default
    with pytest.raises(SchemaError):
        Cardinality("effects", 0, 99).check_schema(schema)  # out of range


def test_check_schema_passes_on_valid_nodes():
    schema = _schema()
    _nested_tree().check_schema(schema)  # must not raise


# ---- DSL ---------------------------------------------------------------

def test_dsl_coerces_tuples():
    node = C.implies(("intent", "read"), ("effects", "read_only"))
    assert node == Implies(LabelRef("intent", "read"), LabelRef("effects", "read_only"))
    assert isinstance(C.all_of(("a", "b"), ("c", "d")), And)
    assert isinstance(C.any_of(("a", "b")), Or)
    assert C.at_least("effects", 2) == Cardinality("effects", 2, None)
    assert C.at_most("effects", 2) == Cardinality("effects", 0, 2)
    assert C.exactly("effects", 1) == Cardinality("effects", 1, 1)
    assert C.between_level("risk", "low", "elevated") == And(
        (MinLevel("risk", "low"), MaxLevel("risk", "elevated")))


def test_dsl_rejects_bad_expressions():
    with pytest.raises(SchemaError):
        C.implies(("only",), ("effects", "x"))
    with pytest.raises(SchemaError):
        C.at_least("effects", -1)


def test_dsl_full_surface():
    assert C.not_(("intent", "read")) == Not(LabelRef("intent", "read"))
    assert C.excludes(("a", "x"), ("b", "y")) == Excludes(LabelRef("a", "x"), LabelRef("b", "y"))
    assert C.exactly_one_of(("a", "x"), ("b", "y")) == ExactlyOneOf(
        (LabelRef("a", "x"), LabelRef("b", "y")))
    assert C.iff(("a", "x"), ("b", "y")) == Iff(LabelRef("a", "x"), LabelRef("b", "y"))
    assert C.at_level("risk", "low") == AtLevel("risk", "low")
    assert C.min_level("risk", "low") == MinLevel("risk", "low")
    assert C.max_level("risk", "low") == MaxLevel("risk", "low")
    assert C.any_selected("effects") == AnySelected("effects")
    assert C.any_other_selected("tox") == AnyOtherSelected("tox")
    assert C.is_default("tox") == IsDefault("tox")
    assert C.label("intent", "read") == LabelRef("intent", "read")
    # a Constraint passes through _expr unchanged
    inner = LabelRef("intent", "read")
    assert C.not_(inner).child is inner


def test_dsl_task_name_and_int_validation():
    with pytest.raises(SchemaError):
        C.any_selected(123)
    with pytest.raises(SchemaError):
        C.at_most("effects", True)  # bool is not an int here


def test_exactly_one_of_kleene_outcomes():
    schema = _schema()
    a = _decide(schema, intent={"read"})
    # exactly one of two mutually exclusive label refs on a single task
    node = ExactlyOneOf((LabelRef("intent", "read"), LabelRef("intent", "write")))
    assert node.evaluate(a) is True
    both_open = _open(schema, domains={"intent": {"read", "write", "delete"}})
    assert node.evaluate(both_open) is None
    node2 = ExactlyOneOf((LabelRef("intent", "write"), LabelRef("intent", "delete")))
    assert node2.evaluate(_decide(schema, intent={"read"})) is False  # zero true, all decided


def test_cardinality_over_and_under():
    schema = _schema()
    over = _decide(schema, effects={"read_only", "create", "modify"})
    assert Cardinality("effects", 0, 2).evaluate(over) is False   # too many
    under = _decide(schema, effects={"create"})
    assert Cardinality("effects", 2, None).evaluate(under) is False  # too few


def test_is_default_returns_false_without_default_label():
    schema = _schema()
    # intent has no default; IsDefault.evaluate short-circuits to False.
    assert IsDefault("intent").evaluate(_decide(schema, intent={"read"})) is False
