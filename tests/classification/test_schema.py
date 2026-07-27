"""Phase 1 gate: schema is fully specified and unbypassable by string input.

No torch in this module's call graph.
"""
from __future__ import annotations

import pytest

from gliner2.classification.errors import SchemaError
from gliner2.classification.schema import (
    ClassificationSchema,
    LabelSpec,
    TaskSpec,
)


class _StubConstraint:
    """Minimal constraint stand-in: just enough surface for the builder."""

    def __init__(self, *tasks):
        self._tasks = frozenset(tasks)

    def references(self):
        return self._tasks

    def to_dict(self):
        return {"type": "_Stub", "tasks": sorted(self._tasks)}

    def __eq__(self, other):
        return isinstance(other, _StubConstraint) and self._tasks == other._tasks


def _basic():
    return (ClassificationSchema()
            .single("sentiment", ["positive", "neutral", "negative"])
            .multi("topics", ["billing", "account", "technical"], threshold=0.4, max_labels=2)
            .ordinal("risk", ["safe", "low", "elevated", "dangerous"]))


# ---- T-R1 --------------------------------------------------------------

def test_specs_are_frozen():
    spec = TaskSpec("t", (LabelSpec("a"), LabelSpec("b")))
    with pytest.raises(Exception):
        spec.name = "x"
    with pytest.raises(Exception):
        LabelSpec("a").name = "z"


def test_declaration_order_preserved():
    schema = _basic()
    assert schema.task_order == ("sentiment", "topics", "risk")


def test_to_from_dict_round_trip_equal():
    schema = _basic()
    restored = ClassificationSchema.from_dict(schema.to_dict())
    assert restored == schema
    assert restored.task_order == schema.task_order


def test_round_trip_preserves_descriptions_and_examples():
    schema = (ClassificationSchema()
              .single("intent", {"read": "a read op", "write": "a write op"},
                      instruction="Judge the intent.",
                      examples=[("cat file", "read"), ("rm file", "write")]))
    restored = ClassificationSchema.from_dict(schema.to_dict())
    assert restored == schema


# ---- T-P3a -------------------------------------------------------------

@pytest.mark.parametrize("token", ["[P]", "[L]", "[E]", "[DESCRIPTION]", "(", ")"])
def test_reserved_token_in_label(token):
    with pytest.raises(SchemaError):
        ClassificationSchema().single("t", ["ok", f"bad{token}label"])


def test_reserved_token_in_task_name():
    with pytest.raises(SchemaError):
        ClassificationSchema().single("bad[L]task", ["a", "b"])


def test_reserved_token_in_instruction():
    with pytest.raises(SchemaError):
        ClassificationSchema().single("t", ["a", "b"], instruction="say [OUTPUT]")


def test_reserved_token_in_description():
    with pytest.raises(SchemaError):
        ClassificationSchema().single("t", {"a": "fine", "b": "has [R] marker"})


# ---- T-S3 --------------------------------------------------------------

def test_duplicate_label_within_task():
    with pytest.raises(SchemaError):
        ClassificationSchema().single("t", ["a", "a"])


def test_duplicate_task_name():
    schema = ClassificationSchema().single("t", ["a", "b"])
    with pytest.raises(SchemaError):
        schema.single("t", ["c", "d"])


# ---- T-S4 --------------------------------------------------------------

def test_min_labels_greater_than_max():
    with pytest.raises(SchemaError):
        ClassificationSchema().task("t", ["a", "b", "c"], min_labels=3, max_labels=2)


def test_max_labels_exceeds_label_count():
    with pytest.raises(SchemaError):
        ClassificationSchema().task("t", ["a", "b"], max_labels=3)


def test_ordered_with_one_label():
    with pytest.raises(SchemaError):
        ClassificationSchema().task("t", ["only"], ordered=True)


def test_nonpositive_temperature():
    with pytest.raises(SchemaError):
        ClassificationSchema().single("t", ["a", "b"], temperature=0.0)


@pytest.mark.parametrize("bad", [-0.1, 0.0, 1.0, 1.5])
def test_threshold_out_of_open_interval(bad):
    with pytest.raises(SchemaError):
        ClassificationSchema().single("t", ["a", "b"], threshold=bad)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_candidate_threshold_out_of_range(bad):
    with pytest.raises(SchemaError):
        ClassificationSchema().single("t", ["a", "b"], candidate_threshold=bad)


# ---- T-S5 --------------------------------------------------------------

def test_default_not_in_labels():
    with pytest.raises(SchemaError):
        ClassificationSchema().multi("t", ["a", "b"], default="c")


def test_default_forces_min_labels_at_least_one():
    schema = ClassificationSchema().multi("t", ["a", "b"], default="a", min_labels=0)
    assert schema.task_spec("t").min_labels == 1


# ---- T-S6 --------------------------------------------------------------

def test_example_label_not_in_set():
    with pytest.raises(SchemaError):
        ClassificationSchema().single("t", ["a", "b"], examples=[("text", "c")])


# ---- T-C5 --------------------------------------------------------------

def test_constrain_forward_reference_raises_naming_task():
    schema = ClassificationSchema().single("intent", ["a", "b"])
    with pytest.raises(SchemaError, match="effects"):
        schema.constrain(_StubConstraint("intent", "effects"))


def test_constrain_accepts_declared_tasks():
    schema = (ClassificationSchema()
              .single("intent", ["a", "b"])
              .multi("effects", ["x", "y"]))
    schema.constrain(_StubConstraint("intent", "effects"))
    assert len(schema.constraints) == 1


def test_constrain_rejects_non_constraint():
    schema = ClassificationSchema().single("intent", ["a", "b"])
    with pytest.raises(SchemaError):
        schema.constrain(("intent", "a"))


# ---- T-R2 --------------------------------------------------------------

def _with_constraints():
    return (ClassificationSchema()
            .single("a", ["x", "y"])
            .single("b", ["p", "q"])
            .single("c", ["m", "n"])
            .constrain(_StubConstraint("a"), _StubConstraint("a", "b")))


def test_subset_returns_independent_copy():
    schema = _with_constraints()
    sub = schema.subset("a", "b")
    assert sub is not schema
    sub.single("z", ["1", "2"])
    assert "z" not in schema.task_order  # mutating result does not touch original
    assert sub.task_order == ("a", "b", "z")


def test_subset_strict_raises_on_any_loss():
    schema = _with_constraints()
    with pytest.raises(SchemaError):
        schema.subset("a")  # drops the a/b constraint -> loss


def test_subset_prune_warns_on_pure_drop():
    schema = (ClassificationSchema()
              .single("a", ["x", "y"])
              .single("b", ["p", "q"])
              .constrain(_StubConstraint("b")))
    with pytest.warns(UserWarning):
        sub = schema.subset("a", keep_constraints="prune")
    assert sub.constraints == ()


@pytest.mark.parametrize("mode", ["strict", "prune", "error_on_mixed"])
def test_mixed_references_raise_under_all_modes(mode):
    schema = _with_constraints()
    with pytest.raises(SchemaError):
        # subset("a") keeps a, drops b/c; the a/b constraint is mixed.
        schema.subset("a", keep_constraints=mode)


def test_error_on_mixed_silently_drops_pure():
    schema = (ClassificationSchema()
              .single("a", ["x", "y"])
              .single("b", ["p", "q"])
              .constrain(_StubConstraint("b")))
    sub = schema.subset("a", keep_constraints="error_on_mixed")
    assert sub.constraints == ()


def test_drop_is_complement_of_subset():
    schema = _with_constraints()
    dropped = schema.drop("c")
    assert dropped.task_order == ("a", "b")


# ---- T-S7 --------------------------------------------------------------

def test_single_multi_ordinal_match_raw_task():
    single = ClassificationSchema().single("t", ["a", "b"]).task_spec("t")
    assert single == TaskSpec("t", (LabelSpec("a"), LabelSpec("b")),
                              min_labels=1, max_labels=1)

    multi = ClassificationSchema().multi("t", ["a", "b"], max_labels=2).task_spec("t")
    assert multi == TaskSpec("t", (LabelSpec("a"), LabelSpec("b")), max_labels=2)

    ordinal = ClassificationSchema().ordinal("t", ["a", "b", "c"]).task_spec("t")
    assert ordinal == TaskSpec("t", (LabelSpec("a"), LabelSpec("b"), LabelSpec("c")),
                               min_labels=1, max_labels=1, ordered=True)


def test_is_exclusive_property():
    assert ClassificationSchema().single("t", ["a", "b"]).task_spec("t").is_exclusive
    assert not ClassificationSchema().multi("t", ["a", "b"]).task_spec("t").is_exclusive
