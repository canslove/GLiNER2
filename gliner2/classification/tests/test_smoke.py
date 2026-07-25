"""Phase 0 gate: the package imports and its public surface is present."""
from __future__ import annotations


def test_package_imports_and_exposes_public_surface():
    import gliner2.classification as classification

    assert hasattr(classification, "__all__")
    for name in classification.__all__:
        assert hasattr(classification, name), name


def test_errors_are_importable_with_correct_bases():
    from gliner2.classification.errors import InfeasibleError, SchemaError

    assert issubclass(SchemaError, ValueError)
    assert issubclass(InfeasibleError, RuntimeError)
    err = InfeasibleError("boom", violations=("a", "b"))
    assert err.violations == ("a", "b")
