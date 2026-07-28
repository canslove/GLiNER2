"""Length-adaptive proposal budget tests."""

from gliner2.models.boundary.proposal import resolve_boundary_budget


def test_adaptive_budget_is_shape_derived_and_bucketed():
    kwargs = dict(base_k=16, alpha=0.08, k_max=128, bucket=8)
    assert resolve_boundary_budget(129, **kwargs) == 16
    assert resolve_boundary_budget(513, **kwargs) == 48
    assert resolve_boundary_budget(2049, **kwargs) == 128


def test_zero_alpha_preserves_legacy_budget():
    assert resolve_boundary_budget(
        4097, base_k=16, alpha=0.0, k_max=128, bucket=8
    ) == 16
