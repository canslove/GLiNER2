"""Architecture-aware LoRA target resolution for the boundary model."""

from __future__ import annotations

from gliner2.training.lora import _resolve_targets, _task_module_names
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model


def test_boundary_task_module_names():
    model = build_tiny_boundary_model()
    names = _task_module_names(model)
    assert "boundary_head" in names
    assert "classifier" in names


def test_extractive_head_alias_targets_boundary_head_not_encoder():
    model = build_tiny_boundary_model()
    resolved = _resolve_targets(model, ["extractive_head"])
    assert resolved, "expected boundary_head Linear layers"
    assert all(name.startswith("boundary_head") for name in resolved)
    assert not any(name.startswith("encoder.") for name in resolved)


def test_classification_head_alias_targets_classifier():
    model = build_tiny_boundary_model()
    resolved = _resolve_targets(model, ["classification_head"])
    assert resolved
    assert all(name.startswith("classifier") for name in resolved)


def test_all_task_heads_alias_covers_boundary_and_classifier():
    model = build_tiny_boundary_model()
    resolved = _resolve_targets(model, ["all_task_heads"])
    assert any(n.startswith("boundary_head") for n in resolved)
    assert any(n.startswith("classifier") for n in resolved)
    assert not any(n.startswith("encoder.") for n in resolved)


def test_encoder_alias_still_resolves_encoder_only():
    model = build_tiny_boundary_model()
    resolved = _resolve_targets(model, ["encoder"])
    assert resolved
    assert all(name.startswith("encoder.") for name in resolved)
