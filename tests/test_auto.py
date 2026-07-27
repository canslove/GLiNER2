"""Tests for ``AutoExtractor`` registry dispatch and safe loading (PR 2)."""

from __future__ import annotations

import pytest

from gliner2.auto import (
    AutoExtractor,
    UnknownArchitectureError,
    ArchitectureMismatchError,
    ArchitectureRegistrationError,
)
from gliner2.configuration import ExtractorConfig
from tests.fixtures.tiny_span_checkpoint import save_tiny_span_checkpoint


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_span_is_registered_after_import():
    import gliner2.inference.engine  # noqa: F401
    assert "span" in AutoExtractor._registry


def test_register_conflict_raises_without_exist_ok():
    class Dummy:
        pass

    AutoExtractor.register("span", Dummy, exist_ok=True)  # temporarily override
    try:
        with pytest.raises(ArchitectureRegistrationError):
            AutoExtractor.register("span", object)  # different class, no exist_ok
    finally:
        # Restore the real span class.
        from gliner2.inference.engine import SpanExtractor
        AutoExtractor.register("span", SpanExtractor, exist_ok=True)


def test_register_rejects_unknown_architecture_name():
    with pytest.raises(ValueError):
        AutoExtractor.register("nonsense", object)


# ---------------------------------------------------------------------------
# Resolution / unknown architecture
# ---------------------------------------------------------------------------

def test_resolve_span_returns_span_class():
    cls = AutoExtractor._resolve_class("span")
    from gliner2.inference.engine import SpanExtractor
    assert cls is SpanExtractor


def test_resolve_unregistered_architecture_raises(monkeypatch):
    """A known-but-unregistered architecture must raise UnknownArchitectureError."""
    import gliner2.inference.engine  # noqa: F401  (ensure builtins registered)
    registry_without_boundary = {
        k: v for k, v in AutoExtractor._registry.items() if k != "boundary"
    }
    monkeypatch.setattr(AutoExtractor, "_registry", registry_without_boundary)
    # Prevent _ensure_registered from re-adding boundary.
    monkeypatch.setattr("gliner2.auto._ensure_registered", lambda: None)
    with pytest.raises(UnknownArchitectureError):
        AutoExtractor._resolve_class("boundary")


# ---------------------------------------------------------------------------
# Dispatch via from_config
# ---------------------------------------------------------------------------

def test_from_config_dispatches_span(tiny_tokenizer, tiny_encoder_config):
    cfg = ExtractorConfig(model_name="tiny-bert-fixture", max_width=8)
    model = AutoExtractor.from_config(
        cfg, encoder_config=tiny_encoder_config, tokenizer=tiny_tokenizer
    )
    from gliner2.inference.engine import SpanExtractor
    assert isinstance(model, SpanExtractor)
    assert model.architecture == "span"


# ---------------------------------------------------------------------------
# Dispatch via from_pretrained (round-trip on a saved tiny checkpoint)
# ---------------------------------------------------------------------------

def test_from_pretrained_dispatches_span(tmp_path):
    save_tiny_span_checkpoint(tmp_path)
    model = AutoExtractor.from_pretrained(str(tmp_path))
    from gliner2.inference.engine import SpanExtractor
    assert isinstance(model, SpanExtractor)
    assert model.architecture == "span"


def test_from_pretrained_architecture_mismatch_raises(tmp_path):
    save_tiny_span_checkpoint(tmp_path)
    with pytest.raises(ArchitectureMismatchError):
        AutoExtractor.from_pretrained(str(tmp_path), architecture="boundary")
