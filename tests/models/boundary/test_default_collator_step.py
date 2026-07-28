"""Boundary training via the fallback target path (Finding 3 / Phase 1.3).

When a boundary model is trained with a collator left at its default
``architecture="span"``, no ``batch.targets`` is attached and the model must
fall back to ``_targets_from_structure``. Those targets are built on CPU and
must be moved to the model device before meeting the logits. On CPU the device
move is a no-op, but this exercises the fallback path end-to-end so it cannot
silently regress.
"""

from __future__ import annotations

import torch

from gliner2.training import ExtractorCollator
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model


def test_boundary_training_step_with_default_span_collator():
    model = build_tiny_boundary_model()
    model.train()

    # Default architecture is "span": the boundary metadata (and thus
    # batch.targets) is intentionally NOT built, forcing the fallback.
    collator = ExtractorCollator(model.processor, is_training=True)
    batch = collator([
        ("apple released iphone .", {"entities": {"company": ["apple"], "product": ["iphone"]}}),
    ])
    assert getattr(batch, "targets", None) is None  # fallback will build them

    out = model(batch)
    assert out.total_loss is not None
    assert torch.isfinite(out.total_loss)

    out.total_loss.backward()
    grads = [p.grad for p in model.boundary_head.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)
