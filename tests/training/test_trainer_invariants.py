"""Trainer generalization: alias, strict config, disjoint optimizer groups."""

from __future__ import annotations

import pytest
import torch

from gliner2.training.trainer import (
    ExtractorTrainer,
    GLiNER2Trainer,
    TrainingConfig,
)
from tests.fixtures.tiny_span_checkpoint import build_tiny_span_model


def test_gliner2trainer_is_extractor_trainer_alias():
    assert GLiNER2Trainer is ExtractorTrainer


def test_strict_training_defaults():
    config = TrainingConfig()
    assert config.strict_training is True
    assert config.allow_invalid_samples is False
    assert config.log_proposal_metrics is True


def test_optimizer_groups_are_disjoint_and_complete(tmp_path):
    model = build_tiny_span_model()
    config = TrainingConfig(output_dir=str(tmp_path / "out"), num_workers=0, fp16=False)
    trainer = ExtractorTrainer(model=model, config=config)

    optimizer = trainer._create_optimizer()  # asserts disjoint/complete internally
    grouped = sum(len(g["params"]) for g in optimizer.param_groups)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert grouped == trainable

    # Encoder params land in the encoder LR group, task params in the task group.
    enc_group, task_group = optimizer.param_groups
    assert enc_group["lr"] == config.encoder_lr
    assert task_group["lr"] == config.task_lr
