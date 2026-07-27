"""Shared pytest fixtures for the GLiNER2 test suite."""

from __future__ import annotations

import pytest
import torch

from tests.fixtures.tiny_encoder import build_tiny_encoder_config
from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer
from tests.fixtures.tiny_span_checkpoint import build_tiny_span_model


@pytest.fixture(autouse=True)
def _deterministic_seed():
    """Seed RNGs before every test for reproducibility."""
    torch.manual_seed(0)
    yield


@pytest.fixture
def tiny_tokenizer():
    return build_tiny_tokenizer()


@pytest.fixture
def tiny_encoder_config(tiny_tokenizer):
    return build_tiny_encoder_config(vocab_size=len(tiny_tokenizer))


@pytest.fixture
def tiny_span_model():
    return build_tiny_span_model()
