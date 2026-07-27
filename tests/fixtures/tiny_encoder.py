"""Build a tiny BERT encoder config with no network access.

Used to construct span and boundary models on CPU without downloading any
pretrained weights.
"""

from __future__ import annotations

from transformers import BertConfig


def build_tiny_encoder_config(
    vocab_size: int = 128,
    hidden_size: int = 32,
    num_hidden_layers: int = 1,
    num_attention_heads: int = 4,
    intermediate_size: int = 64,
    max_position_embeddings: int = 512,
) -> BertConfig:
    """Return a small ``BertConfig`` with dropout disabled for determinism."""
    return BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        type_vocab_size=2,
    )
