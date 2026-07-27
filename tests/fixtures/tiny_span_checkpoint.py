"""Build a tiny, deterministic span (legacy) checkpoint entirely offline.

The resulting model is a fully-initialized ``GLiNER2`` (span architecture)
backed by the tiny tokenizer and tiny encoder. Weights are seeded so golden
regression tests are reproducible.
"""

from __future__ import annotations

import torch

from .tiny_encoder import build_tiny_encoder_config
from .tiny_tokenizer import build_tiny_tokenizer


def build_tiny_span_model(seed: int = 13, max_width: int = 8):
    """Create a deterministic tiny span model (``GLiNER2``).

    Args:
        seed: RNG seed used for both torch global state and weight init.
        max_width: Legacy span max width.

    Returns:
        A ``GLiNER2`` instance on CPU in eval mode.
    """
    from gliner2.inference.engine import GLiNER2
    from gliner2 import ExtractorConfig

    tokenizer = build_tiny_tokenizer()
    encoder_config = build_tiny_encoder_config(vocab_size=len(tokenizer))

    config = ExtractorConfig(
        model_name="tiny-bert-fixture",
        max_width=max_width,
        counting_layer="count_lstm",
        token_pooling="first",
    )

    torch.manual_seed(seed)
    model = GLiNER2(config, encoder_config=encoder_config, tokenizer=tokenizer)

    # Re-seed and re-initialize task heads so values are reproducible and not
    # left at whatever the module default init produced.
    torch.manual_seed(seed)
    _reinit_(model)
    model.eval()
    return model


def _reinit_(model) -> None:
    """Deterministically re-initialize *all* parameters.

    Every parameter (encoder included) is overwritten from the freshly-seeded
    RNG in a stable ``named_parameters`` order. This makes the built model a
    pure function of the seed, independent of process history — e.g. one-time
    lazy initialization on the first model construction in a process no longer
    perturbs the weights. Without this, the first build in a process would
    differ from later builds and the golden signature would not reproduce.
    """
    with torch.no_grad():
        for _, param in model.named_parameters():
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param)
            else:
                param.uniform_(-0.05, 0.05)


def save_tiny_span_checkpoint(directory, seed: int = 13, max_width: int = 8):
    """Build a tiny span model and save it to ``directory``.

    Returns the built model (pre-save instance).
    """
    model = build_tiny_span_model(seed=seed, max_width=max_width)
    model.save_pretrained(str(directory))
    return model
