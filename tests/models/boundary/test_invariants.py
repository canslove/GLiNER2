"""Static complexity invariants for the boundary package.

These guard the blueprint's non-negotiables: half-open coordinates with no
length cutoff and no width-embedding table (which would reintroduce a maximum
span width / dense ``[L, W, D]`` tensor).
"""

from __future__ import annotations

import pathlib

import gliner2.models.boundary as boundary_pkg

FORBIDDEN_SUBSTRINGS = [
    "max_width",
    "width_embedding",
    "WidthEmbedding",
    "width_embed",
    "max_span_width",
]

BOUNDARY_DIR = pathlib.Path(boundary_pkg.__file__).parent


def test_no_forbidden_width_identifiers_in_boundary_source():
    offenders = {}
    for path in BOUNDARY_DIR.glob("*.py"):
        text = path.read_text()
        hits = [tok for tok in FORBIDDEN_SUBSTRINGS if tok in text]
        if hits:
            offenders[path.name] = hits
    assert not offenders, f"forbidden width identifiers found: {offenders}"


def test_no_nn_embedding_over_span_width():
    # A width table would appear as an nn.Embedding sized by a width bound.
    for path in BOUNDARY_DIR.glob("*.py"):
        text = path.read_text()
        assert "nn.Embedding" not in text, f"unexpected embedding table in {path.name}"
