"""Zero-subword words must not desynchronize word<->embedding alignment.

A text word that tokenizes to no subwords used to be dropped from
``text_word_first_positions`` while char-offset mappings still kept a row for it,
shifting every downstream word/embedding index. The processor now keeps a
placeholder row (and warns) so positions stay 1:1 with words.
"""

from __future__ import annotations

import logging

from gliner2.processor import SchemaTransformer
from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer


def test_zero_subword_word_keeps_placeholder_and_warns(monkeypatch, caplog):
    proc = SchemaTransformer(tokenizer=build_tiny_tokenizer())

    original = proc._tokenize_cached

    def fake_tokenize(token):
        if token == "ZEROWORD":
            return []
        return original(token)

    monkeypatch.setattr(proc, "_tokenize_cached", fake_tokenize)

    text_tokens = ["alpha", "ZEROWORD", "beta"]
    with caplog.at_level(logging.WARNING):
        out = proc._format_input_with_mapping([["task"]], text_tokens)

    positions = out["text_word_first_positions"]
    # One first-subword position per text word — the empty word is not dropped.
    assert len(positions) == len(text_tokens)
    # Positions remain monotonically non-decreasing.
    assert all(positions[i] <= positions[i + 1] for i in range(len(positions) - 1))
    # The silent drop is now an explicit warning.
    assert any("no subwords" in record.message for record in caplog.records)


def test_all_words_tokenize_normally_has_no_warning(caplog):
    proc = SchemaTransformer(tokenizer=build_tiny_tokenizer())
    with caplog.at_level(logging.WARNING):
        out = proc._format_input_with_mapping([["task"]], ["alpha", "beta", "gamma"])
    assert len(out["text_word_first_positions"]) == 3
    assert not any("no subwords" in record.message for record in caplog.records)
