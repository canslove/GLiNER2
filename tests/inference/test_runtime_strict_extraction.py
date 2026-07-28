"""Runtime extraction failure handling (strict vs resilient).

Guards Finding 11: a per-sample extraction failure must be observable. In the
default strict mode the exception propagates; with ``strict_extraction=False``
it is logged and the offending sample yields ``{}`` (distinguishable only by
opt-in).
"""

from __future__ import annotations

import types

import pytest
import torch

from gliner2.inference.runtime import ExtractorRuntimeMixin


class _FakeBatch:
    def __init__(self, n):
        self.input_ids = torch.zeros(n, 1, dtype=torch.long)
        self.attention_mask = torch.ones(n, 1, dtype=torch.long)
        self.task_types = [["entities"] for _ in range(n)]
        self.schema_tokens_list = [[["tok"]] for _ in range(n)]
        self.text_tokens = [["w"] for _ in range(n)]
        self.original_texts = ["text" for _ in range(n)]
        self.start_mappings = [[0] for _ in range(n)]
        self.end_mappings = [[1] for _ in range(n)]
        self.original_schemas = [{} for _ in range(n)]
        self._n = n

    def __len__(self):
        return self._n


class _FakeRuntime(ExtractorRuntimeMixin):
    """Minimal runtime that always fails inside ``_extract_sample``."""

    def __init__(self):
        self.encoder = lambda input_ids, attention_mask: types.SimpleNamespace(
            last_hidden_state=torch.zeros(len(input_ids), 1, 4)
        )
        self.processor = types.SimpleNamespace(
            extract_embeddings_from_batch=lambda hidden, ids, batch: (
                [torch.zeros(1, 4) for _ in range(len(batch))],
                [[[torch.zeros(4)]] for _ in range(len(batch))],
            )
        )

    def compute_span_rep_batched(self, embs):
        return [None for _ in embs]

    def _extract_sample(self, **kwargs):
        raise RuntimeError("boom")


def _run(runtime, n=2):
    batch = _FakeBatch(n)
    return runtime._extract_from_batch(
        batch,
        threshold=0.5,
        metadata_list=[{} for _ in range(n)],
        include_confidence=True,
        include_spans=True,
    )


def test_strict_extraction_default_propagates():
    runtime = _FakeRuntime()
    assert runtime.strict_extraction is True
    with pytest.raises(RuntimeError, match="boom"):
        _run(runtime)


def test_resilient_extraction_logs_and_returns_empty(caplog):
    runtime = _FakeRuntime()
    runtime.strict_extraction = False
    with caplog.at_level("ERROR"):
        results = _run(runtime, n=2)
    assert results == [{}, {}]
    assert "extraction failed for sample" in caplog.text
