"""Synthetic training/eval corpora used by overfit and regression tests."""

from __future__ import annotations

from typing import List

from gliner2.training.data import InputExample


def span_golden_texts() -> List[str]:
    """A small, fixed set of texts for span golden regression."""
    return [
        "Apple acquired Apple Records.",
        "John Smith works at Google in NYC.",
        "The quick brown fox jumps over the lazy dog.",
    ]


def span_golden_schema() -> dict:
    """A fixed entity schema for span golden regression."""
    return {"entities": {"company": "", "person": "", "location": ""}}


def boundary_entity_examples() -> List[InputExample]:
    """Six examples exercising short, long, nested, shared-start, final-token,
    and negative cases for boundary entity extraction."""
    return [
        # 1. short entity
        InputExample(text="Apple released iPhone 15.", entities={"company": ["Apple"]}),
        # 2. entity longer than eight tokens
        InputExample(
            text="the quick brown fox jumps over the lazy dog near the ridge.",
            entities={"phrase": ["the quick brown fox jumps over the lazy dog"]},
        ),
        # 3. nested entities
        InputExample(
            text="John Smith works at Google.",
            entities={"person": ["John Smith"], "name": ["John"]},
        ),
        # 4. two labels sharing one start
        InputExample(
            text="Apple Records is a label.",
            entities={"company": ["Apple"], "org": ["Apple Records"]},
        ),
        # 5. entity ending at final text token
        InputExample(
            text="The founder is Elon Musk",
            entities={"person": ["Elon Musk"]},
        ),
        # 6. negative example with no target entity
        InputExample(
            text="The cat sat on the mat.",
            entities={"company": []},
        ),
    ]
