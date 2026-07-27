"""Build a tiny word-level fast tokenizer with no network access.

The tokenizer is a ``PreTrainedTokenizerFast`` wrapping a ``WordLevel`` model
with whitespace pre-tokenization. Every training/eval word used by the test
suite is placed in the vocabulary so that ``tokenizer.tokenize(word)`` returns
a single subword; unknown words fall back to ``[UNK]``. This keeps the
word -> subword mapping one-to-one, which makes golden-output comparisons
deterministic and easy to reason about.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


# A broad base vocabulary covering the words used across the test corpora.
_BASE_WORDS: List[str] = [
    ".", "!", "?", ",", "(", ")", "|", ":",
    "the", "a", "an", "of", "at", "in", "on", "for", "and", "to", "near",
    "reactor", "core", "temperature", "reached", "450", "degrees", "midnight",
    "coastal", "facility", "heavy", "rainfall", "disrupted", "satellite",
    "uplink", "during", "summit", "conference", "protocol", "seven", "was",
    "activated", "after", "seismic", "anomaly", "below", "northern", "ridge",
    "cargo", "manifest", "listed", "twelve", "crates", "synthetic", "polymer",
    "offshore", "platform",
    "apple", "acquired", "records", "released", "iphone", "15", "company",
    "product", "google", "works", "john", "smith", "elon", "musk", "founded",
    "spacex", "microsoft", "amazon", "nyc", "person", "location", "organization",
    "positive", "negative", "neutral", "sentiment", "flux_high", "flux_low",
    "flux_neutral", "vortex_mode", "zx9_alpha", "vibrance", "morph_class",
    "qx_report", "component", "reading", "site",
    "buyer", "target", "price", "acquisition", "$999", "$", "999",
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven",
    "cat", "dog", "sat", "mat", "ran", "fast", "slow", "big", "small",
    "red", "blue", "green", "yellow", "black", "white",
    "word", "text", "token", "span", "entity", "type", "label", "value",
    "start", "end", "boundary", "long", "short", "nested", "overlap",
    "quick", "brown", "fox", "jumps", "over", "lazy",
]

_SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]


def build_tiny_tokenizer(
    extra_words: Optional[Iterable[str]] = None,
) -> PreTrainedTokenizerFast:
    """Construct a deterministic offline word-level tokenizer.

    Args:
        extra_words: Additional (lowercased) words to guarantee are present
            in the vocabulary.

    Returns:
        A ``PreTrainedTokenizerFast`` ready to pass to ``SchemaTransformer``.
    """
    words: List[str] = []
    seen = set()

    def _add(tok: str) -> None:
        if tok not in seen:
            seen.add(tok)
            words.append(tok)

    for tok in _SPECIAL_TOKENS:
        _add(tok)
    for tok in _BASE_WORDS:
        _add(tok.lower())
    if extra_words:
        for tok in extra_words:
            _add(str(tok).lower())

    vocab = {tok: idx for idx, tok in enumerate(words)}

    model = WordLevel(vocab=vocab, unk_token="[UNK]")
    backend = Tokenizer(model)
    backend.pre_tokenizer = Whitespace()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    return tokenizer
