"""Span golden regression gate.

Freezes the legacy span behavior: state-dict keys, training-forward losses,
and formatted extraction output. The span refactor must reproduce these to
1e-6 tolerance.
"""

import torch

from tests.fixtures.tiny_span_checkpoint import build_tiny_span_model
from tests.fixtures.span_golden import (
    assert_signature_matches,
    compute_span_signature,
    load_golden,
)


def test_old_span_forward_matches_golden_output():
    model = build_tiny_span_model()
    signature = compute_span_signature(model)
    golden = load_golden()
    assert_signature_matches(signature, golden)


def test_old_span_decode_matches_golden_output():
    model = build_tiny_span_model()
    signature = compute_span_signature(model)
    golden = load_golden()
    # Formatted extraction specifically.
    from tests.fixtures.span_golden import _formatted_close

    assert _formatted_close(signature["formatted"], golden["formatted"], 1e-6)


def test_old_state_dict_keys_load_without_missing_keys():
    model = build_tiny_span_model()
    keys = set(model.state_dict().keys())
    prefixes = {k.split(".")[0] for k in keys}
    # Legacy span module names must be preserved exactly.
    assert {"encoder", "span_rep", "classifier", "count_pred", "count_embed"} <= prefixes


def test_span_build_is_deterministic():
    a = compute_span_signature(build_tiny_span_model())
    b = compute_span_signature(build_tiny_span_model())
    assert a["state_keys"] == b["state_keys"]
    assert a["losses"] == b["losses"]
