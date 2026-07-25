"""Phase 4 gate: given planted logits, the scorer reproduces them exactly,
under adversarial label reordering, in one encoder pass.
"""
from __future__ import annotations

import math

import pytest
import torch

from gliner2.classification.compiler import compile_schema
from gliner2.classification.errors import SchemaError
from gliner2.classification.scoring import ClassificationScorer
from gliner2.classification.schema import ClassificationSchema
from gliner2.joint_ie.candidates import center_logit


def _schema(**overrides):
    return ClassificationSchema().single(
        "sentiment", ["pos", "neu", "neg"], **overrides)


LOGITS = {"sentiment": {"pos": 2.0, "neu": 0.0, "neg": -1.0}}
TASKS = [("sentiment", ["pos", "neu", "neg"])]


# ---- T-S1 --------------------------------------------------------------

def test_one_encoder_pass_per_batch(planted):
    model = planted(TASKS, LOGITS)
    scorer = ClassificationScorer(model)
    compiled = compile_schema(_schema())
    scores = scorer.batch_score(["a"] * 8, compiled, batch_size=8)
    assert model.encoder.calls == 1
    assert len(scores) == 8
    assert scores[0].logit("sentiment", "pos") == 2.0
    assert scores[0].logit("sentiment", "neg") == -1.0


def test_per_document_schemas_supported(planted):
    model = planted(TASKS, LOGITS)
    scorer = ClassificationScorer(model)
    compiled = compile_schema(_schema())
    scores = scorer.batch_score(["a", "b", "c"], [compiled, compiled, compiled],
                                batch_size=2)
    assert len(scores) == 3
    assert model.encoder.calls == 2  # two chunks


# ---- T-S2 : alignment from encoded tokens ------------------------------

def test_alignment_survives_shuffled_prompt_order(planted):
    # The fake ENCODES labels in this shuffled order, planting per-name logits.
    shuffled = [("sentiment", ["neg", "pos", "neu"])]
    model = planted(shuffled, LOGITS)
    scorer = ClassificationScorer(model)
    # The compiled schema DECLARES a different order; a dict-derived path would
    # misalign. The token-derived path stays correct.
    compiled = compile_schema(_schema())
    scores = scorer.score("a", compiled)
    assert scores.logit("sentiment", "pos") == 2.0
    assert scores.logit("sentiment", "neu") == 0.0
    assert scores.logit("sentiment", "neg") == -1.0


# ---- T-S2b : count mismatch raises -------------------------------------

def test_logit_name_count_mismatch_raises(planted):
    model = planted(TASKS, LOGITS)

    # Replace the processor's embedding extraction to return one FEWER label row
    # than the [L] tokens in schema_tokens_list.
    def extract(encoded, input_ids, batch):
        n = len(batch)
        token = [torch.zeros((2, 4)) for _ in range(n)]
        schema = []
        for _ in range(n):
            rows = [torch.zeros(4), torch.full((4,), 2.0), torch.full((4,), 0.0)]  # 2 labels only
            schema.append([rows])
        return token, schema

    model.processor.extract_embeddings_from_batch = extract
    scorer = ClassificationScorer(model)
    compiled = compile_schema(_schema())
    with pytest.raises(SchemaError, match="mismatch"):
        scorer.score("a", compiled)


# ---- T-U2 : utility sign vs threshold ----------------------------------

@pytest.mark.parametrize("threshold", [0.1, 0.5, 0.9])
def test_utility_positive_iff_probability_meets_threshold(planted, threshold):
    logits = {"t": {"a": 1.3, "b": -0.4, "c": 0.0}}
    tasks = [("t", ["a", "b", "c"])]
    model = planted(tasks, logits)
    scorer = ClassificationScorer(model)
    compiled = compile_schema(
        ClassificationSchema().multi("t", ["a", "b", "c"], threshold=threshold))
    scores = scorer.score("x", compiled)
    for label in ("a", "b", "c"):
        util = scores.utility("t", label)
        prob = scores.probability("t", label)  # sigmoid (multi task)
        # center_logit is strict: utility > 0 iff sigmoid prob > threshold.
        assert (util > 0) == (prob > threshold)


def test_softmax_probabilities_sum_to_one(planted):
    model = planted(TASKS, LOGITS)
    scorer = ClassificationScorer(model)
    compiled = compile_schema(_schema())  # exclusive -> auto == softmax
    scores = scorer.score("x", compiled)
    total = sum(scores.probability("sentiment", l) for l in ("pos", "neu", "neg"))
    assert math.isclose(total, 1.0, rel_tol=1e-9)


# ---- T-U3 : temperature applied ----------------------------------------

def test_temperature_applied_to_utility(planted):
    model = planted(TASKS, LOGITS)
    scorer = ClassificationScorer(model)
    compiled = compile_schema(_schema(temperature=2.0, threshold=0.5))
    scores = scorer.score("x", compiled)
    # utility = center_logit(logit / temperature, threshold)
    assert math.isclose(scores.utility("sentiment", "pos"),
                        center_logit(2.0 / 2.0, 0.5), rel_tol=1e-9)


def test_utility_is_activation_independent(planted):
    model = planted(TASKS, LOGITS)
    scorer = ClassificationScorer(model)
    sig = scorer.score("x", compile_schema(_schema(activation="sigmoid")))
    smx = scorer.score("x", compile_schema(_schema(activation="softmax")))
    for label in ("pos", "neu", "neg"):
        assert math.isclose(sig.utility("sentiment", label),
                            smx.utility("sentiment", label), rel_tol=1e-12)


# ---- T-S8 : lifecycle --------------------------------------------------

def test_to_and_eval_propagate(planted):
    model = planted(TASKS, LOGITS)
    scorer = ClassificationScorer(model)
    scorer.eval()
    assert model.processor.is_training is False
    returned = scorer.to(device="cpu")
    assert returned is scorer


# ---- T-S9 : deterministic top under ties -------------------------------

def test_top_is_deterministic_under_ties(planted):
    logits = {"t": {"a": 1.0, "b": 1.0, "c": 0.0}}
    tasks = [("t", ["a", "b", "c"])]
    model = planted(tasks, logits)
    scorer = ClassificationScorer(model)
    compiled = compile_schema(ClassificationSchema().multi("t", ["a", "b", "c"]))
    scores = scorer.score("x", compiled)
    top = scores.top("t", k=2)
    assert [label for label, _ in top] == ["a", "b"]  # tie broken by name


# ---- T-S10 : fingerprint ------------------------------------------------

def test_scores_fingerprint_matches_compiled(planted):
    model = planted(TASKS, LOGITS)
    scorer = ClassificationScorer(model)
    compiled = compile_schema(_schema())
    scores = scorer.score("x", compiled)
    assert scores.fingerprint == compiled.fingerprint


def test_scores_tasks_are_immutable(planted):
    model = planted(TASKS, LOGITS)
    scorer = ClassificationScorer(model)
    scores = scorer.score("x", compile_schema(_schema()))
    with pytest.raises(TypeError):
        scores.tasks["sentiment"]["pos"] = 9.0
