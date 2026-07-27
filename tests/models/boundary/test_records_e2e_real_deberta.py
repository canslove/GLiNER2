"""End-to-end record extraction on a real encoder (natural-anchor structure).

Trains a records-enabled boundary model with the ``ExtractorTrainer`` on
``InputExample`` structures carrying Instance Formation metadata, then runs the
public ``extract`` path and reloads the checkpoint. Marked ``slow`` (fine-tunes
a real DeBERTa-v3 encoder).
"""

from __future__ import annotations

import pytest
import torch

from gliner2 import AutoExtractor, ExtractorConfig
from gliner2.inference.engine import BoundaryExtractor
from gliner2.inference.schema import Schema
from gliner2.training.data import InputExample, Structure
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig

MODEL_NAME = "microsoft/deberta-v3-xsmall"

EXAMPLES = [
    InputExample(
        text="Alice bought apples and Bob bought oranges",
        structures=[
            Structure("purchase", mode="natural", anchor="buyer",
                      buyer="Alice", item="apples"),
            Structure("purchase", mode="natural", anchor="buyer",
                      buyer="Bob", item="oranges"),
        ],
    ),
    InputExample(
        text="Carol bought grapes and Dave bought melons",
        structures=[
            Structure("purchase", mode="natural", anchor="buyer",
                      buyer="Carol", item="grapes"),
            Structure("purchase", mode="natural", anchor="buyer",
                      buyer="Dave", item="melons"),
        ],
    ),
]


def _build_model() -> BoundaryExtractor:
    cfg = ExtractorConfig(
        model_name=MODEL_NAME,
        architecture="boundary",
        boundary_head=dict(
            boundary_dim=96, pair_dim=96, start_top_k=32, end_top_k=32,
            ends_per_start=16, starts_per_end=16, candidate_budget=96,
            training_candidate_budget=128, max_gold_per_query=16,
            end_block_size=32, dropout=0.0,
            enable_records=True, record_dim=96, record_instance_queries=16,
        ),
        token_pooling="first",
    )
    model = BoundaryExtractor(cfg)
    model.boundary_head.loss_weights["pair"] = 4.0
    return model


def _purchase_schema() -> Schema:
    s = Schema()
    (
        s.structure("purchase", mode="natural", anchor="buyer")
        .field("buyer", dtype="str", cardinality="required_one")
        .field("item", dtype="str")
    )
    return s


@pytest.mark.slow
def test_records_train_infer_reload_real_deberta(tmp_path):
    torch.manual_seed(0)
    model = _build_model()

    config = TrainingConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=300,
        batch_size=2,
        gradient_accumulation_steps=1,
        encoder_lr=2e-5,
        task_lr=5e-3,
        warmup_ratio=0.0,
        scheduler_type="constant",
        logging_steps=25,
        eval_strategy="no",
        fp16=False,
        bf16=False,
    )
    trainer = ExtractorTrainer(model, config)
    result = trainer.train(train_data=EXAMPLES)
    last_loss = result["train_metrics_history"][-1]["loss"]
    assert torch.isfinite(torch.tensor(last_loss))

    schema = _purchase_schema()
    out = model.extract(EXAMPLES[0].text, schema, include_spans=True)
    assert "purchase" in out
    assert isinstance(out["purchase"], list)
    buyers = {
        value["text"] if isinstance(value := inst.get("buyer"), dict) else value
        for inst in out["purchase"]
    }
    assert {"Alice", "Bob"} & buyers, f"expected anchors recovered, got {buyers}"

    # Save (trainer checkpoint) -> architecture-aware reload -> re-infer.
    save_dir = tmp_path / "run" / "final"
    reloaded = AutoExtractor.from_pretrained(str(save_dir))
    assert type(reloaded).__name__ == "BoundaryExtractor"
    assert reloaded.enable_records is True
    out2 = reloaded.extract(EXAMPLES[0].text, schema, include_spans=True)
    assert "purchase" in out2 and isinstance(out2["purchase"], list)
