"""Public-API end-to-end overfit test for the boundary architecture."""

from __future__ import annotations

import pytest
import torch

from gliner2 import BoundaryExtractor, ExtractorConfig
from gliner2.training import ExtractorTrainer, TrainingConfig
from gliner2.training.data import InputExample

MODEL_NAME = "microsoft/deberta-v3-xsmall"
TEXT = "Alice works at Acme."
ENTITY_TYPES = ["person", "company"]
EXPECTED = {"person": ["Alice"], "company": ["Acme"]}


@pytest.mark.slow
def test_boundary_overfit_on_real_deberta_v3(tmp_path):
    torch.manual_seed(0)

    try:
        model = BoundaryExtractor(
            ExtractorConfig(
                model_name=MODEL_NAME,
                architecture="boundary",
                token_pooling="first",
                boundary_head={
                    "boundary_dim": 64,
                    "pair_dim": 64,
                    "start_top_k": 16,
                    "end_top_k": 16,
                    "ends_per_start": 8,
                    "starts_per_end": 8,
                    "candidate_budget": 32,
                    "training_candidate_budget": 32,
                    "max_gold_per_query": 4,
                    "end_block_size": 16,
                    "dropout": 0.0,
                },
            )
        )
    except Exception as exc:
        pytest.skip(f"could not load {MODEL_NAME}: {exc}")

    examples = [InputExample(text=TEXT, entities=EXPECTED)]
    trainer = ExtractorTrainer(
        model=model,
        config=TrainingConfig(
            output_dir=str(tmp_path / "boundary-overfit"),
            max_steps=100,
            batch_size=1,
            encoder_lr=2e-5,
            task_lr=5e-3,
            weight_decay=0.0,
            scheduler_type="constant",
            warmup_ratio=0.0,
            fp16=False,
            bf16=False,
            eval_strategy="no",
            save_best=False,
            logging_steps=100,
            num_workers=0,
            pin_memory=False,
            seed=0,
            deterministic=True,
        ),
    )

    train_result = trainer.train(train_data=examples)
    assert train_result["total_steps"] == 100

    model.eval()
    result = model.extract_entities(
        TEXT,
        ENTITY_TYPES,
        threshold=0.5,
        include_spans=True,
    )

    recovered = {
        label: sorted(item["text"] for item in result["entities"].get(label, []))
        for label in ENTITY_TYPES
    }
    assert recovered == EXPECTED

    for items in result["entities"].values():
        for item in items:
            assert TEXT[item["start"] : item["end"]] == item["text"]
