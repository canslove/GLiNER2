"""LoRA full (non-adapter) checkpoint save smoke test (Finding 2 / Phase 1.2).

Saving a full checkpoint during LoRA training must not raise ``NameError``:
``merge_lora_weights`` has to be imported before it is called in the
full-checkpoint branch of ``_save_checkpoint``.
"""

from __future__ import annotations

from pathlib import Path

from peft.tuners.lora.layer import LoraLayer as _PeftLoraLayer

from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from tests.fixtures.tiny_span_checkpoint import build_tiny_span_model


def test_full_checkpoint_save_under_lora_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setattr("gliner2.training.trainer.torch.cuda.is_available", lambda: False)

    model = build_tiny_span_model()
    peft_model = model.apply_lora(r=4, alpha=8, dropout=0.0, targets=["classification_head"])

    # Build the trainer with use_lora disabled so __init__ does not re-apply
    # LoRA to an already-wrapped PeftModel, then flip into the full-save regime.
    config = TrainingConfig(
        output_dir=str(tmp_path / "out"), fp16=False, bf16=False, local_rank=-1,
        use_lora=False,
    )
    trainer = GLiNER2Trainer(model=peft_model, config=config)
    trainer.config.use_lora = True
    trainer.config.save_adapter_only = False
    trainer.lora_layers = {
        n: m for n, m in peft_model.named_modules() if isinstance(m, _PeftLoraLayer)
    }
    assert trainer.lora_layers, "expected LoRA layers on the peft model"

    trainer._save_checkpoint("full")

    checkpoint_dir = Path(config.output_dir) / "full"
    assert checkpoint_dir.exists()
    assert (checkpoint_dir / "lora_config.json").exists()
