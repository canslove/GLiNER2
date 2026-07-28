"""Gradient accumulation state correctness (Finding 5 / Phase 1.5).

The optimizer-step count must equal ceil(successful_micro / accum): skipped and
OOM'd micro-batches must never drop or misalign a step, a trailing partial
window must be flushed, and no stale gradients may survive into the next epoch.
"""

from __future__ import annotations

import math

import torch

from gliner2.training.data import InputExample
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig
from tests.fixtures.tiny_span_checkpoint import build_tiny_span_model


def _examples(n):
    return [
        InputExample(
            text="apple released iphone .",
            entities={"company": ["apple"], "product": ["iphone"]},
        )
        for _ in range(n)
    ]


def _config(tmp_path, *, accum):
    return TrainingConfig(
        output_dir=str(tmp_path / "out"),
        batch_size=1,
        gradient_accumulation_steps=accum,
        num_epochs=1,
        eval_strategy="no",
        fp16=False,
        bf16=False,
        num_workers=0,
        logging_steps=10_000,
        warmup_ratio=0.0,
        scheduler_type="constant",
        local_rank=-1,
    )


def _instrument(trainer, *, skip_steps=frozenset(), oom_steps=frozenset()):
    """Fake the per-micro backward and count real optimizer steps."""
    calls = {"opt": 0}
    original_step = trainer._optimizer_step

    def counting_step():
        calls["opt"] += 1
        return original_step()

    def fake_backward(batch, step, use_amp, amp_dtype, *, is_last_micro=True):
        if step in oom_steps:
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        trainer._last_train_outputs = {"total_loss": torch.tensor(0.0)}
        # No-supervision batches now contribute a differentiable zero and still
        # advance the deterministic accumulation window.
        return torch.tensor(0.0 if step in skip_steps else 1.0)

    trainer._optimizer_step = counting_step
    trainer._backward_one = fake_backward
    return calls


def test_no_supervision_still_advances_accumulation(tmp_path):
    model = build_tiny_span_model()
    trainer = GLiNER2Trainer(model, _config(tmp_path, accum=4))
    calls = _instrument(trainer, skip_steps={2, 5})  # 10 batches, 8 succeed
    trainer.train(train_data=_examples(10))
    assert calls["opt"] == math.ceil(10 / 4) == 3


def test_partial_window_with_zero_losses_is_flushed(tmp_path):
    model = build_tiny_span_model()
    trainer = GLiNER2Trainer(model, _config(tmp_path, accum=4))
    calls = _instrument(trainer, skip_steps={1, 4, 7})  # 10 batches, 7 succeed
    trainer.train(train_data=_examples(10))
    assert calls["opt"] == math.ceil(10 / 4) == 3


def test_oom_discards_in_flight_window(tmp_path):
    model = build_tiny_span_model()
    trainer = GLiNER2Trainer(model, _config(tmp_path, accum=4))
    # OOM at step 3 discards the 3 accumulated micro-batches (steps 0-2); the
    # remaining 6 successful (steps 4-9) form one full window + a flush.
    calls = _instrument(trainer, oom_steps={3})
    trainer.train(train_data=_examples(10))
    assert calls["opt"] == math.ceil(6 / 4) == 2


def test_no_stale_gradients_after_training(tmp_path):
    # Real forward/backward: after training every gradient buffer must be
    # cleared (set_to_none), so nothing leaks into a subsequent epoch.
    model = build_tiny_span_model()
    trainer = GLiNER2Trainer(model, _config(tmp_path, accum=2))
    trainer.train(train_data=_examples(4))
    leaked = [
        name for name, p in trainer.model.named_parameters()
        if p.grad is not None and float(p.grad.abs().sum()) > 0
    ]
    assert not leaked, f"stale gradients survived: {leaked[:5]}"


def test_partial_window_gradient_matches_mean_of_present_microbatches(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    trainer = object.__new__(GLiNER2Trainer)
    trainer.model = model
    trainer.config = _config(tmp_path, accum=4)
    with torch.no_grad():
        model.weight.fill_(1.0)

    inputs = (torch.tensor([[1.0]]), torch.tensor([[3.0]]))
    expected_model = torch.nn.Linear(1, 1, bias=False)
    expected_model.load_state_dict(model.state_dict())
    expected_loss = sum((expected_model(value) ** 2).mean() for value in inputs) / 2
    expected_loss.backward()

    for value in inputs:
        ((model(value) ** 2).mean() / 4).backward()
    trainer._renormalize_partial_accumulation(2)
    torch.testing.assert_close(model.weight.grad, expected_model.weight.grad)
