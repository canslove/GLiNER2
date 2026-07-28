"""CPU contracts for boundary PR-15 through PR-17."""

from __future__ import annotations

import random

import pytest
import torch

from gliner2.configuration import ExtractorConfig
from gliner2.models.boundary.model import (
    _group_scored_candidates,
    decode_candidates,
)
from gliner2.models.outputs import CandidateTensorBatch
from gliner2.training.sampler import (
    DistributedLengthGroupedSampler,
    LengthGroupedSampler,
)
from gliner2.training.trainer import TrainingConfig


def _padding_waste(order, lengths, batch_size):
    waste = 0
    total = 0
    for start in range(0, len(order), batch_size):
        batch = [lengths[index] for index in order[start:start + batch_size]]
        waste += max(batch) * len(batch) - sum(batch)
        total += max(batch) * len(batch)
    return waste / total


def test_length_grouping_is_deterministic_unique_and_reduces_padding():
    lengths = [max(1, int(2 ** (index / 40))) for index in range(400)]
    sampler = LengthGroupedSampler(
        lengths, batch_size=8, window_batches=50, seed=17
    )
    first = list(sampler)
    assert len(first) == len(lengths)
    assert len(set(first)) == len(lengths)
    assert first == list(sampler)

    sampler.set_epoch(1)
    second = list(sampler)
    assert second != first
    assert set(second) == set(first)

    baseline = list(range(len(lengths)))
    random.Random(17).shuffle(baseline)
    assert _padding_waste(first, lengths, 8) < 0.7 * _padding_waste(
        baseline, lengths, 8
    )


def test_distributed_length_grouping_has_disjoint_equal_step_shards():
    lengths = list(range(1, 103))
    samplers = [
        DistributedLengthGroupedSampler(
            lengths,
            batch_size=8,
            num_replicas=2,
            rank=rank,
            seed=5,
        )
        for rank in range(2)
    ]
    shards = [list(sampler) for sampler in samplers]
    assert set(shards[0]).isdisjoint(shards[1])
    assert len(shards[0]) == len(shards[1])
    assert len(shards[0]) == len(set(shards[0]))
    assert len(shards[1]) == len(set(shards[1]))
    for sampler in samplers:
        sampler.set_epoch(3)
    assert set(samplers[0]).isdisjoint(set(samplers[1]))


def _candidate_batch() -> CandidateTensorBatch:
    indices = torch.tensor(
        [
            [[[0, 2], [1, 3], [2, 4]], [[0, 1], [1, 2], [2, 3]]],
            [[[0, 1], [1, 4], [3, 5]], [[0, 2], [2, 4], [4, 5]]],
        ]
    )
    logits = torch.tensor(
        [[[2.0, -1.0, 2.0], [3.0, 0.0, -3.0]],
         [[-2.0, 4.0, 1.0], [2.0, 2.0, 2.0]]]
    )
    valid = torch.tensor(
        [[[True, False, True], [True, True, True]],
         [[True, True, False], [True, True, True]]]
    )
    query = torch.tensor([[True, False], [True, True]])
    return CandidateTensorBatch(indices, logits + 1, logits, valid, query)


def test_vectorized_pack_preserves_row_major_order_and_offsets():
    candidates = _candidate_batch()
    packed = candidates.pack()
    assert packed.batch_indices.tolist() == [0, 0, 1, 1, 1, 1, 1]
    assert packed.query_indices.tolist() == [0, 0, 0, 0, 1, 1, 1]
    assert list(zip(packed.starts.tolist(), packed.ends.tolist())) == [
        (0, 2), (2, 4), (0, 1), (1, 4), (0, 2), (2, 4), (4, 5)
    ]
    assert packed.offsets.tolist() == [0, 2, 2, 4, 7]


def test_vectorized_decode_grouping_matches_reference_loop():
    candidates = _candidate_batch()
    threshold = 0.5
    probs = torch.sigmoid(candidates.pair_logits)
    expected = [[[] for _ in range(2)] for _ in range(2)]
    for bi in range(2):
        for qi in range(2):
            if not candidates.query_mask[bi, qi]:
                continue
            for ci in range(3):
                if candidates.valid_mask[bi, qi, ci] and probs[bi, qi, ci] >= threshold:
                    start, end = candidates.indices[bi, qi, ci].tolist()
                    expected[bi][qi].append((float(probs[bi, qi, ci]), start, end))
    grouped = _group_scored_candidates(
        candidates, threshold=threshold, probabilities=probs
    )
    assert grouped == expected
    assert decode_candidates(candidates, threshold=threshold) == [
        [
            [(start, end) for _, start, end in sorted(
                scored, key=lambda item: (-item[0], item[1], item[2])
            )]
            for scored in sample
        ]
        for sample in expected
    ]


def test_runtime_config_defaults_and_validation():
    model_config = ExtractorConfig()
    assert model_config.attn_implementation == "sdpa"
    training = TrainingConfig(fp16=False, bf16=False)
    assert training.group_by_length
    assert training.length_group_window_batches == 50
    assert training.fused_optimizer
    with pytest.raises(ValueError, match="attn_implementation"):
        ExtractorConfig(attn_implementation="unknown")
    with pytest.raises(ValueError, match="length_group_window_batches"):
        TrainingConfig(
            fp16=False, bf16=False, length_group_window_batches=0
        )
