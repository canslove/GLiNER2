"""Seeded boundary golden-batch fixtures."""

from __future__ import annotations

import pytest
import torch

from gliner2.configuration import BoundaryHeadSettings
from gliner2.models.boundary.model import BoundaryHead
from gliner2.processing.targets import MentionTarget, TargetGraph, pad_target_graphs
from gliner2.processor import SamplingConfig, SchemaTransformer
from gliner2.training.data import Classification, InputExample, Structure
from gliner2.training.trainer import ExtractorCollator


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="deliberately replace boundary golden tensors",
    )


@pytest.fixture
def golden_batch(tiny_tokenizer):
    """Fixed CPU/fp32 mixed-length batch with ragged gold supervision."""
    torch.manual_seed(1701)
    lengths = [17, 64, 5, 120]
    batch, queries, hidden = 4, 3, 48
    settings = BoundaryHeadSettings(
        boundary_dim=24,
        pair_dim=24,
        start_top_k=12,
        end_top_k=12,
        ends_per_start=6,
        starts_per_end=6,
        candidate_budget=48,
        training_candidate_budget=64,
        max_gold_per_query=16,
        end_block_size=16,
        multihead_pair_compat_heads=1,
        proposal_loss_weight=0.0,
        consistency_loss_weight=0.0,
        rerank_listwise_weight=0.0,
        soft_iou_aux_weight=0.0,
        enable_abstention=False,
        abstention_loss_weight=0.0,
        enable_count_head=False,
        count_loss_weight=0.0,
        negative_query_ratio=0.0,
        dropout=0.0,
    )
    head = BoundaryHead(hidden, settings, query_dim=hidden)
    head.eval()
    head.collect_diagnostics = True
    token_states = torch.randn(batch, max(lengths), hidden)
    text_mask = (
        torch.arange(max(lengths)).unsqueeze(0)
        < torch.tensor(lengths).unsqueeze(1)
    )
    query_states = torch.randn(batch, queries, hidden)
    query_mask = torch.ones(batch, queries, dtype=torch.bool)
    graphs = [
        TargetGraph(
            mentions=tuple(MentionTarget(0, 0, end) for end in range(1, 17))
        ),
        TargetGraph(mentions=(MentionTarget(1, 2, 4), MentionTarget(1, 8, 12))),
        TargetGraph(),  # deliberate zero-gold sample/query set
        TargetGraph(mentions=(MentionTarget(2, 10, 120),)),
    ]
    targets = pad_target_graphs(graphs, [queries] * batch, lengths, 16)
    texts = [
        " ".join(["x"] * 16),
        " ".join(f"c{i}" for i in range(63)),
        " ".join(f"z{i}" for i in range(4)),
        " ".join(f"s{i}" for i in range(119)),
    ]
    examples = [
        InputExample(text=texts[0], entities={"repeated": ["x"]}),
        InputExample(
            text=texts[1],
            classifications=[
                Classification("sentiment", ["positive", "negative"], "positive")
            ],
        ),
        InputExample(text=texts[2], entities={"absent": []}),
        InputExample(
            text=texts[3],
            structures=[Structure("record", value="s0")],
        ),
    ]
    processor = SchemaTransformer(
        tokenizer=tiny_tokenizer,
        sampling_config=SamplingConfig(
            shuffle_entities=False,
            synthetic_entity_label_prob=0.0,
        ),
    )
    preprocessed = ExtractorCollator(
        processor,
        is_training=True,
        architecture="boundary",
        max_gold_per_query=16,
    )([(example.text, example.to_dict()["output"]) for example in examples])
    assert preprocessed.text_word_counts == lengths
    return {
        "head": head,
        "token_states": token_states,
        "text_mask": text_mask,
        "query_states": query_states,
        "query_mask": query_mask,
        "targets": targets,
        "lengths": lengths,
        "preprocessed_batch": preprocessed,
    }
