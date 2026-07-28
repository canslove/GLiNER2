"""Relation role binding, rescue threshold, and directional scoring tests."""

import torch

from gliner2.models.base import QueryLayout
from gliner2.models.boundary.relations import (
    RelationProposalSettings,
    RelationTypeSpec,
    SparseRelationScorer,
    TypedRelationPairGenerator,
)
from gliner2.models.outputs import CandidateTensorBatch


def test_argument_threshold_filters_low_probability_mentions():
    candidates = CandidateTensorBatch(
        indices=torch.tensor([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]]),
        proposal_logits=torch.zeros(1, 2, 2),
        pair_logits=torch.tensor([[[5.0, -3.0], [5.0, -3.0]]]),
        valid_mask=torch.ones(1, 2, 2, dtype=torch.bool),
        query_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    generator = TypedRelationPairGenerator(
        RelationProposalSettings(argument_threshold=0.2)
    )
    pairs = generator.generate(
        candidates,
        [QueryLayout(queries=())],
        [RelationTypeSpec("related", (0,), (1,))],
    )
    assert len(pairs) == 1
    assert (int(pairs.head_start[0]), int(pairs.tail_start[0])) == (0, 4)


def test_directional_biaffine_relation_scorer_runs_and_backpropagates():
    hidden = 8
    scorer = SparseRelationScorer(
        hidden,
        relation_query_dim=2 * hidden,
        use_biaffine_content=True,
    )
    candidates = CandidateTensorBatch(
        indices=torch.tensor([[[[0, 2]], [[3, 5]]]]),
        proposal_logits=torch.zeros(1, 2, 1),
        pair_logits=torch.ones(1, 2, 1),
        valid_mask=torch.ones(1, 2, 1, dtype=torch.bool),
        query_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    pairs = TypedRelationPairGenerator().generate(
        candidates,
        [QueryLayout(queries=())],
        [RelationTypeSpec("related", (0,), (1,))],
    )
    text = torch.randn(1, 6, hidden, requires_grad=True)
    relation = torch.randn(1, 1, 2 * hidden, requires_grad=True)
    logits = scorer(text, relation, candidates, pairs)
    assert logits.shape == (1,)
    logits.sum().backward()
    assert torch.isfinite(text.grad).all()
    assert torch.isfinite(relation.grad).all()


def test_boundary_model_binds_relation_roles_by_position():
    from gliner2 import ExtractorConfig
    from gliner2.inference.engine import BoundaryExtractor
    from gliner2.processor import SamplingConfig, SchemaTransformer
    from gliner2.training import ExtractorCollator
    from tests.fixtures.tiny_boundary_checkpoint import TINY_BOUNDARY_HEAD
    from tests.fixtures.tiny_encoder import build_tiny_encoder_config
    from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer

    tokenizer = build_tiny_tokenizer()
    head = dict(TINY_BOUNDARY_HEAD)
    head.update(enable_relations=True, directional_relation_states=True)
    model = BoundaryExtractor(
        ExtractorConfig(
            model_name="tiny-bert-fixture",
            architecture="boundary",
            boundary_head=head,
        ),
        encoder_config=build_tiny_encoder_config(vocab_size=len(tokenizer)),
        tokenizer=tokenizer,
    )
    processor = SchemaTransformer(
        tokenizer=tokenizer,
        sampling_config=SamplingConfig(
            remove_relations_prob=0.0,
            swap_head_tail_prob=0.0,
            synthetic_entity_label_prob=0.0,
        ),
    )
    batch = ExtractorCollator(
        processor, is_training=True, architecture="boundary"
    )([
        (
            "Alice works for Acme",
            {"relations": [{"works_for": {"subject": "Alice", "object": "Acme"}}]},
        )
    ])
    core = model._encode_core(batch)
    assert len(core["rel_specs"][0]) == 1
    spec = core["rel_specs"][0][0]
    assert spec["spec"].head_query_ids == (0,)
    assert spec["spec"].tail_query_ids == (1,)
    assert spec["query_state"].shape[-1] == 2 * model.hidden_size
