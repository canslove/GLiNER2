"""Record identity preservation through boundary preprocessing."""

from __future__ import annotations

import pytest

from gliner2.processor import SamplingConfig, SchemaTransformer
from gliner2.training import ExtractorCollator


def _deterministic_processor(tokenizer):
    return SchemaTransformer(
        tokenizer=tokenizer,
        sampling_config=SamplingConfig(
            remove_json_structure_prob=0.0,
            shuffle_json_fields=False,
            remove_json_field_prob=0.0,
            synthetic_entity_label_prob=0.0,
            remove_relations_prob=0.0,
            swap_head_tail_prob=0.0,
            remove_classification_prob=0.0,
        ),
    )


def _collate(tokenizer, text, output, *, is_training=True):
    processor = _deterministic_processor(tokenizer)
    collator = ExtractorCollator(
        processor, is_training=is_training, architecture="boundary",
        max_gold_per_query=16,
    )
    return collator([(text, output)])


def test_natural_records_preserve_instance_grouping(tiny_tokenizer):
    output = {
        "json_structures": [
            {"purchase": {"buyer": "Alice", "item": "apples"}},
            {"purchase": {"buyer": "Bob", "item": "oranges"}},
        ],
        "record_metadata": {"purchase": {"mode": "natural", "anchor": "buyer"}},
    }
    batch = _collate(tiny_tokenizer, "Alice bought apples . Bob bought oranges .", output)

    specs = batch.record_specs[0]
    assert len(specs) == 1
    spec = next(iter(specs.values()))
    assert spec.mode == "natural"
    assert spec.anchor_query_id is not None

    records = batch.targets.records[0]
    assert len(records) == 2
    buyers = set()
    for rec in records:
        bf = rec.field_for_query(spec.anchor_query_id)
        assert bf is not None and bf.values  # anchor present
        buyers.add(bf.values[0])  # occurrence-alternative tuple
    # Two distinct anchor bindings -> two distinct instances by coordinate.
    assert len(buyers) == 2


def test_repeated_surface_kept_as_occurrence_alternatives(tiny_tokenizer):
    # "Bob" appears twice; latent_all (default) keeps both as OR alternatives
    # within the same record rather than flattening across records.
    output = {
        "json_structures": [{"sighting": {"who": "Bob"}}],
        "record_metadata": {
            "sighting": {"mode": "latent", "occurrence_policy": "latent_all"}
        },
    }
    batch = _collate(tiny_tokenizer, "Bob greeted Bob warmly today .", output)
    records = batch.targets.records[0]
    assert len(records) == 1
    spec = next(iter(batch.record_specs[0].values()))
    who = records[0].field_for_query(spec.fields[0].query_id)
    # zero_or_more default -> each occurrence is its own value here (list field)
    all_spans = [sp for value in who.values for sp in value]
    assert len(all_spans) == 2
    assert len(set(all_spans)) == 2  # distinct coordinates


def test_error_on_ambiguous_scalar_raises(tiny_tokenizer):
    output = {
        "json_structures": [{"sighting": {"who": "Bob"}}],
        "record_metadata": {
            "sighting": {
                "mode": "latent",
                "occurrence_policy": "error_on_ambiguous",
                "fields": {"who": {"cardinality": "required_one"}},
            }
        },
    }
    with pytest.raises(ValueError, match="ambiguous"):
        _collate(tiny_tokenizer, "Bob greeted Bob warmly today .", output)


def test_legacy_structure_without_metadata_has_no_specs(tiny_tokenizer):
    output = {"json_structures": [{"plain": {"a": "Alice"}}]}
    batch = _collate(tiny_tokenizer, "Alice went home .", output)
    assert batch.record_specs[0] == {}
    # Records are only built for annotated groups.
    assert batch.targets.records is None
