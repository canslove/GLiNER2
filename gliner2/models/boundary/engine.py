"""Public boundary extractor class = shared runtime + boundary model core."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import torch

from gliner2.inference.candidate_decoder import token_boundaries_to_character_offsets
from gliner2.inference.runtime import ExtractorRuntimeMixin
from gliner2.models.boundary.model import BoundaryExtractorModel
from gliner2.models.boundary.record_decode import decode_group


def _resolve_flat_spans(
    scored: List[Tuple[float, int, int]]
) -> List[Tuple[float, int, int]]:
    """Greedy flat-span policy: keep highest-scoring, drop overlapping spans."""
    scored = sorted(scored, key=lambda t: (-t[0], t[1], t[2]))
    kept: List[Tuple[float, int, int]] = []
    for score, s, e in scored:
        if all(e <= ks or s >= ke for _, ks, ke in kept):
            kept.append((score, s, e))
    return kept


class BoundaryExtractor(ExtractorRuntimeMixin, BoundaryExtractorModel):
    """Boundary architecture with the shared public extraction runtime.

    Overrides ``_extract_from_batch`` with the sparse candidate path: encode →
    boundary head → threshold + flat-span resolution → exact half-open
    token→character conversion. Entities, classification, and enabled
    record/event schemas are supported; sparse relation decoding is deferred.
    """

    architecture = "boundary"

    def _extract_from_batch(
        self,
        batch,
        threshold: float,
        metadata_list: List[Dict],
        include_confidence: bool,
        include_spans: bool,
    ) -> List[Dict[str, Any]]:
        core = self._encode_core(batch)
        has_queries = core["query_states"].shape[1] > 0
        candidates = None
        probs = None
        if has_queries:
            out = self.boundary_head(
                core["text_states"], core["text_mask"],
                core["query_states"], core["query_mask"],
                return_candidates=True,
            )
            candidates = out.candidates
            probs = torch.sigmoid(candidates.pair_logits)

        results: List[Dict[str, Any]] = []
        for i in range(len(batch)):
            sample: Dict[str, Any] = {}
            specs = core["ext_specs"][i] if has_queries else []
            offset = core["word_offsets"][i]
            start_map = batch.start_mappings[i]
            end_map = batch.end_mappings[i]
            text = batch.original_texts[i]
            text_len = len(start_map)

            record_results = self._decode_records(
                batch, i, core, candidates, offset, start_map, end_map,
                text, text_len, include_confidence, include_spans,
            )
            for name, instances in record_results.items():
                if instances:
                    sample[name] = instances

            entity_results: "OrderedDict[str, Any]" = OrderedDict()
            for qid, spec in enumerate(specs):
                if spec["task_type"] != "entities":
                    continue  # non-entity queries are decoded by the record head
                scored: List[Tuple[float, int, int]] = []
                if bool(candidates.query_mask[i, qid]):
                    c = candidates.indices.shape[2]
                    for ci in range(c):
                        if not bool(candidates.valid_mask[i, qid, ci]):
                            continue
                        p = float(probs[i, qid, ci])
                        if p < threshold:
                            continue
                        s = int(candidates.indices[i, qid, ci, 0])
                        e = int(candidates.indices[i, qid, ci, 1])
                        scored.append((p, s, e))

                spans: List[Tuple[str, float, int, int]] = []
                for p, s, e in _resolve_flat_spans(scored):
                    ts, te = s - offset, e - offset
                    if ts < 0 or te > text_len or te <= ts:
                        continue
                    char_start, char_end = token_boundaries_to_character_offsets(
                        ts, te, start_map, end_map
                    )
                    surface = text[char_start:char_end].strip()
                    if surface:
                        spans.append((surface, p, char_start, char_end))

                entity_results[spec["field_name"]] = self._format_spans(
                    spans, include_confidence, include_spans, already_finalized=True
                )

            if entity_results:
                sample["entities"] = [entity_results]

            schema = batch.original_schemas[i]
            for cls in core["cls_specs"][i]:
                self._extract_classification_result(
                    sample, cls["task_name"], schema,
                    cls["group_embs"], cls["schema_tokens"],
                )

            results.append(sample)

        return results

    def _decode_records(
        self,
        batch,
        sample_index: int,
        core: Dict[str, Any],
        candidates,
        offset: int,
        start_map,
        end_map,
        text: str,
        text_len: int,
        include_confidence: bool,
        include_spans: bool,
    ) -> Dict[str, Any]:
        """Decode record/event groups into public structure output shapes."""
        if not getattr(self, "enable_records", False):
            return {}
        if candidates is None or candidates.candidate_states is None:
            return {}
        record_specs = getattr(batch, "record_specs", ())
        if sample_index >= len(record_specs) or not record_specs[sample_index]:
            return {}

        settings = self.boundary_settings
        query_states_i = core["query_states"][sample_index]
        out: Dict[str, Any] = {}

        def _format_field(spans, is_scalar):
            formatted: List[Tuple[str, float, int, int]] = []
            for (ts_raw, te_raw) in spans:
                ts, te = ts_raw - offset, te_raw - offset
                if ts < 0 or te > text_len or te <= ts:
                    continue
                cs, ce = token_boundaries_to_character_offsets(ts, te, start_map, end_map)
                surface = text[cs:ce].strip()
                if surface:
                    formatted.append((surface, 1.0, cs, ce))
            if is_scalar:
                if not formatted:
                    return None
                s, conf, cs, ce = formatted[0]
                if include_spans and include_confidence:
                    return {"text": s, "confidence": conf, "start": cs, "end": ce}
                if include_spans:
                    return {"text": s, "start": cs, "end": ce}
                if include_confidence:
                    return {"text": s, "confidence": conf}
                return s
            return self._format_spans(
                formatted, include_confidence, include_spans, already_finalized=True
            )

        for task_index, spec in record_specs[sample_index].items():
            group = self.record_decoder.forward_group(
                spec, query_states_i, candidates, sample_index
            )
            decoded = decode_group(
                group,
                anchor_threshold=settings.record_anchor_threshold,
                field_threshold=settings.record_field_threshold,
                object_threshold=settings.record_anchor_threshold,
            )
            instances = []
            for rec in decoded:
                inst: "OrderedDict[str, Any]" = OrderedDict()
                # Emit every declared field in schema order (legacy shape):
                # scalar -> str/None, list -> list[str] (possibly empty).
                for fspec in spec.fields:
                    spans = rec.fields.get(fspec.query_id, [])
                    value = _format_field(spans, fspec.cardinality.is_scalar)
                    if not fspec.cardinality.is_scalar and value is None:
                        value = []
                    inst[fspec.name] = value
                if any(v is not None and v != [] for v in inst.values()):
                    instances.append(inst)
            if instances:
                out[spec.task_name] = instances
        return out


__all__ = ["BoundaryExtractor"]
