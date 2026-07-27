"""Globally consistent record decoding and count derivation.

Turns a :class:`RecordGroupOutput` into concrete record instances subject to
field cardinality, ABSENT handling, and candidate exclusivity. Instances are
processed in descending object score so exclusive candidates resolve to the
most confident record (a deterministic greedy global assignment). The record
count is *derived* as the number of surviving instances - there is no count
head.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from gliner2.models.boundary.records import RecordGroupOutput


@dataclass
class DecodedRecord:
    """One decoded record: field query id -> selected half-open token spans."""

    fields: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    anchor_span: Optional[Tuple[int, int]] = None
    score: float = 0.0


def _dedup_key(rec: DecodedRecord) -> Tuple:
    return tuple(
        (qid, tuple(sorted(spans)))
        for qid, spans in sorted(rec.fields.items())
    )


def decode_group(
    group: RecordGroupOutput,
    *,
    anchor_threshold: float = 0.5,
    field_threshold: float = 0.5,
    object_threshold: float = 0.5,
) -> List[DecodedRecord]:
    """Decode one record group into a list of :class:`DecodedRecord`."""
    ni = group.num_instances
    if ni == 0:
        return []
    obj_prob = torch.sigmoid(group.object_logits.detach())

    if group.spec.mode == "anchorless":
        select_thr = object_threshold
    else:
        select_thr = anchor_threshold

    order = sorted(
        range(ni), key=lambda i: (-float(obj_prob[i]), i)
    )

    used_exclusive: set = set()   # (field_index, cand_index)
    records: List[DecodedRecord] = []

    for inst in order:
        if float(obj_prob[inst]) < select_thr:
            continue
        rec = DecodedRecord(score=float(obj_prob[inst]))
        anchor_field_idx = None
        if group.spec.mode == "natural":
            anchor_field_idx = group.field_query_ids.index(group.spec.anchor_query_id)
            seed = group.instance_seed[inst]
            if seed is not None:
                rec.anchor_span = group.instance_spans[inst]

        for f_idx, fspec in enumerate(group.field_specs):
            qid = fspec.query_id
            spans_tensor = group.field_spans[f_idx]
            logits_row = group.assign_logits[f_idx][inst].detach()

            # Natural anchor field binds to the instance's own seed span.
            if anchor_field_idx is not None and f_idx == anchor_field_idx:
                if rec.anchor_span is not None:
                    rec.fields.setdefault(qid, []).append(rec.anchor_span)
                continue

            if fspec.cardinality.is_scalar:
                probs = torch.softmax(logits_row, dim=-1)
                chosen = None
                # Argmax over available columns; skip used-exclusive candidates.
                ranked = torch.argsort(probs, descending=True).tolist()
                for col in ranked:
                    if col == 0:
                        chosen = 0  # ABSENT wins
                        break
                    cand_idx = col - 1
                    if fspec.exclusive and (f_idx, cand_idx) in used_exclusive:
                        continue
                    chosen = col
                    break
                if chosen is None or chosen == 0:
                    continue
                if float(probs[chosen]) < field_threshold and fspec.allows_absent:
                    continue
                cand_idx = chosen - 1
                span = (int(spans_tensor[cand_idx, 0]), int(spans_tensor[cand_idx, 1]))
                rec.fields.setdefault(qid, []).append(span)
                if fspec.exclusive:
                    used_exclusive.add((f_idx, cand_idx))
            else:  # list field: independent sigmoid per candidate
                cand_logits = logits_row[1:]
                if cand_logits.numel() == 0:
                    continue
                probs = torch.sigmoid(cand_logits)
                selected: List[Tuple[int, int]] = []
                for cand_idx in range(cand_logits.shape[0]):
                    if float(probs[cand_idx]) < field_threshold:
                        continue
                    if fspec.exclusive and (f_idx, cand_idx) in used_exclusive:
                        continue
                    span = (int(spans_tensor[cand_idx, 0]), int(spans_tensor[cand_idx, 1]))
                    selected.append(span)
                    if fspec.exclusive:
                        used_exclusive.add((f_idx, cand_idx))
                if selected:
                    rec.fields.setdefault(qid, []).extend(selected)

        # Required-one fields that ended up absent invalidate nothing we can
        # fabricate; drop the instance only if it carries no fields at all.
        if not rec.fields:
            continue
        records.append(rec)

    # Latent seeds can produce duplicate records; collapse identical field sets.
    if group.spec.mode in ("latent", "anchorless"):
        best: Dict[Tuple, DecodedRecord] = {}
        for rec in records:
            key = _dedup_key(rec)
            if key not in best or rec.score > best[key].score:
                best[key] = rec
        records = list(best.values())

    return records


def derive_count(records: List[DecodedRecord]) -> int:
    """Record count is the number of selected instances - never predicted."""
    return len(records)


__all__ = ["DecodedRecord", "decode_group", "derive_count"]
