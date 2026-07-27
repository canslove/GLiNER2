"""Training losses for the record head.

Ties together the three record modes with a single, mode-appropriate objective:

* natural  - instances matched to gold by anchor coordinate identity; per-field
  null-aware assignment loss (marginalized over occurrence alternatives).
* latent / anchorless - permutation-invariant Hungarian matching between
  predicted instances and gold records, with unmatched instances trained as
  ``NO_OBJECT`` and matched instances supervised on their field assignments.

Scalar fields use a marginalized softmax NLL over ``{ABSENT} U candidates``;
list fields use per-candidate BCE (absence == all-zero). Gold candidate indices
are resolved from gold spans (which are force-injected as candidates during
training) so identity is coordinate-based, never surface-based.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from gliner2.processing.records import FieldCardinality
from gliner2.processing.targets import RecordTarget, TargetCapacityError
from gliner2.training.matching import linear_sum_assignment
from gliner2.models.boundary.records import RecordGroupOutput


def _span_index(field_spans: torch.LongTensor) -> Dict[Tuple[int, int], int]:
    return {
        (int(field_spans[i, 0]), int(field_spans[i, 1])): i
        for i in range(field_spans.shape[0])
    }


def _resolve_value_cols(
    value_alternatives: Sequence[Tuple[int, int]],
    span_to_idx: Dict[Tuple[int, int], int],
) -> List[int]:
    """Map a value's alternative spans to assignment columns (cand_idx + 1)."""
    cols: List[int] = []
    for span in value_alternatives:
        idx = span_to_idx.get((int(span[0]), int(span[1])))
        if idx is not None:
            cols.append(idx + 1)  # +1 for the ABSENT column at index 0
    return cols


def _scalar_field_nll(
    logits_row: torch.Tensor,        # [1 + Cf]
    target_cols: Sequence[int],
) -> torch.Tensor:
    logp = F.log_softmax(logits_row, dim=-1)
    if not target_cols:
        target_cols = [0]  # ABSENT
    idx = torch.tensor(target_cols, dtype=torch.long, device=logits_row.device)
    return -torch.logsumexp(logp[idx], dim=-1)


def _list_field_bce(
    logits_row: torch.Tensor,        # [1 + Cf]
    positive_cols: Sequence[int],
) -> torch.Tensor:
    cand_logits = logits_row[1:]     # drop ABSENT column
    if cand_logits.numel() == 0:
        return logits_row.new_zeros(())
    target = torch.zeros_like(cand_logits)
    for col in positive_cols:
        target[col - 1] = 1.0
    return F.binary_cross_entropy_with_logits(cand_logits, target, reduction="mean")


def _field_target_cols(
    fspec,
    record: RecordTarget,
    span_to_idx: Dict[Tuple[int, int], int],
) -> Tuple[List[int], bool]:
    """Return (columns, is_scalar) for a gold record's field.

    Scalar: columns are the acceptable candidate columns (occurrence
    alternatives), or empty for ABSENT. List: columns are all gold value
    candidate columns.
    """
    ft = record.field_for_query(fspec.query_id)
    if fspec.cardinality.is_scalar:
        if ft is None or not ft.values:
            return [], True
        # one gold value with occurrence alternatives
        return _resolve_value_cols(ft.values[0], span_to_idx), True
    cols: List[int] = []
    if ft is not None:
        for value in ft.values:
            cols.extend(_resolve_value_cols(value, span_to_idx))
    return cols, False


def _instance_field_loss(
    group: RecordGroupOutput,
    inst: int,
    record: RecordTarget,
    span_indices: List[Dict[Tuple[int, int], int]],
) -> torch.Tensor:
    total = group.object_logits.new_zeros(())
    n_fields = 0
    for f_idx, fspec in enumerate(group.field_specs):
        cols, is_scalar = _field_target_cols(fspec, record, span_indices[f_idx])
        row = group.assign_logits[f_idx][inst]
        if is_scalar:
            total = total + _scalar_field_nll(row, cols)
        else:
            total = total + _list_field_bce(row, cols)
        n_fields += 1
    return total / max(n_fields, 1)


def _instance_field_logprob(
    group: RecordGroupOutput,
    inst: int,
    record: RecordTarget,
    span_indices: List[Dict[Tuple[int, int], int]],
) -> torch.Tensor:
    """Log-likelihood of a gold record's fields under instance ``inst`` (for cost)."""
    total = group.object_logits.new_zeros(())
    for f_idx, fspec in enumerate(group.field_specs):
        cols, is_scalar = _field_target_cols(fspec, record, span_indices[f_idx])
        row = group.assign_logits[f_idx][inst]
        if is_scalar:
            total = total - _scalar_field_nll(row, cols)
        else:
            cand_logits = row[1:]
            if cand_logits.numel() == 0:
                continue
            target = torch.zeros_like(cand_logits)
            for col in cols:
                target[col - 1] = 1.0
            ll = -F.binary_cross_entropy_with_logits(
                cand_logits, target, reduction="sum"
            )
            total = total + ll
    return total


def compute_group_loss(
    group: RecordGroupOutput,
    records: Sequence[RecordTarget],
) -> Dict[str, torch.Tensor]:
    """Compute object + field-assignment losses for one record group.

    Returns a dict with ``object_loss`` and ``field_loss`` tensors.
    """
    device = group.object_logits.device
    zero = torch.zeros((), device=device)
    span_indices = [_span_index(s) for s in group.field_spans]
    ni = group.num_instances

    if group.spec.mode == "natural":
        # Match gold records to instances by anchor coordinate identity.
        anchor_qid = group.spec.anchor_query_id
        anchor_f_idx = group.field_query_ids.index(anchor_qid)
        # instance index by seed candidate index in the anchor field
        seed_to_inst = {
            seed[1]: i
            for i, seed in enumerate(group.instance_seed)
            if seed is not None and seed[0] == anchor_f_idx
        }
        field_loss = zero
        n = 0
        for record in records:
            aft = record.field_for_query(anchor_qid)
            if aft is None or not aft.values:
                continue
            # resolve anchor candidate via any occurrence alternative
            cols = _resolve_value_cols(aft.values[0], span_indices[anchor_f_idx])
            if not cols:
                continue
            anchor_cand = cols[0] - 1
            inst = seed_to_inst.get(anchor_cand)
            if inst is None:
                continue
            field_loss = field_loss + _instance_field_loss(
                group, inst, record, span_indices
            )
            n += 1
        return {
            "object_loss": zero,
            "field_loss": field_loss / max(n, 1),
        }

    # latent / anchorless: Hungarian matching.
    g = len(records)
    if g == 0:
        obj_target = torch.zeros(ni, device=device)
        object_loss = (
            F.binary_cross_entropy_with_logits(group.object_logits, obj_target)
            if ni > 0 else zero
        )
        return {"object_loss": object_loss, "field_loss": zero}
    if ni < g:
        raise TargetCapacityError(
            f"record group task={group.spec.task_index} has {g} gold instances "
            f"but only {ni} instance hypotheses (mode={group.spec.mode}); "
            "increase boundary_head.record_instance_queries or candidate budget."
        )

    obj_logp = F.logsigmoid(group.object_logits)                  # [Ni]
    cost = torch.zeros(ni, g, device=device)
    for i in range(ni):
        for j, record in enumerate(records):
            ll = obj_logp[i] + _instance_field_logprob(group, i, record, span_indices)
            cost[i, j] = -ll
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_rows = set(int(r) for r in row_ind.tolist())
    obj_target = torch.zeros(ni, device=device)
    for r in matched_rows:
        obj_target[r] = 1.0
    object_loss = F.binary_cross_entropy_with_logits(group.object_logits, obj_target)

    field_loss = zero
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        field_loss = field_loss + _instance_field_loss(
            group, int(r), records[int(c)], span_indices
        )
    field_loss = field_loss / max(len(row_ind), 1)
    return {"object_loss": object_loss, "field_loss": field_loss}


__all__ = ["compute_group_loss"]
