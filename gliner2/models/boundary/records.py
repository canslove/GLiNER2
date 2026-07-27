"""Records and events for the boundary architecture: production instance head.

This module implements **Instance Formation and Record Disambiguation**. It
replaces count-first structure decoding with *instance identity* plus
*field-to-instance assignment* built on the same sparse boundary candidates
(no dense grids, no count head):

* **Anchor-driven (natural):** every detected anchor candidate seeds one record
  instance; each non-anchor field candidate is scored against every instance
  with an explicit ``ABSENT`` alternative.
* **Latent anchor:** no declared anchor - a learned selector scores each
  candidate as a potential instance seed, supervised only by record grouping.
* **Anchorless:** document-conditioned learned instance queries cross-attend the
  candidate states and predict object/``NO_OBJECT`` plus per-field pointers.

The low-level primitives ``FieldAssignmentScorer`` and ``RecordSetDecoder`` are
retained (and unit-tested) as building blocks; ``RecordHead`` is the integrated,
schema-aware module used by :class:`BoundaryExtractorModel` and the engine.

Record *count* is never predicted; it is derived from the selected instances by
the global decoder in :mod:`gliner2.models.boundary.record_decode`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from gliner2.models.candidates import CandidateSet
from gliner2.processing.records import FieldCardinality, RecordFieldSpec, RecordSpec


# =============================================================================
# Backward-compatible low-level primitives
# =============================================================================

@dataclass(frozen=True)
class InstanceCandidate:
    """One record/event instance seeded by an anchor (trigger) span."""

    anchor_query_id: int
    anchor_start: int
    anchor_end: int
    score: float


@dataclass
class InstanceCandidateBatch:
    """Padded anchor instances for a batch: states ``[B, N, H]`` + mask ``[B, N]``."""

    states: torch.Tensor
    mask: torch.BoolTensor


def create_anchor_instances(
    anchor_candidates: CandidateSet,
    anchor_query_id: int,
) -> Tuple[InstanceCandidate, ...]:
    """One :class:`InstanceCandidate` per surviving anchor candidate."""
    instances: List[InstanceCandidate] = []
    for start, end, logit in anchor_candidates.for_query(anchor_query_id):
        instances.append(InstanceCandidate(anchor_query_id, start, end, float(logit)))
    return tuple(instances)


class FieldAssignmentScorer(nn.Module):
    """Score assigning each field candidate to each anchor instance.

    Returns ``[B, N, F, C]`` edge logits (anchor N x field F x candidate C).
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.anchor_proj = nn.Linear(hidden_size, hidden_size)
        self.field_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        instance_candidates: InstanceCandidateBatch,
        field_candidate_states: torch.Tensor,   # [B, F, C, H]
        field_query_states: torch.Tensor,       # [B, F, H]
    ) -> torch.Tensor:
        anchor = self.anchor_proj(instance_candidates.states)          # [B, N, H]
        field_q = self.field_proj(field_query_states)                  # [B, F, H]
        query = anchor[:, :, None, :] + field_q[:, None, :, :]         # [B, N, F, H]
        logits = torch.einsum("bnfh,bfch->bnfc", query, field_candidate_states)
        return logits


@dataclass
class RecordSetOutput:
    """Set-decoder output.

    Shapes (``B`` samples, ``I`` instance queries, ``F`` fields, ``C`` candidates):
        object_logits:        [B, I]       object / no-object
        field_pointer_logits: [B, I, F, C] pointer over field candidates
    """

    object_logits: torch.Tensor
    field_pointer_logits: torch.Tensor


class RecordSetDecoder(nn.Module):
    """Fixed instance queries -> object + per-field candidate pointers.

    Object logits are conditioned on the document/schema by cross-attending the
    learned instance queries over the field-candidate states, so the predicted
    record count is input-dependent (well beyond the legacy 19-instance cap).
    """

    def __init__(self, hidden_size: int, instance_queries: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.instance_queries = instance_queries
        self.instance_embed = nn.Parameter(torch.randn(instance_queries, hidden_size) * 0.02)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.object_head = nn.Linear(hidden_size, 1)
        self.inst_proj = nn.Linear(hidden_size, hidden_size)
        self.field_proj = nn.Linear(hidden_size, hidden_size)

    def _condition(
        self,
        inst: torch.Tensor,                      # [B, I, H]
        field_candidate_states: torch.Tensor,    # [B, F, C, H]
        field_mask: Optional[torch.BoolTensor],   # [B, F, C]
    ) -> torch.Tensor:
        b, f, c, h = field_candidate_states.shape
        ctx = field_candidate_states.reshape(b, f * c, h)              # [B, FC, H]
        q = self.q_proj(inst)                                          # [B, I, H]
        k = self.k_proj(ctx)                                          # [B, FC, H]
        v = self.v_proj(ctx)
        attn = torch.einsum("bih,bjh->bij", q, k) / math.sqrt(h)       # [B, I, FC]
        if field_mask is not None:
            m = field_mask.reshape(b, 1, f * c)
            attn = attn.masked_fill(~m, float("-inf"))
            # A fully-masked instance row would produce NaNs; guard it.
            all_masked = ~m.any(dim=-1, keepdim=True)
            attn = attn.masked_fill(all_masked.expand_as(attn), 0.0)
        weights = torch.softmax(attn, dim=-1)
        pooled = torch.einsum("bij,bjh->bih", weights, v)             # [B, I, H]
        return inst + pooled

    def forward(
        self,
        field_query_states: torch.Tensor,        # [B, F, H]
        field_candidate_states: torch.Tensor,    # [B, F, C, H]
        field_mask: Optional[torch.BoolTensor] = None,  # [B, F, C]
    ) -> RecordSetOutput:
        b = field_candidate_states.shape[0]
        inst = self.instance_embed.unsqueeze(0).expand(b, -1, -1)     # [B, I, H]
        inst = self._condition(inst, field_candidate_states, field_mask)
        inst_h = self.inst_proj(inst)                                  # [B, I, H]

        object_logits = self.object_head(inst).squeeze(-1)            # [B, I]

        field_q = self.field_proj(field_query_states)                 # [B, F, H]
        query = inst_h[:, :, None, :] + field_q[:, None, :, :]        # [B, I, F, H]
        logits = torch.einsum("bifh,bfch->bifc", query, field_candidate_states)
        if field_mask is not None:
            logits = logits.masked_fill(~field_mask[:, None, :, :], float("-inf"))
        return RecordSetOutput(object_logits=object_logits, field_pointer_logits=logits)


# =============================================================================
# Integrated, schema-aware record head
# =============================================================================

@dataclass
class RecordGroupOutput:
    """Per-(sample, record group) decoder output consumed by loss + decode.

    ``assign_logits[f]`` has shape ``[Ni, 1 + Cf]``; column 0 is the explicit
    ``ABSENT`` alternative and columns ``1..Cf`` align with ``field_spans[f]``.
    """

    spec: RecordSpec
    object_logits: torch.Tensor                 # [Ni]
    assign_logits: List[torch.Tensor]           # per field: [Ni, 1 + Cf]
    field_query_ids: List[int]
    field_specs: List[RecordFieldSpec]
    field_spans: List[torch.LongTensor]         # per field: [Cf, 2] half-open
    field_cand_mask: List[torch.BoolTensor]     # per field: [Cf]
    field_cand_logits: List[torch.Tensor]       # per field: [Cf] pair logits
    # For natural/latent modes, the (field_index, candidate_index) seed of each
    # instance; None entries for anchorless learned queries.
    instance_seed: List[Optional[Tuple[int, int]]]
    instance_spans: List[Optional[Tuple[int, int]]]

    @property
    def num_instances(self) -> int:
        return int(self.object_logits.shape[0])


class RecordHead(nn.Module):
    """Unified natural / latent / anchorless instance formation head.

    All three modes reduce to (instance states, object logits, null-aware field
    assignment). The head is invoked per sample with that sample's compiled
    :class:`RecordSpec` objects and the boundary candidate batch.
    """

    def __init__(self, hidden_size: int, record_dim: int, instance_queries: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.record_dim = record_dim
        self.instance_queries = instance_queries

        self.inst_proj = nn.Linear(hidden_size, record_dim)
        self.field_proj = nn.Linear(hidden_size, record_dim)
        self.cand_proj = nn.Linear(hidden_size, record_dim)
        self.null_embed = nn.Parameter(torch.randn(record_dim) * 0.02)

        self.object_head = nn.Linear(hidden_size, 1)
        self.latent_seed_head = nn.Linear(hidden_size, 1)

        self.instance_embed = nn.Parameter(torch.randn(instance_queries, hidden_size) * 0.02)
        self.q_proj = nn.Linear(hidden_size, record_dim)
        self.k_proj = nn.Linear(hidden_size, record_dim)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

    # ------------------------------------------------------------------ utils
    def _assign_logits(
        self,
        inst_states: torch.Tensor,        # [Ni, H]
        field_query_states: torch.Tensor,  # [F, H]
        field_cand_states: List[torch.Tensor],  # per field [Cf, H]
    ) -> List[torch.Tensor]:
        inst_q = self.inst_proj(inst_states)                       # [Ni, D]
        field_q = self.field_proj(field_query_states)              # [F, D]
        out: List[torch.Tensor] = []
        for f, cand in enumerate(field_cand_states):
            query = inst_q + field_q[f].unsqueeze(0)               # [Ni, D]
            null_col = query @ self.null_embed                     # [Ni]
            if cand.shape[0] == 0:
                out.append(null_col.unsqueeze(-1))                 # [Ni, 1]
                continue
            cand_p = self.cand_proj(cand)                          # [Cf, D]
            cand_scores = query @ cand_p.t()                       # [Ni, Cf]
            out.append(torch.cat([null_col.unsqueeze(-1), cand_scores], dim=-1))
        return out

    def _anchorless_states(
        self, field_cand_states: List[torch.Tensor]
    ) -> torch.Tensor:
        inst = self.instance_embed                                 # [I, H]
        ctx = [c for c in field_cand_states if c.shape[0] > 0]
        if not ctx:
            return inst
        ctx = torch.cat(ctx, dim=0)                                # [M, H]
        q = self.q_proj(inst)                                      # [I, D]
        k = self.k_proj(ctx)                                       # [M, D]
        v = self.v_proj(ctx)                                       # [M, H]
        attn = (q @ k.t()) / math.sqrt(self.record_dim)            # [I, M]
        weights = torch.softmax(attn, dim=-1)
        pooled = weights @ v                                       # [I, H]
        return inst + pooled

    # ---------------------------------------------------------------- forward
    def forward_group(
        self,
        spec: RecordSpec,
        query_states: torch.Tensor,       # [Q, H] this sample's query states
        candidates,                        # CandidateTensorBatch (sample slice via index)
        sample_index: int,
    ) -> RecordGroupOutput:
        """Decode one record group for one sample."""
        device = query_states.device
        field_specs = list(spec.fields)
        field_query_ids = [f.query_id for f in field_specs]

        # Gather per-field candidate tensors (state / span / mask / logit).
        field_cand_states: List[torch.Tensor] = []
        field_spans: List[torch.LongTensor] = []
        field_cand_mask: List[torch.BoolTensor] = []
        field_cand_logits: List[torch.Tensor] = []
        cand_states_all = candidates.candidate_states
        for f in field_specs:
            qid = f.query_id
            mask = candidates.valid_mask[sample_index, qid]        # [C]
            keep = torch.nonzero(mask, as_tuple=False).flatten()
            spans = candidates.indices[sample_index, qid][keep]    # [Cf, 2]
            logits = candidates.pair_logits[sample_index, qid][keep]  # [Cf]
            states = cand_states_all[sample_index, qid][keep]      # [Cf, H]
            field_cand_states.append(states)
            field_spans.append(spans.to(torch.long))
            field_cand_mask.append(torch.ones(keep.shape[0], dtype=torch.bool, device=device))
            field_cand_logits.append(logits)

        fq = query_states[field_query_ids]                         # [F, H]

        instance_seed: List[Optional[Tuple[int, int]]] = []
        instance_spans: List[Optional[Tuple[int, int]]] = []

        if spec.mode == "natural":
            anchor_field_idx = field_query_ids.index(spec.anchor_query_id)
            anchor_states = field_cand_states[anchor_field_idx]    # [Ca, H]
            anchor_spans = field_spans[anchor_field_idx]
            anchor_logits = field_cand_logits[anchor_field_idx]
            ni = anchor_states.shape[0]
            inst_states = anchor_states
            object_logits = anchor_logits
            for c in range(ni):
                instance_seed.append((anchor_field_idx, c))
                instance_spans.append((int(anchor_spans[c, 0]), int(anchor_spans[c, 1])))
        elif spec.mode == "latent":
            seed_states: List[torch.Tensor] = []
            seed_scores: List[torch.Tensor] = []
            for f_idx, states in enumerate(field_cand_states):
                if states.shape[0] == 0:
                    continue
                scores = self.latent_seed_head(states).squeeze(-1)  # [Cf]
                for c in range(states.shape[0]):
                    seed_states.append(states[c])
                    seed_scores.append(scores[c])
                    instance_seed.append((f_idx, c))
                    sp = field_spans[f_idx][c]
                    instance_spans.append((int(sp[0]), int(sp[1])))
            if seed_states:
                inst_states = torch.stack(seed_states, dim=0)
                object_logits = torch.stack(seed_scores, dim=0)
            else:
                inst_states = query_states.new_zeros((0, self.hidden_size))
                object_logits = query_states.new_zeros((0,))
        else:  # anchorless
            inst_states = self._anchorless_states(field_cand_states)  # [I, H]
            object_logits = self.object_head(inst_states).squeeze(-1)  # [I]
            for _ in range(inst_states.shape[0]):
                instance_seed.append(None)
                instance_spans.append(None)

        assign_logits = self._assign_logits(inst_states, fq, field_cand_states)

        return RecordGroupOutput(
            spec=spec,
            object_logits=object_logits,
            assign_logits=assign_logits,
            field_query_ids=field_query_ids,
            field_specs=field_specs,
            field_spans=field_spans,
            field_cand_mask=field_cand_mask,
            field_cand_logits=field_cand_logits,
            instance_seed=instance_seed,
            instance_spans=instance_spans,
        )


__all__ = [
    "InstanceCandidate",
    "InstanceCandidateBatch",
    "create_anchor_instances",
    "FieldAssignmentScorer",
    "RecordSetOutput",
    "RecordSetDecoder",
    "RecordGroupOutput",
    "RecordHead",
]
