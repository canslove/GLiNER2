"""Architecture-neutral model outputs and candidate tensor batches.

These containers are the contract between a model's candidate production and
the shared runtime/decoder. Coordinates are always half-open ``[start, end)``
token boundaries. The span architecture fills these via an adapter; the
boundary architecture produces them natively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from gliner2.models.candidates import CandidateSet


# =============================================================================
# Dense candidate batch (padded)
# =============================================================================

@dataclass
class CandidateTensorBatch:
    """Padded per-query candidate tensor batch.

    Shapes (``B`` samples, ``Q`` queries, ``C`` candidates per query):
        indices:         [B, Q, C, 2]  half-open [start, end)
        proposal_logits: [B, Q, C]
        pair_logits:     [B, Q, C]
        valid_mask:      [B, Q, C]     True where a real candidate exists
        query_mask:      [B, Q]        True where the query is real (extractive)
    """
    indices: torch.LongTensor
    proposal_logits: torch.Tensor
    pair_logits: torch.Tensor
    valid_mask: torch.BoolTensor
    query_mask: torch.BoolTensor
    # Optional per-candidate contextual states [B, Q, C, H], populated only when
    # the record head is enabled. Legacy/entity-only models leave this None.
    candidate_states: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.indices.dim() != 4 or self.indices.shape[-1] != 2:
            raise ValueError(
                f"indices must be [B, Q, C, 2], got {tuple(self.indices.shape)}"
            )
        b, q, c, _ = self.indices.shape
        if tuple(self.proposal_logits.shape) != (b, q, c):
            raise ValueError("proposal_logits shape must be [B, Q, C]")
        if tuple(self.pair_logits.shape) != (b, q, c):
            raise ValueError("pair_logits shape must be [B, Q, C]")
        if tuple(self.valid_mask.shape) != (b, q, c):
            raise ValueError("valid_mask shape must be [B, Q, C]")
        if tuple(self.query_mask.shape) != (b, q):
            raise ValueError("query_mask shape must be [B, Q]")

    @property
    def shape(self):
        return tuple(self.indices.shape)

    def to(self, device) -> "CandidateTensorBatch":
        return CandidateTensorBatch(
            indices=self.indices.to(device),
            proposal_logits=self.proposal_logits.to(device),
            pair_logits=self.pair_logits.to(device),
            valid_mask=self.valid_mask.to(device),
            query_mask=self.query_mask.to(device),
            candidate_states=(
                self.candidate_states.to(device)
                if self.candidate_states is not None else None
            ),
        )

    def validate(self, text_lengths: torch.LongTensor) -> None:
        """Assert every valid candidate is a legal half-open span in range.

        ``0 <= start < end <= text_length`` for the candidate's sample.
        """
        b, q, c, _ = self.indices.shape
        starts = self.indices[..., 0]
        ends = self.indices[..., 1]
        valid = self.valid_mask
        # end > start
        bad_order = valid & (ends <= starts)
        if bool(bad_order.any()):
            raise ValueError("found valid candidate with end <= start")
        # start >= 0
        if bool((valid & (starts < 0)).any()):
            raise ValueError("found valid candidate with start < 0")
        # end <= text_length (broadcast per-sample length)
        lengths = text_lengths.view(b, 1, 1)
        if bool((valid & (ends > lengths)).any()):
            raise ValueError("found valid candidate with end > text_length")

    def pack(self) -> "PackedCandidateBatch":
        """Flatten valid candidates into a ragged packed representation."""
        b, q, c, _ = self.indices.shape
        device = self.indices.device
        batch_idx: List[int] = []
        query_idx: List[int] = []
        starts: List[int] = []
        ends: List[int] = []
        prop: List[float] = []
        pair: List[float] = []
        offsets: List[int] = [0]

        for bi in range(b):
            for qi in range(q):
                if not bool(self.query_mask[bi, qi]):
                    offsets.append(len(starts))
                    continue
                for ci in range(c):
                    if not bool(self.valid_mask[bi, qi, ci]):
                        continue
                    batch_idx.append(bi)
                    query_idx.append(qi)
                    starts.append(int(self.indices[bi, qi, ci, 0]))
                    ends.append(int(self.indices[bi, qi, ci, 1]))
                    prop.append(float(self.proposal_logits[bi, qi, ci]))
                    pair.append(float(self.pair_logits[bi, qi, ci]))
                offsets.append(len(starts))

        long = lambda xs: torch.tensor(xs, dtype=torch.long, device=device)
        flt = lambda xs: torch.tensor(xs, dtype=self.pair_logits.dtype, device=device)
        return PackedCandidateBatch(
            batch_indices=long(batch_idx),
            query_indices=long(query_idx),
            starts=long(starts),
            ends=long(ends),
            proposal_logits=flt(prop),
            pair_logits=flt(pair),
            offsets=long(offsets),
            num_queries=q,
        )


# =============================================================================
# Packed candidate batch (ragged)
# =============================================================================

@dataclass
class PackedCandidateBatch:
    """Ragged packed candidates across a batch.

    ``offsets`` has length ``B * Q + 1`` and delimits each (sample, query) run
    within the flat ``starts``/``ends`` arrays.
    """
    batch_indices: torch.LongTensor      # [N]
    query_indices: torch.LongTensor      # [N]
    starts: torch.LongTensor             # [N]
    ends: torch.LongTensor               # [N]
    proposal_logits: torch.Tensor        # [N]
    pair_logits: torch.Tensor            # [N]
    offsets: torch.LongTensor            # [B * Q + 1]
    num_queries: int = 0

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def to(self, device) -> "PackedCandidateBatch":
        return PackedCandidateBatch(
            batch_indices=self.batch_indices.to(device),
            query_indices=self.query_indices.to(device),
            starts=self.starts.to(device),
            ends=self.ends.to(device),
            proposal_logits=self.proposal_logits.to(device),
            pair_logits=self.pair_logits.to(device),
            offsets=self.offsets.to(device),
            num_queries=self.num_queries,
        )

    def split_by_sample(self) -> List[CandidateSet]:
        """Regroup packed candidates into one ``CandidateSet`` per sample.

        Requires ``num_queries`` to be set (segments are ``B*Q`` runs).
        """
        if self.num_queries <= 0:
            raise ValueError("split_by_sample requires num_queries > 0")
        q = self.num_queries
        num_segments = self.offsets.shape[0] - 1
        if num_segments % q != 0:
            raise ValueError("offsets length inconsistent with num_queries")
        num_samples = num_segments // q
        device = self.starts.device

        sets: List[CandidateSet] = []
        for bi in range(num_samples):
            q_ids: List[int] = []
            s: List[int] = []
            e: List[int] = []
            lg: List[float] = []
            for qi in range(q):
                seg = bi * q + qi
                lo = int(self.offsets[seg])
                hi = int(self.offsets[seg + 1])
                for k in range(lo, hi):
                    q_ids.append(qi)
                    s.append(int(self.starts[k]))
                    e.append(int(self.ends[k]))
                    lg.append(float(self.pair_logits[k]))
            sets.append(
                CandidateSet(
                    query_ids=torch.tensor(q_ids, dtype=torch.long, device=device),
                    starts=torch.tensor(s, dtype=torch.long, device=device),
                    ends=torch.tensor(e, dtype=torch.long, device=device),
                    logits=torch.tensor(lg, dtype=self.pair_logits.dtype, device=device),
                )
            )
        return sets


# =============================================================================
# Final model output
# =============================================================================

@dataclass
class ExtractorOutput:
    """Architecture-neutral forward output.

    A plain dataclass (not ``transformers.ModelOutput``) so it can carry a
    non-optional ``batch_size`` and component-loss keys. Supports both attribute
    and ``output["total_loss"]`` mapping access for trainer compatibility. Span
    models continue to expose
    ``total_loss/classification_loss/structure_loss/count_loss/batch_size``;
    boundary models additionally expose ``start_loss/end_loss/pair_loss/...``
    via the ``losses`` dict, mirrored as mapping keys.
    """
    loss: Optional[torch.Tensor] = None
    total_loss: Optional[torch.Tensor] = None
    losses: Optional[Dict[str, torch.Tensor]] = None
    candidates: Optional[CandidateTensorBatch] = None
    packed_candidates: Optional[PackedCandidateBatch] = None
    classification_logits: Optional[torch.Tensor] = None
    classification_mask: Optional[torch.BoolTensor] = None
    start_logits: Optional[torch.Tensor] = None
    end_logits: Optional[torch.Tensor] = None
    inside_logits: Optional[torch.Tensor] = None
    metrics: Optional[Dict[str, torch.Tensor]] = None
    batch_size: int = 0

    def __getitem__(self, key):
        # Prefer a real attribute; fall back to the losses dict for component
        # losses (e.g. "start_loss") so trainers can read them uniformly.
        if isinstance(key, str):
            if hasattr(self, key) and getattr(self, key) is not None:
                return getattr(self, key)
            if self.losses is not None and key in self.losses:
                return self.losses[key]
            if hasattr(self, key):
                return getattr(self, key)
            raise KeyError(key)
        raise KeyError(key)

    def __contains__(self, key) -> bool:
        if isinstance(key, str):
            if hasattr(self, key) and getattr(self, key) is not None:
                return True
            return self.losses is not None and key in self.losses
        return False

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
