"""Sparse, typed relation extraction for the boundary architecture.

Relations reuse the entity mention candidates rather than introducing a second
extraction representation. Pair generation is *typed and capped*: for each
relation type we keep only the top-``Rh`` head-typed and top-``Rt`` tail-typed
mentions and score their capped cross product. Work is therefore ``O(Rh*Rt)``
per relation type with fixed caps — never the ``O(N^2)`` all-pairs matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from gliner2.models.base import QueryLayout
from gliner2.models.outputs import CandidateTensorBatch


@dataclass(frozen=True)
class RelationProposalSettings:
    heads_per_relation: int = 32
    tails_per_relation: int = 32
    pair_cap: int = 128


@dataclass(frozen=True)
class RelationTypeSpec:
    """A relation type with its allowed head/tail entity queries."""

    relation_type: str
    head_query_ids: Tuple[int, ...]
    tail_query_ids: Tuple[int, ...]
    allow_self: bool = False


@dataclass
class RelationPairBatch:
    """Flattened typed relation-pair proposals across a batch.

    All index tensors have shape ``[P]`` (P = total proposed pairs). ``*_end``
    are half-open (last covered token is ``end - 1``).
    """

    batch_index: torch.LongTensor
    relation_index: torch.LongTensor
    head_start: torch.LongTensor
    head_end: torch.LongTensor
    tail_start: torch.LongTensor
    tail_end: torch.LongTensor
    head_prob: torch.Tensor
    tail_prob: torch.Tensor
    head_keys: List[Tuple[str, int, int]] = field(default_factory=list)
    tail_keys: List[Tuple[str, int, int]] = field(default_factory=list)
    relation_types: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.batch_index.shape[0])


def _query_type(layout: QueryLayout, query_id: int) -> str:
    try:
        return layout.query(query_id).role_name
    except KeyError:
        return str(query_id)


def _top_mentions(
    candidates: CandidateTensorBatch,
    b: int,
    query_ids: Sequence[int],
    top_k: int,
) -> List[Tuple[float, int, int, int]]:
    """Return up to ``top_k`` (prob, start, end, query_id) for the given queries."""
    scored: List[Tuple[float, int, int, int]] = []
    probs = torch.sigmoid(candidates.pair_logits)
    c = candidates.indices.shape[2]
    for qid in query_ids:
        if qid >= candidates.query_mask.shape[1] or not bool(candidates.query_mask[b, qid]):
            continue
        for ci in range(c):
            if not bool(candidates.valid_mask[b, qid, ci]):
                continue
            p = float(probs[b, qid, ci].detach())
            s = int(candidates.indices[b, qid, ci, 0])
            e = int(candidates.indices[b, qid, ci, 1])
            scored.append((p, s, e, qid))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    return scored[:top_k]


class TypedRelationPairGenerator:
    """Generate typed, capped relation pairs from entity candidates."""

    def __init__(self, settings: RelationProposalSettings | None = None) -> None:
        self.settings = settings or RelationProposalSettings()

    def generate(
        self,
        candidates: CandidateTensorBatch,
        query_layouts: Sequence[QueryLayout],
        relation_schema: Sequence[RelationTypeSpec],
    ) -> RelationPairBatch:
        s = self.settings
        bi: List[int] = []
        ri: List[int] = []
        hs: List[int] = []
        he: List[int] = []
        ts: List[int] = []
        te: List[int] = []
        hp: List[float] = []
        tp: List[float] = []
        hkeys: List[Tuple[str, int, int]] = []
        tkeys: List[Tuple[str, int, int]] = []
        rtypes: List[str] = []

        batch_size = candidates.indices.shape[0]
        for b in range(batch_size):
            layout = query_layouts[b] if b < len(query_layouts) else None
            for r_index, spec in enumerate(relation_schema):
                heads = _top_mentions(candidates, b, spec.head_query_ids, s.heads_per_relation)
                tails = _top_mentions(candidates, b, spec.tail_query_ids, s.tails_per_relation)
                pairs: List[Tuple[float, tuple, tuple]] = []
                for (hprob, h0, h1, hq) in heads:
                    for (tprob, t0, t1, tq) in tails:
                        if not spec.allow_self and (h0, h1) == (t0, t1):
                            continue
                        pairs.append((hprob * tprob, (hprob, h0, h1, hq), (tprob, t0, t1, tq)))
                pairs.sort(key=lambda t: -t[0])
                for _, (hprob, h0, h1, hq), (tprob, t0, t1, tq) in pairs[: s.pair_cap]:
                    bi.append(b)
                    ri.append(r_index)
                    hs.append(h0); he.append(h1); ts.append(t0); te.append(t1)
                    hp.append(hprob); tp.append(tprob)
                    htype = _query_type(layout, hq) if layout is not None else str(hq)
                    ttype = _query_type(layout, tq) if layout is not None else str(tq)
                    hkeys.append((htype, h0, h1))
                    tkeys.append((ttype, t0, t1))
                    rtypes.append(spec.relation_type)

        device = candidates.indices.device
        long = lambda xs: torch.tensor(xs, dtype=torch.long, device=device)
        flt = lambda xs: torch.tensor(xs, dtype=torch.float, device=device)
        return RelationPairBatch(
            batch_index=long(bi), relation_index=long(ri),
            head_start=long(hs), head_end=long(he),
            tail_start=long(ts), tail_end=long(te),
            head_prob=flt(hp), tail_prob=flt(tp),
            head_keys=hkeys, tail_keys=tkeys, relation_types=rtypes,
        )


class SparseRelationScorer(nn.Module):
    """Score proposed relation pairs from boundary endpoint states.

    Uses only local features (four endpoint boundary states, the relation query,
    relative order and a normalized distance) — no dense pair matrix.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        in_dim = 5 * hidden_size + 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        boundary_states: torch.Tensor,          # [B, L, H]
        relation_query_states: torch.Tensor,    # [B, R, H]
        entity_candidates: CandidateTensorBatch,  # (unused indices carrier; kept for API)
        relation_pairs: RelationPairBatch,
    ) -> torch.Tensor:
        if len(relation_pairs) == 0:
            return boundary_states.new_zeros(0)

        b = relation_pairs.batch_index
        length = boundary_states.shape[1]

        def gather(pos: torch.Tensor) -> torch.Tensor:
            pos = pos.clamp(0, max(length - 1, 0))
            return boundary_states[b, pos]

        h_start = gather(relation_pairs.head_start)
        h_end = gather(relation_pairs.head_end - 1)
        t_start = gather(relation_pairs.tail_start)
        t_end = gather(relation_pairs.tail_end - 1)
        rel = relation_query_states[b, relation_pairs.relation_index]

        delta = (relation_pairs.tail_start - relation_pairs.head_start).float()
        order = torch.sign(delta).unsqueeze(-1)
        dist = (delta.abs() / float(max(length, 1))).unsqueeze(-1)

        feats = torch.cat([h_start, h_end, t_start, t_end, rel, order, dist], dim=-1)
        return self.mlp(feats).squeeze(-1)


__all__ = [
    "RelationProposalSettings",
    "RelationTypeSpec",
    "RelationPairBatch",
    "TypedRelationPairGenerator",
    "SparseRelationScorer",
]
