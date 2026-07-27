"""Architecture-neutral sparse candidate score format for joint IE.

The span architecture produces a dense width-oriented :class:`ScoreLattice`;
the boundary architecture produces sparse candidates directly. Both are mapped
onto :class:`CandidateScoreSet` — a flat list of mention scores plus optional
relation-role scores — which then feeds the *unchanged* ``NodeCandidate`` /
``EdgeCandidate`` / ``JointProblem`` optimizer contract. Coordinates are
half-open ``[start, end)`` token offsets throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, List, Mapping, Optional, Sequence, Tuple

from gliner2.joint_ie.candidates import (
    EdgeCandidate,
    JointProblem,
    NodeCandidate,
    center_logit,
    sigmoid,
)


@dataclass(frozen=True)
class MentionScore:
    """One scored span mention (half-open ``[start, end)`` token offsets)."""

    query_id: int
    entity_type: str
    start: int
    end: int
    logit: float
    probability: float

    @property
    def key(self) -> Tuple[str, int, int]:
        return (self.entity_type, self.start, self.end)


@dataclass(frozen=True)
class RelationRoleScore:
    """A mention's compatibility with one role of one relation type."""

    relation_type: str
    role: str  # "head" | "tail"
    mention_id: Hashable
    logit: float
    probability: float


@dataclass
class CandidateScoreSet:
    """Sparse, architecture-neutral candidate scores for one text."""

    text: str
    mentions: Tuple[MentionScore, ...]
    relation_roles: Tuple[RelationRoleScore, ...] = ()
    classifications: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


def score_lattice_to_candidate_score_set(lattice: Any) -> CandidateScoreSet:
    """Convert a dense span :class:`ScoreLattice` into a sparse score set.

    Reads the entity task's single count hypothesis (``role_logits[0]`` shaped
    ``[num_types, L, W]``) and emits one :class:`MentionScore` per valid,
    above-floor span cell, mapping inclusive width cells to half-open spans.
    """
    span_starts = lattice.span_starts
    span_ends = lattice.span_ends
    valid = lattice.valid_span_mask

    mentions: List[MentionScore] = []
    query_id = 0
    for task in lattice.tasks:
        if task.task_type != "entities" or not task.count_hypotheses:
            continue
        hyp = task.count_hypotheses[0]
        role_logits = hyp.role_logits[0]           # [num_types, L, W]
        role_probs = hyp.role_probabilities[0]
        num_types = role_logits.shape[0]
        for t in range(num_types):
            entity_type = task.roles[t] if t < len(task.roles) else str(t)
            length = role_logits.shape[1]
            width = role_logits.shape[2]
            for i in range(length):
                for w in range(width):
                    if not bool(valid[i, w]):
                        continue
                    prob = float(role_probs[t, i, w])
                    start = int(span_starts[i, w])
                    end = int(span_ends[i, w]) + 1  # inclusive -> half-open
                    mentions.append(
                        MentionScore(
                            query_id=query_id,
                            entity_type=entity_type,
                            start=start,
                            end=end,
                            logit=float(role_logits[t, i, w]),
                            probability=prob,
                        )
                    )
            query_id += 1

    return CandidateScoreSet(text=lattice.text, mentions=tuple(mentions))


@dataclass(frozen=True)
class ScoredRelationEdge:
    """A scored (head, tail) relation proposal referencing mention keys."""

    relation_type: str
    head: Hashable
    tail: Hashable
    logit: float
    probability: float


def candidate_score_set_to_problem(
    score_set: CandidateScoreSet,
    edges: Sequence[ScoredRelationEdge] = (),
    *,
    mention_threshold: float = 0.5,
    constraints: Sequence[Any] = (),
    decision_threshold: float = 0.5,
) -> JointProblem:
    """Build a :class:`JointProblem` from sparse mention + edge scores.

    Node/edge utilities are centered log-odds (positive => above threshold), so
    the existing greedy/beam optimizers and constraints work unchanged.
    """
    nodes: List[NodeCandidate] = []
    keep_ids = set()
    for m in score_set.mentions:
        if m.probability < mention_threshold:
            continue
        node = NodeCandidate(
            entity_type=m.entity_type,
            start=m.start,
            end=m.end,
            score=center_logit(m.logit, decision_threshold),
            probability=m.probability,
            candidate_id=m.key,
        )
        nodes.append(node)
        keep_ids.add(m.key)

    edge_cands: List[EdgeCandidate] = []
    for e in edges:
        if e.head not in keep_ids or e.tail not in keep_ids:
            continue
        edge_cands.append(
            EdgeCandidate(
                relation_type=e.relation_type,
                head=e.head,
                tail=e.tail,
                score=center_logit(e.logit, decision_threshold),
                head_probability=e.probability,
                tail_probability=e.probability,
            )
        )

    return JointProblem(
        nodes=tuple(nodes),
        edges=tuple(edge_cands),
        constraints=tuple(constraints),
    )


__all__ = [
    "MentionScore",
    "RelationRoleScore",
    "CandidateScoreSet",
    "ScoredRelationEdge",
    "score_lattice_to_candidate_score_set",
    "candidate_score_set_to_problem",
]
