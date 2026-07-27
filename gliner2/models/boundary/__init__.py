"""Sparse boundary architecture primitives.

Half-open ``[start, end)`` coordinates throughout. There is no width axis and
no production ``[L, L]`` or ``[L, W, D]`` tensor: proposal work streams over end
blocks and stays linear in sequence length for fixed schema/budgets.
"""

from gliner2.models.boundary.encoding import (
    BoundaryEncoding,
    BoundaryEncoder,
    build_boundary_mask,
    shift_left_with_bos,
    shift_right_with_eos,
)
from gliner2.models.boundary.heads import BoundaryMarginals, BoundaryQueryHead
from gliner2.models.boundary.proposal import (
    ProposalSettings,
    ProposalStats,
    BoundaryProposals,
    SparseBoundaryProposer,
)
from gliner2.models.boundary.scoring import SparseBoundaryPairScorer
from gliner2.models.boundary.losses import (
    asymmetric_focal_loss,
    balanced_multilabel_bce,
    build_candidate_labels,
    candidate_pair_loss,
    inside_consistency_loss,
    select_hard_negative_candidates,
)
from gliner2.models.boundary.model import (
    BoundaryExtractorModel,
    BoundaryHead,
    decode_candidates,
    proposal_settings_from_head,
)

__all__ = [
    "BoundaryEncoding",
    "BoundaryEncoder",
    "build_boundary_mask",
    "shift_left_with_bos",
    "shift_right_with_eos",
    "BoundaryMarginals",
    "BoundaryQueryHead",
    "ProposalSettings",
    "ProposalStats",
    "BoundaryProposals",
    "SparseBoundaryProposer",
    "SparseBoundaryPairScorer",
    "asymmetric_focal_loss",
    "balanced_multilabel_bce",
    "build_candidate_labels",
    "candidate_pair_loss",
    "inside_consistency_loss",
    "select_hard_negative_candidates",
    "BoundaryExtractorModel",
    "BoundaryHead",
    "decode_candidates",
    "proposal_settings_from_head",
]
