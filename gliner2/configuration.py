"""Shared, validated configuration for GLiNER2 architectures.

``ExtractorConfig`` is architecture-aware: a missing ``architecture`` field
resolves to ``"span"`` (legacy checkpoints), and span/boundary heads carry
their own validated settings. The config remains a ``PretrainedConfig`` so the
standard save/load contract and ``return_unused_kwargs`` behavior are intact.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping

try:  # Literal is stdlib on 3.8+
    from typing import Literal
except ImportError:  # pragma: no cover
    Literal = None  # type: ignore

from transformers import PretrainedConfig

ArchitectureName = str  # Literal["span", "boundary"] conceptually.

_KNOWN_ARCHITECTURES = ("span", "boundary")


# =============================================================================
# Head settings
# =============================================================================

@dataclass(frozen=True)
class SpanHeadSettings:
    max_width: int = 8
    span_mode: str = "markerV0"
    dropout: float = 0.1


@dataclass(frozen=True)
class BoundaryHeadSettings:
    boundary_dim: int = 128
    pair_dim: int = 128
    boundary_refinement_layers: int = 1
    boundary_ffn_multiplier: float = 2.0
    start_top_k: int = 16
    end_top_k: int = 16
    ends_per_start: int = 8
    starts_per_end: int = 8
    candidate_budget: int = 128
    training_candidate_budget: int = 160
    max_gold_per_query: int = 32
    end_block_size: int = 256
    bidirectional_proposals: bool = True
    use_inside_evidence: bool = True
    dropout: float = 0.1
    export_mode: str = "streaming"  # "streaming" | "vectorized" (export-friendly)
    # -- Record / instance-formation head (Instance Formation & Record
    # Disambiguation). Disabled by default so existing boundary checkpoints and
    # entity-only models are byte-for-byte unaffected. Only architecture-wide
    # *capacities* live here; per-schema mode/anchor is side metadata.
    enable_records: bool = False
    record_dim: int = 128
    record_instance_queries: int = 32       # anchorless / latent capacity
    record_anchor_proposal_threshold: float = 0.2   # lower rescue threshold
    record_anchor_threshold: float = 0.5    # final anchor selection threshold
    record_field_threshold: float = 0.5     # list-field / null decision cutoff
    record_loss_weight: float = 1.0


# =============================================================================
# Validation / normalization functions
# =============================================================================

def normalize_architecture(value: str) -> ArchitectureName:
    """Normalize and validate an architecture name.

    Raises:
        ValueError: If ``value`` is not a known architecture.
    """
    if value is None:
        return "span"
    normalized = str(value).strip().lower()
    if normalized not in _KNOWN_ARCHITECTURES:
        expected = ", ".join(repr(a) for a in _KNOWN_ARCHITECTURES)
        raise ValueError(
            f"Unknown extractor architecture {value!r}.\n"
            f"Expected one of: {expected}."
        )
    return normalized


def validate_span_head(values: Mapping[str, Any]) -> dict:
    """Validate span-head settings, filling defaults."""
    defaults = SpanHeadSettings()
    result = {
        "max_width": int(values.get("max_width", defaults.max_width)),
        "span_mode": str(values.get("span_mode", defaults.span_mode)),
        "dropout": float(values.get("dropout", defaults.dropout)),
    }
    if result["max_width"] <= 0:
        raise ValueError(f"span_head.max_width must be > 0, got {result['max_width']}")
    if not 0.0 <= result["dropout"] < 1.0:
        raise ValueError(f"span_head.dropout must be in [0, 1), got {result['dropout']}")
    return result


def validate_boundary_head(values: Mapping[str, Any]) -> dict:
    """Validate boundary-head settings, filling defaults and enforcing rules."""
    d = BoundaryHeadSettings()
    result = {
        "boundary_dim": int(values.get("boundary_dim", d.boundary_dim)),
        "pair_dim": int(values.get("pair_dim", d.pair_dim)),
        "boundary_refinement_layers": int(
            values.get("boundary_refinement_layers", d.boundary_refinement_layers)
        ),
        "boundary_ffn_multiplier": float(
            values.get("boundary_ffn_multiplier", d.boundary_ffn_multiplier)
        ),
        "start_top_k": int(values.get("start_top_k", d.start_top_k)),
        "end_top_k": int(values.get("end_top_k", d.end_top_k)),
        "ends_per_start": int(values.get("ends_per_start", d.ends_per_start)),
        "starts_per_end": int(values.get("starts_per_end", d.starts_per_end)),
        "candidate_budget": int(values.get("candidate_budget", d.candidate_budget)),
        "training_candidate_budget": int(
            values.get("training_candidate_budget", d.training_candidate_budget)
        ),
        "max_gold_per_query": int(values.get("max_gold_per_query", d.max_gold_per_query)),
        "end_block_size": int(values.get("end_block_size", d.end_block_size)),
        "bidirectional_proposals": bool(
            values.get("bidirectional_proposals", d.bidirectional_proposals)
        ),
        "use_inside_evidence": bool(
            values.get("use_inside_evidence", d.use_inside_evidence)
        ),
        "dropout": float(values.get("dropout", d.dropout)),
        "export_mode": str(values.get("export_mode", d.export_mode)),
        "enable_records": bool(values.get("enable_records", d.enable_records)),
        "record_dim": int(values.get("record_dim", d.record_dim)),
        "record_instance_queries": int(
            values.get("record_instance_queries", d.record_instance_queries)
        ),
        "record_anchor_proposal_threshold": float(
            values.get(
                "record_anchor_proposal_threshold", d.record_anchor_proposal_threshold
            )
        ),
        "record_anchor_threshold": float(
            values.get("record_anchor_threshold", d.record_anchor_threshold)
        ),
        "record_field_threshold": float(
            values.get("record_field_threshold", d.record_field_threshold)
        ),
        "record_loss_weight": float(
            values.get("record_loss_weight", d.record_loss_weight)
        ),
    }
    if result["export_mode"] not in ("streaming", "vectorized"):
        raise ValueError(
            f"boundary_head.export_mode must be 'streaming' or 'vectorized', "
            f"got {result['export_mode']!r}"
        )
    if result["record_dim"] <= 0:
        raise ValueError(
            f"boundary_head.record_dim must be > 0, got {result['record_dim']}"
        )
    if result["record_instance_queries"] <= 0:
        raise ValueError(
            "boundary_head.record_instance_queries must be > 0, got "
            f"{result['record_instance_queries']}"
        )
    for thr_key in (
        "record_anchor_proposal_threshold",
        "record_anchor_threshold",
        "record_field_threshold",
    ):
        if not 0.0 <= result[thr_key] <= 1.0:
            raise ValueError(
                f"boundary_head.{thr_key} must be in [0, 1], got {result[thr_key]}"
            )
    if result["record_anchor_proposal_threshold"] > result["record_anchor_threshold"]:
        raise ValueError(
            "boundary_head.record_anchor_proposal_threshold "
            f"({result['record_anchor_proposal_threshold']}) must be <= "
            f"record_anchor_threshold ({result['record_anchor_threshold']})"
        )

    positive_keys = [
        "boundary_dim", "pair_dim", "start_top_k", "end_top_k",
        "ends_per_start", "starts_per_end", "candidate_budget",
        "max_gold_per_query", "end_block_size",
    ]
    for key in positive_keys:
        if result[key] <= 0:
            raise ValueError(f"boundary_head.{key} must be > 0, got {result[key]}")

    if result["boundary_refinement_layers"] < 0:
        raise ValueError(
            "boundary_head.boundary_refinement_layers must be >= 0, got "
            f"{result['boundary_refinement_layers']}"
        )
    if result["boundary_ffn_multiplier"] <= 0:
        raise ValueError(
            "boundary_head.boundary_ffn_multiplier must be > 0, got "
            f"{result['boundary_ffn_multiplier']}"
        )

    if result["training_candidate_budget"] < result["candidate_budget"]:
        raise ValueError(
            "boundary_head.training_candidate_budget "
            f"({result['training_candidate_budget']}) must be >= candidate_budget "
            f"({result['candidate_budget']})"
        )
    if result["training_candidate_budget"] < result["max_gold_per_query"]:
        raise ValueError(
            "boundary_head.training_candidate_budget "
            f"({result['training_candidate_budget']}) must be >= max_gold_per_query "
            f"({result['max_gold_per_query']})"
        )
    if not 0.0 <= result["dropout"] < 1.0:
        raise ValueError(f"boundary_head.dropout must be in [0, 1), got {result['dropout']}")
    return result


def migrate_config_dict(values: Mapping[str, Any]) -> dict:
    """Migrate a raw config dict to the current schema.

    Idempotent. Fills a missing ``architecture`` with ``"span"`` and moves a
    top-level legacy ``max_width`` into ``span_head`` for span configs.
    """
    migrated = dict(values)
    architecture = migrated.get("architecture") or "span"
    architecture = normalize_architecture(architecture)
    migrated["architecture"] = architecture

    if architecture == "span":
        span_head = dict(migrated.get("span_head") or {})
        if "max_width" in migrated and migrated["max_width"] is not None:
            span_head.setdefault("max_width", migrated["max_width"])
        migrated["span_head"] = validate_span_head(span_head)
    else:
        migrated["boundary_head"] = validate_boundary_head(
            dict(migrated.get("boundary_head") or {})
        )
    migrated.setdefault("config_version", ExtractorConfig.current_config_version)
    return migrated


def architecture_from_config(config: "ExtractorConfig") -> ArchitectureName:
    """Resolve the architecture from a config object (default ``"span"``)."""
    return normalize_architecture(getattr(config, "architecture", None) or "span")


# =============================================================================
# ExtractorConfig
# =============================================================================

class ExtractorConfig(PretrainedConfig):
    """Architecture-aware configuration for GLiNER2 extractors."""

    model_type = "extractor"
    current_config_version = 2

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        architecture: str = None,
        architecture_version: int = 1,
        token_pooling: str = "first",
        max_len: int = None,
        span_head: Mapping[str, Any] = None,
        boundary_head: Mapping[str, Any] = None,
        # Legacy span parameters
        max_width: int = None,
        counting_layer: str = None,
        config_version: int = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        # Missing architecture means an old (span) checkpoint.
        resolved_architecture = architecture or "span"
        self.architecture = normalize_architecture(resolved_architecture)
        self.architecture_version = architecture_version
        self.config_version = config_version or self.current_config_version
        self.model_name = model_name
        self.token_pooling = token_pooling
        self.max_len = max_len

        if self.architecture == "span":
            span_values = dict(span_head or {})
            if max_width is not None:
                span_values.setdefault("max_width", max_width)
            span_values.setdefault("max_width", SpanHeadSettings().max_width)
            self.span_head = validate_span_head(span_values)
            # Preserve attributes expected by the legacy span implementation.
            self.max_width = self.span_head["max_width"]
            self.counting_layer = counting_layer or "count_lstm"
        else:
            self.boundary_head = validate_boundary_head(boundary_head or {})
            if max_width is not None:
                warnings.warn(
                    "max_width is ignored by the boundary architecture",
                    UserWarning,
                    stacklevel=2,
                )

        self.validate()

    def validate(self) -> None:
        """Validate the configuration; raises on invalid settings."""
        normalize_architecture(self.architecture)
        if self.architecture == "span":
            validate_span_head(self.span_head)
        else:
            validate_boundary_head(self.boundary_head)
