"""Architecture-neutral query metadata, encoded-batch container, and base model.

``QuerySpec``/``QueryLayout`` describe the extractive/classification queries of
a sample independent of architecture. ``EncodedBatch`` is the vectorized
encoder output both architectures can consume. ``BaseExtractorModel`` provides
shared encoder loading and architecture-stamping save; the span model keeps its
own encode path until parity is proven (per the blueprint).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

from gliner2.configuration import ExtractorConfig


# =============================================================================
# Query metadata
# =============================================================================

@dataclass(frozen=True)
class QuerySpec:
    """Metadata for one query (an extractive field/entity or a classification)."""
    query_id: int
    task_index: int
    task_type: str
    task_name: str
    role_index: int = 0
    role_name: str = ""
    field_path: Tuple[str, ...] = ()
    extractive: bool = True


@dataclass(frozen=True)
class QueryLayout:
    """Ordered queries for a single sample, with fast id lookup."""
    queries: Tuple[QuerySpec, ...]
    classification_query_ids: Tuple[int, ...] = ()
    extractive_query_ids: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        # Derive id groupings if not provided.
        if not self.classification_query_ids and not self.extractive_query_ids and self.queries:
            cls_ids = tuple(q.query_id for q in self.queries if not q.extractive)
            ext_ids = tuple(q.query_id for q in self.queries if q.extractive)
            object.__setattr__(self, "classification_query_ids", cls_ids)
            object.__setattr__(self, "extractive_query_ids", ext_ids)

    def __len__(self) -> int:
        return len(self.queries)

    def query(self, query_id: int) -> QuerySpec:
        for q in self.queries:
            if q.query_id == query_id:
                return q
        raise KeyError(f"no query with id {query_id}")

    def extractive_count(self) -> int:
        return len(self.extractive_query_ids)

    def classification_count(self) -> int:
        return len(self.classification_query_ids)


# =============================================================================
# Encoded batch
# =============================================================================

@dataclass
class EncodedBatch:
    """Vectorized encoder output shared by both architectures."""
    text_states: torch.Tensor            # [B, L, H]
    text_mask: torch.BoolTensor          # [B, L]
    text_lengths: torch.LongTensor       # [B]
    query_states: torch.Tensor           # [B, Q, H]
    query_mask: torch.BoolTensor         # [B, Q]
    query_layouts: Tuple[QueryLayout, ...]
    classification_states: Optional[torch.Tensor] = None

    def to(self, device) -> "EncodedBatch":
        return EncodedBatch(
            text_states=self.text_states.to(device),
            text_mask=self.text_mask.to(device),
            text_lengths=self.text_lengths.to(device),
            query_states=self.query_states.to(device),
            query_mask=self.query_mask.to(device),
            query_layouts=self.query_layouts,
            classification_states=(
                self.classification_states.to(device)
                if self.classification_states is not None else None
            ),
        )


# =============================================================================
# Base model
# =============================================================================

class BaseExtractorModel(PreTrainedModel):
    """Shared base for extractor architectures.

    Provides encoder construction and architecture-aware serialization. It does
    not impose an ``encode()`` contract on the span model; the boundary model
    uses ``encode()`` while the span model retains its legacy path.
    """
    config_class = ExtractorConfig

    @staticmethod
    def _load_encoder(model_name: str, encoder_config: Optional[PretrainedConfig] = None) -> nn.Module:
        if encoder_config is not None:
            return AutoModel.from_config(encoder_config, trust_remote_code=True)
        return AutoModel.from_pretrained(model_name, trust_remote_code=True)

    def task_module_names(self) -> Tuple[str, ...]:
        raise NotImplementedError

    def save_pretrained(self, *args, **kwargs):
        self.config.architecture = getattr(self, "architecture", self.config.architecture)
        self.config.architectures = [type(self).__name__]
        return super().save_pretrained(*args, **kwargs)
