"""Boundary architecture model and trainable head bundle.

``BoundaryHead`` is the architecture-neutral trainable core: it turns token
states + query states into boundary marginals, sparse candidate proposals,
reranked pair logits, and (given targets) the weighted component losses. It is
deliberately decoupled from the encoder/tokenizer so it can be unit-overfit on
synthetic states (see ``tests/models/boundary/test_overfit_head.py``).

``BoundaryExtractorModel`` wraps a shared transformer encoder + classification
head around ``BoundaryHead`` and provides architecture-stamped serialization.
Half-open ``[start, end)`` coordinates throughout; there is no width axis.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from gliner2.configuration import BoundaryHeadSettings, ExtractorConfig
from gliner2.layers import create_mlp
from gliner2.models.base import BaseExtractorModel, EncodedBatch
from gliner2.models.boundary.encoding import BoundaryEncoder
from gliner2.models.boundary.heads import BoundaryMarginals, BoundaryQueryHead
from gliner2.models.boundary.losses import (
    balanced_multilabel_bce,
    build_candidate_labels,
    candidate_pair_loss,
    inside_consistency_loss,
    select_hard_negative_candidates,
)
from gliner2.models.boundary.proposal import (
    BoundaryProposals,
    ProposalSettings,
    SparseBoundaryProposer,
)
from gliner2.models.boundary.scoring import SparseBoundaryPairScorer, gather_boundary_states
from gliner2.models.base import QueryLayout, QuerySpec
from gliner2.models.outputs import CandidateTensorBatch, ExtractorOutput
from gliner2.processing.targets import (
    MentionTarget,
    PaddedTargetBatch,
    TargetGraph,
    pad_target_graphs,
)


DEFAULT_LOSS_WEIGHTS = {"start": 1.0, "end": 1.0, "pair": 1.0, "inside": 0.5}


def proposal_settings_from_head(settings: BoundaryHeadSettings) -> ProposalSettings:
    """Map validated ``BoundaryHeadSettings`` to runtime ``ProposalSettings``."""
    return ProposalSettings(
        start_top_k=settings.start_top_k,
        end_top_k=settings.end_top_k,
        ends_per_start=settings.ends_per_start,
        starts_per_end=settings.starts_per_end,
        candidate_budget=settings.candidate_budget,
        training_candidate_budget=settings.training_candidate_budget,
        max_gold_per_query=settings.max_gold_per_query,
        end_block_size=settings.end_block_size,
        bidirectional=settings.bidirectional_proposals,
        export_mode=settings.export_mode,
    )


class BoundaryHead(nn.Module):
    """Composable boundary head: encoding, marginals, proposal, scoring, losses."""

    def __init__(
        self,
        hidden_size: int,
        settings: BoundaryHeadSettings,
        query_dim: Optional[int] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        hard_negatives_per_positive: int = 5,
        minimum_hard_negatives: int = 8,
        build_candidate_states: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.settings = settings
        self.query_dim = query_dim if query_dim is not None else hidden_size
        self.loss_weights = dict(loss_weights or DEFAULT_LOSS_WEIGHTS)
        self.hard_negatives_per_positive = hard_negatives_per_positive
        self.minimum_hard_negatives = minimum_hard_negatives

        d = settings.boundary_dim
        # Candidate contextual states for the record head (endpoint-derived);
        # only built when records are enabled so legacy state dicts are unchanged.
        self.candidate_encoder = (
            nn.Linear(2 * d, hidden_size) if build_candidate_states else None
        )
        self.boundary_encoder = BoundaryEncoder(
            hidden_size,
            d,
            settings.dropout,
            settings.boundary_refinement_layers,
            settings.boundary_ffn_multiplier,
        )
        self.boundary_query_head = BoundaryQueryHead(
            hidden_size, d, self.query_dim, settings.dropout
        )
        self.boundary_proposer = SparseBoundaryProposer(
            d, self.query_dim, proposal_settings_from_head(settings)
        )
        self.pair_scorer = SparseBoundaryPairScorer(
            d, self.query_dim, settings.pair_dim,
            use_inside_evidence=settings.use_inside_evidence,
            dropout=settings.dropout,
        )
        self.use_inside_evidence = settings.use_inside_evidence

    def forward(
        self,
        token_states: torch.Tensor,        # [B, L, H]
        text_mask: torch.BoolTensor,       # [B, L]
        query_states: torch.Tensor,        # [B, Q, Hq]
        query_mask: torch.BoolTensor,      # [B, Q]
        targets: Optional[PaddedTargetBatch] = None,
        *,
        return_candidates: bool = True,
    ) -> ExtractorOutput:
        b, l, _ = token_states.shape
        text_lengths = text_mask.sum(dim=1).long()

        encoding = self.boundary_encoder(token_states, text_mask)
        marginals = self.boundary_query_head(
            encoding.states, encoding.mask,
            token_states, text_mask,
            query_states, query_mask,
        )

        gold_pairs = None
        gold_mask = None
        if self.training and targets is not None:
            gold_pairs = targets.mention_pairs
            gold_mask = targets.mention_mask

        proposals = self.boundary_proposer(
            encoding.states, encoding.mask,
            query_states, query_mask,
            marginals.start_logits, marginals.end_logits,
            gold_pairs=gold_pairs, gold_mask=gold_mask,
        )

        inside_prefix = marginals.inside_prefix if self.use_inside_evidence else None
        pair_logits = self.pair_scorer(
            encoding.states, query_states, proposals,
            marginals.start_logits, marginals.end_logits,
            inside_prefix, text_lengths,
        )

        candidates = None
        if return_candidates:
            candidate_states = None
            if self.candidate_encoder is not None:
                starts = proposals.indices[..., 0]
                ends = proposals.indices[..., 1]
                g_start = gather_boundary_states(encoding.states, starts)  # [B,Q,C,d]
                g_end = gather_boundary_states(encoding.states, ends)
                candidate_states = self.candidate_encoder(
                    torch.cat([g_start, g_end], dim=-1)
                )
                candidate_states = candidate_states.masked_fill(
                    ~proposals.valid_mask.unsqueeze(-1), 0.0
                )
            candidates = CandidateTensorBatch(
                indices=proposals.indices,
                proposal_logits=proposals.logits,
                pair_logits=pair_logits,
                valid_mask=proposals.valid_mask,
                query_mask=query_mask,
                candidate_states=candidate_states,
            )

        losses: Optional[Dict[str, torch.Tensor]] = None
        total_loss = None
        if targets is not None:
            losses = self._compute_losses(marginals, proposals, pair_logits, targets, query_mask, encoding.mask, text_mask)
            total_loss = losses["total_loss"]

        return ExtractorOutput(
            loss=total_loss,
            total_loss=total_loss,
            losses=losses,
            candidates=candidates,
            start_logits=marginals.start_logits,
            end_logits=marginals.end_logits,
            inside_logits=marginals.inside_logits,
            batch_size=b,
        )

    def _compute_losses(
        self,
        marginals: BoundaryMarginals,
        proposals: BoundaryProposals,
        pair_logits: torch.Tensor,
        targets: PaddedTargetBatch,
        query_mask: torch.BoolTensor,
        boundary_mask: torch.BoolTensor,
        text_mask: torch.BoolTensor,
    ) -> Dict[str, torch.Tensor]:
        boundary_keep = boundary_mask.unsqueeze(1) & query_mask.unsqueeze(-1)  # [B,Q,L+1]

        start_loss = balanced_multilabel_bce(
            marginals.start_logits, targets.start_targets, boundary_keep
        )
        end_loss = balanced_multilabel_bce(
            marginals.end_logits, targets.end_targets, boundary_keep
        )

        labels = build_candidate_labels(
            proposals.indices, proposals.valid_mask,
            targets.mention_pairs, targets.mention_mask,
        )
        hard = select_hard_negative_candidates(
            pair_logits.detach(), labels, proposals.valid_mask,
            negatives_per_positive=self.hard_negatives_per_positive,
            minimum_negatives=self.minimum_hard_negatives,
        )
        pair_loss = candidate_pair_loss(
            pair_logits, labels, proposals.valid_mask, hard_negative_mask=hard
        )

        if targets.inside_targets is not None:
            inside_loss = inside_consistency_loss(
                marginals.inside_logits, targets.inside_targets, text_mask, query_mask
            )
        else:
            inside_loss = torch.zeros((), device=pair_logits.device, dtype=pair_logits.dtype)

        w = self.loss_weights
        total = (
            w.get("start", 1.0) * start_loss
            + w.get("end", 1.0) * end_loss
            + w.get("pair", 1.0) * pair_loss
            + w.get("inside", 0.5) * inside_loss
        )
        return {
            "total_loss": total,
            "start_loss": start_loss,
            "end_loss": end_loss,
            "pair_loss": pair_loss,
            "inside_loss": inside_loss,
        }


def decode_candidates(
    candidates: CandidateTensorBatch,
    *,
    threshold: float = 0.5,
) -> List[List[List[Tuple[int, int]]]]:
    """Threshold pair logits into per-(sample, query) half-open spans.

    Returns ``preds[b][q]`` = list of ``(start, end)`` with sigmoid(pair) >=
    threshold, ordered by descending score with deterministic tie-break.
    """
    b, q, c, _ = candidates.indices.shape
    probs = torch.sigmoid(candidates.pair_logits)
    out: List[List[List[Tuple[int, int]]]] = []
    for bi in range(b):
        per_query: List[List[Tuple[int, int]]] = []
        for qi in range(q):
            spans: List[Tuple[float, int, int]] = []
            if bool(candidates.query_mask[bi, qi]):
                for ci in range(c):
                    if not bool(candidates.valid_mask[bi, qi, ci]):
                        continue
                    p = float(probs[bi, qi, ci])
                    if p >= threshold:
                        s = int(candidates.indices[bi, qi, ci, 0])
                        e = int(candidates.indices[bi, qi, ci, 1])
                        spans.append((p, s, e))
            spans.sort(key=lambda t: (-t[0], t[1], t[2]))
            per_query.append([(s, e) for _, s, e in spans])
        out.append(per_query)
    return out


def _extractive_field_names(schema_tokens: List[str]) -> List[str]:
    """Child field/label names in a schema group (the token after each marker)."""
    names: List[str] = []
    for j in range(len(schema_tokens) - 1):
        if schema_tokens[j] in ("[E]", "[C]", "[R]"):
            names.append(schema_tokens[j + 1])
    return names


def _iter_inclusive_spans(value: Any):
    """Yield inclusive ``(start, end)`` token spans from a structure-label field.

    Field values are lists of ``(start, end)`` tuples (all surface occurrences),
    possibly nested; ``(-1, -1)`` and ``None`` mark "not found" and are skipped.
    """
    if value is None:
        return
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(x, int) for x in value)
    ):
        if value != (-1, -1):
            yield value
        return
    if isinstance(value, (list, tuple)):
        for sub in value:
            yield from _iter_inclusive_spans(sub)


def _schema_group_name(schema_tokens: List[str]) -> str:
    """Recover the human-readable task/schema name from its prompt token."""
    if len(schema_tokens) > 2:
        return schema_tokens[2].split(" [DESCRIPTION] ")[0]
    return schema_tokens[0] if schema_tokens else ""


class BoundaryExtractorModel(BaseExtractorModel):
    """Boundary architecture: shared encoder + classification head + boundary head."""

    config_class = ExtractorConfig
    architecture = "boundary"

    def task_module_names(self) -> Tuple[str, ...]:
        base = ("classifier", "boundary_head")
        if getattr(self, "enable_records", False):
            return base + ("record_decoder",)
        return base

    def __init__(self, config: ExtractorConfig, encoder_config=None, tokenizer=None):
        super().__init__(config)
        if config.architecture != "boundary":
            raise ValueError(
                f"BoundaryExtractorModel requires architecture='boundary', "
                f"got {config.architecture!r}"
            )
        self.config = config

        from gliner2.processor import SchemaTransformer
        if tokenizer is not None:
            self.processor = SchemaTransformer(tokenizer=tokenizer, token_pooling=config.token_pooling)
        else:
            self.processor = SchemaTransformer(config.model_name, token_pooling=config.token_pooling)

        self.encoder = self._load_encoder(config.model_name, encoder_config)
        self.encoder.resize_token_embeddings(len(self.processor.tokenizer))
        self.hidden_size = self.encoder.config.hidden_size

        self.classifier = create_mlp(
            input_dim=self.hidden_size,
            intermediate_dims=[self.hidden_size * 2],
            output_dim=1,
            dropout=0.0,
            activation="relu",
            add_layer_norm=False,
        )

        settings = BoundaryHeadSettings(**config.boundary_head)
        self.boundary_settings = settings
        self.enable_records = settings.enable_records
        self.boundary_head = BoundaryHead(
            self.hidden_size, settings, query_dim=self.hidden_size,
            build_candidate_states=settings.enable_records,
        )
        if self.enable_records:
            from gliner2.models.boundary.records import RecordHead
            self.record_decoder = RecordHead(
                self.hidden_size,
                settings.record_dim,
                settings.record_instance_queries,
            )

        self._lora_layers = {}
        self._adapter_config = None

    # =========================================================================
    # Encoding
    # =========================================================================

    def _encode_core(self, batch) -> Dict[str, Any]:
        """Encode a ``PreprocessedBatch`` into padded states + query enumeration.

        Every extractive schema child (``[E]``/``[C]``/``[R]`` marker) becomes one
        boundary query; its query embedding is the marker's contextual embedding
        (``embs[1:]``), aligned 1:1 with the gold field order in
        ``structure_labels``. Classification schemas are enumerated separately and
        scored by the shared classifier. No fixed cross-sample query layout is
        required, so training-time task shuffling is handled naturally.
        """
        device = next(self.parameters()).device
        batch = batch.to(device)
        outputs = self.encoder(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        token_embeddings = outputs.last_hidden_state
        all_token_embs, all_schema_embs = self.processor.extract_embeddings_from_batch(
            token_embeddings, batch.input_ids, batch
        )

        n = len(batch)
        h = self.hidden_size
        text_lengths = [t.shape[0] for t in all_token_embs]
        max_l = max(text_lengths) if text_lengths else 0

        text_states = torch.zeros(n, max_l, h, device=device, dtype=token_embeddings.dtype)
        text_mask = torch.zeros(n, max_l, dtype=torch.bool, device=device)
        for i, emb in enumerate(all_token_embs):
            if emb.numel() > 0:
                text_states[i, : text_lengths[i]] = emb
                text_mask[i, : text_lengths[i]] = True

        ext_specs: List[List[Dict[str, Any]]] = []
        ext_embs: List[List[torch.Tensor]] = []
        cls_specs: List[List[Dict[str, Any]]] = []
        word_offsets: List[int] = []

        for i in range(n):
            specs_i: List[Dict[str, Any]] = []
            embs_i: List[torch.Tensor] = []
            cls_i: List[Dict[str, Any]] = []
            text_len_i = len(batch.start_mappings[i]) if batch.start_mappings else text_lengths[i]
            word_offsets.append(max(text_lengths[i] - text_len_i, 0))

            for g in range(batch.schema_counts[i]):
                task_type = batch.task_types[i][g]
                schema_tokens = batch.schema_tokens_list[i][g]
                group_embs = all_schema_embs[i][g]
                if not group_embs:
                    continue
                field_names = _extractive_field_names(schema_tokens)
                name = _schema_group_name(schema_tokens)
                if task_type == "classifications":
                    if len(group_embs) > 1:
                        cls_i.append({
                            "group_index": g,
                            "task_name": name,
                            "schema_tokens": schema_tokens,
                            "group_embs": torch.stack(group_embs),
                        })
                    continue
                for fidx, fname in enumerate(field_names):
                    if 1 + fidx >= len(group_embs):
                        break
                    specs_i.append({
                        "group_index": g,
                        "field_index": fidx,
                        "task_type": task_type,
                        "task_name": name,
                        "field_name": fname,
                    })
                    embs_i.append(group_embs[1 + fidx])

            ext_specs.append(specs_i)
            ext_embs.append(embs_i)
            cls_specs.append(cls_i)

        max_q = max((len(e) for e in ext_embs), default=0)
        query_states = torch.zeros(n, max_q, h, device=device, dtype=token_embeddings.dtype)
        query_mask = torch.zeros(n, max_q, dtype=torch.bool, device=device)
        for i, embs_i in enumerate(ext_embs):
            for j, emb in enumerate(embs_i):
                query_states[i, j] = emb
                query_mask[i, j] = True

        return {
            "text_states": text_states,
            "text_mask": text_mask,
            "text_lengths": torch.tensor(text_lengths, dtype=torch.long, device=device),
            "query_states": query_states,
            "query_mask": query_mask,
            "ext_specs": ext_specs,
            "cls_specs": cls_specs,
            "word_offsets": word_offsets,
        }

    def encode(self, batch) -> EncodedBatch:
        """Encode a ``PreprocessedBatch`` into a dense ``EncodedBatch``."""
        core = self._encode_core(batch)
        layouts: List[QueryLayout] = []
        for specs in core["ext_specs"]:
            queries = tuple(
                QuerySpec(
                    query_id=j,
                    task_index=spec["group_index"],
                    task_type=spec["task_type"],
                    task_name=spec["task_name"],
                    role_index=spec["field_index"],
                    role_name=spec["field_name"],
                    field_path=(spec["task_name"], spec["field_name"]),
                    extractive=True,
                )
                for j, spec in enumerate(specs)
            )
            layouts.append(QueryLayout(queries=queries))
        return EncodedBatch(
            text_states=core["text_states"],
            text_mask=core["text_mask"],
            text_lengths=core["text_lengths"],
            query_states=core["query_states"],
            query_mask=core["query_mask"],
            query_layouts=tuple(layouts),
        )

    def _targets_from_structure(self, batch, core: Dict[str, Any]) -> Optional[PaddedTargetBatch]:
        """Build a :class:`PaddedTargetBatch` from ``structure_labels`` + queries."""
        structure_labels = getattr(batch, "structure_labels", None)
        if not structure_labels:
            return None
        graphs: List[TargetGraph] = []
        query_counts: List[int] = []
        text_lengths: List[int] = []
        for i in range(len(batch)):
            specs = core["ext_specs"][i]
            length = int(core["text_lengths"][i])
            mentions: List[MentionTarget] = []
            for qid, spec in enumerate(specs):
                structure = structure_labels[i][spec["group_index"]]
                if not structure or structure[0] == 0:
                    continue
                fidx = spec["field_index"]
                for inst in structure[1]:
                    if fidx >= len(inst):
                        continue
                    for (s, e_inc) in _iter_inclusive_spans(inst[fidx]):
                        if 0 <= s <= e_inc < length:
                            mentions.append(MentionTarget(qid, s, e_inc + 1))
            graphs.append(TargetGraph(mentions=tuple(mentions)))
            query_counts.append(len(specs))
            text_lengths.append(length)
        if not any(query_counts):
            return None
        return pad_target_graphs(
            graphs, query_counts, text_lengths,
            self.boundary_head.settings.max_gold_per_query,
        )

    def _classification_loss(self, batch, core: Dict[str, Any]) -> torch.Tensor:
        """Binary cross-entropy over classification choices (shared classifier)."""
        device = core["text_states"].device
        total = torch.zeros((), device=device)
        structure_labels = getattr(batch, "structure_labels", None)
        if not structure_labels:
            return total
        for i in range(len(batch)):
            for cls in core["cls_specs"][i]:
                labels_raw = structure_labels[i][cls["group_index"]]
                logits = self.classifier(cls["group_embs"][1:]).squeeze(-1)
                labels = torch.tensor(labels_raw, dtype=logits.dtype, device=device)
                if labels.shape != logits.shape:
                    continue
                total = total + F.binary_cross_entropy_with_logits(
                    logits, labels, reduction="sum"
                )
        return total

    # =========================================================================
    # Forward
    # =========================================================================

    def forward(
        self,
        batch,
        *,
        return_candidates: bool = True,
        return_individual_losses: bool = False,
    ) -> ExtractorOutput:
        core = self._encode_core(batch)
        targets = getattr(batch, "targets", None)
        if targets is not None:
            targets = targets.to(core["text_states"].device)
        elif self.training:
            targets = self._targets_from_structure(batch, core)

        if core["query_states"].shape[1] > 0:
            output = self.boundary_head(
                core["text_states"], core["text_mask"],
                core["query_states"], core["query_mask"],
                targets, return_candidates=return_candidates,
            )
        else:
            # Classification-only batch: no extractive queries to score.
            output = ExtractorOutput(
                candidates=None,
                total_loss=None,
                loss=None,
                losses={},
                batch_size=len(batch),
            )

        cls_loss = self._classification_loss(batch, core)

        record_loss = None
        if (
            self.enable_records
            and self.training
            and targets is not None
            and getattr(targets, "records", None) is not None
            and output.candidates is not None
            and output.candidates.candidate_states is not None
        ):
            record_loss = self._record_loss(batch, core, output.candidates, targets)

        needs_combine = (
            targets is not None or bool(cls_loss.detach()) or record_loss is not None
        )
        if needs_combine:
            span_total = output.total_loss if output.total_loss is not None else torch.zeros((), device=cls_loss.device)
            combined = span_total + cls_loss
            if record_loss is not None:
                combined = combined + record_loss["total"]
            output.total_loss = combined
            output.loss = combined
            if output.losses is not None:
                output.losses["classification_loss"] = cls_loss
                if record_loss is not None:
                    output.losses["record_object_loss"] = record_loss["object"]
                    output.losses["record_field_loss"] = record_loss["field"]
        return output

    def _record_loss(self, batch, core, candidates, targets) -> Dict[str, torch.Tensor]:
        """Aggregate record object + field-assignment losses across the batch."""
        from gliner2.models.boundary.record_loss import compute_group_loss

        device = core["query_states"].device
        obj_total = torch.zeros((), device=device)
        field_total = torch.zeros((), device=device)
        n_groups = 0
        record_specs = getattr(batch, "record_specs", ())
        per_sample_records = targets.records  # List[List[RecordTarget]]
        weight = self.boundary_settings.record_loss_weight

        for i in range(len(batch)):
            if i >= len(record_specs):
                continue
            specs = record_specs[i]
            if not specs:
                continue
            sample_records = (
                per_sample_records[i] if i < len(per_sample_records) else []
            )
            query_states_i = core["query_states"][i]
            for task_index, spec in specs.items():
                group = self.record_decoder.forward_group(
                    spec, query_states_i, candidates, i
                )
                recs = [r for r in sample_records if r.task_index == task_index]
                losses = compute_group_loss(group, recs)
                obj_total = obj_total + losses["object_loss"]
                field_total = field_total + losses["field_loss"]
                n_groups += 1

        denom = max(n_groups, 1)
        obj = obj_total / denom
        field = field_total / denom
        return {"object": obj, "field": field, "total": weight * (obj + field)}

    def score_candidates(self, batch, *, return_auxiliary_logits: bool = False) -> CandidateTensorBatch:
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                output = self.forward(batch, return_candidates=True)
        finally:
            self.train(was_training)
        return output.candidates

    # =========================================================================
    # Serialization
    # =========================================================================

    def save_pretrained(self, save_directory: str, **kwargs):
        from safetensors.torch import save_file

        os.makedirs(save_directory, exist_ok=True)
        self.config.architecture = "boundary"
        self.config.architectures = [type(self).__name__]
        self.config.save_pretrained(save_directory)

        encoder_config_path = os.path.join(save_directory, "encoder_config")
        os.makedirs(encoder_config_path, exist_ok=True)
        self.encoder.config.save_pretrained(encoder_config_path)

        save_file(self.state_dict(), os.path.join(save_directory, "model.safetensors"))
        self.processor.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, repo_or_dir: str, **kwargs):
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download

        config = kwargs.pop("config", None)
        map_location = kwargs.pop("map_location", None)

        def download_or_local(repo, filename):
            if os.path.isdir(repo):
                return os.path.join(repo, filename)
            return hf_hub_download(repo, filename)

        if config is None:
            config = cls.config_class.from_pretrained(download_or_local(repo_or_dir, "config.json"))
        encoder_config = AutoConfig.from_pretrained(
            download_or_local(repo_or_dir, "encoder_config/config.json")
        )
        tokenizer = AutoTokenizer.from_pretrained(repo_or_dir)
        model = cls(config, encoder_config=encoder_config, tokenizer=tokenizer)

        try:
            state_dict = load_file(download_or_local(repo_or_dir, "model.safetensors"))
        except Exception:
            state_dict = torch.load(download_or_local(repo_or_dir, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict)

        model.config._name_or_path = repo_or_dir
        model.name_or_path = repo_or_dir
        if map_location is not None:
            model = model.to(map_location)
        return model


__all__ = [
    "BoundaryHead",
    "BoundaryExtractorModel",
    "decode_candidates",
    "proposal_settings_from_head",
]
