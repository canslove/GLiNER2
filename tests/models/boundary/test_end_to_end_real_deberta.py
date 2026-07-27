"""Full end-to-end lifecycle test of the boundary architecture on a REAL model.

Uses ``microsoft/deberta-v3-xsmall`` as the base encoder and exercises the
boundary model's genuinely-implemented surface across every task family, mapped
onto the boundary head's span-per-query model plus the shared classifier head:

    * entities            -> one query per entity type (multi-mention)
    * json structures     -> one query per field, multi-object via co-ordering
    * relations           -> head/tail queries, paired by rank into instances
    * classification       -> document-contextual choice embeddings + classifier
    * combination          -> all of the above trained jointly on one model

Lifecycle covered: train (real encoder + heads) -> eval metrics -> save_pretrained
-> AutoExtractor.from_pretrained -> inference + reconstruction, asserting exact
`text[start:end]` recovery and perfect classification after overfitting.

Marked ``slow``. Skips cleanly if the encoder/tokenizer are unavailable offline.

NOTE: the processor's ``collate_fn_train`` and the public ``extract_entities``
API are not yet wired for the boundary architecture, so this test drives the
model's native APIs directly (``forward``/loss/candidates, ``save_pretrained``,
``AutoExtractor.from_pretrained``, ``decode_candidates``). It builds token/query
states from the real encoder + tokenizer exactly as those wirings eventually
will.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pytest
import torch

MODEL_NAME = "microsoft/deberta-v3-xsmall"

# Flat span-query layout shared by every document (same schema). Gold surfaces
# are deliberately non-overlapping (no gold span is a substring of another) so
# the model can overfit to an exact F1 of 1.0 at a 0.5 threshold.
Q_COMPANY, Q_PERSON, Q_ITEM, Q_QTY, Q_HEAD, Q_TAIL = range(6)
SPAN_QUERY_LABELS = [
    "company",            # entity
    "person name",        # entity
    "order item",         # json field
    "order quantity",     # json field
    "acquiring company",  # relation head
    "acquired company",   # relation tail
]
Q = len(SPAN_QUERY_LABELS)

TOPIC_LABELS = ["technology", "sports"]

# Gold surfaces are kept mid-sentence (never adjacent to punctuation) so the
# sentencepiece word grouping stays clean.
DOCUMENTS = [
    {
        "text": "Alice from Apple acquired Beats and ordered 3 laptops and 5 monitors today",
        "spans": {
            Q_COMPANY: ["Apple"],
            Q_PERSON: ["Alice"],
            Q_ITEM: ["laptops", "monitors"],
            Q_QTY: ["3", "5"],
            Q_HEAD: ["Apple"],
            Q_TAIL: ["Beats"],
        },
        "topic": 0,  # technology
    },
    {
        "text": "Bob from Sony acquired Bungie and ordered 7 consoles and 2 sensors weekly",
        "spans": {
            Q_COMPANY: ["Sony"],
            Q_PERSON: ["Bob"],
            Q_ITEM: ["consoles", "sensors"],
            Q_QTY: ["7", "2"],
            Q_HEAD: ["Sony"],
            Q_TAIL: ["Bungie"],
        },
        "topic": 1,  # sports (arbitrary label for the overfit target)
    },
]


# =============================================================================
# Model + real-encoder helpers
# =============================================================================

def _load_model():
    from gliner2 import ExtractorConfig
    from gliner2.inference.engine import BoundaryExtractor

    cfg = ExtractorConfig(
        model_name=MODEL_NAME,
        architecture="boundary",
        boundary_head=dict(
            boundary_dim=96, pair_dim=96, start_top_k=32, end_top_k=32,
            ends_per_start=16, starts_per_end=16, candidate_budget=96,
            training_candidate_budget=128, max_gold_per_query=16,
            end_block_size=32, dropout=0.0,
        ),
        token_pooling="first",
    )
    return BoundaryExtractor(cfg)


def _word_alignment(tokenizer, text: str):
    """Real word tokenization with whitespace-trimmed per-word char offsets."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    word_ids = enc.word_ids()
    n_words = max((w for w in word_ids if w is not None), default=-1) + 1
    w_start = [10**9] * n_words
    w_end = [0] * n_words
    first_subword = [None] * n_words
    for pos, (wid, (c0, c1)) in enumerate(zip(word_ids, offsets)):
        if wid is None or c1 <= c0:
            continue
        if first_subword[wid] is None:
            first_subword[wid] = pos
        w_start[wid] = min(w_start[wid], c0)
        w_end[wid] = max(w_end[wid], c1)
    for w in range(n_words):
        while w_start[w] < w_end[w] and text[w_start[w]].isspace():
            w_start[w] += 1
        while w_end[w] > w_start[w] and text[w_end[w] - 1].isspace():
            w_end[w] -= 1
    input_ids = torch.tensor([enc["input_ids"]], dtype=torch.long)
    attn = torch.tensor([enc["attention_mask"]], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "first_idx": torch.tensor(first_subword, dtype=torch.long),
        "w_start": w_start,
        "w_end": w_end,
        "n_words": n_words,
    }


def _surface_to_word_span(surface, text, w_start, w_end, used):
    """Locate a surface string and map it to a half-open word-token span."""
    search_from = 0
    while True:
        c0 = text.find(surface, search_from)
        if c0 < 0:
            raise ValueError(f"surface {surface!r} not found in {text!r}")
        c1 = c0 + len(surface)
        words = [w for w in range(len(w_start)) if w_start[w] >= c0 and w_end[w] <= c1]
        if words:
            span = (min(words), max(words) + 1)
            if span not in used:
                used.add(span)
                return span
        search_from = c0 + 1


def _word_states(model, align) -> torch.Tensor:
    """Real contextual first-subword word states, shape [W, H]."""
    sub = model.encoder(
        input_ids=align["input_ids"], attention_mask=align["attention_mask"]
    ).last_hidden_state[0]
    return sub.index_select(0, align["first_idx"])


def _query_states(model) -> torch.Tensor:
    """Mean-pooled encoder embeddings of the span-query label strings, [Q, H]."""
    tok = model.processor.tokenizer
    enc = tok(SPAN_QUERY_LABELS, padding=True, return_tensors="pt", add_special_tokens=True)
    h = model.encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
    return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _topic_embedding(model, text: str, label: str) -> torch.Tensor:
    """Document-contextual embedding of ``label`` encoded alongside ``text``, [H]."""
    tok = model.processor.tokenizer
    enc = tok(text, label, return_tensors="pt", add_special_tokens=True)
    seq_ids = enc.sequence_ids(0)
    h = model.encoder(
        input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]
    ).last_hidden_state[0]
    label_pos = [i for i, s in enumerate(seq_ids) if s == 1]
    return h[label_pos].mean(dim=0)


def _topic_logits(model, text: str) -> torch.Tensor:
    """Document-contextual classifier logits over TOPIC_LABELS, shape [K]."""
    embs = torch.stack([_topic_embedding(model, text, label) for label in TOPIC_LABELS])
    return model.classifier(embs).squeeze(-1)


# =============================================================================
# Gold assembly
# =============================================================================

def _build_alignments(model):
    tok = model.processor.tokenizer
    aligns, graphs, text_lengths = [], [], []
    from gliner2.processing.targets import MentionTarget, TargetGraph

    for doc in DOCUMENTS:
        align = _word_alignment(tok, doc["text"])
        mentions = []
        for qid, surfaces in sorted(doc["spans"].items()):
            used: set = set()  # per-query: distinct occurrences within a query
            for surface in surfaces:
                s, e = _surface_to_word_span(
                    surface, doc["text"], align["w_start"], align["w_end"], used
                )
                mentions.append(MentionTarget(qid, s, e))
        aligns.append(align)
        graphs.append(TargetGraph(mentions=tuple(mentions)))
        text_lengths.append(align["n_words"])
    return aligns, graphs, text_lengths


def _batch_states(model, aligns, max_l: int):
    b = len(aligns)
    h = model.hidden_size
    text_states = torch.zeros(b, max_l, h)
    text_mask = torch.zeros(b, max_l, dtype=torch.bool)
    for i, align in enumerate(aligns):
        ws = _word_states(model, align)
        n = ws.shape[0]
        text_states[i, :n] = ws
        text_mask[i, :n] = True
    q = _query_states(model).unsqueeze(0).expand(b, -1, -1).contiguous()
    query_mask = torch.ones(b, Q, dtype=torch.bool)
    return text_states, text_mask, q, query_mask


# =============================================================================
# Reconstruction helpers (decode -> surfaces -> objects)
# =============================================================================

def _resolve_overlaps(preds):
    """Greedy flat-span policy: per (sample, query) keep the highest-scoring
    span and drop any that overlap an already-kept one. ``decode_candidates``
    returns spans in descending score order, so first-come = highest score.
    """
    resolved = []
    for per_query in preds:
        rq = []
        for spans in per_query:
            kept: List[Tuple[int, int]] = []
            for (s, e) in spans:
                if all(e <= ks or s >= ke for (ks, ke) in kept):
                    kept.append((s, e))
            rq.append(kept)
        resolved.append(rq)
    return resolved


def _reconstruct(preds, aligns) -> List[Dict]:
    """Turn per-query decoded spans into per-document structured predictions."""
    from gliner2.inference.candidate_decoder import token_boundaries_to_character_offsets

    def surfaces(bi, qid):
        ordered = sorted(preds[bi][qid], key=lambda se: (se[0], se[1]))
        result = []
        for (s, e) in ordered:
            c0, c1 = token_boundaries_to_character_offsets(
                s, e, aligns[bi]["w_start"], aligns[bi]["w_end"]
            )
            result.append(DOCUMENTS[bi]["text"][c0:c1])
        return result

    docs_out = []
    for bi in range(len(aligns)):
        entities = {
            "company": surfaces(bi, Q_COMPANY),
            "person": surfaces(bi, Q_PERSON),
        }
        items = surfaces(bi, Q_ITEM)
        qtys = surfaces(bi, Q_QTY)
        orders = [
            {"item": it, "quantity": qt}
            for it, qt in zip(items, qtys)
        ]
        heads = surfaces(bi, Q_HEAD)
        tails = surfaces(bi, Q_TAIL)
        relations = [
            {"head": hd, "tail": tl} for hd, tl in zip(heads, tails)
        ]
        docs_out.append({"entities": entities, "orders": orders, "relations": relations})
    return docs_out


def _gold_reconstruction() -> List[Dict]:
    out = []
    for doc in DOCUMENTS:
        items = doc["spans"][Q_ITEM]
        qtys = doc["spans"][Q_QTY]
        out.append(
            {
                "entities": {
                    "company": sorted(doc["spans"][Q_COMPANY]),
                    "person": sorted(doc["spans"][Q_PERSON]),
                },
                "orders": [{"item": it, "quantity": qt} for it, qt in zip(items, qtys)],
                "relations": [
                    {"head": hd, "tail": tl}
                    for hd, tl in zip(doc["spans"][Q_HEAD], doc["spans"][Q_TAIL])
                ],
            }
        )
    return out


def _normalize_for_compare(doc_pred: Dict) -> Dict:
    return {
        "entities": {k: sorted(v) for k, v in doc_pred["entities"].items()},
        "orders": sorted([tuple(sorted(o.items())) for o in doc_pred["orders"]]),
        "relations": sorted([tuple(sorted(r.items())) for r in doc_pred["relations"]]),
    }


# =============================================================================
# The end-to-end test
# =============================================================================

@pytest.mark.slow
def test_boundary_full_lifecycle_real_deberta(tmp_path):
    torch.manual_seed(0)
    try:
        model = _load_model()
    except Exception as exc:  # offline / missing tokenizer deps
        pytest.skip(f"could not load {MODEL_NAME}: {exc}")

    from gliner2 import AutoExtractor
    from gliner2.processing.targets import pad_target_graphs
    from gliner2.models.boundary.model import decode_candidates
    from gliner2.training.metrics import (
        candidate_oracle_recall,
        exact_span_counts,
        f1_from_counts,
        gold_from_target_graphs,
    )

    aligns, graphs, text_lengths = _build_alignments(model)
    max_l = max(text_lengths)
    targets = pad_target_graphs(graphs, [Q] * len(graphs), text_lengths, max_gold_per_query=16)
    topic_gold = torch.tensor([doc["topic"] for doc in DOCUMENTS], dtype=torch.long)

    ce = torch.nn.CrossEntropyLoss()

    # Upweight pair separation and mine more hard negatives so the semantically
    # overlapping company-style queries (company / acquirer / acquired) resolve
    # to disjoint spans at a 0.5 threshold.
    model.boundary_head.loss_weights["pair"] = 4.0
    model.boundary_head.hard_negatives_per_positive = 16
    model.boundary_head.minimum_hard_negatives = 24

    # ---- Phase 1: span extraction (real encoder + boundary head) --------------
    span_opt = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": 2e-5},
            {"params": model.boundary_head.parameters(), "lr": 5e-3},
        ],
        weight_decay=0.0,
    )
    model.train()
    initial_loss = None
    for _ in range(600):
        span_opt.zero_grad(set_to_none=True)
        text_states, text_mask, query_states, query_mask = _batch_states(model, aligns, max_l)
        out = model.boundary_head(text_states, text_mask, query_states, query_mask, targets)
        span_loss = out.total_loss
        assert torch.isfinite(span_loss)
        if initial_loss is None:
            initial_loss = float(span_loss.detach())
        span_loss.backward()
        span_opt.step()
    final_loss = float(span_loss.detach())
    assert final_loss < initial_loss * 0.1, (initial_loss, final_loss)

    # ---- Phase 2: classification (encoder frozen, cached doc-contextual embs) --
    model.eval()
    with torch.no_grad():
        topic_embs = torch.stack([
            torch.stack([_topic_embedding(model, doc["text"], label) for label in TOPIC_LABELS])
            for doc in DOCUMENTS
        ])  # [B, K, H]
    cls_opt = torch.optim.AdamW(model.classifier.parameters(), lr=5e-3, weight_decay=0.0)
    for _ in range(200):
        cls_opt.zero_grad(set_to_none=True)
        logits = model.classifier(topic_embs.reshape(-1, model.hidden_size)).reshape(len(DOCUMENTS), len(TOPIC_LABELS))
        cls_loss = ce(logits, topic_gold)
        cls_loss.backward()
        cls_opt.step()
    assert float(cls_loss.detach()) < 1e-2

    # ---- Eval (no gold injection) --------------------------------------------
    model.eval()
    with torch.no_grad():
        text_states, text_mask, query_states, query_mask = _batch_states(model, aligns, max_l)
        out = model.boundary_head(text_states, text_mask, query_states, query_mask)
        oracle = candidate_oracle_recall(out.candidates, targets)
        preds = _resolve_overlaps(decode_candidates(out.candidates, threshold=0.5))
        gold = gold_from_target_graphs(graphs, Q)
        tp, fp, fn = exact_span_counts(preds, gold)
        precision, recall, f1 = f1_from_counts(tp, fp, fn)

        cls_pred = torch.stack([_topic_logits(model, doc["text"]) for doc in DOCUMENTS]).argmax(dim=-1)
        cls_acc = float((cls_pred == topic_gold).float().mean())

    if f1 != 1.0:  # diagnostic dump on failure
        for bi in range(len(aligns)):
            for qi in range(Q):
                print(
                    f"doc{bi} q{qi}({SPAN_QUERY_LABELS[qi]}): "
                    f"pred={sorted(preds[bi][qi])} gold={sorted(gold[bi][qi])}"
                )

    assert oracle == 1.0, oracle
    assert f1 == 1.0, (precision, recall, f1)
    assert cls_acc == 1.0, cls_acc

    pre_save_recon = [_normalize_for_compare(d) for d in _reconstruct(preds, aligns)]
    gold_recon = [_normalize_for_compare(d) for d in _gold_reconstruction()]
    assert pre_save_recon == gold_recon, (pre_save_recon, gold_recon)

    # ---- Save -> Load (public architecture-aware loader) ----------------------
    save_dir = tmp_path / "boundary_ckpt"
    model.save_pretrained(str(save_dir))
    reloaded = AutoExtractor.from_pretrained(str(save_dir))
    assert reloaded.architecture == "boundary"
    assert type(reloaded).__name__ == "BoundaryExtractor"
    reloaded.eval()

    # ---- Inference on the reloaded model -------------------------------------
    with torch.no_grad():
        text_states, text_mask, query_states, query_mask = _batch_states(reloaded, aligns, max_l)
        out2 = reloaded.boundary_head(text_states, text_mask, query_states, query_mask)
        preds2 = _resolve_overlaps(decode_candidates(out2.candidates, threshold=0.5))
        cls_pred2 = torch.stack([_topic_logits(reloaded, doc["text"]) for doc in DOCUMENTS]).argmax(dim=-1)

    post_load_recon = [_normalize_for_compare(d) for d in _reconstruct(preds2, aligns)]
    assert post_load_recon == gold_recon, (post_load_recon, gold_recon)
    assert torch.equal(cls_pred2, topic_gold)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        test_boundary_full_lifecycle_real_deberta(Path(d))
    print("OK")
