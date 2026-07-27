"""End-to-end: boundary head candidates -> pack -> decode -> formatted spans."""

from __future__ import annotations

import torch

from gliner2.configuration import BoundaryHeadSettings
from gliner2.inference.candidate_decoder import decode_candidate_set, format_candidate
from gliner2.models.boundary.model import BoundaryHead
from gliner2.processing.targets import MentionTarget, TargetGraph, pad_target_graphs


def _overfit_head():
    torch.manual_seed(13)
    B, L, Q, H = 1, 16, 2, 32
    settings = BoundaryHeadSettings(
        boundary_dim=24, pair_dim=24, start_top_k=12, end_top_k=12,
        ends_per_start=6, starts_per_end=6, candidate_budget=48,
        training_candidate_budget=64, max_gold_per_query=16,
        end_block_size=16, dropout=0.0,
    )
    model = BoundaryHead(H, settings, query_dim=H)
    token_states = torch.randn(B, L, H)
    text_mask = torch.ones(B, L, dtype=torch.bool)
    query_states = torch.randn(B, Q, H)
    query_mask = torch.ones(B, Q, dtype=torch.bool)

    gold = {(0, 0): [(1, 3), (5, 12)], (0, 1): [(0, 2)]}
    mentions = [MentionTarget(qi, s, e) for (bi, qi), sp in gold.items() for (s, e) in sp]
    targets = pad_target_graphs([TargetGraph(mentions=tuple(mentions))], [Q], [L], 16)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        out = model(token_states, text_mask, query_states, query_mask, targets)
        out.total_loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(token_states, text_mask, query_states, query_mask)
    return out, gold


def test_pack_split_decode_recovers_gold_spans():
    out, gold = _overfit_head()
    packed = out.candidates.pack()
    sets = packed.split_by_sample()
    assert len(sets) == 1
    decoded = decode_candidate_set(
        sets[0], None, thresholds={}, overlap_policy="allow", default_threshold=0.5
    )
    got = {(c.query_id, c.start, c.end) for c in decoded}
    expected = {(qi, s, e) for (bi, qi), sp in gold.items() for (s, e) in sp}
    assert expected <= got


def test_decode_to_character_offsets_exact_slice():
    out, _ = _overfit_head()
    sets = out.candidates.pack().split_by_sample()
    decoded = decode_candidate_set(sets[0], None, thresholds={}, overlap_policy="disallow")

    # 16 single-character word tokens "a b c ..." with single-space separators.
    text = " ".join(["w"] * 16)
    start_map = [i * 2 for i in range(16)]
    end_map = [i * 2 + 1 for i in range(16)]
    for cand in decoded:
        result = format_candidate(
            cand, text, None, start_map, end_map, include_spans=True
        )
        assert text[result["char_start"]:result["char_end"]] == result["text"]
