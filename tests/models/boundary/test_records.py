"""Anchor-driven + anchorless record/event decoders: overfit and capacity."""

from __future__ import annotations

from typing import List, Sequence

import torch

from gliner2.models.boundary.records import (
    FieldAssignmentScorer,
    InstanceCandidateBatch,
    RecordSetDecoder,
)
from gliner2.training.matching import match_record_instances


def _train_set_decoder(decoder, field_query, field_states, gold, steps=600, lr=5e-3):
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()
    ce = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        out = decoder(field_query, field_states)
        obj = out.object_logits[0]
        ptr = out.field_pointer_logits[0]
        res = match_record_instances(obj, ptr, gold)
        row, col = res.row_ind, res.col_ind

        obj_target = torch.zeros_like(obj)
        obj_target[row] = 1.0
        loss = bce(obj, obj_target)
        for i, g in zip(row.tolist(), col.tolist()):
            for f, cidx in enumerate(gold[g]):
                if cidx is not None and cidx >= 0:
                    loss = loss + ce(ptr[i, f].unsqueeze(0), torch.tensor([cidx]))
        loss.backward()
        opt.step()
    return decoder


def _decode(decoder, field_query, field_states):
    out = decoder(field_query, field_states)
    obj = torch.sigmoid(out.object_logits[0]).detach()
    ptr = out.field_pointer_logits[0].detach()
    fields = ptr.shape[1]
    records = set()
    for i in range(obj.shape[0]):
        if float(obj[i]) > 0.5:
            records.add(tuple(int(ptr[i, f].argmax()) for f in range(fields)))
    return records


def test_record_set_decoder_overfits_multiple_records():
    torch.manual_seed(0)
    hidden, fields, cands, inst_q = 16, 2, 4, 6
    field_query = torch.randn(1, fields, hidden)
    field_states = torch.randn(1, fields, cands, hidden)
    gold = [[0, 1], [2, 3], [1, 0]]

    decoder = RecordSetDecoder(hidden, inst_q)
    _train_set_decoder(decoder, field_query, field_states, gold)
    assert _decode(decoder, field_query, field_states) == {(0, 1), (2, 3), (1, 0)}


def test_record_set_decoder_beyond_19_instances():
    torch.manual_seed(0)
    hidden, fields, cands, inst_q = 24, 1, 22, 25
    field_query = torch.randn(1, fields, hidden)
    field_states = torch.randn(1, fields, cands, hidden)
    gold = [[g] for g in range(22)]  # 22 > legacy count cap of 19

    decoder = RecordSetDecoder(hidden, inst_q)
    _train_set_decoder(decoder, field_query, field_states, gold, steps=800, lr=1e-2)
    records = _decode(decoder, field_query, field_states)
    assert records == {(g,) for g in range(22)}


def test_field_assignment_scorer_overfits_anchor_edges():
    torch.manual_seed(0)
    hidden, anchors, fields, cands = 16, 2, 1, 3
    inst = InstanceCandidateBatch(
        states=torch.randn(1, anchors, hidden),
        mask=torch.ones(1, anchors, dtype=torch.bool),
    )
    field_states = torch.randn(1, fields, cands, hidden)
    field_query = torch.randn(1, fields, hidden)
    # anchor 0 -> field cand 2 ; anchor 1 -> field cand 0
    gold = torch.tensor([[2], [0]])

    scorer = FieldAssignmentScorer(hidden)
    opt = torch.optim.Adam(scorer.parameters(), lr=1e-2)
    ce = torch.nn.CrossEntropyLoss()
    for _ in range(400):
        opt.zero_grad()
        logits = scorer(inst, field_states, field_query)[0]  # [N, F, C]
        loss = sum(
            ce(logits[n, f].unsqueeze(0), gold[n, f].unsqueeze(0))
            for n in range(anchors) for f in range(fields)
        )
        loss.backward()
        opt.step()

    logits = scorer(inst, field_states, field_query)[0]
    assert int(logits[0, 0].argmax()) == 2
    assert int(logits[1, 0].argmax()) == 0
