#!/usr/bin/env python3
"""Stream JSONL training data and recommend boundary-head capacities."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def nearest_rank(values: Sequence[int], percentile: float) -> int:
    """Exact nearest-rank percentile (including p99.9)."""
    if not values:
        return 0
    ordered = sorted(int(value) for value in values)
    rank = max(
        1,
        int(
            (Decimal(str(percentile)) * len(ordered) / Decimal(100)).to_integral_value(
                rounding=ROUND_CEILING
            )
        ),
    )
    return ordered[min(rank - 1, len(ordered) - 1)]


def build_capacity_report(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a deterministic report from collator-derived per-sample rows.

    Each row has ``gold_counts`` (one count per query), ``token_length``,
    optional ``sample_id``/``tasks``/``fields``, and optional
    ``pair_elements``. Keeping this core independent makes it directly usable
    by dataset-specific collator adapters.
    """
    counts: List[int] = []
    lengths: List[int] = []
    queries: List[int] = []
    pair_elements: List[int] = []
    offenders: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        row_counts = [int(value) for value in row.get("gold_counts", ())]
        counts.extend(row_counts)
        lengths.append(int(row.get("token_length", 0)))
        queries.append(len(row_counts))
        if "pair_elements" in row:
            pair_elements.append(int(row["pair_elements"]))
        tasks = row.get("tasks", ())
        fields = row.get("fields", ())
        for query, count in enumerate(row_counts):
            offenders.append(
                {
                    "sample_id": row.get("sample_id", index),
                    "task": tasks[query] if query < len(tasks) else "",
                    "field": fields[query] if query < len(fields) else str(query),
                    "count": count,
                }
            )
    quantiles = {
        name: nearest_rank(counts, percentile)
        for name, percentile in (
            ("p50", 50),
            ("p90", 90),
            ("p99", 99),
            ("p99.9", 99.9),
        )
    }
    maximum = max(counts, default=0)
    capacity = max(1, quantiles["p99.9"])
    candidate_budget = max(4 * capacity, 192)
    return {
        "gold_count_histogram": dict(sorted(Counter(counts).items())),
        "gold_count_quantiles": {**quantiles, "max": maximum},
        "top_offenders": sorted(
            offenders,
            key=lambda item: (
                -item["count"],
                str(item["sample_id"]),
                item["task"],
                item["field"],
            ),
        )[:20],
        "token_length_histogram": dict(sorted(Counter(lengths).items())),
        "queries_per_sample_histogram": dict(sorted(Counter(queries).items())),
        "recommended_settings": {
            "max_gold_per_query": capacity,
            "training_candidate_budget": candidate_budget,
            "candidate_budget": candidate_budget,
            "vectorized_pair_elements": nearest_rank(pair_elements, 99),
        },
    }


def _jsonl_rows(path: Path) -> Iterable[Dict[str, Any]]:
    """Read preflight rows from JSONL.

    Dataset adapters should emit the documented collator-derived row shape;
    this avoids guessing schema semantics in a launch-safety tool.
    """
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if "gold_counts" not in row:
                    raise ValueError(
                        f"{path}:{line_number}: missing gold_counts; run the "
                        "dataset's collator adapter first"
                    )
                yield row


def _collator_rows(
    path: Path,
    *,
    model_name: str,
    workers: int,
    batch_size: int,
    start_top_k: int,
) -> Iterable[Dict[str, Any]]:
    """Stream a raw training JSONL through the production target collator."""
    from torch.utils.data import DataLoader

    from gliner2.processor import SchemaTransformer
    from gliner2.training.trainer import ExtractorCollator, ExtractorDataset

    dataset = ExtractorDataset(
        data=path,
        shuffle=False,
        validate=False,
    )
    collator = ExtractorCollator(
        SchemaTransformer(model_name),
        is_training=True,
        architecture="boundary",
        # Dynamic target padding observes every gold span while allocating only
        # to the largest count in the current batch.
        max_gold_per_query=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collator,
    )
    sample_id = 0
    for batch in loader:
        targets = batch.targets
        if targets is None:
            continue
        batch_size_actual = len(batch)
        max_q = targets.mention_mask.shape[1]
        max_l = max(batch.text_word_counts, default=0)
        pair_elements = batch_size_actual * max_q * start_top_k * (max_l + 1)
        for index, layout in enumerate(batch.query_layouts):
            query_count = len(layout.queries)
            counts = (
                targets.mention_mask[index, :query_count].sum(-1).tolist()
            )
            yield {
                "sample_id": sample_id,
                "gold_counts": counts,
                "token_length": batch.text_word_counts[index],
                "tasks": [query.task_name for query in layout.queries],
                "fields": [query.role_name for query in layout.queries],
                "pair_elements": pair_elements,
            }
            sample_id += 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="JSONL of collator-derived rows")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--model-name",
        help="when set, treat input as raw training JSONL and run the real collator",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--start-top-k", type=int, default=16)
    args = parser.parse_args(argv)
    rows = (
        _collator_rows(
            args.input,
            model_name=args.model_name,
            workers=args.workers,
            batch_size=args.batch_size,
            start_top_k=args.start_top_k,
        )
        if args.model_name
        else _jsonl_rows(args.input)
    )
    report = build_capacity_report(rows)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
