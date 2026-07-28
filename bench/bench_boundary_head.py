#!/usr/bin/env python3
"""CPU/CUDA JSON micro-benchmark for the sparse boundary head."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gliner2.configuration import BoundaryHeadSettings
from gliner2.models.boundary.model import BoundaryHead
from gliner2.utils.sync_probe import count_cuda_syncs


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(fn, iters: int, device: torch.device) -> float:
    values = []
    for _ in range(iters):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        values.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(values)


def run(args) -> dict:
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("CPU fp16 benchmark is unsupported; use fp32 or bf16")
    settings = BoundaryHeadSettings(
        boundary_dim=32,
        pair_dim=32,
        start_top_k=min(16, args.length + 1),
        end_top_k=min(16, args.length + 1),
        ends_per_start=8,
        starts_per_end=8,
        candidate_budget=args.budget,
        training_candidate_budget=args.budget,
        max_gold_per_query=min(32, args.budget),
        end_block_size=64,
        dropout=0.0,
    )
    torch.manual_seed(17)
    head = BoundaryHead(64, settings, query_dim=64).to(device=device, dtype=dtype)
    head.eval()
    states = torch.randn(args.batch, args.length, 64, device=device, dtype=dtype)
    text_mask = torch.ones(args.batch, args.length, device=device, dtype=torch.bool)
    queries = torch.randn(args.batch, args.queries, 64, device=device, dtype=dtype)
    query_mask = torch.ones(args.batch, args.queries, device=device, dtype=torch.bool)

    def forward():
        with torch.no_grad():
            return head(states, text_mask, queries, query_mask)

    def forward_backward():
        head.zero_grad(set_to_none=True)
        output = head(states, text_mask, queries, query_mask)
        output.candidates.pair_logits[output.candidates.valid_mask].sum().backward()
        return output

    forward()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        memory_before = torch.cuda.memory_allocated(device)
    else:
        memory_before = 0
    with count_cuda_syncs() as syncs:
        forward_ms = _measure(forward, args.iters, device)
        forward_backward_ms = _measure(forward_backward, args.iters, device)
    memory_delta = (
        torch.cuda.max_memory_allocated(device) - memory_before
        if device.type == "cuda" else 0
    )
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as profiler:
        forward_backward()
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "unknown"
    return {
        "git_sha": sha,
        "device": str(device),
        "dtype": args.dtype,
        "batch": args.batch,
        "length": args.length,
        "queries": args.queries,
        "budget": args.budget,
        "iters": args.iters,
        "forward_median_ms": forward_ms,
        "forward_backward_median_ms": forward_backward_ms,
        "peak_memory_delta_bytes": memory_delta,
        "kernel_count": len(profiler.key_averages()),
        "cuda_sync_count": syncs["n"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--dtype", choices=DTYPES, default="fp32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    row = run(args)
    output = args.output or (
        Path(__file__).parent / "results" / f"{row['git_sha']}-{args.device}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
