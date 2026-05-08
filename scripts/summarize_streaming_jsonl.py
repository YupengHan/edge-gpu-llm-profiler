#!/usr/bin/env python3
"""Summarize a streaming benchmark JSONL produced by benchmark_streaming_openai.py.

Reads JSONL on argv[1] (or stdin) and prints aggregate p50/p90/p95/mean for
TTFT, ITL, E2E, and output pieces/sec. Only non-warmup rows are aggregated.
"""

import json
import statistics
import sys


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    idx = int(round((len(values) - 1) * q))
    return values[idx]


def fmt(v):
    if v is None:
        return "n/a"
    return f"{v:.2f}"


def main():
    if len(sys.argv) > 1 and sys.argv[1] != "-":
        f = open(sys.argv[1], "r", encoding="utf-8")
    else:
        f = sys.stdin

    rows = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("warmup"):
            continue
        rows.append(obj)

    n = len(rows)
    print(f"# Streaming benchmark summary")
    print(f"measured_requests: {n}")
    if not rows:
        return

    def collect(key):
        return [r[key] for r in rows if r.get(key) is not None]

    ttft = collect("ttft_ms")
    itl_mean = collect("itl_ms_mean")
    e2e = collect("e2e_ms")
    pieces = collect("output_pieces")
    tps = collect("output_tps_by_piece")

    def stat_block(label, values, unit):
        if not values:
            print(f"{label}: no data")
            return
        print(f"{label} ({unit}):")
        print(f"  mean = {fmt(statistics.mean(values))}")
        print(f"  p50  = {fmt(percentile(values, 0.50))}")
        print(f"  p90  = {fmt(percentile(values, 0.90))}")
        print(f"  p95  = {fmt(percentile(values, 0.95))}")
        print(f"  min  = {fmt(min(values))}")
        print(f"  max  = {fmt(max(values))}")

    stat_block("TTFT", ttft, "ms")
    stat_block("ITL mean (per-request)", itl_mean, "ms")
    stat_block("E2E", e2e, "ms")
    stat_block("output_pieces", pieces, "pieces")
    stat_block("output_tps_by_piece", tps, "pieces/sec")

    errors = [r for r in rows if r.get("error")]
    if errors:
        print(f"errors: {len(errors)} / {n}")


if __name__ == "__main__":
    main()
