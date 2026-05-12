#!/usr/bin/env python3
"""Summarize a streaming benchmark JSONL produced by benchmark_streaming_openai.py.

Reads JSONL on argv[1] (or stdin) and prints aggregate p50/p90/p95/mean for
TTFT, ITL, E2E, output pieces/sec. Aggregates non-warmup rows.

If rows carry `prompt_index`, the summary auto-splits into a COLD block
(rows whose prompt_index appears for the first time across the whole file)
and a WARM block (rows whose prompt_index has been seen earlier — these
typically hit TRT-LLM's prefix cache and skew TTFT downward).

If rows carry the `itl_ms` raw-gap array (added in step 13), a true global
ITL percentile (flattened across all gaps) is computed in addition to the
"mean of per-request percentiles" view.

Optional `--json <path>` writes a machine-readable summary alongside the
text report so step 18 / step 19 can consume the numbers without re-parsing.
"""

import argparse
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


def collect(rows, key):
    return [r[key] for r in rows if r.get(key) is not None]


def stats_block(values):
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def print_block(label, block, unit):
    if block is None:
        print(f"{label}: no data")
        return
    print(f"{label} ({unit}, n={block['n']}):")
    for k in ("mean", "p50", "p90", "p95", "p99", "min", "max"):
        print(f"  {k:>4} = {fmt(block[k])}")


def split_cold_warm(rows, prior_prompt_indices=None):
    """Return (cold_rows, warm_rows) by prompt_index first-occurrence.

    prior_prompt_indices: iterable of prompt_index values seen *before* `rows`
    started (typically the warmup rows of the same file). Any row whose
    prompt_index appears there is classified as warm even on its first
    occurrence inside `rows`, because the server's prefix cache was warmed
    by the warmup phase.

    Rows without prompt_index all go to 'cold' (we cannot tell).
    """
    seen = set(prior_prompt_indices or ())
    cold, warm = [], []
    for r in rows:
        pi = r.get("prompt_index")
        if pi is None:
            cold.append(r)
        elif pi in seen:
            warm.append(r)
        else:
            seen.add(pi)
            cold.append(r)
    return cold, warm


def summarize_group(label, rows):
    print(f"\n=== {label} (n={len(rows)}) ===")
    if not rows:
        return None
    ttft = collect(rows, "ttft_ms")
    itl_mean = collect(rows, "itl_ms_mean")
    e2e = collect(rows, "e2e_ms")
    tps = collect(rows, "output_tps_by_piece")
    pieces = collect(rows, "output_pieces")

    # Flattened ITL across all per-token gaps (if available)
    flat_itl = []
    for r in rows:
        gaps = r.get("itl_ms")
        if gaps:
            flat_itl.extend(gaps)

    blocks = {
        "ttft_ms": stats_block(ttft),
        "itl_ms_mean_per_request": stats_block(itl_mean),
        "itl_ms_flattened_per_token": stats_block(flat_itl) if flat_itl else None,
        "e2e_ms": stats_block(e2e),
        "output_tps_by_piece": stats_block(tps),
        "output_pieces": stats_block(pieces),
    }

    print_block("TTFT", blocks["ttft_ms"], "ms")
    print_block("ITL mean per-request", blocks["itl_ms_mean_per_request"], "ms")
    if blocks["itl_ms_flattened_per_token"] is not None:
        print_block("ITL flattened per-token", blocks["itl_ms_flattened_per_token"], "ms")
    else:
        print("ITL flattened per-token: not in data (rerun with updated bench client to capture raw gaps)")
    print_block("E2E", blocks["e2e_ms"], "ms")
    print_block("output_tps_by_piece", blocks["output_tps_by_piece"], "pieces/sec")
    print_block("output_pieces", blocks["output_pieces"], "pieces")

    errors = [r for r in rows if r.get("error")]
    if errors:
        print(f"errors: {len(errors)} / {len(rows)}")

    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default="-",
                    help="JSONL path or - for stdin")
    ap.add_argument("--json", dest="json_out",
                    help="Optional path to write machine-readable summary")
    ap.add_argument("--no-split", action="store_true",
                    help="Don't split cold/warm — summarize all non-warmup rows together")
    args = ap.parse_args()

    if args.input != "-":
        f = open(args.input, "r", encoding="utf-8")
    else:
        f = sys.stdin

    warmup_prompt_indices = []
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
            pi = obj.get("prompt_index")
            if pi is not None:
                warmup_prompt_indices.append(pi)
            continue
        rows.append(obj)

    print(f"# Streaming benchmark summary")
    print(f"source: {args.input}")
    print(f"measured_requests: {len(rows)}")

    if not rows:
        return

    summary = {
        "source": args.input,
        "measured_requests": len(rows),
    }

    if args.no_split:
        summary["all"] = summarize_group("ALL MEASURED", rows)
    else:
        cold, warm = split_cold_warm(rows, prior_prompt_indices=warmup_prompt_indices)
        summary["cold"] = summarize_group(
            "COLD — unique-prompt requests (prompt_index first occurrence)", cold)
        if warm:
            summary["warm"] = summarize_group(
                "WARM — cache-hit requests (prompt_index seen before)", warm)
        else:
            print("\n(no warm rows detected — every prompt_index was unique, or "
                  "rows lacked prompt_index)")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as jf:
            json.dump(summary, jf, indent=2)
        print(f"\nJSON summary written to {args.json_out}")


if __name__ == "__main__":
    main()
