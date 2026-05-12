#!/usr/bin/env python3
"""Streaming OpenAI-compatible benchmark client for trtllm-serve.

Sends requests one at a time with stream=True and records:
- request submit timestamp
- first token arrival timestamp
- each subsequent token arrival timestamp
- request done timestamp

Outputs one JSON object per line to --output, suitable for downstream
analysis with summarize_streaming_jsonl.py.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import requests


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    idx = int(round((len(values) - 1) * q))
    return values[idx]


def run_one(base_url, model, prompt, max_tokens, temperature, timeout_s):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    t0 = time.perf_counter()
    token_times = []
    text_parts = []
    finish_reason = None
    usage = None
    error = None

    try:
        with requests.post(url, json=payload, stream=True, timeout=timeout_s) as r:
            r.raise_for_status()
            for raw_line in r.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {}) or {}
                    content = delta.get("content")
                    if content:
                        token_times.append(time.perf_counter())
                        text_parts.append(content)
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish_reason = fr
                u = obj.get("usage")
                if u:
                    usage = u
    except Exception as exc:
        error = repr(exc)

    t_done = time.perf_counter()

    ttft_ms = None
    if token_times:
        ttft_ms = (token_times[0] - t0) * 1000.0

    gaps_ms = [
        (token_times[i] - token_times[i - 1]) * 1000.0
        for i in range(1, len(token_times))
    ]

    e2e_ms = (t_done - t0) * 1000.0
    output_pieces = len(token_times)
    output_tps = output_pieces / (e2e_ms / 1000.0) if e2e_ms > 0 else None

    row = {
        "ttft_ms": ttft_ms,
        "itl_ms_mean": statistics.mean(gaps_ms) if gaps_ms else None,
        "itl_ms_p50": percentile(gaps_ms, 0.50),
        "itl_ms_p90": percentile(gaps_ms, 0.90),
        "itl_ms_p95": percentile(gaps_ms, 0.95),
        "itl_ms": [round(g, 4) for g in gaps_ms],
        "e2e_ms": e2e_ms,
        "output_pieces": output_pieces,
        "output_tps_by_piece": output_tps,
        "finish_reason": finish_reason,
        "usage": usage,
        "text_preview": "".join(text_parts)[:200],
    }
    if error:
        row["error"] = error
    return row


def load_prompts(prompt_arg, prompts_file):
    """Return a list of prompts. Either --prompt (single) or --prompts-file (rotate)."""
    if prompts_file:
        prompts = []
        with open(prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "prompt" not in obj:
                    raise SystemExit(f"prompts-file row missing 'prompt' key: {line[:120]}")
                prompts.append(obj["prompt"])
        if not prompts:
            raise SystemExit(f"--prompts-file {prompts_file} contained no rows")
        return prompts
    if prompt_arg is None:
        raise SystemExit("must pass either --prompt or --prompts-file")
    text = prompt_arg[1:] if prompt_arg.startswith("@") else prompt_arg
    if prompt_arg.startswith("@"):
        text = Path(text).read_text(encoding="utf-8")
    return [text]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt",
                     help="Single prompt string, or @path to read from file.")
    src.add_argument("--prompts-file",
                     help="JSONL file with one {\"prompt\": \"...\"} object per line. "
                          "Requests rotate through this list as request_index % N.")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--requests", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workload-label", default="")
    ap.add_argument("--backend-label", default="")
    args = ap.parse_args()

    prompts = load_prompts(args.prompt, args.prompts_file)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(args.requests + args.warmup):
            prompt_idx = i % len(prompts)
            row = run_one(
                args.base_url,
                args.model,
                prompts[prompt_idx],
                args.max_tokens,
                args.temperature,
                args.timeout,
            )
            row["request_index"] = i
            row["prompt_index"] = prompt_idx
            row["warmup"] = i < args.warmup
            row["model"] = args.model
            row["max_tokens"] = args.max_tokens
            row["workload"] = args.workload_label
            row["backend"] = args.backend_label
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps(
                {k: v for k, v in row.items() if k != "text_preview"},
                ensure_ascii=False,
            ))


if __name__ == "__main__":
    main()
