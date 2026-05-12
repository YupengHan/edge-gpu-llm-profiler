#!/bin/bash
# Step 14 — run the streaming bench while sampling nvidia-smi at 1 Hz.
# Host-side script (not run inside the docker container).  trtllm_serve_step11
# must already be up and reachable at http://localhost:8000.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CSV=logs/stage1/nvidia_smi_during_streaming_512_128.csv
JSONL=results/stage1/benchmark_raw/streaming_512_128.jsonl
LOG=logs/stage1/streaming_512_128_monitored.log

mkdir -p logs/stage1 results/stage1/benchmark_raw

QUERY="timestamp,name,pstate,temperature.gpu,power.draw,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,memory.used,memory.free"

echo "=== Step 14 — monitored streaming bench ==="
echo "CSV: $CSV"
echo "Bench JSONL: $JSONL"
echo

echo "[1/4] pre-bench baseline (5 s at 1 Hz)"
nvidia-smi --query-gpu="$QUERY" --format=csv -l 1 > "$CSV" &
SMI_PID=$!
sleep 5

echo "[2/4] running streaming bench (50 + 5 warmup)"
python3 scripts/benchmark_streaming_openai.py \
  --base-url http://localhost:8000/v1 \
  --model tp_1_pp_1 \
  --prompts-file data/streaming_prompts_512.jsonl \
  --max-tokens 128 --requests 50 --warmup 5 --temperature 0 \
  --workload-label 512_128 --backend-label trtllm-1.0.0-trt \
  --output "$JSONL" \
  > "$LOG" 2>&1
BENCH_EXIT=$?

echo "[3/4] post-bench tail (5 s at 1 Hz)"
sleep 5

echo "[4/4] stopping sampler"
kill -INT "$SMI_PID" 2>/dev/null || true
wait "$SMI_PID" 2>/dev/null || true

echo
echo "bench exit=$BENCH_EXIT"
echo "csv lines: $(wc -l < "$CSV")"
echo "jsonl rows: $(wc -l < "$JSONL")"
echo "jsonl errors: $(grep -c '"error"' "$JSONL" || echo 0)"
exit $BENCH_EXIT
