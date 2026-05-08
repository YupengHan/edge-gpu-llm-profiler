#!/bin/bash
set -euo pipefail
WORKDIR=/workspace/rtx3070-trtllm-latency-lab
cd "$WORKDIR"

mkdir -p logs/stage1 results/stage1/benchmark_raw

export HF_MODEL=Qwen/Qwen2.5-1.5B-Instruct
export HF_HOME=/workspace/rtx3070-trtllm-latency-lab/models/.hf
export WORKSPACE_DIR=trtllm_workspace
export ENGINE_512_128=trtllm_workspace/Qwen/Qwen2.5-1.5B-Instruct/tp_1_pp_1

echo "=== nvidia-smi before main bench ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits

echo
echo "=========================================="
echo "= Step 10a — trtllm-bench latency 512/128 ="
echo "=========================================="
trtllm-bench \
  --model "$HF_MODEL" \
  --workspace "$WORKSPACE_DIR" \
  latency \
  --backend tensorrt \
  --dataset data/synthetic_512_128.jsonl \
  --engine_dir "$ENGINE_512_128" \
  --concurrency 1 \
  --num_requests 100 \
  --warmup 10 \
  --report_json results/stage1/benchmark_raw/trtllm_bench_512_128_latency.json \
  --iteration_log results/stage1/benchmark_raw/trtllm_bench_512_128_iteration.jsonl \
  2>&1 | tee logs/stage1/latency_512_128.log

echo
echo "============================================="
echo "= Step 10b — trtllm-bench throughput 512/128 ="
echo "============================================="
# Capture --help once so we can record any CLI drift on the throughput subcmd
trtllm-bench --model "$HF_MODEL" throughput --help \
  > logs/stage1/trtllm_bench_throughput_help.txt 2>&1 || true

# 1.0.0 throughput may or may not have --request_json; runbook lists it. If it
# fails the runner aborts via set -e and we fix it next pass.
trtllm-bench \
  --model "$HF_MODEL" \
  --workspace "$WORKSPACE_DIR" \
  throughput \
  --backend tensorrt \
  --dataset data/synthetic_512_128.jsonl \
  --engine_dir "$ENGINE_512_128" \
  --concurrency 1 \
  --num_requests 100 \
  --warmup 10 \
  --report_json results/stage1/benchmark_raw/trtllm_bench_512_128_throughput.json \
  --iteration_log results/stage1/benchmark_raw/trtllm_bench_512_128_throughput_iteration.jsonl \
  --request_json results/stage1/benchmark_raw/trtllm_bench_512_128_requests.jsonl \
  2>&1 | tee logs/stage1/throughput_512_128.log

echo
echo "=== nvidia-smi after main bench ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits

echo
echo "=== report files ==="
ls -la results/stage1/benchmark_raw/trtllm_bench_512_128_*
