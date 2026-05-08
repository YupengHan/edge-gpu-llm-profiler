#!/bin/bash
set -euo pipefail
WORKDIR=/workspace/rtx3070-trtllm-latency-lab
cd "$WORKDIR"

mkdir -p logs/stage1 results/stage1/benchmark_raw

export HF_MODEL=Qwen/Qwen2.5-1.5B-Instruct
export HF_HOME=/workspace/rtx3070-trtllm-latency-lab/models/.hf
export WORKSPACE_DIR=trtllm_workspace
export ENGINE_128_128=trtllm_workspace/Qwen/Qwen2.5-1.5B-Instruct/tp_1_pp_1

echo "=== nvidia-smi before latency smoke ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits

echo "=== trtllm-bench latency 128/128 smoke ==="
trtllm-bench \
  --model "$HF_MODEL" \
  --workspace "$WORKSPACE_DIR" \
  latency \
  --backend tensorrt \
  --dataset data/synthetic_128_128.jsonl \
  --engine_dir "$ENGINE_128_128" \
  --concurrency 1 \
  --num_requests 20 \
  --warmup 5 \
  --report_json results/stage1/benchmark_raw/trtllm_bench_128_128_latency.json \
  --iteration_log results/stage1/benchmark_raw/trtllm_bench_128_128_iteration.jsonl \
  2>&1 | tee logs/stage1/latency_128_128.log

echo
echo "=== nvidia-smi after latency smoke ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits

echo
echo "=== report files ==="
ls -la results/stage1/benchmark_raw/trtllm_bench_128_128_*
