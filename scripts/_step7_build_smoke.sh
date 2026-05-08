#!/bin/bash
set -euo pipefail
WORKDIR=/workspace/rtx3070-trtllm-latency-lab
cd "$WORKDIR"

mkdir -p logs/stage1 trtllm_workspace engines/stage1

export HF_MODEL=Qwen/Qwen2.5-1.5B-Instruct
export HF_HOME=/workspace/rtx3070-trtllm-latency-lab/models/.hf
export WORKSPACE_DIR=trtllm_workspace
mkdir -p "$HF_HOME" "$WORKSPACE_DIR"

echo "=== free disk before build ==="
df -h /workspace/rtx3070-trtllm-latency-lab | tail -1

echo "=== nvidia-smi before build ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits

echo "=== trtllm-bench build (128/128 smoke) ==="
# Note: 1.0.0 'trtllm-bench build' has three mutually-exclusive
# (all_or_none) parameter groups:
#   * Dataset:                --dataset
#   * IFB Scheduler Limits:   --max_batch_size + --max_num_tokens
#   * Tuning Heuristics:      --target_input_len + --target_output_len
# Runbook mixed groups; use Dataset-mode (preferred for our reproducible
# synthetic dataset). --max_seq_len is independent and stays.
trtllm-bench \
  --model "$HF_MODEL" \
  --workspace "$WORKSPACE_DIR" \
  --log_level info \
  build \
  --dataset data/synthetic_128_128.jsonl \
  --max_seq_len 256 \
  --tp_size 1 \
  --pp_size 1 \
  2>&1 | tee logs/stage1/build_128_128.log

echo
echo "=== engine dirs after smoke build ==="
find "$WORKSPACE_DIR" -type d -name "tp_1_pp_1" -print | tee logs/stage1/engine_dirs_after_smoke_build.txt

echo
echo "=== free disk after build ==="
df -h /workspace/rtx3070-trtllm-latency-lab | tail -1
