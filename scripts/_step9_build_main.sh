#!/bin/bash
set -euo pipefail
WORKDIR=/workspace/rtx3070-trtllm-latency-lab
cd "$WORKDIR"

mkdir -p logs/stage1 trtllm_workspace

export HF_MODEL=Qwen/Qwen2.5-1.5B-Instruct
export HF_HOME=/workspace/rtx3070-trtllm-latency-lab/models/.hf
export WORKSPACE_DIR=trtllm_workspace
mkdir -p "$HF_HOME" "$WORKSPACE_DIR"

echo "=== free disk before build ==="
df -h /workspace/rtx3070-trtllm-latency-lab | tail -1

echo "=== nvidia-smi before build ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits

echo "=== trtllm-bench build (512/128 main, IFB scheduler limits) ==="
# 1.0.0 build groups are all_or_none and mutually exclusive:
#   * Dataset:                --dataset
#   * IFB Scheduler Limits:   --max_batch_size + --max_num_tokens
#   * Tuning Heuristics:      --target_input_len + --target_output_len
# Runbook step 9 mixes --dataset with --max_batch_size and --max_num_tokens,
# which 1.0.0 rejects. We pick IFB Scheduler Limits to match the runbook's
# max_batch_size=1 / max_num_tokens=640 intent (single-user concurrency=1
# main workload). max_seq_len stays independent.
trtllm-bench \
  --model "$HF_MODEL" \
  --workspace "$WORKSPACE_DIR" \
  --log_level info \
  build \
  --max_batch_size 1 \
  --max_num_tokens 640 \
  --max_seq_len 640 \
  --tp_size 1 \
  --pp_size 1 \
  2>&1 | tee logs/stage1/build_512_128.log

echo
echo "=== engine dirs after main build ==="
find "$WORKSPACE_DIR" -type d -name "tp_1_pp_1" -print | tee logs/stage1/engine_dirs_after_main_build.txt

echo
echo "=== free disk after build ==="
df -h /workspace/rtx3070-trtllm-latency-lab | tail -1
