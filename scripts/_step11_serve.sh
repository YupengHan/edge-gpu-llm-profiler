#!/bin/bash
# Step 11 — start trtllm-serve inside the container, foreground (PID 1).
# This script is the ENTRYPOINT of the detached container started by the host.
# Host launches with `docker run -d --name trtllm_serve_step11 ...`.
set -euo pipefail
WORKDIR=/workspace/rtx3070-trtllm-latency-lab
cd "$WORKDIR"

mkdir -p logs/stage1

export HF_MODEL=Qwen/Qwen2.5-1.5B-Instruct
export HF_HOME=/workspace/rtx3070-trtllm-latency-lab/models/.hf
export ENGINE_512_128=trtllm_workspace/Qwen/Qwen2.5-1.5B-Instruct/tp_1_pp_1

echo "=== nvidia-smi before serve ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits

echo
echo "=========================================="
echo "= Step 11 — trtllm-serve (512/128 engine) ="
echo "=========================================="
# Note: runbook says --backend tensorrt, 1.0.0 actually accepts only [pytorch|trt].
# Use --backend trt for the TensorRT C++ runtime.
exec trtllm-serve serve \
  --host 0.0.0.0 \
  --port 8000 \
  --backend trt \
  --tokenizer "$HF_MODEL" \
  --max_batch_size 1 \
  --max_num_tokens 640 \
  --max_seq_len 640 \
  --tp_size 1 \
  --pp_size 1 \
  --log_level info \
  "$ENGINE_512_128"
