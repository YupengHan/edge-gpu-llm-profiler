#!/bin/bash
# Step 15 — Nsight Systems profile of trtllm-bench latency on the 512/128 engine.
# This script is the docker container entrypoint; the host launches it via
# `docker run ... bash scripts/_step15_nsys.sh`.
set -euo pipefail
WORKDIR=/workspace/rtx3070-trtllm-latency-lab
cd "$WORKDIR"

mkdir -p profiles/stage1/nsys logs/stage1 results/stage1/benchmark_raw

export HF_MODEL=Qwen/Qwen2.5-1.5B-Instruct
export HF_HOME=/workspace/rtx3070-trtllm-latency-lab/models/.hf
export WORKSPACE_DIR=trtllm_workspace
export ENGINE_512_128=trtllm_workspace/Qwen/Qwen2.5-1.5B-Instruct/tp_1_pp_1

NSYS_OUT=profiles/stage1/nsys/qwen25_15b_512_128_latency

echo "=== nvidia-smi before nsys ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits

echo
echo "===================================================="
echo "= Step 15 — nsys profile of trtllm-bench latency    ="
echo "= 20 reqs + 5 warmup, concurrency=1, 512/128 engine ="
echo "===================================================="

# --trace=cuda,nvtx,osrt:
#   cuda  → all CUDA API calls + every kernel/memcpy on the timeline
#   nvtx  → TRT-LLM's NVTX ranges (named scopes for prefill / decode steps)
#   osrt  → libc syscalls (read/write/futex etc) for CPU-side waits
# --sample=cpu  → periodic CPU thread sampling so we can see CPU bottlenecks too
# --force-overwrite=true  → re-run friendly
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  --force-overwrite=true \
  --output "$NSYS_OUT" \
  trtllm-bench \
    --model "$HF_MODEL" \
    --workspace "$WORKSPACE_DIR" \
    latency \
    --backend tensorrt \
    --dataset data/synthetic_512_128.jsonl \
    --engine_dir "$ENGINE_512_128" \
    --concurrency 1 \
    --num_requests 20 \
    --warmup 5 \
    --report_json results/stage1/benchmark_raw/trtllm_bench_512_128_latency_nsys.json \
  2>&1 | tee logs/stage1/nsys_512_128_latency.log

echo
echo "=== nvidia-smi after nsys ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits

echo
echo "=== nsys output files ==="
ls -la profiles/stage1/nsys/

echo
echo "=== generating nsys stats text summary ==="
nsys stats \
  --report cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum,cuda_api_sum,nvtx_sum,osrt_sum \
  --format table \
  "${NSYS_OUT}.nsys-rep" \
  > profiles/stage1/nsys/nsys_summary_512_128.txt \
  2> logs/stage1/nsys_stats_512_128.err || true

echo "wrote profiles/stage1/nsys/nsys_summary_512_128.txt"
