#!/bin/bash
set -euo pipefail
WORKDIR=/workspace/rtx3070-trtllm-latency-lab
cd "$WORKDIR"

mkdir -p logs/stage1 data

export HF_MODEL=Qwen/Qwen2.5-1.5B-Instruct
export HF_HOME=/workspace/rtx3070-trtllm-latency-lab/models/.hf
mkdir -p "$HF_HOME"

PREPARE_DATASET=external/TensorRT-LLM/benchmarks/cpp/prepare_dataset.py
echo "=== prepare_dataset path: $PREPARE_DATASET"
ls -la "$PREPARE_DATASET"

echo "=== prepare_dataset --help (container)"
python3 "$PREPARE_DATASET" --help 2>&1 | tee logs/stage1/prepare_dataset_help_container.txt
echo
echo "=== token-norm-dist --help (container)"
python3 "$PREPARE_DATASET" --tokenizer "$HF_MODEL" token-norm-dist --help 2>&1 | tee logs/stage1/prepare_dataset_token-norm-dist_help.txt || true

echo
echo "=== generating data/synthetic_128_128.jsonl (50 reqs)"
python3 "$PREPARE_DATASET" \
  --stdout \
  --tokenizer "$HF_MODEL" \
  token-norm-dist \
  --input-mean 128 --output-mean 128 \
  --input-stdev 0 --output-stdev 0 \
  --num-requests 50 \
  > data/synthetic_128_128.jsonl

echo "=== generating data/synthetic_512_128.jsonl (100 reqs)"
python3 "$PREPARE_DATASET" \
  --stdout \
  --tokenizer "$HF_MODEL" \
  token-norm-dist \
  --input-mean 512 --output-mean 128 \
  --input-stdev 0 --output-stdev 0 \
  --num-requests 100 \
  > data/synthetic_512_128.jsonl

echo "=== generating data/synthetic_1024_256.jsonl (30 reqs)"
python3 "$PREPARE_DATASET" \
  --stdout \
  --tokenizer "$HF_MODEL" \
  token-norm-dist \
  --input-mean 1024 --output-mean 256 \
  --input-stdev 0 --output-stdev 0 \
  --num-requests 30 \
  > data/synthetic_1024_256.jsonl

echo
echo "=== validation"
wc -l data/synthetic_*.jsonl | tee logs/stage1/dataset_line_counts.txt
head -n 1 data/synthetic_512_128.jsonl | tee logs/stage1/dataset_sample_512_128.txt
head -n 1 data/synthetic_128_128.jsonl | tee logs/stage1/dataset_sample_128_128.txt
head -n 1 data/synthetic_1024_256.jsonl | tee logs/stage1/dataset_sample_1024_256.txt
