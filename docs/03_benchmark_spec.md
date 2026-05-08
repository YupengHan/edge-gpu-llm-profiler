# 03 — Benchmark specification

## Benchmark purpose

Measure local single-user LLM inference latency on RTX 3070 with TensorRT-LLM.

Stage 1 focuses on FP16/no explicit quantization and a single small model:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

## Benchmark modes

Use two complementary benchmark modes.

| Mode | Tool | Purpose | Trust level for metrics |
|---|---|---|---|
| Official backend benchmark | `trtllm-bench` | Engine build, throughput, request latency, reproducible TensorRT-LLM workflow. | Source of backend benchmark data. |
| Streaming client benchmark | Custom OpenAI-compatible streaming client against `trtllm-serve` | Direct TTFT and ITL measurement. | Source of TTFT/ITL data. |

Why both are needed:

- `trtllm-bench` is the official TensorRT-LLM benchmark workflow.
- Streaming TTFT/ITL needs token-arrival timestamps; this is easiest to measure with a streaming client.

## Workload matrix

| Workload ID | Input tokens | Output tokens | Requests | Warmup | Purpose |
|---|---:|---:|---:|---:|---|
| `smoke_128_128` | 128 | 128 | 20 measured | 5 | First successful run; catches setup issues. |
| `main_512_128` | 512 | 128 | 50–100 measured | 5–10 | Main public Stage 1 result. |
| `stretch_1024_256` | 1024 | 256 | 30 measured | 5 | Exposes longer prefill/decode behavior if VRAM allows. |

Main case:

```text
main_512_128
```

Reason:

- It aligns with the Edge LLM Leaderboard's 512 prompt / 128 generation setup.
- It is large enough to make prefill and decode behavior visible.
- It should be more practical on 8GB VRAM than long-context workloads.

## Runtime configuration

Stage 1 uses:

| Parameter | Value |
|---|---|
| Batch size | 1 |
| Concurrency | 1 |
| Tensor parallelism | 1 |
| Pipeline parallelism | 1 |
| Temperature | 0 where supported |
| Quantization | None / do not pass FP8 |
| Precision label | FP16/no explicit quantization |

## Metric definitions

### TTFT — Time to First Token

Time from request submission to first generated token arrival.

Formula:

```text
TTFT = timestamp(first generated token) - timestamp(request submit)
```

Interpretation:

- Includes request/client/server overhead in the streaming benchmark.
- Strongly affected by prefill, scheduling, kernel launch overhead, and server overhead.
- Should be reported as p50/p90/p95 and mean.

### ITL / TPOT — Inter-token latency / Time per Output Token

Time gap between generated token arrivals after the first token.

Formula:

```text
ITL_i = timestamp(token_i) - timestamp(token_{i-1}), for i >= 2
```

Report:

```text
mean ITL
p50 ITL
p90 ITL
p95 ITL
```

Interpretation:

- More representative of decode speed than total latency.
- For streaming client, token counting may be based on streamed text chunks rather than tokenizer-level tokens unless tokenizer decoding is added. Record which counting method is used.

### E2E latency

Full request latency.

Formula:

```text
E2E = timestamp(done) - timestamp(request submit)
```

### Output tokens/sec/user

Single-user output throughput.

Preferred formula when tokenizer token count is available:

```text
tokens/sec/user = generated_token_count / E2E_seconds
```

Fallback formula when only streamed chunks are counted:

```text
chunks/sec/user = streamed_output_piece_count / E2E_seconds
```

If using chunks, label the field clearly as `output_tps_by_piece`, not tokenizer-level `tokens/sec`.

### Peak VRAM

Maximum observed GPU memory used during the benchmark window.

Sources:

- `nvidia-smi` CSV polling.
- `trtllm-serve /metrics` endpoint if available.
- TensorRT-LLM benchmark logs if they print memory details.

## Reproducibility rules

### Always record these fields

| Field | Example |
|---|---|
| Date/time | `2026-05-08T12:34:56-07:00` |
| Host OS | Ubuntu / WSL2 / other |
| GPU | NVIDIA GeForce RTX 3070 |
| Driver | From `nvidia-smi` |
| CUDA runtime/toolkit | From container/PyTorch |
| TensorRT-LLM version | From Python import / logs |
| PyTorch version | From Python import |
| Container image | `nvcr.io/nvidia/tensorrt-llm/release:<tag>` |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Backend | TensorRT-LLM TensorRT backend / PyTorch backend |
| Precision | FP16/no explicit quantization |
| Workload | `512/128` |
| Concurrency | 1 |
| Warmup | e.g. 10 |
| Measured requests | e.g. 100 |
| Engine build options | `max_batch_size`, `max_num_tokens`, `max_seq_len`, TP/PP |

### Stabilization practices

Before benchmarking:

1. Close unrelated GPU processes.
2. Run a warmup pass.
3. Record GPU memory before and after.
4. Avoid running browser/video workloads on the same GPU during measurement.
5. Keep container/image version constant for before/after optimization comparisons.

Optional if permissions allow:

```bash
sudo nvidia-smi -pm 1
```

Do not require this if the machine is not configured for it.

## Dataset generation spec

Use synthetic token-normal distributions with zero stdev for reproducibility:

```bash
python3 ${PREPARE_DATASET} \
  --stdout \
  --tokenizer ${HF_MODEL} \
  token_norm_dist \
  --input-mean 512 \
  --output-mean 128 \
  --input-stdev 0 \
  --output-stdev 0 \
  --num-requests 100 \
  > data/synthetic_512_128.jsonl
```

If local `prepare_dataset.py --help` uses `token-norm-dist`, use that spelling and record it.

## Result storage

Recommended layout:

```text
results/
  stage0/
    environment_report.md
  stage1/
    baseline_summary.md
    streaming_512_128_summary.txt
    benchmark_raw/
      trtllm_bench_128_128_latency.json
      trtllm_bench_512_128_latency.json
      trtllm_bench_512_128_throughput.json
      streaming_512_128.jsonl
```

## JSONL schema for streaming benchmark

Each line in `streaming_512_128.jsonl` should include:

```json
{
  "request_index": 0,
  "warmup": false,
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "backend": "trtllm-serve-tensorrt",
  "workload": "512_128",
  "max_tokens": 128,
  "ttft_ms": 123.4,
  "itl_ms_mean": 12.3,
  "itl_ms_p50": 11.9,
  "itl_ms_p90": 15.2,
  "itl_ms_p95": 17.8,
  "e2e_ms": 1680.5,
  "output_pieces": 128,
  "output_tps_by_piece": 76.1,
  "text_preview": "..."
}
```

If token-level counts are implemented with the tokenizer, add:

```json
{
  "generated_tokens_tokenizer_count": 128,
  "output_tokens_per_sec": 76.1
}
```

## Summary table template

Use this table in README and `baseline_summary.md`:

| Model | Backend | Precision | Workload | Concurrency | TTFT p50 ms | ITL mean ms | E2E p50 ms | Output tok/s/user | Peak VRAM | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-1.5B-Instruct | TensorRT-LLM | FP16 | 512/128 | 1 | TBD | TBD | TBD | TBD | TBD | Stage 1 baseline |

## Result interpretation rules

Avoid overclaiming:

- Do not compare absolute numbers directly against Raspberry Pi / llama.cpp results.
- Do not claim “optimal RTX 3070 performance” after a first run.
- Do not call an optimization successful unless before/after conditions are controlled.
- Report instability if p90/p95 varies significantly.
- If streaming client counts chunks, do not label chunks as tokenizer tokens.

## Minimum benchmark report

`results/stage1/baseline_summary.md` must include:

```text
1. Environment
2. Model and workload
3. trtllm-bench result table
4. streaming TTFT/ITL result table
5. VRAM and utilization observations
6. Reproducibility notes
7. Known limitations
8. Next profiling/optimization step
```
