# Stage 1 Local AI Execution Prompt

You are a local coding/execution agent working inside this repository.

Your mission is to execute **Stage 1: FP16 baseline benchmark and profiling** for a local RTX 3070 TensorRT-LLM project.

Read this file first, then read:

```text
docs/02_stage1_local_ai_runbook.md
docs/03_benchmark_spec.md
docs/04_profiling_playbook.md
docs/07_troubleshooting.md
```

Do not skip the runbook. Do not silently change the project backend.

## Project constraints

| Constraint | Required behavior |
|---|---|
| GPU | Use the local NVIDIA RTX 3070-class GPU. |
| Backend | Prefer TensorRT-LLM. Do not switch to vLLM, llama.cpp, SGLang, or custom C++ unless explicitly instructed later. |
| Precision | Stage 1 is FP16/no explicit quantization. Do not run FP8 in Stage 1. |
| Model | Primary model is `Qwen/Qwen2.5-1.5B-Instruct`. |
| Scenario | Single user: concurrency = 1, batch size = 1. |
| Main workload | 512 input tokens / 128 output tokens. |
| Profiling | Use Python profiling where relevant, then Nsight Systems, then Nsight Compute. |
| Documentation | Save outputs, logs, and summaries after every major step. |

## Do not do these in Stage 1

- Do not quantize to FP8.
- Do not fine-tune the model.
- Do not benchmark unrelated models unless the runbook says it is a smoke-test fallback.
- Do not replace TensorRT-LLM with vLLM or llama.cpp to “make it work.”
- Do not commit model weights, engine binaries, or large `.nsys-rep` / `.ncu-rep` files unless Git LFS is configured.

## Required run order

Follow this order exactly.

```text
0. Create/verify repo directories.
1. Record environment and hardware.
2. Verify Docker + NVIDIA GPU access.
3. Start TensorRT-LLM container or environment.
4. Verify TensorRT-LLM CLI tools.
5. Verify Qwen2.5 tokenizer/model metadata access.
6. Generate synthetic datasets: 128/128, 512/128, optional 1024/256.
7. Build TensorRT-LLM engine for 128/128 smoke test.
8. Run smoke latency benchmark.
9. Build TensorRT-LLM engine for 512/128 main benchmark.
10. Run official trtllm-bench latency and throughput with concurrency=1.
11. Start trtllm-serve with TensorRT backend/engine if available.
12. Implement and run streaming TTFT/ITL benchmark client.
13. Capture VRAM and server metrics.
14. Run Nsight Systems on one representative benchmark.
15. Run Nsight Compute on a small selected kernel range.
16. Write Stage 1 summary.
17. Update README with current status and first result table.
```

## Required outputs

Create these directories if they do not exist:

```bash
mkdir -p results/stage0 results/stage1/benchmark_raw profiles/stage1/nsys profiles/stage1/ncu logs/stage0 logs/stage1 data engines models scripts
```

At minimum, produce:

```text
results/stage0/environment_report.md
results/stage1/baseline_summary.md
results/stage1/benchmark_raw/trtllm_bench_512_128_latency.json
results/stage1/benchmark_raw/streaming_512_128.jsonl
profiles/stage1/nsys/nsys_summary_512_128.txt
profiles/stage1/ncu/ncu_observation_001.md
logs/stage1/stage1_command_log.md
```

If any command fails, append the exact command, error snippet, probable cause, and next action to:

```text
logs/stage1/stage1_command_log.md
docs/07_troubleshooting.md
```

## First commands

Run these before any benchmark work:

```bash
mkdir -p results/stage0 results/stage1/benchmark_raw profiles/stage1/nsys profiles/stage1/ncu logs/stage0 logs/stage1 data engines models scripts

{
  echo "# Stage 1 command log"
  echo
  echo "Started: $(date -Is)"
  echo
} > logs/stage1/stage1_command_log.md

nvidia-smi | tee logs/stage0/nvidia_smi.txt
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,memory.free --format=csv | tee logs/stage0/gpu_query.csv
```

Then follow `docs/02_stage1_local_ai_runbook.md`.

## If TensorRT-LLM fails on RTX 3070

Do not hide the failure. Write a compatibility finding.

Try these in order:

1. Confirm GPU visibility inside container.
2. Confirm TensorRT-LLM import and version.
3. Try a smaller smoke model only for environment validation, such as `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
4. Try a TensorRT-LLM version/release fallback if the current release appears incompatible.
5. If TensorRT-LLM still fails, stop and write `results/stage0/tensorrt_llm_rtx3070_compatibility_failure.md`.

Do not switch the project to vLLM/llama.cpp unless the human owner explicitly changes the scope.

## Final Stage 1 summary format

When Stage 1 is done, `results/stage1/baseline_summary.md` must include:

```text
1. Hardware and software environment
2. TensorRT-LLM compatibility result
3. Model and workload
4. Official trtllm-bench results
5. Streaming TTFT/ITL results
6. VRAM observations
7. Nsight Systems observations
8. Nsight Compute observations
9. Bottleneck hypothesis
10. Next optimization attempt
```
