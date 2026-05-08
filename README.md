# edge-gpu-llm-profiler

**GitHub repo name:** `edge-gpu-llm-profiler`

A fixed-hardware LLM inference systems project: deploy a small open-source LLM on an RTX 3070-class local machine with TensorRT-LLM, measure single-user latency, profile bottlenecks with Python profiling / Nsight Systems / Nsight Compute, and document optimization progress in a GitHub-friendly format.

## One-sentence positioning

This repo is a reproducible local LLM inference benchmark and profiling lab for **RTX 3070 + TensorRT-LLM + Qwen2.5-1.5B-Instruct + FP16**, with future extensions toward quantization and multimodal models.

## Why this repo exists

Most public LLM benchmarks either target datacenter GPUs or CPU/SoC edge devices. This project targets a practical developer workstation GPU: an RTX 3070 with limited VRAM. The goal is not just to run a model, but to show a complete inference-systems workflow:

1. Select a model that fits the hardware.
2. Build and run it locally with TensorRT-LLM.
3. Measure TTFT, ITL/TPOT, end-to-end latency, output throughput, and VRAM usage.
4. Use profilers to locate bottlenecks.
5. Attempt controlled optimizations and document before/after results.
6. Publish progress clearly enough for hiring managers to evaluate the work.

## Initial decisions

| Decision | Choice |
|---|---|
| Primary GPU | NVIDIA GeForce RTX 3070, assumed 8GB VRAM |
| Primary backend | TensorRT-LLM |
| Primary model | `Qwen/Qwen2.5-1.5B-Instruct` |
| First-stage precision | FP16 / no explicit quantization |
| Main benchmark workload | 512 input tokens / 128 output tokens |
| Auxiliary workloads | 128/128 smoke, 1024/256 stretch |
| Main serving mode | Single user, concurrency = 1 |
| Main metrics | TTFT, ITL/TPOT, E2E latency, tokens/sec/user, peak VRAM |
| Main profiling tools | Python profiler, Nsight Systems (`nsys`), Nsight Compute (`ncu`) |

## Critical compatibility note

RTX 3070 support must be validated before treating TensorRT-LLM as working. The latest TensorRT-LLM supported hardware page currently lists Ampere A100 explicitly, while an older TensorRT-LLM overview page lists RTX 30 series as supported for production workloads. Therefore, Stage 0 is mandatory: record the TensorRT-LLM version, driver, CUDA, GPU compute capability, and whether the chosen TensorRT-LLM path actually builds and runs on this machine.

Do **not** silently switch to vLLM, llama.cpp, SGLang, or C/C++ backends during Stage 1. If TensorRT-LLM fails on the RTX 3070, document the failure and try a TensorRT-LLM version/release fallback before using non-preferred backends.

## Document map

| File | Purpose |
|---|---|
| [`STAGE1_LOCAL_AI_PROMPT.md`](./STAGE1_LOCAL_AI_PROMPT.md) | File to paste into a local AI agent such as Claude Code or Codex for Stage 1 execution. |
| [`docs/00_project_brief.md`](./docs/00_project_brief.md) | Project formulation, benchmark gap, model/backend rationale, risks. |
| [`docs/01_stage_roadmap.md`](./docs/01_stage_roadmap.md) | Stage-by-stage plan from environment validation to FP8 and multimodal extensions. |
| [`docs/02_stage1_local_ai_runbook.md`](./docs/02_stage1_local_ai_runbook.md) | Detailed Stage 1 execution order, commands, logging, outputs, and fallback rules. |
| [`docs/03_benchmark_spec.md`](./docs/03_benchmark_spec.md) | Metric definitions, workload spec, reproducibility rules, result schema. |
| [`docs/04_profiling_playbook.md`](./docs/04_profiling_playbook.md) | Python profiler, nsys, ncu workflow and commands. |
| [`docs/05_github_showcase.md`](./docs/05_github_showcase.md) | How to present progress on GitHub for hiring managers. |
| [`docs/06_references.md`](./docs/06_references.md) | Official documentation and external benchmark references. |
| [`docs/07_troubleshooting.md`](./docs/07_troubleshooting.md) | Known failure modes and what to record. |
| [`docs/templates/`](./docs/templates/) | Templates for experiment logs, benchmark tables, profiling observations, and GitHub issues. |

## Recommended first command for a local AI agent

Paste this file into the local agent first:

```text
STAGE1_LOCAL_AI_PROMPT.md
```

Then tell the agent:

```text
Execute Stage 1 exactly as written. Do not change backend away from TensorRT-LLM unless the runbook says to stop and document a failure. Commit or save progress after each major step.
```

## Stage 1 deliverables

At the end of Stage 1, the repo should contain:

```text
results/stage1/environment_report.md
results/stage1/baseline_summary.md
results/stage1/benchmark_raw/*.json
results/stage1/benchmark_raw/*.jsonl
profiles/stage1/nsys/*.txt
profiles/stage1/ncu/*.txt
logs/stage1/*.log
README.md updated with current status and a baseline result table
```

Large model weights, serialized engines, `.nsys-rep`, and `.ncu-rep` files should not be committed unless Git LFS is configured.

## Suggested GitHub topics

```text
llm-inference, tensorrt-llm, rtx3070, nsight-systems, nsight-compute, profiling, benchmark, qwen, local-llm, cuda
```
