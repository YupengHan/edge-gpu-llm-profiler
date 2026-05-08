# 00 — Project brief

## Project title

**RTX 3070 TensorRT-LLM Latency & Profiling Lab**

Recommended repo slug:

```text
rtx3070-trtllm-latency-lab
```

Alternative names:

| Name | Why it is less preferred |
|---|---|
| `local-llm-latency-lab` | Too generic; does not show the fixed hardware/backend. |
| `edge-gpu-llm-profiler` | Good conceptually, but “edge GPU” may sound like Jetson rather than RTX 3070. |
| `trtllm-inference-profiler` | Good backend focus, but hides the RTX 3070 constraint. |
| `rtx3070-llm-benchmark` | Clear, but undersells the profiling and optimization scope. |

## Problem statement

Given a fixed local machine with an NVIDIA RTX 3070 GPU, deploy a small open-source language model locally with TensorRT-LLM, then measure and optimize single-user inference latency.

The first stage focuses on FP16/no explicit quantization. Later stages extend toward quantization, FP8 portability, and multimodal models.

## What makes this project worth showing

This is not a simple “run a local model” demo. The value is the systems workflow:

1. **Hardware-constrained model selection**: choose a model that fits RTX 3070 VRAM.
2. **Deployment backend discipline**: use TensorRT-LLM instead of immediately switching to easier general-purpose backends.
3. **Benchmark methodology**: define reproducible input/output lengths, warmups, repeated runs, and metric schemas.
4. **Latency decomposition**: separate TTFT from ITL/TPOT and end-to-end latency.
5. **Profiler-driven optimization**: use Python profiling, Nsight Systems, and Nsight Compute to identify bottlenecks.
6. **Public documentation**: record every stage in GitHub-readable form.

## Hardware assumption

| Component | Assumption |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 |
| VRAM | Usually 8GB; verify with `nvidia-smi` |
| Architecture | Ampere, compute capability typically 8.6; verify with PyTorch/CUDA |
| OS | Linux or WSL2 Linux preferred |
| Container runtime | Docker + NVIDIA Container Toolkit preferred |

## Compatibility risk

TensorRT-LLM compatibility with an RTX 3070 should be treated as a hypothesis to validate, not as a guaranteed fact.

Why:

- The latest TensorRT-LLM supported hardware page currently lists Blackwell, Hopper, Ada L40/L40S, and Ampere A100 explicitly.
- An older TensorRT-LLM overview page listed RTX 30 series under Ampere production workloads.
- Consumer RTX 30-series support may vary by TensorRT-LLM version, CUDA/driver version, and kernel/plugin path.

Therefore, Stage 0 requires an explicit environment and compatibility report before building the full benchmark story.

## Primary model

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Rationale:

| Criterion | Why Qwen2.5-1.5B-Instruct fits |
|---|---|
| Size | Small enough for 8GB VRAM in FP16 with modest sequence lengths. |
| Quality | Stronger and more modern than TinyLlama-class smoke-test models. |
| Model family | Qwen2/Qwen2.5 architecture is represented in TensorRT-LLM support docs. |
| Benchmark relevance | Appears in edge/local benchmark references, making comparison discussion easier. |
| Future extensions | Can later compare with 0.5B, 3B, coder, or VL variants. |

Known model facts to record in README:

| Field | Value |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Parameters | 1.54B total; 1.31B non-embedding |
| Layers | 28 |
| Attention | GQA: 12 Q heads, 2 KV heads |
| Context length | 32,768 tokens according to model card |

## Backend decision

Primary backend:

```text
TensorRT-LLM
```

Not preferred in Stage 1:

```text
vLLM
llama.cpp
SGLang
custom C/C++ inference code
```

Reasoning:

- The goal is to learn NVIDIA inference deployment/profiling, not merely to maximize ease of setup.
- TensorRT-LLM exposes a path toward TensorRT engines, NVIDIA kernel behavior, Nsight profiling, and quantization workflows.
- A clean TensorRT-LLM project is more aligned with NVIDIA/CUDA/inference-systems hiring signals.

## Stage 1 benchmark target

Primary workload:

```text
input tokens: 512
output tokens: 128
concurrency: 1
batch size: 1
precision: FP16 / no explicit quantization
backend: TensorRT-LLM TensorRT backend when available
```

Auxiliary workloads:

```text
128/128    smoke test
512/128    main public result
1024/256   profiling-heavy stretch case if VRAM allows
```

The 512/128 case is chosen because it aligns with the Edge LLM Leaderboard workload while still being practical for an RTX 3070.

## What to measure

| Metric | Meaning |
|---|---|
| TTFT | Time from request submission to first generated token. |
| ITL / TPOT | Inter-token latency for generated tokens after the first token. |
| E2E latency | Full request latency from submission to completion. |
| Output tokens/sec/user | Single-user generation throughput. |
| Peak VRAM | Maximum GPU memory used during run. |
| GPU utilization | From `nvidia-smi`, Nsight Systems, or server metrics. |
| Kernel characteristics | From Nsight Compute on selected kernels. |

## External benchmark gap

This project should not claim to be directly comparable to datacenter GPU benchmark tables. The public story should be:

> Existing leaderboards and papers provide useful methodology, but RTX 3070 + TensorRT-LLM + Qwen2.5-1.5B-Instruct + FP16 single-user profiling is narrow enough that the most reliable benchmark is a carefully documented local benchmark.

## Non-goals for Stage 1

Do not do these in Stage 1:

- FP8 quantization.
- Accuracy evaluation beyond basic sanity prompts.
- Multimodal deployment.
- Multi-user serving optimization.
- Distributed inference.
- Training or fine-tuning.
- Replacing TensorRT-LLM with vLLM/llama.cpp for convenience.

## Stage 1 success criteria

Stage 1 is successful if the repo contains:

1. A verified environment report.
2. A working TensorRT-LLM deployment path or a clearly documented TensorRT-LLM incompatibility finding.
3. At least one successful FP16/no-quant benchmark for 128/128 or 512/128.
4. TTFT, ITL/TPOT, E2E, tokens/sec/user, and VRAM measurements.
5. One Nsight Systems profile summary.
6. One Nsight Compute kernel-level observation.
7. GitHub-ready README update with a result table and discussion.
