# 01 — Stage roadmap

## Stage overview

| Stage | Name | Main purpose | Output visible on GitHub |
|---:|---|---|---|
| 0 | Feasibility and environment validation | Confirm RTX 3070 + TensorRT-LLM can run the chosen path. | Environment report, compatibility notes, screenshots/logs. |
| 1 | FP16 baseline benchmark and profiling | Run Qwen2.5-1.5B-Instruct locally, measure latency, profile bottlenecks. | Baseline result table, nsys/ncu summaries, first bottleneck analysis. |
| 2 | Quantization extension | Study PTQ/calibration and compare FP16 vs quantized artifacts; FP8 portability if hardware permits. | Quantization notes, calibration workflow, before/after performance table. |
| 3 | Multimodal extension | Adapt benchmark/profiling methodology to a small VLM. | VLM deployment notes, multimodal latency table. |
| 4 | Final polish and portfolio packaging | Turn experiment into a hiring-manager-readable project. | Final report, project architecture diagram, lessons learned. |

## Stage 0 — Feasibility and environment validation

### Goal

Prove the machine can run the project stack or document exactly why it cannot.

### Tasks

1. Record hardware:
   - GPU name.
   - VRAM.
   - compute capability.
   - driver version.
   - CUDA runtime/toolkit version.
2. Confirm Docker + NVIDIA Container Toolkit works.
3. Pull a TensorRT-LLM release container.
4. Run TensorRT-LLM import sanity check.
5. Run `trtllm-bench --help` and `trtllm-serve --help`.
6. Verify `nsys` and `ncu` availability.
7. Record any RTX 3070 support limitations.

### Deliverables

```text
results/stage0/environment_report.md
logs/stage0/*.log
README.md compatibility section update
```

### Success criteria

- `nvidia-smi` works.
- GPU is visible inside Docker.
- TensorRT-LLM import works.
- `trtllm-bench` is available.
- Either `nsys`/`ncu` are available or a clear installation note is written.

### Stop condition

If TensorRT-LLM cannot import or cannot see the GPU, stop and document the failure before attempting benchmark commands.

## Stage 1 — FP16 baseline benchmark and profiling

### Goal

Run Qwen2.5-1.5B-Instruct on RTX 3070 with TensorRT-LLM in FP16/no explicit quantization, then measure and profile single-user inference.

### Tasks

1. Download or allow TensorRT-LLM to fetch `Qwen/Qwen2.5-1.5B-Instruct`.
2. Generate synthetic datasets:
   - 128/128 smoke.
   - 512/128 main.
   - 1024/256 stretch if memory allows.
3. Build TensorRT-LLM engine for the smoke case.
4. Run smoke latency benchmark.
5. Build TensorRT-LLM engine for the main 512/128 case.
6. Run official `trtllm-bench` latency/throughput commands.
7. Run a custom streaming benchmark through `trtllm-serve` for TTFT and ITL.
8. Capture VRAM with `nvidia-smi` and/or `/metrics` endpoint.
9. Capture Nsight Systems trace for one representative run.
10. Use Nsight Compute on a small subset of kernels identified from nsys.
11. Write Stage 1 baseline summary.

### Deliverables

```text
results/stage1/environment_report.md
results/stage1/baseline_summary.md
results/stage1/benchmark_raw/*.json
results/stage1/benchmark_raw/*.jsonl
profiles/stage1/nsys/*.txt
profiles/stage1/ncu/*.txt
logs/stage1/*.log
README.md result table update
```

### Success criteria

- At least one successful TensorRT-LLM run.
- Main 512/128 result if VRAM allows; otherwise 128/128 documented as the first baseline.
- TTFT and ITL measured from streaming, not inferred only from total latency.
- At least one nsys summary and one ncu observation.

## Stage 1A — First optimization attempt

### Goal

Make one controlled optimization based on profiling evidence.

### Candidate knobs

Use only one change at a time:

| Knob | Why it may matter |
|---|---|
| `max_batch_size=1` | Enforces single-user scenario and may reduce allocation/scheduling overhead. |
| `max_seq_len` tuned to workload | Avoids over-provisioning KV cache. |
| `max_num_tokens` tuned to workload | Controls engine scheduling/token limits. |
| `free_gpu_memory_fraction` / KV cache memory fraction | May affect KV allocation and memory headroom. |
| Chunked prefill on/off | May affect TTFT and prefill scheduling. |
| Warmup count | Stabilizes GPU clocks and cache effects. |

### Deliverables

```text
results/stage1/optimization_attempt_001.md
results/stage1/before_after_001.csv
profiles/stage1/nsys_after_001.txt
```

### Success criteria

- A before/after table.
- A hypothesis.
- A result.
- A conclusion, even if performance does not improve.

## Stage 2 — Quantization extension

### Goal

Move from FP16/no explicit quantization toward post-training quantization/calibration experiments.

### Scope

Stage 2 is intentionally separated from Stage 1.

Possible sub-stages:

| Sub-stage | Goal |
|---|---|
| 2A | Study TensorRT-LLM / NVIDIA ModelOpt quantization workflow. |
| 2B | Quantize a compatible small model with calibration data. |
| 2C | Compare FP16 vs quantized artifact on the same workloads. |
| 2D | Investigate FP8 portability and hardware requirements. |

### Important caveat

RTX 3070 does not provide the same FP8 acceleration path as Hopper/Ada/Blackwell-class GPUs. If the goal is actual FP8 speedup, Stage 2 may require a cloud GPU or a second machine. The local RTX 3070 can still be useful for learning quantization workflow, artifact generation, and accuracy/performance comparison where supported.

### Deliverables

```text
results/stage2/quantization_plan.md
results/stage2/calibration_dataset.md
results/stage2/fp16_vs_quantized.md
```

## Stage 3 — Multimodal extension

### Goal

Extend the same benchmark/profiling methodology to a small multimodal model.

### Candidate model direction

Use a small Qwen2.5-VL or Qwen2-VL model only if VRAM allows. Start with the smallest available variant, and do not mix Stage 3 with Stage 1 baseline work.

### New metrics

| Metric | Meaning |
|---|---|
| Image preprocessing latency | Time to prepare image inputs. |
| Vision encoder latency | Vision tower or multimodal encoder time. |
| Text TTFT | Time from request to first generated text token. |
| Text ITL | Inter-token latency after first generated text token. |
| Peak VRAM | Combined vision + language memory footprint. |

### Deliverables

```text
results/stage3/multimodal_benchmark_plan.md
results/stage3/vlm_baseline.md
profiles/stage3/nsys_vlm_summary.md
```

## Stage 4 — Portfolio polish

### Goal

Make the repo understandable in five minutes to a hiring manager.

### Tasks

1. Update README with final status and summary table.
2. Add architecture/workflow diagram.
3. Add “What I learned” section.
4. Add “Profiler-driven findings” section.
5. Add “Future work” section.
6. Make sure logs are summarized, not dumped.
7. Create a small final report under `reports/final_report.md`.

### Deliverables

```text
reports/final_report.md
README.md polished
results/final_summary.csv
```

## Recommended milestone sequence in GitHub

| Milestone | Issues |
|---|---|
| M0 — Environment | Hardware report, Docker/TRT-LLM setup, profiler setup. |
| M1 — FP16 baseline | Dataset generation, engine build, benchmark runs, result schema. |
| M2 — Profiling | nsys trace, ncu deep dive, bottleneck notes. |
| M3 — Optimization | One controlled tuning attempt and before/after result. |
| M4 — Extensions | Quantization plan, multimodal plan. |
| M5 — Portfolio | README polish, final report, plots. |
