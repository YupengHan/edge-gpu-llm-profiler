# 05 — GitHub showcase plan

## Recommended repo name

Use:

```text
rtx3070-trtllm-latency-lab
```

Why this is the best name:

| Part | Signal |
|---|---|
| `rtx3070` | Shows fixed-hardware constraint. |
| `trtllm` | Shows NVIDIA inference backend focus. |
| `latency` | Shows benchmark goal. |
| `lab` | Leaves room for profiling, optimization, FP8, and multimodal extensions. |

Suggested public title:

```text
RTX 3070 TensorRT-LLM Latency & Profiling Lab
```

## Hiring-manager narrative

Use this story in README:

> I built a reproducible local LLM inference benchmark on a constrained RTX 3070 GPU. The project uses TensorRT-LLM to deploy Qwen2.5-1.5B-Instruct, measures single-user TTFT and inter-token latency, then uses Nsight Systems and Nsight Compute to locate bottlenecks and guide optimization attempts.

The repo should show:

1. Systems thinking.
2. Hardware constraints.
3. Benchmark discipline.
4. NVIDIA profiling tool familiarity.
5. Clear before/after reasoning.
6. Honest limitations.

## Suggested repo structure

```text
rtx3070-trtllm-latency-lab/
  README.md
  STAGE1_LOCAL_AI_PROMPT.md
  docs/
    00_project_brief.md
    01_stage_roadmap.md
    02_stage1_local_ai_runbook.md
    03_benchmark_spec.md
    04_profiling_playbook.md
    05_github_showcase.md
    06_references.md
    07_troubleshooting.md
    templates/
  scripts/
    benchmark_streaming_openai.py
    summarize_streaming_jsonl.py
    collect_env.py
  data/
    README.md
  results/
    stage0/
    stage1/
  profiles/
    stage1/
      nsys/
      ncu/
  logs/
  reports/
    final_report.md
```

## README sections

Use this README structure after Stage 1:

```markdown
# RTX 3070 TensorRT-LLM Latency & Profiling Lab

## Summary

## Current status

## Hardware

## Model and backend

## Benchmark methodology

## Stage 1 baseline results

## Profiling findings

## Optimization attempts

## Related benchmarks and gap analysis

## Reproducibility

## Future work

## References
```

## Result table for README

```markdown
| Stage | Model | Backend | Precision | Workload | Concurrency | TTFT p50 ms | ITL mean ms | E2E p50 ms | Output tok/s/user | Peak VRAM | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | Qwen2.5-1.5B-Instruct | TensorRT-LLM | FP16 | 512/128 | 1 | TBD | TBD | TBD | TBD | TBD | Baseline |
```

## Progress display strategy

Hiring managers should see progress without reading every log. Use:

1. README status section.
2. Milestones and issues.
3. Small result tables.
4. Short profiler findings.
5. Before/after optimization notes.
6. Screenshots only when they add clarity.

## Milestones

### M0 — Environment validation

Issues:

- Record RTX 3070 hardware and driver.
- Validate Docker GPU access.
- Validate TensorRT-LLM import and CLI.
- Validate Nsight Systems and Nsight Compute.

Visible output:

```text
results/stage0/environment_report.md
```

### M1 — FP16 baseline

Issues:

- Generate benchmark datasets.
- Build TensorRT-LLM smoke engine.
- Build TensorRT-LLM main engine.
- Run `trtllm-bench` 512/128.
- Run streaming TTFT/ITL client.

Visible output:

```text
results/stage1/baseline_summary.md
```

### M2 — Profiling

Issues:

- Capture nsys timeline.
- Summarize GPU/CPU gaps.
- Run ncu on selected kernels.
- Write bottleneck hypothesis.

Visible output:

```text
profiles/stage1/nsys/nsys_summary_512_128.txt
profiles/stage1/ncu/ncu_observation_001.md
```

### M3 — First optimization

Issues:

- Choose one tuning knob.
- Run before/after benchmark.
- Write conclusion.

Visible output:

```text
results/stage1/optimization_attempt_001.md
```

### M4 — Future extensions

Issues:

- FP8/quantization plan.
- Multimodal plan.

Visible output:

```text
results/stage2/quantization_plan.md
results/stage3/multimodal_benchmark_plan.md
```

## Commit sequence

Recommended commit messages:

```text
init: scaffold RTX 3070 TensorRT-LLM latency lab
add: stage roadmap and benchmark specification
add: environment validation report for RTX 3070
add: Qwen2.5-1.5B TensorRT-LLM smoke benchmark
add: 512/128 FP16 baseline benchmark results
add: streaming TTFT and ITL benchmark client
add: Nsight Systems profiling summary
add: Nsight Compute kernel observation
add: first profiler-driven optimization hypothesis
update: README with Stage 1 baseline and findings
```

## GitHub issue template

Use the included:

```text
.github/ISSUE_TEMPLATE/stage_task.md
```

Each issue should include:

```text
Goal
Commands/files
Expected output
Result
Blockers
Next action
```

## What to pin in README after Stage 1

After Stage 1, README should pin three things:

### 1. Baseline table

A single table with the best stable result.

### 2. Profiling finding

Example style:

```markdown
Nsight Systems shows that the first-token path is dominated by prefill and engine/server setup overhead, while decode has repeated GPU kernels with intermittent CPU gaps. Nsight Compute on selected decode kernels suggests the next optimization should focus on [TBD].
```

### 3. Honest limitation

Example style:

```markdown
This is not a claim of globally optimal RTX 3070 performance. It is a reproducible first baseline on a fixed local machine. TensorRT-LLM consumer RTX compatibility can vary by version, so all environment details are recorded.
```

## What not to show

Avoid:

- Massive raw logs in README.
- Claims like “fastest RTX 3070 benchmark.”
- Uncontrolled comparisons with Raspberry Pi, H100, A100, or llama.cpp.
- Unexplained binary profile artifacts.
- Screenshots without interpretation.

## Optional visual assets

Good visuals after Stage 1:

1. Latency bar chart: TTFT vs ITL vs E2E.
2. VRAM-over-time line chart from `nvidia-smi` CSV.
3. nsys timeline screenshot annotated with prefill/decode regions.
4. Before/after optimization table.

Do not add visuals until numbers are stable.

## Final portfolio summary template

```markdown
## What this project demonstrates

- Deployed an open-source 1.5B LLM locally on RTX 3070 with TensorRT-LLM.
- Built a reproducible benchmark workflow for 512-token prompt / 128-token generation.
- Measured single-user TTFT, ITL, E2E latency, and VRAM usage.
- Used Nsight Systems to identify timeline-level bottlenecks.
- Used Nsight Compute to inspect selected CUDA kernels.
- Designed future extensions for quantization and multimodal inference.
```
