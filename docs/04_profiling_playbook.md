# 04 — Profiling playbook

## Profiling goal

Use profiling to answer:

```text
Where is the bottleneck in local single-user LLM inference on RTX 3070 with TensorRT-LLM?
```

The project should demonstrate a layered profiling workflow:

```text
1. Python/client/server-level timing
2. Nsight Systems timeline
3. Nsight Compute kernel-level deep dive
4. One controlled optimization attempt
```

## Profiling levels

| Level | Tool | Question answered |
|---|---|---|
| Python/client | Python timing, `time.perf_counter`, optional `cProfile`, optional `torch.profiler` | Is overhead in client/server/Python/tokenization/scheduling? |
| System timeline | Nsight Systems (`nsys`) | What is happening across CPU, CUDA API calls, GPU kernels, synchronization, and gaps? |
| Kernel deep dive | Nsight Compute (`ncu`) | Are selected kernels memory-bound, compute-bound, low occupancy, or inefficient? |

## Bottleneck taxonomy

Use this vocabulary in notes:

| Bottleneck type | Signs |
|---|---|
| Prefill dominated | TTFT high; nsys shows large context/prompt processing before first decode. |
| Decode dominated | ITL high; repeated decode kernels dominate timeline. |
| Kernel launch overhead | Many tiny kernels, visible CPU gaps, low GPU utilization. |
| Memory bandwidth bound | ncu shows high memory throughput pressure and low compute utilization. |
| Compute bound | ncu shows high SM/compute utilization. |
| KV cache pressure | VRAM high, allocation/reuse details show cache constraints. |
| CPU/tokenization overhead | GPU idle while CPU/tokenizer/server work runs. |
| Synchronization overhead | Frequent CPU-GPU sync or blocking CUDA API calls. |
| Unknown | Data insufficient; record what additional trace is needed. |

## Step A — Python/client-level measurement

For TTFT/ITL, use the streaming client described in `docs/02_stage1_local_ai_runbook.md`.

Optional add-ons:

1. Log request construction time.
2. Log HTTP connection time if using `requests` hooks or manual timing.
3. Add tokenizer-based token counting after generation.
4. Compare local client on host vs inside container if HTTP overhead seems suspicious.

Minimum Python timing fields:

```text
request_start_perf_counter
first_token_perf_counter
each_token_perf_counter
request_done_perf_counter
```

## Step B — `trtllm-bench` profiling with nsys

Representative command:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  --force-overwrite=true \
  -o profiles/stage1/nsys/qwen25_15b_512_128_latency \
  trtllm-bench \
    --model ${HF_MODEL} \
    --workspace ${WORKSPACE_DIR} \
    latency \
    --backend tensorrt \
    --dataset data/synthetic_512_128.jsonl \
    --engine_dir ${ENGINE_512_128} \
    --concurrency 1 \
    --num_requests 20 \
    --warmup 5 \
    --report_json results/stage1/benchmark_raw/trtllm_bench_512_128_latency_nsys.json
```

Generate text stats:

```bash
nsys stats profiles/stage1/nsys/qwen25_15b_512_128_latency.nsys-rep \
  > profiles/stage1/nsys/nsys_summary_512_128.txt \
  2> logs/stage1/nsys_stats_512_128.err || true
```

What to look for in nsys:

| Observation | Possible interpretation |
|---|---|
| Long gap before first CUDA work | CPU/server/tokenizer/setup overhead. |
| Long prefill kernel block | TTFT dominated by prompt processing. |
| Repeated decode kernels with little idle time | decode is GPU-bound; use ncu. |
| Repeated decode kernels with gaps | launch/scheduling/CPU overhead. |
| GPU memory copies interleaved with kernels | data movement overhead. |
| CUDA API sync calls | synchronization bottleneck. |

## Step C — `trtllm-serve` profiling with nsys

If the streaming path is the main source of TTFT/ITL, also profile the server process.

Start server under nsys:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=cpu \
  --force-overwrite=true \
  -o profiles/stage1/nsys/qwen25_15b_serve_streaming \
  trtllm-serve serve \
    --host 0.0.0.0 \
    --port 8000 \
    --backend tensorrt \
    --tokenizer ${HF_MODEL} \
    --max_batch_size 1 \
    --max_num_tokens 640 \
    --max_seq_len 640 \
    ${ENGINE_512_128}
```

Then run a small streaming benchmark from another terminal:

```bash
python3 scripts/benchmark_streaming_openai.py \
  --base-url http://localhost:8000/v1 \
  --model "Qwen/Qwen2.5-1.5B-Instruct" \
  --prompt "Explain GPU profiling for LLM inference." \
  --max-tokens 128 \
  --requests 10 \
  --warmup 2 \
  --output results/stage1/benchmark_raw/streaming_512_128_nsys.jsonl
```

Stop the server after the run.

## Step D — Choose kernels for ncu

Do not run Nsight Compute blindly on every kernel in a large trace. Use nsys to pick a small region/kernel class.

Good first `ncu` command:

```bash
ncu \
  --target-processes all \
  --set speedOfLight \
  --launch-skip 20 \
  --launch-count 10 \
  -o profiles/stage1/ncu/qwen25_15b_512_128_selected \
  trtllm-bench \
    --model ${HF_MODEL} \
    --workspace ${WORKSPACE_DIR} \
    latency \
    --backend tensorrt \
    --dataset data/synthetic_512_128.jsonl \
    --engine_dir ${ENGINE_512_128} \
    --concurrency 1 \
    --num_requests 10 \
    --warmup 2
```

If too slow, reduce:

```text
--launch-count 3
--num_requests 3
--warmup 1
```

If data is too shallow, use a richer section set later:

```bash
ncu \
  --target-processes all \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section SchedulerStats \
  --launch-skip 20 \
  --launch-count 5 \
  -o profiles/stage1/ncu/qwen25_15b_512_128_mem_sched \
  <benchmark command>
```

## What to extract from ncu

Minimum observation fields:

| Field | Example |
|---|---|
| Kernel name | `...` |
| Kernel role | attention / matmul / layernorm / unknown |
| Duration | ms or us |
| SM utilization | high/medium/low |
| Memory throughput | high/medium/low |
| Occupancy | high/medium/low |
| Bottleneck hypothesis | memory-bound / compute-bound / launch overhead / unclear |
| Next action | tune engine config / inspect KV cache / compare sequence length / etc. |

## Optimization experiment design

Only change one variable at a time.

Template:

```text
Hypothesis:
Changing X will reduce Y because Z.

Baseline:
Model, backend, precision, workload, command, result.

Change:
One exact parameter change.

After:
Same workload, same warmup, same request count.

Result:
Table with p50/p90/p95 TTFT, ITL, E2E, VRAM.

Conclusion:
Keep / revert / needs more data.
```

## Candidate first optimization knobs

| Knob | Controlled comparison |
|---|---|
| `max_seq_len` | Compare over-provisioned vs workload-specific sequence length. |
| `max_num_tokens` | Compare 640 vs larger value for 512/128 workload. |
| `free_gpu_memory_fraction` | Compare default vs slightly lower/higher if KV cache allocation dominates. |
| `enable_chunked_prefill` | Compare on/off for TTFT and throughput. |
| Server vs bench path | Compare `trtllm-bench` latency vs `trtllm-serve` streaming latency. |

## Commit guidance

Commit summaries, not huge binaries.

Recommended commits:

```text
commit 1: add stage1 benchmark/profiling scaffold
commit 2: add environment report for RTX 3070
commit 3: add FP16 baseline benchmark results
commit 4: add streaming TTFT/ITL benchmark client
commit 5: add nsys summary and bottleneck note
commit 6: add ncu observation and first optimization hypothesis
```

## Profiling report template

Use:

```text
docs/templates/profile_observation_template.md
```

for each meaningful profiling observation.
