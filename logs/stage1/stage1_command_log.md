# Stage 1 command log

Started: (see first timestamped entry below)


## Step 0 — Create directories

Timestamp: 2026-05-07T19:58:37-07:00

```bash
mkdir -p results/stage0 results/stage1/benchmark_raw profiles/stage1/nsys profiles/stage1/ncu logs/stage0 logs/stage1 data engines models scripts external
```

Result: All directories created.

## Step 1 — Record hardware and host environment

Timestamp: 2026-05-07T19:59:49-07:00

```bash
nvidia-smi | tee logs/stage0/nvidia_smi.txt
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,memory.free --format=csv | tee logs/stage0/gpu_query.csv
uname -a | tee logs/stage0/uname.txt
lsb_release -a 2>&1 | tee logs/stage0/lsb_release.txt
docker --version | tee logs/stage0/docker_version.txt
nsys --version | tee logs/stage0/nsys_version_host.txt
ncu --version | tee logs/stage0/ncu_version_host.txt
nvidia-ctk --version | tee logs/stage0/nvidia_ctk_version.txt
```

Result:
- GPU = RTX 3070 Laptop GPU (sm_86, 8 GiB, driver 575.57.08, CUDA 12.9).
- Host = Ubuntu 22.04.5, Docker 28.1.1 with NVIDIA runtime, nsys 2025.5, ncu 2024.1.
- Wrote results/stage0/environment_report.md.

## Step 2 — Verify Docker GPU access (in progress)

Timestamp: 2026-05-07T20:03:00-07:00

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi  # smoke test
docker run --rm --gpus all trt-perf:dev bash -c '... probe ...'  # local image, no tensorrt_llm
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14  # running in background
```

Result so far:
- GPU is visible inside Docker via nvidia/cuda:12.4.0-base-ubuntu22.04 (logs/stage0/docker_basic_gpu.txt).
- Pre-existing trt-perf:dev image lacks tensorrt_llm (logs/stage0/trt_perf_dev_probe.txt) — proceeding with official release container.
- TRTLLM_TAG=1.3.0rc14, TRTLLM_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14 (logs/stage0/trtllm_image.env).
- docker pull is running in background; full transcript at logs/stage0/docker_pull_trtllm.log.

## Step 2 / 3 — TensorRT-LLM 1.3.0rc14 import FAILED

Timestamp: 2026-05-07T20:12:51-07:00

```bash
docker run --rm --gpus all nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14 nvidia-smi  # GPU visible inside container
docker run --rm --gpus all -v $PWD:/workspace/rtx3070-trtllm-latency-lab -w /workspace/rtx3070-trtllm-latency-lab nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14 bash scripts/_step3_probe.sh
```

Result:
- GPU is visible inside the 1.3.0rc14 container (logs/stage0/docker_nvidia_smi.txt).
- nvidia-smi inside container reports the host driver 575.57.08 (CUDA 12.9).
- python: 3.12.3, torch: 2.11.0a0+eb65b36914.nv26.02, tensorrt: 10.15.1.29, transformers: 4.57.3, pydantic: 2.12.5.
- torch.cuda.is_available() = False with RuntimeError 'The NVIDIA driver on your system is too old (found version 12090)'.
- import tensorrt_llm raises the same RuntimeError via tensorrt_llm._torch.cuda_tile_utils:53 calling torch.cuda.get_device_properties().
- trtllm-bench --help and trtllm-serve --help both abort with the same import error (logs/stage0/trtllm_bench_help.txt, trtllm_serve_help.txt).
- Container nsys is 2026.1.1.204; container ncu is 2025.4.1.0.

Interpretation:
- The driver advertises CUDA 12.9 but PyTorch 2.11.0a0+nv26.02 inside the 1.3.0rc14 container was compiled against the CUDA 13.x driver ABI.
- This is the env-level failure described in docs/07 §3 (TensorRT-LLM import fails) AND the consumer-Ampere-driver-mismatch failure mode the runbook calls out.
- Per docs/07 'TensorRT-LLM fallback policy', try a TensorRT-LLM release/tag fallback before declaring full compatibility failure.
- Do NOT switch backend (vLLM/llama.cpp/SGLang/C++).

Next action: pull a TensorRT-LLM release tag whose PyTorch is built against CUDA 12.x and retry the import probe.

## Step 3 — TensorRT-LLM 1.0.0 (release fallback) import OK

Timestamp: 2026-05-07T20:27:32-07:00

```bash
docker rmi nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.0.0
docker run --rm --gpus all -v $PWD:/workspace/rtx3070-trtllm-latency-lab -w /workspace/rtx3070-trtllm-latency-lab nvcr.io/nvidia/tensorrt-llm/release:1.0.0 bash scripts/_step3_probe.sh
```

Result:
- python 3.12.3, torch 2.8.0a0+5228986c39.nv25.06, tensorrt 10.11.0.33, tensorrt_llm 1.0.0.
- transformers 4.53.1, pydantic 2.11.5, flashinfer JIT enabled.
- torch.cuda.is_available()=True, GPU recognized as 'NVIDIA GeForce RTX 3070 Laptop GPU' sm_86.
- trtllm-bench --help and trtllm-serve --help both work.
- Container nsys=2025.3.1.90, ncu=2025.2.1.0.
- Image size on disk: 47.9 GB.

Decision: Stage 1 proceeds on TensorRT-LLM 1.0.0 (TRTLLM_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.0.0).

## Step 4 — Qwen2.5 tokenizer/config check

Timestamp: 2026-05-07T20:29:56-07:00

```bash
docker run --rm --gpus all -v $PWD:/workspace/rtx3070-trtllm-latency-lab -w /workspace/rtx3070-trtllm-latency-lab nvcr.io/nvidia/tensorrt-llm/release:1.0.0 bash scripts/_step4_qwen_check.sh
```

Result:
- Tokenizer loaded for Qwen/Qwen2.5-1.5B-Instruct: vocab=151665 (entries), eos='<|im_end|>' id=151645, pad='<|endoftext|>'.
- Sample tokenization 'Hello, benchmark test.' -> 5 tokens, round-trips.
- Config: model_type=qwen2, architectures=['Qwen2ForCausalLM'], torch_dtype=bfloat16, hidden_layers=28, attn_heads=12, kv_heads=2 (GQA 6:1), hidden_size=1536, intermediate_size=8960, vocab_size=151936, max_position_embeddings=32768, rope_theta=1e6, tie_word_embeddings=True.

## Step 5-6 — Generate synthetic datasets

Timestamp: 2026-05-07T20:29:56-07:00

CLI drift: prepare_dataset.py in 1.0.0 uses subcommand 'token-norm-dist' (hyphen); the runbook shows 'token_norm_dist' (underscore). Used the hyphen form per docs/07 §7.

```bash
docker run --rm --gpus all -v $PWD:/workspace/rtx3070-trtllm-latency-lab -w /workspace/rtx3070-trtllm-latency-lab nvcr.io/nvidia/tensorrt-llm/release:1.0.0 bash scripts/_step6_make_datasets.sh
```

Result:
  30 data/synthetic_1024_256.jsonl
  50 data/synthetic_128_128.jsonl
 100 data/synthetic_512_128.jsonl
 180 total
- Each line is {task_id, input_ids[...], output_tokens}.

## Step 7 — Build TensorRT-LLM engine (128/128 smoke)

Timestamp: 2026-05-07T20:31 (first attempt) / 2026-05-07T20:50 (rerun after host reboot)

### CLI drift vs runbook (recorded per Execution discipline #4)

The runbook (docs/02 §Stage 1 scope) specifies `Precision: FP16 / no explicit
quantization`. `trtllm-bench build --help` in the 1.0.0 release container has
no `--dtype` / `--precision` flag (only `-q/--quantization` for INT/FP4/FP8
quant algorithms). dtype is inferred from the HF model config; for
Qwen/Qwen2.5-1.5B-Instruct that is `torch_dtype=bfloat16`, which the build
log confirms (`Set dtype to bfloat16`).

Decision: accept bfloat16 for Stage 1 on TensorRT-LLM 1.0.0 and treat the
runbook's `FP16` as "non-quantized half precision". On RTX 3070 Laptop
(sm_86) bf16 and fp16 share the same Tensor Core throughput
(86 TFLOPS each per the build log), so this should not change the latency
characterization. If a strict-FP16 run is needed later, switch to the
`tensorrt_llm.LLM` Python API with explicit `dtype='float16'` and rebuild;
do not use trtllm-bench for that case.

### First attempt (interrupted by host hang)

```bash
docker run --rm --gpus all \
  -v $PWD:/workspace/rtx3070-trtllm-latency-lab \
  -w /workspace/rtx3070-trtllm-latency-lab \
  nvcr.io/nvidia/tensorrt-llm/release:1.0.0 \
  bash scripts/_step7_build_smoke.sh 2>&1 | tee logs/stage1/step7_build_smoke.log
```

Result: HF model downloaded (95.9s, snapshot 989aa7980e4cf806f80c7fef2b1adb7bc71aa306).
Build entered TRT engine compilation (`Compiler backend is used during engine build`)
and the host froze; user had to hard-reboot. Host has 16 GiB RAM and only
2 GiB swap, so TRT builder peak RAM exhausted swap and locked up the system.

Cleanup after reboot: removed root-owned `trtllm_workspace/tmp5hl3rap8-llm-workspace`
via a throwaway 1.0.0 container.

### Rerun with docker resource limits

```bash
docker run --rm --gpus all \
  --memory 10g --memory-swap 12g \
  --shm-size 2g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $PWD:/workspace/rtx3070-trtllm-latency-lab \
  -w /workspace/rtx3070-trtllm-latency-lab \
  nvcr.io/nvidia/tensorrt-llm/release:1.0.0 \
  bash scripts/_step7_build_smoke.sh 2>&1 | tee logs/stage1/step7_build_smoke.log
```

Rationale: cap container RAM at 10 GiB (physical 16 GiB total, ~3 GiB used
by host) so docker OOM-kills the builder before the host hangs. Swap budget
12 GiB > 10 GiB allows a small overflow without dragging the desktop down.
`--shm-size 2g` and the two `--ulimit` flags follow the NGC container's
own startup banner.

Result:
- Engine build succeeded. Engine generation 64.5 s, total trtllm-bench build
  time ~81.6 s plus ~25 s of warmup-load that 1.0.0 trtllm-bench performs
  after build.
- Reported `Build phase peak memory: 10053.02 MB`; that is right at the
  10 GiB container cap (the 2 GiB swap headroom carried it). For the
  512/128 (max_seq_len 640) and 1024/256 (max_seq_len 1280) builds raise
  `--memory` to 12g and `--memory-swap` to 14g.
- Engine dir: `trtllm_workspace/Qwen/Qwen2.5-1.5B-Instruct/tp_1_pp_1/`
  (root-owned, contains rank0.engine 3.02 GiB plus tokenizer files).
  Logged path: logs/stage1/engine_dirs_after_smoke_build.txt.
- Disk: 240 GiB → 243 GiB used (engine + caches ≈ 3 GiB).
- dtype confirmed `bfloat16` in the post-build engine config dump.

## Step 8 — Smoke latency benchmark (128/128, concurrency=1)

Timestamp: 2026-05-07T20:55-07:00

CLI check first (`trtllm-bench latency --help` saved to logs/stage1/trtllm_bench_latency_help.txt):
- Default `--backend` is `pytorch` in 1.0.0 — must pass `--backend tensorrt`
  explicitly to use the engine we just built.
- All other runbook flags (`--engine_dir`, `--dataset`, `--concurrency`,
  `--num_requests`, `--warmup`, `--report_json`, `--iteration_log`) are present.

```bash
docker run --rm --gpus all \
  --memory 10g --memory-swap 12g --shm-size 2g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $PWD:/workspace/rtx3070-trtllm-latency-lab \
  -w /workspace/rtx3070-trtllm-latency-lab \
  nvcr.io/nvidia/tensorrt-llm/release:1.0.0 \
  bash scripts/_step8_latency_smoke.sh 2>&1 | tee logs/stage1/step8_latency_smoke.log
```

Result (20 requests, 5 warmup, concurrency=1, backend=tensorrt):
- Avg request latency: **1311.30 ms**  (P50 1311.3, P90 1320.4, P95 1334.0)
- TTFT: avg **18.33 ms**, P50 17.28, P90 25.35, P95 25.86
- TPOT: avg **10.18 ms**, P50 10.18, P90 10.26, P95 10.34 (very tight)
- Per-user output throughput: **97.62 tok/s**, per-request gen throughput 98.23 tps/user
- Reports: `results/stage1/benchmark_raw/trtllm_bench_128_128_latency.json`
  (3.7 KB) and `..._iteration.jsonl` (2.5 MB).
- GPU idle after: 300 / 7541 MiB.

## Step 9 — Build TensorRT-LLM engine for main 512/128 workload

Timestamp: 2026-05-07T20:57-07:00

### CLI drift vs runbook (Execution discipline #4)

Runbook step 9 mixes `--dataset` with `--max_batch_size` and `--max_num_tokens`,
but `trtllm-bench build` in 1.0.0 makes those parameter groups
`[all_or_none]` and **mutually exclusive**:

* Dataset:                `--dataset`
* IFB Scheduler Limits:   `--max_batch_size` + `--max_num_tokens`
* Tuning Heuristics:      `--target_input_len` + `--target_output_len`

Decision: pick IFB Scheduler Limits (`--max_batch_size 1 --max_num_tokens 640
--max_seq_len 640`) for step 9, since the runbook's explicit
`max_batch_size=1` is the single-user-concurrency=1 knob and dominates the
intent of the step (the dataset is for benchmarking, not heuristic tuning).
Recorded in scripts/_step9_build_main.sh comments.

### Command

```bash
docker run --rm --gpus all \
  --memory 12g --memory-swap 14g --shm-size 2g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $PWD:/workspace/rtx3070-trtllm-latency-lab \
  -w /workspace/rtx3070-trtllm-latency-lab \
  nvcr.io/nvidia/tensorrt-llm/release:1.0.0 \
  bash scripts/_step9_build_main.sh 2>&1 | tee logs/stage1/step9_build_main.log
```

Result:
- Engine generation 50.7 s, total trtllm-bench build ~53 s plus ~25 s warmup.
- `Build phase peak memory: 11617.62 MB` (under the 12 GiB cap with ~670 MiB
  headroom; the 14 GiB swap budget was not consumed).
- Engine config: max_seq_len=640, max_batch_size=1, max_num_tokens=640,
  dtype=bfloat16, KV cache dtype=None, no quant.
- Engine dir: `trtllm_workspace/Qwen/Qwen2.5-1.5B-Instruct/tp_1_pp_1/`
  (root-owned, rank0.engine 3.01 GiB; **this overwrote the 128/128 smoke
  engine**, which is fine — smoke had served its purpose).
- Disk: 240 GiB → 243 GiB used (steady, since 128/128 engine was replaced
  in place).

## Step 10 — Run official `trtllm-bench` main benchmark (512/128, conc=1)

Timestamp: 2026-05-07T20:59 .. 21:05-07:00

### CLI drift check

`trtllm-bench throughput --help` confirmed all runbook flags are present in
1.0.0 (including `--request_json`); transcript saved to
logs/stage1/trtllm_bench_throughput_help.txt. The `latency` subcommand was
already validated in step 8.

### Command

```bash
docker run --rm --gpus all \
  --memory 10g --memory-swap 12g --shm-size 2g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $PWD:/workspace/rtx3070-trtllm-latency-lab \
  -w /workspace/rtx3070-trtllm-latency-lab \
  nvcr.io/nvidia/tensorrt-llm/release:1.0.0 \
  bash scripts/_step10_main_bench.sh 2>&1 | tee logs/stage1/step10_main_bench.log
```

(This step does not compile, so the 128/128 build's 10 GiB cap is enough.)

### Result — latency subcommand (100 req, warmup 10, concurrency=1)

| Metric | Value |
|---|---|
| Avg request latency | **1302.17 ms** (P50 1302.6, P90 1313.4, P95 1318.9, P99 1340.1) |
| TTFT avg | **51.77 ms** (P50 51.36, P90 53.21, P95 56.27, P99 76.02) |
| TPOT avg | **9.85 ms** (P50 9.85, P90 9.95, P95 10.00, P99 10.11) |
| Per-user output throughput w/ ctx | 98.30 tok/s |
| Generation speed | 101.57 tps |
| Total output throughput | 98.29 tok/s |

### Result — throughput subcommand (same shape, 100 req, warmup 10, concurrency=1)

| Metric | Value |
|---|---|
| Avg request latency | 1339.74 ms (P50 1339.8, P90 1346.2, P99 1354.0) |
| Request throughput | 0.7464 req/sec |
| Total output throughput | 95.54 tok/s |
| Total token throughput (in+out) | 477.69 tok/s |
| Per-user output throughput | 95.54 tps/user |

(Throughput subcommand does not surface TTFT/TPOT — that is what the
streaming benchmark client in step 11 is for.)

### Reports written (under `results/stage1/benchmark_raw/`)

- `trtllm_bench_512_128_latency.json` (3.7 KB)
- `trtllm_bench_512_128_iteration.jsonl` (12.6 MB)
- `trtllm_bench_512_128_throughput.json` (2.7 KB)
- `trtllm_bench_512_128_throughput_iteration.jsonl` (12.6 MB)
- `trtllm_bench_512_128_requests.jsonl` (66 KB)

GPU idle after: 300 / 7541 MiB. Disk: 243 GiB used.

---

## Stage 1 progress checkpoint — pause point

Steps 0–10 complete. Engine `trtllm_workspace/Qwen/Qwen2.5-1.5B-Instruct/tp_1_pp_1/`
holds the 512/128 main build (max_seq_len=640, max_batch_size=1,
dtype=bfloat16). Main 512/128 latency + throughput JSON reports written.

**Resume next session at Step 11** — `trtllm-serve` for streaming TTFT/ITL
(scripts/benchmark_streaming_openai.py is the planned client). Open items
to revisit when resuming:

1. **Step 11 CLI drift risk:** runbook's `trtllm-serve serve …` flags need
   verification with `trtllm-serve serve --help` first; we have not yet
   tested this in 1.0.0.
2. **Engine reuse:** step 11 references `${ENGINE_512_128}` =
   `trtllm_workspace/Qwen/Qwen2.5-1.5B-Instruct/tp_1_pp_1` — reuse the
   step 9 engine directly.
3. **Docker resource template (use throughout the rest of stage 1):**
   * Build / compile-heavy steps:   `--memory 12g --memory-swap 14g`
     (peak 11.6 GiB observed for max_seq_len=640).
   * Smoke build (max_seq_len=256): `--memory 10g --memory-swap 12g`
     was on the edge (peak 10.05 GiB) — use 12g/14g going forward to
     keep margin.
   * Inference-only steps:           `--memory 10g --memory-swap 12g`
     is comfortable.
   * Always include `--shm-size 2g --ulimit memlock=-1 --ulimit stack=67108864`.
4. **Host swap is only 2 GiB.** A future compile that exceeds the docker
   `--memory` cap could still drag the desktop down via swap thrash. If
   step 14 (1024/256 build, max_seq_len 1280) approaches 14 GiB peak,
   consider adding host swap (≥8 GiB) before retrying.
5. **Precision policy on record:** dtype is bfloat16, not FP16, because
   1.0.0 `trtllm-bench build` has no `--dtype` override and inherits from
   the HF config. Treat the runbook's "FP16" as "non-quantized half
   precision" for stage 1; if a strict FP16 comparison is needed, switch
   to the `tensorrt_llm.LLM` Python API with `dtype='float16'` — do not
   use trtllm-bench for that.
6. **Engine path collision:** trtllm-bench saves to
   `trtllm_workspace/<MODEL>/tp_<TP>_pp_<PP>/` regardless of build
   parameters, so a later rebuild silently overwrites this engine. If we
   ever need both 512/128 and 1024/256 engines simultaneously, copy the
   directory aside before the next build.
