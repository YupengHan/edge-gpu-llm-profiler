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

---

## Step 11 — `trtllm-serve` for streaming TTFT/ITL (2026-05-11)

### 11a — Probe `trtllm-serve serve --help` in 1.0.0

The runbook command (`--backend tensorrt`, `--host`, `--port`, …) was
written for an unspecified TRT-LLM version. We probed the actual 1.0.0
CLI in a one-shot foreground container before launching anything
detached.

```bash
docker run --rm --gpus all \
  --memory 4g --memory-swap 5g --shm-size 1g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD":/workspace/rtx3070-trtllm-latency-lab \
  -w /workspace/rtx3070-trtllm-latency-lab \
  nvcr.io/nvidia/tensorrt-llm/release:1.0.0 \
  trtllm-serve serve --help \
  2>&1 | tee logs/stage1/trtllm_serve_serve_help.txt
```

First attempt (without `--gpus all`) failed with
`ImportError: libcuda.so.1: cannot open shared object file`. The
`tensorrt_llm` module loads CUDA bindings even for `--help`; always
pass `--gpus all`.

**CLI drift confirmed (1 item):**

| Runbook | 1.0.0 actual | Note |
|---|---|---|
| `--backend tensorrt` | `--backend [pytorch\|trt]` | Must use **`trt`**, not `tensorrt` |

All other flags match (`--host`, `--port`, `--tokenizer`,
`--max_batch_size`, `--max_num_tokens`, `--max_seq_len`, `--tp_size`,
`--pp_size`). Positional `MODEL` accepts engine path as expected.

### 11b — Launch detached server on existing 512/128 engine

Entrypoint script `scripts/_step11_serve.sh` runs `trtllm-serve serve`
as PID 1 (uses `exec`) so `docker stop` cleanly SIGTERMs the server.

```bash
docker rm -f trtllm_serve_step11 2>/dev/null; true

docker run -d \
  --name trtllm_serve_step11 \
  --gpus all \
  --memory 10g --memory-swap 12g \
  --shm-size 2g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 \
  -v "$PWD":/workspace/rtx3070-trtllm-latency-lab \
  -w /workspace/rtx3070-trtllm-latency-lab \
  nvcr.io/nvidia/tensorrt-llm/release:1.0.0 \
  bash scripts/_step11_serve.sh
```

Inside the entrypoint the actual server command is:

```bash
trtllm-serve serve \
  --host 0.0.0.0 --port 8000 \
  --backend trt \
  --tokenizer Qwen/Qwen2.5-1.5B-Instruct \
  --max_batch_size 1 --max_num_tokens 640 --max_seq_len 640 \
  --tp_size 1 --pp_size 1 --log_level info \
  trtllm_workspace/Qwen/Qwen2.5-1.5B-Instruct/tp_1_pp_1
```

Server startup wall time: **~15 s** (4 polls × 5 s, /health 200 on the
4th poll). Container PID 1 is uvicorn, listening on `0.0.0.0:8000`.

### 11c — Endpoint smoke tests

```bash
curl -s http://localhost:8000/health  | tee logs/stage1/server_health.txt    # 200, empty body
curl -s http://localhost:8000/v1/models | tee logs/stage1/server_models.txt
```

`/v1/models` returns:

```json
{"object":"list","data":[
  {"id":"tp_1_pp_1","object":"model","created":1778563339,
   "owned_by":"tensorrt_llm"}
]}
```

**API drift (important for step 12):** the served model id is
`tp_1_pp_1` (the last path component of the engine directory), **not**
`Qwen/Qwen2.5-1.5B-Instruct`. The streaming bench client and any chat
request must send `"model": "tp_1_pp_1"` in the body — sending the HF
repo name returns a 404 / unknown-model error. Document this so step 12
does not silently 4xx.

Non-streaming chat test (`logs/stage1/server_chat_test.json`):

```json
{"model":"tp_1_pp_1","prompt_tokens":35,"completion_tokens":9,
 "finish_reason":"stop","content":"Hello! How can I assist you today?"}
```

Streaming SSE chat test (`logs/stage1/server_chat_stream_test.txt`):
chunks arrive as standard OpenAI `data: {...}` lines with
`choices[0].delta.content` carrying each text piece — exactly what
`scripts/benchmark_streaming_openai.py` already parses. No client
changes needed beyond the `--model tp_1_pp_1` flag.

### 11d — Resource observations

| When | GPU mem used / free | Note |
|---|---|---|
| Before server | 176 / 7665 MiB | host process baseline |
| After single non-stream chat | 7437 / 404 MiB | engine + KV cache resident |

The 7.4 GiB resident is the steady-state working set for a 1.5B bf16
model + KV cache for `max_seq_len=640, max_batch_size=1`. **There is
only ~400 MiB of headroom on this 8 GiB Laptop GPU**, so anything that
also wants VRAM during step 11/12 (e.g. another CUDA process, a
Chromium tab using NVENC) will OOM the server. Keep the GPU otherwise
idle while the bench runs.

### Open items going into step 12

1. `benchmark_streaming_openai.py` must be invoked with
   `--model tp_1_pp_1` (NOT the HF repo name). Add a one-line note to
   step 12 script.
2. The container `trtllm_serve_step11` stays UP through step 12.
   Cleanup will be a single `docker stop trtllm_serve_step11 &&
   docker rm trtllm_serve_step11` after step 12 is recorded.
3. `usage` field appears in non-streaming responses but is absent from
   the streaming chunks observed so far; the bench client already
   falls back to counting SSE pieces — verify the count matches the
   expected 128 in step 12 results.

---

## Step 12 — Streaming TTFT/ITL benchmark (2026-05-11)

### 12a — First attempt (single prompt) failed methodologically

Ran the original `benchmark_streaming_openai.py` with `--prompt @data/streaming_prompt_512.txt`
(synthetic prompt decoded from step 5's `data/synthetic_512_128.jsonl` line 0).
**Two problems surfaced:**

1. **First run 400'd on every request.** Server log said
   `prompt(570) + max_tokens(128) > max_seq_len(640)`. The decoded synthetic
   text re-tokenizes to 541 tokens (BPE non-canonical decode→encode drift),
   and after Qwen's chat-template wrap totals 570. Truncated the user content
   to 483 tokens so chat-template renders to exactly 512 — fits in 640 with
   128 output headroom. Stored as `data/streaming_prompt_512.txt`.

2. **Same prompt × 55 requests caused two artifacts:**

   - **Prefix caching** (TRT-LLM's `use_paged_context_fmha=True`): request 0
     TTFT 80 ms, requests 1-54 TTFT 15-22 ms. The cache was reusing the prefill
     KV pages, so step 12 numbers no longer compared to step 10's 51.77 ms.
   - **Refusal output**: Qwen-Instruct read the gibberish prompt as
     "I'm sorry, but I can't assist with that." (12 tokens, `finish_reason=stop`).
     Only 11 ITL gaps per request instead of 127. Mean ITL still resembled
     decode timing but the sample was tiny.

   This run is **archived as a prefix-cache demonstration**:
   `results/stage1/benchmark_raw/streaming_512_128_singleprompt_cachebenchmark.jsonl`
   and `logs/stage1/streaming_512_128_singleprompt.log`. It is intentionally
   kept (not deleted) so step 18 can cite it when discussing cache effects.

### 12b — Generated 50 varied 512-token prompts

`scripts/generate_streaming_prompts.py` produces 50 unique natural-language
prompts (5 instruction templates × 10 ML-systems topics). Each prompt is
binary-searched to exactly **512 chat-template tokens**, and every prompt
ends with "Write at least 220 words" so the assistant generates well beyond
128 tokens and `max_tokens=128` is the binding cutoff (`finish_reason=length`).

```bash
docker exec trtllm_serve_step11 python3 scripts/generate_streaming_prompts.py \
  --target-template-total 512 \
  --num-prompts 50 \
  --out data/streaming_prompts_512.jsonl
```

Output: 50 rows, every row's `chat_template_total_tokens == 512`.

### 12c — Modified bench client to rotate prompts

Added a mutually-exclusive `--prompts-file` flag to
`scripts/benchmark_streaming_openai.py`. When set, the client rotates through
the JSONL with `request_index % N`, and records `prompt_index` on each row.
The original `--prompt` mode is unchanged.

### 12d — Second run (rotated prompts, 50+5 reqs)

```bash
python3 scripts/benchmark_streaming_openai.py \
  --base-url http://localhost:8000/v1 \
  --model tp_1_pp_1 \
  --prompts-file data/streaming_prompts_512.jsonl \
  --max-tokens 128 --requests 50 --warmup 5 --temperature 0 \
  --workload-label 512_128 --backend-label trtllm-1.0.0-trt \
  --output results/stage1/benchmark_raw/streaming_512_128.jsonl \
  > logs/stage1/streaming_512_128.log 2>&1
```

Result: **0 errors**, all 55 rows have `output_pieces=128` and
`finish_reason="length"`. TTFT trajectory shows the expected two regimes:

- **Requests 0-49**: 50-58 ms TTFT (each prompt is unique → cold prefill)
- **Requests 50-54**: 15-17 ms TTFT (prompt index wraps to 0-4 → prefix cache hits)

This is the same cache effect as 12a, but now observable inside a single
clean run where the cold regime is the headline number.

### 12e — Summary (cold vs warm split)

`results/stage1/streaming_512_128_summary.txt`. Non-warmup rows = 5..54.
We split by whether the prompt had been seen before in this run:

| Regime | n | TTFT mean | TTFT p50 | TTFT p90 | ITL mean | E2E mean |
|---|---:|---:|---:|---:|---:|---:|
| **COLD (req 5-49, unique prompts)** | 45 | **54.28 ms** | 53.90 | 57.64 | **10.56 ms** | 1395.90 ms |
| WARM (req 50-54, cache hit) | 5 | 15.91 ms | 15.79 | 16.85 | 10.31 ms | 1325.61 ms |

### 12f — Comparison vs step 10

| Metric | step 10 (`trtllm-bench latency`) | step 12 cold (streaming) | delta |
|---|---:|---:|---:|
| TTFT mean | 51.77 ms | 54.28 ms | **+2.5 ms** (HTTP+SSE overhead) |
| TPOT/ITL mean | 9.85 ms | 10.56 ms | **+0.7 ms** (uvicorn per-chunk) |
| E2E mean | 1302.17 ms | 1395.90 ms | **+94 ms** total |

Formula check (step 12 cold): TTFT + ITL × 127 = 54.28 + 10.56 × 127 =
1395.4 ms ≈ measured 1395.9 ms. The two-stage prefill+decode model still
holds end-to-end through the HTTP layer.

Output throughput (tps_by_piece) for cold runs: mean 92.20 pieces/sec,
vs step 10 generation speed 101.57 tps. The ~9 % gap is the SSE chunk
overhead distributed over 127 ITL intervals.

### Open items going into step 13

1. `summarize_streaming_jsonl.py` currently does not auto-detect the
   cold/warm split — it lumps all non-warmup rows together. Step 13 will
   either extend it with a `--cold-only` filter that watches for
   prompt_index repeats, or accept that step 18's baseline summary will
   handle the split manually. Decide in step 13.
2. Server container `trtllm_serve_step11` is still UP. step 12 is done;
   it is safe to `docker stop trtllm_serve_step11` now unless step 13
   needs the server. step 13 only re-summarizes existing JSONL, so stop
   the container before step 14 if step 13 doesn't need it.
3. nvidia-smi after bench: 7437 / 404 MiB — unchanged. KV cache pool
   absorbed the request burst without growing the resident set.

---

## Step 13 — Summarize streaming results (2026-05-11)

### 13a — Extended summarize_streaming_jsonl.py

Three changes vs the step 12 version:

1. **Auto cold/warm split via `prompt_index` first-occurrence.** Warmup
   rows are excluded from the summary, but their `prompt_index` values
   are still recorded into `seen` so that a measured row reusing a
   warmup prompt is correctly classified as **warm**. (Without this,
   step 12 measured rows 50-54 — which reuse warmup prompts 0-4 —
   would have been mis-counted as cold.) Add `--no-split` to disable.
2. **Flattened-per-token ITL block.** If rows carry a new `itl_ms` array
   (raw gaps, added to the bench client in 13b for forward runs), the
   summarizer prints true global ITL p50/p90/p95/p99. Older JSONLs
   (including step 12's main file and the archived single-prompt file)
   don't have this, and the summarizer says so explicitly.
3. **`--json <path>` machine-readable output** for step 18/19 to
   consume without re-parsing the text report.

### 13b — Forward-compat: bench client now saves raw gaps

`scripts/benchmark_streaming_openai.py` row now includes
`itl_ms: [round(g, 4) for g in gaps_ms]`. Old data without this field
remains valid (summarizer handles the absence gracefully). **Did not
re-run step 12**; the headline cold-regime numbers are stable across
runs and the per-request ITL percentiles already converge tightly
(p50 10.60 vs p95 10.76 — a 1.5 % spread), so the flattened p95 would
not change the story. Future bench runs (step 17 stretch and any
re-runs) will capture the raw gaps automatically.

### 13c — Final summary numbers (committed)

```bash
python3 scripts/summarize_streaming_jsonl.py \
  results/stage1/benchmark_raw/streaming_512_128.jsonl \
  --json results/stage1/streaming_512_128_summary.json \
  > results/stage1/streaming_512_128_summary.txt
```

Result, COLD (n=45, unique-prompt requests — this is the step 12
baseline number for stage 1):

| Metric | mean | p50 | p90 | p95 | p99 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| TTFT (ms) | **54.28** | 53.90 | 57.64 | 58.11 | 59.33 | 49.80 | 59.33 |
| ITL mean per-request (ms) | **10.56** | 10.60 | 10.71 | 10.76 | 10.81 | 10.19 | 10.81 |
| E2E (ms) | 1395.90 | 1402.17 | 1415.73 | 1424.08 | 1429.51 | 1345.21 | 1429.51 |
| output_tps_by_piece (pieces/sec) | 91.72 | 91.29 | 94.06 | 94.43 | 95.15 | 89.54 | 95.15 |

WARM (n=5, cache-hit requests — kept as a cache-effect demonstration):

| Metric | mean | min | max |
|---|---:|---:|---:|
| TTFT (ms) | 15.91 | 15.43 | 16.85 |
| ITL mean per-request (ms) | 10.31 | 10.24 | 10.37 |
| E2E (ms) | 1325.61 | 1316.89 | 1332.62 |

Also summarized the archived first-attempt run with `--no-split`:
`results/stage1/streaming_512_128_singleprompt_summary.{txt,json}`.
That run (single prompt × 50 measured) shows TTFT mean 17.53 ms and
output_pieces=12 across the board, which is the prefix-cache + EOS
refusal artefact pattern.

### 13d — Open items / status going into step 14

1. The summarizer doesn't yet auto-categorize stage 1 vs other runs;
   for step 18 (baseline summary) we'll feed it the three workload
   files (smoke 128/128 has no streaming run, main 512/128 has it, and
   1024/256 is step 17). For now just pass the file explicitly.
2. Server container `trtllm_serve_step11` is **still Up**. Step 14
   needs the server alive (sample nvidia-smi while a streaming run is
   in progress). So keep the container running until step 14 finishes.
3. No new GPU work in step 13 — VRAM unchanged at 7437 / 404 MiB.

---

## Step 14 — GPU memory/utilization capture + trtllm-serve /metrics (2026-05-11)

### 14a — Orchestration script

`scripts/_step14_monitored_bench.sh`: 5 s baseline at 1 Hz → bench 50+5
reqs → 5 s tail at 1 Hz, all written to a single CSV via
`nvidia-smi -l 1`. Concurrent shell launches the bench client against
the still-running `trtllm_serve_step11` server.

### 14b — Two passes were needed

**First pass (cache-warm):** Server had been Up 25 minutes serving step 12,
so its prefix cache was fully populated. Every measured request hit
cache and TTFT mean dropped to 17.74 ms — useful as a cache-hit-during-monitoring
view but not a cold-prefill GPU trace. Archived as:
`results/stage1/benchmark_raw/streaming_512_128_step14_cachewarm.jsonl`
+ `logs/stage1/nvidia_smi_during_streaming_512_128_cachewarm.csv`.

Also discovered that `/metrics` is **drain semantics**: the first call after
step 12 returned 13.8 MB (whole bench history), every subsequent call
returned `[]` until new work arrived. The /metrics from the cache-warm
pass was lost to the drain.

**Second pass (cold cache):**

```bash
docker restart trtllm_serve_step11        # clears KV cache pool
# wait /health (30 s engine reload)
curl -s http://localhost:8000/metrics > /dev/null   # drain any stale
bash scripts/_step14_monitored_bench.sh             # 5+75+5 s window
curl -sS -o results/stage1/benchmark_raw/trtllm_serve_metrics_after_streaming.json \
     http://localhost:8000/metrics                  # 6.5 MB this time
```

This pass overwrote `streaming_512_128.jsonl` with a fresh cold-cache
run (TTFT mean 54.66 ms, within 1 % of step 12's 54.28 ms — well within
run-to-run variance) AND captured the raw `itl_ms` arrays that step 13
made the schema support.

### 14c — Re-summarized streaming data (now with flattened ITL)

```bash
python3 scripts/summarize_streaming_jsonl.py \
  results/stage1/benchmark_raw/streaming_512_128.jsonl \
  --json results/stage1/streaming_512_128_summary.json \
  > results/stage1/streaming_512_128_summary.txt
```

**Headline (COLD, n=45):**

| Metric | mean | p50 | p90 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| TTFT (ms) | 54.66 | 54.48 | 57.18 | 58.41 | 63.43 |
| ITL **per-request** mean (ms) | 10.71 | 10.76 | 10.93 | 10.95 | 11.02 |
| **ITL flattened per-token (ms)** (5715 gaps) | **10.71** | **10.66** | **11.63** | **12.03** | **13.09** |
| E2E (ms) | 1415.55 | 1419.99 | 1444.01 | 1447.47 | 1455.51 |

The flattened p95 (12.03 ms) is **10 % higher** than per-request p95
(10.95 ms) — confirming step 13's prediction that per-request averaging
smooths over real per-token outliers. Flattened p99 13.09 ms vs
per-request p99 11.02 ms is even more divergent.
Flattened min 4.44 ms, max 18.92 ms — there is real per-token jitter.

### 14d — nvidia-smi CSV trajectory (1 Hz)

Total samples: 88. Bench window: 78 samples where util.gpu > 50%.

**Baseline (idle with engine loaded, n=10):**

| Metric | mean | min | max |
|---|---:|---:|---:|
| util.gpu (%) | 20 | 14 | 25 |
| util.mem (%) | (low) | — | — |
| power.draw (W) | 38.55 | 32.80 | 58.07 |
| temp.gpu (C) | 61 | 55 | 72 |
| clocks.sm (MHz) | 1597 | 1560 | 1935 |
| memory.used (MiB) | **7436** | 7435 | 7437 |
| pstate | P0 | — | — |

Idle util is **not zero** — engine iter loop keeps GPU busy at ~20 %
while polling /metrics / health.

**Bench window (n=78):**

| Metric | mean | p95 | min | max |
|---|---:|---:|---:|---:|
| util.gpu (%) | **96.85** | 99 | 72 | 99 |
| util.mem (%) | **82.35** | 88 | 55 | 88 |
| power.draw (W) | **114.28** | 114.88 | 86.78 | 115.09 |
| temp.gpu (C) | 69 | 75 | 59 | 75 |
| clocks.sm (MHz) | 1722 | 1770 | 1620 | 1935 |
| memory.used (MiB) | **7437** | 7437 | 7437 | 7437 |

**Key observations:**

- Decode pegs util.gpu at ~97 %: SMs are nearly always working.
- util.mem (memory controller) at 82 % is the dead giveaway that **decode
  is memory-bound** — each generated token re-reads the full KV cache
  through HBM. This matches the textbook expectation.
- Power 114 W is at the laptop's GPU TDP cap (115 W cTGP). No headroom
  for clock boost beyond 1770 MHz on SM clock.
- Temp 75 °C max is well below the 87 °C throttle threshold — no thermal
  throttling during the 75 s bench.
- memory.used stays flat 7437 MiB. KV cache pool is predetermined at
  server startup via `kv_cache_free_gpu_memory_fraction`; request flow
  does not grow it.

CSV: `logs/stage1/nvidia_smi_during_streaming_512_128.csv` (88 lines,
4 KB).

### 14e — trtllm-serve /metrics mining

6.5 MB JSON, 7095 iter records. Each record has `iterLatencyMS`,
`kvCacheStats`, `inflightBatchingStats`. We extracted:

| Quantity | Value | Interpretation |
|---|---|---|
| total iter records | 7095 | ≈ 55 ctx + 7040 decode (55 req × 128 token) — matches expectation |
| context iters | 55 | one per request (chunked prefill not triggered for these sizes) |
| generation iters | 6985 | per-token decode steps |
| **decode iterLatencyMS p50** | **10.75 ms** | matches client-side ITL p50 10.66 ms within 0.1 ms — the engine inner loop **is** the ITL, no extra server-side overhead |
| **context iterLatencyMS p50** | **10.87 ms** | prefill 466 tokens in ~11 ms — sub-linear because of the system-prompt prefix cache hit |
| cacheHitRate steady-state | 0.16–0.23 | not 0 even on a fresh server: Qwen chat-template system-prompt prefix (~46 tokens) is shared across all 50 prompts and gets cached after request #0 |
| reusedBlocks > 0 iters | 6966 / 7095 | almost every iter reuses at least one cached block; cache reuse is woven into normal decode operation |
| numCtxTokens distribution top | {465, 395, 433, 466, 459} | NOT the literal 512 — system-prompt prefix is removed from the "new context" count when cache hits |

This is the first step in stage 1 where we can correlate
**client-perceived ITL** (10.66 ms p50) directly with **engine inner-loop
latency** (10.75 ms p50). The HTTP/SSE layer adds < 0.1 ms per token.

Saved as: `results/stage1/benchmark_raw/trtllm_serve_metrics_after_streaming.json`
(6.5 MB — kept locally, not committed; see open items).

### 14f — Open items going into step 15

1. **Cold "first request" cost is hidden in averages.** Request 0
   TTFT 158.71 ms vs cold-mean 54.66 ms — that's engine first-iter JIT +
   kernel autotuner caches warming up. nsys (step 15) should be configured
   to skip the first 1-2 requests as warmup to avoid the trace being
   dominated by JIT noise.
2. **Should `/metrics` JSON be committed?** 6.5 MB compresses but is not
   git-friendly. Recommend: gzip and commit, or skip and leave it as
   regenerable from a re-run. step 18 baseline will reference the
   summary numbers (which are tiny) — the raw is only useful for ad-hoc
   re-analysis.
3. **Higher-resolution GPU sampling.** 1 Hz can't see per-token GPU
   pauses (each token is ~10 ms). If step 15 nsys doesn't fully cover
   the question "is there a gap between tokens", switch nvidia-smi to
   `--lms` 50 ms in a follow-up run. For now nsys is the right tool.
4. Server container `trtllm_serve_step11` is still Up (engine reload
   was 30 s, cache is now warmed by the cold-cache run). step 15
   does its own bench inside `nsys profile`, so it does NOT use this
   server — safe to `docker stop trtllm_serve_step11` after step 14
   to free 7.4 GiB VRAM for the nsys/trtllm-bench subprocess.

---

## Step 15 — Nsight Systems profile of 512/128 latency (2026-05-11)

### 15a — Tear-down + tool probe

```bash
docker stop trtllm_serve_step11 && docker rm trtllm_serve_step11
# VRAM: 176 / 7665 MiB (server released its 7.4 GiB)

docker run --rm --gpus all nvcr.io/nvidia/tensorrt-llm/release:1.0.0 \
  bash -c "nsys --version; ncu --version"
# nsys 2025.3.1.90-253135822126v0
# ncu  2025.2.1.0 (build 35987062)
```

Both Nsight tools are present in the 1.0.0 container, no host install needed.

### 15b — nsys profile command

`scripts/_step15_nsys.sh` runs **inside** a profiling-permission container:

```bash
docker run --rm \
  --gpus all --ipc host \
  --cap-add=SYS_ADMIN --security-opt seccomp=unconfined \
  --memory 10g --memory-swap 12g --shm-size 2g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD":/workspace/rtx3070-trtllm-latency-lab \
  -w /workspace/rtx3070-trtllm-latency-lab \
  nvcr.io/nvidia/tensorrt-llm/release:1.0.0 \
  bash scripts/_step15_nsys.sh
```

The script wraps `trtllm-bench latency` (20 measured + 5 warmup) with:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=cpu --force-overwrite=true \
  --output profiles/stage1/nsys/qwen25_15b_512_128_latency \
  trtllm-bench --model "$HF_MODEL" --workspace "$WORKSPACE_DIR" \
    latency --backend tensorrt \
    --dataset data/synthetic_512_128.jsonl \
    --engine_dir "$ENGINE_512_128" \
    --concurrency 1 --num_requests 20 --warmup 5 \
    --report_json results/stage1/benchmark_raw/trtllm_bench_512_128_latency_nsys.json
```

Trace flags meaning:

| Flag | Captures |
|---|---|
| `cuda` | All CUDA API calls + every kernel + memcpy/memset on the GPU timeline |
| `nvtx` | TRT-LLM's NVTX ranges with semantic names (layer/op/plugin) |
| `osrt` | libc/OS syscalls — useful for spotting CPU-side blocking |
| `--sample=cpu` | Periodic CPU thread sampling so flamegraphs of host code work |

### 15c — Outputs

```
profiles/stage1/nsys/qwen25_15b_512_128_latency.nsys-rep   38.6 MB
profiles/stage1/nsys/qwen25_15b_512_128_latency.sqlite     200  MB   (auto-generated by nsys stats)
profiles/stage1/nsys/nsys_summary_512_128.txt              190  KB
results/stage1/benchmark_raw/trtllm_bench_512_128_latency_nsys.json
logs/stage1/nsys_512_128_latency.log
```

The `.nsys-rep` is the GUI-loadable trace. Neither it nor `.sqlite` are
git-friendly; commit only the text summary (`nsys_summary_512_128.txt`)
and the bench JSON.

### 15d — Bench-level numbers (cross-check vs step 10/12)

| Metric | step 10 (`trtllm-bench` plain) | step 12 (streaming COLD) | **step 15 (under nsys)** | Comment |
|---|---:|---:|---:|---|
| E2E avg ms | 1302 | 1396 | **1392** | nsys adds ~7 % over plain bench; matches streaming |
| TTFT avg ms | 51.77 | 54.28 | **53.04** | well within run-to-run variance |
| TPOT/ITL avg ms | 9.85 | 10.71 | **10.54** | same |
| TPOT p95 ms | 10.00 | 10.95 (per-req) | **10.74** | step 15 reports its own TPOT inside the bench json |
| output throughput / user (tok/s) | 98.30 | 91.72 | 94.84 | nsys overhead taxes throughput slightly |

The streaming `streaming_metrics` block inside the step 15 bench JSON
even reports the same TTFT/TPOT percentiles, confirming `trtllm-bench
latency` measures from inside the engine (in-process) the same numbers
that step 12 measured from the HTTP client side.

### 15e — Kernel time breakdown (top hits)

Total GPU kernel time (from `cuda_gpu_kern_sum`):

| Category | Time % | Notes |
|---|---:|---|
| **GEMM (all sm80_xmma_gemm_* variants)** | **~61.5 %** | dominant work; multiple tile-size variants because TRT picks a different tile for each layer's matrix shape |
| **Sampling head** | **~31.2 %** | `batchApplyPenalty` 14.2 %, `addBiasSoftMax` 12.7 % (151 936-wide vocab softmax), `topKStage1/2` 4.3 % |
| Activation/Norm fusion | ~3.5 % | `SiluMul`, RMSNorm fused kernels |
| **Attention** | **~3.0 %** | Flash Attn v2 prefill 1.9 % + masked-MHA decode 1.1 % |
| KV cache update / RoPE | ~0.4 % | `applyBiasRopeUpdateKVCacheV2` |
| Misc (copy, fill, length checks) | ~0.4 % | tail of the histogram |

**Three findings worth pinning down for step 16:**

1. **The single largest kernel is `sm80_xmma_gemm_bf16…tile256x128x32`
   at 24.3 %.** 700 instances, 874 μs each — this is one of the
   transformer's MLP up/gate projections (the biggest 2D matmul in the
   model). Step 16 (ncu) should focus a launch range on this kernel.
2. **Sampling > Attention by 10×.** The 151 936-token softmax + topK
   over the LM head is a much bigger cost than all attention combined
   for this small 1.5 B model. People assume "attention dominates" —
   that's wrong at this model size.
3. **MMHA (decode attention) is 1.1 %.** With GQA num_kv_heads=2 and
   max_seq_len=640, each decode step processes a 2-headed attention over
   ≤640 cached tokens — small workload, ~19 μs per instance.

### 15f — CUDA API breakdown (CPU/GPU sync pattern)

`cuda_api_sum` is **91.2 % cudaEventSynchronize**, 3200 calls, ~10 ms avg.

That count (3200) is exactly `25 requests × 128 tokens`. Each decode
iter ends with a `cudaEventSynchronize` for the host driver thread to
wait for the GPU to publish the next token. **CPU is fully blocking on
the GPU during decode** — there's no useful prep work the CPU does in
parallel because each step depends on the previous token's argmax.

This means: shaving CPU-side overhead (Python, scheduler, etc.) **will
not improve TPOT** for single-user decode. Only making the GPU portion
of an iter faster (e.g. quantization, faster attention, batching) moves
the needle.

### 15g — Memory transfer summary

| Direction | Total (MB) | Count | Avg (MB) | Note |
|---|---:|---:|---:|---|
| Host→Device | 3105 | 35 327 | 0.088 | dominated by one 3087 MB transfer = engine load |
| Device→Host | 0.042 | 12 804 | 0.000 | output token ids only |
| Device→Device | 0.032 | 12 800 | 0.000 | tiny tensor reshapes |
| CUDA memset | 3117 | 975 | 3.20 | KV cache pool initialization (3087 MB single memset) |

After engine load, **inference moves essentially zero data to/from host**.
That's the textbook expectation but worth confirming on the trace.

### 15h — NVTX surprise: engine load dominates wall time

NVTX top ranges:

| Time % | Range | Note |
|---:|---|---|
| 34.6 % | `myelinGraphDeserializeBinary` | engine read from disk + plan parsing |
| 23.7 % | `myelin-exec:myelinGraphLoadPersistent` | TRT plugin/kernel load to GPU |
| 15.2 % | `TensorRT:ExecutionContext::enqueue` | the actual inference enqueues |
| 5.0 %  | `myelin-exec:myelinGraphExecute` | per-call execution wrapper |
| 2.0 %  | `myelinGraphUnloadPersistent` | engine tear-down |

**Engine load + unload is ~60 % of the trace's NVTX time** (~30 s in
wall time before the bench's first request). For step 16 ncu we will
isolate a launch range starting well after engine load, otherwise
ncu will spend most of its expensive replay time on TRT framework setup.

### 15i — Open items going into step 16

1. **Target kernel for ncu**: `sm80_xmma_gemm_bf16…tile256x128x32` —
   single biggest contributor (24.3 % of GPU time). Second target:
   `addBiasSoftMax` (12.7 %), since the 151 936-wide softmax is unusual
   and worth understanding the memory access pattern of.
2. **Launch skip**: 25 requests × ~50+ kernel launches per request +
   engine load. Use `--launch-skip 200 --launch-count 10` to land on
   measured-phase kernels, not init phase.
3. `.nsys-rep` (38.6 MB) and `.sqlite` (200 MB) **excluded from commit**
   — too large and easy to regenerate. The text summary, the bench
   JSON, and this command log are sufficient deliverables.
4. step 16 should be a fresh container run (release after step 15
   finishes will set baseline VRAM back to 176 MiB).

