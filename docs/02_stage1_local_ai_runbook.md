# 02 — Stage 1 local AI runbook

This runbook is written for a local AI coding/execution agent. It should be followed in order.

Stage 1 scope:

```text
Model: Qwen/Qwen2.5-1.5B-Instruct
Backend: TensorRT-LLM
Precision: FP16 / no explicit quantization
Hardware: RTX 3070-class local GPU
Scenario: single user, concurrency=1, batch size=1
Primary workload: 512 input tokens / 128 output tokens
```

## Execution discipline

For every major command:

1. Save the command to `logs/stage1/stage1_command_log.md`.
2. Save stdout/stderr to a log file under `logs/stage1/` or `logs/stage0/`.
3. If the command fails, do not skip forward. Record the failure and reason.
4. If the CLI syntax differs from this runbook, run `--help`, update the command, and record the difference.
5. Keep Stage 1 on TensorRT-LLM unless the runbook explicitly says to stop.

Use this helper pattern:

```bash
log_cmd () {
  echo '\n```bash' >> logs/stage1/stage1_command_log.md
  echo "$*" >> logs/stage1/stage1_command_log.md
  echo '```' >> logs/stage1/stage1_command_log.md
  "$@"
}
```

If the shell does not support this helper, manually copy commands into the log.

---

## Step 0 — Create directories

From the repo root:

```bash
mkdir -p \
  results/stage0 \
  results/stage1/benchmark_raw \
  profiles/stage1/nsys \
  profiles/stage1/ncu \
  logs/stage0 \
  logs/stage1 \
  data \
  engines \
  models \
  scripts \
  external

cat > logs/stage1/stage1_command_log.md <<'LOG'
# Stage 1 command log

LOG
```

Expected result:

```text
All directories exist.
```

---

## Step 1 — Record hardware and host environment

Run:

```bash
nvidia-smi | tee logs/stage0/nvidia_smi.txt
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,memory.free --format=csv | tee logs/stage0/gpu_query.csv
uname -a | tee logs/stage0/uname.txt
lsb_release -a 2>&1 | tee logs/stage0/lsb_release.txt || true
docker --version 2>&1 | tee logs/stage0/docker_version.txt || true
nsys --version 2>&1 | tee logs/stage0/nsys_version_host.txt || true
ncu --version 2>&1 | tee logs/stage0/ncu_version_host.txt || true
```

Create `results/stage0/environment_report.md`:

```markdown
# Stage 0 environment report

## Hardware

- GPU:
- VRAM:
- Compute capability:
- Driver version:

## Host

- OS:
- Docker:
- NVIDIA Container Toolkit:
- Nsight Systems:
- Nsight Compute:

## Compatibility status

- TensorRT-LLM container tested: yes/no
- TensorRT-LLM import tested: yes/no
- `trtllm-bench` available: yes/no
- `trtllm-serve` available: yes/no

## Notes
```

Stop condition:

```text
If `nvidia-smi` does not show the RTX 3070, stop. Do not continue to Docker/TRT-LLM.
```

---

## Step 2 — Verify Docker GPU access

Use the TensorRT-LLM release container path first. The official docs recommend the pre-built NGC container as the simplest installation path.

Choose a TensorRT-LLM image tag from the NGC TensorRT-LLM release page. The current documentation example uses `1.3.0rc14`, but the local agent should verify the latest available tag if possible.

Set:

```bash
export TRTLLM_TAG=1.3.0rc14
export TRTLLM_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:${TRTLLM_TAG}
```

Pull:

```bash
docker pull ${TRTLLM_IMAGE} | tee logs/stage0/docker_pull_trtllm.log
```

Run a simple GPU-visible shell:

```bash
docker run --rm --gpus all ${TRTLLM_IMAGE} nvidia-smi | tee logs/stage0/docker_nvidia_smi.txt
```

If profiling from inside the container later fails, rerun the container with additional profiling permissions:

```bash
docker run --rm -it \
  --ipc host \
  --gpus all \
  --cap-add=SYS_ADMIN \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8000:8000 \
  -v "$PWD":/workspace/rtx3070-trtllm-latency-lab \
  -w /workspace/rtx3070-trtllm-latency-lab \
  ${TRTLLM_IMAGE}
```

Expected result:

```text
The RTX 3070 is visible inside the container.
```

Stop condition:

```text
If the GPU is not visible inside Docker, stop and document NVIDIA Container Toolkit setup issue.
```

---

## Step 3 — Start the project container

Start an interactive container mounted to the repo:

```bash
docker run --rm -it \
  --ipc host \
  --gpus all \
  --cap-add=SYS_ADMIN \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8000:8000 \
  -v "$PWD":/workspace/rtx3070-trtllm-latency-lab \
  -w /workspace/rtx3070-trtllm-latency-lab \
  ${TRTLLM_IMAGE}
```

Inside the container, re-run basic checks:

```bash
python3 - <<'PY' | tee logs/stage0/trtllm_python_import.txt
import sys
print("python", sys.version)
try:
    import torch
    print("torch", torch.__version__)
    print("cuda available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
        print("capability", torch.cuda.get_device_capability(0))
except Exception as e:
    print("torch_error", repr(e))
try:
    import tensorrt_llm
    print("tensorrt_llm", getattr(tensorrt_llm, "__version__", "unknown"))
except Exception as e:
    print("tensorrt_llm_error", repr(e))
PY

trtllm-bench --help 2>&1 | tee logs/stage0/trtllm_bench_help.txt
trtllm-serve --help 2>&1 | tee logs/stage0/trtllm_serve_help.txt
nsys --version 2>&1 | tee logs/stage0/nsys_version_container.txt || true
ncu --version 2>&1 | tee logs/stage0/ncu_version_container.txt || true
```

Expected result:

```text
- torch sees CUDA.
- TensorRT-LLM imports.
- trtllm-bench is available.
- trtllm-serve is available.
```

Stop condition:

```text
If TensorRT-LLM import fails, do not continue. Document exact failure.
```

---

## Step 4 — Verify model/tokenizer access

Set:

```bash
export HF_MODEL=Qwen/Qwen2.5-1.5B-Instruct
```

If Hugging Face authentication is needed:

```bash
export HF_TOKEN=<your_token_here>
```

Verify tokenizer:

```bash
python3 - <<'PY' | tee logs/stage1/qwen_tokenizer_check.txt
from transformers import AutoTokenizer
model = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model)
print("model", model)
print("vocab_size", len(tok))
print("eos_token", tok.eos_token)
print("eos_token_id", tok.eos_token_id)
text = "Hello, benchmark test."
ids = tok.encode(text)
print("sample_tokens", len(ids), ids[:20])
PY
```

Optional metadata check:

```bash
python3 - <<'PY' | tee logs/stage1/qwen_config_check.txt
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
for k in ["model_type", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "hidden_size", "vocab_size", "max_position_embeddings"]:
    print(k, getattr(cfg, k, None))
PY
```

Expected result:

```text
Tokenizer and model config load successfully.
```

---

## Step 5 — Obtain TensorRT-LLM benchmark dataset script

`trtllm-bench` is designed to work with TensorRT-LLM's `benchmarks/cpp/prepare_dataset.py` script. If the script is not available inside the container, clone the TensorRT-LLM repo into `external/` for scripts only.

Try to find it:

```bash
python3 - <<'PY' | tee logs/stage1/prepare_dataset_search.txt
import glob, os
roots = ["/app", "/workspace", "/opt", "/usr/local", "."]
for root in roots:
    matches = glob.glob(os.path.join(root, "**", "benchmarks", "cpp", "prepare_dataset.py"), recursive=True)
    for m in matches[:10]:
        print(m)
PY
```

If no path appears, clone:

```bash
git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM external/TensorRT-LLM | tee logs/stage1/clone_trtllm_scripts.log
export PREPARE_DATASET=external/TensorRT-LLM/benchmarks/cpp/prepare_dataset.py
```

If a path appears, set it manually:

```bash
export PREPARE_DATASET=<path_to_prepare_dataset.py>
```

Verify help:

```bash
python3 ${PREPARE_DATASET} --help 2>&1 | tee logs/stage1/prepare_dataset_help.txt
```

Important:

```text
The current docs show `token_norm_dist`, while some older examples show `token-norm-dist`. Use whatever the local `--help` reports, and record the accepted spelling.
```

---

## Step 6 — Generate synthetic datasets

Target datasets:

| Name | Input mean | Output mean | Requests | Purpose |
|---|---:|---:|---:|---|
| `synthetic_128_128.jsonl` | 128 | 128 | 50 | Smoke test |
| `synthetic_512_128.jsonl` | 512 | 128 | 100 | Main benchmark |
| `synthetic_1024_256.jsonl` | 1024 | 256 | 30 | Stretch/profiling-heavy |

Use deterministic token lengths by setting stdev to zero:

```bash
python3 ${PREPARE_DATASET} \
  --stdout \
  --tokenizer ${HF_MODEL} \
  token_norm_dist \
  --input-mean 128 \
  --output-mean 128 \
  --input-stdev 0 \
  --output-stdev 0 \
  --num-requests 50 \
  > data/synthetic_128_128.jsonl

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

python3 ${PREPARE_DATASET} \
  --stdout \
  --tokenizer ${HF_MODEL} \
  token_norm_dist \
  --input-mean 1024 \
  --output-mean 256 \
  --input-stdev 0 \
  --output-stdev 0 \
  --num-requests 30 \
  > data/synthetic_1024_256.jsonl
```

If `token_norm_dist` fails, try the hyphenated spelling reported by local help:

```bash
# Example fallback only if help says this is correct:
# token-norm-dist
```

Validate files:

```bash
wc -l data/synthetic_*.jsonl | tee logs/stage1/dataset_line_counts.txt
head -n 1 data/synthetic_512_128.jsonl | tee logs/stage1/dataset_sample_512_128.txt
```

Expected result:

```text
- 50 lines for 128/128.
- 100 lines for 512/128.
- 30 lines for 1024/256.
```

---

## Step 7 — Build TensorRT-LLM engine for smoke test

Start with the small smoke case.

```bash
export WORKSPACE_DIR=trtllm_workspace
mkdir -p ${WORKSPACE_DIR} engines/stage1

trtllm-bench \
  --model ${HF_MODEL} \
  --workspace ${WORKSPACE_DIR} \
  --log_level info \
  build \
  --dataset data/synthetic_128_128.jsonl \
  --max_batch_size 1 \
  --max_num_tokens 256 \
  --max_seq_len 256 \
  --tp_size 1 \
  --pp_size 1 \
  2>&1 | tee logs/stage1/build_128_128.log
```

Find engine directory:

```bash
find ${WORKSPACE_DIR} -type d -name "tp_1_pp_1" -print | tee logs/stage1/engine_dirs_after_smoke_build.txt
```

Set the engine dir to the printed path:

```bash
export ENGINE_128_128=<printed_engine_dir>
```

Expected result:

```text
TensorRT-LLM engine directory exists.
```

If build fails:

1. Record the exact error.
2. Check whether the error is model support, GPU support, memory, or CLI syntax.
3. If model support is the issue, try `TinyLlama/TinyLlama-1.1B-Chat-v1.0` only as an environment smoke test and write a compatibility note.
4. Do not switch the whole project to TinyLlama unless the owner approves.

---

## Step 8 — Run smoke latency benchmark

```bash
trtllm-bench \
  --model ${HF_MODEL} \
  --workspace ${WORKSPACE_DIR} \
  latency \
  --backend tensorrt \
  --dataset data/synthetic_128_128.jsonl \
  --engine_dir ${ENGINE_128_128} \
  --concurrency 1 \
  --num_requests 20 \
  --warmup 5 \
  --report_json results/stage1/benchmark_raw/trtllm_bench_128_128_latency.json \
  --iteration_log results/stage1/benchmark_raw/trtllm_bench_128_128_iteration.jsonl \
  2>&1 | tee logs/stage1/latency_128_128.log
```

Expected result:

```text
A JSON report exists under results/stage1/benchmark_raw/.
```

---

## Step 9 — Build TensorRT-LLM engine for main 512/128 benchmark

```bash
trtllm-bench \
  --model ${HF_MODEL} \
  --workspace ${WORKSPACE_DIR} \
  --log_level info \
  build \
  --dataset data/synthetic_512_128.jsonl \
  --max_batch_size 1 \
  --max_num_tokens 640 \
  --max_seq_len 640 \
  --tp_size 1 \
  --pp_size 1 \
  2>&1 | tee logs/stage1/build_512_128.log
```

Find engine directory:

```bash
find ${WORKSPACE_DIR} -type d -name "tp_1_pp_1" -print | tee logs/stage1/engine_dirs_after_main_build.txt
```

If multiple `tp_1_pp_1` directories exist, choose the newest one or inspect logs to identify the 512/128 build output.

Set:

```bash
export ENGINE_512_128=<printed_engine_dir>
```

---

## Step 10 — Run official `trtllm-bench` main benchmark

Latency:

```bash
trtllm-bench \
  --model ${HF_MODEL} \
  --workspace ${WORKSPACE_DIR} \
  latency \
  --backend tensorrt \
  --dataset data/synthetic_512_128.jsonl \
  --engine_dir ${ENGINE_512_128} \
  --concurrency 1 \
  --num_requests 100 \
  --warmup 10 \
  --report_json results/stage1/benchmark_raw/trtllm_bench_512_128_latency.json \
  --iteration_log results/stage1/benchmark_raw/trtllm_bench_512_128_iteration.jsonl \
  2>&1 | tee logs/stage1/latency_512_128.log
```

Throughput with concurrency 1:

```bash
trtllm-bench \
  --model ${HF_MODEL} \
  --workspace ${WORKSPACE_DIR} \
  throughput \
  --backend tensorrt \
  --dataset data/synthetic_512_128.jsonl \
  --engine_dir ${ENGINE_512_128} \
  --concurrency 1 \
  --num_requests 100 \
  --warmup 10 \
  --report_json results/stage1/benchmark_raw/trtllm_bench_512_128_throughput.json \
  --iteration_log results/stage1/benchmark_raw/trtllm_bench_512_128_throughput_iteration.jsonl \
  --request_json results/stage1/benchmark_raw/trtllm_bench_512_128_requests.jsonl \
  2>&1 | tee logs/stage1/throughput_512_128.log
```

Expected result:

```text
Official benchmark JSON reports exist.
```

Note:

```text
`trtllm-bench` may not expose TTFT/ITL exactly as required. Use it as official backend benchmark output, but use the streaming benchmark client for TTFT/ITL.
```

---

## Step 11 — Start `trtllm-serve` for streaming TTFT/ITL

The current `trtllm-serve` syntax supports `serve [OPTIONS] MODEL`, where `MODEL` can be a model name, Hugging Face checkpoint path, or TensorRT engine path. Use the TensorRT engine path if available.

Start server:

```bash
trtllm-serve serve \
  --host 0.0.0.0 \
  --port 8000 \
  --backend tensorrt \
  --tokenizer ${HF_MODEL} \
  --max_batch_size 1 \
  --max_num_tokens 640 \
  --max_seq_len 640 \
  ${ENGINE_512_128} \
  2>&1 | tee logs/stage1/trtllm_serve_512_128.log
```

If the command syntax fails:

```bash
trtllm-serve serve --help 2>&1 | tee logs/stage1/trtllm_serve_serve_help.txt
```

Then adjust syntax and document the change.

Health check from another terminal inside the same container or host:

```bash
curl -s http://localhost:8000/health | tee logs/stage1/server_health.txt
curl -s http://localhost:8000/v1/models | tee logs/stage1/server_models.txt
```

Minimal chat test:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 16,
    "temperature": 0
  }' | tee logs/stage1/server_chat_test.json
```

If the model name must equal the engine path or another served name, inspect `/v1/models` and update the benchmark client accordingly.

---

## Step 12 — Implement streaming TTFT/ITL benchmark client

Create:

```text
scripts/benchmark_streaming_openai.py
```

Minimum requirements:

- Sends one request at a time.
- Uses streaming mode.
- Records:
  - request start time.
  - first token time.
  - every subsequent token arrival time.
  - done time.
  - generated token count if available.
- Computes:
  - TTFT ms.
  - mean ITL ms.
  - p50/p90/p95 ITL ms.
  - E2E latency ms.
  - output tokens/sec/user.
- Writes JSONL to `results/stage1/benchmark_raw/streaming_512_128.jsonl`.

Suggested implementation skeleton:

```python
#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from pathlib import Path

import requests


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    idx = int(round((len(values) - 1) * q))
    return values[idx]


def run_one(base_url, model, prompt, max_tokens, temperature):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    t0 = time.perf_counter()
    token_times = []
    text_parts = []

    with requests.post(url, json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for raw_line in r.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data.strip() == "[DONE]":
                break
            obj = json.loads(data)
            delta = obj.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            if content:
                now = time.perf_counter()
                token_times.append(now)
                text_parts.append(content)

    t_done = time.perf_counter()

    ttft_ms = None
    if token_times:
        ttft_ms = (token_times[0] - t0) * 1000

    gaps_ms = [
        (token_times[i] - token_times[i - 1]) * 1000
        for i in range(1, len(token_times))
    ]

    e2e_ms = (t_done - t0) * 1000
    output_pieces = len(token_times)
    output_tps = output_pieces / (e2e_ms / 1000) if e2e_ms > 0 else None

    return {
        "ttft_ms": ttft_ms,
        "itl_ms_mean": statistics.mean(gaps_ms) if gaps_ms else None,
        "itl_ms_p50": percentile(gaps_ms, 0.50),
        "itl_ms_p90": percentile(gaps_ms, 0.90),
        "itl_ms_p95": percentile(gaps_ms, 0.95),
        "e2e_ms": e2e_ms,
        "output_pieces": output_pieces,
        "output_tps_by_piece": output_tps,
        "text_preview": "".join(text_parts)[:200],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--requests", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(args.requests + args.warmup):
            row = run_one(args.base_url, args.model, args.prompt, args.max_tokens, args.temperature)
            row["request_index"] = i
            row["warmup"] = i < args.warmup
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(row)


if __name__ == "__main__":
    main()
```

Install dependencies if necessary:

```bash
python3 -m pip install requests
```

Run:

```bash
python3 scripts/benchmark_streaming_openai.py \
  --base-url http://localhost:8000/v1 \
  --model "Qwen/Qwen2.5-1.5B-Instruct" \
  --prompt "Write a concise explanation of why GPU profiling matters for LLM inference. Use clear technical language." \
  --max-tokens 128 \
  --requests 50 \
  --warmup 5 \
  --output results/stage1/benchmark_raw/streaming_512_128.jsonl \
  2>&1 | tee logs/stage1/streaming_512_128.log
```

If the served model name differs, replace `--model` with the value from `/v1/models`.

---

## Step 13 — Summarize streaming results

Create:

```text
scripts/summarize_streaming_jsonl.py
```

Minimum summary:

- TTFT p50/p90/p95/mean.
- ITL mean/p50/p90/p95.
- E2E mean/p50/p90/p95.
- output pieces/sec mean.

Run:

```bash
python3 scripts/summarize_streaming_jsonl.py \
  results/stage1/benchmark_raw/streaming_512_128.jsonl \
  > results/stage1/streaming_512_128_summary.txt
```

If the summarizer is not implemented yet, manually parse JSONL with Python and write the same metrics.

---

## Step 14 — Capture GPU memory/utilization

During one benchmark run, capture `nvidia-smi`:

```bash
nvidia-smi \
  --query-gpu=timestamp,name,pstate,temperature.gpu,power.draw,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,memory.used,memory.free \
  --format=csv \
  -l 1 \
  > logs/stage1/nvidia_smi_during_streaming_512_128.csv
```

Stop it after the benchmark finishes.

If `trtllm-serve` metrics endpoint is available:

```bash
curl -s http://localhost:8000/metrics | tee results/stage1/benchmark_raw/trtllm_serve_metrics_after_streaming.json
```

---

## Step 15 — Run Nsight Systems

For a representative latency run, use `nsys profile`.

Prefer a smaller request count to keep the trace manageable:

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
    --report_json results/stage1/benchmark_raw/trtllm_bench_512_128_latency_nsys.json \
  2>&1 | tee logs/stage1/nsys_512_128_latency.log
```

Generate stats:

```bash
nsys stats profiles/stage1/nsys/qwen25_15b_512_128_latency.nsys-rep \
  > profiles/stage1/nsys/nsys_summary_512_128.txt \
  2> logs/stage1/nsys_stats_512_128.err || true
```

If `.nsys-rep` is too large, do not commit it. Commit the text summary and a short observation note.

---

## Step 16 — Run Nsight Compute on selected kernels

Do not run full Nsight Compute on the entire benchmark unless necessary; it can be very slow. Start with a small launch range.

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
    --warmup 2 \
  2>&1 | tee logs/stage1/ncu_512_128_selected.log
```

Write an observation note:

```text
profiles/stage1/ncu/ncu_observation_001.md
```

Use the template in:

```text
docs/templates/profile_observation_template.md
```

Minimum observation:

```text
- Which kernel(s) were profiled?
- What is the top-level bottleneck classification: memory-bound, compute-bound, launch overhead, synchronization, or unclear?
- What metric supports this hypothesis?
- What is the next optimization knob?
```

---

## Step 17 — Optional stretch benchmark: 1024/256

Only run if 512/128 succeeds and VRAM is stable.

Build:

```bash
trtllm-bench \
  --model ${HF_MODEL} \
  --workspace ${WORKSPACE_DIR} \
  --log_level info \
  build \
  --dataset data/synthetic_1024_256.jsonl \
  --max_batch_size 1 \
  --max_num_tokens 1280 \
  --max_seq_len 1280 \
  --tp_size 1 \
  --pp_size 1 \
  2>&1 | tee logs/stage1/build_1024_256.log
```

Run latency only:

```bash
export ENGINE_1024_256=<printed_engine_dir>

trtllm-bench \
  --model ${HF_MODEL} \
  --workspace ${WORKSPACE_DIR} \
  latency \
  --backend tensorrt \
  --dataset data/synthetic_1024_256.jsonl \
  --engine_dir ${ENGINE_1024_256} \
  --concurrency 1 \
  --num_requests 30 \
  --warmup 5 \
  --report_json results/stage1/benchmark_raw/trtllm_bench_1024_256_latency.json \
  2>&1 | tee logs/stage1/latency_1024_256.log
```

---

## Step 18 — Write Stage 1 baseline summary

Create:

```text
results/stage1/baseline_summary.md
```

Required sections:

```markdown
# Stage 1 baseline summary

## Environment

## TensorRT-LLM compatibility result

## Model

## Workloads

## Official trtllm-bench results

## Streaming TTFT/ITL results

## VRAM observations

## Nsight Systems observations

## Nsight Compute observations

## Bottleneck hypothesis

## First optimization candidate

## Open issues
```

Do not overclaim. If metrics are unstable, say so and include the likely reason.

---

## Step 19 — Update README

Add/update these README sections:

```markdown
## Current status

## Stage 1 baseline results

| Model | Backend | Precision | Workload | Concurrency | TTFT p50 | ITL mean | E2E p50 | Peak VRAM | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|

## First profiling findings

## Next steps
```

---

## Stage 1 completion checklist

- [ ] `results/stage0/environment_report.md` exists.
- [ ] TensorRT-LLM import/version recorded.
- [ ] Qwen tokenizer/model config check recorded.
- [ ] Synthetic 128/128 and 512/128 datasets generated.
- [ ] 128/128 smoke run completed or failure documented.
- [ ] 512/128 main run completed or failure documented.
- [ ] Streaming TTFT/ITL benchmark completed or failure documented.
- [ ] GPU memory observations recorded.
- [ ] nsys summary generated or failure documented.
- [ ] ncu observation generated or failure documented.
- [ ] `results/stage1/baseline_summary.md` written.
- [ ] README updated.
