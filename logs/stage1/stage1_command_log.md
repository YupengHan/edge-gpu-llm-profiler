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
