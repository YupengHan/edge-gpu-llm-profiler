# Stage 0 environment report

Captured: 2026-05-07T19:58 -07:00 (host: `Alienware`)

## Hardware

- GPU: NVIDIA GeForce RTX 3070 Laptop GPU
- VRAM: 8192 MiB (free at capture: 7391 MiB)
- Compute capability: 8.6 (sm_86, Ampere consumer / mobile)
- Driver version: 575.57.08
- CUDA runtime advertised by driver: 12.9

Active GPU processes at capture (from `logs/stage0/nvidia_smi.txt`):

- `/usr/lib/xorg/Xorg` (PID 2814): 133 MiB — desktop compositor, expected on a laptop.
- `/proc/self/exe` (PID 9917): 293 MiB — unidentified Compute process pre-existing on GPU. Will close before benchmarking if it persists.

## Host

- OS: Ubuntu 22.04.5 LTS (Jammy), kernel `6.8.0-110-generic`
- Docker: `Docker version 28.1.1, build 4eba377`
- NVIDIA Container Toolkit: `1.17.8` (`nvidia-container-runtime` registered in `/etc/docker/daemon.json`)
- Nsight Systems (host): `2025.5.1.121-255136380782v0`
- Nsight Compute (host): `2024.1.1.0 (build 33998838)`

## Compatibility status

- TensorRT-LLM container tested: yes — `nvcr.io/nvidia/tensorrt-llm/release:1.0.0`.
- TensorRT-LLM import tested: yes — `tensorrt_llm 1.0.0` imports inside the 1.0.0 container.
- `trtllm-bench` available: yes (1.0.0 container).
- `trtllm-serve` available: yes (1.0.0 container).
- `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14` import: **FAILED** on this host (driver 575.57.08 / CUDA 12.9). PyTorch in 1.3.0rc14 is `2.11.0a0+nv26.02` and requires the CUDA 13.x driver ABI; `torch.cuda.is_available()` raises `RuntimeError: The NVIDIA driver on your system is too old (found version 12090)`. Image was pulled but unusable here; removed to free disk for the fallback. Full failure transcript at `logs/stage1/step3_container_probe.log`.

## Working container (Stage 1)

- Image: `nvcr.io/nvidia/tensorrt-llm/release:1.0.0`
- python: 3.12.3
- torch: `2.8.0a0+5228986c39.nv25.06` (NV PyTorch 25.06, CUDA 12.x ABI)
- torch.cuda: available, `device_count=1`, `get_device_name(0)='NVIDIA GeForce RTX 3070 Laptop GPU'`, capability `(8, 6)`
- tensorrt: `10.11.0.33`
- tensorrt_llm: `1.0.0`
- transformers: `4.53.1`
- pydantic: `2.11.5`
- nsys (container): `2025.3.1.90-253135822126v0`
- ncu (container): `2025.2.1.0 (build 35987062)`
- flashinfer JIT in use (no prebuilt kernels for this driver/arch combo on first import).

## Notes

- The GPU is an **RTX 3070 Laptop GPU**, not desktop. VRAM is 8 GB. The runbook scope ("RTX 3070-class local GPU") applies; performance numbers should be labeled `RTX 3070 Laptop` to avoid being confused with the 8 GB desktop variant.
- sm_86 is a consumer Ampere arch. Some TensorRT-LLM kernel paths are tuned for datacenter Ampere/Hopper; if engine build fails, follow runbook's compatibility-failure path rather than switching backend.
- Driver 575.x advertises CUDA 12.9. **TensorRT-LLM 1.3.0rc14 is incompatible with this driver** (its bundled PyTorch needs CUDA 13). **TensorRT-LLM 1.0.0 is compatible.** Stage 1 proceeds on 1.0.0 per runbook §"If TensorRT-LLM fails on RTX 3070" item 4 (release/version fallback). Backend remains TensorRT-LLM.
- This finding alone is a legitimate "compatibility result" deliverable for Stage 1 §`baseline_summary` section 2.
