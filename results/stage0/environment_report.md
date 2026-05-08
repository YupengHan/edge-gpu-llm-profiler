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

- TensorRT-LLM container tested: pending (Step 2)
- TensorRT-LLM import tested: pending (Step 3)
- `trtllm-bench` available: pending (Step 3)
- `trtllm-serve` available: pending (Step 3)

## Notes

- The GPU is an **RTX 3070 Laptop GPU**, not desktop. VRAM is 8 GB. The runbook scope ("RTX 3070-class local GPU") applies; performance numbers should be labeled `RTX 3070 Laptop` to avoid being confused with the 8 GB desktop variant.
- sm_86 is a consumer Ampere arch. Some TensorRT-LLM kernel paths are tuned for datacenter Ampere/Hopper; if engine build fails, follow runbook's compatibility-failure path rather than switching backend.
- Driver 575.x advertises CUDA 12.9 — newer than CUDA 12.6/12.8 typically baked into the TensorRT-LLM 1.3.x release containers. Forward-compatibility should hold (driver newer than runtime), but if the container reports CUDA driver/runtime mismatch, document it.
- Host has both nsys 2025.5 and ncu 2024.1; container may bundle different versions. Will record container-side versions in Step 3.
