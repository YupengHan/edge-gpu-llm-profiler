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
