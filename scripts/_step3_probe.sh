#!/bin/bash
set -u
WORKDIR=/workspace/rtx3070-trtllm-latency-lab
cd "$WORKDIR"

mkdir -p logs/stage0 logs/stage1

echo "=== python / torch / tensorrt_llm import probe ==="
python3 - <<'PY' | tee logs/stage0/trtllm_python_import.txt
import sys
print("python", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__)
    print("torch.cuda.is_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch.cuda.device_count", torch.cuda.device_count())
        print("torch.cuda.get_device_name(0)", torch.cuda.get_device_name(0))
        print("torch.cuda.get_device_capability(0)", torch.cuda.get_device_capability(0))
except Exception as e:
    print("torch_error", repr(e))
try:
    import tensorrt
    print("tensorrt", tensorrt.__version__)
except Exception as e:
    print("tensorrt_error", repr(e))
try:
    import tensorrt_llm
    print("tensorrt_llm", getattr(tensorrt_llm, "__version__", "unknown"))
except Exception as e:
    print("tensorrt_llm_error", repr(e))
try:
    import transformers
    print("transformers", transformers.__version__)
except Exception as e:
    print("transformers_error", repr(e))
try:
    import pydantic
    print("pydantic", pydantic.__version__)
except Exception as e:
    print("pydantic_error", repr(e))
PY

echo "=== trtllm-bench --help ==="
trtllm-bench --help 2>&1 | tee logs/stage0/trtllm_bench_help.txt | tail -30
echo "=== trtllm-serve --help ==="
trtllm-serve --help 2>&1 | tee logs/stage0/trtllm_serve_help.txt | tail -30
echo "=== nsys --version (container) ==="
nsys --version 2>&1 | tee logs/stage0/nsys_version_container.txt
echo "=== ncu --version (container) ==="
ncu --version 2>&1 | tee logs/stage0/ncu_version_container.txt
