# 06 — References

This file lists references that should be cited or linked in the public README and reports.

## TensorRT-LLM official docs

### Main docs

- TensorRT-LLM documentation: https://nvidia.github.io/TensorRT-LLM/

### Installation

- Installation guide: https://nvidia.github.io/TensorRT-LLM/installation/installation-guide.html
- Official docs state that the pre-built release container is the simplest way to obtain TensorRT-LLM.
- Official docs also show a `docker pull` / `docker run` workflow and an import sanity check with `python3 -c "import tensorrt_llm"`.

### Supported hardware

- Current supported hardware page: https://nvidia.github.io/TensorRT-LLM/supported-hardware.html
- Older TensorRT-LLM overview page: https://nvidia.github.io/TensorRT-LLM/1.2.0rc0/overview.html

Project note:

```text
Treat RTX 3070 compatibility as a Stage 0 validation item. Current and older docs may differ in how explicitly RTX 30 series is mentioned.
```

### Benchmarking

- TensorRT-LLM benchmarking guide: https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html
- `trtllm-bench` CLI reference: https://nvidia.github.io/TensorRT-LLM/commands/trtllm-bench.html

Relevant details:

- `trtllm-bench` provides build, throughput, and latency workflows.
- `prepare_dataset.py --stdout` is required for the expected streaming dataset format.
- `trtllm-bench` APIs can change; record the actual local CLI version and syntax.

### Serving

- `trtllm-serve` CLI reference: https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html

Relevant details:

- `trtllm-serve` starts an OpenAI-compatible server.
- It exposes `/v1/models`, `/v1/completions`, `/v1/chat/completions`, `/health`, `/metrics`, and `/version`.
- The `MODEL` argument can be a model name, Hugging Face checkpoint path, or TensorRT engine path according to the CLI docs.

### Qwen support

- TensorRT-LLM support matrix: https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html
- TensorRT-LLM Qwen example: https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/qwen/README.md

Project note:

```text
The current support matrix lists Qwen2/Qwen3 language model support and Qwen2.5-VL multimodal support, but the exact Qwen2.5-1.5B TensorRT backend path should still be validated locally.
```

## Model references

### Qwen2.5-1.5B-Instruct

- Hugging Face model card: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

Important facts from model card:

| Field | Value |
|---|---|
| Parameters | 1.54B total, 1.31B non-embedding |
| Layers | 28 |
| Attention | GQA: 12 Q heads, 2 KV heads |
| Context | 32,768 tokens; generation up to 8,192 tokens |
| Architecture | Transformer with RoPE, SwiGLU, RMSNorm, Attention QKV bias, tied word embeddings |

## External benchmark references

### Edge LLM Leaderboard

- Space: https://huggingface.co/spaces/nyunai/edge-llm-leaderboard
- README: https://huggingface.co/spaces/nyunai/edge-llm-leaderboard/blob/main/README.md
- Raspberry Pi 5 CSV: https://huggingface.co/spaces/nyunai/edge-llm-leaderboard/blob/main/dataset/llm-perf-leaderboard-Raspberry%20Pi%205%288GB%29.csv

Relevant details:

- The leaderboard targets edge LLM practical performance and quality.
- It starts with Raspberry Pi 5 8GB / ARM Cortex A76.
- It uses singleton batch, 512-token prompt, and 128 generated tokens.
- It uses llama-bench / llama.cpp, not TensorRT-LLM.

Project note:

```text
Use Edge LLM Leaderboard as workload inspiration, not direct speed comparison.
```

## Profiling references

### Nsight Systems

- Nsight Systems user guide: https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- Nsight Systems product page: https://developer.nvidia.com/nsight-systems

Relevant details:

- `nsys profile` can launch an application and collect a timeline trace.
- `nsys stats` can post-process `.nsys-rep` files into text summaries.

### Nsight Compute

- Nsight Compute CLI guide: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
- Nsight Compute product page: https://developer.nvidia.com/nsight-compute

Relevant details:

- `ncu` is a non-interactive CLI profiler for CUDA kernels.
- It can print results or store report files.
- It may replay kernels and slow execution; use small launch counts.

### PyTorch profiler

- PyTorch profiler docs: https://docs.pytorch.org/docs/stable/profiler.html
- PyTorch profiler recipe: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

Project note:

```text
In Stage 1, PyTorch profiler is optional and mainly useful if using TensorRT-LLM PyTorch backend or investigating Python/server overhead. Nsight tools are the main profiler focus.
```

## Suggested citation language for README

```markdown
The benchmark workload uses 512 input tokens / 128 output tokens to align with the Edge LLM Leaderboard setup, but the results are not directly comparable because that leaderboard runs llama.cpp on Raspberry Pi 5, while this project runs TensorRT-LLM on an RTX 3070.
```

```markdown
TensorRT-LLM provides `trtllm-bench` for build/throughput/latency benchmarking and `trtllm-serve` for OpenAI-compatible serving. This project uses both: `trtllm-bench` for official backend benchmark runs and a streaming client against `trtllm-serve` for TTFT/ITL measurement.
```
