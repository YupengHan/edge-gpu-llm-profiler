# 07 — Troubleshooting and failure documentation

This project should document failures clearly. A failed TensorRT-LLM attempt on RTX 3070 can still be useful if the environment, exact commands, and errors are well recorded.

## Failure documentation rule

For every failure, record:

```text
1. Timestamp
2. Command
3. Environment
4. Error snippet
5. Full log path
6. Hypothesized cause
7. Next action
8. Whether the project scope changed
```

Use this format:

```markdown
## Failure: short name

- Time:
- Stage:
- Command:
- Log file:
- Error snippet:
- Likely cause:
- Tried fixes:
- Next action:
- Scope impact:
```

## Common failure modes

### 1. `nvidia-smi` does not show RTX 3070

Likely causes:

- NVIDIA driver not installed.
- WSL2 GPU passthrough not configured.
- Running inside a VM without GPU access.

Action:

```text
Stop Stage 1. Fix host GPU visibility first.
```

### 2. Docker cannot see GPU

Symptoms:

```text
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

Likely causes:

- NVIDIA Container Toolkit not installed.
- Docker daemon not configured with NVIDIA runtime.

Action:

```text
Document host issue. Do not debug TensorRT-LLM until Docker GPU access works.
```

### 3. TensorRT-LLM import fails

Symptoms:

```text
ModuleNotFoundError: No module named 'tensorrt_llm'
ImportError related to CUDA / libtensorrt / torch
```

Likely causes:

- Wrong container.
- Broken pip install.
- CUDA/PyTorch/TensorRT-LLM mismatch.

Action:

1. Prefer the official TensorRT-LLM release container.
2. Run `python3 -c "import tensorrt_llm"` inside the container.
3. Record image tag and import error.
4. Do not continue to benchmark until resolved.

### 4. RTX 3070 unsupported or plugin build fails

Symptoms:

```text
Unsupported architecture
no kernel image available for execution on the device
invalid device function
build error related to sm_86
TensorRT engine build failure
```

Likely causes:

- TensorRT-LLM release path not supporting this consumer Ampere GPU.
- CUDA architecture flags do not include sm_86.
- Specific plugin/kernel path requires datacenter GPU.

Action order:

1. Record TensorRT-LLM version and image tag.
2. Record GPU compute capability.
3. Try a TinyLlama TensorRT-LLM smoke test only to separate model issue from GPU issue.
4. Try a TensorRT-LLM release/version fallback if reasonable.
5. If still failing, write:

```text
results/stage0/tensorrt_llm_rtx3070_compatibility_failure.md
```

Do not switch to vLLM/llama.cpp silently.

### 5. Qwen2.5 model build fails

Symptoms:

```text
Unsupported model architecture
AutoConfig error
trust_remote_code error
conversion/build failure
```

Likely causes:

- Current TensorRT-LLM version does not support exact Qwen2.5 variant in the selected backend.
- CLI expects a different model path or Python API path.
- Transformers version/tokenizer issue.

Action:

1. Verify tokenizer/config works with Hugging Face `transformers`.
2. Verify TensorRT-LLM support matrix and Qwen example.
3. Try `--trust_remote_code` only if the CLI help supports it and the model requires it.
4. Try Qwen2/Qwen3 tiny equivalent only as a compatibility probe.
5. Document model support failure.

### 6. Out of memory during engine build or benchmark

Symptoms:

```text
CUDA out of memory
failed to allocate memory
KV cache memory too large
```

Likely causes:

- `max_seq_len` too large.
- `max_batch_size` not constrained to 1.
- GPU has other active processes.
- TensorRT-LLM engine/cache allocation too large for 8GB VRAM.

Action order:

1. Check `nvidia-smi` for other processes.
2. Reduce to 128/128 smoke case.
3. Use `--max_batch_size 1`.
4. Use workload-specific `--max_seq_len` and `--max_num_tokens`.
5. Lower KV cache/free memory fraction if applicable.
6. Document exact peak memory.

### 7. `prepare_dataset.py` subcommand name fails

Symptoms:

```text
invalid choice: token_norm_dist
you may have meant token-norm-dist
```

Action:

Run:

```bash
python3 ${PREPARE_DATASET} --help
```

Use the spelling shown by local help and record it in the command log.

### 8. `trtllm-bench` CLI changed

Symptoms:

```text
no such option: --engine_dir
no such option: --report_json
invalid command: latency
```

Action:

1. Run:

```bash
trtllm-bench --help
trtllm-bench build --help
trtllm-bench latency --help
trtllm-bench throughput --help
```

2. Save help outputs to logs.
3. Update the command in the runbook or local notes.
4. Mention API drift in the summary.

### 9. `trtllm-serve` model name mismatch

Symptoms:

```text
OpenAI-compatible request returns model not found
```

Action:

Run:

```bash
curl -s http://localhost:8000/v1/models
```

Use the model ID returned by the server in the streaming benchmark client.

### 10. Streaming benchmark counts chunks, not tokens

Symptoms:

```text
output_pieces differs from max_tokens
```

Explanation:

OpenAI-style streaming may emit text chunks that do not perfectly equal tokenizer tokens. In Stage 1, it is acceptable to record chunk timing for TTFT/ITL as long as the report labels it clearly.

Action:

- Use `output_pieces` and `output_tps_by_piece`.
- Add tokenizer-based counting later if needed.
- Do not claim exact tokenizer-level TPS unless implemented.

### 11. Nsight Systems unavailable inside container

Symptoms:

```text
nsys: command not found
permission denied
```

Action:

1. Try host `nsys` profiling if possible.
2. Start container with profiling permissions:

```bash
--cap-add=SYS_ADMIN --security-opt seccomp=unconfined
```

3. Document whether profiling was run inside container or on host.

### 12. Nsight Compute is too slow

Symptoms:

```text
ncu run takes very long or appears stuck
```

Explanation:

Nsight Compute may replay kernels and collect extensive metrics.

Action:

Reduce scope:

```bash
--set speedOfLight
--launch-skip 20
--launch-count 3
--num_requests 3
--warmup 1
```

Then expand only if needed.

## TensorRT-LLM fallback policy

Allowed in Stage 1:

| Fallback | Allowed? | Purpose |
|---|---|---|
| Different TensorRT-LLM release tag | Yes | Check RTX 3070 compatibility. |
| TensorRT-LLM PyTorch backend | Yes, as diagnostic | Confirm model can run through TensorRT-LLM ecosystem. |
| TinyLlama via TensorRT-LLM | Yes, diagnostic only | Separate environment issue from Qwen model issue. |
| vLLM | No | Requires explicit scope change. |
| llama.cpp | No | Requires explicit scope change. |
| SGLang | No | Requires explicit scope change. |
| Custom C++ inference | No | Requires explicit scope change. |

## Compatibility failure report template

Create `results/stage0/tensorrt_llm_rtx3070_compatibility_failure.md` if needed:

```markdown
# TensorRT-LLM RTX 3070 compatibility failure

## Summary

## Hardware

## Software

## TensorRT-LLM image/version

## Commands attempted

## Error logs

## Analysis

## Attempted mitigations

## Conclusion

## Next possible paths

- Try another TensorRT-LLM release.
- Try TensorRT-LLM PyTorch backend diagnostic.
- Move Stage 1 to a different NVIDIA GPU.
- Explicitly change project scope to a non-TRT-LLM backend.
```

## Do-not-hide rule

A well-documented failure is better than an undocumented backend switch. The project is about systems work, and compatibility findings are legitimate systems results.
