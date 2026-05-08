#!/bin/bash
set -u
WORKDIR=/workspace/rtx3070-trtllm-latency-lab
cd "$WORKDIR"

mkdir -p logs/stage1

export HF_MODEL=Qwen/Qwen2.5-1.5B-Instruct
export HF_HOME=/workspace/rtx3070-trtllm-latency-lab/models/.hf

mkdir -p "$HF_HOME"

echo "=== HF_MODEL=$HF_MODEL"
echo "=== HF_HOME=$HF_HOME"

python3 - <<'PY' | tee logs/stage1/qwen_tokenizer_check.txt
from transformers import AutoTokenizer
model = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model)
print("model", model)
print("vocab_size_len", len(tok))
print("eos_token", tok.eos_token)
print("eos_token_id", tok.eos_token_id)
print("pad_token", tok.pad_token)
print("bos_token", tok.bos_token)
print("model_max_length", tok.model_max_length)
text = "Hello, benchmark test."
ids = tok.encode(text)
print("sample_text", repr(text))
print("sample_tokens_count", len(ids))
print("sample_token_ids", ids[:20])
print("decoded_back", tok.decode(ids))
PY

python3 - <<'PY' | tee logs/stage1/qwen_config_check.txt
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
for k in [
    "model_type",
    "architectures",
    "torch_dtype",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_size",
    "intermediate_size",
    "vocab_size",
    "max_position_embeddings",
    "rope_theta",
    "tie_word_embeddings",
]:
    print(k, getattr(cfg, k, None))
PY
