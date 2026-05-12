#!/usr/bin/env python3
"""Generate N varied natural-language prompts, each padded to exactly K user
content tokens so that the Qwen2.5 chat template wraps to a fixed total.

Usage:
  python3 scripts/generate_streaming_prompts.py \
    --target-template-total 512 \
    --num-prompts 50 \
    --out data/streaming_prompts_512.jsonl

Design:
- 5 instruction templates x 10 topics = 50 unique base prompts.
- Each base prompt contains a long preamble describing a fictional engineer's
  situation (gives natural padding) and ends with an explicit
  "Write at least 220 words" so the assistant won't EOS early.
- We pad/truncate each rendered prompt by appending/dropping deterministic
  filler sentences so that chat_template(rendered) == target tokens.

This makes the streaming benchmark reproducible AND comparable to step 10's
512-token prefill, while sidestepping prefix caching by giving each request
a unique prompt.
"""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

TEMPLATES = [
    # Each template ends with the long-output instruction.
    """You are helping a graduate student who is preparing for a qualifying exam in machine learning systems. The student has a strong undergraduate background in C++ and Python but has only used PyTorch at the level of training small models on a single GPU. They have never written a CUDA kernel and have never deployed a model behind an HTTP server. They want a careful, self-contained explanation that does not assume familiarity with NVIDIA-specific jargon. The exam topic for tonight is: {topic}. The student gives you the following constraint: every concept that is introduced must be motivated by what problem it solves, not by what the API call is named. They also ask that you explain at least one common misconception that an undergraduate would have about this area, and walk through the mental model that resolves that misconception. Please respond with a thorough explanation that is suitable as a study sheet they can reread the morning before the exam. Use clear prose, not a bullet-point list, and write at least 220 words.""",

    """A small startup is migrating their inference workload from a hosted API to a self-hosted setup on a single workstation with a consumer-grade GPU. The on-call engineer needs to make architecture decisions in the next 24 hours and has asked you for guidance on the following subject: {topic}. Their constraints are unusual: latency-sensitive (the product is a coding assistant where the user is watching tokens stream in), small VRAM (under 12 GB), and only one engineer to maintain the system long-term, so simple beats clever. Before answering, please assume the engineer already understands the basics of transformer inference but has never optimized below the framework level. Walk through the trade-offs in detail, explain what would go wrong if they picked the obvious-looking default, and then recommend what you would actually do in their position and why. End by listing the single most useful diagnostic they could run to confirm your recommendation is working. Write at least 220 words.""",

    """Imagine you are reviewing a pull request from a junior teammate. The PR claims to improve performance for the following area: {topic}. The diff is small but the description is hand-wavy. The teammate writes 'this should help on longer prompts' but offers no numbers. Your job is to write a thoughtful review comment that does three things, in order: first, explain the mental model the teammate is probably using and what is correct about it; second, identify the specific failure mode that the change might trigger that the teammate did not consider; third, propose a concrete experiment that would prove or disprove the value of the change before merging. The review should be firm but constructive, the kind of feedback a senior engineer gives that makes the junior actually understand the area better. Avoid platitudes. Write at least 220 words and include at least one specific metric that should be measured.""",

    """Walk me through, step by step, how an experienced practitioner would investigate the following question on a real workstation: {topic}. Assume the practitioner has access to nvidia-smi, Nsight Systems, Nsight Compute, and a Python REPL, but does not have access to vendor support and cannot upload artifacts to the cloud for privacy reasons. The walkthrough should describe what they would look at first, what hypothesis that observation would suggest, how they would test that hypothesis with the next measurement, and how they would know they had identified the actual cause. Do not skip the early steps that experienced people often take for granted: which tool to launch, what its default output really means, how to know whether a number is anomalous. The reader is competent but new to systems-level performance work on GPUs. Please write at least 220 words of narrative prose, not a bulleted checklist.""",

    """An interviewer at a hardware-software co-design team has decided to use the following topic as a deep-dive question for a senior engineering candidate: {topic}. The interviewer wants a model answer that they can use to calibrate candidate responses. A weak candidate will give the textbook definition and stop. A strong candidate will (a) name the underlying mechanism, (b) explain at least one non-obvious second-order effect that arises when this mechanism interacts with another part of the system, (c) describe a real situation in which the textbook intuition leads to the wrong conclusion, and (d) propose what they would measure to detect that situation. Please produce the model answer that captures all four points, in approximately the order above, in clear technical prose. Avoid bullets and avoid restating the question. Write at least 220 words.""",
]

TOPICS = [
    "the KV cache in autoregressive transformer inference and why its memory growth is linear in sequence length",
    "rotary position embeddings (RoPE) and how they let a model generalize to context lengths it was never trained on",
    "the difference between tensor parallelism and pipeline parallelism and when one is preferable over the other on a single workstation",
    "grouped query attention (GQA) and why models like Qwen2.5 ship with num_kv_heads strictly smaller than num_attention_heads",
    "paged KV cache and why it matters specifically for high-throughput single-GPU serving rather than for single-user low-latency chat",
    "CUDA streams, kernel launch overhead, and the conditions under which a tiny kernel becomes the bottleneck in inference",
    "the practical differences between FP16 and BF16 on Ampere-class GPUs, including what each format costs in accuracy and what it saves in throughput",
    "the Nsight Systems profiling workflow for a single-GPU inference job, including which views to use first and what their default settings hide",
    "continuous batching (in-flight batching) and the specific reason it can hurt the latency of a single user while helping throughput for many concurrent users",
    "post-training quantization with int8 versus int4 weights, how each maps to GPU compute units, and why one can be slower despite using less memory",
]


def render(template, topic):
    return template.format(topic=topic)


def fit_to_target(tok, text, target_total):
    """Adjust text so chat_template(user=text) has exactly target_total tokens.

    Strategy: append filler sentences if too short, drop tail tokens if too long.
    Returns the final user-content text plus actual chat-template-total length.
    """
    filler = (
        " Add specific examples drawn from a single-GPU consumer-grade workstation"
        " context with limited VRAM and no networked teammates."
    ) * 6

    def total_for(s):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": s}],
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(ids)

    cur = text
    while total_for(cur) < target_total:
        cur = cur + filler
    # Now cur is too long or exact. Trim user-content tokens until total fits.
    user_ids = tok(cur, add_special_tokens=False)["input_ids"]
    lo, hi = 1, len(user_ids)
    best_text = cur
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = tok.decode(user_ids[:mid], skip_special_tokens=True)
        t = total_for(cand)
        if t <= target_total:
            best_text = cand
            best_total = t
            lo = mid + 1
        else:
            hi = mid - 1
    return best_text, total_for(best_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--target-template-total", type=int, default=512,
                    help="chat_template(user=prompt) total token target")
    ap.add_argument("--num-prompts", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    pairs = []
    for ti, tmpl in enumerate(TEMPLATES):
        for pi, topic in enumerate(TOPICS):
            pairs.append((ti, pi, topic, tmpl))
    assert len(pairs) >= args.num_prompts

    rows = []
    for idx in range(args.num_prompts):
        ti, pi, topic, tmpl = pairs[idx]
        text = render(tmpl, topic)
        fitted, total = fit_to_target(tok, text, args.target_template_total)
        rows.append({
            "prompt_index": idx,
            "template_index": ti,
            "topic_index": pi,
            "topic": topic,
            "chat_template_total_tokens": total,
            "prompt": fitted,
        })

    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    totals = [r["chat_template_total_tokens"] for r in rows]
    print(f"wrote {len(rows)} prompts -> {out}")
    print(f"chat_template_total tokens: min={min(totals)} max={max(totals)} "
          f"mean={sum(totals)/len(totals):.1f}")


if __name__ == "__main__":
    main()
