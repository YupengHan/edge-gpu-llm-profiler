# Stage 1 Step 3-10 学习笔记

> 这是一份**学习向**的中文解读，把 `logs/stage1/stage1_command_log.md` 里 step 3 到 step 10 的实操内容串成一条逻辑线。
> 配套阅读：
> - `docs/02_stage1_local_ai_runbook.md`（runbook 原始规范）
> - `logs/stage1/stage1_command_log.md`（实际执行记录，含命令、输出、踩坑）
>
> **执行环境**：RTX 3070 Laptop（8 GiB, sm_86，驱动 575.57.08 / CUDA 12.9）+ Ubuntu 22.04，16 GiB RAM / 2 GiB swap。
> **后端**：TensorRT-LLM **1.0.0**（不是 1.3.0rc14，原因见 step 3）。
> **模型**：`Qwen/Qwen2.5-1.5B-Instruct`，**bfloat16**（不是严格 fp16，原因见 step 7）。
> **场景**：concurrency=1，batch_size=1。

---

## 整体脉络（step 3-10 在做一件什么事）

把一台普通 RTX 3070 Laptop（8 GiB VRAM）变成一个**可重复、可对照、有数据**的 LLM 推理基准平台，路线是：

> **容器就绪 → 模型/数据就绪 → 编引擎 → 跑 bench**

并且每一步都先用最小规模（smoke）走通流程，再放真实工作负载（main）。

| Step | 阶段 | 关键产出 |
|---|---|---|
| 3 | 容器就绪 | 选定 1.0.0 镜像，`import tensorrt_llm` 成功 |
| 4 | 模型就绪 | Qwen2.5 tokenizer/config 验证通过 |
| 5-6 | 数据就绪 | 三个长度的合成数据集 jsonl |
| 7 | 编引擎（smoke） | 128/128 引擎，验证编译链 |
| 8 | 跑 bench（smoke） | 第一份 latency 数（TTFT/TPOT）|
| 9 | 编引擎（main） | 512/128 引擎（覆盖了 smoke 引擎）|
| 10 | 跑 bench（main） | latency + throughput 双子命令的 baseline |

理解一个核心概念：**TensorRT-LLM 不像 PyTorch 那样可以即跑即推**。它要先把 HF 模型 → TensorRT 图 → 编译成一个 `.engine` 文件（一次性，几十秒到几分钟），之后所有 bench 都基于这个 `.engine` 跑。所以 **step 7、9 是"编译"，step 8、10 是"运行"**。

---

## Step 3 — 启动项目容器

**目的**：确认我们要用的容器能正常 `import tensorrt_llm`。

### 做了什么 / 为什么这么做

1. 先尝试官方最新 tag `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14`，结果 `import tensorrt_llm` 直接报：
   ```
   RuntimeError: The NVIDIA driver on your system is too old (found version 12090)
   ```
2. 原因：这个镜像里的 PyTorch（`torch 2.11.0a0+nv26.02`）是用 **CUDA 13.x driver ABI** 编出来的，但你的本机驱动 575.57.08 只声明 CUDA 12.9。这是典型的"驱动 vs PyTorch 编译时 CUDA 版本不匹配"。
3. 按 `docs/07_troubleshooting.md` 的 fallback 策略，**不切换后端**（不许换 vLLM / llama.cpp），而是回退到 `release:1.0.0`（这个镜像里 `torch 2.8.0a0+nv25.06` 是 CUDA 12.x ABI），import 就过了。

### 学习要点

- GPU 容器栈的兼容三件套是 **驱动 ↔ CUDA runtime ↔ 框架二进制**，不是只看 CUDA 版本号大就行。`nvidia-smi` 显示 CUDA 12.9 不代表所有 CUDA 12.x 镜像都能跑。
- "最新稳定 tag" 不一定适合消费级显卡 + 老一代驱动。**Stage 1 的产出之一就是这条兼容性记录**，本身就是 deliverable。
- 项目硬约束：**TensorRT-LLM 不能换后端**。如果在 RTX 3070 上 TRT-LLM 完全跑不动，写一份 `tensorrt_llm_rtx3070_compatibility_failure.md` 也比偷偷换 vLLM 更有价值。

### 产出

- `logs/stage0/trtllm_image.env`：最终选定 `TRTLLM_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.0.0`
- `logs/stage1/step3_container_probe.log`（1.3.0rc14 失败）
- `logs/stage1/step3_container_probe_1_0_0.log`（1.0.0 成功）

---

## Step 4 — Qwen2.5 tokenizer/config 验证

**目的**：在跑任何东西之前，确认模型能下载、tokenizer 能加载、config 是预期值。

### 做了什么

在 1.0.0 容器里执行 `from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")`，dump 关键字段：

| 字段 | 值 | 意义 |
|---|---|---|
| `vocab_size` | 151936 | 词表大小 |
| eos token | `<|im_end|>` (id=151645) | 生成停止符 |
| `torch_dtype` | **bfloat16** | **决定后续编译 dtype 的默认值** |
| `num_hidden_layers` | 28 | Transformer 层数 |
| `num_attention_heads` | 12 | Q heads |
| `num_kv_heads` | 2 | KV heads（GQA 6:1）|
| `hidden_size` | 1536 | 每层隐藏维度 |
| `max_position_embeddings` | 32768 | RoPE 最大上下文 |

### 学习要点

- HF config 里的 `torch_dtype` 字段是后续编译/推理 dtype 的**默认入口**。先看一眼能省掉很多后续 debug。
- **GQA（Grouped Query Attention）** 的 `num_kv_heads < num_attention_heads`，这是 **KV cache 内存消耗减小的关键**：每层 KV cache 大小 = `2 × num_kv_heads × head_dim × seq_len × dtype_bytes`，而不是按 attention heads 算。后续做显存分析时会再回到这里。
- "tokenize → detokenize 能往返"是一个简单但重要的健全性检查：tokenizer 文件残缺时，引擎跑出来的 token id 解码会乱码。

---

## Step 5-6 — 生成合成数据集

**目的**：bench 需要一批**长度可控、可重复**的请求。真实数据每次 token 数都不同，没法做严格对照。

### 做了什么

用 TRT-LLM 自带的 `benchmarks/cpp/prepare_dataset.py token-norm-dist` 子命令，按"输入 token 数正态分布"生成三个数据集：

| 文件 | 行数 | 输入≈ | 输出 | 用途 |
|---|---|---|---|---|
| `data/synthetic_128_128.jsonl` | 50 | 128 | 128 | smoke |
| `data/synthetic_512_128.jsonl` | 100 | 512 | 128 | **main workload** |
| `data/synthetic_1024_256.jsonl` | 30 | 1024 | 256 | stretch（step 17 才用）|

每行的格式：

```json
{"task_id": 0, "input_ids": [...], "output_tokens": 128}
```

### 踩坑（CLI drift）

runbook 里写的是 `token_norm_dist`（下划线），1.0.0 容器里实际叫 `token-norm-dist`（连字符）。这种"文档和实际 CLI 不一致"在 stage 1 里反复出现，所以**每次都要先 `--help` 再写 script**。

### 学习要点

- LLM bench 必须用合成数据，**因为 latency 是输入/输出 token 数的函数**，shape 不固定就没法做 P50/P90 比较。
- `output_tokens` 字段会强制模型生成正好 128 个 token（不会提前 EOS 截断），保证每个请求的工作量一样。
- "正态分布"是为了模拟真实流量的轻微抖动，不是绝对等长——但分布窄，标准差小。

---

## Step 7 — 编 smoke 引擎（128/128）

**目的**：用最小 shape 把"HF model → TensorRT engine"这条编译链跑通一次，先暴露所有环境/参数问题。

### 命令骨架

```bash
trtllm-bench build \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --max_batch_size 1 --max_num_tokens 256 --max_seq_len 256
```

### 两个大坑

#### 坑 1：精度坑（runbook 说 fp16，实际是 bf16）

runbook 说 "FP16, no quantization"，但 1.0.0 的 `trtllm-bench build` **没有 `--dtype` flag**，dtype 全靠 HF config 的 `torch_dtype` 决定。Qwen2.5 是 `bfloat16`，所以引擎实际是 bf16。

**为什么可以接受**：在 sm_86 上 fp16/bf16 Tensor Core 吞吐都是 86 TFLOPS，不影响延迟数字。但**精度记录上要写清楚是 bf16 不是 fp16**。

**如果将来要严格 fp16 对比**：必须切到 `tensorrt_llm.LLM` Python API 显式 `dtype='float16'` 重编，**不要**继续用 `trtllm-bench`。

#### 坑 2：OOM 坑（第一次直接锁机）

第一次裸跑时，TRT 编译器吃光了 16 GiB 内存 + 2 GiB swap，**整个桌面卡死，被迫硬重启**。

解决方案是给 docker 加资源上限：

```bash
docker run --rm --gpus all \
  --memory 10g --memory-swap 12g \
  --shm-size 2g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $PWD:/workspace/rtx3070-trtllm-latency-lab \
  -w /workspace/rtx3070-trtllm-latency-lab \
  nvcr.io/nvidia/tensorrt-llm/release:1.0.0 \
  bash scripts/_step7_build_smoke.sh
```

让容器先被 OOM-kill，而不是把宿主机一起拖下水。**这条 "docker resource template" 后续每一步都用**。

### 资源用量记录

- 引擎构建峰值内存：**10053 MB**（刚好顶到 10g 上限，靠 12g swap 救回来）
- 引擎生成时间：64.5s，加 warmup 总共 ~106s
- 引擎大小：`rank0.engine` 3.02 GiB

### 学习要点

- TRT 引擎编译是**纯 CPU+RAM 密集型**操作（不是 GPU 那一头），所以瓶颈是内存而不是显存。GPU 在编译期间几乎是闲的。
- 编出来的引擎被写到 `trtllm_workspace/Qwen/Qwen2.5-1.5B-Instruct/tp_1_pp_1/rank0.engine`。**这个路径只看 model name + tp/pp，不带 build 参数 hash**——所以 step 9 重新编 512/128 时会**直接覆盖 smoke 引擎**，是个隐藏陷阱。
- 引擎是 root-owned（容器里以 root 跑出来的），删除时需要 sudo 或起一个一次性容器去删。

---

## Step 8 — Smoke latency benchmark

**目的**：用刚编好的 128/128 引擎跑一遍 `trtllm-bench latency`，确认引擎能正常推理，并拿到第一份延迟数。

### 关键 CLI 检查（必看）

1.0.0 的 `trtllm-bench latency` 默认 `--backend=pytorch`！**必须显式 `--backend tensorrt`**，否则它会去跑 PyTorch in-flight 路径，根本没用上你的 `.engine`，跑出来的数也对不上。

### 结果（20 请求 + 5 warmup, concurrency=1, backend=tensorrt）

| 指标 | 含义 | 数值 |
|---|---|---|
| Avg request latency | 整个请求 end-to-end | **1311.30 ms** |
| TTFT avg | Time To First Token，prefill 阶段耗时 | **18.33 ms** |
| TPOT avg | Time Per Output Token，decode 阶段每 token 间隔 | **10.18 ms** |
| Per-user output throughput | 单用户视角的 token/s | **97.62 tok/s** |
| GPU idle 后显存 | bench 结束后残留 | 300 / 7541 MiB |

### 学习要点（**这是 LLM 推理最重要的两个指标**）

- **TTFT** = 用户按下回车到看到第一个字的延迟，主要由 **prefill**（一次性把 prompt 全过一遍 attention）决定，**受输入长度影响大**。
- **TPOT** = 后续每个 token 的间隔，由 **decode**（一次生成一个 token，KV cache 复用）决定，**跟输入长度关系小**。
- 验算公式：`latency ≈ TTFT + TPOT × (N - 1)`，这里 `18.33 + 10.18 × 127 = 1311.2 ms` ≈ 实测 1311.3 ms ✓
- 这两个指标决定后续优化方向：抱怨"卡很久才开始说话" → TTFT 问题（优化 prefill）；"说得慢" → TPOT 问题（优化 decode 内核 / KV cache）。

### 产出

- `results/stage1/benchmark_raw/trtllm_bench_128_128_latency.json` (3.7 KB)
- `results/stage1/benchmark_raw/trtllm_bench_128_128_iteration.jsonl` (2.5 MB，**每个 iteration 的细粒度 metrics**)

---

## Step 9 — 编 main 引擎（512/128）

**目的**：把引擎参数换成真正要测的 main workload（512 输入 + 128 输出 = `max_seq_len 640`）。

### 踩坑（参数互斥）

runbook 让你同时传 `--dataset` 和 `--max_batch_size`/`--max_num_tokens`，但 1.0.0 里这是**三组互斥的参数**：

| 模式 | 用的 flag |
|---|---|
| Dataset 模式 | `--dataset <path>` |
| **IFB Scheduler Limits** | `--max_batch_size + --max_num_tokens` |
| Tuning Heuristics | `--target_input_len + --target_output_len` |

**选了 IFB**，因为 `max_batch_size=1` 是"单用户 concurrency=1"这个核心场景的开关，dominant 意图比 dataset 更重要。

### 命令

```bash
docker run --rm --gpus all \
  --memory 12g --memory-swap 14g --shm-size 2g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $PWD:/workspace/rtx3070-trtllm-latency-lab \
  -w /workspace/rtx3070-trtllm-latency-lab \
  nvcr.io/nvidia/tensorrt-llm/release:1.0.0 \
  bash scripts/_step9_build_main.sh
```

### 资源用量

- 引擎构建峰值内存：**11617 MB**（在 12g 上限下还有 ~670 MiB 余量，14g swap 没动用）
- 引擎生成时间：50.7s，加 warmup ~78s
- 引擎大小：`rank0.engine` 3.01 GiB
- dtype: bfloat16, KV cache dtype: None（默认 bf16），no quant

### 隐藏副作用：覆盖了 smoke 引擎

引擎路径不带参数 hash，所以这次编译**把 step 7 的 128/128 smoke 引擎覆盖掉了**。在 stage 1 里这没事（smoke 已经完成使命），但要记住：

- 如果以后想同时保留多个引擎，**必须 `cp -a` 备份后再重编**
- 例如未来想做 1024/256 stretch（step 17），就需要先把 512/128 引擎复制走

### 学习要点

- 引擎是按 `(model, tp, pp, max_seq_len, max_batch_size, max_num_tokens, dtype, quant)` 组合特化的。改任何一个都要重编，**而且默认会原地覆盖**。
- `max_seq_len = input_len + output_len` 是 **KV cache 容量上限**的关键参数。给得太小会运行时拒绝请求；给得太大会浪费显存（KV cache 按这个 size 预分配）。
- max_seq_len 的内存代价 ≈ `2 × num_kv_heads × head_dim × max_seq_len × max_batch_size × dtype_bytes × num_layers`，跟 max_batch_size 是相乘关系——所以"加大 batch_size 同时加大 max_seq_len"会平方级吃显存。

---

## Step 10 — 跑 main bench（512/128）

**目的**：用 main 引擎跑两个子命令，拿到 stage 1 的"baseline 数字"。

### 跑两个子命令是因为它们看不同侧面

#### `trtllm-bench latency`（100 请求，warmup 10，concurrency=1）

| 指标 | 数值 |
|---|---|
| Avg request latency | **1302.17 ms**（P50 1302.6, P90 1313.4, P95 1318.9, P99 1340.1）|
| TTFT avg | **51.77 ms**（P50 51.36, P90 53.21, P95 56.27, P99 76.02）|
| TPOT avg | **9.85 ms**（P50 9.85, P90 9.95, P95 10.00, P99 10.11，**非常稳**）|
| Per-user output throughput | 98.30 tok/s |
| Generation speed | 101.57 tps |

#### `trtllm-bench throughput`（同 shape，100 请求，concurrency=1）

| 指标 | 数值 |
|---|---|
| Avg request latency | 1339.74 ms |
| **Request throughput** | **0.7464 req/sec** |
| **Total token throughput**（in+out）| **477.69 tok/s** |
| Total output throughput | 95.54 tok/s |

### 两个子命令的分工

- `latency` 给你**单请求**的细节延迟分布（P50/P90/P95/P99 + TTFT + TPOT）
- `throughput` 给你**系统视角**的吞吐（req/s, tok/s），但**不报 TTFT/TPOT** ——这就是为什么 step 11+ 还要单独写一个 streaming 客户端去采 TTFT/ITL（Inter-Token Latency）

### 与 smoke 对比（学习重点）

| 指标 | 128/128 (step 8) | 512/128 (step 10) | 解读 |
|---|---|---|---|
| TTFT | 18.33 ms | **51.77 ms** | 输入从 128 → 512 (4×)，TTFT 涨到 ~2.8×。**prefill 计算量增加但有缓存/常数项摊薄** |
| TPOT | 10.18 ms | 9.85 ms | **几乎不变**。decode 是单 token 自回归，跟输入长度关系很小 |
| Total latency | 1311 ms | 1302 ms | 几乎一样（输出都是 128 token）|

这就**直接验证了 LLM 推理的两阶段模型**：
- TTFT ∝ prefill 量 ∝ 输入长度
- TPOT ⊥ 输入长度（decode 阶段每步都是同样的"加一个 token + 走 28 层"的工作量）
- 总延迟 ≈ TTFT + TPOT × (output_len - 1)

### 产出

写到 `results/stage1/benchmark_raw/`：
- `trtllm_bench_512_128_latency.json` (3.7 KB)
- `trtllm_bench_512_128_iteration.jsonl` (12.6 MB) — 每个 iteration 的细粒度 metrics
- `trtllm_bench_512_128_throughput.json` (2.7 KB)
- `trtllm_bench_512_128_throughput_iteration.jsonl` (12.6 MB)
- `trtllm_bench_512_128_requests.jsonl` (66 KB) — 每个请求的逐条记录

后两份 jsonl 是后续画时间序列、找尾延迟来源、对照 nsys profile 的原料。

---

## 为什么 step 3-10 这样排（学习总结）

1. **环境先行**：在花时间下数据/编引擎之前先确认 import 能过（step 3）。否则 build 一半才发现是 driver 问题，几十 GB 镜像白下载。
2. **小步快跑**：smoke (step 7-8, 128/128) → main (step 9-10, 512/128)。流程错了在 smoke 阶段就暴露，避免 main 阶段编 5 分钟才发现 CLI drift。
3. **每步都对一遍 `--help`**：因为 1.0.0 vs runbook 的 CLI 一直在飘。这是工程纪律，不是 paranoid。
4. **Latency 和 Throughput 必须分开跑**：单用户场景下 latency 看延迟分布，throughput 看吞吐上限——它们用同一个引擎但回答不同问题。
5. **每个数都能用公式验算**：
   - `latency ≈ TTFT + TPOT × (N - 1)`
   - `TTFT ∝ input_len`
   - `TPOT ⊥ input_len`
   - 能自己推一遍，后续 profiling 才知道哪里反常。
6. **每一步都留命令日志和 stdout**：`logs/stage1/stage1_command_log.md` 是合同，`scripts/_stepN_*.sh` 是脚本，`results/` 是产物。runbook 是规范，CLI drift 改脚本不改 runbook。

---

## 下一步预告（step 11+）

- **step 11**：切换到 `trtllm-serve`（OpenAI 兼容的 HTTP server），因为 `trtllm-bench` 是离线一次性跑完，**没法采流式 ITL**（每两个 token 之间的间隔），那个是真实聊天体验的关键指标。
- **step 12**：写一个 streaming OpenAI 客户端（`scripts/benchmark_streaming_openai.py`）去采 TTFT 和 ITL 的精细分布。
- **step 13-16**：开始上 profiler——`nvidia-smi dmon` 采显存/利用率时间序列、Nsight Systems 看时间线、Nsight Compute 钻特定 kernel。
- **step 17**：可选的 stretch workload（1024/256），需要先备份 512/128 引擎再重编。
- **step 18-19**：把 stage 1 baseline 写成总结报告 + 更新 README。

记住硬约束：**不许换后端**、**镜像锁 1.0.0**、**dtype = bfloat16**、**单用户 concurrency=1**。
