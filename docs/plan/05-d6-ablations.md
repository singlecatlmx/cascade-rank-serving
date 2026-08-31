# D6 · 对照实验（1 天，2 卡）

> 四组独立实验，**按 ROI 排序执行，做不完就按顺序砍**。
> 每组都是"一张表 + 一段分析"，不追求深度，追求覆盖面与可信度。

## 执行顺序（时间不够从后往前砍）

```
E4 打分路径  →  E5 量化  →  E6 拓扑  →  E7 attention backend
```

---

## E4 — 打分路径（优先级 1，约 2 小时）

本 workload 只需要 1 个 token 的 yes/no logit，**是纯 prefill-bound**。

| 变体 | 方式 | 说明 |
|---|---|---|
| D0 | 生成完整回答 | 反面对照，暴露 decode 的浪费 |
| D1 | `max_tokens=1` + `logprobs` | 基本做法 |
| D2 | + `allowed_token_ids=[yes,no]` | 限制词表 |
| D3 | vLLM 原生 score / pooling 路径 | 若 0.25.1 支持 |

**衍生调参**：因为是 prefill-bound，`max_num_batched_tokens` 与 chunked prefill 的
最优值与常规 chat 场景**相反**。扫 2048 / 4096 / 8192 / 16384，画一条曲线。

**要的结论**：「排序 workload 的调参直觉和聊天 workload 是反的」——这是面试里的亮点。

---

## E5 — 量化（优先级 2，约 3 小时）

**实验矩阵形状由 D0 的量化冒烟结果决定**，不要照抄下表，先看 `results/quant_smoke_*.json`。

预期可用档位：

| 档位 | 状态 |
|---|---|
| BF16 | 基线 |
| `fp8` | 高置信可用 |
| `awq_marlin` INT4 | 高置信可用 |
| `gptq_marlin` INT4 | 高置信可用 |
| `+ KV cache FP8` | 与上述正交，单独一维 |
| `modelopt_fp4` | 需实测 |
| `mxfp4` / `mxfp8` | **预期不可用**（`_qutlass_C` 无 sm_120）|

**对象**：D3 产出的 `models/reranker-0.6b-merged/`（自己训的领域模型，比量化官方模型有说服力）
外加 `Qwen3-Reranker-4B` 作为大模型档。

**四维记录**：`MAP@25 × 峰值显存 × candidates_scored_per_s × TTFT p99`

**★ 关键消融（原方案的核心发现，不要省）**：
通用校准集（C4 / pileval）vs **任务相关校准集**（reranker 训练数据子集）。
原作者明确指出任务相关校准集是保住精度的关键。这一条是本组最有价值的产出。

---

## E6 — TP vs DP 拓扑（优先级 3，约 2 小时）

已有硬数据支撑：**P2P = false，卡间实测 43.6 GB/s，与单卡显存带宽比 1:41**
（H100 NVLink 是 1:3.7，本机差约 11 倍）。

| 变体 | 配置 | 预期 |
|---|---|---|
| F0 | 单卡 1 实例 | 基线 |
| F1 | **TP=2** 1 实例 | 显著低于 F0×2，可能低于 F0×1.5 |
| F2 | 2 卡各 1 实例（DP=2）+ 轮询 | 接近 F0×2 |

**要的结论**：「无 P2P 的消费卡上，级联排序服务应该用 DP / stage-split 而非 TP」，
并用 43.6 GB/s 做定量解释。这是原方案完全没有的**原创贡献**。

---

## E7 — Attention backend（优先级 4，约 1.5 小时，可砍）

候选：`flash_attn` / `flashinfer` / `triton_attn` / `flex_attention`

因为是 prefill-bound，backend 的 prefill 性能差异被放大，
常规 chat benchmark 里被 decode 掩盖的差异会暴露出来。

⚠️ 切换旋钮见 D0 的 T0.6 结论。**`VLLM_ATTENTION_BACKEND` 已失效且静默**——
若用它做实验，测出来的会是四份一模一样的默认后端数据。**必须在日志里确认真的切换了。**

---

## 产出物

- `scripts/run_e4_*.sh` ~ `run_e7_*.sh`
- `results/e4_*.json` ~ `e7_*.json`
- `assets/` 下 4 张图

## 已知坑

- 量化模型的产出会吃磁盘，跑完一档及时清理不需要的
- AWQ/GPTQ 量化过程主要吃 CPU RAM（128 GB 够）
- 每换一个变体重启 engine
- E6 的 TP=2 若 hang，检查是否需要显式 `NCCL_P2P_DISABLE=1`
- E5 的质量指标必须用同一个 `eval_set_v1.jsonl`

## 不要做

- ❌ 不做 MoE（Qwen3-30B-A3B）——写进 Future Work 就够了
- ❌ 不做投机解码——本 workload 没有 decode 阶段，无意义
- ❌ 不做 4 卡实验
- ❌ 不为了凑数把不可用的量化档"想办法跑通"，不可用就如实记录并说明原因
