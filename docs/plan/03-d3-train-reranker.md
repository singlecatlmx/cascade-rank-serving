# D3 · 训练最小集（1 天，1 卡）

> 全项目**只训一个模型**：`Qwen3-Reranker-0.6B` 的 LoRA。
> 目的有二：① 证明具备训练能力 ② 给 E5 量化实验提供一个"自己训的领域模型"，
> 这样"量化掉不掉领域精度"才有说服力（量化官方模型说服力弱得多）。

## 前置条件

- [ ] G2 已通过
- [ ] `pip install -c env/constraints.txt peft accelerate datasets` 且验证 vLLM 未损坏

## 任务

### T3.1 训练配方（照抄原方案，不要发明）

| 项 | 值 | 来源 |
|---|---|---|
| 损失 | 组内 CE：`logit = logit_yes - logit_no`，reshape 成 `(-1, group_size)` | 原方案 |
| 组构成 | 1 正 + `group_size-1` 负，label = 正样本下标 | 原方案 |
| 负样本比 | 先 1:8，够用就不加（原方案 1:24 是为 14B 调的） | 缩放 |
| LoRA | `r=64, alpha=128, all-linear` | 原方案 |
| LoRA+ | lora_A 与 lora_B 用不同 lr | 原方案引用 LoRA+ 论文 |
| attention | **`attn_implementation="sdpa"`** | 本环境无 flash-attn wheel |
| 精度 | bf16 | — |
| 并行 | **单卡，不用 DDP**（P2P=false，双卡 DDP 更慢） | 本环境 |
| 序列 | pad 到 8 的倍数 | 原方案 |

### T3.2 tied embeddings 的处理（**必须显式决策**）

Qwen3-0.6B 是 `tie_word_embeddings=True`。原方案在 14B（untied）上"冻结 lm_head"，
在 0.6B 上照搬会**连带冻结输入 embedding**，行为不同。

二选一，并在 REPORT 里写清楚：
- (a) 显式 untie 后冻结 lm_head（贴近原方案语义）
- (b) 接受连带冻结（更省显存，但与原方案不等价）

**推荐 (b)**：LoRA 已经在所有 linear 上了，embedding 是否训练影响很小，且省事。

### T3.3 训练与评估

- 训练集：`eedi-ranker-silver-v3-teacher-blended-cot`
- 验证：fold 0 的 `eval_set_v1.jsonl`
- 保存：只留 best + last，**不要留 5 个 checkpoint**（省盘）
- merge LoRA 回 base，存 `models/reranker-0.6b-merged/`（供 D6 量化用）

### T3.4 复评

用 D1 的 harness 重跑一次端到端，与 `baseline_zeroshot_*.json` 对比。

## 产出物

- `src/train/train_reranker.py`（单文件，≤ 300 行）
- `conf/train/reranker_0.6b.yaml`
- `models/reranker-0.6b-lora/`、`models/reranker-0.6b-merged/`
- `results/finetuned_reranker_*.json`

## Gate G3

**微调后 `MAP@25` 显著优于 zero-shot。**

⚠️ **不达标不要死磕。** 当天下班前若仍不达标，**直接放弃训练**，全程改用 zero-shot 模型，
把时间还给 D4–D5 主线。推理优化的实验用 zero-shot 模型一样能做，
只是 E5 的"领域精度"叙事弱一点——这是可接受的代价，主线塌了才是灾难。

## 已知坑

- Qwen3-Reranker 官方模板固定，改了会丢掉预训练红利
- yes/no 必须**小写**，token id 用 `convert_tokens_to_ids("yes")` 动态取
- 装 peft 时若不带 constraints，会换掉 torch → vLLM 报废
- group 内的负样本要来自**召回结果**（hard negatives），不是随机采样
- 一个 batch 内同一标签不能作为多个 query 的正样本（in-batch 假负例）——这条对 reranker 的分组 CE 也适用

## 不要做

- ❌ 不训 retriever（zero-shot 的 Qwen3-Embedding 够用）
- ❌ 不训 4B（时间不够，且 D6 要用 4B 做量化对象，保持官方权重更干净）
- ❌ 不做超参搜索
- ❌ 不装 flash-attn
- ❌ 不用 DDP / FSDP / DeepSpeed
