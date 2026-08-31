# D2 · 数据与 zero-shot 基线（1 天，1 卡）

> 目标：冻结评测集，拿到没有任何优化的**质量与性能基线**。后面所有数字都跟它比。

## 前置条件

- [ ] G1 已通过
- [ ] 磁盘已扩容
- [ ] `KAGGLE_USERNAME` / `KAGGLE_KEY` 已导出

## 任务

### T2.0 模型就位核验

三个核心模型已在 **D0 的 T0.7** 下载完毕（共 10.4 GB，约 11 分钟）。这里只做一次核验：

```bash
du -sh models/qwen3-embedding-0.6b models/qwen3-reranker-0.6b models/qwen3-reranker-4b
ls models/qwen3-reranker-4b/*.safetensors
```

若缺失，回 D0 重跑下载，**不要在 D2 里边下边做其他事**（会把下载拖延计入基线耗时）。

### T2.1 下载数据

作者已公开全部数据集，**不需要调用任何商业 LLM API**：

| 数据集 | 用途 |
|---|---|
| `eedi-mining-misconceptions-in-mathematics` | 原始比赛数据（1.8k MCQ）；用 Kaggle CLI 下载 |
| `conjuring92/eedi-five-folds` | fold 划分（GroupKFold on ConstructId）|
| `conjuring92/eedi-silver-v3` | 合成数据（1.8k + 10.6k MCQ，4791 标签）|
| `conjuring92/eedi-ranker-silver-v3-teacher-blended-cot` | 粗排训练集（含 CoT 与 teacher 分）|

冠军方案的全部数据已在 D0-A 前置下载。本阶段直接使用上表四项：原始竞赛数据使用 `kaggle competitions download`；其余使用 `kagglehub.dataset_download`。

### T2.2 冻结评测集

- 从 **fold 0** 抽 **200 条** query（question + 错误答案）
- 落盘 `data/eval_set_v1.jsonl`，**纳入 git，全程只读**
- 标签库：4791 条，落盘 `data/label_pool_v1.jsonl`

抽样必须 seed 化，并把 seed 写进文件头。

### T2.3 Prompt 模板

沿用 **Qwen3-Reranker 官方模板**（`<Instruct>...<Query>...<Document>`）。

**关键**：候选标签（Document）必须在 **prompt 末尾**——这是 E1 主线实验的物理基础。
同时保留一个"候选在开头"的变体模板，供 E1 的 A0 对照使用。

代码里显式断言 `enable_thinking=False`。

### T2.4 召回基线

`Qwen3-Embedding-0.6B` zero-shot 编码 4791 个标签 + 200 条 query。

- 4791 × 1024 维只有 20 MB，**用暴力矩阵乘即可，不要引入 FAISS**
- 记录 `recall@32` / `recall@64`

### T2.5 端到端 zero-shot 基线

召回 top-32 → `Qwen3-Reranker-0.6B` zero-shot 打分 → 记录 `MAP@25` + 全套性能指标。

这份结果文件命名为 `baseline_zeroshot_*.json`，是后续所有对比的锚点。

## 产出物

- `data/eval_set_v1.jsonl`、`data/label_pool_v1.jsonl`
- `src/data/`（下载 + prompt 构造）
- `results/baseline_recall_*.json`
- `results/baseline_zeroshot_*.json`

## Gate G2

**`recall@32` > 0.6。**
不达标时先查 instruction 拼装与 query/document 侧是否用了正确的编码方式（query 侧带 instruction、document 侧不带），**大概率是 prompt 问题而非模型问题**。

## 已知坑

- Qwen3-Embedding 用 **EOS-token pooling**（取最后一个 `<|endoftext|>` 位置），不是 mean pooling
- query 侧拼 instruction、document 侧不拼，写反了 recall 会掉很多
- kagglehub 缓存位置默认在 `~/.cache`，注意指到扩容后的盘
- MAP@25 的分母是 `min(len(actual), k)`，不是 `k`

## 不要做

- ❌ 不引入 FAISS / Milvus（4791 条用不上）
- ❌ 不下载 4B 以上模型（D2 只用 0.6B）
- ❌ 不做数据增强 / 合成数据生成（作者已公开，直接用）
- ❌ 不调 prompt 去刷分——基线就该是"没优化过的样子"
