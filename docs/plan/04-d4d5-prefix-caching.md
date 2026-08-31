# D4–D5 · ★ 主线：Prefix Caching 与调度（2 天，2 卡）

> **这是整个项目的卖点，不允许压缩、不允许砍。**
> 一句话：本 workload 是「1 个 query 打 64 个候选」，前缀 100% 共享 —— 这是别人的
> 通用 benchmark 里做不出来的结构性优势，把它量化清楚就是全部价值。

## 为什么这个 workload 特殊

Qwen3-Reranker 的 prompt 结构：

```
[System + Instruct]        ← 全局共享（所有 query）
[Query 题干 + 错误答案]     ← 同一 query 的 64 个候选共享
[Document = 候选标签 i]     ← 唯一变化的部分，且天然在末尾
[yes/no]
```

同一 query 的 64 次请求中，**63 次的 prefill 几乎全部命中 cache**。
这直接对应 vLLM V1 的 `KVCacheManager` / `BlockPool` / free-block LRU 队列。

## 双卡用法

**GPU0 跑 baseline、GPU1 跑 optimized，同时刻同机器。**
目的不是算力，是**消除时间漂移噪声**——云实例的性能波动足以污染 5–10% 的差异，
而我们要测的收益有些就在这个量级。

## 实验矩阵

### E1 — Prompt 结构 × Prefix Cache（核心，预期最大收益）

| 变体 | 候选标签位置 | prefix caching | 预期 hit rate |
|---|---|---|---|
| A0 | prompt **开头** | on | ~0%（反面对照）|
| A1 | prompt **末尾** | on | ~95% |
| A2 | prompt **末尾** | **off** | 0%（隔离 cache 贡献）|

**测**：`candidates_scored_per_s`、TTFT p50/p99、`prefix_cache_hit_rate`
**要的结论**：A1 相对 A0 的吞吐倍数。这个数字就是标题党素材。

### E2 — 提交顺序 × Cache 淘汰（贴 BlockPool LRU）

| 变体 | 提交策略 | 机理 |
|---|---|---|
| B0 | 按 query 分组连续提交（64 候选相邻）| 前缀 block 常驻 |
| B1 | 全局随机打散 | 前缀 block 被反复 evict |
| B2 | 分组但组间交错 K 个 query（扫 K = 2/4/8/16）| 找 cache 容量的拐点 |

**要的结论**：B2 的 K–hit rate 曲线拐点，与 KV cache 容量的定量关系。
**这是把源码知识变成实测曲线的关键一步。**

### E3 — KV Cache 容量 × hit rate

`gpu_memory_utilization` 扫 0.5 / 0.6 / 0.7 / 0.8 / 0.9，
配合 `block_size`（若可调）扫 16 / 32。

**要的结论**：给定 workload，KV cache 该开多大才不浪费。

## 任务顺序

1. 先跑 E1（半天）→ **G4 判定**
2. G4 通过后跑 E2（一天，是全阶段最有深度的一组）
3. E3（半天）
4. 出图：hit rate 时序图、吞吐柱状图、K–hit rate 曲线

## 产出物

- `scripts/run_e1_prompt_layout.sh` / `run_e2_submit_order.sh` / `run_e3_kv_capacity.sh`
- `results/e1_*.json`、`e2_*.json`、`e3_*.json`
- `assets/` 下 3–4 张图

## Gate G4（D4 当天判定）

**E1 的 A1 vs A0 吞吐差异 ≥ 2×。**

不达标说明主线假设不成立，**当天就要决策换方向**（备选：把 E4 打分路径或 E6 拓扑提为主线），
不要用第二天去"再试试"。

## 已知坑

- vLLM V1 **默认开启 prefix caching**，测 A2 要显式关闭
- 每个变体跑之前必须**重启 engine**，否则上一轮的 cache 会污染结果
- 前 20 条预热不计数
- hit rate 要取**稳态**值，不是全程平均
- 同一组对比必须一次开机跑完
- 关注 block 是否因 `max_num_batched_tokens` 太小而被切碎，影响命中统计

## 不要做

- ❌ 不在这个阶段调量化（那是 D6，会引入混淆变量）
- ❌ 不改模型、不改数据
- ❌ 不做"顺便测测别的"——每多一个自由变量，结论就弱一分
