# D1 · 测试台地基（1 天，1 卡）

> **这是全项目最重要的一天。** 所有 E1–E7 实验都跑在这套代码上。
> 这里省的每一分钟，后面都会以"结果不可比、全部重跑"的形式加倍还回来。

## 前置条件

- [ ] G0 已通过

## 设计原则

- **一套 harness，配置驱动**。禁止为单个实验另写采集逻辑
- 输入：一个 config（yaml/dataclass）→ 输出：一个结果 JSON
- 不做插件系统、不做注册表、不做基类。就是几个函数

## 任务

### T1.1 `src/metrics/` — 指标实现

| 函数 | 说明 |
|---|---|
| `recall_at_k(preds, gold, k)` | 召回层用，k=32 |
| `map_at_k(preds, gold, k)` | 端到端主指标，k=25 |
| `ndcg_at_k(preds, gold, k)` | 辅助，k=10 |
| `latency_stats(samples)` | 返回 p50 / p95 / p99 / mean |

用 EEDI 官方 MAP@K 定义（`score / min(len(actual), k)`），写 3 个手算样例断言一致。

### T1.2 `src/bench/` — 统一测试台

必须包含四件事：

1. **预热隔离**：前 20 条只跑不计数（避开 CUDA graph 捕获与 cache 冷启动）
2. **延迟打点**：每条请求记 TTFT / E2E
3. **vLLM 指标抓取**：`gpu_prefix_cache_hit_rate`、已用 KV block 数、GPU 利用率
4. **结果落盘**：见下

### T1.3 结果 JSON schema（**冻结后不再改**）

```json
{
  "git_commit": "abc1234",
  "vllm_version": "0.25.1",
  "torch_version": "2.11.0+cu130",
  "gpu": "RTX 5090 x1",
  "timestamp": "20260901-143022",
  "config": { "...完整配置，一字不落..." },
  "metrics": {
    "quality": {"recall@32": 0.0, "map@25": 0.0},
    "latency_ms": {"ttft_p50": 0, "ttft_p99": 0, "e2e_p50": 0, "e2e_p99": 0},
    "throughput": {"req_per_s": 0, "candidates_scored_per_s": 0},
    "resource": {"peak_mem_gb": 0, "prefix_cache_hit_rate": 0.0}
  }
}
```

文件名：`results/{stage}_{variant}_{YYYYMMDD-HHMMSS}_{cfg_hash8}.json`
**绝不覆盖旧结果。**

`git_commit` 用 `subprocess` 取；若工作区脏，后缀 `-dirty`。

### T1.4 冒烟联调

用 Qwen3-Reranker-0.6B + 20 条假数据跑通一次完整流程。

## 产出物

- `src/metrics/`、`src/bench/`
- `results/smoke_{date}.json`（第一个合法结果文件）

## Gate G1

**端到端跑通，产出格式合法的结果 JSON，且 `prefix_cache_hit_rate` 字段有真实数值（非 0 占位）。**
不过则 D2 不许开始。

## 已知坑

- vLLM 0.25.1 的 metrics 接口与旧版不同，**先在 REPL 里 `dir()` 确认字段名再写代码**，别照抄博客
- `candidates_scored_per_s` 才是本项目的吞吐主指标，不是 `tokens/s`
- 峰值显存用 `torch.cuda.max_memory_allocated()` 不够（vLLM 预分配），要读 `nvidia-smi`

## 不要做

- ❌ 不写 pytest / 不建 tests 目录
- ❌ 不封装 logging / CLI 框架
- ❌ 不做 metrics 的抽象基类
- ❌ 不接真实数据（那是 D2 的事）
