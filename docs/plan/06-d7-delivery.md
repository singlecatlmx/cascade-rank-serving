# D7 · 交付（1 天，1 卡）

> 目标：把一堆 JSON 变成**别人 5 分钟能看懂、面试官愿意追问**的仓库。
> 这一天的产出决定项目的实际价值，**不要压缩**。

## 任务

### T7.1 级联 orchestrator demo（半天）

一个最小可用的端到端服务，证明"不只是跑 benchmark，是真能服务"：

- 2 个 vLLM 实例（召回 embedding + 粗排 reranker），FastAPI 异步编排
- 输入 query → 返回 top-25 标签 + 各级耗时拆解
- 用 `vllm bench serve` 或简单并发脚本打一次压测，画 latency–throughput 曲线，标出 knee point

**保持最小**：不做前端、不做鉴权、不做 Docker。一个 `serve.py` + 一个 `bench_serve.py`。

### T7.2 REPORT.md（核心交付物）

结构建议：

| 章节 | 内容 |
|---|---|
| 1. 问题与约束 | 离线 pipeline → 在线服务的改造，硬件约束（sm_120 / P2P=false / 43.6 GB/s）|
| 2. 系统架构 | 四级漏斗图 + 分层图 |
| 3. 评测方法 | 指标定义、评测协议、为什么这样才可比 |
| 4. ★ Prefix Caching | E1/E2/E3 三组消融 + 图 + 与 KVCacheManager 的机理对应 |
| 5. 打分路径与 prefill-bound 调参 | E4 |
| 6. 量化 | E5 + 校准集消融 |
| 7. 部署拓扑 | E6 + 43.6 GB/s 的定量解释 |
| 8. 环境踩坑记录 | sm_120 kernel 覆盖、无 PTX 无 JIT、`VLLM_ATTENTION_BACKEND` 静默失效 |
| 9. 结论与 Future Work | 砍掉的实验都放这里 |

**每个数字都要能追溯到 `results/` 里的某个 JSON 和某个 commit。**

### T7.3 README.md（简历门面）

读者是 5 分钟内决定要不要细看的面试官。必须在**首屏**给出：

- 一句话业务定位（搜索级联排序服务）
- 一张架构图
- **3–5 条量化结论**，形如「prompt 结构调整使排序吞吐提升 N×」「INT4 量化在 MAP@25 掉 X% 的代价下省 Y% 显存」
- 硬件与软件栈（2×RTX5090 sm_120 / vLLM 0.25.1 / CUDA 13）
- 复现命令

### T7.4 复跑验证

随机挑 2–3 个关键结果，在干净环境重跑一次，确认数字能复现。
**不可复现的结论宁可删掉，也不要留在报告里。**

## 产出物

- `src/serve/`（orchestrator + 压测脚本）
- `REPORT.md`、`README.md`
- `assets/` 全部图表
- 最终 commit 并推送

## 已知坑

- 图表要能独立看懂：标题、轴标签、单位、配置说明缺一不可
- 不要用"显著提升"这类词，一律给数字
- 失败的实验（如 MXFP4 不可用）**要写进报告**，这体现工程判断力，不是减分项

## 不要做

- ❌ 不在这一天开新实验
- ❌ 不重构代码
- ❌ 不做 Web 前端 / Dockerfile / CI
- ❌ 不为了让曲线好看而挑选数据
