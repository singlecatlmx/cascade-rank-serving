# AGENTS.md — cascade-rank-serving

> 本文件是本项目的**唯一真相源**，供任意 AI agent 框架（Copilot / Cursor / Codex / Claude Code）读取。
> 开始任何工作前先完整阅读。与本文件冲突的历史对话一律以本文件为准。

---

## 0. 两条最高优先规则

### 规则 1 · 代码保持精简，禁止过度工程化

这是一个 **7 天的 benchmark 项目**，不是要长期维护的库。代码的唯一使命是：**让实验能跑、结果能比、别人能复现**。任何超出这个目标的工程投入都是负资产。

- 只实现当前任务**直接需要**的东西；不预留「以后可能用得上」的抽象、参数、配置层
- 不新增一次性的 helper / 框架 / 包装；一次性操作不抽象成通用工具
- 不为不会发生的场景加防御式代码；只在系统边界（IO / 子进程 / 网络）做校验
- 不给没改动的代码补 docstring / 注释 / 类型标注
- 一个改动能用 30 行讲清就不写 300 行。评审看不懂 = 太复杂，需要拆小或简化
- 完整禁令见 §4

### 规则 2 · 动手前先读环境事实与阶段文档

- **写任何涉及 GPU / vLLM 的代码前**，必须先读 §2「环境硬事实」。`sm_120` + `P2P=false` + vLLM 0.25.1 这个组合与网上绝大多数教程的前提都不同，**不要凭数据中心卡（A100/H100）的经验硬套**
- **开工前先读 `docs/plan/` 里对应阶段那一篇**，不要跨阶段抢跑
- 若某条经验与本文件冲突，在改动说明里明确指出，**不要静默偏离**
- 若某项事实无法验证：**停下并说明**，不要凭空假设后继续写

---

## 1. 项目定位

**一句话**：把一个离线批处理的四级级联排序 pipeline，改造成 GPU 上可服务的低时延系统，并量化每一步优化的「质量 / 延迟 / 成本」代价。

- **产出目标**：AI Infra 岗位作品集仓库（README + REPORT + 可复现实验）
- **不是**：刷榜项目。绝对分数不重要，**相对优化幅度**才是产出
- **业务包装**：搜索 Query-意图标签 级联语义匹配与排序服务
- **数据蓝本**：Kaggle EEDI 竞赛第一名方案（`rbiswasfc/eedi-mining-misconceptions`，MIT）。作者已公开全部数据集，**不需要调用任何商业 LLM API**
- **数据入口（2026-09-01 实测）**：`kagglehub.competition_download("eedi-mining-misconceptions-in-mathematics")` 返回 Resource not found，但竞赛页面仍提供官方 `kaggle competitions download -c eedi-mining-misconceptions-in-mathematics`。原始竞赛数据用 Kaggle CLI；冠军仓库 `download_datasets.py` 列出的 10 个公开 dataset 全部用 `kagglehub.dataset_download(..., output_dir=...)` 前置下载。注意 `path=` 表示数据集内部路径，误传本地目录会导致 API 400。

### 业务映射（写文档时统一用右列术语）

| 原始任务 | 本项目术语 |
|---|---|
| Question + 错误答案 | Query |
| misconception 池（4.8k） | 意图标签库 |
| retriever → top 32~64 | 召回 |
| pointwise ranker → top 8 | 粗排 |
| pointwise ranker → top 5 | 精排 |
| MAP@25 | 排序质量指标 |

---

## 2. 环境硬事实（已实测，不要重新猜测）

| 项 | 值 |
|---|---|
| GPU | 2 × RTX 5090，**sm_120**，单卡 **31.4 GiB** |
| **P2P** | **false（双向）**，卡间实测 **43.6 GB/s** |
| 通信/显存带宽比 | **1 : 41**（H100 NVLink 是 1 : 3.7，差约 11 倍） |
| vLLM | **0.25.1** |
| torch | **2.11.0+cu130** |
| CUDA | runtime 13.0 / nvcc 13.2 |
| Python 环境 | conda env **`py312`**（不是 base），路径 `/usr/local/miniconda3/envs/py312` |
| 内存 | 128 GB；`/dev/shm` **64 GB**（vLLM 多进程通信够用） |
| 容器可写盘 | overlay `/`，**目标扩容至 200 GB**（原始 50 G） |
| 共享模型库 | **`/model/ModelScope/Qwen/` 50 TB** — 已盘点，见 §3.1 |
| 卡数 | 可 1 / 2 / 4 灵活切换 |

### 由硬事实推导的强制约束

1. **禁用 Tensor Parallel**。P2P 关闭 + 43.6 GB/s，TP=2 的 all-reduce 开销约 40%，而最大模型单卡放得下。只用 单卡 / DP / stage-split。
2. **不装 flash-attn 包**。`torch 2.11 + cu130 + sm_120` 无预编译 wheel。训练用 `attn_implementation="sdpa"`。
3. **不用 4 卡**。P2P=false 时多卡只能 DP，4 卡对本项目零增量收益。
4. **pip 必须带 constraints**，锁死 `torch==2.11.0+cu130` 与 `vllm==0.25.1`。任何触发 torch 变更的安装都会让 vLLM 报废。
5. **`VLLM_ATTENTION_BACKEND` 环境变量在 0.25.1 中已不存在**，backend 选择走 `vllm/v1/attention/selector.py`，不要照抄旧版教程。

### vLLM 二进制的 kernel 编译事实（`cuobjdump` 实测，这是硬证据）

| `.so` | 用途 | SASS 架构 | sm_120 |
|---|---|---|---|
| `_C_stable_libtorch` (73M) | 主 kernel 库：量化 GEMM / Marlin / attention | 75, 80, 86, 89, 90, 100, **120** | ✅ |
| `_moe_C_stable_libtorch` (44M) | MoE kernel | 75, 80, 86, 89, 90, 100, **120** | ✅ |
| `_qutlass_C` (1.2M) | **MX 微缩放量化**（MXFP4/MXFP8/NVFP4 的 CUTLASS kernel） | **仅 100** | ❌ |
| `_flashmla_C` | MLA（DeepSeek 专用，与本项目无关） | 90, 100 | ❌ |

**⚠️ 所有 `.so` 的 PTX 段均为空** → **没有 JIT fallback**。架构支持是二元的：编译了就能用，没编译就彻底不可用，没有中间地带。

### 由此推导的量化实验分档

| 组别 | 方法 | 判断 |
|---|---|---|
| **高置信可用** | `fp8` / `fp8_per_tensor` / `fp8_per_channel` / `fp8_per_block` | 主 kernel 库有 sm_120 |
| **高置信可用** | `awq_marlin` / `gptq_marlin` / `awq` / `gptq` | 同上，Marlin 在 `_C` 内 |
| **大概率不可用** | `mxfp4` / `mxfp8` / `modelopt_mxfp8` / `gpt_oss_mxfp4` | 依赖 `_qutlass_C`，**无 sm_120** |
| **需实测** | `modelopt_fp4`（NVFP4） | 走 qutlass 还是 `_C` 未知，冒烟验证 |
| 不用 | `bitsandbytes` | 非服务级方案 |

> ❗ **纠正一个早期判断**：曾认为「FP4/MXFP8 是 5090 独家卖点」。证据显示 vLLM 0.25.1 的 qutlass 只为 `sm_100`（B100/B200 数据中心卡）编译，**RTX 5090 大概率吃不到 MX 系列**。E5 主线因此回到 `BF16 / FP8 / AWQ-INT4 / GPTQ-INT4 / +KV-FP8`，NVFP4 作为待验证的额外一档。
>
> `get_min_capability()` 的返回值**不可信**（`gptq_marlin` 报 60、`awq_marlin` 报 75，而 Marlin 实际需要 sm_80+），它是配置类的通用下界，不是 kernel 的真实要求。**一律以 `cuobjdump` + 冒烟测试为准。**

### Attention backend（sm_120 相关的 4 个候选）

`flash_attn` / `flashinfer` / `triton_attn` / `flex_attention`
（其余 mamba / linear / SSM / rocm / cpu 系列与本项目无关）

**⚠️ `VLLM_ATTENTION_BACKEND` 环境变量在 0.25.1 中已被移除**，且设置后**不报错也不生效**（静默失效）。选择逻辑在 `vllm/v1/attention/selector.py::get_attn_backend`，正确旋钮需从 `vllm serve --help` 或 `AttentionSelectorConfig` 确认。**不要照抄旧版教程。**

---

## 3. 模型阶梯（Qwen3，不要退回 Qwen2.5）

| 层级 | 模型 | 来源 | 处理 |
|---|---|---|---|
| 召回 | `Qwen/Qwen3-Embedding-0.6B` | ModelScope 下载（已验证）| zero-shot |
| 粗排 | `Qwen/Qwen3-Reranker-0.6B` | ModelScope 下载 | **唯一要训练的**（LoRA）|
| 精排 | `Qwen/Qwen3-Reranker-4B` | ModelScope 下载 | 量化 |
| 重排 / Reasoner | — | — | 7 天内**默认砍掉** |

三个共 10.4 GB，**实测约 16 MB/s，全部下完约 11 分钟**。已前置到 D0，不占关键路径。

### 3.1 共享库 `/model` 盘点结果

**`/model` 是只读的**，下载只能落到容器可写盘（这也是磁盘要扩到 200 GB 的原因）。
两个镜像源：`/model/HuggingFace/` 与 `/model/ModelScope/`（另有 comfyui / offline / ollama / other / stable-diffusion-webui，与本项目无关）。

**❌ 盘点完毕：全库没有任何 vLLM 可用的 reranker 或 Qwen3-Embedding**。唯一命中是 `/model/HuggingFace/Voodisss/Qwen3-Reranker-8B-GGUF-llama_cpp`（GGUF + 8B，不可用）；ModelScope 侧的 `Qwen3-Embedding-8B-GGUF` 同理。三个核心模型必须自行下载。

**✅ 网络实测通畅**，下载不依赖管理员：

| 站点 | 状态 |
|---|---|
| modelscope.cn | 302（正常重定向）✅ |
| huggingface.co | 200 ✅ **直连可用** |
| hf-mirror.com | 200 ✅ |
| pypi.org | 200 ✅ |

实测：Qwen3-Embedding-0.6B（1.2 GB / 13 files）**73 秒**下完 ≈ 16 MB/s。

**✅ `Qwen3-0.6B` 目录完整可用**（`model.safetensors` 1.4 GiB + config + tokenizer 齐全），G0 基线与在线 FP8 冒烟可立即开跑。

**共享库里可用于零下载验证 sm_120 kernel 的官方量化权重**：

| 模型 | 体积 | 用途 |
|---|---|---|
| `Qwen3-0.6B` | 1.2 G | G0 基线 + 在线 FP8 量化验证 |
| `Qwen2.5-14B-Instruct-GPTQ-Int4` | ~9 G | 验证 `gptq_marlin`，单卡可放 |
| `Qwen3-32B-AWQ` | ~18 G | 验证 `awq_marlin`，单卡可放 |
| `Qwen3-32B-FP8` | ~33 G | 验证离线 FP8（单卡放不下，优先用在线 FP8 代替）|

其余可用但**本项目不用**：`Qwen3-4B` / `Qwen3-8B` / `Qwen3-14B` / `Qwen3-32B`（均为 instruct 版，**无 Base 版**）。

**备用方案**：若 Qwen3-Reranker 下载受阻，用共享库的 `Qwen3-0.6B` / `Qwen3-4B` 自行训 yes/no reranker（即原方案做法），代价是丢掉专用 reranker 的预训练红利。

**不用的**：Qwen3.5 / 3.6 / 3.8 系列（新架构、尺寸偏大、无对应 Embedding/Reranker 专用版，7 天冲刺不冒这个险）、`Qwen3-VL-Embedding-8B`（多模态且过大）、所有 GGUF。

### Qwen3 已知坑（踩过一次就够了）

- **thinking 必须关闭**。排序只要 1 个 token 的 yes/no logit，`<think>` 块是纯污染。用 Base 模型，或 `apply_chat_template(..., enable_thinking=False)`。**代码里要有显式断言**。
- **yes/no 用小写**，且 token id 必须 `tokenizer.convert_tokens_to_ids("yes")` 动态取，禁止硬编码。
- **0.6B / 1.7B / 4B 是 tied embeddings**。"冻结 lm_head" 会连带冻结输入 embedding，行为与 14B 不同，必须显式处理并在报告中说明。
- 沿用 Qwen3-Reranker **官方 prompt 模板**（`<Instruct>...<Query>...<Document>`）。好处：Document 天然在末尾，与 prefix caching 诉求吻合。

---

## 4. 开发规范（★ 反过度工程化，最高优先级）

> 这是一个 **7 天的 benchmark 项目**，不是要长期维护的库。
> 代码的唯一使命是：**让实验能跑、结果能比、别人能复现**。任何超出这个目标的工程投入都是负资产。

### 4.1 规模纪律

- 单文件 **≤ 300 行**。超了先问人，不要自动拆分
- 目录深度 **≤ 3 层**
- hydra / yaml 配置层级 **≤ 2 层**
- 新增第三方依赖前，必须先说明「标准库或现有依赖为什么做不到」

### 4.2 明令禁止

| ❌ 禁止 | 原因 |
|---|---|
| 抽象基类 / 插件注册表 / 工厂模式 | 除非已有 **≥ 3 个**具体实现，否则一律直接写 |
| `try/except` 兜底不可能发生的错误 | 只在真实边界（IO、子进程、网络）做异常处理 |
| "以后可能用到"的参数 / 开关 | 用不到就不写 |
| 单元测试脚手架、pytest 目录、mock | benchmark 项目不需要；正确性靠实验结果自证 |
| 封装 logging / CLI / 配置加载 | 直接用 `print` / `argparse` / `OmegaConf` |
| Dockerfile / CI / pre-commit | 服务器是租的容器，没有 CI 需求 |
| Web 前端 / 可视化服务 | 图表用 matplotlib 出静态图存 `assets/` |
| 重构已跑通的代码 | 除非它**阻塞**了新实验 |
| 无意义的 docstring / 类型注解补全 | 只给非显而易见的逻辑写一行注释 |
| 校验和 / SHA / 完整性检查 | 不是发行版软件 |

### 4.3 必须做（这几条不能省）

- ✅ 每个结果 JSON 自带 `git_commit` + `vllm_version` + `gpu` + **完整 config** + `timestamp`
- ✅ 固定随机种子；评测集一次冻结、全程不变
- ✅ 每个实验能用**一条命令**复现，脚本放 `scripts/`，**一个实验一个脚本，直白顺序执行，不追求复用**
- ✅ 所有实验共用 `src/bench/` 这一套测试台；**禁止为单个实验另写采集逻辑**（这是结果不可比的头号原因）

### 4.4 代码风格

- 直白 > 聪明。宁可重复三行代码，不要为复用造一层抽象
- 函数式脚本优先，能不写 class 就不写
- 变量名可长，逻辑不可绕

### 4.5 标准工作流（每个任务都照做）

1. 读 `docs/plan/` 对应阶段那一篇 + §2 环境事实，**别急着写**
2. 做**最小**改动实现目标
3. **先跑对，再跑快**：新配置先与 baseline 对齐质量指标，再谈吞吐
4. 跑 `src/bench/` 测试台，记录质量 + P50/P95/P99 + hit rate
5. **只有 E2E 不回退，才把新配置设为默认**
6. 改动说明附：文件清单 + 服务器命令 + git 命令 + 实测数字（见 §9）

### 4.6 不可触碰的约束

- **不改动已冻结的评测集与随机种子**。一旦 `eval_set_v1.jsonl` 生成，全程只读
- **不为「让数字好看」放宽评测协议**（跳过预热轮、换评测集、只报最好的一次 run）
- **不把 microbenchmark 快、但 E2E 无收益的配置设为默认**
- **不静默 fallback**：不支持的量化档 / backend 要 fail closed 并报错退出，绝不悄悄退回默认路径——否则测出来的全是假数据
- **每个优化都要能单独开关**（feature flag），便于把收益归因到单一变量
- BF16 未量化路径始终保留为可一键回退的 baseline

---

## 5. 目录结构

```
cascade-rank-serving/
├── AGENTS.md              # 本文件（唯一真相源）
├── README.md              # 业务包装 + 架构图 + 结论摘要（简历门面）
├── REPORT.md              # benchmark 报告，图表主战场
├── docs/plan/             # ★ 分阶段任务书，开工前读对应那一篇
│   ├── README.md          #   导航 + 全局时间表 + Gate 总表
│   ├── 00-d0-env-freeze.md
│   ├── 01-d1-bench-harness.md
│   ├── 02-d2-data-baseline.md
│   ├── 03-d3-train-reranker.md
│   ├── 04-d4d5-prefix-caching.md
│   ├── 05-d6-ablations.md
│   └── 06-d7-delivery.md
├── env/
│   ├── probe.py           # 环境探测 → env_report.json
│   ├── constraints.txt    # 锁死 torch / vllm 版本
│   └── setup.sh           # 训练栈安装
├── conf/                  # 一个实验变体一个 yaml
├── src/
│   ├── data/              # 下载、fold 划分、prompt 构造
│   ├── metrics/           # recall@32 / MAP@25 / NDCG / 延迟统计
│   ├── bench/             # ★ 统一测试台（地基，最先写、最后改）
│   ├── serve/             # vLLM 封装 + 级联 orchestrator
│   └── train/             # Reranker-0.6B LoRA（SDPA）
├── scripts/               # run_e1_*.sh 等一键实验
├── results/               # 结果 JSON（体积小，纳入 git）
└── assets/                # 图表
```

---

## 6. 指标与实验协议

### 6.1 指标定义

| 类别 | 指标 |
|---|---|
| 质量 | `recall@32`（召回）、`MAP@25`（端到端）、`NDCG@10` |
| 延迟 | TTFT / E2E 的 p50 / p95 / p99 |
| 吞吐 | req/s、prefill tokens/s、**candidates-scored/s**（本项目主指标） |
| 资源 | 峰值显存、GPU 利用率、**prefix cache hit rate** |
| 成本 | **GPU-秒 / 千次查询**（对外讲故事的最终口径） |

### 6.2 评测协议（违反则结果作废）

1. 固定评测集 `eval_set_v1.jsonl`（fold 0 抽 200 条），**全程不变**
2. 所有采样 / shuffle 必须 seed 化
3. 前 20 条为预热轮，不计入统计（避开 CUDA graph 捕获与 cache 冷启动）
4. **一组对比实验必须在同一台实例、同一次开机内跑完**。跨实例、跨重启的数字不可比

### 6.3 结果文件命名

```
results/{stage}_{variant}_{YYYYMMDD-HHMMSS}_{cfg_hash8}.json
```

**绝不覆盖旧结果。** 换了配置就换文件名。

---

## 7. 实验矩阵

### 主线（不允许压缩）

**E1 — Prompt 结构 × Prefix Cache**
| 变体 | 候选标签位置 | 预期 hit rate |
|---|---|---|
| A0 | prompt 开头 | ~0%（反面对照） |
| A1 | prompt 末尾 | ~95% |
| A2 | 末尾 + 关闭 prefix caching | 0%（隔离 cache 贡献） |

**E2 — 提交顺序 × Cache 淘汰**（对应 KVCacheManager / BlockPool LRU）
| 变体 | 策略 |
|---|---|
| B0 | 按 query 分组连续提交 |
| B1 | 全局随机打散 |
| B2 | 分组但组间交错 K 个 query（扫 K） |

**E3 — KV Cache 容量 × hit rate**：`gpu_memory_utilization` 扫 0.5 / 0.6 / 0.7 / 0.8 / 0.9

### 对照实验

**E4 — 打分路径**：生成完整回答 / `max_tokens=1`+`logprobs` / `+allowed_token_ids` / vLLM 原生 score 路径

**E5 — 量化**：`BF16 / FP8 / (AWQ-INT4 · GPTQ-INT4 若 Marlin 可用) / MXFP8 / NVFP4 / +KV-FP8`
四维记录「MAP@25 × 显存 × 吞吐 × TTFT」。
**关键消融**：通用校准集 vs 任务相关校准集（原方案的核心发现）。

**E6 — TP vs DP 拓扑**：单卡 1 实例 / TP=2 1 实例 / 2 卡各 1 实例（DP=2）。用 43.6 GB/s 实测带宽做理论解释。

**E7 — Attention backend**：`flash_attn` / `flashinfer` / `triton_attn` / `flex_attention`。
本 workload 是**纯 prefill-bound**（只出 1 个 token），backend 的 prefill 性能差异被放大。

---

## 8. 7 天计划与切卡策略

| Day | 阶段 | 卡数 | Gate |
|---|---|---|---|
| **D0-A** | **无卡准备**：下模型/数据、冻评测集、验 tokenizer、算共享前缀占比 | **0** | 出口条件见 `docs/plan/00-d0a-nogpu-prep.md` |
| D0-B | 有卡环境冻结 + 量化冒烟 | 1 | vLLM 加载 0.6B 并返回 logprobs |
| D1 | `src/bench/` 地基 + metrics | 1 | 端到端跑通，产出合法结果 JSON |
| D2 | 数据 + zero-shot 基线 | 1 | `recall@32` > 0.6 |
| D3 | Reranker-0.6B LoRA 训练 | 1 | MAP@25 显著优于 zero-shot |
| **D4–D5** | ★ Prefix Caching 主线（E1–E3） | **2** | prompt 结构消融跑出 **≥ 2× 吞吐差异** |
| D6 | E4 / E5 / E6 / E7 | **2** | — |
| D7 | orchestrator demo + README + REPORT | 1 | — |

**成本约束**：单卡 2.43 元/h，双卡 4.86 元/h，全程约 46 GPU-小时 ≈ 185 元。
**凡不需要 GPU 的工作一律放无卡模式做**；开卡后只做非 GPU 不可的事，跑完即关。

**超时时的砍除顺序**：E7 → E6 → E5 → D3 训练（退化为纯 zero-shot）。
**D4–D5 主线不允许压缩。**

2 卡阶段的用法是 **GPU0 跑 baseline、GPU1 跑 optimized**，同时刻同机器消除时间漂移噪声——不是为了算力。

---

## 9. 与人协作的沟通规范

用户在 Windows 本机开发（**无 Python 环境**），实验在远程 Linux 服务器跑，**文件靠手动同步**。因此：

1. **每次改动结束，必须给出三件套**：
   - 改了 / 新建了哪些文件（逐个列全，漏一个就会跑到旧代码）
   - 服务器上依次执行哪些命令（标注**预计耗时**与**顺序依赖**）
   - 对应的 git 命令（并说清当前仓库状态：已提交几个 / 有没有推送 / 服务器要不要 pull）
2. 结果写**新文件名**，不覆盖旧结果——否则看不出跑的是不是新代码
3. 不要只说"已提交"就结束，用户看不到本地终端

### Git

- 远程：GitHub 私库 `cascade-rank-serving`，分支 `main`
- 服务器用只读 deploy key 拉取
- `results/*.json` 体积小，**纳入 git**（这样报告里每个数字都能追溯到 commit）
- 模型权重、量化产物、HF cache **不进 git**

---

## 10. 参考与导航

| 需要什么 | 去哪 |
|---|---|
| 当前阶段该做什么 | `docs/plan/README.md` → 对应阶段那一篇 |
| 环境能不能跑某个量化档 | §2「kernel 编译事实」，不确定就 `cuobjdump` 实测 |
| 方案蓝本 / 原始 writeup | `rbiswasfc/eedi-mining-misconceptions`（MIT） |
| 指标怎么算、结果怎么存 | §6 |
| 实验变体清单 | §7 |

---

## 11. 当前进度

- [x] 需求分析、业务包装、技术选型
- [x] 环境探测：GPU / P2P / 量化注册表 / attention backend / **cuobjdump kernel 架构**
- [x] AGENTS.md + `docs/plan/` 分阶段任务书
- [ ] **磁盘扩容至 200 GB**（阻塞 D2 之后所有工作）
- [ ] 查 `/model` 共享库是否预置 Qwen3（可省数十 GB 与下载时间）
- [ ] D0：建仓 → `env/probe.py` 固化 → 量化冒烟三连 → 冻结 `constraints.txt`
