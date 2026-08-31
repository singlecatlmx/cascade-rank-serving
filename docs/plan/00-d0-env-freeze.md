# D0-B · 有卡环境冻结（1–2 小时，1 卡）

> **前置：必须先完成 [00-d0a-nogpu-prep.md](00-d0a-nogpu-prep.md)。**
> 本阶段只做**非 GPU 不可**的事：开卡即跑，跑完即关。**目标 GPU 时长 ≤ 2 小时。**

## 前置条件

- [ ] D0-A 全部出口条件已满足（模型到位、评测集冻结、tokenizer 坑位已验）
- [ ] 实例已切到 **1 卡** 模式
- [ ] `conda activate py312`

## 任务

### T0.4 固化探测脚本

把已跑过的探测逻辑写进 `env/probe.py`，输出 `results/env_report.json`：
GPU / cap / P2P / d2d 带宽 / 量化注册表 / `cuobjdump` 架构 / 磁盘 / 版本。

一个文件、顺序执行、不要抽象。

### T0.5 ★ 量化冒烟（本阶段核心，**零下载**，全部单卡可跑）

共享库已有现成的官方量化权重，直接拿来验证 sm_120 上的 kernel 到底能不能用。
**只关心能否成功加载并产出 logits，不关心质量。**

| 序 | 验证目标 | 模型（`/model/ModelScope/Qwen/`） | 体积 | 预期 |
|---|---|---|---|---|
| 1 | BF16 基线 | `Qwen3-0.6B` | 1.2 G | 必须成功 = **G0** |
| 2 | `fp8`（在线量化）| `Qwen3-0.6B` + `--quantization fp8` | — | 高置信成功 |
| 3 | `gptq_marlin` | `Qwen2.5-14B-Instruct-GPTQ-Int4` | ~9 G | 高置信成功 |
| 4 | `awq_marlin` | `Qwen3-32B-AWQ` | ~18 G | 高置信成功 |
| 5 | `modelopt_fp4` / `mxfp4` | 无现成权重 | — | 看 vLLM 是否直接报 not supported on sm_120 |

> 序 3/4 是本阶段**最有价值的一步**：它把「Marlin 在 sm_120 能不能用」这个影响 E5 实验矩阵形状的最大不确定性，
> 提前到 D0 第一小时就消除了，而且**一字节不用下**。

结果写入 `results/quant_smoke_{date}.json`，**这张表直接决定 E5 的实验矩阵形状**。

### T0.6 确认 attention backend 的正确旋钮
```bash
vllm serve --help 2>&1 | grep -i -B2 -A6 "attention"
sed -n '80,220p' $VLLM/v1/attention/selector.py
```

找到能真正切换 backend 的参数名，记进 AGENTS.md §2。
**不要用 `VLLM_ATTENTION_BACKEND`——它在 0.25.1 已失效且静默。**

## 产出物

- `env/probe.py`
- `results/env_report.json`
- `results/quant_smoke_{date}.json`
- AGENTS.md §2 补入 attention backend 旋钮名与量化实测结论

## Gate G0

**vLLM 能加载 Qwen3-Reranker-0.6B（BF16）并返回 `logprobs`。**
不过则停工排查，**不要开始 D1**。

## 已知坑

- conda 环境是 `py312`，不是 base
- `pip install` 不带 constraints 会换掉 torch，vLLM 立刻报废
- Qwen3-Reranker 需关闭 thinking；yes/no 用**小写**且动态取 token id
- 所有 `.so` 无 PTX → 架构不支持就是彻底不可用，没有 JIT 兜底

## 不要做

- ❌ 不在有卡模式下做任何 D0-A 能做的事（下载 / 装包 / 写纯 CPU 代码）
- ❌ 不写数据处理、不写训练、不写 orchestrator
- ❌ 不下载任何模型（量化冒烟用共享库现成权重，零下载）
- ❌ 不做"顺手优化"
