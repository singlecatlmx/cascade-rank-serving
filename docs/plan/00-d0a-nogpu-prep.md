# D0-A · 无卡准备（不限时，GPU 关机状态下完成）

> **原则：凡是不需要 GPU 的事，一律在无卡模式下做完。**
> 省钱只是次要收益（约 14 元）；真正的价值是——
> ① tokenizer / prompt / 数据层面的坑全部前置踩完，上卡后不浪费 GPU 时间调试
> ② 写代码时没有计费表在跑，不会为省钱跳过验证
> ③ 上卡后直接用**算好的参数**，不用瞎试

## 前置条件

- [ ] 磁盘已扩容到 **200 GB**
- [ ] GitHub 空私库 `cascade-rank-serving` 已建
- [ ] 实例以**无卡模式**启动
- [ ] `conda activate py312`

---

## 任务

### A.1 本地建仓并推送

```powershell
cd c:\Workspace\Infra\cascade-rank-serving
git init -b main
git add AGENTS.md docs/
git commit -m "docs: AGENTS.md + staged plan"
git remote add origin https://github.com/singlecatlmx/cascade-rank-serving.git
git push -u origin main
```

服务器侧 clone 到容器可写盘。

### A.2 目录与缓存位置

把所有缓存指到扩容后的盘，避免写满系统盘：

```bash
mkdir -p /workspace/{models,data,results,cache}
cat >> ~/.bashrc <<'EOF'
export HF_HOME=/workspace/cache/hf
export MODELSCOPE_CACHE=/workspace/cache/modelscope
export HF_HUB_ENABLE_HF_TRANSFER=1
EOF
source ~/.bashrc
df -h /workspace
```

### A.3 冻结版本约束

```bash
pip freeze > env/env_baseline.txt
python - <<'PY'
import importlib.metadata as md
with open("env/constraints.txt","w") as f:
    for p in ("torch","vllm","transformers"):
        try: f.write(f"{p}=={md.version(p)}\n")
        except Exception: pass
print(open("env/constraints.txt").read())
PY
```

之后**所有** `pip install` 必须带 `-c env/constraints.txt`。

### A.4 装训练栈并验证 vLLM 未损坏

```bash
pip install -c env/constraints.txt \
    modelscope peft accelerate datasets kagglehub hydra-core omegaconf matplotlib

python -c "import torch, vllm; print(torch.__version__, vllm.__version__)"
```

> ⚠️ **无卡模式下 `import vllm` 是否可行需实测**。若 import 时因检测不到 GPU 而报错，
> 说明 harness 里凡 import vllm 的模块都只能上卡后再联调——**把这个结论记进 AGENTS.md §2**，
> 并把 `src/bench/` 拆成「纯 CPU 的指标与数据部分」和「需 vLLM 的执行部分」两个文件。

### A.5 ★ 下载三个核心模型（后台，约 11 分钟）

```bash
cd /workspace && mkdir -p models
nohup bash -c '
modelscope download --model Qwen/Qwen3-Embedding-0.6B --local_dir models/qwen3-embedding-0.6b
modelscope download --model Qwen/Qwen3-Reranker-0.6B  --local_dir models/qwen3-reranker-0.6b
modelscope download --model Qwen/Qwen3-Reranker-4B    --local_dir models/qwen3-reranker-4b
' > models/download.log 2>&1 &
```

HF 直连也可用（实测 200），若 ModelScope 某仓库缺文件，换 `huggingface-cli download`。

校验：`du -sh models/*` → 两个 0.6B 各 ~1.2 G，4B ~8 G。

### A.6 下载数据集

```bash
export KAGGLE_USERNAME=***  KAGGLE_KEY=***
```

只下这四个（其余 7 天内用不到）：

| 数据集 | 用途 |
|---|---|
| `eedi-mining-misconceptions-in-mathematics` | 原始比赛数据 |
| `conjuring92/eedi-five-folds` | fold 划分 |
| `conjuring92/eedi-silver-v3` | 合成数据（12.4k MCQ / 4791 标签）|
| `conjuring92/eedi-ranker-silver-v3-teacher-blended-cot` | 粗排训练集 |

### A.7 ★★ Tokenizer 侧验证（**最容易踩坑的地方，全部无卡可做**）

这一步的价值最高——把 Qwen3 的四个已知坑在无卡阶段一次性验完：

```bash
python - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/workspace/models/qwen3-reranker-0.6b")

# ① yes/no token id：必须是单 token
for w in ("yes","no","Yes","No"):
    ids = tok.encode(w, add_special_tokens=False)
    print(f"{w!r:6} -> {ids}  单token={len(ids)==1}")

# ② thinking 是否被关掉：模板里不应出现 <think>
msgs = [{"role":"user","content":"ping"}]
for flag in (True, False):
    try:
        s = tok.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True, enable_thinking=flag)
        print(f"enable_thinking={flag}: has_think={'<think>' in s}")
    except TypeError as e:
        print(f"enable_thinking={flag}: 模板不接受该参数 -> {e}")

# ③ tied embeddings
import json
cfg = json.load(open("/workspace/models/qwen3-reranker-0.6b/config.json"))
print("tie_word_embeddings =", cfg.get("tie_word_embeddings"))
print("vocab_size =", cfg.get("vocab_size"), "hidden =", cfg.get("hidden_size"))
PY
```

把结论写进 AGENTS.md §3「Qwen3 已知坑」，**上卡后不要再为这些事浪费 GPU 时间**。

### A.8 ★ 序列长度统计（直接决定上卡后的关键参数）

用真实数据构造 prompt 并统计 token 长度分布：

```
p50 / p90 / p99 / max  的 prompt 长度
```

产出两个上卡后立刻要用的参数：

| 参数 | 由什么决定 |
|---|---|
| `max_model_len` | p99 长度 + 余量（**开太大会白白吃掉 KV cache 空间**）|
| `max_num_batched_tokens` | 平均长度 × 目标并发（E4 的扫描区间由此确定）|

顺带算出**共享前缀占比**：`(前缀 token 数) / (总 prompt token 数)`。
这个数字是 E1 实验收益的**理论上界**，无卡就能算出来——如果它只有 30%，
那 D4 主线的预期收益要立刻下调，甚至当场调整方案。**这是最值钱的一次无卡计算。**

### A.9 写 `src/metrics/`（纯 CPU）

`recall@k` / `map@k` / `ndcg@k` / `latency_stats`。
用手算样例断言 MAP@25 定义正确（分母是 `min(len(actual), k)`）。

### A.10 写 `src/data/` 并冻结评测集

- fold 0 抽 200 条 → `data/eval_set_v1.jsonl`（seed 写进文件头，**全程只读**）
- 标签库 4791 条 → `data/label_pool_v1.jsonl`
- prompt 构造：Qwen3-Reranker 官方模板（Document 在**末尾**）+ 一个"候选在开头"的 A0 对照模板

---

## 产出物

- `env/constraints.txt`、`env/env_baseline.txt`
- `models/` 三个核心模型就位
- `data/eval_set_v1.jsonl`、`data/label_pool_v1.jsonl`
- `src/metrics/`、`src/data/`
- **序列长度报告 + 共享前缀占比**（写进 AGENTS.md 或 `results/prompt_stats.json`）
- AGENTS.md §3 的 Qwen3 坑位结论已实测确认

## 出口条件（满足后才开卡）

- [ ] 三个模型下载完整
- [ ] `eval_set_v1.jsonl` 已冻结
- [ ] `src/metrics/` 自测通过
- [ ] yes/no 单 token、thinking 关闭方式、tied embeddings 三项已确认
- [ ] **共享前缀占比已算出**，主线预期收益有量化依据

## 不要做

- ❌ 不试图在无卡模式下跑 vLLM 推理（必然失败，浪费时间）
- ❌ 不写训练脚本（D3 的事）
- ❌ 不写 orchestrator（D7 的事）
- ❌ 不做 GPU 相关的性能猜测——上卡实测
