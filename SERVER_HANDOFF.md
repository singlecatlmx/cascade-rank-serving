# 服务器新窗口交接提示词

请在服务器 `/workspace/cascade-rank-serving` 中接手 `cascade-rank-serving` 项目。

开始任何操作前，完整阅读：

1. `AGENTS.md`（唯一真相源）
2. `docs/plan/README.md`
3. 当前阶段 `docs/plan/00-d0-env-freeze.md`

工作方式：

- 现在直接在 Linux 服务器上开发、运行、提交和 push；不要再设计 Windows 手动同步流程。
- GitHub 是公共仓库，服务器有写权限。每次实验前保持工作区 clean，结果 JSON 直接 commit/push。
- 当前仓库已主动清理为规划基线：只保留 `AGENTS.md`、`docs/plan/`、Git 配置与本交接文件。旧运行代码、仓库内数据快照和旧结果已从工作树删除，这是预期状态；不要整批恢复，只按当前 Gate 最小重建。
- `/workspace/data`、`/workspace/models`、`/workspace/cache` 在仓库外；先只读盘点，存在就复用，禁止重复下载。
- 优先给阶段级一键脚本；命令保持最少。禁止过度工程化，单文件不超过 300 行。
- 不安装 flash-attn，不使用 Tensor Parallel。若确实需要安装依赖，先按 `AGENTS.md` 的固定版本重建 `env/constraints.txt`，任何 pip install 必须带该 constraints。

已确认事实：

- 无卡 `import vllm` 成功。
- D0-A 有效结果：A0 shared prefix mean 23.12%，A1 mean 88.37%。
- 4791 标签映射来自 `eedi-silver-v3/misconception_mapping.csv`，不是官方竞赛的 2587 条映射。
- Qwen3-Reranker 官方 prompt 的 thinking 关闭形态包含一个空 `<think>\n\n</think>` suffix，不能断言完全没有 `<think>` 标签。
- vLLM 完整 CLI 帮助使用 `vllm serve --help=all`；attention backend 旋钮为 `--attention-backend`。
- D0-B 已进入 G0，但 `g0_reranker_bf16` 失败，后续量化实验未运行。

当前第一任务：

1. 先执行只读检查：`git status --short`，确认 GPU、模型、数据和缓存仍在。
2. 读取并分析：`/workspace/cache/d0b/20260901-092130/g0_reranker_bf16.json` 和同目录日志。
3. 明确失败根因后，只重建 D0-B 当前 Gate 直接需要的最小文件并修复；不要重写完整框架，也不要直接恢复全部旧运行文件。
4. 先单独重跑 G0。只有 Reranker-0.6B BF16 返回 yes/no logprobs，才继续 FP8/GPTQ/AWQ 冒烟。
5. 成功结果写唯一文件名并直接 commit/push；失败证据保留在 `/workspace/cache/d0b/`，不要把大日志放进 Git。

每次回复必须给三件套：实际改动文件、服务器最少执行命令及预计时间、Git 状态与核心命令。
