#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_DIR"
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py312
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4

if [[ -n "$(git status --porcelain)" ]]; then
  printf 'D3 training requires a clean worktree.\n' >&2
  git status --short >&2
  exit 1
fi
if [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -ne 1 ]]; then
  printf 'D3 training requires exactly one visible GPU.\n' >&2
  exit 1
fi
for required in \
  /workspace/models/qwen3-reranker-0.6b/model.safetensors \
  /workspace/data/raw/eedi-ranker-silver-v3-teacher-blended-cot/train.parquet \
  /workspace/cache/d2/20260903-074803/recall_top64.jsonl \
  data/label_pool_v1.jsonl; do
  [[ -f "$required" ]] || { printf 'Missing D3 input: %s\n' "$required" >&2; exit 1; }
done
for output in /workspace/models/reranker-0.6b-lora /workspace/models/reranker-0.6b-merged; do
  [[ ! -e "$output" ]] || { printf 'Refusing to overwrite: %s\n' "$output" >&2; exit 1; }
done

RUN_TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
WORK_DIR=/workspace/cache/d3/$RUN_TIMESTAMP
mkdir -p "$WORK_DIR"

python -m src.train.train_reranker --smoke-steps 2 2>&1 | tee "$WORK_DIR/smoke.log"
python -m src.train.train_reranker 2>&1 | tee "$WORK_DIR/train.log"
python -m src.bench.zeroshot \
  --model /workspace/models/reranker-0.6b-merged \
  --labels data/label_pool_v1.jsonl \
  --candidates /workspace/cache/d2/20260903-074803/recall_top64.jsonl \
  --stage finetuned \
  --variant reranker_0.6b_lora \
  --result-prefix finetuned_reranker \
  --training-config conf/train/reranker_0.6b.yaml \
  2>&1 | tee "$WORK_DIR/eval.log"

printf 'D3 complete. Logs: %s\n' "$WORK_DIR"
git status --short
