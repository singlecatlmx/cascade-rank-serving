#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_DIR"
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py312
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2

if [[ -n "$(git status --porcelain)" ]]; then
  printf 'D2 baseline requires a clean worktree.\n' >&2
  git status --short >&2
  exit 1
fi
if [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -ne 1 ]]; then
  printf 'D2 baseline requires exactly one visible GPU.\n' >&2
  exit 1
fi

EMBEDDING=/workspace/models/qwen3-embedding-0.6b
RERANKER=/workspace/models/qwen3-reranker-0.6b
for model in "$EMBEDDING" "$RERANKER"; do
  [[ -f "$model/model.safetensors" ]] || { printf 'Incomplete model: %s\n' "$model" >&2; exit 1; }
done
[[ -s data/eval_set_v1.jsonl && -s data/label_pool_v1.jsonl ]] || {
  printf 'Frozen D2 data is missing.\n' >&2
  exit 1
}

RUN_TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
WORK_DIR=/workspace/cache/d2/$RUN_TIMESTAMP
mkdir -p "$WORK_DIR"
CANDIDATES=$WORK_DIR/recall_top64.jsonl

python -m src.data.recall \
  --model "$EMBEDDING" \
  --eval data/eval_set_v1.jsonl \
  --labels data/label_pool_v1.jsonl \
  --candidates-output "$CANDIDATES" \
  --results-dir "$WORK_DIR" \
  2>&1 | tee "$WORK_DIR/recall.log"

python -m src.bench.zeroshot \
  --model "$RERANKER" \
  --labels data/label_pool_v1.jsonl \
  --candidates "$CANDIDATES" \
  2>&1 | tee "$WORK_DIR/zeroshot.log"

mv "$WORK_DIR"/baseline_recall_*.json results/
printf 'D2 baseline complete. Logs and candidates: %s\n' "$WORK_DIR"
git status --short
