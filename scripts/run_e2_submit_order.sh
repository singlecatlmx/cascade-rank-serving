#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_DIR"
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py312

if [[ -n "$(git status --porcelain)" ]]; then
  printf 'E2 requires a clean worktree.\n' >&2
  exit 1
fi
if [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -ne 2 ]]; then
  printf 'E2 requires exactly two visible GPUs.\n' >&2
  exit 1
fi
MODEL=/workspace/models/reranker-0.6b-merged
CANDIDATES=/workspace/cache/d2/20260903-074803/recall_top64.jsonl
[[ -f "$MODEL/model.safetensors" && -f "$CANDIDATES" ]] || { printf 'Missing E2 input.\n' >&2; exit 1; }

RUN_TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
WORK_DIR=/workspace/cache/d4/e2/$RUN_TIMESTAMP
mkdir -p "$WORK_DIR"

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 python -m src.bench.zeroshot \
  --model "$MODEL" --labels data/label_pool_v1.jsonl --candidates "$CANDIDATES" \
  --results-dir "$WORK_DIR" --stage e2 --variant b0_grouped --result-prefix e2_b0 \
  --submission-mode candidate --submission-order grouped --candidate-k 64 \
  >"$WORK_DIR/b0.log" 2>&1 &
PID_B0=$!
CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 python -m src.bench.zeroshot \
  --model "$MODEL" --labels data/label_pool_v1.jsonl --candidates "$CANDIDATES" \
  --results-dir "$WORK_DIR" --stage e2 --variant b1_random --result-prefix e2_b1 \
  --submission-mode candidate --submission-order random --candidate-k 64 \
  >"$WORK_DIR/b1.log" 2>&1 &
PID_B1=$!
wait "$PID_B0" "$PID_B1"

for k in 2 4 8 16; do
  CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 python -m src.bench.zeroshot \
    --model "$MODEL" --labels data/label_pool_v1.jsonl --candidates "$CANDIDATES" \
    --results-dir "$WORK_DIR" --stage e2 --variant "b2_interleave_k${k}" --result-prefix "e2_b2_k${k}" \
    --submission-mode candidate --submission-order interleave --interleave-k "$k" --candidate-k 64 \
    >"$WORK_DIR/b2_k${k}.log" 2>&1
done
mv "$WORK_DIR"/e2_*.json results/
printf 'E2 complete. Logs: %s\n' "$WORK_DIR"
