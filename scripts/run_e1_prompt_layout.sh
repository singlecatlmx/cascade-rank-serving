#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_DIR"
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py312

if [[ -n "$(git status --porcelain)" ]]; then
  printf 'E1 requires a clean worktree.\n' >&2
  exit 1
fi
if [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -ne 2 ]]; then
  printf 'E1 requires exactly two visible GPUs.\n' >&2
  exit 1
fi
MODEL=/workspace/models/reranker-0.6b-merged
CANDIDATES=/workspace/cache/d2/20260903-074803/recall_top64.jsonl
[[ -f "$MODEL/model.safetensors" && -f "$CANDIDATES" ]] || { printf 'Missing E1 input.\n' >&2; exit 1; }

RUN_TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
WORK_DIR=/workspace/cache/d4/e1/$RUN_TIMESTAMP
mkdir -p "$WORK_DIR"

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 python -m src.bench.zeroshot \
  --model "$MODEL" --labels data/label_pool_v1.jsonl --candidates "$CANDIDATES" \
  --results-dir "$WORK_DIR" --stage e1 --variant a0_cache_on --result-prefix e1_a0 \
  --prompt-variant a0_document_first --submission-mode group --candidate-k 64 \
  >"$WORK_DIR/a0.log" 2>&1 &
PID_A0=$!
CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 python -m src.bench.zeroshot \
  --model "$MODEL" --labels data/label_pool_v1.jsonl --candidates "$CANDIDATES" \
  --results-dir "$WORK_DIR" --stage e1 --variant a1_cache_on --result-prefix e1_a1 \
  --prompt-variant a1_document_last --submission-mode group --candidate-k 64 \
  >"$WORK_DIR/a1.log" 2>&1 &
PID_A1=$!
wait "$PID_A0" "$PID_A1"

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 python -m src.bench.zeroshot \
  --model "$MODEL" --labels data/label_pool_v1.jsonl --candidates "$CANDIDATES" \
  --results-dir "$WORK_DIR" --stage e1 --variant a2_cache_off --result-prefix e1_a2 \
  --prompt-variant a1_document_last --submission-mode group --candidate-k 64 \
  --disable-prefix-caching >"$WORK_DIR/a2.log" 2>&1
mv "$WORK_DIR"/e1_*.json results/
printf 'E1 complete. Logs: %s\n' "$WORK_DIR"
