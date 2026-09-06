#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_DIR"
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py312

if [[ -n "$(git status --porcelain)" ]]; then
  printf 'E4 requires a clean worktree.\n' >&2
  exit 1
fi
if [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -ne 2 ]]; then
  printf 'E4 requires exactly two visible GPUs.\n' >&2
  exit 1
fi
MODEL=/workspace/models/reranker-0.6b-merged
CANDIDATES=/workspace/cache/d2/20260903-074803/recall_top64.jsonl
[[ -f "$MODEL/model.safetensors" && -f "$CANDIDATES" ]] || { printf 'Missing E4 input.\n' >&2; exit 1; }

RUN_TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
WORK_DIR=/workspace/cache/d6/e4/$RUN_TIMESTAMP
mkdir -p "$WORK_DIR"

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 python -m src.bench.zeroshot \
  --model "$MODEL" --labels data/label_pool_v1.jsonl --candidates "$CANDIDATES" \
  --results-dir "$WORK_DIR" --stage e4 --variant d0_full_decode --result-prefix e4_d0 \
  --prompt-variant a1_document_last --submission-mode group --candidate-k 64 \
  --decode-tokens 16 --no-allowed-token-ids >"$WORK_DIR/d0.log" 2>&1 &
PID_D0=$!
CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 python -m src.bench.zeroshot \
  --model "$MODEL" --labels data/label_pool_v1.jsonl --candidates "$CANDIDATES" \
  --results-dir "$WORK_DIR" --stage e4 --variant d1_one_token --result-prefix e4_d1 \
  --prompt-variant a1_document_last --submission-mode group --candidate-k 64 \
  --decode-tokens 1 --no-allowed-token-ids >"$WORK_DIR/d1.log" 2>&1 &
PID_D1=$!
wait "$PID_D0" "$PID_D1"

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 python -m src.bench.zeroshot \
  --model "$MODEL" --labels data/label_pool_v1.jsonl --candidates "$CANDIDATES" \
  --results-dir "$WORK_DIR" --stage e4 --variant d2_allowed_token_ids --result-prefix e4_d2 \
  --prompt-variant a1_document_last --submission-mode group --candidate-k 64 \
  --decode-tokens 1 >"$WORK_DIR/d2.log" 2>&1
mv "$WORK_DIR"/e4_*.json results/
printf 'E4 complete. Logs: %s\n' "$WORK_DIR"
