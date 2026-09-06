#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_DIR"
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py312

if [[ -n "$(git status --porcelain)" ]]; then
  printf 'E5 requires a clean worktree.\n' >&2
  exit 1
fi
if [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -ne 2 ]]; then
  printf 'E5 requires exactly two visible GPUs.\n' >&2
  exit 1
fi
MODEL=/workspace/models/reranker-0.6b-merged
CANDIDATES=/workspace/cache/d2/20260903-074803/recall_top64.jsonl
[[ -f "$MODEL/model.safetensors" && -f "$CANDIDATES" ]] || { printf 'Missing E5 input.\n' >&2; exit 1; }

RUN_TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
WORK_DIR=/workspace/cache/d6/e5/$RUN_TIMESTAMP
mkdir -p "$WORK_DIR"

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 python -m src.bench.zeroshot \
  --model "$MODEL" --labels data/label_pool_v1.jsonl --candidates "$CANDIDATES" \
  --results-dir "$WORK_DIR" --stage e5 --variant bf16 --result-prefix e5_bf16 \
  --prompt-variant a1_document_last --submission-mode group --candidate-k 64 \
  >"$WORK_DIR/bf16.log" 2>&1 &
PID_BF16=$!
CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 python -m src.bench.zeroshot \
  --model "$MODEL" --labels data/label_pool_v1.jsonl --candidates "$CANDIDATES" \
  --results-dir "$WORK_DIR" --stage e5 --variant fp8 --result-prefix e5_fp8 \
  --prompt-variant a1_document_last --submission-mode group --candidate-k 64 \
  --quantization fp8 >"$WORK_DIR/fp8.log" 2>&1 &
PID_FP8=$!
wait "$PID_BF16" "$PID_FP8"

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 python -m src.bench.zeroshot \
  --model "$MODEL" --labels data/label_pool_v1.jsonl --candidates "$CANDIDATES" \
  --results-dir "$WORK_DIR" --stage e5 --variant fp8_kv_fp8 --result-prefix e5_fp8_kvfp8 \
  --prompt-variant a1_document_last --submission-mode group --candidate-k 64 \
  --quantization fp8 --kv-cache-dtype fp8 >"$WORK_DIR/fp8_kvfp8.log" 2>&1
mv "$WORK_DIR"/e5_*.json results/
printf 'E5 complete. Logs: %s\n' "$WORK_DIR"
