#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_DIR"
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py312

if [[ -n "$(git status --porcelain)" ]]; then
  printf 'E3 requires a clean worktree.\n' >&2
  exit 1
fi
MODEL=/workspace/models/reranker-0.6b-merged
CANDIDATES=/workspace/cache/d2/20260903-074803/recall_top64.jsonl
[[ -f "$MODEL/model.safetensors" && -f "$CANDIDATES" ]] || { printf 'Missing E3 input.\n' >&2; exit 1; }

RUN_TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
WORK_DIR=/workspace/cache/d4/e3/$RUN_TIMESTAMP
mkdir -p "$WORK_DIR"
for utilization in 0.5 0.6 0.7 0.8 0.9; do
  tag=${utilization/./}
  CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 python -m src.bench.zeroshot \
    --model "$MODEL" --labels data/label_pool_v1.jsonl --candidates "$CANDIDATES" \
    --results-dir "$WORK_DIR" --stage e3 --variant "gpu_memory_utilization_${utilization}" --result-prefix "e3_u${tag}" \
    --submission-mode candidate --submission-order grouped --candidate-k 64 --gpu-memory-utilization "$utilization" \
    >"$WORK_DIR/u${tag}.log" 2>&1
done
mv "$WORK_DIR"/e3_*.json results/
printf 'E3 complete. Logs: %s\n' "$WORK_DIR"
