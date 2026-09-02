#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_DIR"
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py312
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2

if [[ -n "$(git status --porcelain)" ]]; then
  printf 'D1 smoke requires a clean worktree.\n' >&2
  git status --short >&2
  exit 1
fi
if [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -ne 1 ]]; then
  printf 'D1 smoke requires exactly one visible GPU.\n' >&2
  exit 1
fi

RUN_TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
WORK_DIR=/workspace/cache/d1/$RUN_TIMESTAMP
mkdir -p "$WORK_DIR"
TEMP_RESULT=$WORK_DIR/smoke.json
python -m src.bench.smoke \
  --model /workspace/models/qwen3-reranker-0.6b \
  --warmup 20 --requests 20 --output "$TEMP_RESULT" \
  2>&1 | tee "$WORK_DIR/smoke.log"
CONFIG_HASH=$(python -c 'import hashlib,json,sys; c=json.load(open(sys.argv[1]))["config"]; print(hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:8])' "$TEMP_RESULT")
RESULT=results/d1_smoke_${RUN_TIMESTAMP}_${CONFIG_HASH}.json
if [[ -e "$RESULT" ]]; then
  printf 'Refusing to overwrite %s\n' "$RESULT" >&2
  exit 1
fi
mkdir -p results
mv "$TEMP_RESULT" "$RESULT"
printf 'D1 smoke complete. Result: %s\nLogs: %s\n' "$RESULT" "$WORK_DIR"
