#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_DIR"
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py312

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2

DIRTY_STATUS=$(git status --porcelain)
if [[ -n "$DIRTY_STATUS" ]]; then
  printf 'G0 requires a clean worktree so its git_commit is reproducible:\n%s\n' \
    "$DIRTY_STATUS" >&2
  exit 1
fi

MODEL=/workspace/models/qwen3-reranker-0.6b
if [[ ! -d "$MODEL" ]]; then
  printf 'Required model directory missing: %s\n' "$MODEL" >&2
  exit 1
fi

RUN_TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
WORK_DIR=/workspace/cache/d0b/$RUN_TIMESTAMP
mkdir -p "$WORK_DIR"
TEMP_RESULT=$WORK_DIR/g0_reranker_bf16.json

set +e
python env/vllm_smoke.py \
  --model "$MODEL" \
  --output "$TEMP_RESULT" \
  2>&1 | tee "$WORK_DIR/g0_reranker_bf16.log"
EXIT_CODE=${PIPESTATUS[0]}
set -e

if [[ $EXIT_CODE -ne 0 ]]; then
  printf 'Gate G0 failed. Evidence: %s\n' "$WORK_DIR" >&2
  exit "$EXIT_CODE"
fi

CONFIG_HASH=$(
  python -c 'import hashlib,json,sys; c=json.load(open(sys.argv[1]))["config"]; print(hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:8])' \
    "$TEMP_RESULT"
)
mkdir -p results
RESULT=results/d0b_g0_bf16_${RUN_TIMESTAMP}_${CONFIG_HASH}.json
if [[ -e "$RESULT" ]]; then
  printf 'Refusing to overwrite %s\n' "$RESULT" >&2
  exit 1
fi
cp "$TEMP_RESULT" "$RESULT"

printf 'Gate G0 passed. Result: %s\nLog: %s\n' \
  "$RESULT" "$WORK_DIR/g0_reranker_bf16.log"
