#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_DIR"
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py312

VLLM_IMPORT_LOG=/workspace/d0a-logs/import_vllm_after_install.log
if [[ ! -f "$VLLM_IMPORT_LOG" ]]; then
  printf 'Missing no-GPU vLLM import log. Run run_nogpu_prep.sh first.\n' >&2
  exit 1
fi
if grep -q 'IMPORT_VLLM_OK' "$VLLM_IMPORT_LOG"; then
  printf 'No-GPU import vllm: PASS\n'
else
  printf 'No-GPU import vllm: FAIL (CPU modules remain independent of vLLM)\n'
fi

for required_path in \
  /workspace/data/raw/competition/train.csv \
  /workspace/data/raw/competition/misconception_mapping.csv \
  /workspace/data/raw/eedi-five-folds/folds.parquet \
  /workspace/models/qwen3-embedding-0.6b/config.json \
  /workspace/models/qwen3-reranker-0.6b/config.json \
  /workspace/models/qwen3-reranker-4b/config.json; do
  if [[ ! -f "$required_path" ]]; then
    printf 'Missing D0-A input: %s\n' "$required_path" >&2
    exit 1
  fi
done

for dataset_dir in \
  eedi-five-folds eedi-silver-v3 eedi-embed-pretrain-mix-final \
  eedi-embed-mix-silver-v3 eedi-ranker-silver-v3-teacher-blended-cot \
  eedi-tutor-mix-v8 eedi-cot-sonnet-6k eedi-cot-train-silver-v3 \
  eedi-misconception-clusters eedi-cot-gen-base; do
  dataset_path="/workspace/data/raw/$dataset_dir"
  if [[ ! -d "$dataset_path" ]] || \
    ! find "$dataset_path" -type f -print -quit | grep -q .; then
    printf 'Missing downloaded dataset: %s\n' "$dataset_path" >&2
    exit 1
  fi
done

for model_dir in \
  /workspace/models/qwen3-embedding-0.6b \
  /workspace/models/qwen3-reranker-0.6b \
  /workspace/models/qwen3-reranker-4b; do
  if ! find "$model_dir" -maxdepth 1 -type f -name '*.safetensors' -print -quit | grep -q .; then
    printf 'Missing model weights in: %s\n' "$model_dir" >&2
    exit 1
  fi
done

if [[ -f data/eval_set_v1.jsonl && -f data/label_pool_v1.jsonl ]]; then
  printf 'Reusing frozen eval_set_v1.jsonl and label_pool_v1.jsonl.\n'
elif [[ -e data/eval_set_v1.jsonl || -e data/label_pool_v1.jsonl ]]; then
  printf 'Only one frozen data file exists; refusing to continue.\n' >&2
  exit 1
else
  python -m src.data.prepare_eval
fi

python -m src.metrics.metrics
python -m src.data.prompt_stats

printf '\nD0-A finalize complete. New files:\n'
git status --short
