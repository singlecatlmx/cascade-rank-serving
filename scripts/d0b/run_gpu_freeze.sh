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
  printf 'Commit or remove local changes before D0-B so results have a clean git commit.\n' >&2
  printf '%s\n' "$DIRTY_STATUS" >&2
  exit 1
fi
if [[ ! -s data/eval_set_v1.jsonl || ! -s data/label_pool_v1.jsonl ]]; then
  printf 'D0-A frozen data is missing.\n' >&2
  exit 1
fi

RERANKER=/workspace/models/qwen3-reranker-0.6b
QWEN_BF16=/model/ModelScope/Qwen/Qwen3-0.6B
QWEN_GPTQ=/model/ModelScope/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4
QWEN_AWQ=/model/ModelScope/Qwen/Qwen3-32B-AWQ
for model_dir in "$RERANKER" "$QWEN_BF16" "$QWEN_GPTQ" "$QWEN_AWQ"; do
  if [[ ! -d "$model_dir" ]]; then
    printf 'Required model directory missing: %s\n' "$model_dir" >&2
    exit 1
  fi
done

if [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -ne 1 ]]; then
  printf 'D0-B requires the instance to expose exactly one GPU.\n' >&2
  exit 1
fi

D0B_TS=$(date -u +%Y%m%d-%H%M%S)
WORK_DIR=/workspace/cache/d0b/$D0B_TS
mkdir -p "$WORK_DIR"

printf '== D0-B environment probe ==\n'
python env/probe.py 2>&1 | tee "$WORK_DIR/probe.log"

run_case() {
  local case_name=$1
  local model_path=$2
  local quantization=$3
  local prompt_mode=$4
  local output_path=$WORK_DIR/$case_name.json
  local log_path=$WORK_DIR/$case_name.log

  printf '\n== Smoke: %s ==\n' "$case_name"
  set +e
  python env/vllm_smoke.py \
    --case "$case_name" \
    --model "$model_path" \
    --quantization "$quantization" \
    --prompt-mode "$prompt_mode" \
    --output "$output_path" \
    2>&1 | tee "$log_path"
  local exit_code=${PIPESTATUS[0]}
  set -e
  printf '%s\t%s\t%s\t%s\n' \
    "$case_name" "$exit_code" "$output_path" "$log_path" >> "$WORK_DIR/manifest.tsv"
  return "$exit_code"
}

run_case g0_reranker_bf16 "$RERANKER" none reranker || {
  printf 'Gate G0 failed. Stop before other GPU experiments.\n' >&2
  exit 1
}
run_case qwen3_bf16 "$QWEN_BF16" none plain || true
run_case qwen3_online_fp8 "$QWEN_BF16" fp8 plain || true
run_case qwen25_gptq_marlin "$QWEN_GPTQ" gptq_marlin plain || true
run_case qwen3_awq_marlin "$QWEN_AWQ" awq_marlin plain || true
run_case qwen3_modelopt_fp4_probe "$QWEN_BF16" modelopt_fp4 plain || true
run_case qwen3_mxfp4_probe "$QWEN_BF16" mxfp4 plain || true

export D0B_TS WORK_DIR
python - <<'PY'
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

work_dir = Path(os.environ["WORK_DIR"])
cases = []
for line in (work_dir / "manifest.tsv").read_text(encoding="utf-8").splitlines():
    name, exit_code, result_path, log_path = line.split("\t")
    path = Path(result_path)
    if path.exists():
        case = json.loads(path.read_text(encoding="utf-8"))
    else:
        log_tail = Path(log_path).read_text(encoding="utf-8", errors="replace")[-4000:]
        case = {"config": {"case": name}, "status": "process_failed", "log_tail": log_tail}
    case["process_exit_code"] = int(exit_code)
    cases.append(case)

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
config = {
    "cuda_visible_devices": "0",
    "tensor_parallel_size": 1,
    "case_order": [case["config"]["case"] for case in cases],
}
timestamp = datetime.now(timezone.utc)
config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
report = {
    "git_commit": commit,
    "vllm_version": metadata.version("vllm"),
    "gpu": cases[0]["gpu"],
    "timestamp": timestamp.isoformat(),
    "config": config,
    "temporary_work_dir": str(work_dir),
    "gate_g0": cases[0]["status"] == "passed",
    "cases": cases,
}
output = Path("results") / (
    f"d0b_quant_smoke_{timestamp.strftime('%Y%m%d-%H%M%S')}_{config_hash}.json"
)
if output.exists():
    raise SystemExit(f"refusing to overwrite {output}")
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"wrote {output}")
for case in cases:
    print(case["config"]["case"], case["status"], case.get("error", ""))
PY

printf '\nD0-B complete. Temporary logs: %s\n' "$WORK_DIR"
git status --short
