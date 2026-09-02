#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_DIR"
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py312
export CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2

if [[ -n "$(git status --porcelain)" ]]; then
  printf 'D0-B requires a clean worktree.\n' >&2
  git status --short >&2
  exit 1
fi
if [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -ne 1 ]]; then
  printf 'D0-B requires exactly one visible GPU.\n' >&2
  exit 1
fi

declare -A MODELS=(
  [g0_reranker_bf16]=/workspace/models/qwen3-reranker-0.6b
  [qwen3_bf16]=/model/ModelScope/Qwen/Qwen3-0.6B
  [qwen3_online_fp8]=/model/ModelScope/Qwen/Qwen3-0.6B
  [qwen25_gptq_marlin]=/model/ModelScope/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4
  [qwen3_awq_marlin]=/model/ModelScope/Qwen/Qwen3-32B-AWQ
)
for model in "${MODELS[@]}"; do
  [[ -d "$model" ]] || { printf 'Missing model: %s\n' "$model" >&2; exit 1; }
done

RUN_TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
WORK_DIR=/workspace/cache/d0b/$RUN_TIMESTAMP
mkdir -p "$WORK_DIR"
python env/probe.py 2>&1 | tee "$WORK_DIR/probe.log"

run_case() {
  local case_name=$1 model=$2 quantization=$3 prompt_mode=$4
  local output="$WORK_DIR/$case_name.json"
  set +e
  python env/vllm_smoke.py --case "$case_name" --model "$model" \
    --quantization "$quantization" --prompt-mode "$prompt_mode" --output "$output" \
    2>&1 | tee "$WORK_DIR/$case_name.log"
  local code=${PIPESTATUS[0]}
  set -e
  printf '%s\t%s\n' "$case_name" "$code" >> "$WORK_DIR/manifest.tsv"
  return "$code"
}

run_case g0_reranker_bf16 "${MODELS[g0_reranker_bf16]}" none reranker
run_case qwen3_bf16 "${MODELS[qwen3_bf16]}" none plain || true
run_case qwen3_online_fp8 "${MODELS[qwen3_online_fp8]}" fp8 plain || true
run_case qwen25_gptq_marlin "${MODELS[qwen25_gptq_marlin]}" gptq_marlin plain || true
run_case qwen3_awq_marlin "${MODELS[qwen3_awq_marlin]}" awq_marlin plain || true
run_case qwen3_modelopt_fp4_probe "${MODELS[qwen3_bf16]}" modelopt_fp4 plain || true
run_case qwen3_mxfp4_probe "${MODELS[qwen3_bf16]}" mxfp4 plain || true

python - "$WORK_DIR" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

work_dir = Path(sys.argv[1])
cases = []
for line in (work_dir / "manifest.tsv").read_text().splitlines():
    name, code = line.split("\t")
    path = work_dir / f"{name}.json"
    case = json.loads(path.read_text()) if path.exists() else {"config": {"case": name}, "status": "process_failed"}
    case["process_exit_code"] = int(code)
    cases.append(case)
config = {"cuda_visible_devices": "0", "tensor_parallel_size": 1, "case_order": [c["config"]["case"] for c in cases]}
timestamp = datetime.now(timezone.utc)
digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
output = Path("results") / f"d0b_quant_smoke_{timestamp.strftime('%Y%m%d-%H%M%S')}_{digest}.json"
if output.exists():
    raise SystemExit(f"refusing to overwrite {output}")
report = {
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "vllm_version": metadata.version("vllm"),
    "gpu": cases[0].get("gpu", "unknown"),
    "timestamp": timestamp.isoformat(),
    "config": config,
    "temporary_work_dir": str(work_dir),
    "gate_g0": cases[0].get("status") == "passed",
    "cases": cases,
}
output.write_text(json.dumps(report, indent=2) + "\n")
print(f"wrote {output}")
PY

printf 'D0-B complete. Logs: %s\n' "$WORK_DIR"
