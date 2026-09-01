#!/usr/bin/env bash
set -euo pipefail
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ ! -f "$REPO_DIR/AGENTS.md" || ! -d "$REPO_DIR/.git" ]]; then
  printf 'Cannot locate the cascade-rank-serving repository.\n' >&2
  exit 1
fi
cd "$REPO_DIR"
WORK_DIR=/workspace
LOG_DIR=$WORK_DIR/cache/d0a/logs
MODEL_DIR=$WORK_DIR/models
RAW_DATA_DIR=$WORK_DIR/data/raw
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py312
mkdir -p env "$LOG_DIR" "$MODEL_DIR" "$RAW_DATA_DIR"
mkdir -p "$WORK_DIR/cache/hf"
mkdir -p "$WORK_DIR/cache/modelscope" "$WORK_DIR/cache/pip"
export HF_HOME=$WORK_DIR/cache/hf
export MODELSCOPE_CACHE=$WORK_DIR/cache/modelscope
export PIP_CACHE_DIR=$WORK_DIR/cache/pip
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
for setting in \
  'export HF_HOME=/workspace/cache/hf' \
  'export MODELSCOPE_CACHE=/workspace/cache/modelscope' \
  'export PIP_CACHE_DIR=/workspace/cache/pip' \
  'export HF_HUB_ENABLE_HF_TRANSFER=1'; do
  grep -qxF "$setting" ~/.bashrc || printf '%s\n' "$setting" >> ~/.bashrc
done
printf '== Preflight ==\n'
python --version
which python
nproc
free -h
df -h "$WORK_DIR"
git status --short --branch
git rev-parse HEAD
printf '\n== Freeze installed versions ==\n'
python -m pip freeze > env/env_baseline.txt
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import torch
versions = {name: version(name) for name in ("torch", "vllm", "transformers")}
print("package metadata:", versions)
print("torch runtime:", torch.__version__)
if versions["torch"] != "2.11.0":
    raise SystemExit(f"unexpected torch metadata version: {versions['torch']}")
if torch.__version__ != "2.11.0+cu130":
    raise SystemExit(f"unexpected torch runtime version: {torch.__version__}")
if versions["vllm"] != "0.25.1":
    raise SystemExit(f"unexpected vllm version: {versions['vllm']}")
Path("env/constraints.txt").write_text(
    "".join(f"{name}=={value}\n" for name, value in versions.items()),
    encoding="utf-8",
)
print(Path("env/constraints.txt").read_text(encoding="utf-8"))
PY
printf '\n== Test import vLLM before installing dependencies ==\n'
set +e
python - <<'PY' 2>&1 | tee "$LOG_DIR/import_vllm_before_install.log"
import importlib.metadata as md
import torch

print("torch metadata:", md.version("torch"))
print("vllm metadata:", md.version("vllm"))
print("torch import:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
import vllm
print("vllm import:", vllm.__version__)
print("IMPORT_VLLM_OK")
PY
VLLM_BEFORE_RC=${PIPESTATUS[0]}
set -e
printf 'import vllm before install exit code: %s\n' "$VLLM_BEFORE_RC"

printf '\n== Install D0-A dependencies ==\n'
python -m pip install \
  --progress-bar off \
  -c env/constraints.txt \
  modelscope peft accelerate datasets kagglehub kaggle \
  hydra-core omegaconf matplotlib \
  2>&1 | tee "$LOG_DIR/pip_install.log"

python - <<'PY' | tee "$LOG_DIR/versions_after_install.log"
import importlib.metadata as md
import torch

for name in (
    "torch", "vllm", "transformers", "modelscope", "peft",
    "accelerate", "datasets", "kagglehub", "kaggle",
):
    print(f"{name}=={md.version(name)}")
print("torch runtime:", torch.__version__)
assert md.version("torch") == "2.11.0"
assert torch.__version__ == "2.11.0+cu130"
assert md.version("vllm") == "0.25.1"
PY

printf '\n== Test import vLLM after installing dependencies ==\n'
set +e
python - <<'PY' 2>&1 | tee "$LOG_DIR/import_vllm_after_install.log"
import importlib.metadata as md
import torch

print("torch metadata:", md.version("torch"))
print("vllm metadata:", md.version("vllm"))
print("torch import:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
import vllm
print("vllm import:", vllm.__version__)
print("IMPORT_VLLM_OK")
PY
VLLM_AFTER_RC=${PIPESTATUS[0]}
set -e
printf 'import vllm after install exit code: %s\n' "$VLLM_AFTER_RC"

if [[ $VLLM_BEFORE_RC -eq 0 && $VLLM_AFTER_RC -ne 0 ]]; then
  printf 'Dependency installation broke import vllm. Stop and inspect the logs.\n' >&2
  exit 1
fi

printf '\n== Download models serially ==\n'
modelscope download --model Qwen/Qwen3-Embedding-0.6B \
  --local_dir "$MODEL_DIR/qwen3-embedding-0.6b" \
  2>&1 | tee "$LOG_DIR/download_embedding_0.6b.log"
modelscope download --model Qwen/Qwen3-Reranker-0.6B \
  --local_dir "$MODEL_DIR/qwen3-reranker-0.6b" \
  2>&1 | tee "$LOG_DIR/download_reranker_0.6b.log"
modelscope download --model Qwen/Qwen3-Reranker-4B \
  --local_dir "$MODEL_DIR/qwen3-reranker-4b" \
  2>&1 | tee "$LOG_DIR/download_reranker_4b.log"

du -sh "$MODEL_DIR"/* | tee "$LOG_DIR/model_sizes.log"
find "$MODEL_DIR" -maxdepth 2 -type f \
  \( -name 'config.json' -o -name 'tokenizer_config.json' \
  -o -name '*.safetensors' -o -name '*.index.json' \) \
  -printf '%p\t%s bytes\n' | sort | tee "$LOG_DIR/model_files.log"

printf '\n== Verify reranker tokenizer ==\n'
python - <<'PY' 2>&1 | tee "$LOG_DIR/tokenizer_checks.log"
import json
from pathlib import Path
from transformers import AutoTokenizer

model_dir = Path("/workspace/models/qwen3-reranker-0.6b")
tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
print("tokenizer class:", type(tok).__name__)
print("chat template present:", tok.chat_template is not None)

encoded = {}
for word in ("yes", "no", "Yes", "No"):
    ids = tok.encode(word, add_special_tokens=False)
    encoded[word] = ids
    print(
        repr(word), "encode=", ids, "single_token=", len(ids) == 1,
        "convert_tokens_to_ids=", tok.convert_tokens_to_ids(word),
    )

messages = [{"role": "user", "content": "ping"}]
for flag in (True, False):
    try:
        rendered = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=flag,
        )
        print(
            f"enable_thinking={flag}",
            "has_<think>=", "<think>" in rendered,
            "has_</think>=", "</think>" in rendered,
        )
        print("rendered:", repr(rendered))
    except (TypeError, ValueError) as exc:
        print(f"enable_thinking={flag} failed:", type(exc).__name__, str(exc))

config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
print("tie_word_embeddings:", config.get("tie_word_embeddings"))
print("vocab_size:", config.get("vocab_size"))
print("hidden_size:", config.get("hidden_size"))

assert len(encoded["yes"]) == 1
assert len(encoded["no"]) == 1
assert tok.convert_tokens_to_ids("yes") == encoded["yes"][0]
assert tok.convert_tokens_to_ids("no") == encoded["no"][0]
assert config.get("tie_word_embeddings") is True
PY

RERANKER_DIR=$MODEL_DIR/qwen3-reranker-0.6b
if command -v rg >/dev/null 2>&1; then
  rg -n -C 3 --glob 'README*' --glob '*.py' \
    --glob 'tokenizer_config.json' \
    '<Instruct>|<Query>|<Document>|yes_token_id|no_token_id|enable_thinking|prefix|suffix' \
    "$RERANKER_DIR" 2>&1 | tee "$LOG_DIR/official_prompt_refs.log" || true
else
  find "$RERANKER_DIR" -maxdepth 2 -type f \
    \( -iname 'README*' -o -name '*.py' -o -name 'tokenizer_config.json' \) \
    -print0 | xargs -0 grep -nE \
    '<Instruct>|<Query>|<Document>|yes_token_id|no_token_id|enable_thinking|prefix|suffix' \
    2>&1 | tee "$LOG_DIR/official_prompt_refs.log" || true
fi

printf '\n== Read Kaggle credentials ==\n'
if [[ -z ${KAGGLE_USERNAME:-} ]]; then
  read -r -p 'Kaggle username: ' KAGGLE_USERNAME
  export KAGGLE_USERNAME
fi
if [[ -z ${KAGGLE_KEY:-} ]]; then
  read -r -s -p 'Kaggle API key: ' KAGGLE_KEY
  printf '\n'
  export KAGGLE_KEY
fi

printf '\n== Download datasets serially ==\n'
mkdir -p "$RAW_DATA_DIR/competition"
kaggle competitions download -c eedi-mining-misconceptions-in-mathematics \
  -p "$RAW_DATA_DIR/competition" 2>&1 | tee "$LOG_DIR/download_competition.log"
python - <<'PY'
from zipfile import ZipFile

archive = "/workspace/data/raw/competition/eedi-mining-misconceptions-in-mathematics.zip"
with ZipFile(archive) as source:
    source.extractall("/workspace/data/raw/competition")
PY
python - <<'PY' 2>&1 | tee "$LOG_DIR/download_data.log"
import kagglehub, os

handles = (
    "conjuring92/eedi-five-folds", "conjuring92/eedi-silver-v3",
    "conjuring92/eedi-embed-pretrain-mix-final", "conjuring92/eedi-embed-mix-silver-v3",
    "conjuring92/eedi-ranker-silver-v3-teacher-blended-cot", "conjuring92/eedi-tutor-mix-v8",
    "conjuring92/eedi-cot-sonnet-6k", "conjuring92/eedi-cot-train-silver-v3",
    "conjuring92/eedi-misconception-clusters", "conjuring92/eedi-cot-gen-base",
)
for handle in handles:
    print(f"downloading dataset: {handle}")
    path = f"/workspace/data/raw/{handle.split('/', 1)[1]}"
    result = kagglehub.dataset_download(handle, output_dir=path)
    print("saved to:", result)

for required_file in (
    "/workspace/data/raw/competition/train.csv", "/workspace/data/raw/competition/misconception_mapping.csv",
    "/workspace/data/raw/eedi-silver-v3/train.csv", "/workspace/data/raw/eedi-silver-v3/misconception_mapping.csv",
    "/workspace/data/raw/eedi-five-folds/folds.parquet",
):
    if not os.path.isfile(required_file):
        raise SystemExit(f"required data file missing: {required_file}")
PY
unset KAGGLE_USERNAME KAGGLE_KEY

du -sh "$RAW_DATA_DIR"/* | tee "$LOG_DIR/data_sizes.log"
find "$RAW_DATA_DIR" -maxdepth 4 -type f -printf '%p\t%s bytes\n' \
  | sort | tee "$LOG_DIR/data_files.log"

printf '\n== Inspect data schemas without loading full datasets ==\n'
python - <<'PY' 2>&1 | tee "$LOG_DIR/data_schema.log"
import csv
import json
from pathlib import Path

root = Path("/workspace/data/raw")
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    suffix = path.suffix.lower()
    print(f"\nFILE: {path}")
    print(f"SIZE: {path.stat().st_size}")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            print("CSV COLUMNS:", next(csv.reader(handle), None))
    elif suffix == ".jsonl":
        first = None
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    first = json.loads(line)
                    break
        print("JSONL KEYS:", sorted(first) if isinstance(first, dict) else type(first).__name__)
    elif suffix == ".parquet":
        import pyarrow.parquet as pq
        print("PARQUET COLUMNS:", pq.ParquetFile(path).schema_arrow.names)
PY

printf '\n== Package logs for local review ==\n'
cp env/constraints.txt "$LOG_DIR/constraints.txt"
cp env/env_baseline.txt "$LOG_DIR/env_baseline.txt"
git status --short --branch | tee "$LOG_DIR/git_status_after_run.log"

D0A_LOG_TS=$(date +%Y%m%d-%H%M%S)
ARCHIVE_PATH=$WORK_DIR/cache/d0a/d0a-probe-$D0A_LOG_TS.tgz
tar -czf "$ARCHIVE_PATH" -C "$WORK_DIR/cache/d0a" logs

printf '\nD0-A collection finished.\n'
printf 'vLLM import before install exit code: %s\n' "$VLLM_BEFORE_RC"
printf 'vLLM import after install exit code:  %s\n' "$VLLM_AFTER_RC"
printf 'Log archive: %s\n' "$ARCHIVE_PATH"
printf 'Copy env/constraints.txt, env/env_baseline.txt, and the archive back to Windows.\n'
