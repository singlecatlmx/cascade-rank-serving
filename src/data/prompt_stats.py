import hashlib
import importlib.metadata as metadata
import json
import math
import random
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

from src.data.prompts import INSTRUCTION, SUFFIX, SYSTEM_MESSAGE, prompt_token_ids


MODEL_DIR = Path("/workspace/models/qwen3-reranker-0.6b")
EVAL_PATH = Path("data/eval_set_v1.jsonl")
LABEL_PATH = Path("data/label_pool_v1.jsonl")
CANDIDATES_PER_QUERY = 32
LENGTH_MARGIN = 64
TARGET_CONCURRENCIES = [8, 16, 32, 64]


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values):
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def common_prefix_length(token_lists):
    common = token_lists[0]
    for tokens in token_lists[1:]:
        limit = min(len(common), len(tokens))
        index = 0
        while index < limit and common[index] == tokens[index]:
            index += 1
        common = common[:index]
    return len(common)


def round_up(value, multiple):
    return math.ceil(value / multiple) * multiple


def main():
    if not MODEL_DIR.is_dir():
        raise SystemExit(f"tokenizer directory missing: {MODEL_DIR}")
    eval_rows = load_jsonl(EVAL_PATH)
    if not eval_rows or "_meta" not in eval_rows[0]:
        raise SystemExit("eval_set_v1.jsonl metadata header missing")
    meta = eval_rows[0]["_meta"]
    queries = eval_rows[1:]
    labels = load_jsonl(LABEL_PATH)
    if len(queries) != 200 or len(labels) != 4791:
        raise SystemExit(f"unexpected frozen data size: queries={len(queries)}, labels={len(labels)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model_config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    if model_config.get("tie_word_embeddings") is not True:
        raise SystemExit("expected tied embeddings for Qwen3-Reranker-0.6B")
    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    no_ids = tokenizer.encode("no", add_special_tokens=False)
    assert len(yes_ids) == 1 and len(no_ids) == 1
    assert tokenizer.convert_tokens_to_ids("yes") == yes_ids[0]
    assert tokenizer.convert_tokens_to_ids("no") == no_ids[0]
    suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)

    label_by_id = {row["label_id"]: row["label"] for row in labels}
    all_label_ids = sorted(label_by_id)
    rng = random.Random(meta["seed"])
    candidates_by_query = {}
    for query in queries:
        actual = query["actual"][0]
        negatives = [label_id for label_id in all_label_ids if label_id != actual]
        candidates_by_query[query["query_id"]] = [
            actual,
            *rng.sample(negatives, CANDIDATES_PER_QUERY - 1),
        ]

    variants = {}
    for variant in ("a0_document_first", "a1_document_last"):
        lengths = []
        shared_lengths = []
        shared_ratios = []
        for query in queries:
            candidate_ids = candidates_by_query[query["query_id"]]
            token_lists = [
                prompt_token_ids(
                    tokenizer,
                    query["query"],
                    label_by_id[label_id],
                    variant,
                    suffix_tokens,
                )
                for label_id in candidate_ids
            ]
            shared = common_prefix_length(token_lists)
            shared_lengths.extend([shared] * len(token_lists))
            lengths.extend(len(tokens) for tokens in token_lists)
            shared_ratios.extend(shared / len(tokens) for tokens in token_lists)
        variants[variant] = {
            "prompt_tokens": summarize(lengths),
            "shared_prefix_tokens": summarize(shared_lengths),
            "shared_prefix_ratio": summarize(shared_ratios),
        }

    a1_lengths = variants["a1_document_last"]["prompt_tokens"]
    max_model_len = round_up(a1_lengths["p99"] + LENGTH_MARGIN + 1, 128)
    batch_tokens = {
        str(concurrency): round_up(a1_lengths["mean"] * concurrency, 128)
        for concurrency in TARGET_CONCURRENCIES
    }
    config = {
        "model_dir": str(MODEL_DIR),
        "eval_set": str(EVAL_PATH),
        "label_pool": str(LABEL_PATH),
        "seed": meta["seed"],
        "queries": len(queries),
        "candidates_per_query": CANDIDATES_PER_QUERY,
        "candidate_sampling": "actual_plus_seeded_random_negatives",
        "prompt_variants": ["a0_document_first", "a1_document_last"],
        "system_message": SYSTEM_MESSAGE,
        "instruction": INSTRUCTION,
        "suffix": SUFFIX,
        "length_margin_tokens": LENGTH_MARGIN,
        "target_concurrencies": TARGET_CONCURRENCIES,
    }
    timestamp = datetime.now(timezone.utc)
    config_text = json.dumps(config, sort_keys=True, ensure_ascii=False)
    config_hash = hashlib.sha256(config_text.encode()).hexdigest()[:8]
    result = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "vllm_version": metadata.version("vllm"),
        "gpu": "none (D0-A no-GPU)",
        "timestamp": timestamp.isoformat(),
        "config": config,
        "tokenizer_checks": {
            "yes_token_id": yes_ids[0],
            "no_token_id": no_ids[0],
            "thinking_disabled": True,
            "tie_word_embeddings": True,
        },
        "variants": variants,
        "recommended": {
            "max_model_len": max_model_len,
            "max_num_batched_tokens_scan": batch_tokens,
        },
    }
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / (
        f"d0a_prompt_stats_{timestamp.strftime('%Y%m%d-%H%M%S')}_{config_hash}.json"
    )
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing result: {output_path}")
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    print(json.dumps(result["recommended"], indent=2))
    for variant in ("a0_document_first", "a1_document_last"):
        ratio = variants[variant]["shared_prefix_ratio"]
        print(
            f"{variant} shared prefix ratio:",
            f"mean={ratio['mean']:.2%}",
            f"p50={ratio['p50']:.2%}",
            f"p90={ratio['p90']:.2%}",
        )


if __name__ == "__main__":
    main()
