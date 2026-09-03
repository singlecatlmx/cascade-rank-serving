import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.data.prompts import INSTRUCTION
from src.metrics import latency_stats, recall_at_k


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def git_commit():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        commit += "-dirty"
    return commit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--candidates-output", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    eval_lines = read_jsonl(args.eval)
    meta, queries = eval_lines[0]["_meta"], eval_lines[1:]
    labels = read_jsonl(args.labels)
    if meta != {"version": "eval_set_v1", "seed": 20260901, "fold": 0, "size": 200, "source": "eedi-mining-misconceptions-in-mathematics", "label_source": "conjuring92/eedi-silver-v3"}:
        raise RuntimeError(f"unexpected frozen eval metadata: {meta}")
    if len(queries) != 200 or len(labels) != 4791:
        raise RuntimeError(f"expected 200 queries and 4791 labels, got {len(queries)} and {len(labels)}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible GPU, found {torch.cuda.device_count()}")

    config = {
        "stage": "baseline",
        "variant": "recall",
        "model": args.model,
        "dtype": "bfloat16",
        "attention": "sdpa",
        "pooling": "last_eos_token",
        "normalize": True,
        "query_instruction": INSTRUCTION,
        "document_instruction": None,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "top_k": [32, 64],
        "warmup_queries": 20,
        "measured_queries": 180,
        "seed": meta["seed"],
        "eval": args.eval,
        "labels": args.labels,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="left")
    if tokenizer.pad_token != "<|endoftext|>":
        raise RuntimeError(f"unexpected embedding pad token: {tokenizer.pad_token}")
    torch.manual_seed(meta["seed"])
    torch.cuda.manual_seed_all(meta["seed"])
    model = AutoModel.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval().cuda()

    def encode(texts):
        batch = tokenizer(texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt").to("cuda")
        if not torch.all(batch["attention_mask"][:, -1] == 1):
            raise RuntimeError("last-token pooling received right-padded input")
        if not torch.all(batch["input_ids"][:, -1] == tokenizer.pad_token_id):
            raise RuntimeError("embedding input does not end with <|endoftext|>")
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state[:, -1]
        return F.normalize(hidden.float(), p=2, dim=1)

    label_embeddings = []
    label_texts = [row["label"] for row in labels]
    for start in range(0, len(label_texts), args.batch_size):
        label_embeddings.append(encode(label_texts[start : start + args.batch_size]).cpu())
    label_embeddings = torch.cat(label_embeddings).cuda()
    label_ids = torch.tensor([row["label_id"] for row in labels], device="cuda")

    candidate_rows = []
    latencies = []
    measured_started = None
    for index, query in enumerate(queries):
        if index == 20:
            measured_started = time.perf_counter()
        text = f"Instruct: {INSTRUCTION}\nQuery:{query['query']}"
        started = time.perf_counter()
        query_embedding = encode([text])
        scores = query_embedding @ label_embeddings.T
        top_scores, top_indices = torch.topk(scores[0], k=64)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if index >= 20:
            latencies.append(elapsed_ms)
        candidate_rows.append({
            "query_id": query["query_id"],
            "query": query["query"],
            "actual": query["actual"],
            "candidate_ids": label_ids[top_indices].cpu().tolist(),
            "scores": [float(value) for value in top_scores.cpu()],
        })
    measured_seconds = time.perf_counter() - measured_started
    measured_rows = candidate_rows[20:]
    recall32 = sum(recall_at_k(row["candidate_ids"], row["actual"], 32) for row in measured_rows) / len(measured_rows)
    recall64 = sum(recall_at_k(row["candidate_ids"], row["actual"], 64) for row in measured_rows) / len(measured_rows)

    candidates_path = Path(args.candidates_output)
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    if candidates_path.exists():
        raise SystemExit(f"refusing to overwrite {candidates_path}")
    candidates_path.write_text("\n".join(json.dumps(row) for row in [{"_meta": config}, *candidate_rows]) + "\n")

    stats = latency_stats(latencies)
    timestamp = datetime.now(timezone.utc)
    digest = sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    result_path = Path(args.results_dir) / f"baseline_recall_{timestamp.strftime('%Y%m%d-%H%M%S')}_{digest}.json"
    result_path.parent.mkdir(exist_ok=True)
    result = {
        "git_commit": git_commit(),
        "vllm_version": metadata.version("vllm"),
        "torch_version": torch.__version__,
        "gpu": f"{torch.cuda.get_device_name(0)} x1",
        "timestamp": timestamp.strftime("%Y%m%d-%H%M%S"),
        "config": config,
        "metrics": {
            "quality": {"recall@32": recall32, "recall@64": recall64, "map@25": None, "ndcg@10": None},
            "latency_ms": {"ttft_p50": stats["p50"], "ttft_p95": stats["p95"], "ttft_p99": stats["p99"], "e2e_p50": stats["p50"], "e2e_p95": stats["p95"], "e2e_p99": stats["p99"]},
            "throughput": {"req_per_s": 180 / measured_seconds, "candidates_scored_per_s": 180 * len(labels) / measured_seconds},
            "resource": {"peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "prefix_cache_hit_rate": 0.0},
        },
    }
    if result_path.exists():
        raise SystemExit(f"refusing to overwrite {result_path}")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": str(result_path), "candidates": str(candidates_path), "recall@32": recall32, "recall@64": recall64}))
    if recall32 <= 0.6:
        raise SystemExit("Gate G2 failed: recall@32 must be greater than 0.6")


if __name__ == "__main__":
    main()
