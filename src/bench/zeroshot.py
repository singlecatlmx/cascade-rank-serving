import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from src.bench.smoke import gpu_memory_gb, metric_value
from src.data.prompts import prompt_token_ids
from src.metrics import latency_stats, map_at_k, ndcg_at_k, recall_at_k


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
    parser.add_argument("--labels", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=640)
    args = parser.parse_args()

    candidate_lines = read_jsonl(args.candidates)
    candidate_meta, rows = candidate_lines[0]["_meta"], candidate_lines[1:]
    labels = {row["label_id"]: row["label"] for row in read_jsonl(args.labels)}
    if len(rows) != 200 or len(labels) != 4791:
        raise RuntimeError(f"expected 200 queries and 4791 labels, got {len(rows)} and {len(labels)}")
    if candidate_meta["seed"] != 20260901:
        raise RuntimeError(f"unexpected candidate seed: {candidate_meta['seed']}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible GPU, found {torch.cuda.device_count()}")

    config = {
        "stage": "baseline",
        "variant": "zeroshot",
        "model": args.model,
        "dtype": "bfloat16",
        "quantization": None,
        "tensor_parallel_size": 1,
        "enable_prefix_caching": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "enforce_eager": True,
        "disable_log_stats": False,
        "prompt_variant": "a1_document_last",
        "candidate_k": 32,
        "submission": "one query group per generate call",
        "warmup_queries": 20,
        "measured_queries": 180,
        "temperature": 0,
        "max_tokens": 1,
        "logprobs": 2,
        "logprob_token_ids": ["yes", "no"],
        "allowed_token_ids": ["yes", "no"],
        "seed": candidate_meta["seed"],
        "labels": args.labels,
        "candidates": Path(args.candidates).name,
        "candidate_config": candidate_meta,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    if tokenizer.encode("yes", add_special_tokens=False) != [yes_id] or tokenizer.encode("no", add_special_tokens=False) != [no_id]:
        raise RuntimeError("yes/no must each be one token")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        disable_log_stats=False,
        seed=candidate_meta["seed"],
    )
    sampling = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=2,
        logprob_token_ids=[yes_id, no_id],
        allowed_token_ids=[yes_id, no_id],
    )

    def run(row):
        candidate_ids = row["candidate_ids"][:32]
        prompts = [
            TokensPrompt(prompt_token_ids=prompt_token_ids(tokenizer, row["query"], labels[label_id], "a1_document_last"))
            for label_id in candidate_ids
        ]
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        e2e_ms = (time.perf_counter() - started) * 1000
        scores = []
        ttft = []
        for output in outputs:
            if output.metrics is None or output.metrics.first_token_latency <= 0:
                raise RuntimeError("vLLM did not return valid request metrics")
            final = output.outputs[0].logprobs[-1]
            if yes_id not in final or no_id not in final:
                raise RuntimeError("yes/no missing from returned logprobs")
            ttft.append(output.metrics.first_token_latency * 1000)
            scores.append(final[yes_id].logprob - final[no_id].logprob)
        ranking = [item for _, item in sorted(zip(scores, candidate_ids), reverse=True)]
        return ranking, max(ttft), e2e_ms

    for row in rows[:20]:
        run(row)
    warmup_metrics = llm.get_metrics()
    warmup_queries = metric_value(warmup_metrics, "vllm:prefix_cache_queries", (int, float))
    warmup_hits = metric_value(warmup_metrics, "vllm:prefix_cache_hits", (int, float))
    measured_started = time.perf_counter()
    measured = [run(row) for row in rows[20:]]
    measured_seconds = time.perf_counter() - measured_started

    snapshot = llm.get_metrics()
    prefix_queries = metric_value(snapshot, "vllm:prefix_cache_queries", (int, float)) - warmup_queries
    prefix_hits = metric_value(snapshot, "vllm:prefix_cache_hits", (int, float)) - warmup_hits
    if prefix_queries <= 0:
        raise RuntimeError("vLLM prefix cache counters were not recorded")
    hit_rate = prefix_hits / prefix_queries
    kv_usage = metric_value(snapshot, "vllm:kv_cache_usage_perc", (int, float))
    num_blocks = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks or 0
    ttft = latency_stats([item[1] for item in measured])
    e2e = latency_stats([item[2] for item in measured])
    quality = {
        "recall@32": sum(recall_at_k(item[0], row["actual"], 32) for item, row in zip(measured, rows[20:])) / 180,
        "map@25": sum(map_at_k(item[0], row["actual"], 25) for item, row in zip(measured, rows[20:])) / 180,
        "ndcg@10": sum(ndcg_at_k(item[0], row["actual"], 10) for item, row in zip(measured, rows[20:])) / 180,
    }
    timestamp = datetime.now(timezone.utc)
    digest = sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    result_path = Path(args.results_dir) / f"baseline_zeroshot_{timestamp.strftime('%Y%m%d-%H%M%S')}_{digest}.json"
    result_path.parent.mkdir(exist_ok=True)
    result = {
        "git_commit": git_commit(),
        "vllm_version": metadata.version("vllm"),
        "torch_version": torch.__version__,
        "gpu": f"{torch.cuda.get_device_name(0)} x1",
        "timestamp": timestamp.strftime("%Y%m%d-%H%M%S"),
        "config": config,
        "metrics": {
            "quality": quality,
            "latency_ms": {"ttft_p50": ttft["p50"], "ttft_p95": ttft["p95"], "ttft_p99": ttft["p99"], "e2e_p50": e2e["p50"], "e2e_p95": e2e["p95"], "e2e_p99": e2e["p99"]},
            "throughput": {"req_per_s": 180 / measured_seconds, "candidates_scored_per_s": 180 * 32 / measured_seconds},
            "resource": {"peak_mem_gb": gpu_memory_gb(), "prefix_cache_hit_rate": hit_rate, "kv_cache_usage_perc": kv_usage, "kv_cache_used_blocks": kv_usage * num_blocks},
        },
    }
    if result_path.exists():
        raise SystemExit(f"refusing to overwrite {result_path}")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": str(result_path), "quality": quality, "prefix_cache_hit_rate": hit_rate}))


if __name__ == "__main__":
    main()
