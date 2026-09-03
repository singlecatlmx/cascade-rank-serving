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

from src.metrics import latency_stats, map_at_k, ndcg_at_k, recall_at_k


SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
)
INSTRUCTION = (
    "Given a math question, its correct answer, and an incorrect answer, retrieve "
    "the misconception that best explains the incorrect answer."
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def make_prompts(tokenizer, count):
    prompts = []
    for index in range(count):
        query = f"Question {index}: What is {index} + 1? Correct answer: {index + 1}. Incorrect answer: {index + 2}."
        document = f"The learner adds one to the value in question {index}."
        rendered = (
            f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
            f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {document}{SUFFIX}"
        )
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        prompts.append(TokensPrompt(prompt_token_ids=token_ids))
    return prompts


def metric_value(metrics, name, kind):
    values = [metric.value for metric in metrics if metric.name == name and isinstance(metric.value, kind)]
    return sum(values) if values else 0.0


def gpu_memory_gb():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return max(float(line.strip()) for line in output.splitlines() if line.strip()) / 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.warmup < 20 or args.requests <= 0:
        raise SystemExit("warmup must be at least 20 and requests must be positive")

    config = {
        "model": args.model,
        "dtype": "bfloat16",
        "quantization": None,
        "tensor_parallel_size": 1,
        "enable_prefix_caching": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "enforce_eager": True,
        "disable_log_stats": False,
        "warmup_requests": args.warmup,
        "measured_requests": args.requests,
        "max_tokens": 1,
        "logprobs": 20,
        "seed": 0,
    }
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    if tokenizer.encode("yes", add_special_tokens=False) != [yes_id] or tokenizer.encode("no", add_special_tokens=False) != [no_id]:
        raise RuntimeError("yes/no must each be one token")
    prompts = make_prompts(tokenizer, args.requests)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        disable_log_stats=False,
        seed=0,
    )
    sampling = SamplingParams(temperature=0, max_tokens=1, logprobs=20, allowed_token_ids=[yes_id, no_id])

    def run(prompt):
        before = time.perf_counter()
        output = llm.generate([prompt], sampling, use_tqdm=False)[0]
        e2e_ms = (time.perf_counter() - before) * 1000
        if output.metrics is None:
            raise RuntimeError("vLLM did not return request metrics")
        if output.metrics.first_token_latency <= 0:
            raise RuntimeError("vLLM returned invalid TTFT")
        logprobs = output.outputs[0].logprobs[-1]
        if yes_id not in logprobs or no_id not in logprobs:
            raise RuntimeError("yes/no missing from returned logprobs")
        score = logprobs[yes_id].logprob - logprobs[no_id].logprob
        return output.metrics.first_token_latency * 1000, e2e_ms, score

    for index in range(args.warmup):
        run(prompts[index % len(prompts)])
    warmup_metrics = llm.get_metrics()
    warmup_queries = metric_value(warmup_metrics, "vllm:prefix_cache_queries", (int, float))
    warmup_hits = metric_value(warmup_metrics, "vllm:prefix_cache_hits", (int, float))
    measurement_started = time.perf_counter()
    measured = [run(prompts[index % len(prompts)]) for index in range(args.requests)]
    measured_seconds = time.perf_counter() - measurement_started
    peak_memory = gpu_memory_gb()
    metrics = llm.get_metrics()
    queries = metric_value(metrics, "vllm:prefix_cache_queries", (int, float)) - warmup_queries
    hits = metric_value(metrics, "vllm:prefix_cache_hits", (int, float)) - warmup_hits
    hit_rate = hits / queries if queries else 0.0
    if queries <= 0 or hit_rate <= 0:
        raise RuntimeError(f"prefix cache metrics are not positive: queries={queries}, hits={hits}")
    kv_usage = metric_value(metrics, "vllm:kv_cache_usage_perc", (int, float))
    num_blocks = getattr(llm.llm_engine.vllm_config.cache_config, "num_gpu_blocks", 0)
    total_elapsed = time.perf_counter() - started
    ttft = latency_stats([item[0] for item in measured])
    e2e = latency_stats([item[1] for item in measured])
    candidates = args.requests
    preds = sorted(range(args.requests), key=lambda index: measured[index][2], reverse=True)
    quality = {
        "recall@32": recall_at_k(preds, [0], 32),
        "map@25": map_at_k(preds, [0], 25),
        "ndcg@10": ndcg_at_k(preds, [0], 10),
    }
    timestamp = datetime.now(timezone.utc)
    config_hash = sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        commit += "-dirty"
    output_path = Path(args.output)
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite {output_path}")
    result = {
        "git_commit": commit,
        "vllm_version": metadata.version("vllm"),
        "torch_version": torch.__version__,
        "gpu": f"{torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}",
        "timestamp": timestamp.strftime("%Y%m%d-%H%M%S"),
        "config": config,
        "metrics": {
            "quality": quality,
            "latency_ms": {
                "ttft_p50": ttft["p50"], "ttft_p95": ttft["p95"], "ttft_p99": ttft["p99"],
                "e2e_p50": e2e["p50"], "e2e_p95": e2e["p95"], "e2e_p99": e2e["p99"],
            },
            "throughput": {"req_per_s": args.requests / measured_seconds, "candidates_scored_per_s": candidates / measured_seconds},
            "resource": {"peak_mem_gb": peak_memory, "prefix_cache_hit_rate": hit_rate, "kv_cache_usage_perc": kv_usage, "kv_cache_used_blocks": kv_usage * num_blocks},
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "prefix_cache_hit_rate": hit_rate, "measured_seconds": measured_seconds, "total_seconds": total_elapsed, "config_hash": config_hash}))


if __name__ == "__main__":
    main()
