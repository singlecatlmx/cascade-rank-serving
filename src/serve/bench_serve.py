import argparse
import asyncio
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import httpx
import torch

from src.metrics import map_at_k


def git_commit():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        commit += "-dirty"
    return commit


def gpu_name():
    if not torch.cuda.is_available():
        return "not visible to client"
    return f"{torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}"


def read_eval(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    meta, rows = rows[0], rows[1:]
    if meta.get("_meta", {}).get("seed") != 20260901 or len(rows) != 200:
        raise RuntimeError("eval-jsonl must be eval_set_v1 with seed 20260901 and 200 rows")
    return rows


def latency_stats(values):
    values = sorted(values)
    index = lambda q: values[min(len(values) - 1, int((len(values) - 1) * q))]
    return {"p50": index(0.50), "p95": index(0.95), "p99": index(0.99), "mean": statistics.mean(values)}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/rank")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-ms", type=int, default=150)
    parser.add_argument("--eval-jsonl")
    parser.add_argument("--baseline-timeout-ms", type=int, default=1000)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.requests <= 0 or args.concurrency <= 0 or args.timeout_ms <= 0:
        raise SystemExit("requests, concurrency, and timeout-ms must be positive")
    eval_rows = read_eval(args.eval_jsonl) if args.eval_jsonl else None
    measured_eval_rows = eval_rows[20:] if eval_rows else None
    payload = {"query": "What is 20% of 50?", "correct_answer": "10", "incorrect_answer": "20"}
    semaphore = asyncio.Semaphore(args.concurrency)

    async def send(client, body):
        async with semaphore:
            started = time.perf_counter()
            response = await client.post(args.url, json=body)
            response.raise_for_status()
            return (time.perf_counter() - started) * 1000, response.json()

    async def send_eval(client, row, timeout_ms):
        body = {"query": row["query"], "correct_answer": "", "incorrect_answer": "", "timeout_ms": timeout_ms}
        return await send(client, body)

    async with httpx.AsyncClient(timeout=30) as client:
        for _ in range(20):
            await send(client, {**payload, "timeout_ms": args.timeout_ms})
        started = time.perf_counter()
        if measured_eval_rows:
            baseline = await asyncio.gather(*(send_eval(client, row, args.baseline_timeout_ms) for row in measured_eval_rows))
            measured = await asyncio.gather(*(send_eval(client, row, args.timeout_ms) for row in measured_eval_rows))
        else:
            baseline = None
            measured = await asyncio.gather(*(send(client, {**payload, "timeout_ms": args.timeout_ms}) for _ in range(args.requests)))
    elapsed = time.perf_counter() - started
    latencies = [item[0] for item in measured]
    degraded = [item[1]["degraded"] for item in measured]
    metrics = {
        "requests": len(measured),
        "concurrency": args.concurrency,
        "qps": len(measured) / elapsed,
        "latency_ms": latency_stats(latencies),
        "degraded_rate": sum(degraded) / len(degraded),
    }
    if eval_rows:
        gold = [row["actual"] for row in measured_eval_rows]
        baseline_maps = [map_at_k([item["label_id"] for item in result["results"]], gold[index], 25) for index, (_, result) in enumerate(baseline)]
        target_maps = [map_at_k([item["label_id"] for item in result["results"]], gold[index], 25) for index, (_, result) in enumerate(measured)]
        baseline_degraded = [result["degraded"] for _, result in baseline]
        degraded_maps = [value for value, flag in zip(target_maps, degraded) if flag]
        metrics["quality"] = {
            "baseline_map_at_25": sum(baseline_maps) / len(baseline_maps),
            "target_map_at_25": sum(target_maps) / len(target_maps),
            "paired_map_loss": sum(baseline_maps) / len(baseline_maps) - sum(target_maps) / len(target_maps),
            "baseline_degraded_rate": sum(baseline_degraded) / len(baseline_degraded),
            "target_degraded_rate": sum(degraded) / len(degraded),
            "target_map_at_25_degraded": sum(degraded_maps) / len(degraded_maps) if degraded_maps else None,
        }
    result = {
        "git_commit": git_commit(),
        "vllm_version": metadata.version("vllm"),
        "gpu": gpu_name(),
        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        "config": {
            "url": args.url,
            "requests": len(measured),
            "concurrency": args.concurrency,
            "timeout_ms": args.timeout_ms,
            "baseline_timeout_ms": args.baseline_timeout_ms if eval_rows else None,
            "eval_jsonl": args.eval_jsonl,
            "warmup_requests": 20,
        },
        "metrics": metrics,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise SystemExit(f"refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
