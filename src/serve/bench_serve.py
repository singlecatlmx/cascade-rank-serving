import argparse
import asyncio
import json
import statistics
import time

import httpx


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/rank")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if args.requests <= 0 or args.concurrency <= 0:
        raise SystemExit("requests and concurrency must be positive")
    payload = {"query": "What is 20% of 50?", "correct_answer": "10", "incorrect_answer": "20"}
    semaphore = asyncio.Semaphore(args.concurrency)

    async def send(client):
        async with semaphore:
            started = time.perf_counter()
            response = await client.post(args.url, json=payload)
            response.raise_for_status()
            return (time.perf_counter() - started) * 1000, response.json()["degraded"]

    async with httpx.AsyncClient(timeout=30) as client:
        for _ in range(20):
            await send(client)
        started = time.perf_counter()
        measured = await asyncio.gather(*(send(client) for _ in range(args.requests)))
    latencies = sorted(item[0] for item in measured)
    elapsed = time.perf_counter() - started
    index = lambda q: latencies[min(len(latencies) - 1, int((len(latencies) - 1) * q))]
    print(json.dumps({
        "requests": args.requests,
        "concurrency": args.concurrency,
        "qps": args.requests / elapsed,
        "latency_ms": {"p50": index(0.50), "p95": index(0.95), "p99": index(0.99), "mean": statistics.mean(latencies)},
        "degraded_rate": sum(item[1] for item in measured) / args.requests,
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
