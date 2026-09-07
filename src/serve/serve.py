import argparse
import asyncio
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from src.data.prompts import prompt_token_ids


class RankRequest(BaseModel):
    query: str
    correct_answer: str = ""
    incorrect_answer: str = ""
    timeout_ms: int = 150


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def build_query(request):
    return (
        f"Question: {request.query}\nCorrect Answer: {request.correct_answer}\n"
        f"Incorrect Answer: {request.incorrect_answer}"
    )


def create_app(args):
    labels = {row["label_id"]: row["label"] for row in read_jsonl(args.labels)}
    embedding_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, local_files_only=True, padding_side="left")
    embedding_model = AutoModel.from_pretrained(
        args.embedding_model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).eval().cuda()
    label_ids = list(labels)

    def embed(texts):
        batch = embedding_tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            hidden = embedding_model(**batch).last_hidden_state[:, -1]
        return F.normalize(hidden.float(), p=2, dim=1)

    label_embeddings = []
    for start in range(0, len(label_ids), 64):
        label_embeddings.append(embed([labels[label_id] for label_id in label_ids[start : start + 64]]).cpu())
    label_embeddings = torch.cat(label_embeddings).cuda()

    reranker_tokenizer = AutoTokenizer.from_pretrained(args.reranker_model, local_files_only=True)
    yes_id = reranker_tokenizer.convert_tokens_to_ids("yes")
    no_id = reranker_tokenizer.convert_tokens_to_ids("no")
    if reranker_tokenizer.encode("yes", add_special_tokens=False) != [yes_id] or reranker_tokenizer.encode("no", add_special_tokens=False) != [no_id]:
        raise RuntimeError("yes/no must each be one token")
    reranker = LLM(
        model=args.reranker_model,
        dtype="bfloat16",
        quantization=None,
        kv_cache_dtype=args.kv_cache_dtype,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=640,
        enforce_eager=True,
        disable_log_stats=False,
        seed=20260901,
    )
    sampling = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=2,
        logprob_token_ids=[yes_id, no_id],
        allowed_token_ids=[yes_id, no_id],
    )
    lock = asyncio.Lock()

    def rank(query):
        recall_started = time.perf_counter()
        query_embedding = embed([f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {query}"])
        _, top_indices = torch.topk(query_embedding @ label_embeddings.T, k=args.candidate_k)
        candidate_ids = [label_ids[index] for index in top_indices[0].cpu().tolist()]
        recall_ms = (time.perf_counter() - recall_started) * 1000
        rerank_started = time.perf_counter()
        prompts = [
            TokensPrompt(prompt_token_ids=prompt_token_ids(reranker_tokenizer, query, labels[label_id], "a1_document_last"))
            for label_id in candidate_ids
        ]
        outputs = reranker.generate(prompts, sampling, use_tqdm=False)
        scored = []
        for label_id, output in zip(candidate_ids, outputs):
            if output.metrics is None or output.metrics.first_token_latency <= 0:
                raise RuntimeError("vLLM did not return request metrics")
            logprobs = output.outputs[0].logprobs[-1]
            if yes_id not in logprobs or no_id not in logprobs:
                raise RuntimeError("yes/no missing from returned logprobs")
            scored.append((logprobs[yes_id].logprob - logprobs[no_id].logprob, label_id))
        return [label_id for _, label_id in sorted(scored, reverse=True)], recall_ms, (time.perf_counter() - rerank_started) * 1000

    async def score_with_lock(query):
        async with lock:
            return await asyncio.to_thread(rank, query)

    app = FastAPI(title="Cascade Rank Serving", version="0.1.0")

    @app.get("/health")
    async def health():
        return {"status": "ok", "candidate_k": args.candidate_k, "kv_cache_dtype": args.kv_cache_dtype}

    @app.post("/v1/rank")
    async def rank_endpoint(request: RankRequest):
        started = time.perf_counter()
        query = build_query(request)
        fallback = False
        fallback_reason = None
        try:
            ranked, recall_ms, rerank_ms = await asyncio.wait_for(score_with_lock(query), max(request.timeout_ms, 1) / 1000)
        except asyncio.TimeoutError:
            fallback = True
            fallback_reason = "timeout"
            recall_started = time.perf_counter()
            query_embedding = await asyncio.to_thread(embed, [f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {query}"])
            _, top_indices = torch.topk(query_embedding @ label_embeddings.T, k=25)
            ranked = [label_ids[index] for index in top_indices[0].cpu().tolist()]
            recall_ms = (time.perf_counter() - recall_started) * 1000
            rerank_ms = 0.0
        except RuntimeError:
            fallback = True
            fallback_reason = "rerank_error"
            recall_started = time.perf_counter()
            query_embedding = await asyncio.to_thread(embed, [f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {query}"])
            _, top_indices = torch.topk(query_embedding @ label_embeddings.T, k=25)
            ranked = [label_ids[index] for index in top_indices[0].cpu().tolist()]
            recall_ms = (time.perf_counter() - recall_started) * 1000
            rerank_ms = 0.0
        total_ms = (time.perf_counter() - started) * 1000
        return {
            "results": [{"label_id": label_id, "label": labels[label_id], "rank": index + 1} for index, label_id in enumerate(ranked[:25])],
            "degraded": fallback,
            "fallback": "recall_top25" if fallback else None,
            "fallback_reason": fallback_reason if fallback else None,
            "timing_ms": {"total": round(total_ms, 3), "recall": round(recall_ms, 3), "rerank": round(rerank_ms, 3)},
            "candidate_count": args.candidate_k,
            "kv_cache_dtype": args.kv_cache_dtype,
        }

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", default="/workspace/models/qwen3-embedding-0.6b")
    parser.add_argument("--reranker-model", default="/workspace/models/reranker-0.6b-merged")
    parser.add_argument("--labels", default="data/label_pool_v1.jsonl")
    parser.add_argument("--candidate-k", type=int, default=32)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.candidate_k <= 64:
        raise SystemExit("candidate-k must be between 1 and 64")
    if args.kv_cache_dtype not in {"auto", "fp8"}:
        raise SystemExit("kv-cache-dtype must be auto or fp8")
    import uvicorn

    uvicorn.run(create_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
