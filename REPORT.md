# Benchmark Report

## 1. Scope

This project converts an offline misconception-mining pipeline into an online search intent-ranking service. Query text, correct answer, and incorrect answer are embedded against a 4,791-label pool; a Qwen3 reranker scores the retrieved candidates with one-token yes/no logits.

The evaluation set is frozen at 200 queries (fold 0, seed `20260901`). Every run warms up 20 queries and reports the remaining 180. Results are stored with the full configuration, Git commit, vLLM version, GPU, and timestamp.

## 2. Hardware Constraint

The target machine has two RTX 5090 cards (`sm_120`) with P2P disabled and measured inter-card bandwidth of 43.6 GB/s. vLLM 0.25.1 contains sm_120 SASS in its main kernel library, but the MX qutlass library is compiled only for sm_100. This is why the project uses single-card instances, BF16/FP8, and keeps TP out of the default path.

## 3. System

The offline step embeds the label pool once. Online requests perform a brute-force matrix multiply, retain 32 or 64 candidates, and score them with the merged LoRA Reranker-0.6B. The FastAPI demo returns stage timings and falls back to recall top-25 when reranking exceeds the request timeout.

## 4. E1: Prompt Layout and Prefix Cache

Formal group-mode results:

| Variant | Cache hit | Candidates/s | E2E p99 |
|---|---:|---:|---:|
| A0, document first | 0.407 | 692.36 | 119.58 ms |
| A1, document last | 0.847 | 1,001.64 | 51.47 ms |
| A2, cache off | 0.000 | 586.30 | 128.28 ms |

A1 is 1.45x A0. The hit-rate separation validates prompt-prefix placement, but the 2x G4 gate is not met. Candidate-by-candidate diagnostic runs were also retained; they showed Python request overhead masking cache savings, so they are not used for the formal gate.

## 5. E4: Scoring Path

| Variant | Decode | Allowed IDs | Candidates/s | TTFT p99 | E2E p99 |
|---|---:|---|---:|---:|---:|
| D0 | 16 tokens | no | 242.03 | 59.80 ms | 331.92 ms |
| D1 | 1 token | no | 977.32 | 43.42 ms | 52.95 ms |
| D2 | 1 token | yes/no | 1,005.25 | 41.55 ms | 51.49 ms |

The service therefore uses one-token logit scoring. Full generation is a clear anti-pattern for this prefill-bound workload, while vocabulary restriction is a small but measurable optimization.

## 6. E5: Quantization

| Variant | MAP@25 | Candidates/s | TTFT p99 | E2E p99 | Peak memory |
|---|---:|---:|---:|---:|---:|
| BF16 | 0.3953 | 976.30 | 49.85 ms | 61.98 ms | 27.95 GB |
| Online FP8 | 0.3843 | 908.77 | 65.54 ms | 75.05 ms | 27.91 GB |
| FP8 + KV-FP8 | 0.3977 | 911.89 | 48.94 ms | 57.34 ms | 28.31 GB |

FP8 weight quantization did not improve this short-prefill workload and lost 0.011 MAP. KV-FP8 is kept as an explicit feature flag because it preserved quality and reduced tail latency in this run. BF16 remains the default rollback path.

The earlier D0-B probes found `gptq_marlin` and `awq_marlin` falling back to generic loaders, while `modelopt_fp4` lacked a configuration and MXFP4 lacked sm_120 kernels. They are recorded as engineering decisions, not hidden failures.

## 7. Delivery Decisions

The delivery path favors one-card BF16 inference with a 32-candidate default, an optional 64-candidate setting, and explicit `--kv-cache-dtype fp8`. The API exposes recall and rerank timing, candidate count, and a timeout fallback so the benchmark numbers map to an operational search workflow.

## 8. Reproduction

```bash
conda activate py312
./scripts/run_e4_scoring_paths.sh
./scripts/run_e5_quantization.sh
python -m src.serve.serve --port 8000
python -m src.serve.bench_serve --requests 100 --concurrency 4
```

No E2/E3 run is presented as a headline result because G4 did not pass and the project deliberately avoids spending GPU time on a cache mainline whose measured benefit is below the decision threshold.
