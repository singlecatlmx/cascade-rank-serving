import argparse
import json
import subprocess
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def load_first_eval_prompt(tokenizer):
    from src.data.prompts import prompt_token_ids

    with Path("data/eval_set_v1.jsonl").open(encoding="utf-8") as handle:
        next(handle)
        query = json.loads(next(handle))
    labels = {
        row["label_id"]: row["label"]
        for row in (
            json.loads(line)
            for line in Path("data/label_pool_v1.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    return prompt_token_ids(
        tokenizer,
        query["query"],
        labels[query["actual"][0]],
        "a1_document_last",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--quantization", default="none")
    parser.add_argument("--prompt-mode", choices=("plain", "reranker"), default="plain")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc)
    config = {
        "case": args.case,
        "model": args.model,
        "quantization": args.quantization,
        "prompt_mode": args.prompt_mode,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.85,
        "max_model_len": 512,
        "enforce_eager": True,
        "max_tokens": 1,
        "allowed_tokens": ["yes", "no"],
    }
    result = {
        "git_commit": git_commit(),
        "vllm_version": metadata.version("vllm"),
        "gpu": "unknown",
        "timestamp": timestamp.isoformat(),
        "config": config,
        "status": "failed",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite {output_path}")

    started = time.perf_counter()
    try:
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.inputs.data import TokensPrompt

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(f"expected one visible GPU, found {torch.cuda.device_count()}")
        if not Path(args.model).is_dir():
            raise FileNotFoundError(f"model directory missing: {args.model}")
        result["gpu"] = torch.cuda.get_device_name(0)

        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        yes_ids = tokenizer.encode("yes", add_special_tokens=False)
        no_ids = tokenizer.encode("no", add_special_tokens=False)
        if len(yes_ids) != 1 or len(no_ids) != 1:
            raise RuntimeError(f"yes/no are not single tokens: yes={yes_ids}, no={no_ids}")

        llm_args = {
            "model": args.model,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": config["gpu_memory_utilization"],
            "max_model_len": config["max_model_len"],
            "enforce_eager": True,
        }
        if args.quantization != "none":
            llm_args["quantization"] = args.quantization
        llm = LLM(**llm_args)
        actual_quantization = llm.model_config.quantization
        expected_quantization = None if args.quantization == "none" else args.quantization
        if actual_quantization != expected_quantization:
            raise RuntimeError(
                "quantization fallback detected: "
                f"requested={expected_quantization}, actual={actual_quantization}"
            )
        result["resolved_model_config"] = {
            "quantization": actual_quantization,
            "dtype": str(llm.model_config.dtype),
        }

        if args.prompt_mode == "reranker":
            prompt = TokensPrompt(prompt_token_ids=load_first_eval_prompt(tokenizer))
        else:
            prompt = "Answer yes or no: Is two plus two equal to four?"
        sampling = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[yes_ids[0], no_ids[0]],
        )
        request_output = llm.generate([prompt], sampling, use_tqdm=False)[0]
        completion = request_output.outputs[0]
        step_logprobs = completion.logprobs[-1]
        if yes_ids[0] not in step_logprobs or no_ids[0] not in step_logprobs:
            raise RuntimeError(f"yes/no missing from logprobs: {list(step_logprobs)}")

        result["status"] = "passed"
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        result["output"] = {
            "text": completion.text,
            "token_ids": list(completion.token_ids),
            "yes_token_id": yes_ids[0],
            "no_token_id": no_ids[0],
            "yes_logprob": float(step_logprobs[yes_ids[0]].logprob),
            "no_logprob": float(step_logprobs[no_ids[0]].logprob),
        }
    except Exception as exc:
        result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case": args.case, "status": result["status"], "output": str(output_path)}))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
