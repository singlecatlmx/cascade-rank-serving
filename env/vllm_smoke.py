import argparse
import json
import subprocess
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
)
INSTRUCTION = (
    "Given a math question, its correct answer, and an incorrect answer, retrieve "
    "the misconception that best explains the incorrect answer."
)
QUERY = "Question: What is 2 + 2? Correct answer: 4. Incorrect answer: 5."
DOCUMENT = "The learner adds one too many when combining two quantities."
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="g0_reranker_bf16")
    parser.add_argument("--model", required=True)
    parser.add_argument("--quantization", default="none")
    parser.add_argument("--prompt-mode", choices=("plain", "reranker"), default="plain")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = {
        "case": args.case,
        "model": args.model,
        "dtype": "bfloat16",
        "quantization": None if args.quantization == "none" else args.quantization,
        "prompt_mode": args.prompt_mode,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.85,
        "max_model_len": 512,
        "enforce_eager": True,
        "max_tokens": 1,
        "logprobs": 20,
        "allowed_tokens": ["yes", "no"],
        "prompt": {
            "instruction": INSTRUCTION,
            "query": QUERY,
            "document": DOCUMENT,
        },
    }
    result = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "vllm_version": metadata.version("vllm"),
        "gpu": "unknown",
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
        from vllm.inputs import TokensPrompt

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                f"expected one visible GPU, found {torch.cuda.device_count()}"
            )
        model_path = Path(args.model)
        if not model_path.is_dir():
            raise FileNotFoundError(f"model directory missing: {model_path}")
        result["gpu"] = torch.cuda.get_device_name(0)

        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        yes_token_id = tokenizer.convert_tokens_to_ids("yes")
        no_token_id = tokenizer.convert_tokens_to_ids("no")
        if tokenizer.encode("yes", add_special_tokens=False) != [yes_token_id]:
            raise RuntimeError("yes is not a single token")
        if tokenizer.encode("no", add_special_tokens=False) != [no_token_id]:
            raise RuntimeError("no is not a single token")

        if args.prompt_mode == "reranker":
            prompt = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                "<|im_start|>user\n"
                f"<Instruct>: {INSTRUCTION}\n<Query>: {QUERY}\n<Document>: {DOCUMENT}"
                f"{SUFFIX}"
            )
        else:
            prompt = "Answer yes or no: Is two plus two equal to four?"
        if args.prompt_mode == "reranker" and not prompt.endswith(SUFFIX):
            raise RuntimeError("prompt does not end with the official empty thinking suffix")
        prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_token_ids) > config["max_model_len"]:
            raise RuntimeError(f"prompt has {len(prompt_token_ids)} tokens")

        llm_args = {
            "model": args.model,
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": config["gpu_memory_utilization"],
            "max_model_len": config["max_model_len"],
            "enforce_eager": True,
        }
        if args.quantization != "none":
            llm_args["quantization"] = args.quantization
        llm = LLM(**llm_args)
        expected_quantization = None if args.quantization == "none" else args.quantization
        if llm.model_config.quantization != expected_quantization:
            raise RuntimeError(
                "quantization fallback detected: "
                f"requested={expected_quantization}, actual={llm.model_config.quantization}"
            )
        if args.quantization == "none" and llm.model_config.dtype != torch.bfloat16:
            raise RuntimeError(f"unexpected dtype: {llm.model_config.dtype}")
        result["resolved_model_config"] = {
            "quantization": llm.model_config.quantization,
            "dtype": str(llm.model_config.dtype),
        }

        sampling = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=config["logprobs"],
            allowed_token_ids=[yes_token_id, no_token_id],
        )
        request_output = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_token_ids)],
            sampling,
            use_tqdm=False,
        )[0]
        completion = request_output.outputs[0]
        step_logprobs = completion.logprobs[-1]
        if yes_token_id not in step_logprobs or no_token_id not in step_logprobs:
            raise RuntimeError(f"yes/no missing from logprobs: {list(step_logprobs)}")

        result["status"] = "passed"
        result["output"] = {
            "text": completion.text,
            "token_ids": list(completion.token_ids),
            "prompt_tokens": len(prompt_token_ids),
            "yes_token_id": yes_token_id,
            "no_token_id": no_token_id,
            "yes_logprob": float(step_logprobs[yes_token_id].logprob),
            "no_logprob": float(step_logprobs[no_token_id].logprob),
        }
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case": args.case, "status": result["status"], "output": str(output_path)}))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
