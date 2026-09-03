import argparse
import json
import math
import random
import time
from collections import deque
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from src.data.prompts import prompt_token_ids
from src.metrics import map_at_k


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def load_groups(path):
    columns = [
        "query_id", "content_id", "SubjectName", "ConstructName", "QuestionText",
        "CorrectAnswerText", "InCorrectAnswerText", "MisconceptionName", "label",
    ]
    frame = pd.read_parquet(path, columns=columns)
    groups = []
    for query_id, rows in frame.groupby("query_id", sort=True):
        positive = rows[rows["label"] == 1]
        negatives = rows[rows["label"] == 0]
        if len(positive) != 1 or len(negatives) < 8:
            raise RuntimeError(f"invalid hard-negative group {query_id}: pos={len(positive)} neg={len(negatives)}")
        row = positive.iloc[0]
        query = (
            f"Subject: {row['SubjectName']}\nTopic: {row['ConstructName']}\n"
            f"Question: {row['QuestionText']}\nCorrect Answer: {row['CorrectAnswerText']}\n"
            f"Incorrect Answer: {row['InCorrectAnswerText']}"
        )
        groups.append({
            "query_id": query_id,
            "positive_id": int(row["content_id"]),
            "query": query,
            "positive": row["MisconceptionName"],
            "negatives": list(zip(negatives["content_id"].astype(int), negatives["MisconceptionName"])),
        })
    return groups


def group_batches(groups, size, rng):
    order = list(range(len(groups)))
    rng.shuffle(order)
    pending = deque(order)
    while pending:
        batch = []
        positive_ids = set()
        deferred = []
        while pending and len(batch) < size:
            index = pending.popleft()
            positive_id = groups[index]["positive_id"]
            if positive_id in positive_ids:
                deferred.append(index)
            else:
                batch.append(groups[index])
                positive_ids.add(positive_id)
        pending.extend(deferred)
        yield batch


def make_batch(groups, tokenizer, group_size, max_length, pad_to_multiple_of, rng):
    token_rows = []
    targets = []
    for group in groups:
        candidates = [(group["positive_id"], group["positive"])]
        candidates += rng.sample(group["negatives"], group_size - 1)
        rng.shuffle(candidates)
        targets.append(next(i for i, item in enumerate(candidates) if item[0] == group["positive_id"]))
        token_rows.extend(
            {"input_ids": prompt_token_ids(tokenizer, group["query"], document, "a1_document_last")[-max_length:]}
            for _, document in candidates
        )
    batch = tokenizer.pad(token_rows, padding=True, pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt")
    return {key: value.cuda() for key, value in batch.items()}, torch.tensor(targets, device="cuda")


def evaluate(model, tokenizer, candidates_path, labels_path, max_length, pad_to_multiple_of):
    candidate_lines = read_jsonl(candidates_path)
    rows = candidate_lines[21:]
    labels = {row["label_id"]: row["label"] for row in read_jsonl(labels_path)}
    scores = []
    model.eval()
    with torch.inference_mode():
        for row in rows:
            token_rows = [
                {"input_ids": prompt_token_ids(tokenizer, row["query"], labels[label_id], "a1_document_last")[-max_length:]}
                for label_id in row["candidate_ids"][:32]
            ]
            batch = tokenizer.pad(token_rows, padding=True, pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt")
            batch = {key: value.cuda() for key, value in batch.items()}
            logits = model(**batch, logits_to_keep=1).logits[:, -1]
            values = (logits[:, tokenizer.convert_tokens_to_ids("yes")] - logits[:, tokenizer.convert_tokens_to_ids("no")]).float().cpu().tolist()
            ranking = [item for _, item in sorted(zip(values, row["candidate_ids"][:32]), reverse=True)]
            scores.append(map_at_k(ranking, row["actual"], 25))
    model.train()
    return sum(scores) / len(scores)


def save_adapter(model, tokenizer, path):
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/train/reranker_0.6b.yaml")
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible GPU, found {torch.cuda.device_count()}")
    if config["group_size"] != 9 or config["tied_embeddings"] != "accept_frozen" or config["devices"] != 1:
        raise RuntimeError("D3 requires group_size=9 and frozen tied embeddings")
    if config["dtype"] != "bfloat16" or config["attention"] != "sdpa":
        raise RuntimeError("D3 requires bfloat16 and SDPA")
    if not args.smoke_steps:
        for output in (config["output_lora"], config["output_merged"]):
            if Path(output).exists():
                raise SystemExit(f"refusing to overwrite {output}")

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], local_files_only=True, padding_side="left", truncation_side="left"
    )
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    if tokenizer.encode("yes", add_special_tokens=False) != [yes_id] or tokenizer.encode("no", add_special_tokens=False) != [no_id]:
        raise RuntimeError("yes/no must each be one token")
    groups = load_groups(config["train_data"])
    print(json.dumps({"groups": len(groups), "group_size": config["group_size"], "config": config}))

    base = AutoModelForCausalLM.from_pretrained(
        config["model"], local_files_only=True, dtype=torch.bfloat16, attn_implementation=config["attention"]
    )
    if not base.config.tie_word_embeddings:
        raise RuntimeError("expected tied input/output embeddings")
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(
        r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"], bias="none", task_type=TaskType.CAUSAL_LM,
    ))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.cuda()
    if model.get_input_embeddings().weight.requires_grad or model.get_output_embeddings().weight.requires_grad:
        raise RuntimeError("tied embeddings must remain frozen")
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    lora_a = [parameter for name, parameter in trainable if "lora_A" in name]
    lora_b = [parameter for name, parameter in trainable if "lora_B" in name]
    if len(lora_a) + len(lora_b) != len(trainable):
        raise RuntimeError("unexpected trainable parameters outside LoRA A/B")
    optimizer = torch.optim.AdamW([
        {"params": lora_a, "lr": config["lr_lora_a"]},
        {"params": lora_b, "lr": config["lr_lora_b"]},
    ], weight_decay=config["weight_decay"], betas=(config["adam_beta1"], config["adam_beta2"]), eps=config["adam_epsilon"])
    steps_per_epoch = math.ceil(len(groups) / config["groups_per_batch"])
    total_steps = args.smoke_steps or steps_per_epoch * config["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(total_steps * config["warmup_ratio"])), total_steps
    )
    rng = random.Random(config["seed"])
    best_map = -1.0
    started = time.perf_counter()
    step = 0
    model.train()
    for epoch in range(config["epochs"]):
        for selected in group_batches(groups, config["groups_per_batch"], rng):
            batch, targets = make_batch(
                selected, tokenizer, config["group_size"], config["max_length"], config["pad_to_multiple_of"], rng
            )
            logits = model(**batch, logits_to_keep=1).logits[:, -1]
            rank_logits = (logits[:, yes_id] - logits[:, no_id]).reshape(len(selected), config["group_size"])
            loss = F.cross_entropy(rank_logits.float(), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step == 1 or step % 50 == 0:
                print(json.dumps({"step": step, "total_steps": total_steps, "loss": loss.item(), "seconds": time.perf_counter() - started}), flush=True)
            if args.smoke_steps and step >= args.smoke_steps:
                print(json.dumps({"smoke": "PASS", "steps": step, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}))
                return
            if step % config["eval_every"] == 0 or step == total_steps:
                score = evaluate(
                    model, tokenizer, config["eval_candidates"], config["label_pool"],
                    config["max_length"], config["pad_to_multiple_of"],
                )
                print(json.dumps({"step": step, "validation_map@25": score}), flush=True)
                if score > best_map:
                    best_map = score
                    save_adapter(model, tokenizer, Path(config["output_lora"]) / "best")
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    save_adapter(model, tokenizer, Path(config["output_lora"]) / "last")
    del optimizer, scheduler, model, base
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(
        config["model"], local_files_only=True, dtype=torch.bfloat16, attn_implementation=config["attention"]
    )
    merged = PeftModel.from_pretrained(base, Path(config["output_lora"]) / "best").merge_and_unload()
    merged.save_pretrained(config["output_merged"], safe_serialization=True)
    tokenizer.save_pretrained(config["output_merged"])
    print(json.dumps({"training": "PASS", "steps": step, "best_map@25": best_map, "seconds": time.perf_counter() - started}))


if __name__ == "__main__":
    main()
