import json
import random
from pathlib import Path

import pandas as pd


SEED = 20260901
EVAL_SIZE = 200
RAW_DIR = Path("/workspace/data/raw")
EVAL_PATH = Path("data/eval_set_v1.jsonl")
LABEL_PATH = Path("data/label_pool_v1.jsonl")


def build_queries(train, folds):
    fold_zero_ids = set(folds.loc[folds["kfold"] == 0, "QuestionId"].astype(int))
    fold_zero = train[train["QuestionId"].astype(int).isin(fold_zero_ids)]
    queries = []
    for row in fold_zero.to_dict(orient="records"):
        correct = row["CorrectAnswer"]
        correct_text = row[f"Answer{correct}Text"]
        for answer in "ABCD":
            misconception_id = row[f"Misconception{answer}Id"]
            if answer == correct or pd.isna(misconception_id):
                continue
            query_id = f"{int(row['QuestionId'])}_{answer}"
            query_text = (
                f"Subject: {row['SubjectName']}\n"
                f"Topic: {row['ConstructName']}\n"
                f"Question: {row['QuestionText']}\n"
                f"Correct Answer: {correct_text}\n"
                f"Incorrect Answer: {row[f'Answer{answer}Text']}"
            )
            queries.append(
                {
                    "query_id": query_id,
                    "question_id": int(row["QuestionId"]),
                    "answer": answer,
                    "query": query_text,
                    "actual": [int(misconception_id)],
                }
            )
    return sorted(queries, key=lambda item: item["query_id"])


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    if EVAL_PATH.exists() or LABEL_PATH.exists():
        raise SystemExit("frozen data already exists; refusing to overwrite")

    train_path = RAW_DIR / "competition/train.csv"
    mapping_path = RAW_DIR / "eedi-silver-v3/misconception_mapping.csv"
    folds_path = RAW_DIR / "eedi-five-folds/folds.parquet"
    for path in (train_path, mapping_path, folds_path):
        if not path.is_file():
            raise SystemExit(f"required input missing: {path}")

    train = pd.read_csv(train_path)
    mapping = pd.read_csv(mapping_path).sort_values("MisconceptionId")
    folds = pd.read_parquet(folds_path)
    if not {"QuestionId", "kfold"}.issubset(folds.columns):
        raise SystemExit(f"unexpected folds columns: {list(folds.columns)}")
    if len(mapping) != 4791:
        raise SystemExit(f"expected 4791 labels, found {len(mapping)}")
    if mapping["MisconceptionId"].nunique() != len(mapping):
        raise SystemExit("duplicate MisconceptionId values in label mapping")

    queries = build_queries(train, folds)
    if len(queries) < EVAL_SIZE:
        raise SystemExit(f"fold 0 has only {len(queries)} labeled queries")
    selected = random.Random(SEED).sample(queries, EVAL_SIZE)
    selected.sort(key=lambda item: item["query_id"])
    label_ids = set(mapping["MisconceptionId"].astype(int))
    missing_label_ids = sorted({item["actual"][0] for item in selected} - label_ids)
    if missing_label_ids:
        raise SystemExit(f"eval labels missing from label pool: {missing_label_ids}")

    EVAL_PATH.parent.mkdir(exist_ok=True)
    meta = {
        "_meta": {
            "version": "eval_set_v1",
            "seed": SEED,
            "fold": 0,
            "size": EVAL_SIZE,
            "source": "eedi-mining-misconceptions-in-mathematics",
            "label_source": "conjuring92/eedi-silver-v3",
        }
    }
    write_jsonl(EVAL_PATH, [meta, *selected])
    labels = [
        {
            "label_id": int(row.MisconceptionId),
            "label": row.MisconceptionName,
        }
        for row in mapping.itertuples(index=False)
    ]
    write_jsonl(LABEL_PATH, labels)
    print(f"wrote {EVAL_PATH}: {len(selected)} queries + metadata")
    print(f"wrote {LABEL_PATH}: {len(labels)} labels")


if __name__ == "__main__":
    main()
