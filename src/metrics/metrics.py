import math


def _validate(preds, gold, k):
    if k <= 0:
        raise ValueError("k must be positive")
    return list(preds[:k]), set(gold)


def recall_at_k(preds, gold, k):
    _, gold_set = _validate(preds, gold, k)
    return len(set(preds[:k]) & gold_set) / len(gold_set) if gold_set else 0.0


def map_at_k(preds, gold, k):
    ranked, gold_set = _validate(preds, gold, k)
    if not gold_set:
        return 0.0
    hits = 0
    score = 0.0
    for rank, item in enumerate(ranked, 1):
        if item in gold_set:
            hits += 1
            score += hits / rank
    return score / min(len(gold_set), k)


def ndcg_at_k(preds, gold, k):
    ranked, gold_set = _validate(preds, gold, k)
    gains = [1.0 if item in gold_set else 0.0 for item in ranked]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold_set), k) + 1))
    return dcg / ideal if ideal else 0.0


def latency_stats(samples):
    values = sorted(float(value) for value in samples)
    if not values:
        raise ValueError("latency samples cannot be empty")

    def percentile(q):
        position = (len(values) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return values[lower]
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "mean": sum(values) / len(values),
    }


assert recall_at_k([1, 2, 3], [2], 3) == 1.0
assert round(map_at_k([2, 1, 3], [1, 2], 3), 6) == round((1 + 2 / 2) / 2, 6)
assert round(ndcg_at_k([2, 1, 3], [1, 2], 3), 6) == 1.0
