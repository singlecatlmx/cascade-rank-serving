import math
import statistics


def recall_at_k(actual, predicted, k):
    actual = set(actual)
    if not actual:
        return 0.0
    return len(actual.intersection(predicted[:k])) / len(actual)


def map_at_k(actual, predicted, k):
    actual = set(actual)
    if not actual:
        return 0.0
    hits = 0
    score = 0.0
    seen = set()
    for rank, item in enumerate(predicted[:k], start=1):
        if item in actual and item not in seen:
            hits += 1
            score += hits / rank
            seen.add(item)
    return score / min(len(actual), k)


def ndcg_at_k(actual, predicted, k):
    actual = set(actual)
    if not actual:
        return 0.0
    seen = set()
    dcg = 0.0
    for rank, item in enumerate(predicted[:k], start=1):
        if item in actual and item not in seen:
            dcg += 1.0 / math.log2(rank + 1)
            seen.add(item)
    ideal_hits = min(len(actual), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def latency_stats(values):
    values = list(values)
    if not values:
        raise ValueError("latency values cannot be empty")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def main():
    assert recall_at_k([1, 3], [1, 2, 4], 3) == 0.5
    assert math.isclose(map_at_k([1, 3], [1, 2, 3], 25), 5 / 6)
    assert math.isclose(ndcg_at_k([1, 3], [1, 2, 3], 3), 0.9197207891481876)
    stats = latency_stats([1, 2, 3, 4, 5])
    assert stats["p50"] == 3.0 and math.isclose(stats["p95"], 4.8)
    print("CPU metrics self-check passed")


if __name__ == "__main__":
    main()
