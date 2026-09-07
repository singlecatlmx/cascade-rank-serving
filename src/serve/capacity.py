import argparse
import json
import subprocess
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
from pathlib import Path

import matplotlib.pyplot as plt


FORMAL_FILES = {
    "a0_cache_on": "e1_a0_20260906-083104_76fbe0c5.json",
    "a1_cache_on": "e1_a1_20260906-083058_a63fd7c8.json",
    "a2_cache_off": "e1_a2_20260906-083147_d96795f2.json",
    "d0_full_decode": "e4_d0_20260906-084114_4a18855d.json",
    "d1_one_token": "e4_d1_20260906-084034_eb42238c.json",
    "d2_allowed_token_ids": "e4_d2_20260906-084148_95fb6596.json",
    "bf16": "e5_bf16_20260906-084932_14485910.json",
    "fp8": "e5_fp8_20260906-084934_c127d47c.json",
    "fp8_kv_fp8": "e5_fp8_kvfp8_20260906-085026_0a5a7c37.json",
}


def load_points(results_dir):
    points = []
    for name, filename in FORMAL_FILES.items():
        path = Path(results_dir) / filename
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data["metrics"]
        points.append({
            "name": name,
            "source": filename,
            "map_at_25": metrics["quality"]["map@25"],
            "p99_ms": metrics["latency_ms"]["e2e_p99"],
            "req_per_s": metrics["throughput"]["req_per_s"],
            "candidates_scored_per_s": metrics["throughput"]["candidates_scored_per_s"],
            "gpu_seconds_per_1000_queries": 1000 / metrics["throughput"]["req_per_s"],
            "peak_mem_gb": metrics["resource"]["peak_mem_gb"],
        })
    return points


def best_under(points, budget):
    feasible = [point for point in points if point["p99_ms"] <= budget]
    return max(feasible, key=lambda point: (point["map_at_25"], -point["p99_ms"])) if feasible else None


def pareto(points):
    frontier = []
    for point in sorted(points, key=lambda item: item["p99_ms"]):
        if not frontier or point["map_at_25"] > max(item["map_at_25"] for item in frontier):
            frontier.append(point)
    return frontier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--assets-dir", default="assets")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    points = load_points(args.results_dir)
    budgets = {str(budget): best_under(points, budget) for budget in (150, 200, 300)}
    frontier = pareto(points)
    config = {
        "stage": "d7",
        "variant": "capacity_plan",
        "source_files": [point["source"] for point in points],
        "budgets_ms": [150, 200, 300],
        "card_cost_yuan_per_hour": 2.43,
        "note": "Observed single-card benchmark points only; no synthetic load curve or unrun topology result.",
    }
    timestamp = datetime.now(timezone.utc)
    digest = sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    output_path = Path(args.output_dir) / f"d7_capacity_{timestamp.strftime('%Y%m%d-%H%M%S')}_{digest}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "vllm_version": metadata.version("vllm"),
        "gpu": "not applicable (result recombination; no inference run)",
        "timestamp": timestamp.strftime("%Y%m%d-%H%M%S"),
        "config": config,
        "observed_points": points,
        "pareto_frontier": frontier,
        "best_under_budget": budgets,
        "knee_point": None,
    }, indent=2) + "\n", encoding="utf-8")

    Path(args.assets_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    ax.scatter([point["p99_ms"] for point in points], [point["map_at_25"] for point in points], color="#4c78a8", alpha=0.7)
    ax.plot([point["p99_ms"] for point in frontier], [point["map_at_25"] for point in frontier], color="#e45756", linewidth=2)
    for budget in (150, 200, 300):
        ax.axvline(budget, color="#999999", linestyle="--", linewidth=0.8)
        ax.text(budget + 2, ax.get_ylim()[0], f"{budget} ms", rotation=90, va="bottom", fontsize=8)
    for point in frontier:
        ax.annotate(point["name"], (point["p99_ms"], point["map_at_25"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("E2E P99 latency (ms)")
    ax.set_ylabel("MAP@25")
    ax.set_title("Observed quality-latency Pareto frontier")
    fig.tight_layout()
    chart_path = Path(args.assets_dir) / "capacity_pareto.png"
    fig.savefig(chart_path, bbox_inches="tight")
    print(json.dumps({"result": str(output_path), "chart": str(chart_path), "frontier": [point["name"] for point in frontier]}))


if __name__ == "__main__":
    main()
