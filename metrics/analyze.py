"""
Metrics Analysis for LLM Inference Benchmarks

Responsibilities:
- Load raw benchmark logs (JSONL)
- Compute latency percentiles (P50 / P95 / P99)
- Compute throughput metrics
- Produce summary statistics for comparison

This module is read-only with respect to benchmark outputs.
"""

import json
import argparse
from typing import Dict, List
import numpy as np


def load_results(path: str) -> List[Dict]:
    """
    Load JSONL benchmark results.
    """
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def latency_percentiles(latencies: np.ndarray) -> Dict[str, float]:
    """
    Compute latency percentiles.
    """
    return {
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
    }


def compute_throughput(
    records: List[Dict],
) -> Dict[str, float]:
    """
    Compute throughput metrics.
    """
    if len(records) == 0:
        return {}

    start_time = min(r["arrival_time"] for r in records)
    end_time = max(r["arrival_time"] + r["latency_s"] for r in records)
    duration = max(1e-6, end_time - start_time)

    total_requests = len(records)
    total_generated_tokens = sum(r["max_new_tokens"] for r in records)

    return {
        "requests_per_sec": total_requests / duration,
        "tokens_per_sec": total_generated_tokens / duration,
        "duration_s": duration,
    }


def analyze(path: str) -> Dict[str, float]:
    """
    Analyze benchmark results and return summary metrics.
    """
    records = load_results(path)

    latencies = np.array([r["latency_s"] for r in records], dtype=np.float32)

    summary = {}
    summary.update(latency_percentiles(latencies))
    summary.update(compute_throughput(records))
    summary["num_requests"] = len(records)

    return summary


def print_summary(summary: Dict[str, float]):
    """
    Pretty-print metrics summary.
    """
    print("📊 Benchmark Summary")
    print("-" * 30)
    print(f"Requests       : {summary['num_requests']}")
    print(f"Duration (s)   : {summary['duration_s']:.2f}")
    print(f"Req/sec        : {summary['requests_per_sec']:.2f}")
    print(f"Tokens/sec     : {summary['tokens_per_sec']:.2f}")
    print(f"P50 Latency    : {summary['p50']:.3f}s")
    print(f"P95 Latency    : {summary['p95']:.3f}s")
    print(f"P99 Latency    : {summary['p99']:.3f}s")


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM benchmark results")
    parser.add_argument("--input", type=str, required=True, help="Path to JSONL results file")
    args = parser.parse_args()

    summary = analyze(args.input)
    print_summary(summary)


if __name__ == "__main__":
    main()
