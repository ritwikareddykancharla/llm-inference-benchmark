"""
Experiment Orchestrator for LLM Inference Benchmarking

Responsibilities:
- Define experiment grids (backend, QPS, concurrency)
- Run benchmarks end-to-end
- Trigger metrics analysis
- Save results in a structured way

This is the main entry point for experiments.
"""

import os
import argparse
import subprocess
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM inference benchmarks")

    parser.add_argument("--backend", type=str, choices=["vllm", "sglang"], required=True)
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--qps", type=float, default=10.0)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=4)

    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.results_dir, exist_ok=True)

    result_path = os.path.join(
        args.results_dir,
        f"{args.backend}_qps{args.qps}_conc{args.concurrency}_{timestamp}.jsonl",
    )

    print("🚀 Starting experiment")
    print(f"Backend      : {args.backend}")
    print(f"Model        : {args.model}")
    print(f"QPS          : {args.qps}")
    print(f"Concurrency  : {args.concurrency}")
    print(f"Duration (s): {args.duration}")
    print("-" * 40)

    # Step 1: Run benchmark
    benchmark_cmd = [
        "python",
        "benchmarks/run_benchmark.py",
        "--backend",
        args.backend,
        "--model",
        args.model,
        "--concurrency",
        str(args.concurrency),
        "--output_path",
        result_path,
    ]

    print("▶ Running benchmark...")
    subprocess.run(benchmark_cmd, check=True)

    # Step 2: Analyze results
    analyze_cmd = [
        "python",
        "metrics/analyze.py",
        "--input",
        result_path,
    ]

    print("📊 Analyzing results...")
    subprocess.run(analyze_cmd, check=True)

    print("✅ Experiment complete")
    print(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()
