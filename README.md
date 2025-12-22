# llm-inference-benchmark

Benchmarking framework for low-latency LLM serving, evaluating batching, caching, and concurrency tradeoffs using vLLM and SGLang.

---

## Overview

This repository implements a **benchmarking framework for serving large language models (LLMs)** under realistic request workloads. The goal is to quantify system-level tradeoffs between **latency, throughput, and concurrency** when deploying LLMs in production environments.

The framework focuses on evaluating modern LLM serving stacks and isolates the impact of batching strategies, KV-cache reuse, and request scheduling on end-to-end performance.

---

## System Architecture

### Components

**Serving Backends**
- **vLLM** for optimized KV-cache management and continuous batching
- **SGLang** for programmable serving and request scheduling

**Workload Generation**
- Synthetic and trace-driven request workloads
- Configurable request rates, sequence lengths, and concurrency levels

**Metrics Collection**
- End-to-end latency (P50 / P95 / P99)
- Throughput (tokens/sec)
- GPU utilization and memory footprint

---

### Data Flow

```text
Client Requests
      |
      v
+----------------------+
|  Load Generator      |
| (Rate / Concurrency) |
+----------------------+
            |
            v
+------------------------------+
|   LLM Serving Backend        |
|   vLLM / SGLang              |
|  - Dynamic Batching          |
|  - KV-Cache Reuse            |
+------------------------------+
            |
            v
+------------------------------+
|   Inference Execution        |
|   (GPU / CUDA Kernels)       |
+------------------------------+
            |
            v
+------------------------------+
|   Metrics Collection         |
|  Latency / Throughput        |
|  GPU Utilization             |
+------------------------------+
````

---

## Benchmark Dimensions

The framework evaluates the following dimensions:

* **Batching Strategy**

  * Static vs dynamic batching
* **KV-Cache Reuse**

  * Impact on latency and memory efficiency
* **Concurrency**

  * Throughput saturation and tail latency behavior
* **Deployment Configuration**

  * Model size, sequence length, and hardware constraints

Results are reported using latency percentiles and aggregate throughput metrics.

---

## Evaluation

Benchmark results are analyzed to identify:

* Throughput--latency tradeoffs under increasing load
* Tail latency degradation at high concurrency
* GPU memory bottlenecks driven by KV-cache growth
* System-level constraints relevant to high-traffic LLM services

The evaluation is designed to reflect production serving conditions rather than single-request performance.

---

## Repository Structure

```text
llm-inference-benchmark/
├── backends/          # vLLM and SGLang serving configs
├── workloads/         # Synthetic and trace-based workloads
├── benchmarks/        # Benchmark runners
├── metrics/           # Latency and throughput analysis
├── scripts/           # Experiment orchestration
├── results/           # Logged benchmark outputs
├── README.md
├── LICENSE
└── .gitignore
```

---

## Design Principles

* Reproducible benchmarking under controlled workloads
* Explicit separation of serving, workload generation, and metrics
* Focus on system-level tradeoffs over micro-optimizations
* Production-oriented evaluation methodology

---

## Notes

This project is intended as an applied ML systems benchmarking study and is structured to support extension to additional serving backends, hardware configurations, and workload traces.

---

## License

MIT License
