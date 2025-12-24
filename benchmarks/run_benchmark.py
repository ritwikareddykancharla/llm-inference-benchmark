"""
Benchmark Runner

Responsibilities:
- Consume a workload stream
- Dispatch requests to a serving backend
- Collect latency and throughput metrics
- Log raw results for offline analysis

This module is backend-agnostic.
"""

import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

from workloads.synthetic import SyntheticWorkload, LLMRequest
from backends.vllm_server import VLLMServer
from backends.sglang_server import SGLangServer


class BenchmarkRunner:
    def __init__(
        self,
        backend: str,
        model: str,
        concurrency: int,
        output_path: str,
    ):
        """
        Args:
            backend: 'vllm' or 'sglang'
            model: model name or path
            concurrency: number of concurrent workers
            output_path: path to save raw results (JSONL)
        """
        self.backend_name = backend
        self.concurrency = concurrency
        self.output_path = output_path

        if backend == "vllm":
            self.server = VLLMServer(model=model)
        elif backend == "sglang":
            self.server = SGLangServer(model=model)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.lock = threading.Lock()

    def run(self, workload: SyntheticWorkload):
        """
        Run benchmark on a given workload.
        """
        self.server.start()

        results: List[Dict] = []

        def _handle_request(req: LLMRequest):
            prompt = "x " * req.prompt_tokens
            start = time.time()

            response = self.server.generate(
                prompt=prompt,
                max_tokens=req.max_new_tokens,
            )

            end = time.time()

            record = {
                "request_id": req.request_id,
                "arrival_time": req.arrival_time,
                "prompt_tokens": req.prompt_tokens,
                "max_new_tokens": req.max_new_tokens,
                "latency_s": response["latency_s"],
                "wall_time_s": end - start,
            }

            with self.lock:
                results.append(record)

        # Dispatch requests with bounded concurrency
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = []
            for req in workload.generate():
                futures.append(executor.submit(_handle_request, req))

            for f in as_completed(futures):
                pass

        self.server.stop()

        # Save raw results
        with open(self.output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        print(f"✅ Benchmark complete. Results saved to {self.output_path}")
