"""
Synthetic Workload Generator for LLM Inference Benchmarking

Responsibilities:
- Generate realistic LLM request workloads
- Control arrival rate (QPS), concurrency, and sequence lengths
- Support steady-state and bursty traffic patterns

This module is backend-agnostic.
"""

from dataclasses import dataclass
from typing import Iterator, List, Dict
import random
import time
import math


@dataclass
class LLMRequest:
    """
    Represents a single LLM inference request.
    """
    request_id: int
    prompt_tokens: int
    max_new_tokens: int
    arrival_time: float


class SyntheticWorkload:
    def __init__(
        self,
        qps: float,
        duration_s: int,
        prompt_len_range: tuple = (32, 512),
        max_new_tokens_range: tuple = (64, 256),
        burst_prob: float = 0.0,
        burst_multiplier: float = 3.0,
        seed: int = 42,
    ):
        """
        Args:
            qps: average queries per second
            duration_s: total workload duration in seconds
            prompt_len_range: (min, max) prompt length in tokens
            max_new_tokens_range: (min, max) generation length
            burst_prob: probability of traffic burst at any second
            burst_multiplier: QPS multiplier during burst
            seed: RNG seed for reproducibility
        """
        self.qps = qps
        self.duration_s = duration_s
        self.prompt_len_range = prompt_len_range
        self.max_new_tokens_range = max_new_tokens_range
        self.burst_prob = burst_prob
        self.burst_multiplier = burst_multiplier

        random.seed(seed)

    def _requests_in_second(self) -> int:
        """
        Sample number of requests arriving in one second.
        Uses Poisson arrivals with optional bursts.
        """
        effective_qps = self.qps
        if random.random() < self.burst_prob:
            effective_qps *= self.burst_multiplier

        # Poisson arrivals
        return max(0, int(random.poisson(effective_qps))) if hasattr(random, "poisson") \
            else max(0, int(random.gauss(effective_qps, math.sqrt(effective_qps))))

    def generate(self) -> Iterator[LLMRequest]:
        """
        Generate a stream of LLMRequest objects.
        """
        request_id = 0
        start_time = time.time()

        for sec in range(self.duration_s):
            num_requests = self._requests_in_second()
            for _ in range(num_requests):
                prompt_tokens = random.randint(*self.prompt_len_range)
                max_new_tokens = random.randint(*self.max_new_tokens_range)

                yield LLMRequest(
                    request_id=request_id,
                    prompt_tokens=prompt_tokens,
                    max_new_tokens=max_new_tokens,
                    arrival_time=start_time + sec,
                )
                request_id += 1

    def to_list(self) -> List[LLMRequest]:
        """
        Materialize workload into a list (for offline benchmarks).
        """
        return list(self.generate())


if __name__ == "__main__":
    # Quick sanity test
    workload = SyntheticWorkload(
        qps=10,
        duration_s=5,
        burst_prob=0.2,
        burst_multiplier=4.0,
    )

    requests = workload.to_list()
    print(f"Generated {len(requests)} requests")
    print(requests[:3])
