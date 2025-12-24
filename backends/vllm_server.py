"""
vLLM Serving Backend Wrapper

Responsibilities:
- Launch vLLM OpenAI-compatible server
- Send inference requests
- Measure latency and throughput

This module does NOT generate workloads or analyze metrics.
"""

import subprocess
import time
import requests
from typing import Dict, List


class VLLMServer:
    def __init__(
        self,
        model: str,
        host: str = "localhost",
        port: int = 8000,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
    ):
        """
        Args:
            model: HF model name or local path
            host: server host
            port: server port
            max_model_len: max sequence length
            gpu_memory_utilization: fraction of GPU memory to use
        """
        self.model = model
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}/v1"
        self.process = None
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization

    def start(self):
        """
        Start vLLM server as a subprocess.
        """
        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model,
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--port",
            str(self.port),
        ]

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to be ready
        self._wait_until_ready()
        print("✅ vLLM server started")

    def _wait_until_ready(self, timeout_s: int = 60):
        start = time.time()
        while time.time() - start < timeout_s:
            try:
                r = requests.get(f"{self.base_url}/models")
                if r.status_code == 200:
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)

        raise RuntimeError("vLLM server failed to start")

    def stop(self):
        """
        Stop vLLM server.
        """
        if self.process is not None:
            self.process.terminate()
            self.process.wait()
            print("🛑 vLLM server stopped")

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> Dict:
        """
        Send a single generation request.

        Returns:
            dict with latency and output tokens
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        start_time = time.time()
        response = requests.post(
            f"{self.base_url}/completions",
            json=payload,
        )
        end_time = time.time()

        response.raise_for_status()
        result = response.json()

        return {
            "latency_s": end_time - start_time,
            "output_text": result["choices"][0]["text"],
            "usage": result.get("usage", {}),
        }
