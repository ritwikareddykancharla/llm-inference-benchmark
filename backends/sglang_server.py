"""
SGLang Serving Backend Wrapper

Responsibilities:
- Launch an SGLang server
- Send inference requests
- Measure end-to-end latency

SGLang emphasizes programmable request scheduling and
explicit control over execution flow.
"""

import subprocess
import time
import requests
from typing import Dict


class SGLangServer:
    def __init__(
        self,
        model: str,
        host: str = "localhost",
        port: int = 30000,
        tp_size: int = 1,
    ):
        """
        Args:
            model: HF model name or local path
            host: server host
            port: server port
            tp_size: tensor parallel size
        """
        self.model = model
        self.host = host
        self.port = port
        self.tp_size = tp_size
        self.base_url = f"http://{host}:{port}"
        self.process = None

    def start(self):
        """
        Start SGLang server as a subprocess.
        """
        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model,
            "--tp-size",
            str(self.tp_size),
            "--port",
            str(self.port),
        ]

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self._wait_until_ready()
        print("✅ SGLang server started")

    def _wait_until_ready(self, timeout_s: int = 60):
        start = time.time()
        while time.time() - start < timeout_s:
            try:
                r = requests.get(f"{self.base_url}/health")
                if r.status_code == 200:
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)

        raise RuntimeError("SGLang server failed to start")

    def stop(self):
        """
        Stop SGLang server.
        """
        if self.process is not None:
            self.process.terminate()
            self.process.wait()
            print("🛑 SGLang server stopped")

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> Dict:
        """
        Send a single generation request.

        Returns:
            dict with latency and output text
        """
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        }

        start_time = time.time()
        response = requests.post(
            f"{self.base_url}/generate",
            json=payload,
        )
        end_time = time.time()

        response.raise_for_status()
        result = response.json()

        return {
            "latency_s": end_time - start_time,
            "output_text": result.get("text", ""),
            "meta": result.get("meta", {}),
        }
