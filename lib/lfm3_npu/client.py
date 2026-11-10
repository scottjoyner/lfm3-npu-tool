"""
Python client library for LFM3-1.7B inference servers.

Provides a simple Python interface to the OpenAI-compatible API.
"""

import json
import urllib.request
from typing import List, Optional, Dict, Any


class LFM3Client:
    """Python client for LFM3 inference servers."""

    def __init__(self, base_url: str = "http://localhost:1239"):
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, data: Optional[dict] = None) -> dict:
        url = f"{self.base_url}{path}"
        if data:
            body = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method=method,
            )
        else:
            req = urllib.request.Request(url, method=method)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def health(self) -> dict:
        """Get server health status."""
        return self._request("GET", "/health")

    def list_models(self) -> dict:
        """List available models."""
        return self._request("GET", "/v1/models")

    def model_info(self) -> dict:
        """Get detailed model information."""
        return self._request("GET", "/v1/model/info")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stream: bool = False,
    ) -> dict:
        """Create a chat completion."""
        data = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        return self._request("POST", "/v1/chat/completions", data)

    def chat(self, message: str, max_tokens: int = 50, temperature: float = 0.7) -> str:
        """Simple chat interface - returns just the assistant's response text."""
        response = self.chat_completion(
            messages=[{"role": "user", "content": message}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    import sys
    client = LFM3Client()
    try:
        print(f"Health: {client.health()}")
        response = client.chat("Hello, how are you?", max_tokens=30)
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the server is running: lfm3-npu-server-mgr start npu")