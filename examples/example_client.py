#!/usr/bin/env python3
"""
Example: Use the Python client library to interact with the NPU server.

Requires the NPU server to be running on port 1239.
"""

import sys
import os

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SERVER_DIR, "lib"))

from lfm3_npu.client import LFM3Client


def main():
    client = LFM3Client(base_url="http://localhost:1239")

    try:
        health = client.health()
        print(f"Health: {health}")
    except Exception as e:
        print(f"Server not reachable: {e}")
        print("Start the server first: lfm3-npu-server-mgr start npu")
        return 1

    models = client.list_models()
    print(f"Models: {models}")

    info = client.model_info()
    print(f"Model info: {info}")

    response = client.chat_completion(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        max_tokens=30,
        temperature=0.7,
    )
    print(f"Response: {response}")


if __name__ == "__main__":
    sys.exit(main())