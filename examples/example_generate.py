#!/usr/bin/env python3
"""
Example: Generate text using LFM3-1.7B NPU inference engine directly.

This demonstrates using the Python API instead of the CLI.
"""

import sys
import os

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SERVER_DIR, "lib"))

from lfm3_npu import LFM3NPUInference, load_config


def main():
    config = load_config()

    engine = LFM3NPUInference(
        model_path=config.get("model.path"),
        tokenizer_path=config.get("model.tokenizer"),
        config=config,
    )
    engine.load()
    engine.warmup()

    prompts = [
        "The quick brown fox",
        "Once upon a time",
        "The capital of France is",
        "Hello, how are you?",
    ]

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        result = engine.generate(
            prompt,
            max_new_tokens=20,
            temperature=0.7,
            top_p=0.9,
        )
        print(f"Generated: '{result}'")
        print(f"Full: '{prompt}{result}'")


if __name__ == "__main__":
    main()