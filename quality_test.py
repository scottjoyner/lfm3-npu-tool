#!/usr/bin/env python3
"""
NPU Quality Test - Generate text with NPU server to assess quality degradation.
"""

import json
import urllib.request

NPU_URL = "http://localhost:1239"
CPU_URL = "http://localhost:1238"


def generate(url, prompt, max_tokens=40, temperature=0.7):
    payload = json.dumps({
        "model": "lfm3-1.7b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read())
    choices = data.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "")
    return "NO RESPONSE"


PROMPTS = [
    "The quick brown fox",
    "Once upon a time",
    "Write a short story about a robot",
    "Explain quantum computing",
    "What is the capital of France?",
]

print("=" * 60)
print("NPU Server Quality Test (center-tap conv replacement)")
print("=" * 60)

for prompt in PROMPTS:
    print(f"\nPrompt: '{prompt}'")
    try:
        result = generate(NPU_URL, prompt, max_tokens=40)
        # Check for repetition (simple heuristic)
        words = result.split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            print(f"  NPU: {result[:100]}... (unique ratio: {unique_ratio:.2f})")
        else:
            print(f"  NPU: {result}")
    except Exception as e:
        print(f"  NPU ERROR: {e}")

print("\n" + "=" * 60)
print("CPU Server Quality Test (full conv - reference)")
print("=" * 60)

for prompt in PROMPTS:
    print(f"\nPrompt: '{prompt}'")
    try:
        result = generate(CPU_URL, prompt, max_tokens=40)
        words = result.split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            print(f"  CPU: {result[:100]}... (unique ratio: {unique_ratio:.2f})")
        else:
            print(f"  CPU: {result}")
    except Exception as e:
        print(f"  CPU ERROR: {e}")
