#!/usr/bin/env python3
"""
LFM3-1.7B Quality-Scored Benchmark.

Evaluates output quality on CPU (1238) and NPU (1239) servers using:
- Correctness checks for factual/arithmetic prompts
- Repetition detection (degradation symptom)
- Coherence scoring
- Reasoning trace presence (NPU advantage)

Usage:
    python3 quality_benchmark.py [--server cpu|npu|both]
"""

import json
import re
import sys
import time
import urllib.request
import urllib.error

CPU_HOST = "127.0.0.1"
CPU_PORT = 1238
NPU_HOST = "127.0.0.1"
NPU_PORT = 1239

# Test cases with expected quality markers
TEST_CASES = [
    {
        "prompt": "What is 2+2?",
        "expected_contains": ["4"],
        "expected_not_contains": ["5", "6", "8", "?"],
        "category": "arithmetic",
    },
    {
        "prompt": "The capital of France is",
        "expected_contains": ["Paris"],
        "expected_not_contains": ["London", "Berlin", "Tokyo"],
        "category": "factual",
    },
    {
        "prompt": "Write a haiku about coding",
        "expected_contains": None,
        "expected_not_contains": ["assistant:", "user:", "1.", "2.", "3."],
        "category": "creative",
    },
    {
        "prompt": "Explain quantum computing in one sentence",
        "expected_contains": ["quantum"],
        "expected_not_contains": ["assistant:", "user:", "1.0"],
        "category": "explanation",
    },
    {
        "prompt": "Count to 5",
        "expected_contains": ["1", "2", "3", "4", "5"],
        "expected_not_contains": ["6", "7", "8", "assistant:"],
        "category": "instruction",
    },
]


def http_post(url, payload, timeout=60):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def generate(server, prompt, max_tokens=50, temperature=0.7):
    if server == "cpu":
        url = f"http://{CPU_HOST}:{CPU_PORT}/v1/chat/completions"
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    else:
        url = f"http://{NPU_HOST}:{NPU_PORT}/v1/chat/completions"
        payload = {
            "model": "lfm3:1.7b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    result = http_post(url, payload)
    choice = result["choices"][0]
    msg = choice["message"]
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""
    usage = result.get("usage", {})
    return {
        "content": content,
        "reasoning": reasoning,
        "decode_tps": usage.get("decoding_speed_tps", 0),
        "ttft": usage.get("prefill_duration_ttft", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def score_repetition(text):
    """Score repetition: lower is better. Returns 0-100 (0=no repetition)."""
    if not text:
        return 100  # Empty output is bad
    words = text.split()
    if len(words) < 3:
        return 0
    # Count repeated bigrams
    bigrams = list(zip(words, words[1:]))
    if not bigrams:
        return 0
    bigram_counts = {}
    for bg in bigrams:
        bigram_counts[bg] = bigram_counts.get(bg, 0) + 1
    repeated = sum(1 for c in bigram_counts.values() if c > 1)
    ratio = repeated / len(bigrams)
    return min(100, int(ratio * 100))


def score_garbled(text):
    """Detect garbled output patterns. Returns 0-100 (0=clean)."""
    if not text:
        return 100
    score = 0
    # Conversation loop markers
    for marker in ["assistant:", "user:", "system:"]:
        count = text.lower().count(marker.lower())
        if count > 2:
            score += 30
    # Numbered list explosion
    if re.search(r'\d+\.\s*\d+\.\s*\d+', text):
        score += 25
    # Repeated single characters
    if re.search(r'(.)\1{5,}', text):
        score += 20
    # "Answer: 8" style wrong math
    if re.search(r'(?i)answer:\s*[5-9]', text):
        score += 30
    return min(100, score)


def check_quality(content, reasoning, test_case):
    """Score output quality 0-100 based on test case expectations."""
    # NPU server puts output in reasoning_content; combine both for checking
    full_text = content + " " + reasoning if reasoning else content
    score = 100
    details = []

    # Check expected contains
    if test_case["expected_contains"]:
        for expected in test_case["expected_contains"]:
            if expected.lower() in full_text.lower():
                details.append(f"✓ contains '{expected}'")
            else:
                score -= 20
                details.append(f"✗ missing '{expected}'")

    # Check expected NOT contains
    if test_case["expected_not_contains"]:
        for unexpected in test_case["expected_not_contains"]:
            if unexpected.lower() in full_text.lower():
                score -= 15
                details.append(f"✗ contains unwanted '{unexpected}'")
            else:
                details.append(f"✓ avoids '{unexpected}'")

    # Repetition penalty (on content, or full_text if content is empty)
    rep_text = content if content else full_text
    rep_score = score_repetition(rep_text)
    if rep_score > 30:
        score -= 15
        details.append(f"⚠ repetition score: {rep_score}/100")
    else:
        details.append(f"✓ repetition: {rep_score}/100")

    # Garbled penalty (on content, or full_text if content is empty)
    garbled_text = content if content else full_text
    garbled_score = score_garbled(garbled_text)
    if garbled_score > 20:
        score -= 20
        details.append(f"⚠ garbled score: {garbled_score}/100")
    else:
        details.append(f"✓ garbled: {garbled_score}/100")

    # Reasoning bonus (NPU)
    if reasoning:
        details.append(f"✓ reasoning trace present ({len(reasoning)} chars)")

    return max(0, min(100, score)), details


def run_benchmark(server):
    """Run quality benchmark on a single server."""
    print(f"\n{'=' * 70}")
    print(f"QUALITY BENCHMARK: {server.upper()} server")
    print(f"{'=' * 70}")

    # Check server reachable
    try:
        if server == "cpu":
            urllib.request.urlopen(f"http://{CPU_HOST}:{CPU_PORT}/health", timeout=5)
        else:
            urllib.request.urlopen(f"http://{NPU_HOST}:{NPU_PORT}/v1/models", timeout=5)
    except Exception as e:
        print(f"Server {server} not reachable: {e}")
        return None

    results = []
    total_score = 0
    total_decode = 0
    total_ttft = 0
    successful = 0

    for tc in TEST_CASES:
        try:
            result = generate(server, tc["prompt"], max_tokens=50)
            score, details = check_quality(
                result["content"], result["reasoning"], tc
            )
            total_score += score
            total_decode += result["decode_tps"] or 0
            total_ttft += result["ttft"] or 0
            successful += 1

            status = "PASS" if score >= 70 else "FAIL"
            print(f"\n[{status}] {tc['category']}: '{tc['prompt']}'")
            print(f"  Score: {score}/100")
            print(f"  Content: {result['content'][:120]}")
            if result["reasoning"]:
                print(f"  Reasoning: {result['reasoning'][:80]}...")
            print(f"  Decode: {result['decode_tps']:.1f} tok/s, TTFT: {result['ttft']:.3f}s")
            for d in details:
                print(f"    {d}")

            results.append({
                "prompt": tc["prompt"],
                "category": tc["category"],
                "score": score,
                "content": result["content"],
                "reasoning": result["reasoning"],
                "decode_tps": result["decode_tps"],
                "ttft": result["ttft"],
                "details": details,
            })
        except Exception as e:
            print(f"\n[ERROR] {tc['category']}: '{tc['prompt']}' - {e}")
            results.append({
                "prompt": tc["prompt"],
                "category": tc["category"],
                "score": 0,
                "error": str(e),
            })

    avg_score = total_score / successful if successful else 0
    avg_decode = total_decode / successful if successful else 0
    avg_ttft = total_ttft / successful if successful else 0

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {server.upper()}")
    print(f"  Tests passed: {successful}/{len(TEST_CASES)}")
    print(f"  Average quality score: {avg_score:.1f}/100")
    print(f"  Average decode: {avg_decode:.1f} tok/s")
    print(f"  Average TTFT: {avg_ttft:.3f}s")
    print(f"{'=' * 70}")

    return {
        "server": server,
        "avg_score": avg_score,
        "avg_decode": avg_decode,
        "avg_ttft": avg_ttft,
        "tests_passed": successful,
        "total_tests": len(TEST_CASES),
        "results": results,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="LFM3-1.7B Quality-Scored Benchmark"
    )
    parser.add_argument(
        "--server", choices=["cpu", "npu", "both"], default="both",
        help="Server(s) to benchmark (default: both)"
    )
    args = parser.parse_args()

    servers = ["cpu", "npu"] if args.server == "both" else [args.server]

    all_results = []
    for server in servers:
        result = run_benchmark(server)
        if result:
            all_results.append(result)

    # Comparison
    if len(all_results) == 2:
        cpu = all_results[0] if all_results[0]["server"] == "cpu" else all_results[1]
        npu = all_results[0] if all_results[0]["server"] == "npu" else all_results[1]

        print(f"\n{'=' * 70}")
        print("QUALITY COMPARISON: CPU vs NPU")
        print(f"{'=' * 70}")
        print(f"  CPU:    avg_score={cpu['avg_score']:.1f}/100, "
              f"decode={cpu['avg_decode']:.1f} tok/s, TTFT={cpu['avg_ttft']:.3f}s")
        print(f"  NPU:    avg_score={npu['avg_score']:.1f}/100, "
              f"decode={npu['avg_decode']:.1f} tok/s, TTFT={npu['avg_ttft']:.3f}s")
        print(f"  Quality gap: {npu['avg_score'] - cpu['avg_score']:+.1f} points "
              f"(NPU {'better' if npu['avg_score'] > cpu['avg_score'] else 'worse'})")
        if cpu['avg_decode'] > 0:
            print(f"  Speed ratio: {npu['avg_decode'] / cpu['avg_decode']:.1f}x "
                  f"({'faster' if npu['avg_decode'] > cpu['avg_decode'] else 'slower'})")
        else:
            print(f"  Speed ratio: N/A (CPU decode TPS not reported)")
        print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
