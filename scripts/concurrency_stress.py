#!/usr/bin/env python3
"""
LFM3-1.7B Concurrency Stress Test.

Sends N concurrent requests to CPU (1238) and/or NPU (1239) servers and
measures:
- Success rate
- Time to first token (TTFT)
- Decode speed (tok/s)
- Total time per request
- Output quality (degradation detection)

Usage:
    python3 concurrency_stress.py [--server cpu|npu|both] --concurrency 4 --requests 20
"""

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.request
import urllib.error

CPU_HOST = "127.0.0.1"
CPU_PORT = 1238
NPU_HOST = "127.0.0.1"
NPU_PORT = 1239

STRESS_PROMPTS = [
    "What is 2+2?",
    "The capital of France is",
    "Write a haiku about coding",
    "Explain quantum computing in one sentence",
    "Count to 10",
    "Tell me a joke about computers",
    "What is the tallest mountain in the world?",
    "Write a short poem about the ocean",
]


def http_post(url, payload, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def single_request(server, prompt, max_tokens=50, temperature=0.7, request_id=0):
    """Send a single request and return timing + quality metrics."""
    start = time.perf_counter()

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

    try:
        result = http_post(url, payload)
        total = time.perf_counter() - start
        choice = result["choices"][0]
        msg = choice["message"]
        content = msg.get("content", "") or ""
        reasoning = msg.get("reasoning_content", "") or ""
        usage = result.get("usage", {})

        # Quality: check for degradation markers
        has_garbled = any(m in content.lower() for m in ["assistant:", "user:"]) or \
                      any(m in reasoning.lower() for m in ["assistant:", "user:"]) if reasoning else False
        full_text = content + " " + reasoning if reasoning else content
        repeated_words = len(set(full_text.split())) / max(len(full_text.split()), 1) if full_text else 0

        return {
            "request_id": request_id,
            "server": server,
            "success": True,
            "total_time": total,
            "ttft": usage.get("prefill_duration_ttft"),
            "decode_tps": usage.get("decoding_speed_tps"),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "content_length": len(content),
            "reasoning_length": len(reasoning),
            "has_garbled": has_garbled,
            "vocab_diversity": repeated_words,
            "content": content[:200],
        }
    except Exception as e:
        total = time.perf_counter() - start
        return {
            "request_id": request_id,
            "server": server,
            "success": False,
            "total_time": total,
            "error": str(e),
        }


def run_stress(server, concurrency, total_requests):
    """Run concurrent stress test on a server."""
    print(f"\n{'=' * 70}")
    print(f"CONCURRENCY STRESS: {server.upper()} server")
    print(f"  Concurrency: {concurrency}, Total requests: {total_requests}")
    print(f"{'=' * 70}")

    # Check reachable
    try:
        if server == "cpu":
            urllib.request.urlopen(f"http://{CPU_HOST}:{CPU_PORT}/health", timeout=5)
        else:
            urllib.request.urlopen(f"http://{NPU_HOST}:{NPU_PORT}/v1/models", timeout=5)
    except Exception as e:
        print(f"Server {server} not reachable: {e}")
        return None

    results = []
    start_all = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(total_requests):
            prompt = STRESS_PROMPTS[i % len(STRESS_PROMPTS)]
            futures.append(
                executor.submit(single_request, server, prompt, 50, 0.7, i)
            )
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    total_elapsed = time.perf_counter() - start_all

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if successful:
        ttfts = [r["ttft"] for r in successful if r["ttft"] is not None]
        decodes = [r["decode_tps"] for r in successful if r["decode_tps"] is not None]
        total_times = [r["total_time"] for r in successful]
        garbled_count = sum(1 for r in successful if r["has_garbled"])

        print("\nResults:")
        print(f"  Successful:  {len(successful)}/{total_requests}")
        print(f"  Failed:      {len(failed)}/{total_requests}")
        print(f"  Total time:  {total_elapsed:.1f}s")
        print(f"  Throughput:  {len(successful) / total_elapsed:.1f} req/s")
        if ttfts:
            print(f"  TTFT:        avg={statistics.mean(ttfts):.3f}s, "
                  f"p95={statistics.quantiles(ttfts, n=20)[18] if len(ttfts) > 1 else ttfts[0]:.3f}s")
        if decodes:
            print(f"  Decode:      avg={statistics.mean(decodes):.1f} tok/s")
        if total_times:
            print(f"  Total/req:   avg={statistics.mean(total_times):.2f}s, "
                  f"max={max(total_times):.2f}s")
        print(f"  Garbled:     {garbled_count}/{len(successful)} requests")
        if decodes:
            print("  Vocab div:   N/A (per-request)")

    else:
        print(f"\nAll {total_requests} requests failed.")
        for r in failed[:5]:
            print(f"  Req {r['request_id']}: {r['error']}")

    return {
        "server": server,
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful": len(successful),
        "failed": len(failed),
        "total_elapsed": total_elapsed,
        "throughput_rps": len(successful) / total_elapsed if total_elapsed > 0 else 0,
        "avg_ttft": statistics.mean(ttfts) if ttfts else 0,
        "avg_decode": statistics.mean(decodes) if decodes else 0,
        "avg_total_time": statistics.mean(total_times) if total_times else 0,
        "garbled_count": garbled_count if successful else 0,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="LFM3-1.7B Concurrency Stress Test"
    )
    parser.add_argument(
        "--server", choices=["cpu", "npu", "both"], default="both",
        help="Server(s) to stress test (default: both)"
    )
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Concurrent requests (default: 4)")
    parser.add_argument("--requests", type=int, default=20,
                        help="Total requests (default: 20)")
    args = parser.parse_args()

    servers = ["cpu", "npu"] if args.server == "both" else [args.server]

    all_results = []
    for server in servers:
        result = run_stress(server, args.concurrency, args.requests)
        if result:
            all_results.append(result)

    # Comparison
    if len(all_results) == 2:
        cpu = all_results[0] if all_results[0]["server"] == "cpu" else all_results[1]
        npu = all_results[0] if all_results[0]["server"] == "npu" else all_results[1]

        print(f"\n{'=' * 70}")
        print("STRESS COMPARISON: CPU vs NPU")
        print(f"{'=' * 70}")
        print(f"  CPU:    success={cpu['successful']}/{cpu['total_requests']}, "
              f"throughput={cpu['throughput_rps']:.1f} req/s, "
              f"avg_decode={cpu['avg_decode']:.1f} tok/s, "
              f"garbled={cpu['garbled_count']}")
        print(f"  NPU:    success={npu['successful']}/{npu['total_requests']}, "
              f"throughput={npu['throughput_rps']:.1f} req/s, "
              f"avg_decode={npu['avg_decode']:.1f} tok/s, "
              f"garbled={npu['garbled_count']}")
        print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
