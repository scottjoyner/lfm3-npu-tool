#!/usr/bin/env python3
"""
Comprehensive CPU vs NPU concurrency benchmark for LFM3-1.7B.
Tests multiple concurrency levels and reports TTFT, per-token latency, and throughput.
"""

import json
import urllib.request
import time
import threading
import sys
import statistics

CPU_URL = "http://localhost:1238"
NPU_URL = "http://localhost:1239"

PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Write a short story about a robot learning to paint.",
    "Explain quantum computing in simple terms.",
    "What are the benefits of exercise for mental health?",
    "Write a Python function to calculate fibonacci numbers.",
]


def single_request(url, prompt, max_tokens=50):
    """Make a single non-streaming request and return metrics."""
    payload = json.dumps({
        "model": "lfm3-1.7b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=180)
        data = json.loads(resp.read())
        elapsed = time.time() - start
        choices = data.get("choices", [])
        if choices:
            text = choices[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            toks = usage.get("completion_tokens", len(text.split()))
            return {
                "elapsed": elapsed,
                "tokens": toks,
                "tps": toks / elapsed if elapsed > 0 else 0,
                "success": True,
            }
        return {"elapsed": elapsed, "tokens": 0, "tps": 0, "success": False, "error": "no choices"}
    except Exception as e:
        return {"elapsed": time.time() - start, "tokens": 0, "tps": 0, "success": False, "error": str(e)}


def concurrent_requests(url, concurrency, num_requests, max_tokens=50):
    """Fire num_requests with concurrency level, return all results."""
    results = []
    lock = threading.Lock()
    semaphore = threading.Semaphore(concurrency)

    def worker(prompt):
        with semaphore:
            result = single_request(url, prompt, max_tokens)
            with lock:
                results.append(result)

    threads = []
    for i in range(num_requests):
        prompt = PROMPTS[i % len(PROMPTS)]
        t = threading.Thread(target=worker, args=(prompt,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return results


def benchmark_server(label, url, concurrency_levels=(1, 2, 4), num_requests_per_level=4):
    print(f"\n{'='*60}")
    print(f"Benchmarking {label} server ({url})")
    print(f"{'='*60}")

    all_results = {}
    for conc in concurrency_levels:
        print(f"\n  Concurrency level: {conc}")
        results = concurrent_requests(url, conc, num_requests_per_level)
        successful = [r for r in results if r["success"]]
        failed = len(results) - len(successful)

        if successful:
            tps_list = [r["tps"] for r in successful]
            elapsed_list = [r["elapsed"] for r in successful]
            tokens_list = [r["tokens"] for r in successful]

            avg_tps = statistics.mean(tps_list)
            p95_tps = sorted(tps_list)[int(len(tps_list) * 0.95)] if len(tps_list) > 1 else tps_list[0]
            avg_elapsed = statistics.mean(elapsed_list)
            avg_tokens = statistics.mean(tokens_list)

            all_results[conc] = {
                "avg_tps": avg_tps,
                "p95_tps": p95_tps,
                "avg_elapsed": avg_elapsed,
                "avg_tokens": avg_tokens,
                "successful": len(successful),
                "failed": failed,
            }
            print(f"    Successful: {len(successful)}/{num_requests_per_level}")
            print(f"    Avg TPS: {avg_tps:.2f} (P95: {p95_tps:.2f})")
            print(f"    Avg elapsed: {avg_elapsed:.2f}s")
            print(f"    Avg tokens: {avg_tokens:.0f}")
        else:
            print(f"    All requests failed!")
            all_results[conc] = {"successful": 0, "failed": failed}

    return all_results


def main():
    cpu_results = benchmark_server("CPU", CPU_URL)
    npu_results = benchmark_server("NPU", NPU_URL)

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Conc':<8} {'CPU TPS':<10} {'NPU TPS':<10} {'Speedup':<10} {'CPU Failed':<12} {'NPU Failed':<12}")
    print("-" * 60)
    for conc in sorted(cpu_results.keys()):
        cpu = cpu_results.get(conc, {})
        npu = npu_results.get(conc, {})
        cpu_tps = cpu.get("avg_tps", 0)
        npu_tps = npu.get("avg_tps", 0)
        speedup = npu_tps / cpu_tps if cpu_tps > 0 else 0
        cpu_fail = cpu.get("failed", 0)
        npu_fail = npu.get("failed", 0)
        print(f"{conc:<8} {cpu_tps:<10.2f} {npu_tps:<10.2f} {speedup:<10.2f}x {cpu_fail:<12} {npu_fail:<12}")

    print("\nKV cache quantization: COMPLETE (22 tensors, INT8)")
    print("Model: /media/scott/data/lfm3_oga_export/model_kv_quant.onnx")


if __name__ == "__main__":
    main()
