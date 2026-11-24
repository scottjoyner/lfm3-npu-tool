#!/usr/bin/env python3
"""
LFM3-1.7B Benchmark Cron Job.

Runs periodic benchmarks on both CPU and NPU servers and logs results.
Intended for systemd timer invocation (e.g., hourly).

Results are appended to var/log/benchmark-cron.log for trend analysis.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

CPU_PORT = 1238
NPU_PORT = 1239
CPU_HOST = "127.0.0.1"
NPU_HOST = "127.0.0.1"
LOG_FILE = "/media/scott/data/lfm3-npu-tool/var/log/benchmark-cron.log"


def log(msg):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    print(msg, flush=True)


def fetch_json(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            resp_data = resp.read()
        return json.loads(req.read().decode("utf-8")), None
    except Exception as e:
        return None, str(e)


def post_json(url, payload, timeout=30):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")), None
    except Exception as e:
        return None, str(e)


def benchmark_server(server, prompt="The capital of France is", max_tokens=30):
    """Run a single benchmark iteration on a server."""
    if server == "cpu":
        url = f"http://{CPU_HOST}:{CPU_PORT}/v1/chat/completions"
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
    else:
        url = f"http://{NPU_HOST}:{NPU_PORT}/v1/chat/completions"
        payload = {
            "model": "lfm3:1.7b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }

    start = time.perf_counter()
    result, err = post_json(url, payload)
    total = time.perf_counter() - start

    if not result:
        return {"success": False, "error": err, "total_time": total}

    usage = result.get("usage", {})
    return {
        "success": True,
        "total_time": total,
        "ttft": usage.get("prefill_duration_ttft"),
        "decode_tps": usage.get("decoding_speed_tps"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def main():
    log("=" * 60)
    log("LFM3-1.7B Benchmark Cron Run")
    log("=" * 60)

    all_ok = True

    for server in ["cpu", "npu"]:
        log(f"\n--- Benchmarking {server.upper()} server ---")

        # Check server reachable first
        health_path = "/health" if server == "cpu" else "/v1/models"
        data, err = fetch_json(
            f"http://{CPU_HOST if server == 'cpu' else NPU_HOST}:"
            f"{CPU_PORT if server == 'cpu' else NPU_PORT}{health_path}",
            timeout=5,
        )
        if not data:
            log(f"  {server.upper()} server not reachable: {err}")
            all_ok = False
            continue

        # Run 3 iterations
        results = []
        for i in range(3):
            r = benchmark_server(server)
            results.append(r)
            if r["success"]:
                log(f"  Iter {i+1}: {r['total_time']:.2f}s, "
                    f"decode={r.get('decode_tps') or 0:.1f} tok/s, "
                    f"ttft={r.get('ttft') or 0:.3f}s")
            else:
                log(f"  Iter {i+1}: FAILED - {r['error']}")

        successful = [r for r in results if r["success"]]
        if successful:
            avg_total = sum(r["total_time"] for r in successful) / len(successful)
            decodes = [r["decode_tps"] for r in successful if r.get("decode_tps")]
            ttfts = [r["ttft"] for r in successful if r.get("ttft")]
            log(f"  {server.upper()} summary: {len(successful)}/3 successful, "
                f"avg_total={avg_total:.2f}s")
            if decodes:
                log(f"  {server.upper()} avg decode: {sum(decodes)/len(decodes):.1f} tok/s")
            if ttfts:
                log(f"  {server.upper()} avg TTFT: {sum(ttfts)/len(ttfts):.3f}s")
        else:
            log(f"  {server.upper()} summary: ALL ITERATIONS FAILED")
            all_ok = False

    log("\nBenchmark cron run complete.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
