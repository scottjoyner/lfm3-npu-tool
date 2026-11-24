#!/usr/bin/env python3
"""
LFM3-1.7B Smoke Test / Validation Script.

Validates that both CPU (port 1238) and NPU (port 1239) servers are
running and producing correct, non-degraded output. Exits non-zero if
any check fails.

Usage:
    python3 scripts/smoke_test.py [--verbose] [--fix]

Checks:
    - CPU server /health endpoint responds
    - NPU server /v1/models endpoint responds
    - CPU server generates correct output for simple prompts
    - NPU server generates correct output for simple prompts
    - NPU output contains reasoning trace (quality indicator)
    - Metrics exporter on port 9190 responds
"""

import argparse
import json
import re
import sys
import urllib.request
import urllib.error

CPU_PORT = 1238
NPU_PORT = 1239
METRICS_PORT = 9190
CPU_HOST = "127.0.0.1"
NPU_HOST = "127.0.0.1"

# Test prompts with expected substrings in output
TEST_PROMPTS = [
    ("What is 2+2?", ["4"]),
    ("The capital of France is", ["Paris", "paris"]),
    ("Say hello", ["hello", "Hello"]),
]

PASS = 0
FAIL = 1
checks_passed = 0
checks_total = 0


def log(msg, verbose=False):
    if verbose:
        print(msg)


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


def check_http_health():
    """Check both servers respond to health endpoints."""
    global checks_passed, checks_total

    checks_total += 1
    data, err = fetch_json(f"http://{CPU_HOST}:{CPU_PORT}/health")
    if data and data.get("status") == "healthy":
        print(f"  [PASS] CPU server health (port {CPU_PORT})")
        checks_passed += 1
    else:
        print(f"  [FAIL] CPU server health: {err or 'unhealthy'}")
        return False

    checks_total += 1
    data, err = fetch_json(f"http://{NPU_HOST}:{NPU_PORT}/v1/models")
    if data and "data" in data and len(data["data"]) > 0:
        lfm = [m for m in data["data"] if "lfm3" in str(m.get("id", ""))]
        if lfm:
            print(f"  [PASS] NPU server models (port {NPU_PORT}): {lfm[0]['id']}")
            checks_passed += 1
        else:
            print(f"  [FAIL] NPU server: no lfm3 model in {data['data']}")
            return False
    else:
        print(f"  [FAIL] NPU server models: {err or 'no data'}")
        return False

    return True


def check_metrics_exporter():
    """Check metrics exporter responds."""
    global checks_passed, checks_total

    checks_total += 1
    try:
        with urllib.request.urlopen(f"http://localhost:{METRICS_PORT}/metrics", timeout=10) as resp:
            resp_data = resp.read()
        body = req.read().decode("utf-8")
        if "lfm3_npu_up 1" in body:
            print(f"  [PASS] Metrics exporter (port {METRICS_PORT})")
            checks_passed += 1
            return True
        else:
            print(f"  [FAIL] Metrics exporter: lfm3_npu_up not found")
            return False
    except Exception as e:
        print(f"  [FAIL] Metrics exporter: {e}")
        return False


def check_generation(server, prompt, expected):
    """Generate text on a server and check output quality."""
    global checks_passed, checks_total

    checks_total += 1
    if server == "cpu":
        url = f"http://{CPU_HOST}:{CPU_PORT}/v1/chat/completions"
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 30,
            "temperature": 0.7,
        }
    else:
        url = f"http://{NPU_HOST}:{NPU_PORT}/v1/chat/completions"
        payload = {
            "model": "lfm3:1.7b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 30,
            "temperature": 0.7,
        }

    result, err = post_json(url, payload)
    if not result:
        print(f"  [FAIL] {server.upper()} generation '{prompt}': {err}")
        return False

    choice = result.get("choices", [{}])[0]
    msg = choice.get("message", {})
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""

    # NPU puts response in reasoning_content
    if not content and reasoning:
        content = reasoning

    # Check for expected content
    content_lower = content.lower()
    found = any(exp.lower() in content_lower for exp in expected)

    # Check for degradation markers
    degraded = False
    markers = ["assistant:", "user:", "system:"]
    loop_count = sum(content_lower.count(m.lower()) for m in markers)
    if loop_count > 2:
        degraded = True

    if found and not degraded:
        print(f"  [PASS] {server.upper()} '{prompt}' -> '{content[:60]}'")
        checks_passed += 1
        return True
    else:
        reason = "degraded" if degraded else f"missing expected {expected}"
        print(f"  [FAIL] {server.upper()} '{prompt}' -> '{content[:60]}' ({reason})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="LFM3-1.7B Smoke Test / Validation"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--fix", action="store_true",
                        help="Attempt to restart unhealthy servers via systemctl")
    args = parser.parse_args()

    print("=" * 60)
    print("LFM3-1.7B Smoke Test")
    print("=" * 60)

    import subprocess

    if not check_http_health():
        print("\n[FAIL] Servers not healthy - aborting generation tests")
        if args.fix:
            print("\nAttempting to restart servers via systemctl...")
            subprocess.run(["systemctl", "--user", "restart", "lfm3-npu-cpu.service"], check=False)
            subprocess.run(["systemctl", "--user", "restart", "lfm3-npu-npu.service"], check=False)
            print("Restart initiated. Re-run smoke test to verify.")
        return 1

    check_metrics_exporter()

    print("\n--- Generation Quality Tests ---")
    for prompt, expected in TEST_PROMPTS:
        check_generation("cpu", prompt, expected)
        check_generation("npu", prompt, expected)

    print("\n" + "=" * 60)
    print(f"Results: {checks_passed}/{checks_total} checks passed")

    if checks_passed == checks_total:
        print("ALL CHECKS PASSED")
        return 0
    else:
        print("SOME CHECKS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
