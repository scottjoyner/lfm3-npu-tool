#!/usr/bin/env python3
"""
LFM3-1.7B Quality Monitor.

Monitors inference output quality on both CPU and NPU servers by running
smoke test generation checks. Automatically restarts the CPU server when
output quality degrades (known ONNX Runtime issue after ~20 min uptime).

Usage:
    python3 scripts/quality_monitor.py [--interval 300] [--once] [--fix]

Checks every interval:
    - CPU server generates correct, non-degraded output
    - NPU server generates correct, non-degraded output with reasoning traces
    - Auto-restarts CPU server if quality falls below threshold

Exit codes:
    0 - All quality checks passed
    1 - Quality degradation detected (or --once with failures)
"""

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error

CPU_PORT = 1238
NPU_PORT = 1239
CPU_HOST = "127.0.0.1"
NPU_HOST = "127.0.0.1"

# Test prompts with expected substrings
TEST_PROMPTS = [
    ("What is 2+2?", ["4"]),
    ("Say hello", ["hello", "Hello"]),
    ("The capital of France is", ["Paris", "paris"]),
]

QUALITY_THRESHOLD = 0.7  # Minimum fraction of checks that must pass


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def fetch_json(url, timeout=10):
    try:
        req = urllib.request.urlopen(url, timeout=timeout)
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


def check_generation(server, prompt, expected, verbose=False):
    """Generate text and check quality. Returns (passed, content, reason)."""
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
        return False, "", f"request failed: {err}"

    choice = result.get("choices", [{}])[0]
    msg = choice.get("message", {})
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""

    # NPU puts response in reasoning_content
    if not content and reasoning:
        content = reasoning

    full_text = content + " " + reasoning

    # Check for expected content
    content_lower = full_text.lower()
    found = any(exp.lower() in content_lower for exp in expected)

    # Check for degradation markers
    degraded = False
    reasons = []

    loop_markers = ["assistant:", "user:", "system:"]
    loop_count = sum(content_lower.count(m.lower()) for m in loop_markers)
    if loop_count > 2:
        degraded = True
        reasons.append(f"conversation loop ({loop_count} markers)")

    if re.search(r'\d+\.\s*\d+\.\s*\d+', full_text):
        degraded = True
        reasons.append("numbered list explosion")

    if re.search(r'(.)\1{5,}', full_text):
        degraded = True
        reasons.append("repeated characters")

    # Wrong math pattern: answer is 5-9 for 2+2
    if "2+2" in prompt.lower() or "what is 2" in prompt.lower():
        if re.search(r'(?i)(?:answer|is|equals?):\s*[5-9]', full_text):
            degraded = True
            reasons.append("wrong math")

    # NPU-specific: should produce reasoning traces (Qwen XML format)
    # Absence of reasoning_content indicates quality degradation
    if server == "npu" and not reasoning and not content:
        degraded = True
        reasons.append("no reasoning trace (NPU quality degraded)")

    if verbose:
        log(f"  {server.upper()} '{prompt}' -> '{content[:60]}' "
            f"{'(degraded: ' + ', '.join(reasons) + ')' if degraded else ''}")

    if found and not degraded:
        return True, content, ""
    else:
        reason = "; ".join(reasons) if reasons else f"missing expected {expected}"
        return False, content, reason


def run_quality_check(verbose=False):
    """Run all quality checks. Returns (passed, total, cpu_passed, npu_passed)."""
    results = {"cpu": [], "npu": []}

    for prompt, expected in TEST_PROMPTS:
        cpu_passed, _, cpu_reason = check_generation("cpu", prompt, expected, verbose)
        results["cpu"].append((cpu_passed, cpu_reason))

        npu_passed, _, npu_reason = check_generation("npu", prompt, expected, verbose)
        results["npu"].append((npu_passed, npu_reason))

    cpu_passed_count = sum(1 for p, _ in results["cpu"] if p)
    npu_passed_count = sum(1 for p, _ in results["npu"] if p)
    total = len(TEST_PROMPTS)

    return cpu_passed_count, npu_passed_count, total, results


def restart_cpu():
    """Restart the CPU server via systemctl."""
    log("Restarting CPU server due to quality degradation...")
    result = subprocess.run(
        ["systemctl", "--user", "restart", "lfm3-npu-cpu.service"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0:
        log("CPU server restarted successfully.")
        return True
    else:
        log(f"Failed to restart CPU server: {result.stderr}")
        return False


def restart_npu():
    """Restart the NPU server via systemctl."""
    log("Restarting NPU server due to quality degradation (no reasoning trace)...")
    result = subprocess.run(
        ["systemctl", "--user", "restart", "lfm3-npu-npu.service"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode == 0:
        log("NPU server restarted successfully.")
        return True
    else:
        log(f"Failed to restart NPU server: {result.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="LFM3-1.7B Quality Monitor"
    )
    parser.add_argument("--interval", type=int, default=300,
                        help="Check interval in seconds (default: 300)")
    parser.add_argument("--once", action="store_true",
                        help="Run one check and exit")
    parser.add_argument("--fix", action="store_true",
                        help="Auto-restart CPU server on quality degradation")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--restart-cooldown", type=int, default=120,
                        help="Minimum seconds between CPU restarts (default: 120)")
    args = parser.parse_args()

    log("=== LFM3-1.7B Quality Monitor ===")
    log(f"Interval: {args.interval}s, Fix: {args.fix}, Verbose: {args.verbose}")
    log(f"Restart cooldown: {args.restart_cooldown}s")

    last_cpu_restart_time = 0
    last_npu_restart_time = 0

    while True:
        cpu_passed, npu_passed, total, results = run_quality_check(args.verbose)

        cpu_ok = cpu_passed == total
        npu_ok = npu_passed == total

        log(f"Quality: CPU {cpu_passed}/{total}, NPU {npu_passed}/{total}")

        # Report failures
        for i, (passed, reason) in enumerate(results["cpu"]):
            if not passed:
                log(f"  CPU FAIL '{TEST_PROMPTS[i][0]}': {reason}")
        for i, (passed, reason) in enumerate(results["npu"]):
            if not passed:
                log(f"  NPU FAIL '{TEST_PROMPTS[i][0]}': {reason}")

        # Auto-restart CPU on degradation (with cooldown)
        if not cpu_ok and args.fix:
            now = time.time()
            if now - last_cpu_restart_time >= args.restart_cooldown:
                log("CPU quality degraded - triggering restart...")
                if restart_cpu():
                    last_cpu_restart_time = now
                    log("Waiting 15s for CPU server to stabilize...")
                    time.sleep(15)
                else:
                    log("CPU restart failed - will retry on next cycle.")
            else:
                remaining = int(args.restart_cooldown - (now - last_cpu_restart_time))
                log(f"CPU quality degraded but restart in cooldown ({remaining}s remaining)")

        # Auto-restart NPU on reasoning trace loss (with cooldown)
        if not npu_ok and args.fix:
            now = time.time()
            if now - last_npu_restart_time >= args.restart_cooldown:
                log("NPU quality degraded (no reasoning trace) - triggering restart...")
                if restart_npu():
                    last_npu_restart_time = now
                    log("Waiting 15s for NPU server to stabilize...")
                    time.sleep(15)
                else:
                    log("NPU restart failed - will retry on next cycle.")
            else:
                remaining = int(args.restart_cooldown - (now - last_npu_restart_time))
                log(f"NPU quality degraded but restart in cooldown ({remaining}s remaining)")

        if args.once:
            if cpu_ok and npu_ok:
                log("All quality checks passed.")
                return 0
            else:
                log("Quality degradation detected.")
                return 1

        log(f"Next check in {args.interval}s...")
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
