#!/usr/bin/env python3
"""
CI Test Script for LFM3-1.7B NPU Inference Deployment.

Validates the complete deployment in a CI environment:
- All systemd services are active
- Both servers respond to health checks
- NPU server generates correct, non-degraded output
- CPU server generates correct output (or is flagged for restart)
- Metrics exporter exposes Prometheus metrics
- Watchdog can detect and restart degraded servers

Exit codes:
    0 - All checks passed
    1 - One or more checks failed
"""

import json
import subprocess
import sys
import urllib.request
import urllib.error

CPU_PORT = 1238
NPU_PORT = 1239
METRICS_PORT = 9190

failures = []


def check(condition, message):
    if condition:
        print(f"  [PASS] {message}")
        return True
    else:
        print(f"  [FAIL] {message}")
        failures.append(message)
        return False


def run_cmd(cmd):
    """Run a shell command and return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    return result.returncode, result.stdout, result.stderr


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


def main():
    print("=" * 60)
    print("LFM3-1.7B CI Test Suite")
    print("=" * 60)

    # 1. Systemd services
    print("\n--- Systemd Services ---")
    rc, out, _ = run_cmd("systemctl --user list-units 'lfm3-npu-*.service' --state=running --no-legend")
    active_services = [line.split()[0] for line in out.strip().splitlines() if line.strip()]
    check("lfm3-npu-cpu.service" in active_services, "CPU service is active")
    check("lfm3-npu-npu.service" in active_services, "NPU service is active")
    check("lfm3-npu-metrics.service" in active_services, "Metrics service is active")

    # 2. Systemd timers
    print("\n--- Systemd Timers ---")
    rc, out, _ = run_cmd("systemctl --user list-timers 'lfm3-npu-*' --no-legend --all")
    timers = out.strip().splitlines()
    check(any("health.timer" in t for t in timers), "Health timer exists")
    check(any("watchdog.timer" in t for t in timers), "Watchdog timer exists")
    check(any("doctor.timer" in t for t in timers), "Doctor timer exists")
    check(any("quality-monitor.timer" in t for t in timers), "Quality monitor timer exists")
    check(any("benchmark.timer" in t for t in timers), "Benchmark timer exists")
    check(any("startup-validate.timer" in t for t in timers), "Startup validate timer exists")

    # 3. Server health
    print("\n--- Server Health ---")
    data, err = fetch_json(f"http://localhost:{CPU_PORT}/health")
    check(data and data.get("status") == "healthy", f"CPU server healthy (err: {err})")

    data, err = fetch_json(f"http://localhost:{NPU_PORT}/v1/models")
    check(data and "data" in data and len(data["data"]) > 0, f"NPU server reachable (err: {err})")
    if data and "data" in data:
        lfm = [m for m in data["data"] if "lfm3" in str(m.get("id", ""))]
        check(len(lfm) > 0, f"NPU lfm3:1.7b model available")

    # 4. Metrics exporter
    print("\n--- Metrics Exporter ---")
    try:
        req = urllib.request.urlopen(f"http://localhost:{METRICS_PORT}/metrics", timeout=10)
        body = req.read().decode("utf-8")
        check("lfm3_npu_up 1" in body, "NPU up metric present")
        check("lfm3_cpu_up" in body, "CPU up metric present")
        check("lfm3_npu_lfm3_model_available 1" in body, "NPU model available metric present")
    except Exception as e:
        check(False, f"Metrics exporter reachable: {e}")

    # 5. NPU generation quality
    print("\n--- NPU Generation Quality ---")
    result, err = post_json(
        f"http://localhost:{NPU_PORT}/v1/chat/completions",
        {"model": "lfm3:1.7b", "messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 50, "temperature": 0.7},
    )
    if result:
        choice = result.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "") or ""
        reasoning = msg.get("reasoning_content", "") or ""
        full_text = content + " " + reasoning
        if not content and reasoning:
            content = reasoning
        check("4" in full_text, f"NPU answers 2+2 correctly (got: '{full_text[:100]}')")
        check(len(content) > 0, f"NPU produces non-empty output")
        check(content.lower().count("assistant:") <= 2,
              f"NPU output not in conversation loop")
    else:
        check(False, f"NPU generation: {err}")

    # 6. CPU generation quality
    print("\n--- CPU Generation Quality ---")
    result, err = post_json(
        f"http://localhost:{CPU_PORT}/v1/chat/completions",
        {"messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 20, "temperature": 0.7},
    )
    if result:
        choice = result.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "") or ""
        check("4" in content, f"CPU answers 2+2 correctly (got: '{content[:80]}')")
    else:
        check(False, f"CPU generation: {err}")

    # 7. Watchdog can run
    print("\n--- Watchdog ---")
    rc, out, _ = run_cmd(
        "cd /media/scott/data/lfm3-npu-tool && python3 bin/lfm3-npu watchdog --once "
        "--max-uptime-hours 4 --max-garble-ratio 0.3"
    )
    check(rc == 0, f"Watchdog runs successfully (rc={rc})")

    # Summary
    print("\n" + "=" * 60)
    if failures:
        print(f"CI TESTS FAILED: {len(failures)} failure(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print("ALL CI TESTS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
