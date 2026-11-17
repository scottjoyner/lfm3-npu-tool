#!/usr/bin/env python3
"""
LFM3-1.7B Startup Validation Script.

Validates that all LFM3-1.7B services are running correctly after system boot.
Intended to be run via systemd timer @reboot or cron @reboot.

Exit codes:
    0 - All validations passed
    1 - One or more validations failed
"""

import json
import subprocess
import sys
import time
import urllib.request
import urllib.error

CPU_PORT = 1238
NPU_PORT = 1239
METRICS_PORT = 9190
CPU_HOST = "127.0.0.1"
NPU_HOST = "127.0.0.1"

failures = []


def log(msg):
    print(msg, flush=True)


def check(condition, message):
    if condition:
        log(f"  [PASS] {message}")
        return True
    else:
        log(f"  [FAIL] {message}")
        failures.append(message)
        return False


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
    print("LFM3-1.7B Startup Validation")
    print("=" * 60)

    # Wait for services to come up (max 60 seconds)
    log("Waiting for services to start...")
    for i in range(12):
        cpu_ok, _ = fetch_json(f"http://{CPU_HOST}:{CPU_PORT}/health", timeout=3)
        npu_ok, _ = fetch_json(f"http://{NPU_HOST}:{NPU_PORT}/v1/models", timeout=3)
        if cpu_ok and npu_ok:
            log("  All services are up")
            break
        log(f"  Waiting... ({i+1}/12)")
        time.sleep(5)
    else:
        log("  Services did not come up in time")

    # 1. Systemd services
    print("\n--- Systemd Services ---")
    result = subprocess.run(
        ["systemctl", "--user", "list-units", "lfm3-npu-*.service", "--state=running", "--no-legend"],
        capture_output=True, text=True, timeout=10
    )
    active = [line.split()[0] for line in result.stdout.strip().splitlines() if line.strip()]
    check("lfm3-npu-cpu.service" in active, "CPU service active")
    check("lfm3-npu-npu.service" in active, "NPU service active")
    check("lfm3-npu-metrics.service" in active, "Metrics service active")
    check("lfm3-npu-quality-monitor.service" in active, "Quality monitor service active")

    # 2. Systemd timers
    print("\n--- Systemd Timers ---")
    result = subprocess.run(
        ["systemctl", "--user", "list-timers", "lfm3-npu-*", "--no-legend", "--all"],
        capture_output=True, text=True, timeout=10
    )
    timers = result.stdout.strip().splitlines()
    check(any("health.timer" in t for t in timers), "Health timer active")
    check(any("watchdog.timer" in t for t in timers), "Watchdog timer active")
    check(any("doctor.timer" in t for t in timers), "Doctor timer active")
    check(any("quality-monitor.timer" in t for t in timers), "Quality monitor timer active")
    check(any("benchmark.timer" in t for t in timers), "Benchmark timer active")

    # 3. Server health
    print("\n--- Server Health ---")
    data, err = fetch_json(f"http://{CPU_HOST}:{CPU_PORT}/health")
    check(data and data.get("status") == "healthy", f"CPU server healthy (err: {err})")

    data, err = fetch_json(f"http://{NPU_HOST}:{NPU_PORT}/v1/models")
    check(data and "data" in data and len(data["data"]) > 0, f"NPU server reachable (err: {err})")
    if data and "data" in data:
        lfm = [m for m in data["data"] if "lfm3" in str(m.get("id", ""))]
        check(len(lfm) > 0, "NPU lfm3:1.7b model available")

    # 4. Metrics exporter
    print("\n--- Metrics Exporter ---")
    try:
        req = urllib.request.urlopen(f"http://localhost:{METRICS_PORT}/metrics", timeout=10)
        body = req.read().decode("utf-8")
        check("lfm3_npu_up 1" in body, "NPU up metric present")
        check("lfm3_cpu_up 1" in body, "CPU up metric present")
        check("lfm3_disk_used_percent" in body, "Disk metrics present")
    except Exception as e:
        check(False, f"Metrics exporter: {e}")

    # 5. NPU generation quality
    print("\n--- NPU Generation Quality ---")
    result, err = post_json(
        f"http://{NPU_HOST}:{NPU_PORT}/v1/chat/completions",
        {"model": "lfm3:1.7b", "messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 50, "temperature": 0.7},
    )
    if result:
        choice = result.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "") or ""
        reasoning = msg.get("reasoning_content", "") or ""
        full_text = content + " " + reasoning
        check("4" in full_text, f"NPU answers 2+2 correctly (got: '{full_text[:100]}')")
        check(len(reasoning) > 0, "NPU produces reasoning trace")
    else:
        check(False, f"NPU generation: {err}")

    # 6. Disk space
    print("\n--- Disk Space ---")
    import shutil
    for path, label in [("/", "root"), ("/media/scott/data", "data"), ("/tmp", "tmp")]:
        try:
            stat = shutil.disk_usage(path)
            free_gb = stat.free / (1024**3)
            check(free_gb > 1, f"{label} ({path}): {free_gb:.1f}GB free")
        except Exception as e:
            check(False, f"{label} ({path}): {e}")

    # Summary
    print("\n" + "=" * 60)
    if failures:
        print(f"STARTUP VALIDATION FAILED: {len(failures)} failure(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print("STARTUP VALIDATION PASSED: All systems operational")
        return 0


if __name__ == "__main__":
    sys.exit(main())
