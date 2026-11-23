#!/usr/bin/env python3
"""
Integration tests against running LFM3-1.7B servers.
Tests enterprise module integration with live HTTP endpoints.
"""

import sys
import os
import json
import urllib.request
import urllib.error

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SERVER_DIR, "lib"))

CPU_PORT = 1238
NPU_PORT = 1239

failures = []


def fetch_json(url, method="GET", data=None, timeout=10):
    """Fetch JSON from a URL. Returns (data, error)."""
    try:
        if data:
            body = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json"},
                method=method,
            )
        else:
            req = urllib.request.Request(url, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")), None
    except Exception as e:
        return None, str(e)


def check(condition, message):
    if condition:
        print(f"  [PASS] {message}")
        return True
    else:
        print(f"  [FAIL] {message}")
        failures.append(message)
        return False


def test_cpu_health():
    """Test CPU server /health endpoint includes enterprise module data."""
    print("\n--- CPU Server /health ---")
    data, err = fetch_json(f"http://localhost:{CPU_PORT}/health")
    if err:
        check(False, f"CPU /health reachable: {err}")
        return
    check(data is not None, "CPU /health returns data")
    if data:
        check(data.get("status") == "healthy", "CPU server status is healthy")
        check("autoscaler" in data, "Health includes autoscaler info")
        check("disaster_recovery" in data, "Health includes DR info")
        check("ab_testing" in data, "Health includes A/B testing info")
        if "autoscaler" in data:
            check("current_replicas" in data["autoscaler"],
                  "Autoscaler has current_replicas")
        if "disaster_recovery" in data:
            check("primary" in data["disaster_recovery"],
                  "DR has primary field")


def test_cpu_models():
    """Test CPU server /v1/models endpoint."""
    print("\n--- CPU Server /v1/models ---")
    data, err = fetch_json(f"http://localhost:{CPU_PORT}/v1/models")
    if err:
        check(False, f"CPU /v1/models reachable: {err}")
        return
    check(data is not None, "CPU /v1/models returns data")
    if data and "data" in data:
        check(len(data["data"]) > 0, "CPU has at least one model")


def test_cpu_generate():
    """Test CPU server generation endpoint."""
    print("\n--- CPU Server Generation ---")
    data, err = fetch_json(
        f"http://localhost:{CPU_PORT}/v1/chat/completions",
        method="POST",
        data={
            "model": "lfm3-1.7b",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 30,
            "temperature": 0.7,
        },
        timeout=30,
    )
    if err:
        check(False, f"CPU generation reachable: {err}")
        return
    if data:
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "") or ""
        check(len(content) > 0, f"CPU produces non-empty output (got: '{content[:80]}')")
        check("4" in content, f"CPU answers 2+2 correctly (got: '{content[:80]}')")


def test_enterprise_modules_import():
    """Test all enterprise modules can be imported together."""
    print("\n--- Enterprise Module Imports ---")
    try:
        from lfm3_npu import (
            get_auto_scaler, AutoScaler, ScalingPolicy,
            get_ab_test_manager, ABTestManager,
            get_dr_manager, DisasterRecoveryManager,
            get_promotion_pipeline, PromotionPipeline,
            EnterpriseMetricsCollector,
            validate_config, ConfigSchema,
        )
        check(True, "All enterprise modules importable")
    except Exception as e:
        check(False, f"Enterprise module import failed: {e}")


def test_enterprise_metrics():
    """Test EnterpriseMetricsCollector produces valid Prometheus output."""
    print("\n--- Enterprise Metrics ---")
    try:
        from lfm3_npu import EnterpriseMetricsCollector
        emc = EnterpriseMetricsCollector()
        emc.record_autoscaler_replicas(3)
        emc.record_autoscaler_scale_up()
        emc.record_dr_region_healthy("us-east-1", True)
        emc.record_promotion_request("staging", "canary", "pending")
        emc.record_cost_total("tenant-1", "standard", 15.50)
        emc.record_key_rotation()
        emc.record_security_finding("high", 2)
        metrics = emc.get_metrics().decode()
        check("lfm3_autoscaler_current_replicas" in metrics,
              "Metrics include autoscaler replicas")
        check("lfm3_dr_region_healthy" in metrics,
              "Metrics include DR region health")
        check("lfm3_promotion_requests_total" in metrics,
              "Metrics include promotion requests")
        check("lfm3_cost_total_usd" in metrics,
              "Metrics include cost tracking")
        check("lfm3_key_rotation_total" in metrics,
              "Metrics include key rotation")
        check("lfm3_security_findings_total" in metrics,
              "Metrics include security findings")
    except Exception as e:
        check(False, f"Enterprise metrics test failed: {e}")


def test_config_validate_with_new_sections():
    """Test config-validate accepts new module sections."""
    print("\n--- Config Validate with New Sections ---")
    try:
        from lfm3_npu import load_config, validate_config
        import copy
        config = load_config()
        validated = validate_config(copy.deepcopy(config._data))
        check("model_promotion" in validated,
              "Validated config includes model_promotion")
        check("disaster_recovery" in validated,
              "Validated config includes disaster_recovery")
        check("auto_scaling" in validated,
              "Validated config includes auto_scaling")
        check("ab_testing" in validated,
              "Validated config includes ab_testing")
        check("cost_tracking" in validated,
              "Validated config includes cost_tracking")
        check("alerting" in validated,
              "Validated config includes alerting")
        check("security_scan" in validated,
              "Validated config includes security_scan")
        check("key_rotation" in validated,
              "Validated config includes key_rotation")
    except Exception as e:
        check(False, f"Config validate test failed: {e}")


def main():
    print("=" * 60)
    print("LFM3-1.7B Enterprise Module Integration Tests")
    print("=" * 60)

    tests = [
        test_enterprise_modules_import,
        test_enterprise_metrics,
        test_config_validate_with_new_sections,
        test_cpu_health,
        test_cpu_models,
        test_cpu_generate,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failures.append(test.__name__)
            failed += 1

    print("\n" + "=" * 60)
    if failures:
        print(f"INTEGRATION TESTS FAILED: {len(failures)} failure(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print("ALL INTEGRATION TESTS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
