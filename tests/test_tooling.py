#!/usr/bin/env python3
"""
Test suite for LFM3-1.7B NPU tooling.
"""

import sys
import os

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SERVER_DIR, "lib"))


def test_config_loading():
    """Test that configuration loads correctly."""
    from lfm3_npu import load_config, Config

    config = load_config()
    assert isinstance(config, Config), "Config should be a Config instance"
    assert config.get("model.path") is not None, "model.path should be set"
    assert config.get("npu.provider") == "VitisAIExecutionProvider"
    assert config.get("server.cpu.port") == 1238
    assert config.get("server.npu.port") == 1239
    print("  ✓ Config loading works")


def test_module_imports():
    """Test that all modules can be imported."""
    from lfm3_npu import (
        Config,
        LFM3NPUInference,
        ServerManager,
        bench_model,
        load_config,
        convert_model,
    )
    print("  ✓ All module imports work")


def test_converter():
    """Test that the converter module exists and has expected functions."""
    from lfm3_npu.converter import convert, convert_centertap, convert_bypass

    assert callable(convert)
    assert callable(convert_centertap)
    assert callable(convert_bypass)
    print("  ✓ Converter module works")


def test_benchmark():
    """Test that the benchmark module has expected functions."""
    from lfm3_npu.benchmark import bench_model
    assert callable(bench_model)
    print("  ✓ Benchmark module works")


def test_server_manager():
    """Test that ServerManager can be instantiated."""
    from lfm3_npu import ServerManager, load_config

    config = load_config()
    manager = ServerManager(config)
    assert manager is not None
    print("  ✓ ServerManager instantiation works")


def test_smoke_test_script_exists():
    """Test that smoke test script exists and is executable."""
    import os
    script = os.path.join(SERVER_DIR, "scripts", "smoke_test.py")
    assert os.path.exists(script), "smoke_test.py should exist"
    assert os.access(script, os.X_OK), "smoke_test.py should be executable"
    print("  ✓ Smoke test script exists and is executable")


def test_ci_test_script_exists():
    """Test that CI test script exists and is executable."""
    import os
    script = os.path.join(SERVER_DIR, "scripts", "ci_test.py")
    assert os.path.exists(script), "ci_test.py should exist"
    assert os.access(script, os.X_OK), "ci_test.py should be executable"
    print("  ✓ CI test script exists and is executable")


def test_quality_monitor_script_exists():
    """Test that quality monitor script exists and is executable."""
    import os
    script = os.path.join(SERVER_DIR, "scripts", "quality_monitor.py")
    assert os.path.exists(script), "quality_monitor.py should exist"
    assert os.access(script, os.X_OK), "quality_monitor.py should be executable"
    print("  ✓ Quality monitor script exists and is executable")


def test_deploy_script_exists():
    """Test that deploy script exists and is executable."""
    import os
    script = os.path.join(SERVER_DIR, "deploy.sh")
    assert os.path.exists(script), "deploy.sh should exist"
    assert os.access(script, os.X_OK), "deploy.sh should be executable"
    print("  ✓ Deploy script exists and is executable")


def test_grafana_dashboard_valid_json():
    """Test that Grafana dashboard JSON is valid."""
    import json
    dashboard_path = os.path.join(SERVER_DIR, "etc", "grafana-dashboard-lfm3-npu.json")
    assert os.path.exists(dashboard_path), "Grafana dashboard should exist"
    with open(dashboard_path) as f:
        d = json.load(f)
    assert "dashboard" in d, "Dashboard should have 'dashboard' key"
    assert "panels" in d["dashboard"], "Dashboard should have panels"
    assert len(d["dashboard"]["panels"]) >= 13, "Should have at least 13 panels"
    print(f"  ✓ Grafana dashboard JSON is valid ({len(d['dashboard']['panels'])} panels)")


def test_prometheus_rules_valid_yaml():
    """Test that Prometheus rules YAML is valid."""
    import os
    rules_path = os.path.join(SERVER_DIR, "etc", "prometheus-rules-lfm3-npu.yml")
    assert os.path.exists(rules_path), "Prometheus rules should exist"
    with open(rules_path) as f:
        content = f.read()
    assert "groups:" in content, "Rules should have groups"
    assert "LFM3NPUServerDown" in content, "Should have NPU down alert"
    assert "LFM3CPUServerDown" in content, "Should have CPU down alert"
    assert "LFM3CPUDecodeSlow" in content, "Should have CPU decode slow alert"
    assert "LFM3CPUUptimeHigh" in content, "Should have CPU uptime high alert"
    assert "LFM3DiskUsageHigh" in content, "Should have disk usage high alert"
    assert "LFM3DiskUsageCritical" in content, "Should have disk usage critical alert"
    assert "LFM3NPUUnresponsiveRestarts" in content, "Should have NPU unresponsive restarts alert"
    assert "LFM3CPUUnresponsiveRestarts" in content, "Should have CPU unresponsive restarts alert"
    print("  ✓ Prometheus rules YAML is valid with all alerts")


def test_watchdog_service_has_execstartpre():
    """Test that NPU service has ExecStartPre for port cleanup."""
    import os
    svc_path = os.path.join(SERVER_DIR, "etc", "lfm3-npu-npu.service")
    assert os.path.exists(svc_path), "NPU service should exist"
    with open(svc_path) as f:
        content = f.read()
    assert "ExecStartPre" in content, "NPU service should have ExecStartPre"
    assert "pkill" in content, "ExecStartPre should kill stale processes"
    print("  ✓ NPU service has ExecStartPre for port cleanup")


def test_metrics_service_has_cpu_metrics():
    """Test that metrics exporter exposes CPU metrics."""
    import os
    script_path = os.path.join(SERVER_DIR, "scripts", "npu_metrics_exporter.py")
    assert os.path.exists(script_path), "Metrics exporter should exist"
    with open(script_path) as f:
        content = f.read()
    assert "lfm3_cpu_up" in content, "Should expose CPU up metric"
    assert "lfm3_cpu_avg_decode_ms" in content, "Should expose CPU decode ms"
    assert "cpu_port" in content, "Should accept CPU port argument"
    print("  ✓ Metrics exporter exposes CPU metrics")


def test_quality_monitor_has_degradation_detection():
    """Test that quality monitor has degradation detection logic."""
    import os
    script_path = os.path.join(SERVER_DIR, "scripts", "quality_monitor.py")
    assert os.path.exists(script_path), "Quality monitor should exist"
    with open(script_path) as f:
        content = f.read()
    assert "conversation loop" in content, "Should detect conversation loops"
    assert "numbered list explosion" in content, "Should detect list explosions"
    assert "repeated characters" in content, "Should detect repeated chars"
    assert "wrong math" in content, "Should detect wrong math"
    assert "restart_cooldown" in content, "Should have restart cooldown"
    print("  ✓ Quality monitor has degradation detection")


def test_cli_has_all_commands():
    """Test that CLI has all expected commands."""
    import subprocess
    result = subprocess.run(
        ["python3", os.path.join(SERVER_DIR, "bin", "lfm3-npu"), "--help"],
        capture_output=True, text=True, timeout=10
    )
    output = result.stdout
    for cmd in ["smoke-test", "quality-monitor", "version-check", "benchmark-cron", "validate"]:
        assert cmd in output, f"CLI should have {cmd} command"
    print("  ✓ CLI has all expected commands")


def test_startup_validate_script_exists():
    """Test that startup validation script exists and is executable."""
    import os
    script = os.path.join(SERVER_DIR, "scripts", "startup-validate.py")
    assert os.path.exists(script), "startup-validate.py should exist"
    assert os.access(script, os.X_OK), "startup-validate.py should be executable"
    print("  ✓ Startup validation script exists and is executable")


def test_startup_validate_has_all_checks():
    """Test that startup validation covers all critical areas."""
    import os
    script_path = os.path.join(SERVER_DIR, "scripts", "startup-validate.py")
    assert os.path.exists(script_path), "startup-validate.py should exist"
    with open(script_path) as f:
        content = f.read()
    assert "Systemd Services" in content, "Should check systemd services"
    assert "Systemd Timers" in content, "Should check systemd timers"
    assert "Server Health" in content, "Should check server health"
    assert "Metrics Exporter" in content, "Should check metrics exporter"
    assert "NPU Generation Quality" in content, "Should check NPU generation"
    assert "Disk Space" in content, "Should check disk space"
    print("  ✓ Startup validation covers all critical areas")


def test_metrics_exporter_has_disk_metrics():
    """Test that metrics exporter exposes disk metrics."""
    import os
    script_path = os.path.join(SERVER_DIR, "scripts", "npu_metrics_exporter.py")
    assert os.path.exists(script_path), "Metrics exporter should exist"
    with open(script_path) as f:
        content = f.read()
    assert "lfm3_disk_total_bytes" in content, "Should expose disk total"
    assert "lfm3_disk_used_percent" in content, "Should expose disk used percent"
    assert "_collect_disk_metrics" in content, "Should have disk collection method"
    print("  ✓ Metrics exporter exposes disk metrics")


def test_metrics_exporter_has_quality_monitor_metrics():
    """Test that metrics exporter exposes quality monitor metrics."""
    import os
    script_path = os.path.join(SERVER_DIR, "scripts", "npu_metrics_exporter.py")
    assert os.path.exists(script_path), "Metrics exporter should exist"
    with open(script_path) as f:
        content = f.read()
    assert "lfm3_qm_cpu_restarts" in content, "Should expose CPU restarts"
    assert "lfm3_qm_npu_restarts" in content, "Should expose NPU restarts"
    assert "lfm3_qm_npu_unresponsive_restarts" in content, "Should expose NPU unresponsive restarts"
    assert "_read_quality_monitor_state" in content, "Should read QM state"
    print("  ✓ Metrics exporter exposes quality monitor metrics")


def test_quality_monitor_has_unresponsive_detection():
    """Test that quality monitor detects unresponsive servers."""
    import os
    script_path = os.path.join(SERVER_DIR, "scripts", "quality_monitor.py")
    assert os.path.exists(script_path), "Quality monitor should exist"
    with open(script_path) as f:
        content = f.read()
    assert "check_server_reachable" in content, "Should check server reachability"
    assert "npu_unresponsive_restarts" in content, "Should track NPU unresponsive restarts"
    assert "unreachable" in content.lower(), "Should detect unreachable servers"
    print("  ✓ Quality monitor has unresponsive detection")


def test_quality_monitor_writes_state():
    """Test that quality monitor writes state file."""
    import os
    script_path = os.path.join(SERVER_DIR, "scripts", "quality_monitor.py")
    assert os.path.exists(script_path), "Quality monitor should exist"
    with open(script_path) as f:
        content = f.read()
    assert "STATE_FILE" in content, "Should define state file path"
    assert "write_state" in content, "Should have write_state function"
    assert "cpu_restarts" in content, "Should track CPU restarts"
    assert "npu_restarts" in content, "Should track NPU restarts"
    print("  ✓ Quality monitor writes state file")


# ---------------------------------------------------------------------------
# Enterprise middleware tests
# ---------------------------------------------------------------------------


def test_enterprise_module_imports():
    """Test that the enterprise middleware module imports cleanly."""
    from lfm3_npu.enterprise import (
        APIKeyAuth,
        RateLimiter,
        CircuitBreaker,
        CircuitOpenError,
        EnterpriseMiddleware,
        log_event,
        require_api_key,
    )
    assert callable(APIKeyAuth)
    assert callable(RateLimiter)
    assert callable(CircuitBreaker)
    assert issubclass(CircuitOpenError, Exception)
    print("  ✓ Enterprise module imports work")


def test_api_key_auth():
    """Test API key authentication behavior."""
    import os
    from lfm3_npu.enterprise import APIKeyAuth

    # No keys configured -> open access
    os.environ.pop("LFM3_API_KEYS", None)
    auth = APIKeyAuth()
    assert not auth.enabled
    assert auth.authenticate(None) == "anonymous"
    assert auth.authenticate("anything") == "anonymous"

    # Keys configured -> strict validation
    os.environ["LFM3_API_KEYS"] = "sk-live-abc123:prod-web,sk-live-def456:dev-cli"
    # Force re-init by creating a fresh instance (module caches nothing global here)
    auth2 = APIKeyAuth.__new__(APIKeyAuth)
    APIKeyAuth.__init__(auth2)
    assert auth2.enabled is True
    assert auth2.authenticate("sk-live-abc123") == "prod-web"
    assert auth2.authenticate("sk-live-def456") == "dev-cli"
    assert auth2.authenticate("wrong-key") is None
    assert auth2.authenticate(None) is None
    assert auth2.authenticate("") is None
    print("  ✓ API key authentication works")


def test_rate_limiter():
    """Test token bucket rate limiting behavior."""
    from lfm3_npu.enterprise import RateLimiter

    # High rate, small burst: allow burst then reject
    limiter = RateLimiter.__new__(RateLimiter)
    RateLimiter.__init__(limiter)
    limiter.rate = 1000.0  # effectively unlimited refill for this test
    limiter.capacity = 3

    class FakeRequest:
        class Client:
            host = "10.0.0.1"
        client = Client()

    req = FakeRequest()

    # First 3 allowed (burst=3)
    for i in range(3):
        assert limiter.allow(req, None) is None, f"request {i} should be allowed"

    # 4th should be rate limited
    retry = limiter.allow(req, None)
    assert retry is not None and retry > 0, "4th request should be rate limited"
    print("  ✓ Rate limiter enforces burst limit")


def test_circuit_breaker():
    """Test circuit breaker open / half-open / closed transitions."""
    from lfm3_npu.enterprise import CircuitBreaker, CircuitOpenError

    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

    # Closed: calls succeed
    for _ in range(2):
        with cb.call("test"):
            pass
    assert cb.state == CircuitBreaker.CLOSED

    # Two consecutive failures open the circuit
    for _ in range(2):
        try:
            with cb.call("test"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
    assert cb.state == CircuitBreaker.OPEN

    # Open: calls are rejected immediately
    try:
        with cb.call("test"):
            pass
        assert False, "should have raised CircuitOpenError"
    except CircuitOpenError as e:
        assert e.retry_after > 0

    # Wait for recovery timeout -> half-open allows one trial
    import time
    time.sleep(0.15)
    try:
        with cb.call("test"):
            pass  # trial succeeds
        assert cb.state == CircuitBreaker.CLOSED
    except CircuitOpenError:
        assert False, "half-open trial should succeed"
    print("  ✓ Circuit breaker open/half-open/closed transitions work")


def test_structured_logging():
    """Test structured JSON log event emission."""
    from lfm3_npu.enterprise import log_event, get_logger, _JsonFormatter
    import io
    import logging
    import json

    logger = get_logger()
    # Swap in a StringIO stream handler so we can assert on the JSON output
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(_JsonFormatter())
    logger.handlers = [handler]

    log_event(
        "test.event",
        request_id="req-123",
        client_ip="10.0.0.1",
        status_code=200,
        latency_ms=42.5,
        provider="CPUExecutionProvider",
    )
    output = stream.getvalue()
    assert "req-123" in output
    assert "10.0.0.1" in output
    assert "200" in output
    assert "42.5" in output
    assert "CPUExecutionProvider" in output
    parsed = json.loads(output.strip().splitlines()[-1])
    assert parsed["request_id"] == "req-123"
    assert parsed["status_code"] == 200
    print("  ✓ Structured logging emits valid JSON")


def test_config_has_enterprise_defaults():
    """Test that config documents enterprise sections."""
    from lfm3_npu import load_config
    config = load_config()
    assert "security" in config.to_dict(), "Config should document security section"
    assert "rate_limit" in config.to_dict(), "Config should document rate_limit section"
    assert "circuit_breaker" in config.to_dict(), "Config should document circuit_breaker section"
    print("  ✓ Config documents enterprise sections")


def test_cpu_service_has_enterprise_env():
    """Test that CPU systemd service documents enterprise env vars."""
    import os
    svc_path = os.path.join(SERVER_DIR, "etc", "lfm3-npu-cpu.service")
    assert os.path.exists(svc_path)
    with open(svc_path) as f:
        content = f.read()
    assert "LFM3_ENV=production" in content
    assert "LFM3_LOG_FORMAT=json" in content
    assert "LFM3_RATE_LIMIT_RPM" in content
    assert "LFM3_API_KEYS" in content
    print("  ✓ CPU service documents enterprise env vars")


def test_batched_server_has_live_ready():
    """Test that batched CPU server exposes /live and /ready probes."""
    import os
    path = os.path.join(SERVER_DIR, "lib", "lfm3_npu", "batched_server.py")
    assert os.path.exists(path)
    with open(path) as f:
        content = f.read()
    assert "@app.get(\"/live\")" in content, "Should have /live probe"
    assert "@app.get(\"/ready\")" in content, "Should have /ready probe"
    assert "check_rate_limit" in content, "Should have rate-limit dependency"
    assert "_circuit_breaker" in content, "Should have circuit breaker"
    print("  ✓ Batched server has /live, /ready, rate limit, circuit breaker")


def test_npu_server_has_live_ready():
    """Test that NPU server exposes /live and /ready probes."""
    import os
    path = os.path.join(SERVER_DIR, "lib", "lfm3_npu", "npu_server.py")
    assert os.path.exists(path)
    with open(path) as f:
        content = f.read()
    assert "@app.get(\"/live\")" in content, "Should have /live probe"
    assert "@app.get(\"/ready\")" in content, "Should have /ready probe"
    assert "_circuit_breaker" in content, "Should have circuit breaker"
    print("  ✓ NPU server has /live, /ready, circuit breaker")


def test_otel_collector_jaeger_path():
    """Test that OTel collector exports to Jaeger's OTLP HTTP path."""
    import os
    path = os.path.join(SERVER_DIR, "docker", "otel-collector-config.yaml")
    assert os.path.exists(path)
    with open(path) as f:
        content = f.read()
    assert "endpoint: jaeger:4318/api/otlp" in content, (
        "OTel exporter endpoint must include /api/otlp prefix so the "
        "appended /v1/traces yields /api/otlp/v1/traces (Jaeger's path)"
    )
    print("  ✓ OTel collector -> Jaeger path is correct")




def test_startup_validate_service_exists():
    """Test that startup validate systemd service exists."""
    import os
    svc_path = os.path.join(SERVER_DIR, "etc", "lfm3-npu-startup-validate.service")
    timer_path = os.path.join(SERVER_DIR, "etc", "lfm3-npu-startup-validate.timer")
    assert os.path.exists(svc_path), "Startup validate service should exist"
    assert os.path.exists(timer_path), "Startup validate timer should exist"
    with open(svc_path) as f:
        content = f.read()
    assert "Type=oneshot" in content, "Should be oneshot service"
    with open(timer_path) as f:
        timer_content = f.read()
    assert "OnBootSec" in timer_content, "Timer should run on boot"
    print("  ✓ Startup validate service/timer exists")


def main():
    """Run all tests."""
    print("Running LFM3-1.7B NPU tooling tests...\n")

    tests = [
        test_config_loading,
        test_module_imports,
        test_converter,
        test_benchmark,
        test_server_manager,
        test_smoke_test_script_exists,
        test_ci_test_script_exists,
        test_quality_monitor_script_exists,
        test_deploy_script_exists,
        test_grafana_dashboard_valid_json,
        test_prometheus_rules_valid_yaml,
        test_watchdog_service_has_execstartpre,
        test_metrics_service_has_cpu_metrics,
        test_quality_monitor_has_degradation_detection,
        test_cli_has_all_commands,
        test_startup_validate_script_exists,
        test_startup_validate_has_all_checks,
        test_metrics_exporter_has_disk_metrics,
        test_metrics_exporter_has_quality_monitor_metrics,
        test_quality_monitor_writes_state,
        test_startup_validate_service_exists,
        test_quality_monitor_has_unresponsive_detection,
        test_enterprise_module_imports,
        test_api_key_auth,
        test_rate_limiter,
        test_circuit_breaker,
        test_structured_logging,
        test_config_has_enterprise_defaults,
        test_cpu_service_has_enterprise_env,
        test_batched_server_has_live_ready,
        test_npu_server_has_live_ready,
        test_otel_collector_jaeger_path,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)} tests")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())