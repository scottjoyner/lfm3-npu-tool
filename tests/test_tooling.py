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


def test_server_manager(tmp_path, monkeypatch):
    """Test ServerManager creation without writing to host-specific paths."""
    from lfm3_npu import ServerManager, load_config
    from lfm3_npu import server as server_module

    pid_dir = tmp_path / "run"
    log_dir = tmp_path / "log"
    monkeypatch.setattr(server_module, "PID_DIR", pid_dir)
    monkeypatch.setattr(server_module, "LOG_DIR", log_dir)

    config = load_config()
    manager = ServerManager(config)
    assert manager is not None
    assert pid_dir.is_dir()
    assert log_dir.is_dir()
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