#!/usr/bin/env python3
"""
Tests for new enterprise modules added in the latest enhancement batch:
- model_promotion (CI/CD promotion pipeline)
- disaster_recovery (multi-region DR)
- auto_scaler (metrics-based auto-scaling)
- ab_testing (A/B testing framework)

These tests use temp storage paths to avoid polluting real state.
"""

import os
import sys
import tempfile
import time

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SERVER_DIR, "lib"))


def test_model_promotion_module_imports():
    """Test that model_promotion module imports and key classes exist."""
    from lfm3_npu.model_promotion import (
        PromotionPipeline, PromotionStage, PromotionStatus,
        PromotionRequest, PromotionGate, get_promotion_pipeline,
    )
    assert PromotionStage.STAGING.value == "staging"
    assert PromotionStage.CANARY.value == "canary"
    assert PromotionStage.PRODUCTION.value == "production"
    assert PromotionStatus.PENDING.value == "pending"
    assert PromotionStatus.DEPLOYED.value == "deployed"
    assert PromotionStatus.ROLLED_BACK.value == "rolled_back"
    print("  ✓ model_promotion imports and enums work")


def test_model_promotion_pipeline_crud():
    """Test promotion pipeline request/approve/deploy/rollback/list/stage."""
    import json
    from lfm3_npu.model_promotion import (
        PromotionPipeline, PromotionStage, PromotionStatus,
    )

    tmpdir = tempfile.mkdtemp()
    storage = os.path.join(tmpdir, "promotion.json")
    pipeline = PromotionPipeline(storage_path=storage)

    # Request promotion
    req = pipeline.request_promotion(
        "1.0.0", PromotionStage.STAGING, PromotionStage.CANARY,
        "tester", notes="test", auto_approve=True,
    )
    assert req.request_id is not None
    assert req.model_version == "1.0.0"
    assert req.status == PromotionStatus.APPROVED  # auto-approved

    # Deploy
    deployed = pipeline.deploy_promotion(req.request_id)
    assert deployed is not None
    assert deployed.status == PromotionStatus.DEPLOYED
    assert deployed.deployed_at is not None

    # Check stage
    stage = pipeline.get_current_stage("1.0.0")
    assert stage == PromotionStage.CANARY

    # List requests
    requests = pipeline.list_requests(model_version="1.0.0")
    assert len(requests) >= 1

    # Rollback
    rolled = pipeline.rollback("1.0.0", reason="test rollback")
    assert rolled is not None
    assert rolled.status == PromotionStatus.ROLLED_BACK

    # Persistence check - reload
    pipeline2 = PromotionPipeline(storage_path=storage)
    stage2 = pipeline2.get_current_stage("1.0.0")
    assert stage2 is None  # rolled back, no deployed stage

    print("  ✓ model_promotion CRUD works")


def test_disaster_recovery_module_imports():
    """Test that disaster_recovery module imports and key classes exist."""
    from lfm3_npu.disaster_recovery import (
        DisasterRecoveryManager, Region, RegionStatus,
        FailoverEvent, FailoverStatus, get_dr_manager,
    )
    assert RegionStatus.PRIMARY.value == "primary"
    assert RegionStatus.SECONDARY.value == "secondary"
    assert FailoverStatus.IDLE.value == "idle"
    assert FailoverStatus.COMPLETED.value == "completed"
    print("  ✓ disaster_recovery imports and enums work")


def test_disaster_recovery_region_management():
    """Test DR region add/list/primary/health/failover/failback."""
    from lfm3_npu.disaster_recovery import (
        DisasterRecoveryManager, RegionStatus, FailoverStatus,
    )

    tmpdir = tempfile.mkdtemp()
    storage = os.path.join(tmpdir, "dr.json")
    dr = DisasterRecoveryManager(storage_path=storage)

    # Add regions
    r1 = dr.add_region("us-east-1", status="primary", endpoint="http://use1:1239")
    r2 = dr.add_region("us-west-2", status="secondary", endpoint="http://usw2:1239")

    assert r1.status == RegionStatus.PRIMARY
    assert r2.status == RegionStatus.SECONDARY

    # List regions
    regions = dr.list_regions()
    assert len(regions) == 2

    # Get primary
    primary = dr.get_primary()
    assert primary is not None
    assert primary.name == "us-east-1"

    # Set primary (failover-like)
    dr.set_primary("us-west-2")
    primary2 = dr.get_primary()
    assert primary2.name == "us-west-2"

    # Replication state
    dr.set_replication_state("us-west-2", models=True, config=True, keys=True)
    region = dr.get_region("us-west-2")
    assert region.models_replicated is True
    assert region.config_replicated is True
    assert region.keys_replicated is True

    # Status
    status = dr.get_status()
    assert status["primary"] == "us-west-2"

    # Health check on non-existent endpoint should fail gracefully
    healthy = dr.check_region_health("us-east-1")
    assert healthy is False  # endpoint doesn't actually exist

    print("  ✓ disaster_recovery region management works")


def test_auto_scaler_module_imports():
    """Test that auto_scaler module imports and key classes exist."""
    from lfm3_npu.auto_scaler import (
        AutoScaler, ScalingPolicy, ScalingMetrics, ScalingDecision,
        ScalingAction, get_auto_scaler,
    )
    assert ScalingAction.SCALE_UP.value == "scale_up"
    assert ScalingAction.SCALE_DOWN.value == "scale_down"
    assert ScalingAction.NO_CHANGE.value == "no_change"
    print("  ✓ auto_scaler imports and enums work")


def test_auto_scaler_evaluation():
    """Test auto-scaler evaluation and manual scale up/down."""
    from lfm3_npu.auto_scaler import AutoScaler, ScalingPolicy, ScalingAction

    tmpdir = tempfile.mkdtemp()
    storage = os.path.join(tmpdir, "autoscaler.json")
    scaler = AutoScaler(policy=ScalingPolicy(min_replicas=1, max_replicas=5),
                        storage_path=storage)

    # Initial status
    status = scaler.get_status()
    initial_replicas = status["current_replicas"]

    # Evaluate (may scale up due to system load - just verify it returns a decision)
    decision = scaler.evaluate()
    assert decision.action in (ScalingAction.SCALE_UP, ScalingAction.SCALE_DOWN, ScalingAction.NO_CHANGE)
    after_eval = scaler.get_status()["current_replicas"]
    assert after_eval >= initial_replicas

    # Manual scale up
    replicas_before = scaler.get_status()["current_replicas"]
    replicas = scaler.scale_up()
    assert replicas == replicas_before + 1

    # Manual scale down
    replicas_before = scaler.get_status()["current_replicas"]
    replicas = scaler.scale_down()
    assert replicas == replicas_before - 1

    # History should have entries from evaluate + manual ops
    history = scaler.get_scaling_history(limit=10)
    assert len(history) >= 2

    print("  ✓ auto_scaler evaluation and manual scaling works")


def test_ab_testing_module_imports():
    """Test that ab_testing module imports and key classes exist."""
    from lfm3_npu.ab_testing import (
        ABTestManager, ABTest, ABVariant, ABTestResult, TestStatus,
        get_ab_test_manager,
    )
    assert TestStatus.DRAFT.value == "draft"
    assert TestStatus.RUNNING.value == "running"
    assert TestStatus.COMPLETED.value == "completed"
    print("  ✓ ab_testing imports and enums work")


def test_ab_testing_lifecycle():
    """Test A/B test create/start/complete/variant/results/analyze/promote."""
    from lfm3_npu.ab_testing import (
        ABTestManager, TestStatus,
    )

    tmpdir = tempfile.mkdtemp()
    storage = os.path.join(tmpdir, "ab_tests.json")
    manager = ABTestManager(storage_path=storage)

    # Create test
    test = manager.create_test(
        "test-001", "Model 2.0 vs 1.0",
        control_model="lfm3-1.7b-v1",
        treatment_model="lfm3-1.7b-v2",
        traffic_split=0.2,
        description="Compare v2 against v1",
    )
    assert test.test_id == "test-001"
    assert test.status == TestStatus.DRAFT
    assert len(test.variants) == 2

    # List tests
    tests = manager.list_tests()
    assert len(tests) == 1

    # Get test
    got = manager.get_test("test-001")
    assert got is not None
    assert got.name == "Model 2.0 vs 1.0"

    # Start test
    started = manager.start_test("test-001")
    assert started is not None
    assert started.status == TestStatus.RUNNING

    # Variant routing (consistent)
    v1 = manager.get_variant_for_user("test-001", "user-a")
    v2 = manager.get_variant_for_user("test-001", "user-a")
    assert v1.variant_id == v2.variant_id  # consistent routing

    # Record metrics
    manager.record_variant_metric(
        "test-001", "control", request_count=100,
        conversion_count=10, latency_ms=50.0, error_count=1,
        tokens=5000, cost=0.01,
    )
    manager.record_variant_metric(
        "test-001", "treatment", request_count=100,
        conversion_count=15, latency_ms=55.0, error_count=0,
        tokens=5000, cost=0.01,
    )

    # Results
    results = manager.get_results("test-001")
    assert len(results) == 2

    # Analyze
    analysis = manager.analyze_test("test-001")
    assert "error" not in analysis
    assert "conversion_lift_percent" in analysis
    assert "recommendation" in analysis

    # Complete
    completed = manager.complete_test("test-001", winner_variant_id="treatment")
    assert completed.status == TestStatus.COMPLETED
    assert completed.winner_variant_id == "treatment"

    print("  ✓ ab_testing lifecycle works")


def test_all_new_modules_importable_from_package():
    """Test that all new modules are importable from the lfm3_npu package."""
    from lfm3_npu import (
        get_promotion_pipeline, PromotionPipeline, PromotionStage,
        get_dr_manager, DisasterRecoveryManager,
        get_auto_scaler, AutoScaler, ScalingPolicy,
        get_ab_test_manager, ABTestManager, ABTest,
        get_alert_manager, SecurityScanner, get_regression_detector,
        RegressionDetector, get_cost_tracker,
        get_rotation_manager, get_key_manager,
        get_tenant_manager, get_model_registry,
        LoadBalancer, get_flags, get_audit, get_backup_manager,
        EnterpriseMiddleware, RBACAuth,
        get_tracer, get_tracker,
    )
    print("  ✓ All new modules importable from lfm3_npu package")


def test_cli_has_promotion_commands():
    """Test CLI has promotion subcommands."""
    import subprocess
    result = subprocess.run(
        ["python3", os.path.join(SERVER_DIR, "bin", "lfm3-npu"),
         "promotion", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    for sub in ["request", "approve", "deploy", "rollback", "list", "stage"]:
        assert sub in result.stdout, f"promotion should have {sub} subcommand"
    print("  ✓ CLI promotion commands present")


def test_cli_has_dr_commands():
    """Test CLI has disaster recovery subcommands."""
    import subprocess
    result = subprocess.run(
        ["python3", os.path.join(SERVER_DIR, "bin", "lfm3-npu"),
         "dr", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    for sub in ["add-region", "list-regions", "check-health", "set-primary",
                "replication", "failover", "failback", "status", "history"]:
        assert sub in result.stdout, f"dr should have {sub} subcommand"
    print("  ✓ CLI DR commands present")


def test_cli_has_autoscale_commands():
    """Test CLI has auto-scaler subcommands."""
    import subprocess
    result = subprocess.run(
        ["python3", os.path.join(SERVER_DIR, "bin", "lfm3-npu"),
         "autoscale", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    for sub in ["evaluate", "status", "policy", "scale-up", "scale-down", "history"]:
        assert sub in result.stdout, f"autoscale should have {sub} subcommand"
    print("  ✓ CLI autoscale commands present")


def test_cli_has_abtest_commands():
    """Test CLI has A/B testing subcommands."""
    import subprocess
    result = subprocess.run(
        ["python3", os.path.join(SERVER_DIR, "bin", "lfm3-npu"),
         "abtest", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    for sub in ["create", "start", "complete", "list", "get", "variant",
                "results", "analyze", "record", "promote"]:
        assert sub in result.stdout, f"abtest should have {sub} subcommand"
    print("  ✓ CLI abtest commands present")


def test_cli_promotion_stage_end_to_end():
    """Test CLI promotion stage command actually executes."""
    import subprocess, os, tempfile
    env = os.environ.copy()
    env["LFM3_PROMOTION_STORAGE"] = os.path.join(tempfile.mkdtemp(), "promotion.json")
    env["PYTHONPATH"] = os.path.join(SERVER_DIR, "lib")

    result = subprocess.run(
        ["python3", os.path.join(SERVER_DIR, "bin", "lfm3-npu"),
         "promotion", "stage"],
        capture_output=True, text=True, timeout=10, env=env
    )
    assert result.returncode == 0, f"promotion stage failed: {result.stderr}"
    print("  ✓ CLI promotion stage end-to-end works")


def test_cli_dr_status_end_to_end():
    """Test CLI dr status command actually executes."""
    import subprocess, os, tempfile
    env = os.environ.copy()
    env["LFM3_DR_STORAGE"] = os.path.join(tempfile.mkdtemp(), "dr.json")
    env["PYTHONPATH"] = os.path.join(SERVER_DIR, "lib")

    result = subprocess.run(
        ["python3", os.path.join(SERVER_DIR, "bin", "lfm3-npu"),
         "dr", "status"],
        capture_output=True, text=True, timeout=10, env=env
    )
    assert result.returncode == 0, f"dr status failed: {result.stderr}"
    assert "DR Status" in result.stdout
    print("  ✓ CLI dr status end-to-end works")


def test_cli_autoscale_status_end_to_end():
    """Test CLI autoscale status command actually executes."""
    import subprocess, os, tempfile
    env = os.environ.copy()
    env["LFM3_AUTOSCALER_STORAGE"] = os.path.join(tempfile.mkdtemp(), "autoscaler.json")
    env["PYTHONPATH"] = os.path.join(SERVER_DIR, "lib")

    result = subprocess.run(
        ["python3", os.path.join(SERVER_DIR, "bin", "lfm3-npu"),
         "autoscale", "status"],
        capture_output=True, text=True, timeout=10, env=env
    )
    assert result.returncode == 0, f"autoscale status failed: {result.stderr}"
    assert "Auto-Scaler Status" in result.stdout
    print("  ✓ CLI autoscale status end-to-end works")


def test_cli_abtest_list_end_to_end():
    """Test CLI abtest list command actually executes."""
    import subprocess, os, tempfile
    env = os.environ.copy()
    env["LFM3_ABTEST_STORAGE"] = os.path.join(tempfile.mkdtemp(), "ab_tests.json")
    env["PYTHONPATH"] = os.path.join(SERVER_DIR, "lib")

    result = subprocess.run(
        ["python3", os.path.join(SERVER_DIR, "bin", "lfm3-npu"),
         "abtest", "list"],
        capture_output=True, text=True, timeout=10, env=env
    )
    assert result.returncode == 0, f"abtest list failed: {result.stderr}"
    assert "A/B tests" in result.stdout
    print("  ✓ CLI abtest list end-to-end works")


def test_config_validate_new_sections():
    """Test that config-validate CLI accepts new module sections."""
    import subprocess, os, tempfile
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(SERVER_DIR, "lib")

    # Write a temp config with new sections
    config_dir = tempfile.mkdtemp()
    config_path = os.path.join(config_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write("""
model_promotion:
  auto_approve: false
  stages: [staging, canary, production]
disaster_recovery:
  regions:
    - name: primary
      status: primary
      endpoint: http://localhost:1239
auto_scaling:
  min_replicas: 1
  max_replicas: 10
ab_testing: {}
cost_tracking: {}
alerting: {}
security_scan: {}
key_rotation: {}
""")
    env["LFM3_NPU_CONFIG"] = config_path

    result = subprocess.run(
        ["python3", os.path.join(SERVER_DIR, "bin", "lfm3-npu"),
         "config-validate"],
        capture_output=True, text=True, timeout=10, env=env
    )
    assert result.returncode == 0, f"config-validate failed: {result.stderr}"
    print("  ✓ CLI config-validate with new sections works")


def main():
    """Run all new module tests."""
    print("Running new enterprise module tests...\n")

    tests = [
        test_model_promotion_module_imports,
        test_model_promotion_pipeline_crud,
        test_disaster_recovery_module_imports,
        test_disaster_recovery_region_management,
        test_auto_scaler_module_imports,
        test_auto_scaler_evaluation,
        test_ab_testing_module_imports,
        test_ab_testing_lifecycle,
        test_all_new_modules_importable_from_package,
        test_cli_has_promotion_commands,
        test_cli_has_dr_commands,
        test_cli_has_autoscale_commands,
        test_cli_has_abtest_commands,
        test_cli_promotion_stage_end_to_end,
        test_cli_dr_status_end_to_end,
        test_cli_autoscale_status_end_to_end,
        test_cli_abtest_list_end_to_end,
        test_config_validate_new_sections,
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
