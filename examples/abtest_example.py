#!/usr/bin/env python3
"""
Example: A/B Testing
Demonstrates test creation, traffic splitting, variant routing, metrics,
analysis, and winner promotion.
"""
import sys
import os
import tempfile

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SERVER_DIR, "lib"))

from lfm3_npu import get_ab_test_manager, ABTestManager


def main():
    print("=== LFM3-1.7B A/B Testing Example ===\n")

    # Use temp storage for the example
    tmpdir = tempfile.mkdtemp()
    storage = os.path.join(tmpdir, "ab_tests.json")
    manager = ABTestManager(storage_path=storage)

    # 1. Create an A/B test
    print("1. Creating A/B test: v2 vs v1")
    test = manager.create_test(
        "test-001", "Model 2.0 vs 1.0",
        control_model="lfm3-1.7b-v1",
        treatment_model="lfm3-1.7b-v2",
        traffic_split=0.2,
        description="Compare v2 against v1",
    )
    print(f"   Test ID: {test.test_id}")
    print(f"   Control: {test.variants[0].model_version} ({test.variants[0].traffic_weight*100:.0f}%)")
    print(f"   Treatment: {test.variants[1].model_version} ({test.variants[1].traffic_weight*100:.0f}%)\n")

    # 2. Start the test
    print("2. Starting A/B test")
    manager.start_test("test-001")
    print(f"   Status: running\n")

    # 3. Route users to variants (consistent routing)
    print("3. Variant routing (consistent hash-based)")
    for user_id in ["user-a", "user-b", "user-c", "user-d", "user-e"]:
        variant = manager.get_variant_for_user("test-001", user_id)
        print(f"   {user_id} -> {variant.variant_id} ({variant.model_version})")
    print()

    # 4. Record metrics for both variants
    print("4. Recording metrics")
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
    print("   Recorded 100 requests for each variant\n")

    # 5. Get results
    print("5. Test results")
    results = manager.get_results("test-001")
    for r in results:
        print(f"   {r.variant_id} ({r.variant_name}): "
              f"{r.request_count} requests, "
              f"{r.conversion_rate:.1%} conversion, "
              f"{r.avg_latency_ms:.0f}ms avg latency, "
              f"{r.error_rate:.1%} error rate")
    print()

    # 6. Analyze the test
    print("6. Analysis")
    analysis = manager.analyze_test("test-001")
    print(f"   Conversion lift: {analysis['conversion_lift_percent']:+.2f}%")
    print(f"   Latency change: {analysis['latency_change_percent']:+.2f}%")
    print(f"   Cost change: {analysis['cost_change_percent']:+.2f}%")
    print(f"   Statistically significant: {analysis['statistically_significant']}")
    print(f"   Recommendation: {analysis['recommendation']}\n")

    # 7. Complete and promote winner
    print("7. Completing test and promoting winner")
    manager.complete_test("test-001", winner_variant_id="treatment")
    promoted = manager.promote_winner("test-001")
    if promoted:
        print(f"   Promoted model: {promoted}")

    print("\n✓ A/B testing example complete")


if __name__ == "__main__":
    main()
