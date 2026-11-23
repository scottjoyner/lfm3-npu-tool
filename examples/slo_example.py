#!/usr/bin/env python3
"""
Example: Service-Level Objective (SLO) Tracking
Demonstrates recording request outcomes, checking SLO compliance, and
monitoring error budget consumption.
"""
import sys
import os
import time
import random

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SERVER_DIR, "lib"))

from lfm3_npu import SLOTarget


def main():
    print("=== LFM3-1.7B SLO Tracking Example ===\n")

    # Create a tracker with custom targets
    target = SLOTarget(
        availability=99.9,      # 99.9% availability target
        latency_p95_ms=5000.0,  # 5 second p95 latency target
        window_seconds=3600,    # 1 hour evaluation window
    )
    from lfm3_npu.slo import SLOTracker
    tracker = SLOTracker(target=target)

    # Simulate recording request outcomes
    print("1. Recording request outcomes...")
    for i in range(100):
        success = random.random() < 0.995  # 99.5% success rate
        latency = random.uniform(50, 2000)  # 50ms to 2s latency
        tracker.record(success=success, latency_ms=latency, endpoint="chat/completions")

    # Take a snapshot
    print("2. Taking SLO snapshot...")
    snap = tracker.snapshot()
    print(f"   Total requests: {snap['total_requests']}")
    print(f"   Success: {snap['total_success']}  Errors: {snap['total_errors']}")
    print(f"   Availability: {snap['availability_percent']}% (target: {snap['availability_target_percent']}%)")
    print(f"   P95 latency: {snap['p95_latency_ms']}ms (target: {snap['latency_p95_target_ms']}ms)")
    print(f"   Error budget used: {snap['error_budget_used_percent']}%")
    print(f"   Error budget remaining: {snap['error_budget_remaining_percent']}%")
    print(f"   SLO met: {snap['slo_met']}\n")

    # Record more requests with some errors to consume error budget
    print("3. Recording additional requests with higher error rate...")
    for i in range(50):
        success = random.random() < 0.90  # 90% success rate
        latency = random.uniform(100, 3000)
        tracker.record(success=success, latency_ms=latency, endpoint="chat/completions")

    snap2 = tracker.snapshot()
    print(f"   Total requests: {snap2['total_requests']}")
    print(f"   Availability: {snap2['availability_percent']}%")
    print(f"   Error budget remaining: {snap2['error_budget_remaining_percent']}%")
    print(f"   SLO met: {snap2['slo_met']}\n")

    if not snap2["slo_met"]:
        print("  ⚠️  SLO NOT MET - Error budget may be exhausted")
        print("  Consider investigating latency or error rate issues")
    else:
        print("  ✓ SLO met - Error budget is healthy")

    print("\n✓ SLO tracking example complete")


if __name__ == "__main__":
    main()
