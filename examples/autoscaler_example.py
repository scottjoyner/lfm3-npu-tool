#!/usr/bin/env python3
"""
Example: Auto-Scaling
Demonstrates metrics-based scaling decisions and manual scaling.
"""
import sys
import os
import tempfile

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SERVER_DIR, "lib"))

from lfm3_npu import (
    get_auto_scaler, AutoScaler, ScalingPolicy, ScalingAction,
)


def main():
    print("=== LFM3-1.7B Auto-Scaling Example ===\n")

    # Use temp storage for the example
    tmpdir = tempfile.mkdtemp()
    storage = os.path.join(tmpdir, "autoscaler.json")

    # 1. Create auto-scaler with custom policy
    print("1. Creating auto-scaler with custom policy")
    policy = ScalingPolicy(
        min_replicas=2,
        max_replicas=20,
        target_cpu_percent=70.0,
        target_memory_percent=80.0,
        target_queue_depth=10,
        target_latency_p95_ms=1000.0,
        max_error_rate=0.05,
    )
    scaler = AutoScaler(policy=policy, storage_path=storage)
    status = scaler.get_status()
    print(f"   Current replicas: {status['current_replicas']}")
    print(f"   Min/Max: {status['min_replicas']}/{status['max_replicas']}\n")

    # 2. Evaluate scaling (no load balancer, so metrics come from system)
    print("2. Evaluating scaling decision")
    decision = scaler.evaluate()
    print(f"   Action: {decision.action.value}")
    print(f"   Reason: {decision.reason}")
    print(f"   Replicas: {decision.current_replicas} -> {decision.target_replicas}\n")

    # 3. Manual scale up
    print("3. Manual scale up")
    replicas = scaler.scale_up()
    print(f"   Scaled up to {replicas} replicas\n")

    # 4. Manual scale down
    print("4. Manual scale down")
    replicas = scaler.scale_down()
    print(f"   Scaled down to {replicas} replicas\n")

    # 5. Update policy
    print("5. Updating scaling policy")
    scaler.scale_up()
    scaler.scale_up()
    print(f"   Current replicas: {scaler.get_status()['current_replicas']}\n")

    # 6. Scaling history
    print("6. Scaling history")
    history = scaler.get_scaling_history(limit=10)
    for entry in history:
        print(f"   {entry['action']}: {entry['current_replicas']}->{entry['target_replicas']} - {entry['reason']}")

    print("\n✓ Auto-scaling example complete")


if __name__ == "__main__":
    main()
