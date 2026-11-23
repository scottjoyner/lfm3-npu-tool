#!/usr/bin/env python3
"""
Example: Disaster Recovery (Multi-Region)
Demonstrates region management, health checking, failover, and failback.
"""
import sys
import os
import tempfile

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SERVER_DIR, "lib"))

from lfm3_npu import get_dr_manager, DisasterRecoveryManager


def main():
    print("=== LFM3-1.7B Disaster Recovery Example ===\n")

    # Use temp storage for the example
    tmpdir = tempfile.mkdtemp()
    storage = os.path.join(tmpdir, "dr.json")
    dr = DisasterRecoveryManager(storage_path=storage)

    # 1. Add regions
    print("1. Adding DR regions")
    dr.add_region("us-east-1", status="primary", endpoint="http://use1:1239")
    dr.add_region("us-west-2", status="secondary", endpoint="http://usw2:1239")
    print(f"   Primary: {dr.get_primary().name}")
    print(f"   Secondaries: {[r.name for r in dr.get_secondaries()]}\n")

    # 2. Check health (endpoints don't exist, so will be unhealthy)
    print("2. Checking region health")
    health = dr.check_all_regions()
    for name, healthy in health.items():
        print(f"   {name}: {'healthy' if healthy else 'unhealthy'}\n")

    # 3. Set replication state for secondary
    print("3. Setting replication state for us-west-2")
    dr.set_replication_state("us-west-2", models=True, config=True, keys=True)
    region = dr.get_region("us-west-2")
    print(f"   Models replicated: {region.models_replicated}")
    print(f"   Config replicated: {region.config_replicated}")
    print(f"   Keys replicated: {region.keys_replicated}\n")

    # 4. Check DR status
    print("4. DR status")
    status = dr.get_status()
    print(f"   Primary: {status['primary']}")
    print(f"   Primary healthy: {status['primary_healthy']}")
    print(f"   Ready secondaries: {status['ready_secondaries']}\n")

    # 5. Failover to secondary
    print("5. Failing over to us-west-2")
    try:
        event = dr.failover("us-west-2", reason="primary outage")
        print(f"   Failover event: {event.event_id}")
        print(f"   Status: {event.status.value}")
        print(f"   From: {event.from_region} -> To: {event.to_region}\n")
    except ValueError as e:
        print(f"   Failover skipped: {e}\n")

    # 6. Failback to primary
    print("6. Failing back to us-east-1")
    try:
        event = dr.failback("us-east-1")
        print(f"   Failback event: {event.event_id}")
        print(f"   Status: {event.status.value}\n")
    except ValueError as e:
        print(f"   Failback skipped: {e}\n")

    # 7. Failover history
    print("7. Failover history")
    for e in dr.get_failover_history():
        print(f"   {e.event_id}: {e.from_region} -> {e.to_region} [{e.status.value}]")

    print("\n✓ Disaster recovery example complete")


if __name__ == "__main__":
    main()
