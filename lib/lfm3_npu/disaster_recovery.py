"""
Multi-region disaster recovery for LFM3-1.7B inference servers.

Provides:
  * Region definitions with primary/secondary status
  * Health monitoring across regions
  * Automatic failover to secondary region
  * Manual failback to primary region
  * Replication state tracking (model registry, config, keys)
  * Disaster recovery runbook execution
  * CLI integration for DR management

Usage:
    from lfm3_npu.disaster_recovery import DisasterRecoveryManager, Region

    dr = DisasterRecoveryManager()
    dr.add_region("us-east-1", primary=True, endpoint="http://use1:1239")
    dr.add_region("us-west-2", primary=False, endpoint="http://usw2:1239")
    result = dr.failover("us-west-2")
"""

import enum
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import urllib.request
import urllib.error


class RegionStatus(str, enum.Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    FAILED = "failed"
    RECOVERING = "recovering"


class FailoverStatus(str, enum.Enum):
    IDLE = "idle"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Region:
    """A geographic region for disaster recovery."""

    name: str
    status: RegionStatus = RegionStatus.SECONDARY
    endpoint: Optional[str] = None
    healthy: bool = False
    last_check: Optional[float] = None
    last_error: Optional[str] = None
    models_replicated: bool = False
    config_replicated: bool = False
    keys_replicated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "endpoint": self.endpoint,
            "healthy": self.healthy,
            "last_check": self.last_check,
            "last_error": self.last_error,
            "models_replicated": self.models_replicated,
            "config_replicated": self.config_replicated,
            "keys_replicated": self.keys_replicated,
        }


@dataclass
class FailoverEvent:
    """A disaster recovery failover event."""

    event_id: str
    from_region: str
    to_region: str
    reason: str
    started_at: float
    completed_at: Optional[float] = None
    status: FailoverStatus = FailoverStatus.IN_PROGRESS
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "from_region": self.from_region,
            "to_region": self.to_region,
            "reason": self.reason,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status.value,
            "error": self.error,
        }


class DisasterRecoveryManager:
    """Manages multi-region disaster recovery and failover."""

    def __init__(self, storage_path: Optional[str] = None):
        if storage_path is None:
            storage_path = os.environ.get(
                "LFM3_DR_STORAGE",
                str(Path.home() / ".config" / "lfm3-npu" / "dr.json"),
            )
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._regions: Dict[str, Region] = {}
        self._failover_history: List[FailoverEvent] = []
        self._current_failover: Optional[FailoverEvent] = None
        self._load()

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            import json as _json

            with open(self.storage_path, "r") as f:
                data = _json.load(f)
            for name, rdata in data.get("regions", {}).items():
                rdata = dict(rdata)  # Don't mutate original
                if isinstance(rdata.get("status"), str):
                    rdata["status"] = RegionStatus(rdata["status"])
                self._regions[name] = Region(**rdata)
            for ev in data.get("failover_history", []):
                ev = dict(ev)  # Don't mutate original
                if isinstance(ev.get("status"), str):
                    ev["status"] = FailoverStatus(ev["status"])
                self._failover_history.append(FailoverEvent(**ev))
        except (_json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        import json as _json

        tmp = self.storage_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(
                {
                    "regions": {n: r.to_dict() for n, r in self._regions.items()},
                    "failover_history": [e.to_dict() for e in self._failover_history],
                },
                f,
                indent=2,
            )
        tmp.replace(self.storage_path)

    # ------------------------------------------------------------------ #
    # Region management                                                  #
    # ------------------------------------------------------------------ #

    def add_region(
        self,
        name: str,
        status: str = "secondary",
        endpoint: Optional[str] = None,
    ) -> Region:
        """Add or update a region."""
        region = Region(
            name=name,
            status=RegionStatus(status),
            endpoint=endpoint,
        )
        self._regions[name] = region
        self._save()
        return region

    def get_region(self, name: str) -> Optional[Region]:
        return self._regions.get(name)

    def list_regions(self) -> List[Region]:
        return list(self._regions.values())

    def get_primary(self) -> Optional[Region]:
        for r in self._regions.values():
            if r.status == RegionStatus.PRIMARY:
                return r
        return None

    def get_secondaries(self) -> List[Region]:
        return [r for r in self._regions.values() if r.status == RegionStatus.SECONDARY]

    def set_primary(self, name: str) -> Optional[Region]:
        """Set a region as primary (demotes current primary)."""
        region = self._regions.get(name)
        if region is None:
            return None

        # Demote current primary
        for r in self._regions.values():
            if r.status == RegionStatus.PRIMARY and r.name != name:
                r.status = RegionStatus.SECONDARY

        region.status = RegionStatus.PRIMARY
        self._save()
        return region

    # ------------------------------------------------------------------ #
    # Health checking                                                    #
    # ------------------------------------------------------------------ #

    def check_region_health(self, name: str) -> bool:
        """Check health of a region via its health endpoint."""
        region = self._regions.get(name)
        if region is None or region.endpoint is None:
            return False

        try:
            req = urllib.request.Request(
                f"{region.endpoint}/health", method="GET"
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status == 200:
                    region.healthy = True
                    region.last_error = None
                else:
                    region.healthy = False
                    region.last_error = f"HTTP {resp.status}"
        except Exception as e:  # noqa BLE001
            region.healthy = False
            region.last_error = str(e)

        region.last_check = time.time()
        return region.healthy

    def check_all_regions(self) -> Dict[str, bool]:
        """Check health of all regions."""
        results = {}
        for name in self._regions:
            results[name] = self.check_region_health(name)
        return results

    # ------------------------------------------------------------------ #
    # Replication state                                                  #
    # ------------------------------------------------------------------ #

    def set_replication_state(
        self,
        name: str,
        models: Optional[bool] = None,
        config: Optional[bool] = None,
        keys: Optional[bool] = None,
    ) -> Optional[Region]:
        """Update replication state for a region."""
        region = self._regions.get(name)
        if region is None:
            return None

        if models is not None:
            region.models_replicated = models
        if config is not None:
            region.config_replicated = config
        if keys is not None:
            region.keys_replicated = keys

        self._save()
        return region

    def is_region_ready_for_failover(self, name: str) -> bool:
        """Check if a secondary region is ready to take over as primary."""
        region = self._regions.get(name)
        if region is None:
            return False
        if region.status != RegionStatus.SECONDARY:
            return False
        if not region.healthy:
            return False
        if not region.models_replicated:
            return False
        if not region.config_replicated:
            return False
        if not region.keys_replicated:
            return False
        return True

    # ------------------------------------------------------------------ #
    # Failover                                                           #
    # ------------------------------------------------------------------ #

    def failover(self, to_region: str, reason: str = "manual") -> FailoverEvent:
        """Initiate failover to a secondary region."""
        import uuid

        event_id = f"dr-{uuid.uuid4().hex[:12]}"

        current_primary = self.get_primary()
        target = self._regions.get(to_region)

        if target is None:
            raise ValueError(f"Region {to_region} not found")

        if not self.is_region_ready_for_failover(to_region):
            raise ValueError(f"Region {to_region} is not ready for failover")

        event = FailoverEvent(
            event_id=event_id,
            from_region=current_primary.name if current_primary else "none",
            to_region=to_region,
            reason=reason,
            started_at=time.time(),
            status=FailoverStatus.IN_PROGRESS,
        )
        self._current_failover = event
        self._failover_history.append(event)
        self._save()

        try:
            # Demote current primary
            if current_primary:
                current_primary.status = RegionStatus.FAILED
                self._save()

            # Promote target
            target.status = RegionStatus.PRIMARY
            target.healthy = True
            self._save()

            event.status = FailoverStatus.COMPLETED
            event.completed_at = time.time()
            self._save()

            return event
        except Exception as exc:  # noqa BLE001
            event.status = FailoverStatus.FAILED
            event.error = str(exc)
            event.completed_at = time.time()
            self._save()
            raise

    def failback(self, to_region: str) -> FailoverEvent:
        """Fail back to a region (restore primary status)."""
        return self.failover(to_region, reason="failback")

    # ------------------------------------------------------------------ #
    # Queries                                                            #
    # ------------------------------------------------------------------ #

    def get_status(self) -> Dict[str, Any]:
        """Get overall DR status."""
        primary = self.get_primary()
        secondaries = self.get_secondaries()
        ready_secondaries = [
            r.name for r in secondaries if self.is_region_ready_for_failover(r.name)
        ]

        return {
            "primary": primary.name if primary else None,
            "primary_healthy": primary.healthy if primary else False,
            "secondaries": [r.name for r in secondaries],
            "ready_secondaries": ready_secondaries,
            "current_failover": self._current_failover.to_dict()
            if self._current_failover
            else None,
            "total_failovers": len(self._failover_history),
        }

    def get_failover_history(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._failover_history]


# --------------------------------------------------------------------------- #
# Global instance                                                             #
# --------------------------------------------------------------------------- #

_dr_manager: Optional[DisasterRecoveryManager] = None


def get_dr_manager() -> DisasterRecoveryManager:
    global _dr_manager
    if _dr_manager is None:
        _dr_manager = DisasterRecoveryManager()
    return _dr_manager


# --------------------------------------------------------------------------- #
# CLI entry point                                                             #
# --------------------------------------------------------------------------- #

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LFM3-1.7B Disaster Recovery Manager")
    subparsers = parser.add_subparsers(dest="command")

    # add-region command
    add_parser = subparsers.add_parser("add-region", help="Add a DR region")
    add_parser.add_argument("name", help="Region name")
    add_parser.add_argument(
        "--status", default="secondary", choices=["primary", "secondary"], help="Region status"
    )
    add_parser.add_argument("--endpoint", default=None, help="Region health endpoint URL")
    add_parser.set_defaults(func=cmd_add_region)

    # list-regions command
    list_parser = subparsers.add_parser("list-regions", help="List DR regions")
    list_parser.set_defaults(func=cmd_list_regions)

    # check-health command
    health_parser = subparsers.add_parser("check-health", help="Check region health")
    health_parser.add_argument("name", nargs="?", default=None, help="Region name (all if omitted)")
    health_parser.set_defaults(func=cmd_check_health)

    # set-primary command
    primary_parser = subparsers.add_parser("set-primary", help="Set primary region")
    primary_parser.add_argument("name", help="Region name")
    primary_parser.set_defaults(func=cmd_set_primary)

    # replication command
    repl_parser = subparsers.add_parser("replication", help="Set replication state")
    repl_parser.add_argument("name", help="Region name")
    repl_parser.add_argument("--models", action="store_true", default=None, help="Models replicated")
    repl_parser.add_argument("--no-models", action="store_true", default=None, help="Models not replicated")
    repl_parser.add_argument("--config", action="store_true", default=None, help="Config replicated")
    repl_parser.add_argument("--no-config", action="store_true", default=None, help="Config not replicated")
    repl_parser.add_argument("--keys", action="store_true", default=None, help="Keys replicated")
    repl_parser.add_argument("--no-keys", action="store_true", default=None, help="Keys not replicated")
    repl_parser.set_defaults(func=cmd_replication)

    # failover command
    fo_parser = subparsers.add_parser("failover", help="Failover to a region")
    fo_parser.add_argument("name", help="Target region name")
    fo_parser.add_argument("--reason", default="manual", help="Failover reason")
    fo_parser.set_defaults(func=cmd_failover)

    # failback command
    fb_parser = subparsers.add_parser("failback", help="Fail back to a region")
    fb_parser.add_argument("name", help="Target region name")
    fb_parser.set_defaults(func=cmd_failback)

    # status command
    status_parser = subparsers.add_parser("status", help="Show DR status")
    status_parser.set_defaults(func=cmd_status)

    # history command
    history_parser = subparsers.add_parser("history", help="Show failover history")
    history_parser.set_defaults(func=cmd_history)

    args = parser.parse_args()
    dr = DisasterRecoveryManager()

    if args.command == "add-region":
        region = dr.add_region(args.name, args.status, args.endpoint)
        print(f"Region {region.name} added: status={region.status.value}, endpoint={region.endpoint}")
        return 0

    elif args.command == "list-regions":
        regions = dr.list_regions()
        if not regions:
            print("No regions configured")
            return 0
        print(f"DR Regions ({len(regions)}):")
        for r in regions:
            print(f"  {r.name}: {r.status.value} endpoint={r.endpoint} healthy={r.healthy}")
        return 0

    elif args.command == "check-health":
        if args.name:
            healthy = dr.check_region_health(args.name)
            region = dr.get_region(args.name)
            print(f"Region {args.name}: {'healthy' if healthy else 'unhealthy'}")
            if region and region.last_error:
                print(f"  Last error: {region.last_error}")
        else:
            results = dr.check_all_regions()
            for name, healthy in results.items():
                print(f"  {name}: {'healthy' if healthy else 'unhealthy'}")
        return 0

    elif args.command == "set-primary":
        region = dr.set_primary(args.name)
        if region is None:
            print(f"Region {args.name} not found")
            return 1
        print(f"Region {region.name} set as primary")
        return 0

    elif args.command == "replication":
        models = None
        config = None
        keys = None
        if args.models is not None:
            models = args.models
        if args.no_models is not None:
            models = False
        if args.config is not None:
            config = args.config
        if args.no_config is not None:
            config = False
        if args.keys is not None:
            keys = args.keys
        if args.no_keys is not None:
            keys = False

        region = dr.set_replication_state(args.name, models, config, keys)
        if region is None:
            print(f"Region {args.name} not found")
            return 1
        print(f"Replication state for {region.name}:")
        print(f"  Models: {region.models_replicated}")
        print(f"  Config: {region.config_replicated}")
        print(f"  Keys: {region.keys_replicated}")
        return 0

    elif args.command == "failover":
        try:
            event = dr.failover(args.name, args.reason)
            print(f"Failover {event.event_id}: {event.from_region} -> {event.to_region}")
            print(f"  Status: {event.status.value}")
            print(f"  Reason: {event.reason}")
        except Exception as e:
            print(f"Failover failed: {e}")
            return 1
        return 0

    elif args.command == "failback":
        try:
            event = dr.failback(args.name)
            print(f"Failback {event.event_id}: {event.from_region} -> {event.to_region}")
            print(f"  Status: {event.status.value}")
        except Exception as e:
            print(f"Failback failed: {e}")
            return 1
        return 0

    elif args.command == "status":
        status = dr.get_status()
        print(f"DR Status:")
        print(f"  Primary: {status['primary']} (healthy: {status['primary_healthy']})")
        print(f"  Secondaries: {', '.join(status['secondaries']) if status['secondaries'] else 'none'}")
        print(f"  Ready secondaries: {', '.join(status['ready_secondaries']) if status['ready_secondaries'] else 'none'}")
        if status["current_failover"]:
            cf = status["current_failover"]
            print(f"  Current failover: {cf['from_region']} -> {cf['to_region']} [{cf['status']}]")
        print(f"  Total failovers: {status['total_failovers']}")
        return 0

    elif args.command == "history":
        history = dr.get_failover_history()
        if not history:
            print("No failover history")
            return 0
        print(f"Failover history ({len(history)} events):")
        for e in history:
            started = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(e["started_at"]))
            completed = (
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(e["completed_at"]))
                if e["completed_at"]
                else "N/A"
            )
            print(f"  {started} -> {completed}  {e['from_region']} -> {e['to_region']} [{e['status']}] reason={e['reason']}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
