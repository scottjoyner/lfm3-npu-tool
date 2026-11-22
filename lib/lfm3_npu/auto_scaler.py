"""
Auto-scaling for LFM3-1.7B inference servers.

Provides:
  * Metrics-based scaling decisions (queue depth, latency, error rate, CPU/memory)
  * Scale-up and scale-down policies with cooldown periods
  * Integration with load balancer and server metrics
  * Simulated replica management (for environments without K8s)
  * CLI integration for auto-scaling management

Usage:
    from lfm3_npu.auto_scaler import AutoScaler, ScalingPolicy

    scaler = AutoScaler(load_balancer, metrics_collector)
    decision = scaler.evaluate()
    if decision.should_scale_up:
        scaler.scale_up()
"""

import enum
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class ScalingAction(str, enum.Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_CHANGE = "no_change"


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""

    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_percent: float = 70.0
    target_memory_percent: float = 80.0
    target_queue_depth: int = 10
    target_latency_p95_ms: float = 1000.0
    max_error_rate: float = 0.05
    scale_up_cooldown_seconds: int = 60
    scale_down_cooldown_seconds: int = 300
    evaluation_interval_seconds: int = 30

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "target_cpu_percent": self.target_cpu_percent,
            "target_memory_percent": self.target_memory_percent,
            "target_queue_depth": self.target_queue_depth,
            "target_latency_p95_ms": self.target_latency_p95_ms,
            "max_error_rate": self.max_error_rate,
            "scale_up_cooldown_seconds": self.scale_up_cooldown_seconds,
            "scale_down_cooldown_seconds": self.scale_down_cooldown_seconds,
            "evaluation_interval_seconds": self.evaluation_interval_seconds,
        }


@dataclass
class ScalingMetrics:
    """Current metrics for scaling decisions."""

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    queue_depth: int = 0
    latency_p95_ms: float = 0.0
    error_rate: float = 0.0
    requests_per_second: float = 0.0
    active_replicas: int = 1
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingDecision:
    """Result of a scaling evaluation."""

    action: ScalingAction
    current_replicas: int
    target_replicas: int
    reason: str
    metrics: ScalingMetrics
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "current_replicas": self.current_replicas,
            "target_replicas": self.target_replicas,
            "reason": self.reason,
            "metrics": {
                "cpu_percent": self.metrics.cpu_percent,
                "memory_percent": self.metrics.memory_percent,
                "queue_depth": self.metrics.queue_depth,
                "latency_p95_ms": self.metrics.latency_p95_ms,
                "error_rate": self.metrics.error_rate,
                "requests_per_second": self.metrics.requests_per_second,
            },
            "timestamp": self.timestamp,
        }


class AutoScaler:
    """Auto-scaling manager for inference servers."""

    def __init__(
        self,
        load_balancer=None,
        policy: Optional[ScalingPolicy] = None,
        storage_path: Optional[str] = None,
    ):
        self.load_balancer = load_balancer
        self.policy = policy or ScalingPolicy()
        if storage_path is None:
            storage_path = os.environ.get(
                "LFM3_AUTOSCALER_STORAGE",
                str(Path.home() / ".config" / "lfm3-npu" / "autoscaler.json"),
            )
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._current_replicas = self.policy.min_replicas
        self._last_scale_up: Optional[float] = None
        self._last_scale_down: Optional[float] = None
        self._scaling_history: List[ScalingDecision] = []
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
            self._current_replicas = data.get("current_replicas", self.policy.min_replicas)
            self._last_scale_up = data.get("last_scale_up")
            self._last_scale_down = data.get("last_scale_down")
        except (_json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        import json as _json

        tmp = self.storage_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(
                {
                    "current_replicas": self._current_replicas,
                    "last_scale_up": self._last_scale_up,
                    "last_scale_down": self._last_scale_down,
                },
                f,
                indent=2,
            )
        tmp.replace(self.storage_path)

    # ------------------------------------------------------------------ #
    # Policy management                                                  #
    # ------------------------------------------------------------------ #

    def set_policy(self, policy: ScalingPolicy) -> None:
        self.policy = policy

    def get_policy(self) -> ScalingPolicy:
        return self.policy

    # ------------------------------------------------------------------ #
    # Metrics collection                                                 #
    # ------------------------------------------------------------------ #

    def _collect_metrics(self) -> ScalingMetrics:
        """Collect current metrics from load balancer and system."""
        metrics = ScalingMetrics()
        metrics.active_replicas = self._current_replicas

        # Try to get metrics from load balancer
        if self.load_balancer:
            try:
                lb_status = self.load_balancer.get_status()
                # Aggregate backend metrics
                total_queue = 0
                total_latency = 0.0
                total_errors = 0
                total_requests = 0
                healthy_count = 0

                for backend_name, backend in lb_status.get("backends", {}).items():
                    if backend.get("healthy"):
                        healthy_count += 1
                    total_queue += backend.get("active_slots", 0)
                    total_latency += backend.get("avg_latency_ms", 0)
                    total_errors += backend.get("total_errors", 0)
                    total_requests += backend.get("total_requests", 0)

                metrics.queue_depth = total_queue
                if healthy_count > 0:
                    metrics.latency_p95_ms = total_latency / healthy_count
                if total_requests > 0:
                    metrics.error_rate = total_errors / total_requests
                    metrics.requests_per_second = total_requests / 60.0  # rough estimate
            except Exception:
                pass

        # Try to get system metrics
        try:
            import psutil

            metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics.memory_percent = psutil.virtual_memory().percent
        except Exception:
            pass

        return metrics

    # ------------------------------------------------------------------ #
    # Scaling evaluation                                                 #
    # ------------------------------------------------------------------ #

    def _can_scale_up(self) -> bool:
        if self._current_replicas >= self.policy.max_replicas:
            return False
        if self._last_scale_up is None:
            return True
        return (time.time() - self._last_scale_up) >= self.policy.scale_up_cooldown_seconds

    def _can_scale_down(self) -> bool:
        if self._current_replicas <= self.policy.min_replicas:
            return False
        if self._last_scale_down is None:
            return True
        return (time.time() - self._last_scale_down) >= self.policy.scale_down_cooldown_seconds

    def evaluate(self) -> ScalingDecision:
        """Evaluate current metrics and make a scaling decision."""
        metrics = self._collect_metrics()
        reasons: List[str] = []

        # Check scale-up conditions
        scale_up = False
        if metrics.cpu_percent > self.policy.target_cpu_percent:
            reasons.append(f"CPU {metrics.cpu_percent:.1f}% > {self.policy.target_cpu_percent}%")
            scale_up = True
        if metrics.memory_percent > self.policy.target_memory_percent:
            reasons.append(f"Memory {metrics.memory_percent:.1f}% > {self.policy.target_memory_percent}%")
            scale_up = True
        if metrics.queue_depth > self.policy.target_queue_depth:
            reasons.append(f"Queue depth {metrics.queue_depth} > {self.policy.target_queue_depth}")
            scale_up = True
        if metrics.latency_p95_ms > self.policy.target_latency_p95_ms:
            reasons.append(f"P95 latency {metrics.latency_p95_ms:.0f}ms > {self.policy.target_latency_p95_ms}ms")
            scale_up = True
        if metrics.error_rate > self.policy.max_error_rate:
            reasons.append(f"Error rate {metrics.error_rate:.1%} > {self.policy.max_error_rate:.1%}")
            scale_up = True

        # Check scale-down conditions (only if no scale-up needed)
        scale_down = False
        if not scale_up:
            if (
                metrics.cpu_percent < self.policy.target_cpu_percent * 0.5
                and metrics.memory_percent < self.policy.target_memory_percent * 0.5
                and metrics.queue_depth < self.policy.target_queue_depth * 0.3
                and metrics.latency_p95_ms < self.policy.target_latency_p95_ms * 0.5
                and metrics.error_rate < self.policy.max_error_rate * 0.5
            ):
                scale_down = True
                reasons.append("All metrics well below thresholds")

        # Make decision
        if scale_up and self._can_scale_up():
            target = min(self._current_replicas + 1, self.policy.max_replicas)
            action = ScalingAction.SCALE_UP
            self._current_replicas = target
            self._last_scale_up = time.time()
            reason = "; ".join(reasons)
        elif scale_down and self._can_scale_down():
            target = max(self._current_replicas - 1, self.policy.min_replicas)
            action = ScalingAction.SCALE_DOWN
            self._current_replicas = target
            self._last_scale_down = time.time()
            reason = "; ".join(reasons)
        else:
            target = self._current_replicas
            action = ScalingAction.NO_CHANGE
            reason = "Metrics within acceptable range" if not reasons else "; ".join(reasons)

        decision = ScalingDecision(
            action=action,
            current_replicas=self._current_replicas,
            target_replicas=target,
            reason=reason,
            metrics=metrics,
        )

        self._scaling_history.append(decision)
        # Keep only last 1000 decisions
        if len(self._scaling_history) > 1000:
            self._scaling_history = self._scaling_history[-1000:]

        self._save()
        return decision

    def scale_up(self) -> int:
        """Manually scale up by one replica."""
        if self._current_replicas >= self.policy.max_replicas:
            return self._current_replicas
        self._current_replicas += 1
        self._last_scale_up = time.time()
        self._record_manual_scale(ScalingAction.SCALE_UP, "manual scale-up")
        self._save()
        return self._current_replicas

    def scale_down(self) -> int:
        """Manually scale down by one replica."""
        if self._current_replicas <= self.policy.min_replicas:
            return self._current_replicas
        self._current_replicas -= 1
        self._last_scale_down = time.time()
        self._record_manual_scale(ScalingAction.SCALE_DOWN, "manual scale-down")
        self._save()
        return self._current_replicas

    def _record_manual_scale(self, action: ScalingAction, reason: str) -> None:
        """Record a manual scaling operation in history."""
        metrics = ScalingMetrics(active_replicas=self._current_replicas)
        decision = ScalingDecision(
            action=action,
            current_replicas=self._current_replicas,
            target_replicas=self._current_replicas,
            reason=reason,
            metrics=metrics,
        )
        self._scaling_history.append(decision)
        if len(self._scaling_history) > 1000:
            self._scaling_history = self._scaling_history[-1000:]

    # ------------------------------------------------------------------ #
    # Queries                                                            #
    # ------------------------------------------------------------------ #

    def get_status(self) -> Dict[str, Any]:
        return {
            "current_replicas": self._current_replicas,
            "min_replicas": self.policy.min_replicas,
            "max_replicas": self.policy.max_replicas,
            "last_scale_up": self._last_scale_up,
            "last_scale_down": self._last_scale_down,
            "policy": self.policy.to_dict(),
        }

    def get_scaling_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [d.to_dict() for d in self._scaling_history[-limit:]]


# --------------------------------------------------------------------------- #
# Global instance                                                             #
# --------------------------------------------------------------------------- #

_auto_scaler: Optional[AutoScaler] = None


def get_auto_scaler() -> AutoScaler:
    global _auto_scaler
    if _auto_scaler is None:
        _auto_scaler = AutoScaler()
    return _auto_scaler


# --------------------------------------------------------------------------- #
# CLI entry point                                                             #
# --------------------------------------------------------------------------- #

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LFM3-1.7B Auto-Scaler")
    subparsers = parser.add_subparsers(dest="command")

    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate scaling metrics")
    eval_parser.set_defaults(func=cmd_evaluate)

    # status command
    status_parser = subparsers.add_parser("status", help="Show auto-scaler status")
    status_parser.set_defaults(func=cmd_status)

    # policy command
    policy_parser = subparsers.add_parser("policy", help="Set scaling policy")
    policy_parser.add_argument("--min-replicas", type=int, default=None)
    policy_parser.add_argument("--max-replicas", type=int, default=None)
    policy_parser.add_argument("--target-cpu", type=float, default=None)
    policy_parser.add_argument("--target-memory", type=float, default=None)
    policy_parser.add_argument("--target-queue", type=int, default=None)
    policy_parser.add_argument("--target-latency", type=float, default=None)
    policy_parser.add_argument("--max-error-rate", type=float, default=None)
    policy_parser.set_defaults(func=cmd_policy)

    # scale-up command
    up_parser = subparsers.add_parser("scale-up", help="Manually scale up")
    up_parser.set_defaults(func=cmd_scale_up)

    # scale-down command
    down_parser = subparsers.add_parser("scale-down", help="Manually scale down")
    down_parser.set_defaults(func=cmd_scale_down)

    # history command
    history_parser = subparsers.add_parser("history", help="Show scaling history")
    history_parser.add_argument("--limit", type=int, default=50, help="Number of entries")
    history_parser.set_defaults(func=cmd_history)

    args = parser.parse_args()
    scaler = AutoScaler()

    if args.command == "evaluate":
        decision = scaler.evaluate()
        print(f"Scaling decision: {decision.action.value}")
        print(f"  Current replicas: {decision.current_replicas}")
        print(f"  Target replicas: {decision.target_replicas}")
        print(f"  Reason: {decision.reason}")
        print(f"  Metrics:")
        print(f"    CPU: {decision.metrics.cpu_percent:.1f}%")
        print(f"    Memory: {decision.metrics.memory_percent:.1f}%")
        print(f"    Queue depth: {decision.metrics.queue_depth}")
        print(f"    P95 latency: {decision.metrics.latency_p95_ms:.0f}ms")
        print(f"    Error rate: {decision.metrics.error_rate:.1%}")
        return 0

    elif args.command == "status":
        status = scaler.get_status()
        print(f"Auto-Scaler Status:")
        print(f"  Current replicas: {status['current_replicas']}")
        print(f"  Min replicas: {status['min_replicas']}")
        print(f"  Max replicas: {status['max_replicas']}")
        print(f"  Policy:")
        print(f"    Target CPU: {status['policy']['target_cpu_percent']}%")
        print(f"    Target Memory: {status['policy']['target_memory_percent']}%")
        print(f"    Target queue depth: {status['policy']['target_queue_depth']}")
        print(f"    Target P95 latency: {status['policy']['target_latency_p95_ms']}ms")
        print(f"    Max error rate: {status['policy']['max_error_rate']:.1%}")
        if status["last_scale_up"]:
            print(f"  Last scale up: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(status['last_scale_up']))}")
        if status["last_scale_down"]:
            print(f"  Last scale down: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(status['last_scale_down']))}")
        return 0

    elif args.command == "policy":
        policy = scaler.get_policy()
        if args.min_replicas is not None:
            policy.min_replicas = args.min_replicas
        if args.max_replicas is not None:
            policy.max_replicas = args.max_replicas
        if args.target_cpu is not None:
            policy.target_cpu_percent = args.target_cpu
        if args.target_memory is not None:
            policy.target_memory_percent = args.target_memory
        if args.target_queue is not None:
            policy.target_queue_depth = args.target_queue
        if args.target_latency is not None:
            policy.target_latency_p95_ms = args.target_latency
        if args.max_error_rate is not None:
            policy.max_error_rate = args.max_error_rate
        scaler.set_policy(policy)
        print(f"Scaling policy updated:")
        print(f"  Min replicas: {policy.min_replicas}")
        print(f"  Max replicas: {policy.max_replicas}")
        print(f"  Target CPU: {policy.target_cpu_percent}%")
        print(f"  Target Memory: {policy.target_memory_percent}%")
        print(f"  Target queue depth: {policy.target_queue_depth}")
        print(f"  Target P95 latency: {policy.target_latency_p95_ms}ms")
        print(f"  Max error rate: {policy.max_error_rate:.1%}")
        return 0

    elif args.command == "scale-up":
        replicas = scaler.scale_up()
        print(f"Scaled up to {replicas} replicas")
        return 0

    elif args.command == "scale-down":
        replicas = scaler.scale_down()
        print(f"Scaled down to {replicas} replicas")
        return 0

    elif args.command == "history":
        history = scaler.get_scaling_history(args.limit)
        if not history:
            print("No scaling history")
            return 0
        print(f"Scaling history ({len(history)} entries):")
        for d in history:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(d["timestamp"]))
            print(f"  {ts}  {d['action']}  {d['current_replicas']}->{d['target_replicas']}  {d['reason']}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
