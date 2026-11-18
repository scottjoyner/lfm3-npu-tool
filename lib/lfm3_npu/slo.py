"""
Service-Level Objective (SLO) and error-budget tracking for LFM3-1.7B
inference servers.

Tracks per-endpoint availability and latency SLOs, computes error-budget
remaining, and exposes Prometheus metrics plus a /slo HTTP endpoint.

SLO targets (configurable):
  * availability:  percentage of successful requests (default 99.9%)
  * latency_p95:   p95 request latency threshold in ms (default 5000ms)

Error budget = 1 - (actual availability / target availability).
When error budget is exhausted, alerts should fire (see prometheus-rules).

All metrics are Prometheus-compatible and are consumed by the metrics
exporter (scripts/npu_metrics_exporter.py).
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional


@dataclass
class SLOTarget:
    """SLO target definition."""

    availability: float = 99.9          # percent
    latency_p95_ms: float = 5000.0      # milliseconds
    window_seconds: int = 3600          # evaluation window (1 hour)


@dataclass
class _WindowBucket:
    """Rolling time-window bucket for request outcomes."""

    start: float
    success_count: int = 0
    error_count: int = 0
    latencies: deque = field(default_factory=deque)

    def record(self, success: bool, latency_ms: float) -> None:
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        self.latencies.append(latency_ms)

    def total(self) -> int:
        return self.success_count + self.error_count

    def availability(self) -> float:
        total = self.total()
        if total == 0:
            return 100.0
        return 100.0 * self.success_count / total

    def p95_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(0.95 * (len(sorted_lat) - 1))
        return sorted_lat[idx]


class SLOTracker:
    """
    Rolling-window SLO tracker.

    Maintains a deque of per-minute buckets over the evaluation window.
    Thread-safe.
    """

    def __init__(self, target: Optional[SLOTarget] = None) -> None:
        self.target = target or SLOTarget()
        self._buckets: Deque[_WindowBucket] = deque()
        self._lock = threading.Lock()
        self._current: Optional[_WindowBucket] = None
        # Cumulative counters (for Prometheus counters)
        self._total_requests = 0
        self._total_errors = 0
        self._total_success = 0

    def _current_bucket(self, now: float) -> _WindowBucket:
        if self._current is None or now - self._current.start >= 60.0:
            self._current = _WindowBucket(start=now - (now % 60.0))
            self._buckets.append(self._current)
            self._evict_old(now)
        return self._current

    def _evict_old(self, now: float) -> None:
        cutoff = now - self.target.window_seconds
        while self._buckets and self._buckets[0].start < cutoff:
            self._buckets.popleft()

    def record(self, success: bool, latency_ms: float, endpoint: str = "") -> None:
        """Record a request outcome."""
        now = time.monotonic()
        with self._lock:
            bucket = self._current_bucket(now)
            bucket.record(success, latency_ms)
            self._total_requests += 1
            if success:
                self._total_success += 1
            else:
                self._total_errors += 1

    def snapshot(self) -> Dict:
        """Return current SLO evaluation snapshot."""
        now = time.monotonic()
        with self._lock:
            self._evict_old(now)
            total_success = sum(b.success_count for b in self._buckets)
            total_error = sum(b.error_count for b in self._buckets)
            total_requests = total_success + total_error
            all_latencies = []
            for b in self._buckets:
                all_latencies.extend(b.latencies)

        availability = 100.0 * total_success / total_requests if total_requests else 100.0
        all_latencies.sort()
        p95 = all_latencies[int(0.95 * (len(all_latencies) - 1))] if all_latencies else 0.0

        target_avail = self.target.availability
        error_budget_used = max(0.0, (target_avail - availability) / target_avail) * 100.0
        error_budget_remaining = max(0.0, 100.0 - error_budget_used)

        latency_slo_met = p95 <= self.target.latency_p95_ms
        availability_slo_met = availability >= target_avail

        return {
            "window_seconds": self.target.window_seconds,
            "total_requests": total_requests,
            "total_success": total_success,
            "total_errors": total_error,
            "availability_percent": round(availability, 4),
            "availability_target_percent": target_avail,
            "availability_slo_met": availability_slo_met,
            "p95_latency_ms": round(p95, 2),
            "latency_p95_target_ms": self.target.latency_p95_ms,
            "latency_slo_met": latency_slo_met,
            "error_budget_used_percent": round(error_budget_used, 4),
            "error_budget_remaining_percent": round(error_budget_remaining, 4),
            "error_budget_exhausted": error_budget_remaining <= 0.0,
            "slo_met": availability_slo_met and latency_slo_met,
        }

    def prometheus_metrics(self, provider: str, model: str) -> str:
        """Render Prometheus text-format metrics."""
        snap = self.snapshot()
        labels = f'provider="{provider}",model="{model}"'
        lines = [
            "# HELP lfm3_slo_availability_percent Current availability percentage over SLO window",
            "# TYPE lfm3_slo_availability_percent gauge",
            f'lfm3_slo_availability_percent{{{labels}}} {snap["availability_percent"]}',
            "# HELP lfm3_slo_availability_target_percent Target availability percentage",
            "# TYPE lfm3_slo_availability_target_percent gauge",
            f'lfm3_slo_availability_target_percent{{{labels}}} {snap["availability_target_percent"]}',
            "# HELP lfm3_slo_p95_latency_ms p95 latency in milliseconds over SLO window",
            "# TYPE lfm3_slo_p95_latency_ms gauge",
            f'lfm3_slo_p95_latency_ms{{{labels}}} {snap["p95_latency_ms"]}',
            "# HELP lfm3_slo_latency_target_ms Target p95 latency in milliseconds",
            "# TYPE lfm3_slo_latency_target_ms gauge",
            f'lfm3_slo_latency_target_ms{{{labels}}} {snap["latency_p95_target_ms"]}',
            "# HELP lfm3_slo_error_budget_remaining_percent Error budget remaining percentage",
            "# TYPE lfm3_slo_error_budget_remaining_percent gauge",
            f'lfm3_slo_error_budget_remaining_percent{{{labels}}} {snap["error_budget_remaining_percent"]}',
            "# HELP lfm3_slo_error_budget_exhausted 1 if error budget is exhausted",
            "# TYPE lfm3_slo_error_budget_exhausted gauge",
            f'lfm3_slo_error_budget_exhausted{{{labels}}} {1 if snap["error_budget_exhausted"] else 0}',
            "# HELP lfm3_slo_slo_met 1 if both availability and latency SLOs are met",
            "# TYPE lfm3_slo_slo_met gauge",
            f'lfm3_slo_slo_met{{{labels}}} {1 if snap["slo_met"] else 0}',
            "# HELP lfm3_slo_total_requests Total requests in SLO window",
            "# TYPE lfm3_slo_total_requests counter",
            f'lfm3_slo_total_requests{{{labels}}} {snap["total_requests"]}',
            "# HELP lfm3_slo_total_errors Total errors in SLO window",
            "# TYPE lfm3_slo_total_errors counter",
            f'lfm3_slo_total_errors{{{labels}}} {snap["total_errors"]}',
        ]
        return "\n".join(lines) + "\n"


# Global trackers (one per provider)
_trackers: Dict[str, SLOTracker] = {}
_trackers_lock = threading.Lock()


def get_tracker(provider: str, model: str) -> SLOTracker:
    """Get or create an SLO tracker for a provider/model pair."""
    key = f"{provider}:{model}"
    with _trackers_lock:
        if key not in _trackers:
            _trackers[key] = SLOTracker()
        return _trackers[key]


def record_request(provider: str, model: str, success: bool,
                   latency_ms: float, endpoint: str = "") -> None:
    """Convenience: record a request outcome to the appropriate tracker."""
    get_tracker(provider, model).record(success, latency_ms, endpoint)


def prometheus_metrics_all() -> str:
    """Render Prometheus metrics for all trackers."""
    with _trackers_lock:
        trackers = list(_trackers.items())
    parts = []
    for key, tracker in trackers:
        provider, model = key.split(":", 1)
        parts.append(tracker.prometheus_metrics(provider, model))
    return "\n".join(parts)
