"""
SLO Alerting for LFM3-1.7B inference servers.

Provides:
  * SLO definition and tracking
  * Error budget calculation and tracking
  * Alerting when SLOs are breached
  * Integration with alerting webhooks
  * CLI integration for SLO management
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SLOTarget:
    """An SLO target definition."""

    name: str
    description: str
    target_value: float
    unit: str
    direction: str
    window_seconds: int = 3600


@dataclass
class SLOResult:
    """Result of an SLO check."""

    test_id: str
    timestamp: float
    provider: str
    model: str
    status: str
    uptime_seconds: Optional[float] = None
    active_slots: int = 0
    free_slots: int = 0
    avg_decode_ms: float = 0.0
    tls: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SLOAlert:
    """An SLO alert."""

    alert_id: str
    timestamp: float
    slo_name: str
    severity: str
    message: str
    current_value: float
    target_value: float
    unit: str
    error_budget_remaining: float


class SLOTracker:
    """Tracks SLOs and generates alerts when SLOs are breached."""

    def __init__(
        self,
        storage_path: Optional[str] = None,
        alert_webhook: Optional[str] = None,
    ):
        self.storage_path = Path(
            storage_path or os.environ.get(
                "LFM3_SLO_STORAGE",
                str(Path.home() / ".config" / "lfm3-npu" / "slo_data.json"),
            )
        )
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.alert_webhook = alert_webhook or os.environ.get("LFM3_SLO_WEBHOOK")

        self.slo_targets: Dict[str, SLOTarget] = {}
        self._load_targets()

        self.slo_results: List[SLOResult] = []
        self._load_results()

    def _load_targets(self) -> None:
        try:
            import json
            if self.storage_path.exists():
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                for name, tdata in data.get("targets", {}).items():
                    self.slo_targets[tdata["name"]] = SLOTarget(**tdata)
        except Exception:
            pass

    def _save_targets(self) -> None:
        try:
            import json
            data = {"targets": [asdict(t) for t in self.slo_targets.values()]}
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_results(self) -> None:
        try:
            import json
            results_path = Path(str(self.storage_path) + ".results")
            if results_path.exists():
                with open(results_path, "r") as f:
                    data = json.load(f)
                self.slo_results = [SLOResult(**r) for r in data]
        except Exception:
            pass

    def _save_results(self) -> None:
        try:
            import json
            results_path = Path(str(self.storage_path) + ".results")
            data = {"results": [asdict(r) for r in self.slo_results]}
            with open(results_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def add_slo_target(self, target: SLOTarget) -> None:
        self.slo_targets[target.name] = target
        self._save_targets()

    def check_slo(self, name: str, current_value: float) -> SLOResult:
        target = self.slo_targets.get(name)
        if target is None:
            return SLOResult(
                name=name,
                timestamp=time.time(),
                provider="lfm3-npu",
                model="lfm3-1.7b",
                status="unknown",
                current_value=current_value,
                target_value=0.0,
                unit="unknown",
                slo_met=False,
                error_budget_remaining=0.0,
                error_budget_exhausted=True,
                reason="SLO target not found",
            )

        if target.direction == "higher_is_better":
            slo_met = current_value >= target.target_value
        else:
            slo_met = current_value <= target.target_value

        if target.target_value != 0:
            if target.direction == "higher_is_better":
                error_budget_used = max(0.0, (target.target_value - current_value) / target.target_value)
            else:
                error_budget_used = max(0.0, (current_value - target.target_value) / target.target_value)
        else:
            error_budget_used = 0.0

        error_budget_remaining = max(0.0, 1.0 - error_budget_used)
        error_budget_exhausted = error_budget_remaining <= 0.0

        result = SLOResult(
            test_id=name,
            timestamp=time.time(),
            provider="lfm3-npu",
            model="lfm3-1.7b",
            status="healthy" if slo_met else "degraded",
            current_value=current_value,
            target_value=target.target_value,
            unit=target.unit,
            slo_met=slo_met,
            error_budget_remaining=error_budget_remaining,
            error_budget_exhausted=error_budget_exhausted,
            reason="SLO met" if slo_met else "SLO breached",
        )

        self.slo_results.append(result)
        self._save_results()
        return result

    def check_slo_with_alert(
        self, name: str, current_value: float, webhook_url: Optional[str] = None
    ) -> SLOResult:
        result = self.check_slo(name, current_value)
        if not result.slo_met:
            self._send_alert(result, webhook_url or self.alert_webhook)
        return result

    def _send_alert(
        self,
        result: SLOResult,
        webhook_url: Optional[str] = None,
    ) -> None:
        webhook_url = webhook_url or self.alert_webhook
        if not webhook_url:
            return

        import urllib.request
        import json as json_module

        payload = {
            "alert_id": result.test_id,
            "timestamp": result.timestamp,
            "slo_name": result.name,
            "severity": "critical" if result.error_budget_exhausted else "warning",
            "message": result.reason,
            "current_value": result.current_value,
            "target_value": result.target_value,
            "unit": result.unit,
            "error_budget_remaining": result.error_budget_remaining,
        }

        try:
            data = json_module.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status >= 400:
                    print(f"Alert webhook returned status {resp.status}")
        except Exception as e:
            print(f"Failed to send alert webhook: {e}")

    def get_slo_status(self, name: str) -> Optional[SLOResult]:
        for result in reversed(self.slo_results):
            if result.name == name:
                return result
        return None

    def get_all_slo_status(self) -> List[SLOResult]:
        return self.slo_results


if __name__ == "__main__":
    tracker = SLOTracker()

    tracker.slo_targets["availability"] = SLOTarget(
        name="availability",
        description="Service availability percentage",
        target_value=99.9,
        unit="percent",
        direction="higher_is_better",
    )
    tracker.slo_targets["latency_p95_ms"] = SLOTarget(
        name="latency_p95_ms",
        description="P95 latency in milliseconds",
        target_value=5000.0,
        unit="ms",
        direction="lower_is_better",
    )
    tracker.slo_targets["error_rate"] = SLOTarget(
        name="error_rate",
        description="Error rate percentage",
        target_value=1.0,
        unit="percent",
        direction="lower_is_better",
    )

    tracker.check_slo("availability", 99.5)
    tracker.check_slo("latency_p95_ms", 5500.0)
    tracker.check_slo("error_rate", 2.0)

    for result in tracker.slo_results:
        status = "MET" if result.slo_met else "BREACHED"
        print(f"{result.name}: {status} - {result.error_budget_remaining:.1%} error budget remaining")
