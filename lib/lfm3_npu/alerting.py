"""
Real-time alerting for LFM3-1.7B inference servers.

Provides:
  * Slack webhook notifications
  * PagerDuty event integration
  * Generic webhook alerts
  * Alert severity levels (info, warning, critical)
  * Alert deduplication and rate limiting
  * Integration with SLO error budgets, circuit breaker, and watchdog

Usage:
    from lfm3_npu.alerting import AlertManager, AlertSeverity

    alerts = AlertManager()
    alerts.send_critical("CPU server down on port 1238")
    alerts.send_warning("CPU quality degradation detected")
    alerts.send_info("NPU server restarted successfully")

CLI:
    lfm3-npu alert test --webhook-url https://hooks.slack.com/services/...
    lfm3-npu alert send --severity critical --message "Server down"
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import urllib.request
import urllib.error

# Alert severity levels
ALERT_INFO = "info"
ALERT_WARNING = "warning"
ALERT_CRITICAL = "critical"

SEVERITY_EMOJI = {
    ALERT_INFO: ":information_source:",
    ALERT_WARNING: ":warning:",
    ALERT_CRITICAL: ":rotating_light:",
}

SEVERITY_COLOR = {
    ALERT_INFO: "#36a64f",
    ALERT_WARNING: "#ffcc00",
    ALERT_CRITICAL: "#ff0000",
}


@dataclass
class Alert:
    """An alert notification."""

    severity: str
    message: str
    source: str = "lfm3-inference"
    timestamp: float = field(default_factory=time.time)
    details: Optional[Dict[str, Any]] = None
    alert_id: Optional[str] = None

    def __post_init__(self):
        if self.alert_id is None:
            import hashlib

            self.alert_id = hashlib.sha256(
                f"{self.severity}:{self.message}:{self.source}:{int(self.timestamp)}".encode()
            ).hexdigest()[:12]

    def to_slack_payload(self) -> Dict[str, Any]:
        """Convert to Slack webhook payload."""
        emoji = SEVERITY_EMOJI.get(self.severity, ":bell:")
        color = SEVERITY_COLOR.get(self.severity, "#cccccc")

        fields = []
        if self.details:
            for k, v in self.details.items():
                fields.append({"title": k, "value": str(v), "short": True})

        return {
            "username": "LFM3-1.7B Alerts",
            "icon_emoji": ":robot_face:",
            "attachments": [
                {
                    "color": color,
                    "title": f"{emoji} {self.severity.upper()}: {self.message}",
                    "fields": fields,
                    "footer": f"{self.source} • {self.alert_id}",
                    "ts": int(self.timestamp),
                }
            ],
        }

    def to_pagerduty_payload(self) -> Dict[str, Any]:
        """Convert to PagerDuty Events API v2 payload."""
        event_action = "trigger"
        if self.severity == ALERT_INFO:
            event_action = "trigger"  # PagerDuty doesn't have info, use trigger
        elif self.severity == ALERT_CRITICAL:
            event_action = "trigger"

        return {
            "routing_key": os.environ.get("PAGERDUTY_ROUTING_KEY", ""),
            "event_action": event_action,
            "dedup_key": self.alert_id,
            "payload": {
                "summary": f"{self.severity.upper()}: {self.message}",
                "source": self.source,
                "severity": self.severity,
                "details": self.details or {},
                "timestamp": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp)
                ),
                "group": "lfm3-inference",
                "class": "inference-server",
                "component": self.source,
            },
        }


class AlertManager:
    """Manages alert delivery via webhooks."""

    def __init__(
        self,
        slack_webhook_url: Optional[str] = None,
        pagerduty_routing_key: Optional[str] = None,
        webhook_urls: Optional[List[str]] = None,
        dedup_window_seconds: int = 300,
    ):
        self.slack_webhook_url = slack_webhook_url or os.environ.get(
            "LFM3_SLACK_WEBHOOK_URL", ""
        )
        self.pagerduty_routing_key = pagerduty_routing_key or os.environ.get(
            "PAGERDUTY_ROUTING_KEY", ""
        )
        self.webhook_urls = webhook_urls or []
        self.dedup_window = dedup_window_seconds
        self._sent_alerts: Dict[str, float] = {}  # alert_id -> timestamp

    # ------------------------------------------------------------------ #
    # Core sending                                                       #
    # ------------------------------------------------------------------ #

    def _should_dedup(self, alert: Alert) -> bool:
        """Check if this alert was recently sent (deduplication)."""
        if alert.alert_id in self._sent_alerts:
            last_sent = self._sent_alerts[alert.alert_id]
            if time.time() - last_sent < self.dedup_window:
                return True
        return False

    def _mark_sent(self, alert: Alert) -> None:
        """Record that an alert was sent."""
        self._sent_alerts[alert.alert_id] = time.time()
        # Clean old entries
        cutoff = time.time() - self.dedup_window * 2
        self._sent_alerts = {
            k: v for k, v in self._sent_alerts.items() if v > cutoff
        }

    def _send_webhook(
        self, url: str, payload: Dict[str, Any], timeout: float = 10.0
    ) -> bool:
        """Send a webhook payload. Returns True on success."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp.read()
            return True
        except Exception as e:  # noqa BLE001
            print(f"[Alert] Webhook failed ({url}): {e}", flush=True)
            return False

    def send(self, alert: Alert) -> Dict[str, bool]:
        """Send an alert through all configured channels.

        Returns a dict of channel -> success status.
        """
        results: Dict[str, bool] = {}

        if self._should_dedup(alert):
            print(f"[Alert] Deduped: {alert.alert_id}", flush=True)
            return results

        # Slack
        if self.slack_webhook_url:
            results["slack"] = self._send_webhook(
                self.slack_webhook_url, alert.to_slack_payload()
            )

        # PagerDuty (only for warning/critical)
        if self.pagerduty_routing_key and alert.severity in (
            ALERT_WARNING,
            ALERT_CRITICAL,
        ):
            results["pagerduty"] = self._send_webhook(
                "https://events.pagerduty.com/v2/enqueue",
                alert.to_pagerduty_payload(),
            )

        # Generic webhooks
        for i, url in enumerate(self.webhook_urls):
            results[f"webhook_{i}"] = self._send_webhook(url, alert.to_slack_payload())

        self._mark_sent(alert)
        return results

    # ------------------------------------------------------------------ #
    # Convenience methods                                                #
    # ------------------------------------------------------------------ #

    def send_info(
        self,
        message: str,
        source: str = "lfm3-inference",
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, bool]:
        """Send an info alert."""
        return self.send(
            Alert(severity=ALERT_INFO, message=message, source=source, details=details)
        )

    def send_warning(
        self,
        message: str,
        source: str = "lfm3-inference",
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, bool]:
        """Send a warning alert."""
        return self.send(
            Alert(
                severity=ALERT_WARNING, message=message, source=source, details=details
            )
        )

    def send_critical(
        self,
        message: str,
        source: str = "lfm3-inference",
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, bool]:
        """Send a critical alert."""
        return self.send(
            Alert(
                severity=ALERT_CRITICAL,
                message=message,
                source=source,
                details=details,
            )
        )


# --------------------------------------------------------------------------- #
# Global instance                                                             #
# --------------------------------------------------------------------------- #

_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


# --------------------------------------------------------------------------- #
# CLI entry point                                                             #
# --------------------------------------------------------------------------- #

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LFM3-1.7B Alert Manager")
    parser.add_argument("--test", action="store_true", help="Send a test alert")
    parser.add_argument(
        "--severity",
        choices=["info", "warning", "critical"],
        default="info",
        help="Alert severity",
    )
    parser.add_argument("--message", default="Test alert from LFM3-1.7B", help="Alert message")
    parser.add_argument("--webhook-url", default=None, help="Slack webhook URL")
    parser.add_argument("--source", default="lfm3-inference", help="Alert source")

    args = parser.parse_args()

    alerts = AlertManager(slack_webhook_url=args.webhook_url)

    if args.test or args.message:
        alert = Alert(
            severity=args.severity,
            message=args.message,
            source=args.source,
        )
        results = alerts.send(alert)
        print(f"Alert sent: {results}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
