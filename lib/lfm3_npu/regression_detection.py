"""
Performance regression detection for LFM3-1.7B inference servers.

Provides:
  * Baseline benchmark storage and comparison
  * Regression detection for TTFT, decode speed, p95 latency, error rate
  * Configurable thresholds (percentage change)
  * Regression report generation (JSON/text)
  * CLI integration for running regression checks
  * Integration with benchmark results and CI/CD

Usage:
    from lfm3_npu.regression_detection import RegressionDetector, RegressionResult

    detector = RegressionDetector()
    result = detector.check_regression("npu", current_metrics)
    if result.regression_detected:
        print(f"Regression: {result.reason}")

CLI:
    lfm3-npu regress check --server npu --baseline-file baseline.json
    lfm3-npu regress baseline --server npu --output baseline.json
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default regression thresholds (percentage change that triggers alert)
DEFAULT_THRESHOLDS = {
    "ttft_ms": 20.0,        # 20% increase in time-to-first-token
    "decode_tps": 15.0,     # 15% decrease in decode tokens/sec
    "p95_latency_ms": 20.0, # 20% increase in p95 latency
    "error_rate": 50.0,     # 50% increase in error rate
    "availability": 5.0,    # 5% decrease in availability
}


@dataclass
class RegressionResult:
    """Result of a regression check."""

    server: str
    metric: str
    baseline_value: float
    current_value: float
    change_percent: float
    threshold_percent: float
    regression_detected: bool
    severity: str  # "info", "warning", "critical"

    @property
    def direction(self) -> str:
        """Whether the metric went up or down."""
        if self.change_percent > 0:
            return "increased"
        return "decreased"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "server": self.server,
            "metric": self.metric,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "change_percent": round(self.change_percent, 2),
            "threshold_percent": self.threshold_percent,
            "regression_detected": self.regression_detected,
            "severity": self.severity,
            "direction": self.direction,
        }


class RegressionDetector:
    """Detects performance regressions by comparing current metrics to baseline."""

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        baseline_path: Optional[str] = None,
    ):
        self.thresholds = thresholds or dict(DEFAULT_THRESHOLDS)
        if baseline_path is None:
            baseline_path = os.environ.get(
                "LFM3_REGRESSION_BASELINE",
                str(Path.home() / ".config" / "lfm3-npu" / "baseline.json"),
            )
        self.baseline_path = Path(baseline_path)
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        self._baselines: Dict[str, Dict[str, float]] = {}
        self._load_baselines()

    # ------------------------------------------------------------------ #
    # Baseline persistence                                               #
    # ------------------------------------------------------------------ #

    def _load_baselines(self) -> None:
        """Load baselines from disk."""
        if not self.baseline_path.exists():
            return
        try:
            import json as _json

            with open(self.baseline_path, "r") as f:
                data = _json.load(f)
            self._baselines = data
        except (_json.JSONDecodeError, OSError) as e:
            print(f"[Regression] Warning: could not load baseline: {e}", flush=True)

    def _save_baselines(self) -> None:
        """Save baselines to disk."""
        import json as _json

        tmp = self.baseline_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            _json.dump(self._baselines, f, indent=2)
        tmp.replace(self.baseline_path)

    # ------------------------------------------------------------------ #
    # Baseline management                                                #
    # ------------------------------------------------------------------ #

    def set_baseline(self, server: str, metrics: Dict[str, float]) -> None:
        """Set or update the baseline for a server."""
        self._baselines[server] = dict(metrics)
        self._save_baselines()

    def get_baseline(self, server: str) -> Optional[Dict[str, float]]:
        """Get the baseline for a server."""
        return self._baselines.get(server)

    def list_baselines(self) -> List[str]:
        """List all servers with baselines."""
        return list(self._baselines.keys())

    def delete_baseline(self, server: str) -> bool:
        """Delete a baseline."""
        if server in self._baselines:
            del self._baselines[server]
            self._save_baselines()
            return True
        return False

    # ------------------------------------------------------------------ #
    # Regression detection                                               #
    # ------------------------------------------------------------------ #

    def _metric_direction(self, metric: str) -> str:
        """Return 'higher_is_better' or 'lower_is_better' for a metric."""
        higher_is_better = {"decode_tps", "availability"}
        return "higher_is_better" if metric in higher_is_better else "lower_is_better"

    def check_regression(
        self,
        server: str,
        current_metrics: Dict[str, float],
    ) -> List[RegressionResult]:
        """Check current metrics against baseline for regressions.

        Returns a list of RegressionResult for metrics that exceed thresholds.
        """
        baseline = self._baselines.get(server)
        if baseline is None:
            return []

        results: List[RegressionResult] = []

        for metric, threshold in self.thresholds.items():
            if metric not in baseline or metric not in current_metrics:
                continue

            baseline_val = baseline[metric]
            current_val = current_metrics[metric]

            if baseline_val == 0:
                continue

            change_percent = ((current_val - baseline_val) / baseline_val) * 100

            direction = self._metric_direction(metric)

            # Determine if this is a regression
            is_regression = False
            if direction == "lower_is_better":
                # Metric increased (bad) beyond threshold
                is_regression = change_percent > threshold
            else:
                # Metric decreased (bad) beyond threshold
                is_regression = change_percent < -threshold

            # Determine severity
            if not is_regression:
                continue

            if abs(change_percent) >= threshold * 2:
                severity = "critical"
            elif abs(change_percent) >= threshold * 1.5:
                severity = "warning"
            else:
                severity = "info"

            results.append(
                RegressionResult(
                    server=server,
                    metric=metric,
                    baseline_value=baseline_val,
                    current_value=current_val,
                    change_percent=change_percent,
                    threshold_percent=threshold,
                    regression_detected=True,
                    severity=severity,
                )
            )

        return results

    def check_all_servers(
        self,
        current_metrics: Dict[str, Dict[str, float]],
    ) -> List[RegressionResult]:
        """Check all servers against their baselines."""
        results: List[RegressionResult] = []
        for server, metrics in current_metrics.items():
            results.extend(self.check_regression(server, metrics))
        return results

    # ------------------------------------------------------------------ #
    # Report generation                                                  #
    # ------------------------------------------------------------------ #

    def generate_report(
        self,
        results: List[RegressionResult],
        format: str = "text",
    ) -> str:
        """Generate a regression report."""
        if format == "json":
            return json.dumps(
                {"regressions": [r.to_dict() for r in results]},
                indent=2,
            )

        lines = []
        lines.append("=" * 70)
        lines.append("LFM3-1.7B Performance Regression Report")
        lines.append("=" * 70)
        lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
        lines.append("")

        if not results:
            lines.append("No regressions detected. All metrics within thresholds.")
            return "\n".join(lines)

        lines.append(f"Regressions detected: {len(results)}")
        lines.append("")

        # Sort by severity
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        sorted_results = sorted(results, key=lambda r: severity_order.get(r.severity, 3))

        for r in sorted_results:
            lines.append(f"[{r.severity.upper()}] {r.server} - {r.metric}")
            lines.append(f"  Baseline: {r.baseline_value:.2f}  Current: {r.current_value:.2f}")
            lines.append(f"  Change: {r.change_percent:+.2f}% ({r.direction})")
            lines.append(f"  Threshold: ±{r.threshold_percent:.1f}%")
            lines.append("")

        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Global instance                                                             #
# --------------------------------------------------------------------------- #

_regression_detector: Optional[RegressionDetector] = None


def get_regression_detector() -> RegressionDetector:
    global _regression_detector
    if _regression_detector is None:
        _regression_detector = RegressionDetector()
    return _regression_detector


# --------------------------------------------------------------------------- #
# CLI entry point                                                             #
# --------------------------------------------------------------------------- #

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LFM3-1.7B Regression Detection")
    subparsers = parser.add_subparsers(dest="command")

    # baseline command
    baseline_parser = subparsers.add_parser("baseline", help="Set baseline metrics")
    baseline_parser.add_argument("--server", required=True, help="Server name")
    baseline_parser.add_argument("--ttft-ms", type=float, required=True, help="Baseline TTFT ms")
    baseline_parser.add_argument("--decode-tps", type=float, required=True, help="Baseline decode TPS")
    baseline_parser.add_argument("--p95-latency-ms", type=float, required=True, help="Baseline p95 latency ms")
    baseline_parser.add_argument("--error-rate", type=float, default=0.0, help="Baseline error rate")
    baseline_parser.add_argument("--availability", type=float, default=100.0, help="Baseline availability %%")
    baseline_parser.add_argument("--output", default=None, help="Output file")

    # check command
    check_parser = subparsers.add_parser("check", help="Check for regressions")
    check_parser.add_argument("--server", required=True, help="Server name")
    check_parser.add_argument("--ttft-ms", type=float, required=True, help="Current TTFT ms")
    check_parser.add_argument("--decode-tps", type=float, required=True, help="Current decode TPS")
    check_parser.add_argument("--p95-latency-ms", type=float, required=True, help="Current p95 latency ms")
    check_parser.add_argument("--error-rate", type=float, default=0.0, help="Current error rate")
    check_parser.add_argument("--availability", type=float, default=100.0, help="Current availability %%")
    check_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    check_parser.add_argument("--output", default=None, help="Output file")

    args = parser.parse_args()

    detector = RegressionDetector()

    if args.command == "baseline":
        metrics = {
            "ttft_ms": args.ttft_ms,
            "decode_tps": args.decode_tps,
            "p95_latency_ms": args.p95_latency_ms,
            "error_rate": args.error_rate,
            "availability": args.availability,
        }
        detector.set_baseline(args.server, metrics)
        print(f"Baseline set for {args.server}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        if args.output:
            with open(args.output, "w") as f:
                json.dump({args.server: metrics}, f, indent=2)
            print(f"Baseline saved to {args.output}")
        return 0

    elif args.command == "check":
        metrics = {
            "ttft_ms": args.ttft_ms,
            "decode_tps": args.decode_tps,
            "p95_latency_ms": args.p95_latency_ms,
            "error_rate": args.error_rate,
            "availability": args.availability,
        }
        results = detector.check_regression(args.server, metrics)
        report = detector.generate_report(results, format=args.format)

        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"Report saved to {args.output}")
        else:
            print(report)

        # Exit with error if critical regression detected
        if any(r.severity == "critical" for r in results):
            return 1
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
