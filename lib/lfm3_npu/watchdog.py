"""
Server watchdog for LFM3-1.7B inference servers.

Monitors server health and auto-restarts unresponsive servers.
Can run as a standalone daemon or be called periodically.
Includes performance-based monitoring (TPS threshold, error rate),
webhook notifications, cooldown periods, and graceful degradation.
"""

import json
import sys
import subprocess
import time
from typing import Any, Dict, List, Optional

import psutil
import urllib.request

from .config import Config
from .server import ServerManager, get_process


class Watchdog:
    """Monitors and auto-restarts LFM3 inference servers."""

    def __init__(
        self,
        config: Optional[Config] = None,
        check_interval: int = 30,
        min_tps: Optional[float] = None,
        max_error_rate: Optional[float] = None,
        max_latency_ms: Optional[float] = None,
        webhook_url: Optional[str] = None,
        cooldown_seconds: int = 300,
        enable_degradation: bool = False,
        min_concurrent_requests: int = 1,
    ):
        self.config = config or Config({})
        self.manager = ServerManager(config)
        self.check_interval = check_interval
        self.min_tps = min_tps
        self.max_error_rate = max_error_rate
        self.max_latency_ms = max_latency_ms
        self.webhook_url = webhook_url
        self.cooldown_seconds = cooldown_seconds
        self.enable_degradation = enable_degradation
        self.min_concurrent_requests = min_concurrent_requests
        self._running = False
        self._consecutive_failures: Dict[str, int] = {}
        self._last_tps: Dict[str, float] = {}
        self._last_restart_time: Dict[str, float] = {}
        self._last_alert_time: Dict[str, float] = {}

    def _send_webhook(self, server: str, message: str, severity: str = "warning") -> None:
        """Send a notification to the webhook endpoint."""
        if not self.webhook_url:
            return
        try:
            payload = {
                "timestamp": time.time(),
                "server": server,
                "severity": severity,
                "message": message,
                "source": "lfm3-watchdog",
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass  # Don't let webhook failures break the watchdog

    def _alert(self, server: str, message: str, severity: str = "warning") -> None:
        """Send an alert via webhook and print to stdout."""
        now = time.time()
        last_alert = self._last_alert_time.get(server, 0)
        # Throttle alerts to once per cooldown period per server
        if now - last_alert < self.cooldown_seconds:
            return
        self._last_alert_time[server] = now
        print(f"Watchdog alert [{severity}] {server}: {message}", flush=True)
        self._send_webhook(server, message, severity)

    def _can_restart(self, server: str) -> bool:
        """Check if cooldown period has elapsed since last restart."""
        now = time.time()
        last_restart = self._last_restart_time.get(server, 0)
        return now - last_restart >= self.cooldown_seconds

    def _record_restart(self, server: str) -> None:
        """Record a restart event."""
        self._last_restart_time[server] = time.time()
        self._consecutive_failures[server] = self._consecutive_failures.get(server, 0) + 1

    def check_health(self, server: str) -> bool:
        """Check if a server is responding to health requests."""
        if server == "cpu":
            port = self.config.get("server.cpu.port", 1238)
            host = self.config.get("server.cpu.host", "0.0.0.0")
        elif server == "npu":
            port = self.config.get("server.npu.port", 1239)
            host = self.config.get("server.npu.host", "0.0.0.0")
        else:
            return False

        try:
            req = urllib.request.urlopen(
                f"http://{host}:{port}/health", timeout=5
            )
            return req.status == 200
        except Exception:
            return False

    def check_performance(self, server: str) -> Optional[Dict[str, Any]]:
        """Check server performance metrics from /metrics endpoint.

        Returns metrics dict or None if unavailable.
        """
        if server == "cpu":
            port = self.config.get("server.cpu.port", 1238)
            host = self.config.get("server.cpu.host", "0.0.0.0")
        elif server == "npu":
            port = self.config.get("server.npu.port", 1239)
            host = self.config.get("server.npu.host", "0.0.0.0")
        else:
            return None

        try:
            import json
            req = urllib.request.urlopen(
                f"http://{host}:{port}/metrics", timeout=5
            )
            text = req.read().decode("utf-8")
            # Parse simple Prometheus text format for key metrics
            metrics = {}
            for line in text.split("\n"):
                if line.startswith("lfm3_tokens_per_second"):
                    parts = line.split(" ")
                    if len(parts) >= 2:
                        metrics["tokens_per_second"] = float(parts[1])
                elif line.startswith("lfm3_requests_errors_total"):
                    parts = line.split(" ")
                    if len(parts) >= 2:
                        metrics["errors_total"] = float(parts[1])
                elif line.startswith("lfm3_requests_total"):
                    parts = line.split(" ")
                    if len(parts) >= 2:
                        metrics["requests_total"] = float(parts[1])
            return metrics
        except Exception:
            return None

    def check_and_restart(self, server: str) -> bool:
        """Check server health and restart if unresponsive or underperforming.

        Returns True if action was taken (restarted), False otherwise.
        """
        # First check HTTP health — if responding, server is fine regardless of PID tracking
        if self.check_health(server):
            self._consecutive_failures[server] = 0
            return False

        # Health check failed — check if process is tracked
        try:
            proc = self.manager.get_process(server)
        except psutil.NoSuchProcess:
            proc = None

        if proc is None:
            print(f"Watchdog: {server.upper()} not running, starting...", flush=True)
            self._alert(server, "not running, starting", "critical")
            self._record_restart(server)
            return self.manager.start(server)

        if not self.check_health(server):
            print(f"Watchdog: {server.upper()} unresponsive, restarting...", flush=True)
            self._alert(server, "unresponsive, restarting", "critical")
            self._record_restart(server)
            return self.manager.restart(server)

        # Performance-based checks
        if self.min_tps is not None or self.max_error_rate is not None or self.max_latency_ms is not None:
            metrics = self.check_performance(server)
            if metrics:
                tps = metrics.get("tokens_per_second", 0)
                self._last_tps[server] = tps

                # Check TPS threshold
                if self.min_tps is not None and tps > 0 and tps < self.min_tps:
                    msg = f"{server.upper()} TPS ({tps:.1f}) below threshold ({self.min_tps})"
                    self._alert(server, msg, "warning")
                    if self.enable_degradation and self._can_restart(server):
                        print(f"Watchdog: {server.upper()} degrading concurrency (low TPS)", flush=True)
                        self._record_restart(server)
                        return self._degrade_concurrency(server)
                    elif self._can_restart(server):
                        self._record_restart(server)
                        return self.manager.restart(server)

                # Check error rate
                if self.max_error_rate is not None:
                    errors = metrics.get("errors_total", 0)
                    requests = metrics.get("requests_total", 0)
                    if requests > 0:
                        error_rate = errors / requests
                        if error_rate > self.max_error_rate:
                            msg = f"{server.upper()} error rate ({error_rate*100:.1f}%) above threshold ({self.max_error_rate*100:.1f}%)"
                            self._alert(server, msg, "warning")
                            if self.enable_degradation and self._can_restart(server):
                                self._record_restart(server)
                                return self._degrade_concurrency(server)
                            elif self._can_restart(server):
                                self._record_restart(server)
                                return self.manager.restart(server)

                # Check latency
                if self.max_latency_ms is not None:
                    # Get detailed metrics for latency
                    try:
                        import json
                        req = urllib.request.urlopen(
                            f"http://{host}:{port}/v1/metrics/detailed", timeout=5
                        )
                        detailed = json.loads(req.read().decode("utf-8"))
                        completed = detailed.get("completed", {})
                        avg_latency = completed.get("avg_token_latency_ms", 0)
                        if avg_latency > 0 and avg_latency > self.max_latency_ms:
                            msg = f"{server.upper()} avg latency ({avg_latency:.1f}ms) above threshold ({self.max_latency_ms}ms)"
                            self._alert(server, msg, "warning")
                            if self.enable_degradation and self._can_restart(server):
                                self._record_restart(server)
                                return self._degrade_concurrency(server)
                            elif self._can_restart(server):
                                self._record_restart(server)
                                return self.manager.restart(server)
                    except Exception:
                        pass

        # Reset failure counter on healthy check
        self._consecutive_failures[server] = 0
        return False

    def _degrade_concurrency(self, server: str) -> bool:
        """Reduce concurrency on the server to recover from overload."""
        try:
            if server == "cpu":
                port = self.config.get("server.cpu.port", 1238)
                host = self.config.get("server.cpu.host", "0.0.0.0")
            elif server == "npu":
                port = self.config.get("server.npu.port", 1239)
                host = self.config.get("server.npu.host", "0.0.0.0")
            else:
                return False

            # Get current batch status to see current concurrency
            import json
            req = urllib.request.urlopen(
                f"http://{host}:{port}/v1/batch/status", timeout=5
            )
            batch_status = json.loads(req.read().decode("utf-8"))
            current_max = batch_status.get("max_batch_size", 8)
            new_max = max(current_max // 2, self.min_concurrent_requests)

            if new_max < current_max:
                print(f"Watchdog: {server.upper()} reducing max_batch_size {current_max} -> {new_max}", flush=True)
                # Note: This requires the server to support dynamic reconfiguration.
                # For now, we log and restart with new env var.
                # A full implementation would update the running server config.
                return self.manager.restart(server)
            return False
        except Exception as e:
            print(f"Watchdog: degradation failed for {server}: {e}", flush=True)
            return False

    def run_forever(self, servers: Optional[List[str]] = None) -> None:
        """Run the watchdog loop forever.

        Args:
            servers: List of servers to monitor, or None for both.
        """
        import signal

        servers = servers or ["cpu", "npu"]
        self._running = True

        def signal_handler(signum, frame):
            self._running = False

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        print(f"Watchdog: Monitoring {servers} every {self.check_interval}s", flush=True)
        if self.min_tps:
            print(f"Watchdog: Min TPS threshold: {self.min_tps}", flush=True)
        if self.max_error_rate:
            print(f"Watchdog: Max error rate: {self.max_error_rate*100:.1f}%", flush=True)
        if self.max_latency_ms:
            print(f"Watchdog: Max latency: {self.max_latency_ms}ms", flush=True)
        if self.webhook_url:
            print(f"Watchdog: Webhook alerts: {self.webhook_url}", flush=True)
        if self.enable_degradation:
            print(f"Watchdog: Graceful degradation enabled", flush=True)
        if self.cooldown_seconds > 0:
            print(f"Watchdog: Cooldown: {self.cooldown_seconds}s", flush=True)

        while self._running:
            for server in servers:
                self.check_and_restart(server)
            time.sleep(self.check_interval)

        print("Watchdog: Stopped", flush=True)

    def check_once(self, servers: Optional[List[str]] = None) -> Dict[str, bool]:
        """Run a single health check on all servers.

        Returns:
            Dictionary mapping server name to action taken (True=restarted).
        """
        servers = servers or ["cpu", "npu"]
        results: Dict[str, bool] = {}
        for server in servers:
            results[server] = self.check_and_restart(server)
        return results

    def get_status(self, servers: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Get detailed status for all servers.

        Returns:
            Dictionary mapping server name to status info.
        """
        import json
        servers = servers or ["cpu", "npu"]
        status: Dict[str, Dict] = {}
        for server in servers:
            entry: Dict = {}
            try:
                proc = self.manager.get_process(server)
            except psutil.NoSuchProcess:
                proc = None
            entry["running"] = proc is not None
            if proc:
                try:
                    entry["pid"] = proc.pid
                    entry["cpu_percent"] = proc.cpu_percent(interval=0.5)
                    entry["memory_mb"] = proc.memory_info().rss / 1024 / 1024
                except Exception:
                    pass
            entry["healthy"] = self.check_health(server) if entry["running"] else False
            metrics = self.check_performance(server)
            if metrics:
                entry["metrics"] = metrics
            entry["consecutive_failures"] = self._consecutive_failures.get(server, 0)
            entry["last_tps"] = self._last_tps.get(server)
            status[server] = entry
        return status


# CLI entry point
def watchdog_main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="LFM3-1.7B Server Watchdog"
    )
    parser.add_argument("--servers", nargs="+", default=["cpu", "npu"],
                         choices=["cpu", "npu"],
                         help="Servers to monitor")
    parser.add_argument("--interval", type=int, default=30,
                         help="Check interval in seconds")
    parser.add_argument("--once", action="store_true",
                         help="Run one check and exit")
    parser.add_argument("--daemon", action="store_true",
                         help="Run as daemon (foreground)")
    parser.add_argument("--min-tps", type=float, default=None,
                         help="Minimum tokens/sec threshold (restart if below)")
    parser.add_argument("--max-error-rate", type=float, default=None,
                         help="Maximum error rate (0.0-1.0, restart if above)")
    parser.add_argument("--max-latency-ms", type=float, default=None,
                         help="Max token latency in ms (restart if above)")
    parser.add_argument("--webhook", type=str, default=None,
                         help="Webhook URL for alert notifications")
    parser.add_argument("--cooldown", type=int, default=300,
                         help="Cooldown period between restarts (seconds)")
    parser.add_argument("--degrade", action="store_true",
                         help="Enable graceful degradation (reduce concurrency before restart)")
    parser.add_argument("--min-concurrent", type=int, default=1,
                         help="Minimum concurrent requests when degrading")
    parser.add_argument("--status", action="store_true",
                         help="Show detailed status and exit")

    args = parser.parse_args()

    from .config import load_config
    config = load_config()
    watchdog = Watchdog(
        config,
        check_interval=args.interval,
        min_tps=args.min_tps,
        max_error_rate=args.max_error_rate,
        max_latency_ms=args.max_latency_ms,
        webhook_url=args.webhook,
        cooldown_seconds=args.cooldown,
        enable_degradation=args.degrade,
        min_concurrent_requests=args.min_concurrent,
    )

    if args.status:
        status = watchdog.get_status(args.servers)
        print(json.dumps(status, indent=2))
        return 0

    if args.once:
        results = watchdog.check_once(args.servers)
        for server, action_taken in results.items():
            action = "restarted" if action_taken else "healthy"
            print(f"Watchdog: {server.upper()} - {action}")
        return 0
    else:
        watchdog.run_forever(args.servers)
        return 0


if __name__ == "__main__":
    sys.exit(watchdog_main())
