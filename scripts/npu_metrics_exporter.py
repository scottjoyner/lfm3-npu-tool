#!/usr/bin/env python3
"""
LFM3-1.7B NPU Prometheus Metrics Exporter.

Exposes NPU server metrics in Prometheus text format on a configurable port.
Scrapes the FastFlowLM NPU server (/v1/models, /v1/chat/completions) and
process metrics (CPU, memory) to produce Prometheus-compatible output.

Periodically sends test requests to the NPU server to measure real request
durations (TTFT + decode time) for the request_duration_seconds histogram.

Usage:
    python3 npu_metrics_exporter.py --port 9190 --npu-port 1239 --interval 15

Then configure Prometheus to scrape: http://localhost:9190/metrics
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error

import psutil

DEFAULT_PORT = 9190
DEFAULT_NPU_PORT = 1239
DEFAULT_INTERVAL = 15
DEFAULT_TEST_PROMPT = "What is 2+2?"

# Prometheus metric lines
METRICS_HEADER = """# HELP lfm3_npu_up Whether the NPU server is reachable
# TYPE lfm3_npu_up gauge
# HELP lfm3_npu_total_models Total number of models available on NPU server
# TYPE lfm3_npu_total_models gauge
# HELP lfm3_npu_lfm3_model_available Whether lfm3:1.7b model is available
# TYPE lfm3_npu_lfm3_model_available gauge
# HELP lfm3_npu_requests_total Total number of requests made by exporter
# TYPE lfm3_npu_requests_total counter
# HELP lfm3_npu_requests_errors_total Total number of failed requests
# TYPE lfm3_npu_requests_errors_total counter
# HELP lfm3_npu_process_cpu_percent NPU server process CPU usage percent
# TYPE lfm3_npu_process_cpu_percent gauge
# HELP lfm3_npu_process_memory_rss_bytes NPU server process RSS memory in bytes
# TYPE lfm3_npu_process_memory_rss_bytes gauge
# HELP lfm3_npu_process_memory_vms_bytes NPU server process VMS memory in bytes
# TYPE lfm3_npu_process_memory_vms_bytes gauge
# HELP lfm3_npu_process_threads NPU server process thread count
# TYPE lfm3_npu_process_threads gauge
# HELP lfm3_npu_request_duration_seconds Histogram of request durations
# TYPE lfm3_npu_request_duration_seconds histogram
# HELP lfm3_npu_ttft_seconds Time to first token
# TYPE lfm3_npu_ttft_seconds gauge
# HELP lfm3_npu_decode_tokens_per_second Decode speed in tokens/sec
# TYPE lfm3_npu_decode_tokens_per_second gauge
# HELP lfm3_npu_uptime_seconds Exporter uptime in seconds
# TYPE lfm3_npu_uptime_seconds gauge
# HELP lfm3_npu_in_warmup Whether exporter is in warmup period (errors not counted)
# TYPE lfm3_npu_in_warmup gauge
# HELP lfm3_npu_last_request_duration Last test request duration in seconds
# TYPE lfm3_npu_last_request_duration gauge
# HELP lfm3_npu_last_request_success Whether last test request succeeded
# TYPE lfm3_npu_last_request_success gauge
# HELP lfm3_cpu_up Whether the CPU server is reachable
# TYPE lfm3_cpu_up gauge
# HELP lfm3_cpu_uptime_seconds CPU server uptime in seconds
# TYPE lfm3_cpu_uptime_seconds gauge
# HELP lfm3_cpu_avg_decode_ms CPU server average decode time in ms
# TYPE lfm3_cpu_avg_decode_ms gauge
# HELP lfm3_cpu_active_slots CPU server active request slots
# TYPE lfm3_cpu_active_slots gauge
# HELP lfm3_cpu_free_slots CPU server free request slots
# TYPE lfm3_cpu_free_slots gauge
# HELP lfm3_disk_total_bytes Total disk space in bytes
# TYPE lfm3_disk_total_bytes gauge
# HELP lfm3_disk_used_bytes Used disk space in bytes
# TYPE lfm3_disk_used_bytes gauge
# HELP lfm3_disk_free_bytes Free disk space in bytes
# TYPE lfm3_disk_free_bytes gauge
# HELP lfm3_disk_used_percent Disk usage percent
# TYPE lfm3_disk_used_percent gauge
# HELP lfm3_qm_cpu_restarts Total CPU restarts by quality monitor
# TYPE lfm3_qm_cpu_restarts counter
# HELP lfm3_qm_npu_restarts Total NPU restarts by quality monitor
# TYPE lfm3_qm_npu_restarts counter
# HELP lfm3_qm_last_check Last quality monitor check timestamp
# TYPE lfm3_qm_last_check gauge
# HELP lfm3_qm_cpu_passed CPU quality checks passed
# TYPE lfm3_qm_cpu_passed counter
# HELP lfm3_qm_cpu_total Total CPU quality checks
# TYPE lfm3_qm_cpu_total counter
# HELP lfm3_qm_npu_passed NPU quality checks passed
# TYPE lfm3_qm_npu_passed counter
# HELP lfm3_qm_npu_total Total NPU quality checks
# TYPE lfm3_qm_npu_total counter
# HELP lfm3_qm_npu_unresponsive_restarts Total NPU unresponsive restarts by quality monitor
# TYPE lfm3_qm_npu_unresponsive_restarts counter
"""


class MetricsCollector:
    """Collects and stores metrics from the NPU server."""

    def __init__(self, npu_port: int, npu_host: str = "127.0.0.1",
                 cpu_port: int = 1238, cpu_host: str = "127.0.0.1",
                 test_prompt: str = DEFAULT_TEST_PROMPT,
                 max_duration_samples: int = 1000,
                 warmup_seconds: int = 30,
                 disk_paths: list[str] | None = None):
        self.npu_port = npu_port
        self.npu_host = npu_host
        self.cpu_port = cpu_port
        self.cpu_host = cpu_host
        self.test_prompt = test_prompt
        self.max_duration_samples = max_duration_samples
        self.warmup_seconds = warmup_seconds
        self._start_time = time.time()
        self._lock = threading.Lock()
        self._durations: list[float] = []
        self._ttfts: list[float] = []
        self._decodes: list[float] = []
        self._requests_total = 0
        self._requests_errors = 0
        self._last_ttft = 0.0
        self._last_decode_tps = 0.0
        self._last_duration = 0.0
        self._last_success = False
        # CPU metrics cache
        self._cpu_up = 0
        self._cpu_uptime = 0.0
        self._cpu_avg_decode_ms = 0.0
        self._cpu_active_slots = 0
        self._cpu_free_slots = 0
        # Disk metrics cache
        self._disk_paths = disk_paths or ["/", "/media/scott/data", "/tmp"]
        self._disk_total_bytes = 0
        self._disk_used_bytes = 0
        self._disk_free_bytes = 0
        self._disk_used_percent = 0.0
        # Quality monitor metrics cache
        self._qm_cpu_restarts = 0
        self._qm_npu_restarts = 0
        self._qm_npu_unresponsive_restarts = 0
        self._qm_last_check = 0.0
        self._qm_cpu_passed = 0
        self._qm_cpu_total = 0
        self._qm_npu_passed = 0
        self._qm_npu_total = 0

    def _read_quality_monitor_state(self):
        """Read quality monitor state from state file."""
        import os
        state_file = "/media/scott/data/lfm3-npu-tool/var/log/quality-monitor-state.json"
        try:
            if os.path.exists(state_file):
                with open(state_file) as f:
                    import json
                    state = json.load(f)
                self._qm_cpu_restarts = state.get("cpu_restarts", 0)
                self._qm_npu_restarts = state.get("npu_restarts", 0)
                self._qm_npu_unresponsive_restarts = state.get("npu_unresponsive_restarts", 0)
                self._qm_last_check = state.get("last_check", 0.0)
                self._qm_cpu_passed = state.get("cpu_passed", 0)
                self._qm_cpu_total = state.get("cpu_total", 0)
                self._qm_npu_passed = state.get("npu_passed", 0)
                self._qm_npu_total = state.get("npu_total", 0)
        except Exception:
            pass

    def fetch_json(self, url: str, timeout: float = 10.0):
        """Fetch JSON from URL, return None on failure."""
        try:
            req = urllib.request.urlopen(url, timeout=timeout)
            return json.loads(req.read().decode("utf-8"))
        except Exception:
            return None

    def find_npu_process(self):
        """Find the NPU server process by command line."""
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = " ".join(proc.info["cmdline"] or [])
                if "flm" in cmdline and "serve" in cmdline and str(self.npu_port) in cmdline:
                    return proc
            except Exception:
                continue
        return None

    def get_process_metrics(self, proc):
        """Get CPU and memory metrics for a process."""
        try:
            with proc.oneshot():
                return {
                    "cpu_percent": proc.cpu_percent(interval=0.1),
                    "memory_rss_bytes": proc.memory_info().rss,
                    "memory_vms_bytes": proc.memory_info().vms,
                    "num_threads": proc.num_threads(),
                }
        except Exception:
            return None

    def _collect_disk_metrics(self):
        """Collect disk usage metrics for configured paths."""
        import shutil

        total = 0
        used = 0
        free = 0
        for path in self._disk_paths:
            try:
                disk = shutil.disk_usage(path)
                total += disk.total
                used += disk.used
                free += disk.free
            except Exception:
                continue

        if total > 0:
            self._disk_total_bytes = total
            self._disk_used_bytes = used
            self._disk_free_bytes = free
            self._disk_used_percent = (used / total) * 100.0

    def send_test_request(self) -> dict:
        """Send a test request to the NPU server and measure duration."""
        payload = {
            "model": "lfm3:1.7b",
            "messages": [{"role": "user", "content": self.test_prompt}],
            "max_tokens": 20,
            "temperature": 0.7,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://{self.npu_host}:{self.npu_port}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            duration = time.perf_counter() - start
            usage = result.get("usage", {})
            ttft = usage.get("prefill_duration_ttft", 0) or 0
            decode_tps = usage.get("decoding_speed_tps", 0) or 0
            return {
                "success": True,
                "duration": duration,
                "ttft": ttft,
                "decode_tps": decode_tps,
            }
        except Exception:
            duration = time.perf_counter() - start
            return {
                "success": False,
                "duration": duration,
                "ttft": 0,
                "decode_tps": 0,
            }

    def scrape(self) -> dict:
        """Scrape all available metrics from the NPU server."""
        elapsed = time.time() - self._start_time
        in_warmup = elapsed < self.warmup_seconds

        # Send a test request to measure real duration
        test_result = self.send_test_request()
        with self._lock:
            self._requests_total += 1
            # During warmup, don't count errors (NPU server may still be starting)
            if not test_result["success"]:
                if not in_warmup:
                    self._requests_errors += 1
            else:
                self._durations.append(test_result["duration"])
                self._ttfts.append(test_result["ttft"])
                self._decodes.append(test_result["decode_tps"])
                # Trim to max samples
                if len(self._durations) > self.max_duration_samples:
                    self._durations = self._durations[-self.max_duration_samples:]
                if len(self._ttfts) > self.max_duration_samples:
                    self._ttfts = self._ttfts[-self.max_duration_samples:]
                if len(self._decodes) > self.max_duration_samples:
                    self._decodes = self._decodes[-self.max_duration_samples:]
                # Update last values
                self._last_ttft = test_result["ttft"]
                self._last_decode_tps = test_result["decode_tps"]
                self._last_duration = test_result["duration"]
                self._last_success = True

            # Re-read after update
            durations = list(self._durations)
            ttfts = list(self._ttfts)
            decodes = list(self._decodes)
            requests_total = self._requests_total
            requests_errors = self._requests_errors
            last_ttft = self._last_ttft
            last_decode_tps = self._last_decode_tps
            last_duration = self._last_duration
            last_success = self._last_success

        # Check /v1/models
        models_data = self.fetch_json(
            f"http://{self.npu_host}:{self.npu_port}/v1/models", timeout=10
        )
        up = 0
        total_models = 0
        lfm3_available = 0
        if models_data and "data" in models_data:
            up = 1
            models = models_data["data"]
            total_models = len(models)
            lfm_models = [m for m in models if "lfm3" in str(m.get("id", ""))]
            if lfm_models:
                lfm3_available = 1

        # Check CPU server /health endpoint
        cpu_data = self.fetch_json(
            f"http://{self.cpu_host}:{self.cpu_port}/health", timeout=5
        )
        if cpu_data and cpu_data.get("status") == "healthy":
            self._cpu_up = 1
            self._cpu_uptime = cpu_data.get("uptime_seconds", 0.0)
            self._cpu_avg_decode_ms = cpu_data.get("avg_decode_ms", 0.0)
            self._cpu_active_slots = cpu_data.get("active_slots", 0)
            self._cpu_free_slots = cpu_data.get("free_slots", 0)
        else:
            self._cpu_up = 0
            self._cpu_uptime = 0.0
            self._cpu_avg_decode_ms = 0.0
            self._cpu_active_slots = 0
            self._cpu_free_slots = 0

        # Collect disk usage metrics
        self._collect_disk_metrics()

        # Read quality monitor state
        self._read_quality_monitor_state()

        # Process metrics
        proc = self.find_npu_process()
        process_cpu_percent = 0
        process_memory_rss_bytes = 0
        process_memory_vms_bytes = 0
        process_threads = 0
        if proc:
            pm = self.get_process_metrics(proc)
            if pm:
                process_cpu_percent = pm["cpu_percent"]
                process_memory_rss_bytes = pm["memory_rss_bytes"]
                process_memory_vms_bytes = pm["memory_vms_bytes"]
                process_threads = pm["num_threads"]

        return {
            "up": up,
            "total_models": total_models,
            "lfm3_available": lfm3_available,
            "process_cpu_percent": process_cpu_percent,
            "process_memory_rss_bytes": process_memory_rss_bytes,
            "process_memory_vms_bytes": process_memory_vms_bytes,
            "process_threads": process_threads,
            "durations": durations,
            "ttfts": ttfts,
            "decodes": decodes,
            "requests_total": requests_total,
            "requests_errors": requests_errors,
            "last_ttft": last_ttft,
            "last_decode_tps": last_decode_tps,
            "last_duration": last_duration,
            "last_success": last_success,
            "uptime_seconds": elapsed,
            "in_warmup": in_warmup,
            "cpu_up": self._cpu_up,
            "cpu_uptime": self._cpu_uptime,
            "cpu_avg_decode_ms": self._cpu_avg_decode_ms,
            "cpu_active_slots": self._cpu_active_slots,
            "cpu_free_slots": self._cpu_free_slots,
            "disk_total_bytes": self._disk_total_bytes,
            "disk_used_bytes": self._disk_used_bytes,
            "disk_free_bytes": self._disk_free_bytes,
            "disk_used_percent": self._disk_used_percent,
            "qm_cpu_restarts": self._qm_cpu_restarts,
            "qm_npu_restarts": self._qm_npu_restarts,
            "qm_last_check": self._qm_last_check,
            "qm_cpu_passed": self._qm_cpu_passed,
            "qm_cpu_total": self._qm_cpu_total,
            "qm_npu_passed": self._qm_npu_passed,
            "qm_npu_total": self._qm_npu_total,
            "qm_npu_unresponsive_restarts": self._qm_npu_unresponsive_restarts,
        }


def format_prometheus(metrics: dict) -> str:
    """Format metrics in Prometheus text format."""
    lines = []
    lines.append(METRICS_HEADER)
    lines.append(f'lfm3_npu_up {metrics["up"]}')
    lines.append(f'lfm3_npu_total_models {metrics["total_models"]}')
    lines.append(f'lfm3_npu_lfm3_model_available {metrics["lfm3_available"]}')
    lines.append(f'lfm3_npu_requests_total {metrics["requests_total"]}')
    lines.append(f'lfm3_npu_requests_errors_total {metrics["requests_errors"]}')
    lines.append(f'lfm3_npu_process_cpu_percent {metrics["process_cpu_percent"]}')
    lines.append(f'lfm3_npu_process_memory_rss_bytes {metrics["process_memory_rss_bytes"]}')
    lines.append(f'lfm3_npu_process_memory_vms_bytes {metrics["process_memory_vms_bytes"]}')
    lines.append(f'lfm3_npu_process_threads {metrics["process_threads"]}')
    lines.append(f'lfm3_npu_ttft_seconds {metrics["last_ttft"]}')
    lines.append(f'lfm3_npu_decode_tokens_per_second {metrics["last_decode_tps"]}')
    lines.append(f'lfm3_npu_uptime_seconds {metrics["uptime_seconds"]}')
    lines.append(f'lfm3_npu_in_warmup {1 if metrics["in_warmup"] else 0}')
    lines.append(f'lfm3_npu_last_request_duration {metrics["last_duration"]}')
    lines.append(f'lfm3_npu_last_request_success {1 if metrics["last_success"] else 0}')

    # CPU server metrics
    lines.append(f'lfm3_cpu_up {metrics["cpu_up"]}')
    lines.append(f'lfm3_cpu_uptime_seconds {metrics["cpu_uptime"]}')
    lines.append(f'lfm3_cpu_avg_decode_ms {metrics["cpu_avg_decode_ms"]}')
    lines.append(f'lfm3_cpu_active_slots {metrics["cpu_active_slots"]}')
    lines.append(f'lfm3_cpu_free_slots {metrics["cpu_free_slots"]}')

    # Disk usage metrics
    lines.append(f'lfm3_disk_total_bytes {metrics["disk_total_bytes"]}')
    lines.append(f'lfm3_disk_used_bytes {metrics["disk_used_bytes"]}')
    lines.append(f'lfm3_disk_free_bytes {metrics["disk_free_bytes"]}')
    lines.append(f'lfm3_disk_used_percent {metrics["disk_used_percent"]}')

    # Quality monitor metrics
    lines.append(f'lfm3_qm_cpu_restarts {metrics["qm_cpu_restarts"]}')
    lines.append(f'lfm3_qm_npu_restarts {metrics["qm_npu_restarts"]}')
    lines.append(f'lfm3_qm_last_check {metrics["qm_last_check"]}')
    lines.append(f'lfm3_qm_cpu_passed {metrics["qm_cpu_passed"]}')
    lines.append(f'lfm3_qm_cpu_total {metrics["qm_cpu_total"]}')
    lines.append(f'lfm3_qm_npu_passed {metrics["qm_npu_passed"]}')
    lines.append(f'lfm3_qm_npu_total {metrics["qm_npu_total"]}')
    lines.append(f'lfm3_qm_npu_unresponsive_restarts {metrics["qm_npu_unresponsive_restarts"]}')

    # Request duration histogram
    durations = metrics["durations"]
    buckets = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    if durations:
        bucket_counts = {b: 0 for b in buckets}
        for d in durations:
            for b in buckets:
                if d <= b:
                    bucket_counts[b] += 1
        total = len(durations)
        cumulative = 0
        for b in buckets:
            cumulative += bucket_counts[b]
            lines.append(f'lfm3_npu_request_duration_seconds_bucket{{le="{b}"}} {cumulative}')
        lines.append(f'lfm3_npu_request_duration_seconds_bucket{{le="+Inf"}} {total}')
        lines.append(f'lfm3_npu_request_duration_seconds_sum {sum(durations)}')
        lines.append(f'lfm3_npu_request_duration_seconds_count {total}')
    else:
        for b in buckets:
            lines.append(f'lfm3_npu_request_duration_seconds_bucket{{le="{b}"}} 0')
        lines.append('lfm3_npu_request_duration_seconds_bucket{le="+Inf"} 0')
        lines.append('lfm3_npu_request_duration_seconds_sum 0')
        lines.append('lfm3_npu_request_duration_seconds_count 0')

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="LFM3-1.7B NPU Prometheus Metrics Exporter"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Exporter listen port (default: {DEFAULT_PORT})")
    parser.add_argument("--npu-port", type=int, default=DEFAULT_NPU_PORT,
                        help=f"NPU server port (default: {DEFAULT_NPU_PORT})")
    parser.add_argument("--npu-host", default="127.0.0.1",
                        help="NPU server host (default: 127.0.0.1)")
    parser.add_argument("--cpu-port", type=int, default=1238,
                        help="CPU server port (default: 1238)")
    parser.add_argument("--cpu-host", default="127.0.0.1",
                        help="CPU server host (default: 127.0.0.1)")
    parser.add_argument("--disk-paths", type=str, default="/,/media/scott/data,/tmp",
                        help="Comma-separated disk paths to monitor (default: /,/media/scott/data,/tmp)")
    parser.add_argument("--test-prompt", default=DEFAULT_TEST_PROMPT,
                        help=f"Test prompt for duration measurement (default: '{DEFAULT_TEST_PROMPT}')")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Max duration samples to keep (default: 1000)")
    parser.add_argument("--interval", type=int, default=15,
                        help="Scrape interval in seconds (default: 15)")
    parser.add_argument("--once", action="store_true",
                        help="Scrape once and exit (for testing)")
    args = parser.parse_args()

    import http.server
    import socketserver

    collector = MetricsCollector(
        npu_port=args.npu_port,
        npu_host=args.npu_host,
        cpu_port=args.cpu_port,
        cpu_host=args.cpu_host,
        disk_paths=args.disk_paths.split(","),
        test_prompt=args.test_prompt,
        max_duration_samples=args.max_samples,
    )

    class MetricsHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/metrics":
                metrics = collector.scrape()
                body = format_prometheus(metrics).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/health":
                metrics = collector.scrape()
                healthy = metrics["up"] == 1
                body = json.dumps({"healthy": healthy, "up": metrics["up"]}).encode("utf-8")
                self.send_response(200 if healthy else 503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass  # Suppress default logging

    if args.once:
        metrics = collector.scrape()
        print(format_prometheus(metrics))
        return 0

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("0.0.0.0", args.port), MetricsHandler) as httpd:
        print(f"NPU metrics exporter listening on port {args.port}")
        print(f"  Scraping NPU server at {args.npu_host}:{args.npu_port}")
        print(f"  Metrics endpoint: http://0.0.0.0:{args.port}/metrics")
        print(f"  Health endpoint:  http://0.0.0.0:{args.port}/health")
        print("  Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nExporter stopped.")
            return 0


if __name__ == "__main__":
    sys.exit(main())
