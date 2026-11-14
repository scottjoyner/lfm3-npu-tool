#!/usr/bin/env python3
"""
LFM3-1.7B NPU Prometheus Metrics Exporter.

Exposes NPU server metrics in Prometheus text format on a configurable port.
Scrapes the FastFlowLM NPU server (/v1/models, /v1/chat/completions) and
process metrics (CPU, memory) to produce Prometheus-compatible output.

Usage:
    python3 npu_metrics_exporter.py --port 9190 --npu-port 1239 --interval 15

Then configure Prometheus to scrape: http://localhost:9190/metrics
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

import psutil

DEFAULT_PORT = 9190
DEFAULT_NPU_PORT = 1239
DEFAULT_INTERVAL = 15

# Prometheus metric lines
METRICS_HEADER = """# HELP lfm3_npu_up Whether the NPU server is reachable
# TYPE lfm3_npu_up gauge
# HELP lfm3_npu_total_models Total number of models available on NPU server
# TYPE lfm3_npu_total_models gauge
# HELP lfm3_npu_lfm3_model_available Whether lfm3:1.7b model is available
# TYPE lfm3_npu_lfm3_model_available gauge
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
"""


def fetch_json(url, timeout=10):
    """Fetch JSON from URL, return None on failure."""
    try:
        req = urllib.request.urlopen(url, timeout=timeout)
        return json.loads(req.read().decode("utf-8"))
    except Exception:
        return None


def find_npu_process(port):
    """Find the NPU server process by command line."""
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
            if "flm" in cmdline and "serve" in cmdline and str(port) in cmdline:
                return proc
        except Exception:
            continue
    return None


def get_process_metrics(proc):
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


def scrape_npu_metrics(npu_port, npu_host="127.0.0.1"):
    """Scrape all available metrics from the NPU server."""
    metrics = {
        "up": 0,
        "total_models": 0,
        "lfm3_available": 0,
        "process_cpu_percent": 0,
        "process_memory_rss_bytes": 0,
        "process_memory_vms_bytes": 0,
        "process_threads": 0,
        "request_duration_seconds": [],
    }

    # Check /v1/models
    models_data = fetch_json(f"http://{npu_host}:{npu_port}/v1/models", timeout=10)
    if models_data and "data" in models_data:
        metrics["up"] = 1
        models = models_data["data"]
        metrics["total_models"] = len(models)
        lfm_models = [m for m in models if "lfm3" in str(m.get("id", ""))]
        if lfm_models:
            metrics["lfm3_available"] = 1

    # Process metrics
    proc = find_npu_process(npu_port)
    if proc:
        pm = get_process_metrics(proc)
        if pm:
            metrics["process_cpu_percent"] = pm["cpu_percent"]
            metrics["process_memory_rss_bytes"] = pm["memory_rss_bytes"]
            metrics["process_memory_vms_bytes"] = pm["memory_vms_bytes"]
            metrics["process_threads"] = pm["num_threads"]

    return metrics


def format_prometheus(metrics):
    """Format metrics in Prometheus text format."""
    lines = []
    lines.append(METRICS_HEADER)
    lines.append(f'lfm3_npu_up {metrics["up"]}')
    lines.append(f'lfm3_npu_total_models {metrics["total_models"]}')
    lines.append(f'lfm3_npu_lfm3_model_available {metrics["lfm3_available"]}')
    lines.append(f'lfm3_npu_process_cpu_percent {metrics["process_cpu_percent"]}')
    lines.append(f'lfm3_npu_process_memory_rss_bytes {metrics["process_memory_rss_bytes"]}')
    lines.append(f'lfm3_npu_process_memory_vms_bytes {metrics["process_memory_vms_bytes"]}')
    lines.append(f'lfm3_npu_process_threads {metrics["process_threads"]}')

    # Request duration histogram (simple bucket counts)
    durations = metrics["request_duration_seconds"]
    if durations:
        buckets = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
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
        # Empty histogram with zero counts
        for b in [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]:
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
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                        help=f"Scrape interval in seconds (default: {DEFAULT_INTERVAL})")
    parser.add_argument("--once", action="store_true",
                        help="Scrape once and exit (for testing)")
    args = parser.parse_args()

    import http.server
    import socketserver

    class MetricsHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/metrics":
                metrics = scrape_npu_metrics(args.npu_port, args.npu_host)
                body = format_prometheus(metrics).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/health":
                metrics = scrape_npu_metrics(args.npu_port, args.npu_host)
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
        metrics = scrape_npu_metrics(args.npu_port, args.npu_host)
        print(format_prometheus(metrics))
        return 0

    with socketserver.TCPServer(("0.0.0.0", args.port), MetricsHandler) as httpd:
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
