#!/usr/bin/env python3
"""
Start the LFM3-1.7B Prometheus + Grafana monitoring stack.

Usage:
    python3 scripts/start_monitoring.py up
    python3 scripts/start_monitoring.py down
    python3 scripts/start_monitoring.py status

Access:
    Grafana: http://localhost:3000 (admin / set GRAFANA_ADMIN_PASSWORD before starting)
    Prometheus: http://localhost:9090

The metrics exporter must be running on port 9190:
    systemctl --user start lfm3-npu-metrics.service
"""

import os
import subprocess

TOOL_DIR = "/media/scott/data/lfm3-npu-tool"
COMPOSE_FILE = os.path.join(TOOL_DIR, "docker", "prometheus-grafana.yml")

def cmd_up(args):
    """Start Prometheus + Grafana."""
    import os
    os.chdir(TOOL_DIR)
    subprocess.run(["docker", "compose", "-f", COMPOSE_FILE, "up", "-d"], check=True)
    print("\n✓ Monitoring stack started.")
    print("\nAccess points:")
    print("  Grafana:   http://localhost:3000 (admin/admin)")
    print("  Prometheus: http://localhost:9090")
    print("\nMake sure the metrics exporter is running:")
    print("  systemctl --user start lfm3-npu-metrics.service")
    print("  curl http://localhost:9190/metrics")

def cmd_down(args):
    """Stop the monitoring stack."""
    import os
    os.chdir(TOOL_DIR)
    subprocess.run(["docker", "compose", "-f", COMPOSE_FILE, "down"], check=True)
    print("✓ Monitoring stack stopped.")

def cmd_status(args):
    """Show stack status."""
    import os
    os.chdir(TOOL_DIR)
    result = subprocess.run(["docker", "compose", "-f", COMPOSE_FILE, "ps"], capture_output=True, text=True)
    print(result.stdout)

def main():
    parser = __import__("argparse").ArgumentParser(
        description="Start/stop LFM3-1.7B Prometheus + Grafana monitoring stack"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    up_parser = subparsers.add_parser("up", help="Start monitoring stack")
    up_parser.set_defaults(func=cmd_up)

    down_parser = subparsers.add_parser("down", help="Stop monitoring stack")
    down_parser.set_defaults(func=cmd_down)

    status_parser = subparsers.add_parser("status", help="Show stack status")
    status_parser.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
