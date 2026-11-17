#!/usr/bin/env python3
"""
Start the LFM3-1.7B observability stack (Jaeger + OpenTelemetry Collector).

Prerequisites:
- Docker and Docker Compose installed
- NPU server running (port 1239)
- CPU server running (port 1238)

Usage:
    python3 scripts/start_observability.py [--up|--down|--status]

After starting:
- Jaeger UI: http://localhost:16686
- OTLP HTTP endpoint: http://localhost:4318
- OTLP gRPC endpoint: http://localhost:4317

To enable tracing on the CPU server:
    export LFM3_TRACING_ENABLED=true
    export LFM3_OTLP_ENDPOINT=http://localhost:4318
    systemctl --user restart lfm3-npu-cpu.service
"""

import argparse
import os
import subprocess
import sys

COMPOSE_FILE = "/media/scott/data/lfm3-npu-tool/docker/observability.yml"


def check_docker():
    """Verify Docker and Docker Compose are available."""
    try:
        subprocess.run(["docker", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Docker not found. Install Docker first.")
        sys.exit(1)

    try:
        subprocess.run(
            ["docker", "compose", "version"], capture_output=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Docker Compose not found. Install Docker Compose first.")
        sys.exit(1)

    print("✓ Docker and Docker Compose available")


def cmd_up(args):
    """Start the observability stack."""
    check_docker()
    print("Starting observability stack...")
    subprocess.run(
        ["docker", "compose", "-f", COMPOSE_FILE, "up", "-d"],
        check=True,
    )
    print("\n✓ Observability stack started.")
    print("\nAccess points:")
    print("  Jaeger UI:        http://localhost:16686")
    print("  OTLP HTTP:        http://localhost:4318")
    print("  OTLP gRPC:        http://localhost:4317")
    print("  OTel metrics:     http://localhost:8888")
    print("\nTo enable tracing on the CPU server:")
    print("  export LFM3_TRACING_ENABLED=true")
    print("  export LFM3_OTLP_ENDPOINT=http://localhost:4318")
    print("  systemctl --user restart lfm3-npu-cpu.service")


def cmd_down(args):
    """Stop the observability stack."""
    check_docker()
    print("Stopping observability stack...")
    subprocess.run(
        ["docker", "compose", "-f", COMPOSE_FILE, "down"],
        check=True,
    )
    print("✓ Observability stack stopped.")


def cmd_status(args):
    """Show stack status."""
    check_docker()
    result = subprocess.run(
        ["docker", "compose", "-f", COMPOSE_FILE, "ps"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Manage LFM3-1.7B observability stack (Jaeger + OTel Collector)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    up_parser = subparsers.add_parser("up", help="Start observability stack")
    up_parser.set_defaults(func=cmd_up)

    down_parser = subparsers.add_parser("down", help="Stop observability stack")
    down_parser.set_defaults(func=cmd_down)

    status_parser = subparsers.add_parser("status", help="Show stack status")
    status_parser.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
