#!/usr/bin/env python3
"""
LFM3-1.7B Unified Health Check Script.

Checks the health of both CPU (port 1238) and NPU (port 1239) servers
and returns a unified status. Intended for use with load balancers,
monitoring systems, and container health checks.

Usage:
    python3 health_check.py [--server cpu|npu|both] [--exit-on-unhealthy]

Exit codes:
    0 - All specified servers are healthy
    1 - One or more servers are unhealthy/unreachable
    2 - Invalid arguments
"""

import argparse
import json
import sys
import urllib.request
import urllib.error

CPU_HOST = "127.0.0.1"
CPU_PORT = 1238
NPU_HOST = "127.0.0.1"
NPU_PORT = 1239


def fetch_json(url: str, timeout: float = 5.0) -> dict | None:
    """Fetch JSON from URL, return None on failure."""
    try:
        req = urllib.request.urlopen(url, timeout=timeout)
        return json.loads(req.read().decode("utf-8"))
    except Exception:
        return None


def check_cpu_health() -> dict:
    """Check CPU server health via /health endpoint."""
    result = {
        "server": "cpu",
        "port": CPU_PORT,
        "healthy": False,
        "details": {},
    }
    try:
        data = fetch_json(f"http://{CPU_HOST}:{CPU_PORT}/health", timeout=5)
        if data and data.get("status") == "healthy":
            result["healthy"] = True
            result["details"] = {
                "provider": data.get("provider", "unknown"),
                "model": data.get("model", "unknown"),
                "uptime_seconds": data.get("uptime_seconds", 0),
                "avg_decode_ms": data.get("avg_decode_ms", 0),
                "active_slots": data.get("active_slots", 0),
                "free_slots": data.get("free_slots", 0),
                "port": data.get("port", CPU_PORT),
            }
    except Exception as e:
        result["details"] = {"error": str(e)}
    return result


def check_npu_health() -> dict:
    """Check NPU server health via /v1/models endpoint."""
    result = {
        "server": "npu",
        "port": NPU_PORT,
        "healthy": False,
        "details": {},
    }
    try:
        data = fetch_json(f"http://{NPU_HOST}:{NPU_PORT}/v1/models", timeout=5)
        if data and "data" in data and len(data["data"]) > 0:
            result["healthy"] = True
            lfm_models = [m for m in data["data"] if "lfm3" in str(m.get("id", ""))]
            result["details"] = {
                "provider": "FastFlowLM",
                "total_models": len(data["data"]),
                "lfm3_models": [m.get("id", "unknown") for m in lfm_models],
            }
    except Exception as e:
        result["details"] = {"error": str(e)}
    return result


def main():
    parser = argparse.ArgumentParser(
        description="LFM3-1.7B Unified Health Check"
    )
    parser.add_argument(
        "--server", choices=["cpu", "npu", "both"], default="both",
        help="Server(s) to check (default: both)"
    )
    parser.add_argument(
        "--exit-on-unhealthy", action="store_true",
        help="Exit with code 1 if any server is unhealthy"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output JSON format"
    )
    args = parser.parse_args()

    servers = ["cpu", "npu"] if args.server == "both" else [args.server]

    results = []
    for server in servers:
        if server == "cpu":
            results.append(check_cpu_health())
        elif server == "npu":
            results.append(check_npu_health())

    if args.json:
        output = {
            "healthy": all(r["healthy"] for r in results),
            "servers": results,
        }
        print(json.dumps(output, indent=2))
        all_healthy = output["healthy"]
    else:
        all_healthy = True
        for r in results:
            status = "HEALTHY" if r["healthy"] else "UNREACHABLE"
            print(f"{r['server'].upper()} (port {r['port']}): {status}")
            for key, value in r["details"].items():
                print(f"  {key}: {value}")
            if not r["healthy"]:
                all_healthy = False

    if args.exit_on_unhealthy and not all_healthy:
        sys.exit(1)

    sys.exit(0 if all_healthy else 1)


if __name__ == "__main__":
    main()
