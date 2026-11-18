#!/usr/bin/env python3
"""Fail-closed systemd ExecCondition for the shared LFM3 CPU port.

Exit 0 only when a reliable local listener inventory proves port 1238 free.
Exit 1 (unit skipped, not auto-restarted) on occupancy or unverifiable state.
No processes are killed and no HTTP request or model invocation occurs.
"""
from __future__ import annotations

import subprocess


def main() -> int:
    try:
        check = subprocess.run(
            ["ss", "-H", "-ltn", "( sport = :1238 )"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        print("HOLD: CPU port ownership could not be verified", flush=True)
        return 1
    if check.returncode != 0 or check.stdout.strip():
        print("HOLD: CPU port 1238 occupied or unverifiable", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
