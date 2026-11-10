"""
Server management for LFM3-1.7B inference servers.

Provides start/stop/restart/status for CPU and NPU inference servers
using subprocess management with PID files.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import psutil
import urllib.request

from .config import Config

PID_DIR: Path = Path("/media/scott/data/lfm3-npu-tool/var/run")
LOG_DIR: Path = Path("/media/scott/data/lfm3-npu-tool/var/log")


def ensure_dirs() -> None:
    """Create required runtime directories."""
    PID_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def pid_file(server: str) -> Path:
    """Get the PID file path for a server."""
    return PID_DIR / f"{server}.pid"


def log_file(server: str) -> Path:
    """Get the log file path for a server."""
    return LOG_DIR / f"{server}.log"


def is_running(server: str) -> bool:
    """Check if a server process is running."""
    pfile = pid_file(server)
    if not pfile.exists():
        return False
    try:
        pid = int(pfile.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError, PermissionError):
        return False


def get_process(server: str) -> Optional[psutil.Process]:
    """Get the server process if running."""
    pfile = pid_file(server)
    if not pfile.exists():
        return None
    try:
        pid = int(pfile.read_text().strip())
        return psutil.Process(pid)
    except (ProcessLookupError, ValueError, PermissionError):
        return None


class ServerManager:
    """Manages CPU and NPU inference servers."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config({})
        ensure_dirs()

    def get_process(self, server: str) -> Optional[psutil.Process]:
        """Get the server process if running."""
        return get_process(server)

    def _get_env(self) -> dict:
        """Get environment variables for NPU subprocesses."""
        env = os.environ.copy()
        ld_parts = self.config.get("npu.ld_library_path", [])
        ld_parts.append(env.get("LD_LIBRARY_PATH", ""))
        env["LD_LIBRARY_PATH"] = ":".join([p for p in ld_parts if p])
        env["TMPDIR"] = "/media/scott/data/tmp"
        return env

    def start(self, server: str) -> bool:
        """Start a server.

        Args:
            server: 'cpu' or 'npu'.

        Returns:
            True if server started successfully.
        """
        if is_running(server):
            pid = pid_file(server).read_text().strip()
            print(f"{server.upper()} server is already running (PID: {pid})")
            return True

        if server == "cpu":
            server_script = "/media/scott/data/lfm3-npu-tool/lib/lfm3_npu/cpu_server.py"
            port = self.config.get("server.cpu.port", 1238)
            host = self.config.get("server.cpu.host", "0.0.0.0")
            python_exec = self.config.get(
                "npu.python_executable", "/media/scott/data/ryzen-ai-venv/bin/python"
            )
            cmd = [python_exec, server_script, "--host", host, "--port", str(port)]
        elif server == "npu":
            server_script = "/media/scott/data/lfm3-npu-tool/lib/lfm3_npu/npu_server.py"
            port = self.config.get("server.npu.port", 1239)
            host = self.config.get("server.npu.host", "0.0.0.0")
            python_exec = self.config.get(
                "npu.python_executable", "/media/scott/data/ryzen-ai-venv/bin/python"
            )
            cmd = [python_exec, server_script, "--host", host, "--port", str(port)]
        else:
            print(f"Unknown server: {server}. Use 'cpu' or 'npu'.")
            return False

        logf = log_file(server).open("w")
        pidf = pid_file(server).open("w")

        print(f"Starting {server.upper()} server on {host}:{port}...", flush=True)
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=self._get_env(),
            cwd="/media/scott/data/lfm3-npu-tool",
            start_new_session=True,
        )
        pidf.write(str(proc.pid))
        pidf.close()

        # Wait for server to be ready
        for _ in range(30):
            time.sleep(1)
            try:
                req = urllib.request.urlopen(
                    f"http://{host}:{port}/health", timeout=2
                )
                if req.status == 200:
                    print(
                        f"{server.upper()} server is ready (PID: {proc.pid})",
                        flush=True,
                    )
                    return True
            except Exception:
                pass

        print(
            f"{server.upper()} server started (PID: {proc.pid}) but may still be "
            f"initializing. Check logs: {log_file(server)}",
            flush=True,
        )
        return True

    def stop(self, server: str) -> bool:
        """Stop a server.

        Args:
            server: 'cpu' or 'npu'.

        Returns:
            True if server was stopped.
        """
        proc = get_process(server)
        if proc is None:
            print(f"{server.upper()} server is not running.")
            pfile = pid_file(server)
            if pfile.exists():
                pfile.unlink()
            return False

        print(f"Stopping {server.upper()} server (PID: {proc.pid})...", flush=True)
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except psutil.TimeoutExpired:
            proc.kill()
            proc.wait()

        pfile = pid_file(server)
        if pfile.exists():
            pfile.unlink()
        print(f"{server.upper()} server stopped.", flush=True)
        return True

    def restart(self, server: str) -> bool:
        """Restart a server."""
        self.stop(server)
        time.sleep(1)
        return self.start(server)

    def status(self, server: Optional[str] = None) -> None:
        """Show server status.

        Args:
            server: 'cpu', 'npu', or None for both.
        """
        servers = [server] if server else ["cpu", "npu"]
        for s in servers:
            proc = get_process(s)
            if proc:
                try:
                    cpu_pct = proc.cpu_percent(interval=0.5)
                    mem_mb = proc.memory_info().rss / 1024 / 1024
                    print(
                        f"{s.upper()} server: RUNNING (PID: {proc.pid}, "
                        f"CPU: {cpu_pct:.1f}%, MEM: {mem_mb:.0f} MB)"
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    print(f"{s.upper()} server: RUNNING (PID: {proc.pid})")
            else:
                print(f"{s.upper()} server: NOT RUNNING")