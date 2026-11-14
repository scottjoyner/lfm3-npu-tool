"""
Server management for LFM3-1.7B inference servers.

Provides start/stop/restart/status for CPU and NPU inference servers
using subprocess management with PID files.

CPU server: ONNX Runtime CPUExecutionProvider (port 1238)
NPU server: FastFlowLM v1.0.12 with liblfm3_npu.so (port 1239)

Both servers expose OpenAI-compatible HTTP APIs.
"""

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import urllib.request
import urllib.error

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
    """Check if a server process is running via PID file."""
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
    """Get the server process if running via PID file."""
    pfile = pid_file(server)
    if not pfile.exists():
        return None
    try:
        pid = int(pfile.read_text().strip())
        return psutil.Process(pid)
    except (ProcessLookupError, ValueError, PermissionError, psutil.NoSuchProcess):
        # Stale PID file — try to find the process by command line
        return find_server_process(server)


def find_server_process(server: str) -> Optional[psutil.Process]:
    """Find a server process by command line, even without PID file.

    Looks for known process patterns:
    - cpu: cpu_batched_server.py --port 1238
    - npu: flm serve lfm3:1.7b -p 1239
    """
    patterns = {
        "cpu": ["cpu_batched_server.py"],
        "npu": ["flm", "serve", "lfm3:1.7b"],
    }
    pattern_list = patterns.get(server, [])
    if not pattern_list:
        return None

    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info["cmdline"]
            if not cmdline:
                continue
            cmdline_str = " ".join(cmdline)
            if all(p in cmdline_str for p in pattern_list):
                return psutil.Process(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
            continue
    return None


def get_server_pid(server: str) -> Optional[int]:
    """Get server PID from PID file or by finding the process."""
    proc = get_process(server)
    if proc:
        return proc.pid
    proc = find_server_process(server)
    if proc:
        return proc.pid
    return None


class ServerManager:
    """Manages CPU and NPU inference servers."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config({})
        ensure_dirs()

    def get_process(self, server: str) -> Optional[psutil.Process]:
        """Get the server process if running (PID file or found process)."""
        proc = get_process(server)
        if proc:
            return proc
        return find_server_process(server)

    def _get_cpu_env(self) -> dict:
        """Get environment variables for CPU server subprocess."""
        env = os.environ.copy()
        ld_parts = self.config.get("npu.ld_library_path", [])
        ld_parts.append(env.get("LD_LIBRARY_PATH", ""))
        env["LD_LIBRARY_PATH"] = ":".join([p for p in ld_parts if p])
        env["TMPDIR"] = "/media/scott/data/tmp"
        env["XLNX_CACHE_DIR"] = "/media/scott/data/vitisai_cache"
        return env

    def _get_npu_env(self) -> dict:
        """Get environment variables for NPU (FLM) server subprocess."""
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/tmp/lib:/opt/xilinx/xrt/lib:" + env.get(
            "LD_LIBRARY_PATH", ""
        )
        env["XILINX_XRT"] = "/opt/xilinx/xrt"
        env["FLM_MODEL_PATH"] = "/home/scott/.config/flm/models"
        env["TMPDIR"] = "/media/scott/data/tmp"
        return env

    def _http_health(self, host: str, port: int, path: str = "/health") -> bool:
        """Check HTTP health endpoint."""
        try:
            req = urllib.request.urlopen(
                f"http://{host}:{port}{path}", timeout=5
            )
            return req.status == 200
        except Exception:
            return False

    def _wait_for_ready(self, host: str, port: int, timeout: int = 60,
                        health_path: str = "/health") -> bool:
        """Wait for server to respond to health check."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._http_health(host, port, health_path):
                return True
            time.sleep(1)
        return False

    def start(self, server: str) -> bool:
        """Start a server.

        Args:
            server: 'cpu' or 'npu'.

        Returns:
            True if server started successfully.
        """
        # Check if already running (PID file or found process)
        proc = self.get_process(server)
        if proc:
            pid = proc.pid
            print(f"{server.upper()} server is already running (PID: {pid})")
            return True

        if server == "cpu":
            port = self.config.get("server.cpu.port", 1238)
            host = self.config.get("server.cpu.host", "0.0.0.0")
            python_exec = self.config.get(
                "npu.python_executable", "/media/scott/data/ryzen-ai-venv/bin/python"
            )
            server_script = "/media/scott/data/lfm3-npu-tool/lib/lfm3_npu/cpu_batched_server.py"
            cmd = [python_exec, "-B", server_script, "--host", host, "--port", str(port)]
            env = self._get_cpu_env()
            health_path = "/health"
            logf_path = log_file("cpu")
        elif server == "npu":
            port = self.config.get("server.npu.port", 1239)
            host = self.config.get("server.npu.host", "127.0.0.1")
            flm_binary = self.config.get("npu.flm_binary", "/tmp/flm")
            cmd = [flm_binary, "serve", "lfm3:1.7b", "-p", str(port), "--host", host]
            env = self._get_npu_env()
            health_path = "/v1/models"
            logf_path = log_file("npu")
        else:
            print(f"Unknown server: {server}. Use 'cpu' or 'npu'.")
            return False

        logf = logf_path.open("w")
        pidf = pid_file(server).open("w")

        print(f"Starting {server.upper()} server on {host}:{port}...", flush=True)
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            cwd="/media/scott/data/lfm3-npu-tool",
            start_new_session=True,
        )
        pidf.write(str(proc.pid))
        pidf.close()

        # Wait for server to be ready
        ready = self._wait_for_ready(host, port, health_path=health_path)
        if ready:
            print(
                f"{server.upper()} server is ready (PID: {proc.pid})",
                flush=True,
            )
            return True
        else:
            print(
                f"{server.upper()} server started (PID: {proc.pid}) but may still be "
                f"initializing. Check logs: {logf_path}",
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
        proc = self.get_process(server)
        if proc is None:
            # Try finding by port
            pid = get_server_pid(server)
            if pid is None:
                print(f"{server.upper()} server is not running.")
                pfile = pid_file(server)
                if pfile.exists():
                    pfile.unlink()
                return False
            try:
                proc = psutil.Process(pid)
            except psutil.NoSuchProcess:
                print(f"{server.upper()} server is not running.")
                return False

        pid = proc.pid
        print(f"Stopping {server.upper()} server (PID: {pid})...", flush=True)
        try:
            # Kill the entire process group (start_new_session=True creates a new group)
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            proc.wait(timeout=15)
        except psutil.TimeoutExpired:
            print(f"{server.upper()} server did not terminate, killing...", flush=True)
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        except ProcessLookupError:
            pass

        pfile = pid_file(server)
        if pfile.exists():
            pfile.unlink()
        print(f"{server.upper()} server stopped.", flush=True)
        return True

    def restart(self, server: str) -> bool:
        """Restart a server."""
        self.stop(server)
        time.sleep(2)
        return self.start(server)

    def status(self, server: Optional[str] = None) -> List[Dict[str, Any]]:
        """Show server status with health info.

        Args:
            server: 'cpu', 'npu', or None for both.

        Returns:
            List of status dicts for each server.
        """
        servers = [server] if server else ["cpu", "npu"]
        results: List[Dict[str, Any]] = []
        for s in servers:
            entry: Dict[str, Any] = {"server": s}
            proc = self.get_process(s)
            if proc:
                entry["running"] = True
                entry["pid"] = proc.pid
                try:
                    entry["cpu_percent"] = proc.cpu_percent(interval=0.5)
                    entry["memory_mb"] = proc.memory_info().rss / 1024 / 1024
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                # Check health
                if s == "cpu":
                    port = self.config.get("server.cpu.port", 1238)
                    host = self.config.get("server.cpu.host", "0.0.0.0")
                    health_path = "/health"
                else:
                    port = self.config.get("server.npu.port", 1239)
                    host = self.config.get("server.npu.host", "127.0.0.1")
                    health_path = "/v1/models"
                entry["healthy"] = self._http_health(host, port, health_path)
                if entry["healthy"]:
                    try:
                        req = urllib.request.urlopen(
                            f"http://{host}:{port}{health_path}", timeout=5
                        )
                        import json
                        data = json.loads(req.read().decode("utf-8"))
                        entry["health_data"] = data
                    except Exception:
                        pass
            else:
                entry["running"] = False
                entry["healthy"] = False
            results.append(entry)
            self._print_status_entry(entry)
        return results

    def _print_status_entry(self, entry: Dict[str, Any]) -> None:
        """Print a single server status entry."""
        s = entry["server"]
        if entry.get("running"):
            pid = entry.get("pid", "?")
            mem = entry.get("memory_mb", 0)
            cpu = entry.get("cpu_percent", 0)
            healthy = entry.get("healthy", False)
            health_str = "HEALTHY" if healthy else "UNHEALTHY"
            print(
                f"{s.upper()} server: RUNNING (PID: {pid}, "
                f"CPU: {cpu:.1f}%, MEM: {mem:.0f} MB) - {health_str}"
            )
            # Print key health metrics
            hd = entry.get("health_data", {})
            if s == "cpu" and hd:
                uptime = hd.get("uptime_seconds", 0)
                tps = hd.get("avg_decode_ms", 0)
                print(f"  Uptime: {uptime:.0f}s, Avg decode: {tps:.1f}ms")
                model = hd.get("model", "unknown")
                print(f"  Model: {model}, Port: {hd.get('port', '?')}")
            elif s == "npu" and hd:
                models = hd.get("data", [])
                for m in models:
                    if "lfm3" in str(m.get("id", "")):
                        print(f"  Model: {m.get('id')} ({m.get('owned_by', '?')})")
                        break
        else:
            print(f"{s.upper()} server: NOT RUNNING")
