"""
Load balancer with health-based failover for LFM3-1.7B inference servers.

Provides a single entry point that:
  * Health-checks backend servers (CPU on 1238, NPU on 1239)
  * Routes traffic to the preferred backend (NPU) when healthy
  * Automatically fails over to the secondary backend (CPU) when primary is down
  * Supports sticky session affinity via X-Session-ID header
  * Exposes its own /health and /status endpoints for monitoring

Usage:
    python -m lfm3_npu.load_balancer --port 1240 --primary npu --secondary cpu

Or via CLI:
    lfm3-npu lb start --port 1240 --primary npu --secondary cpu
    lfm3-npu lb status
    lfm3-npu lb stop
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import web

# Backend configuration
CPU_HOST = "127.0.0.1"
CPU_PORT = 1238
NPU_HOST = "127.0.0.1"
NPU_PORT = 1239

HEALTH_CHECK_INTERVAL = 5.0  # seconds
HEALTH_CHECK_TIMEOUT = 3.0   # seconds
FAILURE_THRESHOLD = 3        # consecutive failures before marking unhealthy
RECOVERY_THRESHOLD = 2       # consecutive successes before marking healthy


@dataclass
class Backend:
    """Represents a backend inference server."""

    name: str
    host: str
    port: int
    url: str
    health_path: str = "/health"
    healthy: bool = False
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_check: float = 0.0
    last_error: Optional[str] = None
    total_requests: int = 0
    total_errors: int = 0
    total_latency_ms: float = 0.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}{self.health_path}"

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_requests if self.total_requests else 0.0

    @property
    def error_rate(self) -> float:
        return self.total_errors / self.total_requests if self.total_requests else 0.0


class LoadBalancer:
    """Health-aware load balancer with automatic failover."""

    def __init__(
        self,
        primary: str = "npu",
        secondary: str = "cpu",
        port: int = 1240,
        host: str = "0.0.0.0",
        health_interval: float = HEALTH_CHECK_INTERVAL,
    ):
        self.port = port
        self.host = host
        self.health_interval = health_interval

        # Define backends
        self.backends: Dict[str, Backend] = {
            "cpu": Backend("cpu", CPU_HOST, CPU_PORT, f"http://{CPU_HOST}:{CPU_PORT}", health_path="/health"),
            "npu": Backend("npu", NPU_HOST, NPU_PORT, f"http://{NPU_HOST}:{NPU_PORT}", health_path="/v1/models"),
        }

        # Routing preference
        self.primary_name = primary
        self.secondary_name = secondary

        # Sticky sessions: session_id -> backend_name
        self.sticky_sessions: Dict[str, str] = {}
        self.sticky_lock = asyncio.Lock()

        # State
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        _health_task: Optional[asyncio.Task] = None
        self._session = None

        # Start time
        self.start_time = time.time()

    # ------------------------------------------------------------------ #
    # Health checking                                                    #
    # ------------------------------------------------------------------ #

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def _check_backend_health(self, backend: Backend) -> bool:
        """Check a single backend's health. Returns True if healthy."""
        await self._ensure_session()
        try:
            start = time.monotonic()
            async with self._session.get(backend.health_url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    latency_ms = (time.monotonic() - start) * 1000
                    backend.consecutive_successes += 1
                    backend.consecutive_failures = 0
                    backend.last_error = None

                    # Mark healthy after recovery threshold
                    if not backend.healthy and backend.consecutive_successes >= RECOVERY_THRESHOLD:
                        backend.healthy = True
                        print(
                            f"[LB] Backend {backend.name} recovered (healthy)",
                            flush=True,
                        )

                    # Track latency
                    backend.total_requests += 1
                    backend.total_latency_ms += latency_ms
                    return True
                else:
                    return self._record_failure(backend, f"HTTP {resp.status}")
        except Exception as e:
            return self._record_failure(backend, str(e))

    def _record_failure(self, backend: Backend, error: str) -> bool:
        backend.consecutive_failures += 1
        backend.consecutive_successes = 0
        backend.last_error = error
        backend.total_requests += 1
        backend.total_errors += 1

        if backend.consecutive_failures >= FAILURE_THRESHOLD and backend.healthy:
            backend.healthy = False
            print(
                f"[LB] Backend {backend.name} marked UNHEALTHY: {error}",
                flush=True,
            )
        return False

    async def _health_loop(self):
        """Periodic health check loop for all backends."""
        print(f"[LB] Health check loop started (interval={self.health_interval}s)", flush=True)
        while True:
            try:
                await self._ensure_session()
                results = await asyncio.gather(
                    *(self._check_backend_health(b) for b in self.backends.values()),
                    return_exceptions=True,
                )
                for name, result in zip(self.backends.keys(), results):
                    if isinstance(result, Exception):
                        self._record_failure(self.backends[name], str(result))
            except Exception as e:  # noqa BLE001
                print(f"[LB] Health check error: {e}", flush=True)

            await asyncio.sleep(self.health_interval)

    # ------------------------------------------------------------------ #
    # Backend selection                                                  #
    # ------------------------------------------------------------------ #

    def _select_backend(self, request: web.Request) -> Optional[Backend]:
        """Select the best available backend for a request.

        Preference order:
          1. Sticky session affinity (if X-Session-ID header present)
          2. Primary backend if healthy
          3. Secondary backend if healthy
          4. Any healthy backend
          5. None (all unhealthy)
        """
        # Check sticky session first
        session_id = request.headers.get("X-Session-ID")
        if session_id:
            async def _get_sticky():
                async with self.sticky_lock:
                    return self.sticky_sessions.get(session_id)

            # Note: This is sync context; we check without lock for speed,
            # the lock is only for writes. Minor race is acceptable.
            sticky_backend = self.sticky_sessions.get(session_id)
            if sticky_backend and sticky_backend in self.backends:
                b = self.backends[sticky_backend]
                if b.healthy:
                    return b

        # Primary preferred
        primary = self.backends.get(self.primary_name)
        if primary and primary.healthy:
            return primary

        # Fall back to secondary
        secondary = self.backends.get(self.secondary_name)
        if secondary and secondary.healthy:
            return secondary

        # Any healthy backend
        for b in self.backends.values():
            if b.healthy:
                return b

        return None

    async def _bind_sticky_session(self, session_id: str, backend_name: str):
        async with self.sticky_lock:
            self.sticky_sessions[session_id] = backend_name

    # ------------------------------------------------------------------ #
    # Request proxying                                                   #
    # ------------------------------------------------------------------ #

    async def _proxy_request(self, request: web.Request, backend: Backend) -> web.StreamResponse:
        """Proxy a request to the selected backend."""
        await self._ensure_session()

        # Build target URL
        target_url = f"{backend.base_url}{request.path_qs}"

        # Prepare headers (forward relevant ones, skip hop-by-hop)
        headers = {}
        for key, value in request.headers.items():
            key_lower = key.lower()
            if key_lower in (
                "host",
                "content-length",
                "transfer-encoding",
                "connection",
                "upgrade",
            ):
                continue
            headers[key] = value

        # Add forwarding info
        headers["X-Forwarded-For"] = request.remote or "unknown"
        headers["X-Forwarded-Proto"] = request.scheme
        headers["X-Forwarded-Host"] = request.host

        # Read body if present
        body = None
        if request.can_read_body:
            body = await request.read()

        start = time.monotonic()
        try:
            async with self._session.request(
                method=request.method,
                url=target_url,
                headers=headers,
                data=body,
                allow_redirects=False,
            ) as resp:
                # Build response
                response = web.StreamResponse(status=resp.status, reason=resp.reason)
                response.headers.update(
                    {k: v for k, v in resp.headers.items() if k.lower() not in ("content-length", "transfer-encoding", "connection")}
                )
                await response.prepare(request)

                async for chunk in resp.content.iter_chunked(65536):
                    await response.write(chunk)

                await response.write_eof()

                # Track metrics
                latency_ms = (time.monotonic() - start) * 1000
                backend.total_requests += 1
                backend.total_latency_ms += latency_ms
                if resp.status >= 500:
                    backend.total_errors += 1

                # Bind sticky session if header present
                session_id = request.headers.get("X-Session-ID")
                if session_id:
                    await self._bind_sticky_session(session_id, backend.name)

                return response

        except Exception as e:  # noqa BLE001
            backend.total_requests += 1
            backend.total_errors += 1
            self._record_failure(backend, str(e))

            # Try the other healthy backend as a fallback
            fallback = None
            for b in self.backends.values():
                if b.name != backend.name and b.healthy:
                    fallback = b
                    break

            if fallback:
                print(f"[LB] Retrying on {fallback.name} after {backend.name} failure", flush=True)
                return await self._proxy_request(request, fallback)

            return web.json_response(
                {"error": {"message": f"Backend {backend.name} unavailable", "type": "backend_error"}},
                status=503,
            )

    # ------------------------------------------------------------------ #
    # Web handlers                                                       #
    # ------------------------------------------------------------------ #

    async def _handle_proxy(self, request: web.Request) -> web.StreamResponse:
        """Main proxy handler — routes to the best available backend."""
        backend = self._select_backend(request)
        if backend is None:
            return web.json_response(
                {
                    "error": {
                        "message": "No healthy backends available",
                        "type": "service_unavailable",
                    }
                },
                status=503,
            )
        return await self._proxy_request(request, backend)

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Load balancer health endpoint."""
        backend_status = {}
        any_healthy = False
        for name, b in self.backends.items():
            backend_status[name] = {
                "healthy": b.healthy,
                "url": b.base_url,
                "consecutive_failures": b.consecutive_failures,
                "consecutive_successes": b.consecutive_successes,
                "last_error": b.last_error,
                "total_requests": b.total_requests,
                "total_errors": b.total_errors,
                "avg_latency_ms": round(b.avg_latency_ms, 2),
                "error_rate": round(b.error_rate, 4),
            }
            if b.healthy:
                any_healthy = True

        return web.json_response(
            {
                "status": "healthy" if any_healthy else "critical",
                "primary": self.primary_name,
                "secondary": self.secondary_name,
                "uptime_seconds": round(time.time() - self.start_time, 1),
                "backends": backend_status,
            }
        )

    async def _handle_status(self, request: web.Request) -> web.Response:
        """Detailed status endpoint."""
        return await self._handle_health(request)

    async def _handle_routes(self, request: web.Request) -> web.Response:
        """Return current routing table."""
        primary = self.backends.get(self.primary_name)
        secondary = self.backends.get(self.secondary_name)
        selected = self._select_backend(request)

        return web.json_response(
            {
                "primary": {"name": self.primary_name, "healthy": primary.healthy if primary else False},
                "secondary": {"name": self.secondary_name, "healthy": secondary.healthy if secondary else False},
                "selected_for_request": selected.name if selected else None,
                "sticky_sessions": len(self.sticky_sessions),
            }
        )

    # ------------------------------------------------------------------ #
    # Server lifecycle                                                   #
    # ------------------------------------------------------------------ #

    def _build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self._handle_proxy)
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/status", self._handle_status)
        app.router.add_get("/routes", self._handle_routes)
        return app

    async def start(self):
        """Start the load balancer server."""
        app = self._build_app()
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

        # Start health check loop
        self._health_task = asyncio.create_task(self._health_loop())

        print(
            f"[LB] Load balancer listening on {self.host}:{self.port} "
            f"(primary={self.primary_name}, secondary={self.secondary_name})",
            flush=True,
        )

    async def stop(self):
        """Stop the load balancer server."""
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        if self._session and not self._session.closed:
            await self._session.close()
        print("[LB] Load balancer stopped", flush=True)


# ---------------------------------------------------------------------- #
# CLI entry point                                                        #
# ---------------------------------------------------------------------- #

async def _async_main():
    parser = argparse.ArgumentParser(description="LFM3-1.7B Load Balancer")
    parser.add_argument("--port", type=int, default=1240, help="Listen port")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host")
    parser.add_argument("--primary", default="npu", choices=["cpu", "npu"], help="Primary backend")
    parser.add_argument("--secondary", default="cpu", choices=["cpu", "npu"], help="Secondary backend")
    parser.add_argument("--health-interval", type=float, default=5.0, help="Health check interval (s)")
    args = parser.parse_args()

    lb = LoadBalancer(
        primary=args.primary,
        secondary=args.secondary,
        port=args.port,
        host=args.host,
        health_interval=args.health_interval,
    )

    # Signal handlers
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _request_stop(sig):
        print(f"[LB] Received signal {sig}, shutting down...", flush=True)
        stop_event.set()

    try:
        loop.add_signal_handler(signal.SIGTERM, _request_stop, "SIGTERM")
        loop.add_signal_handler(signal.SIGINT, _request_stop, "SIGINT")
    except (NotImplementedError, OSError):
        pass

    await lb.start()

    # Wait for stop signal
    await stop_event.wait()

    await lb.stop()


def main():
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
