"""
Enterprise-grade middleware for LFM3-1.7B inference servers.

Provides:
  * API key authentication (Bearer token / X-API-Key header)
  * Token-bucket rate limiting per client (IP or API key)
  * Circuit breaker around inference calls for resilience
  * Structured JSON logging with correlation IDs

All components are dependency-free (stdlib + FastAPI) so they can be
dropped into any FastAPI-based inference server without new pip packages.
"""

import hashlib
import json
import logging
import sys
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from typing import Callable, Dict, Optional

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------

_logger: Optional[logging.Logger] = None


def get_logger(name: str = "lfm3.enterprise") -> logging.Logger:
    """Return a structured JSON logger (configured once)."""
    global _logger
    if _logger is None:
        _logger = logging.getLogger(name)
        _logger.setLevel(logging.INFO)
        if not _logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(_JsonFormatter())
            _logger.addHandler(handler)
        _logger.propagate = False
    return _logger


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Carry extra structured fields (request_id, latency_ms, etc.)
        for key in (
            "request_id",
            "client_ip",
            "api_key_hash",
            "endpoint",
            "method",
            "status_code",
            "latency_ms",
            "provider",
            "model",
            "prompt_tokens",
            "completion_tokens",
            "decode_tps",
            "event",
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"))


def log_event(
    event: str,
    *,
    request_id: Optional[str] = None,
    client_ip: Optional[str] = None,
    api_key_hash: Optional[str] = None,
    endpoint: Optional[str] = None,
    method: Optional[str] = None,
    status_code: Optional[int] = None,
    latency_ms: Optional[float] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    decode_tps: Optional[float] = None,
    extra: Optional[Dict] = None,
) -> None:
    """Emit a structured log event."""
    logger = get_logger()
    extra_fields: Dict = {}
    if request_id is not None:
        extra_fields["request_id"] = request_id
    if client_ip is not None:
        extra_fields["client_ip"] = client_ip
    if api_key_hash is not None:
        extra_fields["api_key_hash"] = api_key_hash
    if endpoint is not None:
        extra_fields["endpoint"] = endpoint
    if method is not None:
        extra_fields["method"] = method
    if status_code is not None:
        extra_fields["status_code"] = status_code
    if latency_ms is not None:
        extra_fields["latency_ms"] = round(latency_ms, 3)
    if provider is not None:
        extra_fields["provider"] = provider
    if model is not None:
        extra_fields["model"] = model
    if prompt_tokens is not None:
        extra_fields["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        extra_fields["completion_tokens"] = completion_tokens
    if decode_tps is not None:
        extra_fields["decode_tps"] = round(decode_tps, 2)
    if extra:
        extra_fields.update(extra)
    logger.info(event, extra=extra_fields)


# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------


def _hash_key(key: str) -> str:
    """Return a one-way hash of an API key for safe logging/comparison."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


class APIKeyAuth:
    """
    API key authentication.

    Keys are loaded from the ``LFM3_API_KEYS`` environment variable as a
    comma-separated list.  Each key may optionally carry a label after a
    colon, e.g. ``sk-live-abc123:prod-web``.  When no keys are configured,
    authentication is disabled (open access).
    """

    def __init__(self) -> None:
        raw = __import__("os").environ.get("LFM3_API_KEYS", "")
        self._keys: Dict[str, str] = {}  # hashed_key -> label
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            key, _, label = entry.partition(":")
            key = key.strip()
            if key:
                self._keys[_hash_key(key)] = label.strip() or "unlabeled"
        self.enabled = bool(self._keys)

    def authenticate(self, api_key: Optional[str]) -> Optional[str]:
        """
        Validate an API key.  Returns the key label on success, None on
        failure.
        """
        if not self.enabled:
            return "anonymous"
        if not api_key:
            return None
        return self._keys.get(_hash_key(api_key))


def _extract_api_key(request: Request) -> Optional[str]:
    """Extract API key from Authorization: Bearer or X-API-Key header."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key")


# ---------------------------------------------------------------------------
# Token bucket rate limiter (per client)
# ---------------------------------------------------------------------------


class TokenBucket:
    """A simple thread-safe token bucket."""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate          # tokens per second
        self.capacity = float(capacity)
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume ``tokens``.  Returns True if allowed."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def retry_after(self) -> float:
        """Seconds until at least one token is available."""
        with self._lock:
            if self._tokens >= 1:
                return 0.0
            return (1.0 - self._tokens) / self.rate


class RateLimiter:
    """
    Per-client token bucket rate limiter.

    Clients are identified by API key (when auth is enabled) or by client IP.
    Configuration comes from env: ``LFM3_RATE_LIMIT_RPM`` (requests per
    minute, default 60) and ``LFM3_RATE_LIMIT_BURST`` (default 10).
    """

    def __init__(self) -> None:
        rpm = float(__import__("os").environ.get("LFM3_RATE_LIMIT_RPM", "60"))
        burst = int(__import__("os").environ.get("LFM3_RATE_LIMIT_BURST", "10"))
        self.rate = rpm / 60.0
        self.capacity = burst
        self._buckets: "OrderedDict[str, TokenBucket]" = OrderedDict()
        self._lock = threading.Lock()
        self._max_buckets = 10_000

    def _client_key(self, request: Request, api_key_label: Optional[str]) -> str:
        if api_key_label:
            return f"key:{api_key_label}"
        return f"ip:{request.client.host if request.client else 'unknown'}"

    def allow(self, request: Request, api_key_label: Optional[str]) -> Optional[float]:
        """
        Returns None if the request is allowed, or a retry-after seconds
        float if it is rate-limited.
        """
        key = self._client_key(request, api_key_label)
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(self.rate, self.capacity)
                if len(self._buckets) >= self._max_buckets:
                    self._buckets.popitem(last=False)
                self._buckets[key] = bucket
        if bucket.consume(1):
            return None
        return bucket.retry_after()


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """
    Circuit breaker for inference calls.

    States: CLOSED (normal) -> OPEN (failing fast) -> HALF_OPEN (trial).
    Opens after ``failure_threshold`` consecutive failures; after
    ``recovery_timeout`` seconds allows a single trial call; if the trial
    succeeds the circuit closes, otherwise it re-opens.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exception: type = Exception,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self._state = self.CLOSED
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @contextmanager
    def call(self, name: str = "inference"):
        """
        Context manager.  Raises ``CircuitOpenError`` when the circuit is
        open.  On expected exceptions increments failure count and may open
        the circuit.
        """
        with self._lock:
            now = time.monotonic()
            if self._state == self.OPEN:
                if self._opened_at is not None and now - self._opened_at >= self.recovery_timeout:
                    self._state = self.HALF_OPEN
                else:
                    raise CircuitOpenError(name, self.recovery_timeout)
            elif self._state == self.HALF_OPEN:
                pass  # allow single trial
        try:
            yield
        except self.expected_exception:
            self._record_failure()
            raise
        else:
            self._record_success()

    def _record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == self.HALF_OPEN or self._failures >= self.failure_threshold:
                self._state = self.OPEN
                self._opened_at = time.monotonic()

    def _record_success(self) -> None:
        with self._lock:
            self._state = self.CLOSED
            self._failures = 0
            self._opened_at = None

    def reset(self) -> None:
        with self._lock:
            self._state = self.CLOSED
            self._failures = 0
            self._opened_at = None


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open."""

    def __init__(self, name: str, retry_after: float):
        self.name = name
        self.retry_after = retry_after
        super().__init__(f"Circuit '{name}' is open; retry after {retry_after:.1f}s")


# ---------------------------------------------------------------------------
# Enterprise middleware (auth + rate limit + structured logging)
# ---------------------------------------------------------------------------


class EnterpriseMiddleware(BaseHTTPMiddleware):
    """
    Combined middleware: API key auth, rate limiting, structured JSON
    access logging, and correlation IDs.

    Add to a FastAPI app::

        app.add_middleware(EnterpriseMiddleware,
                           auth=APIKeyAuth(),
                           limiter=RateLimiter(),
                           breaker=None)   # breaker applied per-route instead
    """

    def __init__(
        self,
        app,
        auth: Optional[APIKeyAuth] = None,
        limiter: Optional[RateLimiter] = None,
        exclude_paths: Optional[tuple] = None,
    ) -> None:
        super().__init__(app)
        self.auth = auth or APIKeyAuth()
        self.limiter = limiter or RateLimiter()
        self.exclude_paths = exclude_paths or ("/health", "/live", "/ready", "/docs", "/openapi.json")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())[:12]
        start = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"
        api_key = _extract_api_key(request)
        api_key_label = self.auth.authenticate(api_key)
        api_key_hash = _hash_key(api_key) if api_key else None

        # Authentication
        if self.auth.enabled and api_key_label is None:
            log_event(
                "auth.rejected",
                request_id=request_id,
                client_ip=client_ip,
                api_key_hash=api_key_hash,
                endpoint=request.url.path,
                method=request.method,
                status_code=401,
            )
            return JSONResponse(
                {"error": {"message": "Invalid or missing API key", "type": "auth_error"}},
                status_code=401,
                headers={"X-Request-Id": request_id},
            )

        # Rate limiting (skip for health endpoints)
        if request.url.path not in self.exclude_paths:
            retry_after = self.limiter.allow(request, api_key_label)
            if retry_after is not None:
                log_event(
                    "rate_limited",
                    request_id=request_id,
                    client_ip=client_ip,
                    api_key_hash=api_key_hash,
                    endpoint=request.url.path,
                    method=request.method,
                    status_code=429,
                )
                return JSONResponse(
                    {"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
                    status_code=429,
                    headers={
                        "Retry-After": f"{retry_after:.1f}",
                        "X-Request-Id": request_id,
                    },
                )

        # Attach request_id + api_key to request state for downstream use
        request.state.request_id = request_id
        request.state.api_key_label = api_key_label
        request.state.api_key_hash = api_key_hash

        try:
            response = await call_next(request)
        except Exception as exc:  # pragma: no cover - safety net
            latency_ms = (time.perf_counter() - start) * 1000
            log_event(
                "request.error",
                request_id=request_id,
                client_ip=client_ip,
                api_key_hash=api_key_hash,
                endpoint=request.url.path,
                method=request.method,
                status_code=500,
                latency_ms=latency_ms,
                event="server_error",
                extra={"error": str(exc)},
            )
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-Id"] = request_id
        log_event(
            "request.completed",
            request_id=request_id,
            client_ip=client_ip,
            api_key_hash=api_key_hash,
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )
        return response


def require_api_key(request: Request) -> None:
    """
    Dependency-style helper to enforce auth on a specific route.  Returns
    the API key label or raises HTTPException(401).
    """
    auth: Optional[APIKeyAuth] = getattr(request.app.state, "enterprise_auth", None)
    if auth is None or not auth.enabled:
        return
    api_key = _extract_api_key(request)
    label = auth.authenticate(api_key)
    if label is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    request.state.api_key_label = label
    request.state.api_key_hash = _hash_key(api_key) if api_key else None


# Backwards-friendly namespace
EnterpriseAuth = APIKeyAuth
