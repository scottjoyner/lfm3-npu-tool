"""
Prometheus metrics for LFM3-1.7B inference servers.

Exposes inference latency, token throughput, and error counters
via a /metrics HTTP endpoint.
"""

import time
from threading import Lock
from typing import Dict, Optional

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
except ImportError:
    # Prometheus client not available - metrics will be no-ops
    Counter = Histogram = Gauge = None
    generate_latest = None
    CONTENT_TYPE_LATEST = "text/plain"


class MetricsCollector:
    """Collects and exposes Prometheus metrics for inference servers."""

    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self._lock = Lock()

        if Counter is None:
            self._enabled = False
            return

        self._enabled = True

        # Counters
        self.requests_total = Counter(
            "lfm3_requests_total",
            "Total number of inference requests",
            ["provider", "model", "endpoint"],
        )
        self.requests_errors_total = Counter(
            "lfm3_requests_errors_total",
            "Total number of failed inference requests",
            ["provider", "model", "endpoint"],
        )
        self.tokens_generated_total = Counter(
            "lfm3_tokens_generated_total",
            "Total number of tokens generated",
            ["provider", "model"],
        )
        self.tokens_prefilled_total = Counter(
            "lfm3_tokens_prefilled_total",
            "Total number of tokens prefilled",
            ["provider", "model"],
        )

        # Histograms
        self.prefill_duration_seconds = Histogram(
            "lfm3_prefill_duration_seconds",
            "Time spent in prefill phase",
            ["provider", "model"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )
        self.decode_duration_seconds = Histogram(
            "lfm3_decode_duration_seconds",
            "Time spent in decode phase",
            ["provider", "model"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
        )
        self.token_generation_duration_seconds = Histogram(
            "lfm3_token_generation_duration_seconds",
            "Time to generate a single token",
            ["provider", "model"],
            buckets=[0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
        )

        # Gauges
        self.requests_in_progress = Gauge(
            "lfm3_requests_in_progress",
            "Number of requests currently being processed",
            ["provider", "model"],
        )
        self.tokens_per_second = Gauge(
            "lfm3_tokens_per_second",
            "Current token generation rate",
            ["provider", "model"],
        )

    def record_prefill(self, duration_seconds: float, num_tokens: int) -> None:
        """Record a prefill operation."""
        if not self._enabled:
            return
        with self._lock:
            self.prefill_duration_seconds.labels(
                provider=self.provider, model=self.model
            ).observe(duration_seconds)
            self.tokens_prefilled_total.labels(
                provider=self.provider, model=self.model
            ).inc(num_tokens)
            self.requests_total.labels(
                provider=self.provider, model=self.model, endpoint="prefill"
            ).inc()

    def record_decode(
        self, duration_seconds: float, num_tokens: int
    ) -> None:
        """Record a decode operation."""
        if not self._enabled:
            return
        with self._lock:
            self.decode_duration_seconds.labels(
                provider=self.provider, model=self.model
            ).observe(duration_seconds)
            self.tokens_generated_total.labels(
                provider=self.provider, model=self.model
            ).inc(num_tokens)
            self.requests_total.labels(
                provider=self.provider, model=self.model, endpoint="decode"
            ).inc()

    def record_token_generation(self, duration_seconds: float) -> None:
        """Record a single token generation."""
        if not self._enabled:
            return
        with self._lock:
            self.token_generation_duration_seconds.labels(
                provider=self.provider, model=self.model
            ).observe(duration_seconds)

    def set_in_progress(self, count: int) -> None:
        """Set the number of in-progress requests."""
        if not self._enabled:
            return
        with self._lock:
            self.requests_in_progress.labels(
                provider=self.provider, model=self.model
            ).set(count)

    def set_tokens_per_second(self, tps: float) -> None:
        """Set the current token generation rate."""
        if not self._enabled:
            return
        with self._lock:
            self.tokens_per_second.labels(
                provider=self.provider, model=self.model
            ).set(tps)

    def record_error(self, endpoint: str) -> None:
        """Record a request error."""
        if not self._enabled:
            return
        with self._lock:
            self.requests_errors_total.labels(
                provider=self.provider, model=self.model, endpoint=endpoint
            ).inc()

    def get_metrics(self) -> bytes:
        """Get Prometheus-formatted metrics."""
        if not self._enabled:
            return b"# Prometheus client not available\n"
        return generate_latest()

    def get_content_type(self) -> str:
        """Get the Prometheus content type."""
        return CONTENT_TYPE_LATEST