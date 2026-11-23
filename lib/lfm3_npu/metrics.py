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


class EnterpriseMetricsCollector:
    """Collects Prometheus metrics for enterprise modules."""

    def __init__(self):
        self._enabled = Counter is not None

        if not self._enabled:
            return

        # Auto-scaler metrics
        self.autoscaler_current_replicas = Gauge(
            "lfm3_autoscaler_current_replicas",
            "Current number of replicas",
        )
        self.autoscaler_scale_up_total = Counter(
            "lfm3_autoscaler_scale_up_total",
            "Total number of scale-up operations",
        )
        self.autoscaler_scale_down_total = Counter(
            "lfm3_autoscaler_scale_down_total",
            "Total number of scale-down operations",
        )

        # A/B testing metrics
        self.abtest_active_tests = Gauge(
            "lfm3_abtest_active_tests",
            "Number of currently running A/B tests",
        )
        self.abtest_requests_total = Counter(
            "lfm3_abtest_requests_total",
            "Total A/B test requests routed",
            ["test_id", "variant_id"],
        )

        # Disaster recovery metrics
        self.dr_region_healthy = Gauge(
            "lfm3_dr_region_healthy",
            "Whether a DR region is healthy (1=healthy, 0=unhealthy)",
            ["region"],
        )
        self.dr_failover_total = Counter(
            "lfm3_dr_failover_total",
            "Total number of failover events",
        )

        # Model promotion metrics
        self.promotion_requests_total = Counter(
            "lfm3_promotion_requests_total",
            "Total number of promotion requests",
            ["from_stage", "to_stage", "status"],
        )

        # Cost tracking metrics
        self.cost_total_usd = Gauge(
            "lfm3_cost_total_usd",
            "Total accumulated cost in USD",
            ["tenant", "tier"],
        )

        # Key rotation metrics
        self.key_rotation_total = Counter(
            "lfm3_key_rotation_total",
            "Total number of key rotations",
        )

        # Security scan metrics
        self.security_findings_total = Gauge(
            "lfm3_security_findings_total",
            "Total number of security findings",
            ["severity"],
        )

    def record_autoscaler_replicas(self, count: int) -> None:
        """Record current replica count."""
        if self._enabled:
            self.autoscaler_current_replicas.set(count)

    def record_autoscaler_scale_up(self) -> None:
        """Record a scale-up operation."""
        if self._enabled:
            self.autoscaler_scale_up_total.inc()

    def record_autoscaler_scale_down(self) -> None:
        """Record a scale-down operation."""
        if self._enabled:
            self.autoscaler_scale_down_total.inc()

    def record_abtest_active_tests(self, count: int) -> None:
        """Record number of active A/B tests."""
        if self._enabled:
            self.abtest_active_tests.set(count)

    def record_abtest_request(self, test_id: str, variant_id: str) -> None:
        """Record an A/B test request routed to a variant."""
        if self._enabled:
            self.abtest_requests_total.labels(
                test_id=test_id, variant_id=variant_id
            ).inc()

    def record_dr_region_healthy(self, region: str, healthy: bool) -> None:
        """Record DR region health status."""
        if self._enabled:
            self.dr_region_healthy.labels(region=region).set(1 if healthy else 0)

    def record_dr_failover(self) -> None:
        """Record a failover event."""
        if self._enabled:
            self.dr_failover_total.inc()

    def record_promotion_request(self, from_stage: str, to_stage: str, status: str) -> None:
        """Record a promotion request."""
        if self._enabled:
            self.promotion_requests_total.labels(
                from_stage=from_stage, to_stage=to_stage, status=status
            ).inc()

    def record_cost_total(self, tenant: str, tier: str, usd: float) -> None:
        """Record total cost for a tenant/tier."""
        if self._enabled:
            self.cost_total_usd.labels(tenant=tenant, tier=tier).set(usd)

    def record_key_rotation(self) -> None:
        """Record a key rotation event."""
        if self._enabled:
            self.key_rotation_total.inc()

    def record_security_finding(self, severity: str, count: int) -> None:
        """Record security findings count by severity."""
        if self._enabled:
            self.security_findings_total.labels(severity=severity).set(count)

    def get_metrics(self) -> bytes:
        """Get Prometheus-formatted metrics for enterprise modules."""
        if not self._enabled:
            return b"# Prometheus client not available\n"
        return generate_latest()

    def get_content_type(self) -> str:
        """Get the Prometheus content type."""
        return CONTENT_TYPE_LATEST