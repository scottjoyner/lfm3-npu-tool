"""
Audit logging for LFM3-1.7B inference servers.

Produces an immutable, structured audit trail of all authentication,
authorization, and security-relevant events. Each audit entry is emitted
as a single JSON line with a deterministic schema suitable for SIEM
ingestion (Splunk, Elastic, Sentinel, etc.).

Audit events:
  * auth.success / auth.failure - API key authentication results
  * authz.allowed / authz.denied - authorization decisions
  * rate_limit.exceeded - rate limit tripped
  * validation.rejected - input validation failure
  * inference.started / inference.completed / inference.failed - inference lifecycle
  * circuit_breaker.open / circuit_breaker.closed - circuit state changes
  * admin.shutdown - graceful shutdown initiated

Each entry includes: timestamp, event type, outcome, actor (API key hash
or client IP), resource, request ID, and optional risk score.
"""

import hashlib
import json
import logging
import sys
import time
from typing import Optional, Dict, Any

_logger: Optional[logging.Logger] = None


def get_logger(name: str = "lfm3.audit") -> logging.Logger:
    """Return the audit logger (configured once, JSON-formatted)."""
    global _logger
    if _logger is None:
        _logger = logging.getLogger(name)
        _logger.setLevel(logging.INFO)
        if not _logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(_AuditFormatter())
            _logger.addHandler(handler)
        _logger.propagate = False
    return _logger


class _AuditFormatter(logging.Formatter):
    """Emit one canonical JSON audit record per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "event": record.getMessage(),
            "outcome": getattr(record, "outcome", "unknown"),
            "actor": getattr(record, "actor", None),
            "actor_type": getattr(record, "actor_type", None),
            "resource": getattr(record, "resource", None),
            "request_id": getattr(record, "request_id", None),
            "risk_score": getattr(record, "risk_score", 0),
            "severity": getattr(record, "severity", "info"),
        }
        # Drop None values for cleaner records
        payload = {k: v for k, v in payload.items() if v is not None}
        extra = getattr(record, "audit_extra", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"))


def _hash_actor(actor: Optional[str]) -> Optional[str]:
    """Hash an actor identifier (API key or IP) for privacy in audit logs."""
    if not actor:
        return None
    return hashlib.sha256(actor.encode("utf-8")).hexdigest()[:16]


class AuditLogger:
    """
    Structured audit logger.

    Emits one JSON audit record per security-relevant event. All methods
    are safe to call from async request handlers.
    """

    def __init__(self, service_name: str = "lfm3-inference") -> None:
        self.logger = get_logger()
        self.service_name = service_name

    def _emit(
        self,
        event: str,
        outcome: str,
        *,
        actor: Optional[str] = None,
        actor_type: Optional[str] = None,
        resource: Optional[str] = None,
        request_id: Optional[str] = None,
        risk_score: int = 0,
        severity: str = "info",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.logger.info(
            event,
            extra={
                "outcome": outcome,
                "actor": _hash_actor(actor) if actor else None,
                "actor_type": actor_type,
                "resource": resource,
                "request_id": request_id,
                "risk_score": risk_score,
                "severity": severity,
                "audit_extra": extra,
                "service": self.service_name,
            },
        )

    # ---- Authentication ----
    def auth_success(
        self,
        actor: str,
        actor_type: str = "api_key",
        request_id: Optional[str] = None,
    ) -> None:
        self._emit(
            "auth.success",
            "success",
            actor=actor,
            actor_type=actor_type,
            resource="authentication",
            request_id=request_id,
            severity="info",
        )

    def auth_failure(
        self,
        actor: Optional[str] = None,
        actor_type: str = "api_key",
        reason: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        self._emit(
            "auth.failure",
            "failure",
            actor=actor,
            actor_type=actor_type,
            resource="authentication",
            request_id=request_id,
            risk_score=5,
            severity="warning",
            extra={"reason": reason} if reason else None,
        )

    # ---- Authorization ----
    def authz_allowed(
        self,
        actor: str,
        resource: str,
        request_id: Optional[str] = None,
    ) -> None:
        self._emit(
            "authz.allowed",
            "success",
            actor=actor,
            actor_type="api_key",
            resource=resource,
            request_id=request_id,
            severity="info",
        )

    def authz_denied(
        self,
        actor: str,
        resource: str,
        request_id: Optional[str] = None,
    ) -> None:
        self._emit(
            "authz.denied",
            "denied",
            actor=actor,
            actor_type="api_key",
            resource=resource,
            request_id=request_id,
            risk_score=7,
            severity="warning",
        )

    # ---- Rate limiting ----
    def rate_limit_exceeded(
        self,
        actor: str,
        actor_type: str = "api_key",
        request_id: Optional[str] = None,
    ) -> None:
        self._emit(
            "rate_limit.exceeded",
            "limited",
            actor=actor,
            actor_type=actor_type,
            resource="rate_limit",
            request_id=request_id,
            risk_score=4,
            severity="warning",
        )

    # ---- Input validation ----
    def validation_rejected(
        self,
        actor: Optional[str] = None,
        actor_type: str = "api_key",
        code: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        self._emit(
            "validation.rejected",
            "rejected",
            actor=actor,
            actor_type=actor_type,
            resource="input_validation",
            request_id=request_id,
            risk_score=6,
            severity="warning",
            extra={"code": code} if code else None,
        )

    # ---- Inference lifecycle ----
    def inference_started(
        self,
        actor: str,
        resource: str,
        request_id: Optional[str] = None,
    ) -> None:
        self._emit(
            "inference.started",
            "started",
            actor=actor,
            actor_type="api_key",
            resource=resource,
            request_id=request_id,
            severity="info",
        )

    def inference_completed(
        self,
        actor: str,
        resource: str,
        request_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        self._emit(
            "inference.completed",
            "success",
            actor=actor,
            actor_type="api_key",
            resource=resource,
            request_id=request_id,
            severity="info",
            extra={"latency_ms": round(latency_ms, 2)} if latency_ms is not None else None,
        )

    def inference_failed(
        self,
        actor: str,
        resource: str,
        error: str,
        request_id: Optional[str] = None,
    ) -> None:
        self._emit(
            "inference.failed",
            "failure",
            actor=actor,
            actor_type="api_key",
            resource=resource,
            request_id=request_id,
            risk_score=8,
            severity="error",
            extra={"error": error},
        )

    # ---- Circuit breaker ----
    def circuit_breaker_open(
        self,
        resource: str,
        retry_after: float,
        request_id: Optional[str] = None,
    ) -> None:
        self._emit(
            "circuit_breaker.open",
            "open",
            actor="system",
            actor_type="system",
            resource=resource,
            request_id=request_id,
            risk_score=9,
            severity="critical",
            extra={"retry_after_seconds": round(retry_after, 1)},
        )

    def circuit_breaker_closed(
        self,
        resource: str,
        request_id: Optional[str] = None,
    ) -> None:
        self._emit(
            "circuit_breaker.closed",
            "closed",
            actor="system",
            actor_type="system",
            resource=resource,
            request_id=request_id,
            severity="info",
        )

    # ---- Admin ----
    def admin_shutdown(
        self,
        actor: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        self._emit(
            "admin.shutdown",
            "initiated",
            actor=actor,
            actor_type="admin",
            resource="server",
            request_id=request_id,
            risk_score=3,
            severity="warning",
        )


# Global audit logger instance
_audit: Optional[AuditLogger] = None


def get_audit() -> AuditLogger:
    """Get the global audit logger (initializes on first use)."""
    global _audit
    if _audit is None:
        _audit = AuditLogger()
    return _audit
