"""
Security headers middleware for LFM3-1.7B inference servers.

Adds enterprise-grade HTTP security headers to all responses:

  * Strict-Transport-Security (HSTS) - enforce HTTPS
  * X-Content-Type-Options: nosniff - prevent MIME sniffing
  * X-Frame-Options: DENY - prevent clickjacking
  * Content-Security-Policy - restrict resource sources
  * Referrer-Policy - control referrer leakage
  * Permissions-Policy - disable browser features
  * X-XSS-Protection - legacy XSS filter (defense in depth)
  * X-Request-Id - correlation ID echo

All headers are configurable via environment variables. The middleware
is a Starlette BaseHTTPMiddleware so it drops into any FastAPI app.
"""

from typing import Optional, Dict, Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds security headers to every response.

    Configuration via environment variables:
      LFM3_SECURITY_HSTS_MAX_AGE     - HSTS max-age seconds (default 31536000 = 1 year)
      LFM3_SECURITY_HSTS_INCLUDE_SUBDOMAINS - "1" to include subdomains
      LFM3_SECURITY_HSTS_PRELOAD     - "1" to add preload directive
      LFM3_SECURITY_CSP              - custom Content-Security-Policy value
      LFM3_SECURITY_REFERRER_POLICY  - default "strict-origin-when-cross-origin"
      LFM3_SECURITY_PERMISSIONS      - custom Permissions-Policy value
      LFM3_SECURITY_DISABLE_HSTS     - "1" to skip HSTS (HTTP-only deployments)
    """

    def __init__(self, app, csp: Optional[str] = None) -> None:
        super().__init__(app)
        import os as _os

        self.hsts_max_age = int(_os.environ.get("LFM3_SECURITY_HSTS_MAX_AGE", "31536000"))
        self.hsts_include_subdomains = _os.environ.get(
            "LFM3_SECURITY_HSTS_INCLUDE_SUBDOMAINS", "1"
        ).lower() in ("1", "true", "yes")
        self.hsts_preload = _os.environ.get(
            "LFM3_SECURITY_HSTS_PRELOAD", "1"
        ).lower() in ("1", "true", "yes")
        self.disable_hsts = _os.environ.get(
            "LFM3_SECURITY_DISABLE_HSTS", "0"
        ).lower() in ("1", "true", "yes")
        self.referrer_policy = _os.environ.get(
            "LFM3_SECURITY_REFERRER_POLICY", "strict-origin-when-cross-origin"
        )
        self.permissions_policy = csp or _os.environ.get(
            "LFM3_SECURITY_PERMISSIONS",
            "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
        )
        self.csp = csp or _os.environ.get(
            "LFM3_SECURITY_CSP",
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none';",
        )

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        self._apply_headers(response, request)
        return response

    def _apply_headers(self, response: Response, request: Request) -> None:
        # Echo correlation ID if present
        request_id = getattr(request.state, "request_id", None)
        if request_id:
            response.headers["X-Request-Id"] = request_id

        # HSTS - only over HTTPS or when explicitly enabled
        if not self.disable_hsts:
            hsts = f"max-age={self.hsts_max_age}"
            if self.hsts_include_subdomains:
                hsts += "; includeSubDomains"
            if self.hsts_preload:
                hsts += "; preload"
            response.headers["Strict-Transport-Security"] = hsts

        # MIME sniffing protection
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Clickjacking protection
        response.headers["X-Frame-Options"] = "DENY"

        # Legacy XSS filter (defense in depth for older browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Content Security Policy
        response.headers["Content-Security-Policy"] = self.csp

        # Referrer policy
        response.headers["Referrer-Policy"] = self.referrer_policy

        # Permissions policy
        response.headers["Permissions-Policy"] = self.permissions_policy

        # Remove server version disclosure
        try:
            del response.headers["Server"]
        except KeyError:
            pass
