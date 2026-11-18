"""
Role-Based Access Control (RBAC) for LFM3-1.7B inference servers.

Provides:
  * Role definitions: admin, operator, viewer, auditor
  * Permission model: read, write, admin, audit
  * Role-based endpoint access control for FastAPI
  * CLI integration for managing roles and key assignments

Roles (in hierarchy order):
  * admin     - Full access: read, write, admin, audit
  * operator  - Read + write: can run inference, view metrics, no config changes
  * viewer    - Read-only: can view health, metrics, models, no write
  * auditor   - Audit-only: can view audit logs, metrics, health; no write, no admin

Key format in LFM3_API_KEYS env var:
  sk-...:role:label   (e.g., sk-abc123:admin:prod-web, sk-def456:viewer:monitoring)

If no role is specified, defaults to "viewer".
"""

import hashlib
import os
from typing import Dict, Optional, Set

# ---------------------------------------------------------------------------
# Role and permission definitions
# ---------------------------------------------------------------------------

# Permission bits
PERM_READ = "read"
PERM_WRITE = "write"
PERM_ADMIN = "admin"
PERM_AUDIT = "audit"

# Role -> set of permissions
ROLE_PERMISSIONS: Dict[str, Set[str]] = {
    "admin": {PERM_READ, PERM_WRITE, PERM_ADMIN, PERM_AUDIT},
    "operator": {PERM_READ, PERM_WRITE},
    "viewer": {PERM_READ},
    "auditor": {PERM_READ, PERM_AUDIT},
}

# Role hierarchy (higher index = more privileged)
ROLE_HIERARCHY: Dict[str, int] = {
    "viewer": 0,
    "auditor": 0,
    "operator": 1,
    "admin": 2,
}

# Endpoint -> required permission
ENDPOINT_PERMISSIONS: Dict[str, str] = {
    # Read-only endpoints
    "/health": PERM_READ,
    "/live": PERM_READ,
    "/ready": PERM_READ,
    "/version": PERM_READ,
    "/v1/models": PERM_READ,
    "/v1/model/info": PERM_READ,
    "/slo": PERM_READ,
    "/metrics": PERM_READ,
    "/v1/batch/status": PERM_READ,
    "/v1/metrics/detailed": PERM_READ,
    "/v1/metrics/endpoints": PERM_READ,
    "/v1/fleet/status": PERM_READ,
    "/v1/traces": PERM_READ,
    # Write endpoints (inference)
    "/v1/chat/completions": PERM_WRITE,
    "/v1/batch/chat/completions": PERM_WRITE,
    # Admin endpoints
    "/shutdown": PERM_ADMIN,
    # Audit endpoints (viewed via logs, not HTTP, but documented here)
}


def has_permission(role: str, permission: str) -> bool:
    """Check if a role has a given permission."""
    return permission in ROLE_PERMISSIONS.get(role, set())


def role_level(role: str) -> int:
    """Return the numeric level of a role (higher = more privileged)."""
    return ROLE_HIERARCHY.get(role, 0)


def role_required(endpoint: str) -> str:
    """Return the required permission for an endpoint."""
    return ENDPOINT_PERMISSIONS.get(endpoint, PERM_READ)


# ---------------------------------------------------------------------------
# API key with role
# ---------------------------------------------------------------------------


class KeyInfo:
    """Information about an API key."""

    def __init__(self, key_hash: str, role: str, label: str):
        self.key_hash = key_hash
        self.role = role
        self.label = label

    def has_permission(self, permission: str) -> bool:
        return has_permission(self.role, permission)

    def can_access(self, endpoint: str) -> bool:
        required = role_required(endpoint)
        return self.has_permission(required)

    def to_dict(self) -> Dict[str, str]:
        return {"key_hash": self.key_hash, "role": self.role, "label": self.label}


# ---------------------------------------------------------------------------
# RBAC-aware authentication
# ---------------------------------------------------------------------------


class RBACAuth:
    """
    Role-based access control authentication.

    Keys are loaded from LFM3_API_KEYS env var with format:
      sk-...:role:label
    If no role is specified, defaults to "viewer".
    """

    def __init__(self) -> None:
        raw = os.environ.get("LFM3_API_KEYS", "")
        self._keys: Dict[str, KeyInfo] = {}  # hashed_key -> KeyInfo
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            # Format: key:role:label  (role and label optional)
            parts = entry.split(":")
            key = parts[0].strip() if parts else ""
            role = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "viewer"
            label = parts[2].strip() if len(parts) > 2 and parts[2].strip() else "unlabeled"
            if key:
                key_hash = _hash_key(key)
                self._keys[key_hash] = KeyInfo(key_hash, role, label)
        self.enabled = bool(self._keys)

    def authenticate(self, api_key: Optional[str]) -> Optional[KeyInfo]:
        """
        Validate an API key.  Returns KeyInfo on success, None on failure.
        """
        if not self.enabled:
            # No keys configured - open access with viewer role
            return KeyInfo("", "viewer", "anonymous")
        if not api_key:
            return None
        key_hash = _hash_key(api_key)
        return self._keys.get(key_hash)

    def require_permission(self, key_info: Optional[KeyInfo], permission: str,
                           endpoint: str = "") -> None:
        """
        Check that key_info has the given permission.  Raises PermissionError
        if not authorized.
        """
        if self.enabled and key_info is None:
            raise PermissionError("Authentication required")
        if key_info is not None and not key_info.has_permission(permission):
            raise PermissionError(
                f"Insufficient permissions: '{permission}' required for "
                f"{endpoint or 'this endpoint'} (role: {key_info.role})"
            )

    def require_role(self, key_info: Optional[KeyInfo], min_role: str,
                     endpoint: str = "") -> None:
        """
        Check that key_info has at least the given role level.
        """
        if self.enabled and key_info is None:
            raise PermissionError("Authentication required")
        if key_info is not None and role_level(key_info.role) < role_level(min_role):
            raise PermissionError(
                f"Insufficient role: '{min_role}' required for "
                f"{endpoint or 'this endpoint'} (role: {key_info.role})"
            )

    def list_keys(self) -> Dict[str, Dict[str, str]]:
        """Return all key info (hashes are truncated for safety)."""
        return {kh: ki.to_dict() for kh, ki in self._keys.items()}


def _hash_key(key: str) -> str:
    """Return a one-way hash of an API key for safe logging/comparison."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


# Backwards-compatible alias
APIKeyAuth = RBACAuth
