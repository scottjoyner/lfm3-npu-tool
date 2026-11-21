"""
Self-service API key management for LFM3-1.7B inference servers.

Provides:
  * API key lifecycle: create, list, rotate, revoke, view usage
  * Self-service key management via REST API
  * Key format: sk-<prefix><random>:role:label (compatible with existing RBAC)
  * Key expiration and rotation support
  * Per-key usage tracking (builds on existing token usage tracking)
  * Key metadata: name, description, rate limits, allowed IPs
  * CLI integration for key management

Key format:
    sk-live-<24 random chars>:role:label
    Example: sk-live-aBcDeFgHiJkLmNoPqRsTuVwX:admin:prod-web

API endpoints (require admin role):
  * GET    /v1/keys                    - List all API keys (masked)
  * GET    /v1/keys/{key_hash}         - Get key details
  * POST   /v1/keys                    - Create a new API key
  * PUT    /v1/keys/{key_hash}         - Update key metadata
  * DELETE /v1/keys/{key_hash}         - Revoke/delete a key
  * POST   /v1/keys/{key_hash}/rotate  - Rotate key (generate new value)
  * GET    /v1/keys/{key_hash}/usage   - Get key usage statistics

Self-service endpoints (any authenticated user):
  * GET    /v1/keys/me                 - Get current key's details
  * GET    /v1/keys/me/usage           - Get current key's usage
"""

import hashlib
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Key prefix
KEY_PREFIX = "sk-live-"
KEY_RANDOM_LENGTH = 24


@dataclass
class ApiKey:
    """API key metadata and usage tracking."""

    key_hash: str                    # SHA-256 hash (first 16 chars used as ID)
    key_prefix: str                  # First 8 chars of actual key (for display)
    role: str                        # admin, operator, viewer, auditor
    label: str                       # Human-readable label
    name: Optional[str] = None       # Display name
    description: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None  # Optional expiration timestamp
    active: bool = True
    rate_limit_rpm: Optional[int] = None  # Per-key rate limit override
    allowed_ips: Optional[List[str]] = None  # IP allowlist
    # Usage tracking
    total_requests: int = 0
    total_errors: int = 0
    total_tokens: int = 0
    last_used_at: Optional[float] = None
    last_used_ip: Optional[str] = None

    @property
    def key_id(self) -> str:
        """Short key identifier for display (first 8 chars of hash)."""
        return self.key_hash[:8]

    @property
    def is_expired(self) -> bool:
        """Check if the key has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def masked_key(self) -> str:
        """Return masked key for display (sk-live-****...)."""
        return f"{KEY_PREFIX}{'*' * KEY_RANDOM_LENGTH}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key_hash": self.key_hash,
            "key_id": self.key_id,
            "key_prefix": self.key_prefix,
            "masked_key": self.masked_key,
            "role": self.role,
            "label": self.label,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "active": self.active,
            "rate_limit_rpm": self.rate_limit_rpm,
            "allowed_ips": self.allowed_ips,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "total_tokens": self.total_tokens,
            "last_used_at": self.last_used_at,
            "last_used_ip": self.last_used_ip,
            "is_expired": self.is_expired,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApiKey":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


def generate_api_key(
    role: str = "viewer",
    label: str = "",
    name: Optional[str] = None,
    description: Optional[str] = None,
    expires_in_days: Optional[int] = None,
    rate_limit_rpm: Optional[int] = None,
    allowed_ips: Optional[List[str]] = None,
) -> tuple[str, ApiKey]:
    """
    Generate a new API key.

    Returns:
        (raw_key, ApiKey) - The raw key (only shown once) and metadata.
    """
    random_part = secrets.token_urlsafe(KEY_RANDOM_LENGTH)
    raw_key = f"{KEY_PREFIX}{random_part}"
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    key_prefix = raw_key[:12]  # First 12 chars for display

    expires_at = None
    if expires_in_days is not None:
        expires_at = time.time() + (expires_in_days * 86400)

    key = ApiKey(
        key_hash=key_hash,
        key_prefix=key_prefix,
        role=role,
        label=label or f"key-{key_hash[:8]}",
        name=name,
        description=description,
        expires_at=expires_at,
        rate_limit_rpm=rate_limit_rpm,
        allowed_ips=allowed_ips,
    )
    return raw_key, key


def hash_key(raw_key: str) -> str:
    """Return SHA-256 hash of an API key (first 16 chars used as ID)."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Key Manager                                                                 #
# --------------------------------------------------------------------------- #

class KeyManager:
    """Manages API keys: CRUD, rotation, usage tracking, persistence."""

    def __init__(self, storage_path: Optional[str] = None):
        if storage_path is None:
            storage_path = os.environ.get(
                "LFM3_KEY_STORAGE",
                str(Path.home() / ".config" / "lfm3-npu" / "keys.json"),
            )
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._keys: Dict[str, ApiKey] = {}  # key_hash -> ApiKey
        self._load()

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        """Load keys from disk."""
        if not self.storage_path.exists():
            self._save()
            return
        try:
            import json
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            for kh, kdata in data.items():
                self._keys[kh] = ApiKey.from_dict(kdata)
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        """Save keys to disk."""
        import json
        tmp = self.storage_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({kh: k.to_dict() for kh, k in self._keys.items()}, f, indent=2)
        tmp.replace(self.storage_path)

    # ------------------------------------------------------------------ #
    # CRUD                                                               #
    # ------------------------------------------------------------------ #

    def create_key(
        self,
        role: str = "viewer",
        label: str = "",
        name: Optional[str] = None,
        description: Optional[str] = None,
        expires_in_days: Optional[int] = None,
        rate_limit_rpm: Optional[int] = None,
        allowed_ips: Optional[List[str]] = None,
    ) -> tuple[str, ApiKey]:
        """Create a new API key. Returns (raw_key, ApiKey)."""
        raw_key, key = generate_api_key(
            role=role,
            label=label,
            name=name,
            description=description,
            expires_in_days=expires_in_days,
            rate_limit_rpm=rate_limit_rpm,
            allowed_ips=allowed_ips,
        )
        self._keys[key.key_hash] = key
        self._save()
        return raw_key, key

    def get_key(self, key_hash: str) -> Optional[ApiKey]:
        """Get key metadata by hash."""
        return self._keys.get(key_hash)

    def get_key_by_prefix(self, key_prefix: str) -> Optional[ApiKey]:
        """Find a key by its prefix (first 12 chars of raw key)."""
        for key in self._keys.values():
            if key.key_prefix == key_prefix:
                return key
        return None

    def list_keys(self) -> List[ApiKey]:
        """List all keys."""
        return list(self._keys.values())

    def update_key(
        self,
        key_hash: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        role: Optional[str] = None,
        label: Optional[str] = None,
        rate_limit_rpm: Optional[int] = None,
        allowed_ips: Optional[List[str]] = None,
        active: Optional[bool] = None,
        expires_in_days: Optional[int] = None,
    ) -> Optional[ApiKey]:
        """Update key metadata."""
        key = self._keys.get(key_hash)
        if key is None:
            return None
        if name is not None:
            key.name = name
        if description is not None:
            key.description = description
        if role is not None:
            key.role = role
        if label is not None:
            key.label = label
        if rate_limit_rpm is not None:
            key.rate_limit_rpm = rate_limit_rpm
        if allowed_ips is not None:
            key.allowed_ips = allowed_ips
        if active is not None:
            key.active = active
        if expires_in_days is not None:
            key.expires_at = time.time() + (expires_in_days * 86400)
        key.updated_at = time.time()
        self._save()
        return key

    def revoke_key(self, key_hash: str) -> bool:
        """Revoke (deactivate) a key."""
        key = self._keys.get(key_hash)
        if key is None:
            return False
        key.active = False
        key.updated_at = time.time()
        self._save()
        return True

    def delete_key(self, key_hash: str) -> bool:
        """Permanently delete a key."""
        if key_hash in self._keys:
            del self._keys[key_hash]
            self._save()
            return True
        return False

    def rotate_key(self, key_hash: str) -> Optional[tuple[str, ApiKey]]:
        """Rotate a key: generate new raw key, keep metadata."""
        old_key = self._keys.get(key_hash)
        if old_key is None:
            return None

        # Generate new raw key with same metadata
        raw_key, new_key = generate_api_key(
            role=old_key.role,
            label=old_key.label,
            name=old_key.name,
            description=old_key.description,
            expires_in_days=None,  # Preserve existing expiration
            rate_limit_rpm=old_key.rate_limit_rpm,
            allowed_ips=old_key.allowed_ips,
        )
        new_key.created_at = old_key.created_at
        new_key.expires_at = old_key.expires_at

        # Replace old key with new
        del self._keys[key_hash]
        self._keys[new_key.key_hash] = new_key
        self._save()
        return raw_key, new_key

    def record_key_usage(
        self,
        key_hash: str,
        tokens: int = 0,
        error: bool = False,
        client_ip: Optional[str] = None,
    ) -> None:
        """Record usage for a key."""
        key = self._keys.get(key_hash)
        if key:
            key.total_requests += 1
            if error:
                key.total_errors += 1
            key.total_tokens += tokens
            key.last_used_at = time.time()
            key.last_used_ip = client_ip
            key.updated_at = time.time()

    def is_key_valid(self, key_hash: str) -> bool:
        """Check if a key is valid (exists, active, not expired)."""
        key = self._keys.get(key_hash)
        if key is None:
            return False
        if not key.active:
            return False
        if key.is_expired:
            return False
        return True

    def get_key_usage(self, key_hash: str) -> Optional[Dict[str, Any]]:
        """Get usage statistics for a key."""
        key = self._keys.get(key_hash)
        if key is None:
            return None
        return key.to_dict()


# --------------------------------------------------------------------------- #
# Global instance                                                             #
# --------------------------------------------------------------------------- #

_key_manager: Optional[KeyManager] = None


def get_key_manager() -> KeyManager:
    global _key_manager
    if _key_manager is None:
        _key_manager = KeyManager()
    return _key_manager
