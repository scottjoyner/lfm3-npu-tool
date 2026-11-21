"""
Multi-tenancy support for LFM3-1.7B inference servers.

Provides:
  * Tenant definitions with tier, quotas, rate limits
  * Tenant-scoped usage tracking (separate from API key tracking)
  * Per-tenant rate limiting and token quota enforcement
  * Tenant isolation in metrics and audit logging
  * Persistence to JSON file
  * CLI integration for tenant management

Header: X-Tenant-ID

Tenant tiers:
  * free      - 100k tokens/month, 10 req/min, 2 concurrent, priority 0
  * standard  - 1M tokens/month, 60 req/min, 10 concurrent, priority 1
  * enterprise - unlimited tokens, 600 req/min, 100 concurrent, priority 2
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Tenant tier definitions                                                     #
# --------------------------------------------------------------------------- #

TENANT_TIERS: Dict[str, Dict[str, Any]] = {
    "free": {
        "monthly_token_quota": 100_000,
        "requests_per_minute": 10,
        "concurrent_requests": 2,
        "priority": 0,
    },
    "standard": {
        "monthly_token_quota": 1_000_000,
        "requests_per_minute": 60,
        "concurrent_requests": 10,
        "priority": 1,
    },
    "enterprise": {
        "monthly_token_quota": 0,  # unlimited
        "requests_per_minute": 600,
        "concurrent_requests": 100,
        "priority": 2,
    },
}

DEFAULT_TENANT_ID = "default"
TENANT_HEADER = "X-Tenant-ID"


# --------------------------------------------------------------------------- #
# Tenant data model                                                           #
# --------------------------------------------------------------------------- #

@dataclass
class Tenant:
    """A tenant with quotas, rate limits, and usage tracking."""

    tenant_id: str
    name: str
    tier: str = "free"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    active: bool = True
    # Usage tracking (resets monthly)
    tokens_used: int = 0
    requests_count: int = 0
    last_reset: float = field(default_factory=time.time)
    # Custom overrides (None = use tier defaults)
    monthly_token_quota: Optional[int] = None
    requests_per_minute: Optional[int] = None
    concurrent_requests: Optional[int] = None
    allowed_roles: Optional[List[str]] = None  # None = all roles allowed

    # ------------------------------------------------------------------ #
    # Effective configuration                                            #
    # ------------------------------------------------------------------ #

    def get_effective_quota(self) -> int:
        """Return monthly token quota (0 = unlimited)."""
        if self.monthly_token_quota is not None:
            return self.monthly_token_quota
        return TENANT_TIERS.get(self.tier, TENANT_TIERS["free"])["monthly_token_quota"]

    def get_effective_rpm(self) -> int:
        """Return requests per minute limit."""
        if self.requests_per_minute is not None:
            return self.requests_per_minute
        return TENANT_TIERS.get(self.tier, TENANT_TIERS["free"])["requests_per_minute"]

    def get_effective_concurrent(self) -> int:
        """Return max concurrent requests."""
        if self.concurrent_requests is not None:
            return self.concurrent_requests
        return TENANT_TIERS.get(self.tier, TENANT_TIERS["free"])["concurrent_requests"]

    def get_priority(self) -> int:
        """Return routing priority (higher = more priority)."""
        return TENANT_TIERS.get(self.tier, TENANT_TIERS["free"])["priority"]

    # ------------------------------------------------------------------ #
    # Quota / usage helpers                                              #
    # ------------------------------------------------------------------ #

    def maybe_reset_monthly(self) -> bool:
        """Reset usage counters if a new month has started. Returns True if reset."""
        now = time.time()
        if now - self.last_reset > 30 * 24 * 3600:
            self.tokens_used = 0
            self.requests_count = 0
            self.last_reset = now
            self.updated_at = now
            return True
        return False

    def is_quota_exceeded(self) -> bool:
        """Check if the tenant's monthly token quota is exceeded."""
        self.maybe_reset_monthly()
        quota = self.get_effective_quota()
        if quota <= 0:
            return False  # unlimited
        return self.tokens_used >= quota

    def remaining_tokens(self) -> int:
        """Return remaining tokens in current month (-1 = unlimited)."""
        self.maybe_reset_monthly()
        quota = self.get_effective_quota()
        if quota <= 0:
            return -1
        return max(0, quota - self.tokens_used)

    def record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record token usage for this tenant."""
        self.tokens_used += prompt_tokens + completion_tokens
        self.requests_count += 1
        self.updated_at = time.time()
        self.maybe_reset_monthly()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tenant":
        # Filter out unknown keys for forward compatibility
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


# --------------------------------------------------------------------------- #
# Tenant manager                                                              #
# --------------------------------------------------------------------------- #

class TenantManager:
    """Manages tenants: CRUD, usage tracking, quota enforcement, persistence."""

    def __init__(self, storage_path: Optional[str] = None):
        if storage_path is None:
            storage_path = os.environ.get(
                "LFM3_TENANT_STORAGE",
                str(Path.home() / ".config" / "lfm3-npu" / "tenants.json"),
            )
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._tenants: Dict[str, Tenant] = {}
        self._load()

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        """Load tenants from disk."""
        if not self.storage_path.exists():
            self._create_default_tenant()
            self._save()
            return
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            for tid, tdata in data.items():
                self._tenants[tid] = Tenant.from_dict(tdata)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[Tenants] Warning: could not load {self.storage_path}: {e}", flush=True)
            self._create_default_tenant()

    def _save(self) -> None:
        """Save tenants to disk."""
        tmp = self.storage_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({tid: t.to_dict() for tid, t in self._tenants.items()}, f, indent=2)
        tmp.replace(self.storage_path)

    def _create_default_tenant(self) -> None:
        """Create the default tenant if none exist."""
        self._tenants[DEFAULT_TENANT_ID] = Tenant(
            tenant_id=DEFAULT_TENANT_ID,
            name="Default Tenant",
            tier="standard",
        )

    # ------------------------------------------------------------------ #
    # CRUD                                                               #
    # ------------------------------------------------------------------ #

    async def create_tenant(
        self,
        tenant_id: str,
        name: str,
        tier: str = "free",
        monthly_token_quota: Optional[int] = None,
        requests_per_minute: Optional[int] = None,
        concurrent_requests: Optional[int] = None,
        allowed_roles: Optional[List[str]] = None,
    ) -> Tenant:
        if tenant_id in self._tenants:
            raise ValueError(f"Tenant '{tenant_id}' already exists")
        if tier not in TENANT_TIERS:
            raise ValueError(f"Unknown tier '{tier}'. Valid: {list(TENANT_TIERS.keys())}")
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            tier=tier,
            monthly_token_quota=monthly_token_quota,
            requests_per_minute=requests_per_minute,
            concurrent_requests=concurrent_requests,
            allowed_roles=allowed_roles,
        )
        self._tenants[tenant_id] = tenant
        self._save()
        return tenant

    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        return self._tenants.get(tenant_id)

    async def list_tenants(self) -> List[Tenant]:
        return list(self._tenants.values())

    async def update_tenant(
        self,
        tenant_id: str,
        name: Optional[str] = None,
        tier: Optional[str] = None,
        monthly_token_quota: Optional[int] = None,
        requests_per_minute: Optional[int] = None,
        concurrent_requests: Optional[int] = None,
        allowed_roles: Optional[List[str]] = None,
        active: Optional[bool] = None,
    ) -> Optional[Tenant]:
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return None
        if name is not None:
            tenant.name = name
        if tier is not None:
            if tier not in TENANT_TIERS:
                raise ValueError(f"Unknown tier '{tier}'")
            tenant.tier = tier
        if monthly_token_quota is not None:
            tenant.monthly_token_quota = monthly_token_quota
        if requests_per_minute is not None:
            tenant.requests_per_minute = requests_per_minute
        if concurrent_requests is not None:
            tenant.concurrent_requests = concurrent_requests
        if allowed_roles is not None:
            tenant.allowed_roles = allowed_roles
        if active is not None:
            tenant.active = active
        tenant.updated_at = time.time()
        self._save()
        return tenant

    async def delete_tenant(self, tenant_id: str) -> bool:
        if tenant_id == DEFAULT_TENANT_ID:
            raise ValueError("Cannot delete the default tenant")
        if tenant_id in self._tenants:
            del self._tenants[tenant_id]
            self._save()
            return True
        return False

    # ------------------------------------------------------------------ #
    # Quota / usage                                                      #
    # ------------------------------------------------------------------ #

    async def record_tenant_usage(
        self, tenant_id: str, prompt_tokens: int, completion_tokens: int
    ) -> None:
        tenant = self._tenants.get(tenant_id)
        if tenant:
            tenant.record_usage(prompt_tokens, completion_tokens)
            self._save()

    async def check_tenant_quota(
        self, tenant_id: str, estimated_tokens: int
    ) -> Optional[str]:
        """
        Check if a tenant can make a request with the given estimated tokens.
        Returns None if OK, or an error message string if quota exceeded.
        """
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return f"Tenant '{tenant_id}' not found"
        if not tenant.active:
            return f"Tenant '{tenant_id}' is inactive"

        tenant.maybe_reset_monthly()
        quota = tenant.get_effective_quota()
        used = tenant.tokens_used

        if quota <= 0:
            return None  # unlimited

        if used >= quota:
            return (
                f"Tenant '{tenant_id}' monthly token quota exceeded "
                f"({used}/{quota} tokens)"
            )

        remaining = quota - used
        if estimated_tokens > remaining:
            return (
                f"Tenant '{tenant_id}' would exceed monthly token quota "
                f"({used + estimated_tokens}/{quota})"
            )
        return None

    async def get_tenant_usage(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return None
        tenant.maybe_reset_monthly()
        quota = tenant.get_effective_quota()
        return {
            "tenant_id": tenant.tenant_id,
            "name": tenant.name,
            "tier": tenant.tier,
            "active": tenant.active,
            "tokens_used": tenant.tokens_used,
            "tokens_quota": quota,
            "tokens_remaining": tenant.remaining_tokens(),
            "requests_count": tenant.requests_count,
            "priority": tenant.get_priority(),
            "last_reset": tenant.last_reset,
            "updated_at": tenant.updated_at,
        }

    async def get_all_usage(self) -> List[Dict[str, Any]]:
        result = []
        for tenant in self._tenants.values():
            usage = await self.get_tenant_usage(tenant.tenant_id)
            if usage:
                result.append(usage)
        # Sort by tokens used descending
        result.sort(key=lambda u: u["tokens_used"], reverse=True)
        return result


# --------------------------------------------------------------------------- #
# Global instance                                                             #
# --------------------------------------------------------------------------- #

_tenant_manager: Optional[TenantManager] = None


def get_tenant_manager() -> TenantManager:
    global _tenant_manager
    if _tenant_manager is None:
        _tenant_manager = TenantManager()
    return _tenant_manager


def extract_tenant_id(headers: Dict[str, str]) -> str:
    """Extract tenant ID from request headers. Returns default tenant if not present."""
    tid = headers.get(TENANT_HEADER) or headers.get(TENANT_HEADER.lower())
    if tid:
        return tid.strip()
    return DEFAULT_TENANT_ID
