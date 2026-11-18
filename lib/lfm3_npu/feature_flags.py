"""
Feature flags for LFM3-1.7B inference servers.

Provides runtime-toggleable feature flags with:
  * Boolean flags (on/off)
  * Percentage-based rollout (0-100%)
  * Whitelist/blacklist for specific clients
  * Persistent storage to JSON file
  * Change callbacks for reactive updates
  * Thread-safe access

Enterprise use cases:
  * Kill switches for problematic features
  * Gradual rollout of new inference paths
  * A/B testing between CPU and NPU routing
  * Premium feature gating

All flags are stored in a JSON file (default:
/media/scott/data/lfm3-npu-tool/var/feature_flags.json) and can be
updated at runtime without restarting servers.
"""

import json
import os
import random
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set

DEFAULT_FLAGS_FILE = Path(
    "/media/scott/data/lfm3-npu-tool/var/feature_flags.json"
)

# Built-in flags with sensible defaults
DEFAULT_FLAGS: Dict[str, Dict[str, Any]] = {
    "npu_inference_enabled": {
        "type": "boolean",
        "enabled": True,
        "description": "Enable NPU inference path",
    },
    "cpu_inference_enabled": {
        "type": "boolean",
        "enabled": True,
        "description": "Enable CPU inference path",
    },
    "streaming_enabled": {
        "type": "boolean",
        "enabled": True,
        "description": "Enable streaming chat completions",
    },
    "continuous_batching_enabled": {
        "type": "boolean",
        "enabled": True,
        "description": "Enable continuous batching for CPU server",
    },
    "tracing_enabled": {
        "type": "boolean",
        "enabled": True,
        "description": "Enable OpenTelemetry tracing",
    },
    "rate_limiting_enabled": {
        "type": "boolean",
        "enabled": True,
        "description": "Enable rate limiting",
    },
    "audit_logging_enabled": {
        "type": "boolean",
        "enabled": True,
        "description": "Enable audit logging",
    },
    "new_routing_algorithm": {
        "type": "percentage",
        "enabled": False,
        "percentage": 10,
        "description": "Route 10% of traffic to new routing algorithm",
    },
    "premium_features": {
        "type": "boolean",
        "enabled": False,
        "description": "Premium features gate",
    },
}


class FeatureFlag:
    """Represents a single feature flag."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name = name
        self.flag_type = config.get("type", "boolean")
        self.enabled = config.get("enabled", False)
        self.percentage = config.get("percentage", 0)
        self.whitelist: Set[str] = set(config.get("whitelist", []))
        self.blacklist: Set[str] = set(config.get("blacklist", []))
        self.description = config.get("description", "")

    def is_enabled(self, client_id: Optional[str] = None) -> bool:
        """
        Check if the flag is enabled for a given client.

        Logic:
        1. Blacklist takes precedence - always disabled
        2. Whitelist - always enabled for listed clients
        3. Boolean flags - return enabled state
        4. Percentage flags - hash-based consistent rollout
        """
        # Blacklist check
        if client_id and client_id in self.blacklist:
            return False

        # Whitelist check
        if client_id and client_id in self.whitelist:
            return True

        if self.flag_type == "boolean":
            return self.enabled

        if self.flag_type == "percentage":
            if not self.enabled:
                return False
            if self.percentage >= 100:
                return True
            if self.percentage <= 0:
                return False
            # Consistent hash-based rollout: same client always gets same result
            hash_input = f"{self.name}:{client_id or 'anonymous'}"
            hash_value = int(
                __import__("hashlib").sha256(hash_input.encode()).hexdigest(), 16
            )
            return (hash_value % 100) < self.percentage

        return self.enabled

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.flag_type,
            "enabled": self.enabled,
            "percentage": self.percentage,
            "whitelist": sorted(self.whitelist),
            "blacklist": sorted(self.blacklist),
            "description": self.description,
        }


class FeatureFlags:
    """
    Thread-safe feature flag store with persistence and change callbacks.

    Flags are loaded from a JSON file on initialization and saved back
    on every change. Supports change callbacks for reactive updates.
    """

    def __init__(self, flags_file: Optional[str] = None) -> None:
        self.flags_file = Path(flags_file or DEFAULT_FLAGS_FILE)
        self._lock = threading.RLock()
        self._flags: Dict[str, FeatureFlag] = {}
        self._callbacks: list[Callable[[str, bool], None]] = []
        self._load()

    def _load(self) -> None:
        """Load flags from disk, falling back to defaults."""
        self._flags = {}
        if self.flags_file.exists():
            try:
                with open(self.flags_file, "r") as f:
                    data = json.load(f)
                for name, config in data.items():
                    self._flags[name] = FeatureFlag(name, config)
            except (json.JSONDecodeError, OSError):
                # Corrupt file - use defaults
                pass
        # Ensure all default flags exist
        for name, config in DEFAULT_FLAGS.items():
            if name not in self._flags:
                self._flags[name] = FeatureFlag(name, config)
        self._save()

    def _save(self) -> None:
        """Persist flags to disk."""
        self.flags_file.parent.mkdir(parents=True, exist_ok=True)
        data = {name: flag.to_dict() for name, flag in self._flags.items()}
        tmp_file = self.flags_file.with_suffix(".json.tmp")
        with open(tmp_file, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_file, self.flags_file)

    def is_enabled(self, name: str, client_id: Optional[str] = None) -> bool:
        """Check if a flag is enabled for a client."""
        with self._lock:
            flag = self._flags.get(name)
            if flag is None:
                return False
            return flag.is_enabled(client_id)

    def set_flag(
        self,
        name: str,
        enabled: bool,
        flag_type: str = "boolean",
        percentage: int = 0,
        description: Optional[str] = None,
    ) -> None:
        """Set or create a feature flag."""
        with self._lock:
            old_enabled = False
            if name in self._flags:
                old_enabled = self._flags[name].enabled

            flag = self._flags.get(name) or FeatureFlag(name, {"type": flag_type})
            flag.flag_type = flag_type
            flag.enabled = enabled
            if flag_type == "percentage":
                flag.percentage = max(0, min(100, percentage))
            if description is not None:
                flag.description = description
            self._flags[name] = flag
            self._save()

        # Invoke callbacks outside the lock
        if old_enabled != enabled:
            callbacks = list(self._callbacks)
            for cb in callbacks:
                try:
                    cb(name, enabled)
                except Exception:
                    pass

    def add_whitelist(self, name: str, client_id: str) -> None:
        """Add a client to a flag's whitelist."""
        with self._lock:
            flag = self._flags.get(name)
            if flag is None:
                return
            flag.whitelist.add(client_id)
            self._save()

    def add_blacklist(self, name: str, client_id: str) -> None:
        """Add a client to a flag's blacklist."""
        with self._lock:
            flag = self._flags.get(name)
            if flag is None:
                return
            flag.blacklist.add(client_id)
            self._save()

    def remove_whitelist(self, name: str, client_id: str) -> None:
        """Remove a client from a flag's whitelist."""
        with self._lock:
            flag = self._flags.get(name)
            if flag is None:
                return
            flag.whitelist.discard(client_id)
            self._save()

    def remove_blacklist(self, name: str, client_id: str) -> None:
        """Remove a client from a flag's blacklist."""
        with self._lock:
            flag = self._flags.get(name)
            if flag is None:
                return
            flag.blacklist.discard(client_id)
            self._save()

    def get_flag(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a flag's configuration as a dict."""
        with self._lock:
            flag = self._flags.get(name)
            return flag.to_dict() if flag else None

    def list_flags(self) -> Dict[str, Dict[str, Any]]:
        """Get all flags as a dict."""
        with self._lock:
            return {name: flag.to_dict() for name, flag in self._flags.items()}

    def add_callback(self, callback: Callable[[str, bool], None]) -> None:
        """Register a callback invoked when a flag changes state."""
        with self._lock:
            self._callbacks.append(callback)

    def delete_flag(self, name: str) -> None:
        """Delete a custom flag (built-in flags cannot be deleted)."""
        with self._lock:
            if name in DEFAULT_FLAGS:
                return  # Cannot delete built-in flags
            self._flags.pop(name, None)
            self._save()


# Global feature flag store
_flags: Optional[FeatureFlags] = None


def get_flags() -> FeatureFlags:
    """Get the global feature flag store (initializes on first use)."""
    global _flags
    if _flags is None:
        _flags = FeatureFlags()
    return _flags
