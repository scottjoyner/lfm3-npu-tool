"""
Model versioning and canary routing for LFM3-1.7B inference servers.

Provides:
  * Model registry - track multiple model versions with metadata
  * Canary traffic splitting - route % of traffic to canary model version
  * Canary promotion - promote canary to primary
  * Model rollback - revert to previous primary version
  * Per-version metrics tracking
  * Header-based routing override (X-Model-Version)
  * Persistence to JSON file
  * CLI integration for model management

Usage:
    # Route 10% of traffic to canary model
    lfm3-npu model canary-set lfm3-1.7b-v2 --traffic 10

    # Promote canary to primary
    lfm3-npu model canary-promote

    # Rollback to previous version
    lfm3-npu model rollback

    # List all model versions
    lfm3-npu model list

Header: X-Model-Version (override routing for specific requests)
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Model statuses
MODEL_STATUS_PRIMARY = "primary"
MODEL_STATUS_CANARY = "canary"
MODEL_STATUS_DEPRECATED = "deprecated"
MODEL_STATUS_PENDING = "pending"

# Default model
DEFAULT_MODEL_ID = "lfm3-1.7b"


@dataclass
class ModelVersion:
    """A model version in the registry."""

    model_id: str
    version: str
    status: str = MODEL_STATUS_PENDING  # primary, canary, deprecated, pending
    traffic_percent: int = 0            # % of traffic routed to this version (canary only)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    provider: Optional[str] = None
    checksum: Optional[str] = None
    notes: Optional[str] = None
    # Per-version metrics
    total_requests: int = 0
    total_errors: int = 0
    total_latency_ms: float = 0.0

    @property
    def is_primary(self) -> bool:
        return self.status == MODEL_STATUS_PRIMARY

    @property
    def is_canary(self) -> bool:
        return self.status == MODEL_STATUS_CANARY

    @property
    def is_deprecated(self) -> bool:
        return self.status == MODEL_STATUS_DEPRECATED

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_requests if self.total_requests else 0.0

    @property
    def error_rate(self) -> float:
        return self.total_errors / self.total_requests if self.total_requests else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


class ModelRegistry:
    """Manages model versions: registry, canary routing, promotion, rollback."""

    def __init__(self, storage_path: Optional[str] = None):
        if storage_path is None:
            storage_path = os.environ.get(
                "LFM3_MODEL_REGISTRY",
                str(Path.home() / ".config" / "lfm3-npu" / "models.json"),
            )
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, ModelVersion] = {}  # keyed by version string
        self._load()

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        """Load model registry from disk."""
        if not self.storage_path.exists():
            self._create_default_model()
            self._save()
            return
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            for ver, mdata in data.items():
                self._models[ver] = ModelVersion.from_dict(mdata)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[ModelRegistry] Warning: could not load {self.storage_path}: {e}", flush=True)
            self._create_default_model()

    def _save(self) -> None:
        """Save model registry to disk."""
        tmp = self.storage_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({ver: m.to_dict() for ver, m in self._models.items()}, f, indent=2)
        tmp.replace(self.storage_path)

    def _create_default_model(self) -> None:
        """Create the default primary model."""
        self._models["1.0.0"] = ModelVersion(
            model_id=DEFAULT_MODEL_ID,
            version="1.0.0",
            status=MODEL_STATUS_PRIMARY,
            traffic_percent=0,
            notes="Initial production model",
        )

    # ------------------------------------------------------------------ #
    # CRUD                                                               #
    # ------------------------------------------------------------------ #

    def register_model(
        self,
        version: str,
        model_id: str = DEFAULT_MODEL_ID,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        provider: Optional[str] = None,
        checksum: Optional[str] = None,
        notes: Optional[str] = None,
        status: str = MODEL_STATUS_PENDING,
    ) -> ModelVersion:
        """Register a new model version."""
        if version in self._models:
            raise ValueError(f"Model version '{version}' already exists")
        if status not in (MODEL_STATUS_PRIMARY, MODEL_STATUS_CANARY,
                          MODEL_STATUS_DEPRECATED, MODEL_STATUS_PENDING):
            raise ValueError(f"Invalid status '{status}'")
        model = ModelVersion(
            model_id=model_id,
            version=version,
            status=status,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            provider=provider,
            checksum=checksum,
            notes=notes,
        )
        self._models[version] = model
        self._save()
        return model

    def get_model(self, version: str) -> Optional[ModelVersion]:
        return self._models.get(version)

    def list_models(self) -> List[ModelVersion]:
        return sorted(self._models.values(), key=lambda m: m.created_at, reverse=True)

    def update_model(
        self,
        version: str,
        model_id: Optional[str] = None,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        provider: Optional[str] = None,
        checksum: Optional[str] = None,
        notes: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Optional[ModelVersion]:
        model = self._models.get(version)
        if model is None:
            return None
        if model_id is not None:
            model.model_id = model_id
        if model_path is not None:
            model.model_path = model_path
        if tokenizer_path is not None:
            model.tokenizer_path = tokenizer_path
        if provider is not None:
            model.provider = provider
        if checksum is not None:
            model.checksum = checksum
        if notes is not None:
            model.notes = notes
        if status is not None:
            if status not in (MODEL_STATUS_PRIMARY, MODEL_STATUS_CANARY,
                              MODEL_STATUS_DEPRECATED, MODEL_STATUS_PENDING):
                raise ValueError(f"Invalid status '{status}'")
            model.status = status
        model.updated_at = time.time()
        self._save()
        return model

    def delete_model(self, version: str) -> bool:
        model = self._models.get(version)
        if model is None:
            return False
        if model.is_primary:
            raise ValueError("Cannot delete the primary model version")
        del self._models[version]
        self._save()
        return True

    # ------------------------------------------------------------------ #
    # Canary routing                                                     #
    # ------------------------------------------------------------------ #

    def get_primary(self) -> Optional[ModelVersion]:
        """Get the current primary model version."""
        for m in self._models.values():
            if m.is_primary:
                return m
        return None

    def get_canary(self) -> Optional[ModelVersion]:
        """Get the current canary model version (if any)."""
        for m in self._models.values():
            if m.is_canary:
                return m
        return None

    def set_canary(
        self,
        version: str,
        traffic_percent: int,
    ) -> ModelVersion:
        """Set a model version as canary with the given traffic percentage."""
        if not 0 <= traffic_percent <= 100:
            raise ValueError("traffic_percent must be between 0 and 100")

        # Remove existing canary
        for m in self._models.values():
            if m.is_canary:
                m.status = MODEL_STATUS_PENDING
                m.traffic_percent = 0
                m.updated_at = time.time()

        # Set new canary
        model = self._models.get(version)
        if model is None:
            raise ValueError(f"Model version '{version}' not found")
        model.status = MODEL_STATUS_CANARY
        model.traffic_percent = traffic_percent
        model.updated_at = time.time()
        self._save()
        return model

    def resolve_model_version(
        self,
        requested_version: Optional[str] = None,
    ) -> Optional[ModelVersion]:
        """
        Resolve which model version to use for a request.

        Logic:
          1. If X-Model-Version header matches an existing version, use it
          2. If canary is configured, roll dice based on traffic_percent
          3. Otherwise, use primary
        """
        # Explicit version override
        if requested_version:
            model = self._models.get(requested_version)
            if model and not model.is_deprecated:
                return model

        # Canary traffic split
        canary = self.get_canary()
        if canary and canary.traffic_percent > 0:
            import random
            if random.randint(1, 100) <= canary.traffic_percent:
                return canary

        # Default to primary
        return self.get_primary()

    def promote_canary(self) -> Optional[ModelVersion]:
        """Promote the canary model to primary. Returns the new primary."""
        canary = self.get_canary()
        if canary is None:
            raise ValueError("No canary model to promote")

        # Demote current primary
        primary = self.get_primary()
        if primary:
            primary.status = MODEL_STATUS_DEPRECATED
            primary.updated_at = time.time()

        # Promote canary to primary
        canary.status = MODEL_STATUS_PRIMARY
        canary.traffic_percent = 0
        canary.updated_at = time.time()
        self._save()
        return canary

    def rollback(self) -> Optional[ModelVersion]:
        """
        Rollback: demote current primary to deprecated, promote the most
        recent non-deprecated version to primary.
        """
        primary = self.get_primary()
        if primary is None:
            raise ValueError("No primary model to rollback from")

        # Find previous primary (deprecated versions sorted by updated_at)
        deprecated = sorted(
            [m for m in self._models.values() if m.is_deprecated],
            key=lambda m: m.updated_at,
            reverse=True,
        )

        # Demote current primary
        primary.status = MODEL_STATUS_DEPRECATED
        primary.updated_at = time.time()

        # Promote most recent deprecated version back to primary
        if deprecated:
            previous = deprecated[0]
            previous.status = MODEL_STATUS_PRIMARY
            previous.updated_at = time.time()
            self._save()
            return previous

        # No previous version - just demote, no primary
        self._save()
        return None

    # ------------------------------------------------------------------ #
    # Metrics                                                            #
    # ------------------------------------------------------------------ #

    def record_request(
        self,
        version: str,
        latency_ms: float,
        error: bool = False,
    ) -> None:
        """Record a request metric for a model version."""
        model = self._models.get(version)
        if model:
            model.total_requests += 1
            if error:
                model.total_errors += 1
            model.total_latency_ms += latency_ms
            model.updated_at = time.time()
            self._save()

    def get_model_metrics(self, version: str) -> Optional[Dict[str, Any]]:
        model = self._models.get(version)
        if model is None:
            return None
        return {
            "version": model.version,
            "model_id": model.model_id,
            "status": model.status,
            "traffic_percent": model.traffic_percent,
            "total_requests": model.total_requests,
            "total_errors": model.total_errors,
            "error_rate": round(model.error_rate, 4),
            "avg_latency_ms": round(model.avg_latency_ms, 2),
            "created_at": model.created_at,
            "updated_at": model.updated_at,
        }


# --------------------------------------------------------------------------- #
# Global instance                                                             #
# --------------------------------------------------------------------------- #

_model_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry
