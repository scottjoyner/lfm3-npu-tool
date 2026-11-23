"""
Configuration schema validation for LFM3-1.7B NPU tooling.

Provides pydantic-based validation of the YAML configuration with clear,
actionable error messages. Also supports config hot-reload via file
watching.

Usage::

    from lfm3_npu.config_schema import ConfigSchema, validate_config

    schema = ConfigSchema()
    validated = schema.validate(raw_dict)   # raises ConfigValidationError on failure

Or validate a loaded Config object::

    config = load_config()
    schema.validate(config.to_dict())
"""

import copy
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        message = "Configuration validation failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        super().__init__(message)


# ---------------------------------------------------------------------------
# Pydantic schema models
# ---------------------------------------------------------------------------


class OrtSessionModel(BaseModel):
    intra_op_num_threads: int = Field(default=0, ge=0)
    inter_op_num_threads: int = Field(default=0, ge=0)
    execution_mode: str = Field(default="ORT_SEQUENTIAL")
    graph_optimization_level: str = Field(default="ORT_ENABLE_ALL")
    enable_cpu_mem_arena: bool = Field(default=True)
    enable_mem_reuse: bool = Field(default=True)
    enable_mem_pattern: bool = Field(default=True)
    session_config_entries: Dict[str, str] = Field(default_factory=dict)

    @field_validator("execution_mode")
    @classmethod
    def _validate_exec_mode(cls, v: str) -> str:
        allowed = {"ORT_SEQUENTIAL", "ORT_PARALLEL"}
        if v not in allowed:
            raise ValueError(f"execution_mode must be one of {allowed}, got {v}")
        return v

    @field_validator("graph_optimization_level")
    @classmethod
    def _validate_opt_level(cls, v: str) -> str:
        allowed = {"ORT_DISABLE_ALL", "ORT_ENABLE_BASIC", "ORT_ENABLE_EXTENDED", "ORT_ENABLE_ALL"}
        if v not in allowed:
            raise ValueError(f"graph_optimization_level must be one of {allowed}, got {v}")
        return v


class BenchmarkModel(BaseModel):
    warmup_iterations: int = Field(default=1, ge=0)
    measure_iterations: int = Field(default=1, ge=0)
    concurrency_levels: List[int] = Field(default_factory=lambda: [1, 2, 4, 8])


class ServerPerfModel(BaseModel):
    max_concurrent_requests: int = Field(default=4, ge=1)
    request_timeout_seconds: int = Field(default=120, ge=1)
    enable_request_batching: bool = Field(default=False)
    batch_timeout_ms: int = Field(default=50, ge=0)


class PerformanceModel(BaseModel):
    ort_session: OrtSessionModel = Field(default_factory=OrtSessionModel)
    benchmark: BenchmarkModel = Field(default_factory=BenchmarkModel)
    server: ServerPerfModel = Field(default_factory=ServerPerfModel)


class NPUModel(BaseModel):
    provider: str = Field(default="VitisAIExecutionProvider")
    device_index: int = Field(default=0, ge=0)
    custom_ops_lib: str = Field(default="")
    python_executable: str = Field(default="")
    ld_library_path: List[str] = Field(default_factory=list)

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        allowed = {"VitisAIExecutionProvider", "RyzenAIExecutionProvider", "CPUExecutionProvider"}
        if v not in allowed:
            raise ValueError(f"npu.provider must be one of {allowed}, got {v}")
        return v


class ServerModel(BaseModel):
    port: int = Field(default=1238, ge=1, le=65535)
    host: str = Field(default="0.0.0.0")
    model_path: str = Field(default="")
    model_type: str = Field(default="")


class ServersModel(BaseModel):
    cpu: ServerModel = Field(default_factory=lambda: ServerModel(port=1238))
    npu: ServerModel = Field(default_factory=lambda: ServerModel(port=1239))


class GenerationModel(BaseModel):
    max_tokens: int = Field(default=50, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    do_sample: bool = Field(default=True)


class LoggingModel(BaseModel):
    level: str = Field(default="INFO")
    log_dir: str = Field(default="/var/log/lfm3-npu")

    @field_validator("level")
    @classmethod
    def _validate_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"logging.level must be one of {allowed}, got {v}")
        return v.upper()


class SecurityModel(BaseModel):
    api_keys: str = Field(default="")
    auth_enabled: bool = Field(default=False)


class RateLimitModel(BaseModel):
    requests_per_minute: int = Field(default=60, ge=1)
    burst: int = Field(default=10, ge=1)


class CircuitBreakerModel(BaseModel):
    failure_threshold: int = Field(default=5, ge=1)
    recovery_timeout_seconds: float = Field(default=30.0, ge=0.0)


class SecurityHeadersModel(BaseModel):
    hsts_max_age: int = Field(default=31536000, ge=0)
    hsts_include_subdomains: bool = Field(default=True)
    hsts_preload: bool = Field(default=True)
    disable_hsts: bool = Field(default=False)
    referrer_policy: str = Field(default="strict-origin-when-cross-origin")


class AuditModel(BaseModel):
    service_name: str = Field(default="lfm3-inference")


class LoadShedModel(BaseModel):
    max_active_requests: int = Field(default=100, ge=1)


class ModelPromotionModel(BaseModel):
    storage_path: str = Field(default="/media/scott/data/lfm3-npu-tool/var/promotion.json")
    auto_approve: bool = Field(default=False)
    stages: List[str] = Field(default_factory=lambda: ["staging", "canary", "production"])


class DRRegionModel(BaseModel):
    name: str
    status: str = Field(default="secondary")
    endpoint: Optional[str] = None


class DisasterRecoveryModel(BaseModel):
    storage_path: str = Field(default="/media/scott/data/lfm3-npu-tool/var/dr.json")
    regions: List[DRRegionModel] = Field(default_factory=list)


class AutoScalingModel(BaseModel):
    storage_path: str = Field(default="/media/scott/data/lfm3-npu-tool/var/autoscaler.json")
    min_replicas: int = Field(default=1, ge=1)
    max_replicas: int = Field(default=10, ge=1)
    target_cpu_percent: float = Field(default=70.0, ge=0.0, le=100.0)
    target_memory_percent: float = Field(default=80.0, ge=0.0, le=100.0)
    target_queue_depth: int = Field(default=10, ge=0)
    target_latency_p95_ms: float = Field(default=1000.0, ge=0.0)
    max_error_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    scale_up_cooldown_seconds: int = Field(default=60, ge=0)
    scale_down_cooldown_seconds: int = Field(default=300, ge=0)
    evaluation_interval_seconds: int = Field(default=30, ge=1)


class ABTestingModel(BaseModel):
    storage_path: str = Field(default="/media/scott/data/lfm3-npu-tool/var/ab_tests.json")


class CostTrackingModel(BaseModel):
    storage_path: str = Field(default="/media/scott/data/lfm3-npu-tool/var/cost_tracking.json")
    default_tier: str = Field(default="standard")
    pricing_tiers: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class AlertingModel(BaseModel):
    storage_path: str = Field(default="/media/scott/data/lfm3-npu-tool/var/alerts.json")
    webhook_url: str = Field(default="")
    slack_webhook: str = Field(default="")
    pagerduty_key: str = Field(default="")
    severity_levels: List[str] = Field(default_factory=lambda: ["info", "warning", "critical"])


class SecurityScanModel(BaseModel):
    storage_path: str = Field(default="/media/scott/data/lfm3-npu-tool/var/security_scan.json")
    scan_paths: List[str] = Field(default_factory=list)
    exclude_paths: List[str] = Field(default_factory=list)


class KeyRotationPolicyModel(BaseModel):
    name: str
    max_age_days: Optional[int] = None
    max_usage_count: Optional[int] = None
    max_cost_usd: Optional[float] = None


class KeyRotationModel(BaseModel):
    storage_path: str = Field(default="/media/scott/data/lfm3-npu-tool/var/key_rotation.json")
    policies: List[KeyRotationPolicyModel] = Field(default_factory=list)


class ConfigSchema(BaseModel):
    """Complete configuration schema."""

    model: dict = Field(default_factory=dict)
    npu: NPUModel = Field(default_factory=NPUModel)
    server: ServersModel = Field(default_factory=ServersModel)
    generation: GenerationModel = Field(default_factory=GenerationModel)
    logging: LoggingModel = Field(default_factory=LoggingModel)
    security: SecurityModel = Field(default_factory=SecurityModel)
    rate_limit: RateLimitModel = Field(default_factory=RateLimitModel)
    circuit_breaker: CircuitBreakerModel = Field(default_factory=CircuitBreakerModel)
    security_headers: SecurityHeadersModel = Field(default_factory=SecurityHeadersModel)
    audit: AuditModel = Field(default_factory=AuditModel)
    load_shed: LoadShedModel = Field(default_factory=LoadShedModel)
    performance: PerformanceModel = Field(default_factory=PerformanceModel)
    model_promotion: ModelPromotionModel = Field(default_factory=ModelPromotionModel)
    disaster_recovery: DisasterRecoveryModel = Field(default_factory=DisasterRecoveryModel)
    auto_scaling: AutoScalingModel = Field(default_factory=AutoScalingModel)
    ab_testing: ABTestingModel = Field(default_factory=ABTestingModel)
    cost_tracking: CostTrackingModel = Field(default_factory=CostTrackingModel)
    alerting: AlertingModel = Field(default_factory=AlertingModel)
    security_scan: SecurityScanModel = Field(default_factory=SecurityScanModel)
    key_rotation: KeyRotationModel = Field(default_factory=KeyRotationModel)

    @model_validator(mode="after")
    def _validate_server_ports(self) -> "ConfigSchema":
        # Cross-check CPU != NPU port
        if self.server.cpu.port == self.server.npu.port:
            raise ValueError(
                f"CPU and NPU ports must differ (both {self.server.cpu.port})"
            )
        return self


# ---------------------------------------------------------------------------
# Validation function
# ---------------------------------------------------------------------------


def validate_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a configuration dictionary against the schema.

    Returns the validated (and possibly coerced) dictionary.
    Raises ConfigValidationError with a list of human-readable errors on failure.
    """
    try:
        validated = ConfigSchema.model_validate(data)
        return validated.model_dump()
    except Exception as e:
        # Pydantic v2 ValidationError
        from pydantic import ValidationError

        if isinstance(e, ValidationError):
            errors: List[str] = []
            for err in e.errors():
                loc = ".".join(str(x) for x in err["loc"])
                msg = err["msg"]
                errors.append(f"{loc}: {msg}")
            raise ConfigValidationError(errors)
        raise ConfigValidationError([str(e)])


# ---------------------------------------------------------------------------
# Config hot-reload
# ---------------------------------------------------------------------------


class ConfigWatcher:
    """
    Watches a config file for changes and invokes callbacks when it changes.

    Polls the file's mtime at a configurable interval.  Thread-safe.
    """

    def __init__(self, config_path: str, poll_interval: float = 5.0) -> None:
        self.config_path = Path(config_path)
        self.poll_interval = poll_interval
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_mtime: Optional[float] = None
        self._last_data: Optional[Dict[str, Any]] = None

    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback to invoke when config changes."""
        with self._lock:
            self._callbacks.append(callback)

    def _load(self) -> Optional[Dict[str, Any]]:
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return None

    def check(self) -> bool:
        """Check once for changes.  Returns True if config was reloaded."""
        if not self.config_path.exists():
            return False
        mtime = self.config_path.stat().st_mtime
        with self._lock:
            if self._last_mtime is not None and mtime == self._last_mtime:
                return False
            data = self._load()
            if data is None:
                return False
            if self._last_data is not None and data == self._last_data:
                self._last_mtime = mtime
                return False
            self._last_mtime = mtime
            self._last_data = copy.deepcopy(data)
        callbacks = list(self._callbacks)
    # Invoke callbacks outside the lock
        for cb in callbacks:
            try:
                cb(data)
            except Exception as e:
                print(f"Config reload callback failed: {e}", flush=True)
        return True

    def start(self) -> None:
        """Start the background polling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running:
            self.check()
            time.sleep(self.poll_interval)

    def stop(self) -> None:
        """Stop the background polling thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def running(self) -> bool:
        return self._running


# ---------------------------------------------------------------------------
# Reloadable config container
# ---------------------------------------------------------------------------


class ReloadableConfig:
    """
    A Config that can be hot-reloaded from disk.

    Wraps a ConfigWatcher and re-validates on each change.  Callbacks
    registered via on_reload are invoked with the new Config object.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self._config: Optional[Config] = None
        self._raw_data: Optional[Dict[str, Any]] = None
        self._callbacks: List[Callable[[Config], None]] = []

        path = config_path or os.environ.get("LFM3_NPU_CONFIG", "")
        if not path:
            # Use the same search order as load_config
            from lfm3_npu.config import _find_config_file

            path = _find_config_file()

        self._watcher = ConfigWatcher(path or "", poll_interval=5.0)
        if path:
            self._watcher.add_callback(self._on_file_change)
            # Initial load: force a check to populate watcher state
            self._watcher.check()
            self._on_file_change(None)

    def _on_file_change(self, raw_data: Optional[Dict[str, Any]]) -> None:
        if raw_data is None:
            raw_data = self._watcher._last_data
        if raw_data is None:
            return
        try:
            validated = validate_config(copy.deepcopy(raw_data))
        except ConfigValidationError as e:
            print(f"Config reload validation failed: {e}", flush=True)
            return
        from lfm3_npu.config import Config

        new_config = Config(validated)
        with self._lock:
            self._config = new_config
            self._raw_data = validated
        callbacks = list(self._callbacks)
    # Invoke callbacks outside the lock
        for cb in callbacks:
            try:
                cb(new_config)
            except Exception as e:
                print(f"Config reload callback failed: {e}", flush=True)

    def get(self) -> Optional[Config]:
        """Get the current Config (thread-safe)."""
        with self._lock:
            return self._config

    def raw(self) -> Optional[Dict[str, Any]]:
        """Get the current raw validated dict (thread-safe)."""
        with self._lock:
            return self._raw_data

    def add_callback(self, callback: Callable[[Config], None]) -> None:
        """Register a callback to invoke when config is reloaded."""
        with self._lock:
            self._callbacks.append(callback)

    def start_watching(self) -> None:
        """Start the background config watcher."""
        self._watcher.start()

    def stop_watching(self) -> None:
        """Stop the background config watcher."""
        self._watcher.stop()

    @property
    def watcher(self) -> ConfigWatcher:
        return self._watcher


# ---------------------------------------------------------------------------
# Convenience: validate the defaults
# ---------------------------------------------------------------------------

from lfm3_npu.config import DEFAULTS as _DEFAULTS

# Validate defaults at import time to catch schema errors early
try:
    validate_config(copy.deepcopy(_DEFAULTS))
except ConfigValidationError:
    pass  # Defaults should always be valid; ignore if not
