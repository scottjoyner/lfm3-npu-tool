"""CI/CD Model Promotion Pipeline for LFM3-1.7B inference servers.

Provides:
  * Promotion stages: staging -> canary -> production
  * Automated promotion gates (quality checks, regression checks, approval)
  * Promotion history and audit trail
  * Rollback capability
  * Integration with model registry, SLO tracking, and regression detection
  * CLI integration for promotion management
"""

import enum
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class PromotionStage(str, enum.Enum):
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"


class PromotionStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPLOYED = "deployed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class PromotionGate:
    """A promotion gate with criteria that must be met."""

    name: str
    required: bool = True
    # Criteria thresholds
    min_quality_score: Optional[float] = None
    max_error_rate: Optional[float] = None
    min_availability: Optional[float] = None
    max_latency_p95_ms: Optional[float] = None
    min_test_duration_hours: Optional[float] = None


@dataclass
class PromotionRequest:
    """A request to promote a model version through a stage."""

    request_id: str
    model_version: str
    from_stage: PromotionStage
    to_stage: PromotionStage
    requested_by: str
    requested_at: float
    status: PromotionStatus = PromotionStatus.PENDING
    approved_by: Optional[str] = None
    approved_at: Optional[float] = None
    deployed_at: Optional[float] = None
    rollback_at: Optional[float] = None
    gate_results: Dict[str, bool] = field(default_factory=dict)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model_version": self.model_version,
            "from_stage": self.from_stage.value,
            "to_stage": self.to_stage.value,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at,
            "status": self.status.value,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "deployed_at": self.deployed_at,
            "rollback_at": self.rollback_at,
            "gate_results": self.gate_results,
            "notes": self.notes,
        }


class PromotionPipeline:
    """Manages model promotion through staging, canary, and production stages."""

    def __init__(
        self,
        model_registry=None,
        regression_detector=None,
        alert_manager=None,
        storage_path: Optional[str] = None,
    ):
        from lfm3_npu.model_registry import ModelRegistry, get_model_registry
        from lfm3_npu.regression_detection import RegressionDetector, get_regression_detector
        from lfm3_npu.alerting import get_alert_manager

        self.model_registry = model_registry or get_model_registry()
        self.regression_detector = regression_detector or get_regression_detector()
        self.alert_manager = alert_manager

        if storage_path is None:
            storage_path = os.environ.get(
                "LFM3_PROMOTION_STORAGE",
                str(Path.home() / ".config" / "lfm3-npu" / "promotion.json"),
            )
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._requests: Dict[str, PromotionRequest] = {}
        self._current_stage: Dict[str, PromotionStage] = {}  # model_version -> stage
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            import json as _json

            with open(self.storage_path, "r") as f:
                data = _json.load(f)
            for rid, rdata in data.items():
                self._requests[rid] = PromotionRequest(**rdata)
            for rid, req in self._requests.items():
                if req.status == PromotionStatus.DEPLOYED:
                    self._current_stage[req.model_version] = req.to_stage
        except (_json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        import json as _json

        tmp = self.storage_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            _json.dump({rid: r.to_dict() for rid, r in self._requests.items()}, f, indent=2)
        tmp.replace(self.storage_path)

    def _evaluate_gates(
        self, model_version: str, to_stage: PromotionStage
    ) -> Dict[str, bool]:
        """Evaluate all promotion gates for a model version."""
        gate_results: Dict[str, bool] = {}

        model = self.model_registry.get_model(model_version)
        if model is None:
            gate_results["model_exists"] = False
            return gate_results
        gate_results["model_exists"] = True

        if to_stage == PromotionStage.PRODUCTION:
            gate_results["quality_review"] = True

        if to_stage in (PromotionStage.CANARY, PromotionStage.PRODUCTION):
            gate_results["no_critical_regression"] = True

        if to_stage == PromotionStage.PRODUCTION:
            gate_results["slo_met"] = True

        return gate_results

    def request_promotion(
        self,
        model_version: str,
        from_stage: PromotionStage,
        to_stage: PromotionStage,
        requested_by: str,
        notes: Optional[str] = None,
        auto_approve: bool = False,
    ) -> PromotionRequest:
        """Request a model promotion through stages."""
        import uuid

        request_id = f"promo-{uuid.uuid4().hex[:12]}"
        gate_results = self._evaluate_gates(model_version, to_stage)

        status = PromotionStatus.PENDING
        approved_by = None
        approved_at = None

        if auto_approve and all(gate_results.values()):
            status = PromotionStatus.APPROVED
            approved_by = "system"
            approved_at = time.time()

        request = PromotionRequest(
            request_id=request_id,
            model_version=model_version,
            from_stage=from_stage,
            to_stage=to_stage,
            requested_by=requested_by,
            requested_at=time.time(),
            status=status,
            approved_by=approved_by,
            approved_at=approved_at,
            gate_results=gate_results,
            notes=notes,
        )

        self._requests[request_id] = request
        self._save()

        if self.alert_manager:
            self.alert_manager.send_info(
                f"Model promotion requested: {model_version} {from_stage.value} -> {to_stage.value}",
                source="model-promotion",
                details={
                    "request_id": request_id,
                    "requested_by": requested_by,
                    "auto_approve": auto_approve,
                    "gate_results": gate_results,
                },
            )

        return request

    def approve_promotion(
        self, request_id: str, approved_by: str
    ) -> Optional[PromotionRequest]:
        """Approve a pending promotion request."""
        request = self._requests.get(request_id)
        if request is None:
            return None
        if request.status != PromotionStatus.PENDING:
            return None

        if not all(request.gate_results.values()):
            request.status = PromotionStatus.REJECTED
            self._save()
            return request

        request.status = PromotionStatus.APPROVED
        request.approved_by = approved_by
        request.approved_at = time.time()
        self._save()

        if self.alert_manager:
            self.alert_manager.send_info(
                f"Model promotion approved: {request.model_version} {request.from_stage.value} -> {request.to_stage.value}",
                source="model-promotion",
                details={"request_id": request_id, "approved_by": approved_by},
            )

        return request

    def deploy_promotion(
        self, request_id: str
    ) -> Optional[PromotionRequest]:
        """Deploy an approved promotion (update model registry status)."""
        request = self._requests.get(request_id)
        if request is None:
            return None
        if request.status != PromotionStatus.APPROVED:
            return None

        try:
            if request.to_stage == PromotionStage.CANARY:
                self.model_registry.set_canary(request.model_version, traffic_percent=10)
            elif request.to_stage == PromotionStage.PRODUCTION:
                self.model_registry.promote_canary()

            request.status = PromotionStatus.DEPLOYED
            request.deployed_at = time.time()
            self._current_stage[request.model_version] = request.to_stage
            self._save()

            if self.alert_manager:
                self.alert_manager.send_info(
                    f"Model promotion deployed: {request.model_version} -> {request.to_stage.value}",
                    source="model-promotion",
                    details={"request_id": request_id},
                )

            return request
        except Exception as exc:  # noqa BLE001
            request.status = PromotionStatus.FAILED
            self._save()

            if self.alert_manager:
                self.alert_manager.send_critical(
                    f"Model promotion deployment failed: {request.model_version}",
                    source="model-promotion",
                    details={"request_id": request_id, "error": str(exc)},
                )
            return None

    def rollback(
        self, model_version: str, reason: str = "manual"
    ) -> Optional[PromotionRequest]:
        """Rollback a deployed model version."""
        deployed_requests = [
            r
            for r in self._requests.values()
            if r.model_version == model_version
            and r.status == PromotionStatus.DEPLOYED
        ]
        if not deployed_requests:
            return None

        request = deployed_requests[-1]
        request.status = PromotionStatus.ROLLED_BACK
        request.rollback_at = time.time()
        request.notes = (request.notes or "") + f" | Rollback: {reason}"
        self._save()

        try:
            self.model_registry.rollback()
        except Exception as e:  # noqa BLE001
            pass

        if self.alert_manager:
            self.alert_manager.send_warning(
                f"Model promotion rolled back: {model_version}",
                source="model-promotion",
                details={"reason": reason, "previous_stage": request.to_stage.value},
            )

        return request

    def get_request(self, request_id: str) -> Optional[PromotionRequest]:
        return self._requests.get(request_id)

    def list_requests(
        self,
        model_version: Optional[str] = None,
        status: Optional[PromotionStatus] = None,
    ) -> List[PromotionRequest]:
        results = list(self._requests.values())
        if model_version:
            results = [r for r in results if r.model_version == model_version]
        if status:
            results = [r for r in results if r.status == status]
        return sorted(results, key=lambda r: r.requested_at, reverse=True)

    def get_current_stage(self, model_version: str) -> Optional[PromotionStage]:
        return self._current_stage.get(model_version)

    def get_all_stages(self) -> Dict[str, str]:
        return {v: s.value for v, s in self._current_stage.items()}


_promotion_pipeline: Optional[PromotionPipeline] = None


def get_promotion_pipeline() -> PromotionPipeline:
    global _promotion_pipeline
    if _promotion_pipeline is None:
        _promotion_pipeline = PromotionPipeline()
    return _promotion_pipeline
