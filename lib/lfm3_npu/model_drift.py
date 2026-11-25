"""Model Drift Detection for LFM3-1.7B inference servers.

Provides:
  * Drift detection between CPU and NPU inference outputs
  * Prompt-response comparison using string similarity
  * Drift score calculation (0.0 = identical, 1.0 = completely different)
  * Automatic drift alerting when threshold exceeded
  * Persistent drift history tracking
  * CLI integration for drift analysis

Usage:
    from lfm3_npu.model_drift import DriftDetector, DriftResult

    detector = DriftDetector()
    result = detector.detect_drift(cpu_response, npu_response)
    if result.is_drifted:
        print(f"Drift detected! Score: {result.drift_score}")
"""

import difflib
import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


class DriftResult:
    """Result of drift detection between two model outputs."""

    def __init__(
        self,
        timestamp: float,
        model_id: str,
        prompt: str,
        cpu_output: str,
        npu_output: str,
        drift_score: float,
        is_drifted: bool,
        similarity_ratio: float,
        cpu_length: int,
        npu_length: int,
    ):
        self.timestamp = timestamp
        self.model_id = model_id
        self.prompt = prompt
        self.cpu_output = cpu_output
        self.npu_output = npu_output
        self.drift_score = drift_score
        self.is_drifted = is_drifted
        self.similarity_ratio = similarity_ratio
        self.cpu_length = cpu_length
        self.npu_length = npu_length

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelDriftDetector:
    """Detects drift between CPU and NPU inference outputs."""

    def __init__(
        self,
        drift_threshold: float = 0.15,
        storage_path: Optional[str] = None,
    ):
        self.drift_threshold = drift_threshold
        if storage_path is None:
            storage_path = os.environ.get(
                "LFM3_DRIFT_STORAGE",
                str(Path.home() / ".config" / "lfm3-npu" / "drift_history.json"),
            )
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._drift_history: List[DriftResult] = []
        self._load()

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def _similarity_ratio(self, text1: str, text2: str) -> float:
        """Compute similarity ratio between two strings (0.0 to 1.0)."""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        return difflib.SequenceMatcher(None, text1, text2).ratio()

    def detect_drift(
        self,
        prompt: str,
        cpu_output: str,
        npu_output: str,
        model_id: str = "lfm3-1.7b",
    ) -> DriftResult:
        """
        Detect drift between CPU and NPU inference outputs.

        Args:
            prompt: The input prompt sent to both models.
            cpu_output: Output from CPU server.
            npu_output: Output from NPU server.
            model_id: Model identifier.

        Returns:
            DriftResult with drift score and detection status.
        """
        similarity = self._similarity_ratio(cpu_output, npu_output)
        drift_score = 1.0 - similarity
        is_drifted = drift_score > self.drift_threshold

        result = DriftResult(
            timestamp=time.time(),
            model_id=model_id,
            prompt=prompt,
            cpu_output=cpu_output,
            npu_output=npu_output,
            drift_score=drift_score,
            is_drifted=is_drifted,
            similarity_ratio=similarity,
            cpu_length=len(cpu_output),
            npu_length=len(npu_output),
        )

        self._drift_history.append(result)
        self._save()

        return result

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            import json as _json

            with open(self.storage_path, "r") as f:
                data = _json.load(f)
            for item in data.get("results", []):
                self._drift_history.append(DriftResult(**item))
        except (_json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        import json as _json

        data = {
            "results": [r.to_dict() for r in self._drift_history],
        }
        with open(self.storage_path, "w") as f:
            _json.dump(data, f, indent=2)

    def get_history(self) -> List[DriftResult]:
        """Get all drift detection results."""
        return self._drift_history

    def get_recent_drift(self, limit: int = 10) -> List[DriftResult]:
        """Get recent drift detection results."""
        return self._drift_history[-limit:]
