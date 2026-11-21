"""
A/B Testing framework for LFM3-1.7B inference servers.

Provides:
  * A/B test definitions with traffic splitting
  * Variant management (control vs treatment)
  * Metric collection per variant
  * Statistical significance calculation
  * Test lifecycle: create -> start -> analyze -> complete
  * Integration with model registry for variant routing
  * CLI integration for A/B test management

Usage:
    from lfm3_npu.ab_testing import ABTestManager, ABTest, ABVariant

    manager = ABTestManager(model_registry)
    test = manager.create_test(
        "test-001", "model-2.0", "model-1.0", traffic_split=0.1
    )
    manager.start_test("test-001")
    variant = manager.get_variant("test-001", user_id="user-123")
"""

import enum
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class TestStatus(str, enum.Enum):
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class ABVariant:
    """A variant in an A/B test."""

    variant_id: str
    name: str
    model_version: str
    traffic_weight: float  # 0.0 to 1.0
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "name": self.name,
            "model_version": self.model_version,
            "traffic_weight": self.traffic_weight,
            "description": self.description,
        }


@dataclass
class ABTest:
    """An A/B test definition."""

    test_id: str
    name: str
    description: Optional[str]
    status: TestStatus = TestStatus.DRAFT
    variants: List[ABVariant] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    winner_variant_id: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "variants": [v.to_dict() for v in self.variants],
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "winner_variant_id": self.winner_variant_id,
            "metrics": self.metrics,
        }

    def get_variant_weights(self) -> List[float]:
        """Get normalized traffic weights for all variants."""
        total = sum(v.traffic_weight for v in self.variants)
        if total == 0:
            return [1.0 / len(self.variants)] * len(self.variants)
        return [v.traffic_weight / total for v in self.variants]


@dataclass
class ABTestResult:
    """Results of an A/B test analysis."""

    test_id: str
    variant_id: str
    variant_name: str
    request_count: int
    conversion_count: int
    conversion_rate: float
    avg_latency_ms: float
    error_rate: float
    total_tokens: int
    total_cost: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "variant_name": self.variant_name,
            "request_count": self.request_count,
            "conversion_count": self.conversion_count,
            "conversion_rate": round(self.conversion_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "error_rate": round(self.error_rate, 4),
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 6),
        }


class ABTestManager:
    """Manages A/B tests for model comparison."""

    def __init__(
        self,
        model_registry=None,
        storage_path: Optional[str] = None,
    ):
        from lfm3_npu.model_registry import ModelRegistry, get_model_registry

        self.model_registry = model_registry or get_model_registry()
        if storage_path is None:
            storage_path = os.environ.get(
                "LFM3_ABTEST_STORAGE",
                str(Path.home() / ".config" / "lfm3-npu" / "ab_tests.json"),
            )
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._tests: Dict[str, ABTest] = {}
        self._results: Dict[str, Dict[str, ABTestResult]] = {}  # test_id -> variant_id -> result
        self._assignments: Dict[str, str] = {}  # user_id -> variant_id (for consistent routing)
        self._load()

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            import json as _json

            with open(self.storage_path, "r") as f:
                data = _json.load(f)
            for tid, tdata in data.get("tests", {}).items():
                variants = [ABVariant(**v) for v in tdata.get("variants", [])]
                self._tests[tid] = ABTest(
                    test_id=tdata["test_id"],
                    name=tdata["name"],
                    description=tdata.get("description"),
                    status=TestStatus(tdata.get("status", "draft")),
                    variants=variants,
                    created_at=tdata.get("created_at", time.time()),
                    started_at=tdata.get("started_at"),
                    completed_at=tdata.get("completed_at"),
                    winner_variant_id=tdata.get("winner_variant_id"),
                    metrics=tdata.get("metrics", {}),
                )
        except (_json.JSONDecodeError, OSError):
            pass

    def _save(self) -> None:
        import json as _json

        tmp = self.storage_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(
                {
                    "tests": {tid: t.to_dict() for tid, t in self._tests.items()},
                },
                f,
                indent=2,
            )
        tmp.replace(self.storage_path)

    # ------------------------------------------------------------------ #
    # Test management                                                    #
    # ------------------------------------------------------------------ #

    def create_test(
        self,
        test_id: str,
        name: str,
        control_model: str,
        treatment_model: str,
        traffic_split: float = 0.1,
        description: Optional[str] = None,
    ) -> ABTest:
        """Create a new A/B test with control and treatment variants."""
        if test_id in self._tests:
            raise ValueError(f"Test {test_id} already exists")

        variants = [
            ABVariant(
                variant_id="control",
                name="Control",
                model_version=control_model,
                traffic_weight=1.0 - traffic_split,
                description="Baseline model",
            ),
            ABVariant(
                variant_id="treatment",
                name="Treatment",
                model_version=treatment_model,
                traffic_weight=traffic_split,
                description="New model under test",
            ),
        ]

        test = ABTest(
            test_id=test_id,
            name=name,
            description=description,
            status=TestStatus.DRAFT,
            variants=variants,
        )
        self._tests[test_id] = test
        self._save()
        return test

    def start_test(self, test_id: str) -> Optional[ABTest]:
        """Start an A/B test."""
        test = self._tests.get(test_id)
        if test is None:
            return None
        if test.status != TestStatus.DRAFT:
            return None

        test.status = TestStatus.RUNNING
        test.started_at = time.time()
        self._save()
        return test

    def complete_test(
        self,
        test_id: str,
        winner_variant_id: Optional[str] = None,
    ) -> Optional[ABTest]:
        """Complete an A/B test and optionally declare a winner."""
        test = self._tests.get(test_id)
        if test is None:
            return None

        test.status = TestStatus.COMPLETED
        test.completed_at = time.time()
        test.winner_variant_id = winner_variant_id
        self._save()
        return test

    def get_test(self, test_id: str) -> Optional[ABTest]:
        return self._tests.get(test_id)

    def list_tests(self) -> List[ABTest]:
        return sorted(self._tests.values(), key=lambda t: t.created_at, reverse=True)

    def delete_test(self, test_id: str) -> bool:
        if test_id in self._tests:
            del self._tests[test_id]
            if test_id in self._results:
                del self._results[test_id]
            self._save()
            return True
        return False

    # ------------------------------------------------------------------ #
    # Variant routing                                                    #
    # ------------------------------------------------------------------ #

    def get_variant_for_user(
        self, test_id: str, user_id: str
    ) -> Optional[ABVariant]:
        """Get the variant for a user (consistent routing via hash)."""
        test = self._tests.get(test_id)
        if test is None or test.status != TestStatus.RUNNING:
            return None

        # Consistent hash-based routing
        hash_input = f"{test_id}:{user_id}"
        hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
        normalized = (hash_value % 10000) / 10000.0  # 0.0 to 1.0

        weights = test.get_variant_weights()
        cumulative = 0.0
        for variant, weight in zip(test.variants, weights):
            cumulative += weight
            if normalized < cumulative:
                return variant

        # Fallback to last variant
        return test.variants[-1] if test.variants else None

    def get_variant_for_request(
        self, test_id: str, request_id: Optional[str] = None
    ) -> Optional[ABVariant]:
        """Get the variant for a request (uses request_id or random)."""
        user_id = request_id or f"anon-{time.time()}"
        return self.get_variant_for_user(test_id, user_id)

    # ------------------------------------------------------------------ #
    # Metrics recording                                                  #
    # ------------------------------------------------------------------ #

    def record_variant_metric(
        self,
        test_id: str,
        variant_id: str,
        request_count: int = 1,
        conversion_count: int = 0,
        latency_ms: float = 0.0,
        error_count: int = 0,
        tokens: int = 0,
        cost: float = 0.0,
    ) -> None:
        """Record metrics for a variant in a test."""
        if test_id not in self._results:
            self._results[test_id] = {}

        if variant_id not in self._results[test_id]:
            self._results[test_id][variant_id] = ABTestResult(
                test_id=test_id,
                variant_id=variant_id,
                variant_name="",
                request_count=0,
                conversion_count=0,
                conversion_rate=0.0,
                avg_latency_ms=0.0,
                error_rate=0.0,
                total_tokens=0,
                total_cost=0.0,
            )

        result = self._results[test_id][variant_id]
        result.request_count += request_count
        result.conversion_count += conversion_count
        result.total_tokens += tokens
        result.total_cost += cost

        # Recalculate rates
        if result.request_count > 0:
            result.conversion_rate = result.conversion_count / result.request_count
            result.error_rate = error_count / result.request_count

        # Update avg latency (simple running average)
        if result.request_count == request_count:
            result.avg_latency_ms = latency_ms
        else:
            result.avg_latency_ms = (
                (result.avg_latency_ms * (result.request_count - request_count))
                + latency_ms * request_count
            ) / result.request_count

        # Set variant name from test
        test = self._tests.get(test_id)
        if test:
            for v in test.variants:
                if v.variant_id == variant_id:
                    result.variant_name = v.name
                    break

    # ------------------------------------------------------------------ #
    # Analysis                                                           #
    # ------------------------------------------------------------------ #

    def get_results(self, test_id: str) -> List[ABTestResult]:
        """Get results for all variants in a test."""
        if test_id not in self._results:
            return []
        return list(self._results[test_id].values())

    def analyze_test(self, test_id: str) -> Dict[str, Any]:
        """Analyze an A/B test and determine statistical significance."""
        test = self._tests.get(test_id)
        if test is None:
            return {"error": f"Test {test_id} not found"}

        results = self.get_results(test_id)
        if len(results) < 2:
            return {"error": "Need at least 2 variants for analysis"}

        control = next((r for r in results if r.variant_id == "control"), None)
        treatment = next((r for r in results if r.variant_id == "treatment"), None)

        if not control or not treatment:
            return {"error": "Need both control and treatment variants"}

        # Simple lift calculation
        conversion_lift = 0.0
        if control.conversion_rate > 0:
            conversion_lift = (
                (treatment.conversion_rate - control.conversion_rate)
                / control.conversion_rate
                * 100
            )

        latency_change = 0.0
        if control.avg_latency_ms > 0:
            latency_change = (
                (treatment.avg_latency_ms - control.avg_latency_ms)
                / control.avg_latency_ms
                * 100
            )

        cost_change = 0.0
        if control.total_cost > 0:
            cost_change = (
                (treatment.total_cost - control.total_cost)
                / control.total_cost
                * 100
            )

        # Simple significance check (based on sample size)
        min_samples = 100
        significant = (
            control.request_count >= min_samples
            and treatment.request_count >= min_samples
        )

        return {
            "test_id": test_id,
            "test_name": test.name,
            "status": test.status.value,
            "control": control.to_dict(),
            "treatment": treatment.to_dict(),
            "conversion_lift_percent": round(conversion_lift, 2),
            "latency_change_percent": round(latency_change, 2),
            "cost_change_percent": round(cost_change, 2),
            "statistically_significant": significant,
            "min_samples_required": min_samples,
            "recommendation": self._make_recommendation(
                conversion_lift, latency_change, significant
            ),
        }

    def _make_recommendation(
        self, conversion_lift: float, latency_change: float, significant: bool
    ) -> str:
        if not significant:
            return "Insufficient data - continue running test"
        if conversion_lift > 5 and latency_change < 20:
            return "Promote treatment - better conversion with acceptable latency"
        if conversion_lift < -5:
            return "Reject treatment - worse conversion"
        if latency_change > 50:
            return "Reject treatment - significantly higher latency"
        return "Inconclusive - consider extending test duration"

    # ------------------------------------------------------------------ #
    # Winner promotion                                                   #
    # ------------------------------------------------------------------ #

    def promote_winner(self, test_id: str) -> Optional[str]:
        """Promote the winning variant to production."""
        analysis = self.analyze_test(test_id)
        if "error" in analysis:
            return None

        treatment = analysis["treatment"]
        recommendation = analysis["recommendation"]

        if "Promote treatment" in recommendation:
            model_version = treatment.get("variant_name", "")
            # Find the model version from the variant
            test = self._tests.get(test_id)
            if test:
                for v in test.variants:
                    if v.variant_id == "treatment":
                        model_version = v.model_version
                        break
                test.winner_variant_id = "treatment"
                self._save()
                return model_version
        return None


# --------------------------------------------------------------------------- #
# Global instance                                                             #
# --------------------------------------------------------------------------- #

_ab_test_manager: Optional[ABTestManager] = None


def get_ab_test_manager() -> ABTestManager:
    global _ab_test_manager
    if _ab_test_manager is None:
        _ab_test_manager = ABTestManager()
    return _ab_test_manager


# --------------------------------------------------------------------------- #
# CLI entry point                                                             #
# --------------------------------------------------------------------------- #

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LFM3-1.7B A/B Testing Manager")
    subparsers = parser.add_subparsers(dest="command")

    # create command
    create_parser = subparsers.add_parser("create", help="Create an A/B test")
    create_parser.add_argument("test_id", help="Test ID")
    create_parser.add_argument("name", help="Test name")
    create_parser.add_argument("control_model", help="Control model version")
    create_parser.add_argument("treatment_model", help="Treatment model version")
    create_parser.add_argument("--traffic-split", type=float, default=0.1, help="Traffic split for treatment (0-1)")
    create_parser.add_argument("--description", default=None, help="Test description")
    create_parser.set_defaults(func=cmd_create)

    # start command
    start_parser = subparsers.add_parser("start", help="Start an A/B test")
    start_parser.add_argument("test_id", help="Test ID")
    start_parser.set_defaults(func=cmd_start)

    # complete command
    complete_parser = subparsers.add_parser("complete", help="Complete an A/B test")
    complete_parser.add_argument("test_id", help="Test ID")
    complete_parser.add_argument("--winner", default=None, help="Winning variant ID")
    complete_parser.set_defaults(func=cmd_complete)

    # list command
    list_parser = subparsers.add_parser("list", help="List A/B tests")
    list_parser.set_defaults(func=cmd_list)

    # get command
    get_parser = subparsers.add_parser("get", help="Get test details")
    get_parser.add_argument("test_id", help="Test ID")
    get_parser.set_defaults(func=cmd_get)

    # variant command
    variant_parser = subparsers.add_parser("variant", help="Get variant for user/request")
    variant_parser.add_argument("test_id", help="Test ID")
    variant_parser.add_argument("user_id", help="User ID or request ID")
    variant_parser.set_defaults(func=cmd_variant)

    # results command
    results_parser = subparsers.add_parser("results", help="Get test results")
    results_parser.add_argument("test_id", help="Test ID")
    results_parser.set_defaults(func=cmd_results)

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze test results")
    analyze_parser.add_argument("test_id", help="Test ID")
    analyze_parser.set_defaults(func=cmd_analyze)

    # record command
    record_parser = subparsers.add_parser("record", help="Record variant metrics")
    record_parser.add_argument("test_id", help="Test ID")
    record_parser.add_argument("variant_id", help="Variant ID")
    record_parser.add_argument("--requests", type=int, default=1, help="Request count")
    record_parser.add_argument("--conversions", type=int, default=0, help="Conversion count")
    record_parser.add_argument("--latency", type=float, default=0.0, help="Latency ms")
    record_parser.add_argument("--errors", type=int, default=0, help="Error count")
    record_parser.add_argument("--tokens", type=int, default=0, help="Token count")
    record_parser.add_argument("--cost", type=float, default=0.0, help="Cost")
    record_parser.set_defaults(func=cmd_record)

    # promote command
    promote_parser = subparsers.add_parser("promote", help="Promote test winner")
    promote_parser.add_argument("test_id", help="Test ID")
    promote_parser.set_defaults(func=cmd_promote)

    args = parser.parse_args()
    manager = get_ab_test_manager()

    if args.command == "create":
        try:
            test = manager.create_test(
                args.test_id,
                args.name,
                args.control_model,
                args.treatment_model,
                args.traffic_split,
                args.description,
            )
            print(f"A/B test created: {test.test_id}")
            print(f"  Control: {test.variants[0].model_version} ({test.variants[0].traffic_weight*100:.0f}%)")
            print(f"  Treatment: {test.variants[1].model_version} ({test.variants[1].traffic_weight*100:.0f}%)")
        except ValueError as e:
            print(f"Error: {e}")
            return 1
        return 0

    elif args.command == "start":
        test = manager.start_test(args.test_id)
        if test is None:
            print(f"Test {args.test_id} not found or not in draft status")
            return 1
        print(f"A/B test {args.test_id} started")
        return 0

    elif args.command == "complete":
        test = manager.complete_test(args.test_id, args.winner)
        if test is None:
            print(f"Test {args.test_id} not found")
            return 1
        print(f"A/B test {args.test_id} completed")
        if args.winner:
            print(f"  Winner: {args.winner}")
        return 0

    elif args.command == "list":
        tests = manager.list_tests()
        if not tests:
            print("No A/B tests found")
            return 0
        print(f"A/B tests ({len(tests)}):")
        for t in tests:
            print(f"  {t.test_id}: {t.name} [{t.status.value}]")
        return 0

    elif args.command == "get":
        test = manager.get_test(args.test_id)
        if test is None:
            print(f"Test {args.test_id} not found")
            return 1
        print(f"Test: {test.test_id}")
        print(f"  Name: {test.name}")
        print(f"  Status: {test.status.value}")
        print(f"  Description: {test.description or 'N/A'}")
        print(f"  Variants:")
        for v in test.variants:
            print(f"    {v.variant_id}: {v.model_version} ({v.traffic_weight*100:.0f}%)")
        return 0

    elif args.command == "variant":
        variant = manager.get_variant_for_user(args.test_id, args.user_id)
        if variant is None:
            print(f"No variant found for {args.user_id} in test {args.test_id}")
            return 1
        print(f"Variant: {variant.variant_id} ({variant.name})")
        print(f"  Model: {variant.model_version}")
        return 0

    elif args.command == "results":
        results = manager.get_results(args.test_id)
        if not results:
            print(f"No results for test {args.test_id}")
            return 0
        print(f"Results for test {args.test_id}:")
        for r in results:
            print(f"  {r.variant_id} ({r.variant_name}):")
            print(f"    Requests: {r.request_count}")
            print(f"    Conversions: {r.conversion_count} ({r.conversion_rate:.2%})")
            print(f"    Avg latency: {r.avg_latency_ms:.0f}ms")
            print(f"    Error rate: {r.error_rate:.1%}")
            print(f"    Total cost: {r.total_cost:.6f}")
        return 0

    elif args.command == "analyze":
        analysis = manager.analyze_test(args.test_id)
        if "error" in analysis:
            print(f"Error: {analysis['error']}")
            return 1
        print(f"A/B Test Analysis: {analysis['test_name']}")
        print(f"  Status: {analysis['status']}")
        print(f"  Control conversion: {analysis['control']['conversion_rate']:.2%}")
        print(f"  Treatment conversion: {analysis['treatment']['conversion_rate']:.2%}")
        print(f"  Conversion lift: {analysis['conversion_lift_percent']:+.2f}%")
        print(f"  Latency change: {analysis['latency_change_percent']:+.2f}%")
        print(f"  Cost change: {analysis['cost_change_percent']:+.2f}%")
        print(f"  Statistically significant: {analysis['statistically_significant']}")
        print(f"  Recommendation: {analysis['recommendation']}")
        return 0

    elif args.command == "record":
        manager.record_variant_metric(
            args.test_id,
            args.variant_id,
            request_count=args.requests,
            conversion_count=args.conversions,
            latency_ms=args.latency,
            error_count=args.errors,
            tokens=args.tokens,
            cost=args.cost,
        )
        print(f"Recorded metrics for {args.variant_id} in {args.test_id}")
        return 0

    elif args.command == "promote":
        model_version = manager.promote_winner(args.test_id)
        if model_version is None:
            print(f"Cannot promote winner for {args.test_id}")
            return 1
        print(f"Promoted winner model: {model_version}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
