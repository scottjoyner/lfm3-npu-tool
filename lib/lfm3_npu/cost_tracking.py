"""
Cost tracking and billing for LFM3-1.7B inference servers.

Provides:
  * Per-token pricing models (input/output token rates)
  * Per-tenant cost accumulation from token usage
  * Cost tier definitions (free/standard/enterprise with different rates)
  * Billing period tracking (monthly reset)
  * Cost reports and invoices (JSON/text)
  * Integration with multi-tenancy token usage tracking
  * CLI integration for cost queries and invoice generation

Usage:
    from lfm3_npu.cost_tracking import CostTracker, PricingTier

    tracker = CostTracker()
    cost = tracker.calculate_cost(input_tokens=1000, output_tokens=500, tier="standard")
    tracker.record_tenant_cost("acme-corp", input_tokens=1000, output_tokens=500)

CLI:
    lfm3-npu cost calculate --input-tokens 1000 --output-tokens 500 --tier standard
    lfm3-npu cost report --tenant acme-corp --period 2026-11
    lfm3-npu cost invoice --tenant acme-corp --period 2026-11
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default pricing tiers (cost per 1K tokens)
DEFAULT_PRICING_TIERS: Dict[str, Dict[str, float]] = {
    "free": {
        "input_per_1k": 0.000,
        "output_per_1k": 0.000,
    },
    "standard": {
        "input_per_1k": 0.0015,   # $0.0015 per 1K input tokens
        "output_per_1k": 0.0060,  # $0.0060 per 1K output tokens
    },
    "enterprise": {
        "input_per_1k": 0.0010,   # Discounted enterprise rate
        "output_per_1k": 0.0040,
    },
}

DEFAULT_CURRENCY = "USD"


@dataclass
class PricingTier:
    """Pricing tier definition."""

    tier_name: str
    input_per_1k: float
    output_per_1k: float

    def calculate_input_cost(self, tokens: int) -> float:
        """Calculate cost for input tokens."""
        return (tokens / 1000.0) * self.input_per_1k

    def calculate_output_cost(self, tokens: int) -> float:
        """Calculate cost for output tokens."""
        return (tokens / 1000.0) * self.output_per_1k

    def calculate_total_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate total cost for input and output tokens."""
        return self.calculate_input_cost(input_tokens) + self.calculate_output_cost(output_tokens)


@dataclass
class CostRecord:
    """A single cost record for a tenant."""

    tenant_id: str
    period: str  # YYYY-MM
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    request_count: int = 0
    updated_at: float = field(default_factory=time.time)

    def add_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        input_cost: float,
        output_cost: float,
    ) -> None:
        """Add usage to this record."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
        self.input_cost += input_cost
        self.output_cost += output_cost
        self.total_cost += input_cost + output_cost
        self.request_count += 1
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "period": self.period,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_cost": round(self.input_cost, 6),
            "output_cost": round(self.output_cost, 6),
            "total_cost": round(self.total_cost, 6),
            "request_count": self.request_count,
            "updated_at": self.updated_at,
        }


class CostTracker:
    """Tracks costs per tenant and generates billing reports."""

    def __init__(
        self,
        pricing_tiers: Optional[Dict[str, Dict[str, float]]] = None,
        currency: str = DEFAULT_CURRENCY,
        storage_path: Optional[str] = None,
    ):
        self.pricing_tiers: Dict[str, PricingTier] = {}
        for tier_name, rates in (pricing_tiers or DEFAULT_PRICING_TIERS).items():
            self.pricing_tiers[tier_name] = PricingTier(
                tier_name=tier_name,
                input_per_1k=rates["input_per_1k"],
                output_per_1k=rates["output_per_1k"],
            )
        self.currency = currency
        if storage_path is None:
            storage_path = os.environ.get(
                "LFM3_COST_STORAGE",
                str(Path.home() / ".config" / "lfm3-npu" / "costs.json"),
            )
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._cost_records: Dict[str, CostRecord] = {}  # key: "tenant_id:period"
        self._load()

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        """Load cost records from disk."""
        if not self.storage_path.exists():
            return
        try:
            import json as _json

            with open(self.storage_path, "r") as f:
                data = _json.load(f)
            for key, record_data in data.items():
                self._cost_records[key] = CostRecord(**record_data)
        except (_json.JSONDecodeError, OSError) as e:
            print(f"[Cost] Warning: could not load cost records: {e}", flush=True)

    def _save(self) -> None:
        """Save cost records to disk."""
        import json as _json

        tmp = self.storage_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({k: r.to_dict() for k, r in self._cost_records.items()}, f, indent=2)
        tmp.replace(self.storage_path)

    def _record_key(self, tenant_id: str, period: str) -> str:
        """Generate storage key for a tenant-period record."""
        return f"{tenant_id}:{period}"

    def _current_period(self) -> str:
        """Return current billing period as YYYY-MM."""
        return time.strftime("%Y-%m")

    # ------------------------------------------------------------------ #
    # Cost calculation                                                   #
    # ------------------------------------------------------------------ #

    def get_pricing_tier(self, tier_name: str) -> Optional[PricingTier]:
        """Get a pricing tier by name."""
        return self.pricing_tiers.get(tier_name)

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        tier_name: str = "standard",
    ) -> Dict[str, float]:
        """Calculate cost for given token counts and tier."""
        tier = self.get_pricing_tier(tier_name)
        if tier is None:
            raise ValueError(f"Unknown pricing tier: {tier_name}")
        input_cost = tier.calculate_input_cost(input_tokens)
        output_cost = tier.calculate_output_cost(output_tokens)
        return {
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(input_cost + output_cost, 6),
            "currency": self.currency,
        }

    # ------------------------------------------------------------------ #
    # Cost recording                                                     #
    # ------------------------------------------------------------------ #

    def record_tenant_cost(
        self,
        tenant_id: str,
        input_tokens: int,
        output_tokens: int,
        tier_name: str = "standard",
        period: Optional[str] = None,
    ) -> CostRecord:
        """Record cost for a tenant and return the updated record."""
        if period is None:
            period = self._current_period()

        key = self._record_key(tenant_id, period)
        tier = self.get_pricing_tier(tier_name)
        if tier is None:
            raise ValueError(f"Unknown pricing tier: {tier_name}")

        input_cost = tier.calculate_input_cost(input_tokens)
        output_cost = tier.calculate_output_cost(output_tokens)

        if key in self._cost_records:
            record = self._cost_records[key]
            record.add_usage(input_tokens, output_tokens, input_cost, output_cost)
        else:
            record = CostRecord(
                tenant_id=tenant_id,
                period=period,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=input_cost + output_cost,
                request_count=1,
            )
            self._cost_records[key] = record

        self._save()
        return record

    # ------------------------------------------------------------------ #
    # Reporting                                                          #
    # ------------------------------------------------------------------ #

    def get_tenant_cost(
        self, tenant_id: str, period: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cost record for a tenant in a period."""
        if period is None:
            period = self._current_period()
        key = self._record_key(tenant_id, period)
        record = self._cost_records.get(key)
        if record is None:
            return None
        return record.to_dict()

    def get_all_costs(self, period: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all cost records, optionally filtered by period."""
        if period is None:
            period = self._current_period()
        results = []
        for key, record in self._cost_records.items():
            if record.period == period:
                results.append(record.to_dict())
        # Sort by total cost descending
        results.sort(key=lambda r: r["total_cost"], reverse=True)
        return results

    def get_tenant_summary(
        self, tenant_id: str
    ) -> Dict[str, Any]:
        """Get cost summary for a tenant across all periods."""
        tenant_records = [
            r for k, r in self._cost_records.items() if r.tenant_id == tenant_id
        ]
        if not tenant_records:
            return {
                "tenant_id": tenant_id,
                "total_cost": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_requests": 0,
                "periods": [],
            }

        total_cost = sum(r.total_cost for r in tenant_records)
        total_input = sum(r.input_tokens for r in tenant_records)
        total_output = sum(r.output_tokens for r in tenant_records)
        total_requests = sum(r.request_count for r in tenant_records)
        periods = [
            {
                "period": r.period,
                "total_cost": round(r.total_cost, 6),
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "request_count": r.request_count,
            }
            for r in sorted(tenant_records, key=lambda r: r.period)
        ]

        return {
            "tenant_id": tenant_id,
            "total_cost": round(total_cost, 6),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_requests": total_requests,
            "currency": self.currency,
            "periods": periods,
        }

    def generate_invoice(
        self, tenant_id: str, period: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate an invoice for a tenant in a period."""
        if period is None:
            period = self._current_period()

        record = self.get_tenant_cost(tenant_id, period)
        if record is None:
            return {
                "tenant_id": tenant_id,
                "period": period,
                "status": "no_usage",
                "total_cost": 0.0,
                "currency": self.currency,
                "line_items": [],
            }

        return {
            "tenant_id": tenant_id,
            "period": period,
            "status": "issued",
            "currency": self.currency,
            "input_tokens": record["input_tokens"],
            "output_tokens": record["output_tokens"],
            "input_cost": record["input_cost"],
            "output_cost": record["output_cost"],
            "total_cost": record["total_cost"],
            "request_count": record["request_count"],
            "generated_at": time.time(),
        }


# --------------------------------------------------------------------------- #
# Global instance                                                             #
# --------------------------------------------------------------------------- #

_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker


# --------------------------------------------------------------------------- #
# CLI entry point                                                             #
# --------------------------------------------------------------------------- #

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LFM3-1.7B Cost Tracking")
    subparsers = parser.add_subparsers(dest="command")

    # calculate command
    calc_parser = subparsers.add_parser("calculate", help="Calculate cost for token counts")
    calc_parser.add_argument("--input-tokens", type=int, required=True, help="Input token count")
    calc_parser.add_argument("--output-tokens", type=int, required=True, help="Output token count")
    calc_parser.add_argument("--tier", default="standard", choices=["free", "standard", "enterprise"], help="Pricing tier")
    calc_parser.add_argument("--currency", default=DEFAULT_CURRENCY, help="Currency code")
    calc_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    # record command
    record_parser = subparsers.add_parser("record", help="Record tenant cost")
    record_parser.add_argument("--tenant-id", required=True, help="Tenant ID")
    record_parser.add_argument("--input-tokens", type=int, required=True, help="Input token count")
    record_parser.add_argument("--output-tokens", type=int, required=True, help="Output token count")
    record_parser.add_argument("--tier", default="standard", help="Pricing tier")
    record_parser.add_argument("--period", default=None, help="Billing period YYYY-MM")

    # report command
    report_parser = subparsers.add_parser("report", help="Show cost report")
    report_parser.add_argument("--tenant-id", default=None, help="Tenant ID (all if omitted)")
    report_parser.add_argument("--period", default=None, help="Billing period YYYY-MM")
    report_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    # invoice command
    invoice_parser = subparsers.add_parser("invoice", help="Generate invoice")
    invoice_parser.add_argument("--tenant-id", required=True, help="Tenant ID")
    invoice_parser.add_argument("--period", default=None, help="Billing period YYYY-MM")
    invoice_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    args = parser.parse_args()
    tracker = CostTracker()

    if args.command == "calculate":
        cost = tracker.calculate_cost(
            args.input_tokens, args.output_tokens, args.tier
        )
        if args.format == "json":
            print(json.dumps(cost, indent=2))
        else:
            print(f"Cost calculation ({args.tier} tier, {args.currency}):")
            print(f"  Input tokens:  {args.input_tokens:,}  Cost: {cost['input_cost']:.6f}")
            print(f"  Output tokens: {args.output_tokens:,}  Cost: {cost['output_cost']:.6f}")
            print(f"  Total cost:    {cost['total_cost']:.6f} {args.currency}")
        return 0

    elif args.command == "record":
        record = tracker.record_tenant_cost(
            args.tenant_id,
            args.input_tokens,
            args.output_tokens,
            args.tier,
            args.period,
        )
        print(f"Cost recorded for {args.tenant_id} ({args.period or 'current period'}):")
        print(f"  Input tokens:  {record.input_tokens:,}")
        print(f"  Output tokens: {record.output_tokens:,}")
        print(f"  Total cost:    {record.total_cost:.6f} {tracker.currency}")
        return 0

    elif args.command == "report":
        if args.tenant_id:
            summary = tracker.get_tenant_summary(args.tenant_id)
            if args.format == "json":
                print(json.dumps(summary, indent=2))
            else:
                print(f"Cost summary for {args.tenant_id}:")
                print(f"  Total cost:       {summary['total_cost']:.6f} {summary['currency']}")
                print(f"  Input tokens:     {summary['total_input_tokens']:,}")
                print(f"  Output tokens:    {summary['total_output_tokens']:,}")
                print(f"  Total requests:   {summary['total_requests']:,}")
                print(f"  Periods: {len(summary['periods'])}")
                for p in summary["periods"]:
                    print(f"    {p['period']}: {p['total_cost']:.6f} ({p['request_count']} requests)")
        else:
            all_costs = tracker.get_all_costs(args.period)
            if args.format == "json":
                print(json.dumps(all_costs, indent=2))
            else:
                period_label = args.period or "current period"
                print(f"Cost report ({period_label}, {len(all_costs)} tenants):")
                print()
                for c in all_costs:
                    print(f"  {c['tenant_id']}: {c['total_cost']:.6f} {tracker.currency} "
                          f"({c['input_tokens']:,} in / {c['output_tokens']:,} out, "
                          f"{c['request_count']} requests)")
        return 0

    elif args.command == "invoice":
        invoice = tracker.generate_invoice(args.tenant_id, args.period)
        if args.format == "json":
            print(json.dumps(invoice, indent=2))
        else:
            print(f"Invoice for {invoice['tenant_id']} - Period: {invoice['period']}")
            print(f"  Status: {invoice['status']}")
            if invoice["status"] == "issued":
                print(f"  Input tokens:  {invoice['input_tokens']:,}  Cost: {invoice['input_cost']:.6f}")
                print(f"  Output tokens: {invoice['output_tokens']:,}  Cost: {invoice['output_cost']:.6f}")
                print(f"  Total cost:    {invoice['total_cost']:.6f} {invoice['currency']}")
                print(f"  Requests:      {invoice['request_count']:,}")
            print(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(invoice['generated_at']))}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
