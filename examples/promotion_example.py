#!/usr/bin/env python3
"""
Example: Model Promotion Pipeline (CI/CD)
Demonstrates staging -> canary -> production promotion with gates and rollback.
"""
import sys
import os
import tempfile

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SERVER_DIR, "lib"))

from lfm3_npu import (
    get_promotion_pipeline, PromotionStage,
)
from lfm3_npu.model_registry import get_model_registry


def main():
    print("=== LFM3-1.7B Model Promotion Pipeline Example ===\n")

    # Use temp storage for the example
    tmpdir = tempfile.mkdtemp()
    storage = os.path.join(tmpdir, "promotion.json")
    pipeline = get_promotion_pipeline.__wrapped__ if hasattr(get_promotion_pipeline, '__wrapped__') else None

    # Create a fresh pipeline instance with temp storage
    from lfm3_npu.model_promotion import PromotionPipeline
    pipeline = PromotionPipeline(storage_path=storage)

    # Register a model so promotion gates pass
    model_registry = get_model_registry()
    model_registry.register_model(
        version="2.0.0", model_id="lfm3-1.7b", status="primary"
    )

    # 1. Request promotion from staging to canary
    print("1. Requesting promotion: 2.0.0 staging -> canary")
    req = pipeline.request_promotion(
        "2.0.0", PromotionStage.STAGING, PromotionStage.CANARY,
        "deploy-bot", notes="Canary release for 2.0.0", auto_approve=True
    )
    print(f"   Request ID: {req.request_id}")
    print(f"   Status: {req.status.value}")
    print(f"   Gate results: {req.gate_results}\n")

    # 2. Deploy the canary
    print("2. Deploying canary promotion")
    deployed = pipeline.deploy_promotion(req.request_id)
    print(f"   Deployed: {deployed.status.value}")
    print(f"   Model stage: {pipeline.get_current_stage('2.0.0').value}\n")

    # 3. Promote canary to production
    print("3. Requesting promotion: 2.0.0 canary -> production")
    req2 = pipeline.request_promotion(
        "2.0.0", PromotionStage.CANARY, PromotionStage.PRODUCTION,
        "deploy-bot", notes="Production release for 2.0.0", auto_approve=True
    )
    print(f"   Request ID: {req2.request_id}")
    print(f"   Status: {req2.status.value}\n")

    # 4. Deploy to production
    print("4. Deploying production promotion")
    deployed2 = pipeline.deploy_promotion(req2.request_id)
    print(f"   Deployed: {deployed2.status.value}")
    print(f"   Model stage: {pipeline.get_current_stage('2.0.0').value}\n")

    # 5. List all promotion requests
    print("5. All promotion requests:")
    for r in pipeline.list_requests(model_version="2.0.0"):
        print(f"   {r.request_id}: {r.from_stage.value} -> {r.to_stage.value} [{r.status.value}]")

    print("\n✓ Model promotion example complete")


if __name__ == "__main__":
    main()
