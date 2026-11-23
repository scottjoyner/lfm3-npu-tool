"""
LFM3-1.7B NPU Inference Tooling

A complete tooling suite for running LFM3-1.7B inference on the AMD XDNA2 NPU
alongside CPU, with enterprise-grade deployment, monitoring, optimization, and
CLI tooling.

NPU EXECUTION:
  Full-quality LFM3-1.7B inference on the AMD XDNA2 NPU is provided by
  FastFlowLM v1.0.12 via liblfm3_npu.so. The VitisAI EP approach is
  superseded and documented as historical in EXHAUSTIVE_NPU_APPROACHES.md.

  NPU server:  port 1239  (flm serve lf3:1.7b -p 1239)
  CPU server:  port 1238  (ONNX Runtime)
  Metrics:     port 9190

MODEL FILES:
  /home/scott/.config/flm/models/LFM3-1.7B-NPU2/

ENVIRONMENT:
  LD_LIBRARY_PATH=/tmp/lib:/opt/xilinx/xrt/lib
  XILINX_XRT=/opt/xilinx/xrt
  FLM_MODEL_PATH=/home/scott/.config/flm/models
"""

__version__ = "1.0.0"

# Core inference
from .config import Config, load_config
from .inference import LFM3NPUInference
from .server import ServerManager
from .benchmark import bench_model, bench_concurrency
from .converter import convert_model
from .metrics import MetricsCollector, EnterpriseMetricsCollector
from .compare import compare
from .watchdog import Watchdog, watchdog_main

# Enterprise modules
from .alerting import get_alert_manager
from .api_key_management import get_key_manager
from .audit import get_audit
from .auto_scaler import get_auto_scaler, AutoScaler, ScalingPolicy, ScalingDecision
from .backup_restore import get_backup_manager
from .cost_tracking import get_cost_tracker
from .disaster_recovery import get_dr_manager, DisasterRecoveryManager
from .enterprise import EnterpriseMiddleware
from .feature_flags import get_flags
from .key_rotation import get_rotation_manager
from .load_balancer import LoadBalancer
from .model_promotion import get_promotion_pipeline, PromotionPipeline, PromotionStage
from .model_registry import get_model_registry
from .multi_tenancy import get_tenant_manager
from .rbac import RBACAuth
from .regression_detection import get_regression_detector, RegressionDetector
from .security_scan import SecurityScanner
from .slo import get_tracker, SLOTracker
from .config_schema import validate_config, ConfigSchema
from .tls import tls_ready, https_url
from .tracing import get_tracer
from .validation import validate_prompt
from .ab_testing import get_ab_test_manager, ABTestManager, ABTest

__all__ = [
    "__version__",
    # Core inference
    "Config",
    "load_config",
    "LFM3NPUInference",
    "ServerManager",
    "bench_model",
    "bench_concurrency",
    "convert_model",
    "MetricsCollector",
    "EnterpriseMetricsCollector",
    "compare",
    "Watchdog",
    "watchdog_main",
    # Enterprise modules
    "get_alert_manager",
    "get_key_manager",
    "get_audit",
    "get_auto_scaler",
    "AutoScaler",
    "ScalingPolicy",
    "ScalingDecision",
    "get_backup_manager",
    "get_cost_tracker",
    "get_dr_manager",
    "DisasterRecoveryManager",
    "EnterpriseMiddleware",
    "get_flags",
    "get_rotation_manager",
    "LoadBalancer",
    "get_promotion_pipeline",
    "PromotionPipeline",
    "PromotionStage",
    "get_model_registry",
    "get_tenant_manager",
    "RBACAuth",
    "get_regression_detector",
    "RegressionDetector",
    "SecurityScanner",
    "get_tracker",
    "SLOTracker",
    "tls_ready",
    "https_url",
    "get_tracer",
    "validate_prompt",
    "validate_config",
    "ConfigSchema",
    "get_ab_test_manager",
    "ABTestManager",
    "ABTest",
]