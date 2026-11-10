"""
LFM3-1.7B NPU Inference Tooling

A complete tooling suite for running LFM3-1.7B inference on the AMD XDNA2 NPU
using the Ryzen AI 1.8.0 SDK with VitisAI ExecutionProvider.

IMPORTANT NOTE ON NPU QUALITY:
LFM3's depthwise Conv layers trigger a crash in AMD's proprietary
libvaiml.so::vaiml_get_capability_v5 during VitisAI EP GetCapability().
The crash occurs BEFORE any inference and cannot be worked around from Python.

Workaround: Conv layers are replaced with simplified operations:
- Center-tap Mul (recommended): output[c,t] = input[c,t] * w[c,0,3]
  Preserves center weight; better quality but still repetitive.
- Identity bypass: Conv replaced entirely with Identity.
  Fastest, lowest quality (very repetitive).

Full-quality NPU inference requires AMD or LiquidAI intervention.
The CPU server provides full-quality inference.
"""

__version__ = "1.0.0"

from .config import Config, load_config
from .inference import LFM3NPUInference
from .server import ServerManager
from .benchmark import bench_model, bench_concurrency
from .converter import convert_model
from .metrics import MetricsCollector
from .compare import compare
from .watchdog import Watchdog, watchdog_main

__all__ = [
    "__version__",
    "Config",
    "load_config",
    "LFM3NPUInference",
    "ServerManager",
    "bench_model",
    "bench_concurrency",
    "convert_model",
    "MetricsCollector",
    "compare",
    "Watchdog",
    "watchdog_main",
]