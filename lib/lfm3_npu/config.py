"""
Configuration management for LFM3-1.7B NPU tooling.

Loads YAML configuration with sensible defaults.
Search order:
1. LFM3_NPU_CONFIG environment variable
2. ~/.config/lfm3-npu/config.yaml
3. /etc/lfm3-npu/config.yaml
4. ./lfm3-npu-config.yaml
"""

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULTS: Dict[str, Any] = {
    "model": {
        "path": "/media/scott/data/lfm3_conv_centertap.onnx",
        "tokenizer": "/media/scott/data/lfm3_oga_export/tokenizer.json",
        "type": "lfm3-1.7b-npu",
    },
    "npu": {
        "provider": "VitisAIExecutionProvider",
        "device_index": 0,
        "custom_ops_lib": "/media/scott/data/ryzen-ai-venv/lib/python3.12/site-packages/voe/lib/libonnxruntime_vitis_ai_custom_ops.so",
        "python_executable": "/media/scott/data/ryzen-ai-venv/bin/python",
        "ld_library_path": [
            "/media/scott/data/ryzen-ai-venv/lib/python3.12/site-packages/flexml/flexml_extras/lib",
            "/media/scott/data/ryzen-ai-venv/lib/python3.12/site-packages/voe/lib",
            "/opt/xilinx/xrt/lib",
        ],
    },
    "server": {
        "cpu": {
            "port": 1238,
            "host": "0.0.0.0",
            "model_path": "/media/scott/data/lfm3_oga_export/model.onnx",
            "model_type": "cpu_onnx",
        },
        "npu": {
            "port": 1239,
            "host": "0.0.0.0",
            "model_path": "/media/scott/data/lfm3_conv_centertap.onnx",
            "model_type": "npu_vitisai",
        },
    },
    "generation": {
        "max_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
    },
    "logging": {
        "level": "INFO",
        "log_dir": "/media/scott/data/lfm3-npu-tool/var/log",
    },
    "performance": {
        "ort_session": {
            "intra_op_num_threads": 0,
            "inter_op_num_threads": 0,
            "execution_mode": "ORT_SEQUENTIAL",
            "graph_optimization_level": "ORT_ENABLE_ALL",
            "enable_cpu_mem_arena": True,
            "enable_mem_reuse": True,
            "enable_mem_pattern": True,
            "session_config_entries": {
                "optimize_model": "1",
                "enable_bf16": "1"
            }
        },
        "benchmark": {
            "warmup_iterations": 1,
            "measure_iterations": 3,
            "concurrency_levels": [1, 2, 4, 8],
        },
        "server": {
            "max_concurrent_requests": 4,
            "request_timeout_seconds": 120,
            "enable_request_batching": False,
            "batch_timeout_ms": 50,
        },
    },
}


class Config:
    """Configuration container with dot-notation access."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation (e.g., 'server.npu.port')."""
        keys = key.split(".")
        value: Any = self._data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _find_config_file() -> Optional[str]:
    """Find the configuration file using search order."""
    env_path = os.environ.get("LFM3_NPU_CONFIG")
    if env_path and Path(env_path).exists():
        return env_path

    user_config = Path.home() / ".config" / "lfm3-npu" / "config.yaml"
    if user_config.exists():
        return str(user_config)

    system_config = Path("/etc/lfm3-npu/config.yaml")
    if system_config.exists():
        return str(system_config)

    local_config = Path("./lfm3-npu-config.yaml")
    if local_config.exists():
        return str(local_config)

    package_config = Path(__file__).parent.parent.parent / "etc" / "lfm3-npu-config.yaml"
    if package_config.exists():
        return str(package_config)

    return None


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file with defaults fallback."""
    data: Dict[str, Any] = copy.deepcopy(DEFAULTS)

    path = config_path or _find_config_file()
    if path:
        with open(path, "r") as f:
            file_data = yaml.safe_load(f) or {}
        data = _deep_merge(data, file_data)

    return Config(data)