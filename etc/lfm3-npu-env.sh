#!/bin/bash
# =============================================================================
# LFM3-1.7B NPU Tooling - Environment Setup
# =============================================================================
# Source this file to set up the environment for NPU inference:
#
#   source /etc/lfm3-npu-env.sh
#   or
#   . /path/to/etc/lfm3-npu-env.sh
# =============================================================================

# Ryzen AI SDK Python 3.12 environment
export LFM3_NPU_PYTHON="/media/scott/data/ryzen-ai-venv/bin/python"
export LFM3_NPU_VENV="/media/scott/data/ryzen-ai-venv"

# Temporary directory (avoid /tmp quota issues)
export TMPDIR="/media/scott/data/tmp"

# Library paths for VitisAI EP and XRT
export LD_LIBRARY_PATH="/media/scott/data/ryzen-ai-venv/lib/python3.12/site-packages/flexml/flexml_extras/lib:/media/scott/data/ryzen-ai-venv/lib/python3.12/site-packages/voe/lib:/opt/xilinx/xrt/lib:${LD_LIBRARY_PATH:-}"

# Tooling installation directory
export LFM3_NPU_TOOL_DIR="/media/scott/data/lfm3-npu-tool"
export LFM3_NPU_LIB_DIR="${LFM3_NPU_TOOL_DIR}/lib"
export LFM3_NPU_ETC_DIR="${LFM3_NPU_TOOL_DIR}/etc"
export LFM3_NPU_LOG_DIR="${LFM3_NPU_TOOL_DIR}/var/log"
export LFM3_NPU_RUN_DIR="${LFM3_NPU_TOOL_DIR}/var/run"

# Model paths
export LFM3_NPU_MODEL_ORIGINAL="/media/scott/data/lfm3_oga_export/model.onnx"
export LFM3_NPU_MODEL_NPU="/media/scott/data/lfm3_conv_centertap.onnx"
export LFM3_NPU_MODEL_CPU="/media/scott/data/lfm3_oga_export/model.onnx"
export LFM3_NPU_TOKENIZER="/media/scott/data/lfm3_oga_export/tokenizer.json"

# Server ports
export LFM3_NPU_CPU_PORT="1238"
export LFM3_NPU_NPU_PORT="1239"

# Default generation parameters
export LFM3_NPU_MAX_TOKENS="50"
export LFM3_NPU_TEMPERATURE="0.7"
export LFM3_NPU_TOP_P="0.9"

# Python path for the tooling package
export PYTHONPATH="${LFM3_NPU_LIB_DIR}:${PYTHONPATH:-}"

echo "LFM3-1.7B NPU Tooling environment loaded."
echo "  Python: ${LFM3_NPU_PYTHON}"
echo "  NPU Model: ${LFM3_NPU_MODEL_NPU}"
echo "  CPU Model: ${LFM3_NPU_MODEL_CPU}"
echo "  CPU Port: ${LFM3_NPU_CPU_PORT}"
echo "  NPU Port: ${LFM3_NPU_NPU_PORT}"