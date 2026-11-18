FROM ubuntu:26.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Use an isolated Python environment (Ubuntu 26.04 enforces PEP 668).
ENV VIRTUAL_ENV=/opt/lfm3-venv
RUN python3 -m venv "$VIRTUAL_ENV"

# Create directories
RUN mkdir -p /media/scott/data/tmp /media/scott/data/lfm3-npu-tool \
    /home/scott/.config/flm/models/LFM3-1.7B-NPU2 \
    /home/scott/.config/systemd/user

# Set environment
ENV TMPDIR=/media/scott/data/tmp
ENV LD_LIBRARY_PATH=/tmp/lib:/opt/xilinx/xrt/lib
ENV XILINX_XRT=/opt/xilinx/xrt
ENV FLM_MODEL_PATH=/home/scott/.config/flm/models
ENV PATH=/opt/lfm3-venv/bin:/opt/xilinx/xrt/bin:/usr/bin:/bin
ENV PYTHONPATH=/media/scott/data/lfm3-npu-tool/lib

# Copy tooling
COPY . /media/scott/data/lfm3-npu-tool/

# Install Python dependencies
RUN python -m pip install --no-cache-dir fastapi uvicorn onnxruntime tokenizers numpy pydantic pyyaml psutil

# Copy FLM binary and NPU engine (built separately on host)
# These must be present at /tmp/flm and /tmp/lib/liblfm3_npu.so before building
# See: https://github.com/ROCm/FastFlowLM/releases/tag/v1.0.12

# Expose ports
EXPOSE 1238 1239 9190

# Default command: start all services
CMD ["/media/scott/data/lfm3-npu-tool/start-all.sh", "start"]
