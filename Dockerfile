FROM ubuntu:26.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ryzen AI SDK dependencies
WORKDIR /opt
COPY ryzen-ai-venv /media/scott/data/ryzen-ai-venv

# Set environment
ENV TMPDIR=/media/scott/data/tmp
ENV LD_LIBRARY_PATH="/media/scott/data/ryzen-ai-venv/lib/python3.12/site-packages/flexml/flexml_extras/lib:/media/scott/data/ryzen-ai-venv/lib/python3.12/site-packages/voe/lib:/opt/xilinx/xrt/lib"
ENV PYTHONPATH=/media/scott/data/lfm3-npu-tool/lib

# Copy tooling
COPY . /media/scott/data/lfm3-npu-tool/

# Install Python dependencies
RUN pip3 install fastapi uvicorn onnxruntime onnxruntime-genai tokenizers numpy pydantic psutil

# Expose ports
EXPOSE 1238 1239

# Default command: start CPU server
CMD ["/media/scott/data/ryzen-ai-venv/bin/python", "/media/scott/data/lfm3-npu-tool/lib/lfm3_npu/cpu_server.py", "--host", "0.0.0.0", "--port", "1238"]