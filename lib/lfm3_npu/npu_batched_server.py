#!/usr/bin/env python3
"""
LFM3-1.7B NPU Batched Inference Server - Continuous Batching on VitisAI EP.
Port 1239. Uses center-tap Conv replacement model.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["LFM3_MODEL_PATH"] = "/media/scott/data/lfm3_conv_centertap.onnx"
os.environ["LFM3_PROVIDER"] = "VitisAIExecutionProvider"
os.environ["LFM3_CUSTOM_OPS_PATH"] = "/media/scott/data/ryzen-ai-venv/lib/python3.12/site-packages/voe/lib/libonnxruntime_vitis_ai_custom_ops.so"
os.environ["LFM3_SERVER_NAME"] = "lfm3-1.7b-npu"
os.environ["LFM3_PROVIDER_LABEL"] = "NPU"

from lfm3_npu.batched_server import app  # noqa: E402

if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=1239)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
