#!/usr/bin/env python3
"""
LFM3-1.7B CPU Batched Inference Server - Continuous Batching on CPU EP.
Port 1238. Full-quality inference with original Conv layers.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["LFM3_MODEL_PATH"] = "/media/scott/data/lfm3_oga_export/model.onnx"
os.environ["LFM3_PROVIDER"] = "CPUExecutionProvider"
os.environ["LFM3_SERVER_NAME"] = "lfm3-1.7b-cpu"
os.environ["LFM3_PROVIDER_LABEL"] = "CPU"

from lfm3_npu.batched_server import app  # noqa: E402

if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=1238)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
