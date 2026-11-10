# LFM3-1.7B NPU Inference Tooling

Complete tooling suite for running LFM3-1.7B inference on the AMD XDNA2 NPU
using the Ryzen AI 1.8.0 SDK with VitisAI ExecutionProvider.

## Important Note on NPU Quality

LFM3's depthwise Conv layers trigger a crash in AMD's proprietary
`libvaiml.so::vaiml_get_capability_v5` during VitisAI EP `GetCapability()`.
The crash occurs **before any inference** and cannot be worked around from Python.

**Workaround:** Conv layers are replaced with simplified operations:
- **Center-tap Mul** (recommended): `output[c,t] = input[c,t] × w[c,0,3]`
  Preserves center weight; better quality but still repetitive.
- **Identity bypass**: Conv replaced entirely with Identity.
  Fastest, lowest quality (very repetitive).

**Full-quality NPU inference requires:**
- AMD to fix `libvaiml.so::vaiml_get_capability_v5`
- LiquidAI to provide VitisAI-precompiled LFM3 NPU artifacts
- FastFlowLM to add LFM3 support

**The CPU server provides full-quality inference** and is the production-ready solution.

## Installation

```bash
cd /media/scott/data/lfm3-npu-tool
./install.sh
```

This installs `lfm3-npu`, `lfm3-npu-server-mgr`, `lfm3-npu-health` to `/usr/local/bin/`
and copies the config to `~/.config/lfm3-npu/config.yaml`.

## Quick Start

### Source the Ryzen AI Environment

```bash
source /media/scott/data/lfm3-npu-tool/etc/lfm3-npu-env.sh
```

### Start the CPU Server (Full Quality, Production)

```bash
# Using the manager
lfm3-npu-server-mgr start cpu

# Check health
lfm3-npu-health cpu

# Or using systemd
sudo systemctl start lfm3-npu-cpu
sudo systemctl enable lfm3-npu-cpu
```

### Start the NPU Server (Experimental, Degraded Quality)

```bash
# Requires root for /dev/accel0 access
sudo lfm3-npu-server-mgr start npu

# Check health
lfm3-npu-health npu

# Or using systemd
sudo systemctl start lfm3-npu-npu
sudo systemctl enable lfm3-npu-npu
```

### Generate Text

```bash
# CPU (full quality)
lfm3-npu generate "The quick brown fox" --max-tokens 30 --temperature 0.7

# NPU (degraded quality)
lfm3-npu --model /media/scott/data/lfm3_conv_centertap.onnx generate "The quick brown fox" --max-tokens 30
```

### Benchmark

```bash
# NPU only
lfm3-npu bench --tokens 50 --iterations 3

# Compare CPU vs NPU
lfm3-npu bench --compare --tokens 50 --iterations 3
```

### Convert a Model

```bash
lfm3-npu convert --model /path/to/model.onnx --output model_npu.onnx --method centertap
```

## Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List models |
| `/v1/model/info` | GET | Model info |
| `/v1/chat/completions` | POST | Chat completion |

### Example API Call

```bash
curl -X POST http://localhost:1238/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 30, "temperature": 0.7}'
```

## Python Client

```python
from lfm3_npu.client import LFM3Client

client = LFM3Client(base_url="http://localhost:1238")
response = client.chat("Hello, how are you?", max_tokens=30)
print(response)
```

## Directory Layout

```
lfm3-npu-tool/
├── bin/
│   ├── lfm3-npu              # Unified CLI
│   ├── lfm3-npu-server-mgr   # Server manager
│   ├── lfm3-npu-health       # Health check
│   └── start_npu_server.sh   # NPU server startup script
├── lib/
│   └── lfm3_npu/
│       ├── __init__.py
│       ├── config.py         # Configuration
│       ├── inference.py      # NPU inference engine
│       ├── server.py         # Server manager
│       ├── benchmark.py      # Benchmarking
│       ├── converter.py      # Model conversion
│       ├── cpu_server.py     # CPU FastAPI server (port 1238)
│       ├── npu_server.py     # NPU FastAPI server (port 1239)
│       └── client.py         # Python REST client
├── etc/
│   ├── lfm3-npu-config.yaml  # Default configuration
│   └── lfm3-npu-env.sh       # Environment setup
├── examples/
│   ├── example_generate.py
│   └── example_client.py
├── tests/
│   └── test_tooling.py       # 5 tests, all passing
├── var/
│   ├── log/                  # Server logs
│   └── run/                  # PID files
├── install.sh                # Installation script
└── README.md                 # This file
```

## Known Limitations

1. **NPU text quality is degraded** - Conv layers are simplified, losing local context mixing
2. **NPU server requires root** for `/dev/accel0` access
3. **Full-quality NPU support requires external help** (AMD, LiquidAI, FastFlowLM)
4. **CPU server** provides full-quality inference without root

## License

Internal use only.