# LFM3-1.7B NPU Inference Tooling

Complete tooling suite for running LFM3-1.7B inference on the AMD XDNA2 NPU.

## Status: NPU Inference Fully Working ✅

**LFM3-1.7B runs fully on the NPU via FastFlowLM v1.0.12+** (released Nov 5, 2026).
All 24 layers + LM head execute on the AMD Strix Halo NPU. No VitisAI EP workaround needed.

| Component | Port | Provider | Quality | Status |
|-----------|------|----------|---------|--------|
| FLM NPU Server | 1239 | liblfm3_npu.so | **Full quality** | Production-ready ✅ |
| VitisAI EP (stub) | 1239 | VitisAIExecutionProvider | Degraded | Deprecated |

## Quick Start

### Install FastFlowLM (v1.0.12+)

```bash
# Download and extract FLM v1.0.12 for Linux
cd /tmp
curl -sL -o fastflowlm_linux.tar.gz \
  https://github.com/ROCm/FastFlowLM/releases/download/v1.0.12/fastflowlm_1.0.12_linux.tar.gz
tar xzf fastflowlm_linux.tar.gz

# Run LFM3-1.7B on NPU (interactive)
export LD_LIBRARY_PATH=/tmp/lib:/opt/xilinx/xrt/lib
export XILINX_XRT=/opt/xilinx/xrt
/tmp/flm run lfm3:1.7b

# Start NPU server (OpenAI-compatible API on port 1239)
/tmp/flm serve lfm3:1.7b -p 1239 --host 127.0.0.1
```

### CPU Server (Full Quality, Port 1238)

The existing CPU inference server remains available at port 1238:

```bash
cd /media/scott/data/lfm3-npu-server
# Server is already running (~4.7 days uptime)
curl http://localhost:1238/health
```

## NPU Server API Example

```bash
curl -s -X POST http://localhost:1239/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"lfm3:1.7b","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":100}'
```

**Expected output:**
```
"2 + 2 equals **4**."
```
- Prefill: ~69 tok/s
- Decode: ~27 tok/s
- TTFT: ~0.25s

## Performance (LFM3-1.7B on Strix Halo NPU)

| Metric | Value |
|--------|-------|
| Prefill speed | ~69 tok/s |
| Decode speed | ~27 tok/s |
| Time to first token | ~0.25s |
| Model load time | ~2.9s |
| Layers on NPU | All 24 + LM head |
| Quantization | Q4_0 |

## Directory Layout

```
lfm3-npu-tool/
├── bin/                          # CLI tools
├── lib/lfm3_npu/                 # Python package
├── etc/                          # Config files
├── examples/                     # Example scripts
├── tests/                        # Tooling test suite (5/5 passing)
├── var/                          # Logs and PID files
├── install.sh                    # Installation script
├── README.md                     # This file
├── FINAL_SUMMARY.md              # Overall project summary
├── EXHAUSTIVE_NPU_APPROACHES.md  # All VitisAI EP approaches attempted
├── STUB_VAIML_README.md          # Stub libvaiml.so documentation
├── NPU_ACCESSIBILITY_RESULTS.md  # NPU hardware accessibility proof
├── VAIP_NPU_PATH.md              # VAIP config bypass investigation
└── VAIML_COMPILE_X2_V2_ATTEMPT.md # libvaiml2.so compilation attempt
```

## Key Files

| File | Purpose |
|------|---------|
| `/tmp/flm` | FLM v1.0.12 binary (LFM3 NPU support) |
| `/tmp/lib/liblfm3_npu.so` | LFM3 NPU engine library |
| `/home/scott/.config/flm/models/LFM3-1.7B-NPU2/` | Downloaded model files |
| `/media/scott/data/lfm3-npu-server/` | CPU server (port 1238) |
| Port 1239 | NPU server (FLM LFM3) |

## Background: VitisAI EP Approaches Exhausted

All 13+ approaches to run LFM3 via VitisAI ExecutionProvider were attempted and documented in `EXHAUSTIVE_NPU_APPROACHES.md`. All were blocked by:

1. **`libvaiml.so` SIGSEGV** — Protobuf crash during `dlopen()` (AMD proprietary, unfixable from user space)
2. **No Strix Halo xclbin** — Existing SDK xclbins target `xilinx_v1_ipu_0_0` (bank 131071), not Strix (banks 0/1)
3. **No Strix Halo platform file** — No `.xpfm`/`.xsa` exists for `v++ --package`
4. **No ONNX→xclbin pipeline** — xcompiler finds 0 ops in LFM3

FastFlowLM v1.0.12 bypasses all of these by using its own `liblfm3_npu.so` engine directly with XRT.

## License

Internal use only.