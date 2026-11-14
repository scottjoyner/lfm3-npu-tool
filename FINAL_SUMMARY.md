# LFM3-1.7B NPU Harness — Final Summary

## Overview

This document summarizes the complete LFM3-1.7B inference harness built for AMD XDNA2 NPU alongside CPU, including all optimizations, tooling, benchmarks, and known limitations.

## BREAKTHROUGH: NPU Inference Fully Working via FastFlowLM v1.0.12+ ✅

**As of November 14, 2026, LFM3-1.7B runs fully on the NPU via FastFlowLM v1.0.12.**
All 24 layers + LM head execute on the AMD Strix Halo NPU. No VitisAI EP workaround needed.

| Component | Port | Provider | Quality | Status |
|-----------|------|----------|---------|--------|
| FLM NPU Server | 1239 | liblfm3_npu.so | **Full quality** | Production-ready ✅ |
| VitisAI EP (stub) | 1239 | VitisAIExecutionProvider | Degraded | Deprecated |

### How to Run LFM3 on NPU

```bash
# Download FLM v1.0.12 for Linux
cd /tmp
curl -sL -o fastflowlm_linux.tar.gz \
  https://github.com/ROCm/FastFlowLM/releases/download/v1.0.12/fastflowlm_1.0.12_linux.tar.gz
tar xzf fastflowlm_linux.tar.gz

# Interactive chat on NPU
export LD_LIBRARY_PATH=/tmp/lib:/opt/xilinx/xrt/lib
export XILINX_XRT=/opt/xilinx/xrt
/tmp/flm run lfm3:1.7b

# NPU server (OpenAI-compatible API)
/tmp/flm serve lfm3:1.7b -p 1239 --host 127.0.0.1

# API example
curl -X POST http://localhost:1239/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"lfm3:1.7b","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":100}'
```

### CLI Tooling (bin/lfm3-npu)

```bash
# Generate via NPU server
lfm3-npu generate "What is 2+2?" --server npu --max-tokens 50

# Start/stop servers
lfm3-npu server start npu   # FLM NPU server (port 1239)
lfm3-npu server start cpu   # CPU ONNX server (port 1238)
lfm3-npu server status      # Health check both servers

# Benchmark both servers
lfm3-npu bench --server both --prompt "Hello world" --tokens 50 --iterations 3

# Live side-by-side comparison
lfm3-npu compare-servers --prompts "What is 2+2?" "The capital of France is"

# Watchdog (auto-restart unresponsive servers)
lfm3-npu watchdog --once
lfm3-npu watchdog --daemon --interval 30

# Systemd services (requires sudo)
# etc/lfm3-npu-cpu.service  -> port 1238
# etc/lfm3-npu-npu.service  -> port 1239
```

### Performance (LFM3-1.7B on Strix Halo NPU via FastFlowLM)

| Metric | Value |
|--------|-------|
| Prefill speed | ~69 tok/s |
| Decode speed | ~27-32 tok/s |
| Time to first token | ~0.17-0.43s |
| Model load time | ~2.9s |
| Layers on NPU | All 24 + LM head |
| Quantization | Q4_0 |
| Prefix cache | 8 slots, 4GB budget |

### CPU vs NPU Comparison (Live HTTP API)

| Metric | CPU (ONNX, port 1238) | NPU (FLM, port 1239) | Improvement |
|--------|----------------------|---------------------|-------------|
| TTFT | ~10s (total) | ~0.2s | ~50x faster |
| Decode speed | ~2 tok/s (effective) | ~27-32 tok/s | ~13x faster |
| Total time per request | ~4-10s | ~1.8-2.2s | ~2.5-5x faster |
| Output quality | Degraded after extended uptime | Consistently high | NPU superior |
| Reasoning traces | None | Present (Qwen XML format) | NPU advantage |
| Prefix cache | Not available | 8 slots, 4GB budget | NPU advantage |

## System Configuration

| Component | Details |
|-----------|---------|
| NPU SDK | AMD Ryzen AI 1.8.0 (`/media/scott/data/ryzen-ai-venv/`, Python 3.12) |
| System Python | 3.14 (SDK requires 3.12 from snap core24) |
| CPU Server | Port 1238, CPUExecutionProvider |
| NPU Server | Port 1239, FastFlowLM v1.0.12, liblfm3_npu.so |
| Model | LFM3-1.7B Q4_0 |
| FLM Model Path | `/home/scott/.config/flm/models/LFM3-1.7B-NPU2/` |

## NPU Hardware Status

| Property | Value |
|----------|-------|
| Device | AMD NPU Strix Halo |
| Architecture | aie2p, 6x8 topology |
| BDF | 0000:c4:00.1 |
| Firmware | 1.1.2.65 |
| XRT Version | 2.25.37 |

**NPU hardware IS accessible from user space via pyxrt/XRT:**
- `pyxrt.device(0)` opens the NPU ✅
- `pyxrt.hw_context(device, xclbin_path)` loads xclbins ✅
- Kernels can be found ✅
- Buffer management works with `cacheable` flag ✅

## Historical: VitisAI EP Approaches Exhausted

All 13+ approaches to run LFM3 via ONNX Runtime + VitisAI ExecutionProvider were attempted and documented in `EXHAUSTIVE_NPU_APPROACHES.md`. All were blocked by:

1. **`libvaiml.so` SIGSEGV** — Protobuf crash during `dlopen()` (AMD proprietary, unfixable from user space)
2. **No Strix Halo xclbin** — Existing SDK xclbins target bank 131071, not Strix (banks 0/1)
3. **No Strix Halo platform file** — No `.xpfm`/`.xsa` exists for `v++ --package`
4. **No ONNX→xclbin pipeline** — xcompiler finds 0 ops in LFM3

FastFlowLM v1.0.12 bypasses all of these by using its own `liblfm3_npu.so` engine directly with XRT.

## Complete Tooling Suite (`/media/scott/data/lfm3-npu-tool/`)

- **CLI** (`bin/lfm3-npu`): 12 commands — generate, server, bench, perf, health, convert, config, status, compare, compare-servers, watchdog, fleet-monitor
- **Python package** (`lib/lfm3_npu/`): ServerManager, Watchdog, benchmark, converter, compare, config, metrics
- **Config** (`etc/lfm3-npu-config.yaml`): with performance tuning section
- **Env setup** (`etc/lfm3-npu-env.sh`)
- **Systemd services**: `etc/lfm3-npu-cpu.service`, `etc/lfm3-npu-npu.service`
- **Dockerfile**: Container deployment
- **Install script**: `install.sh`
- **Documentation**: `README.md`, `FINAL_SUMMARY.md`, `BENCHMARK_RESULTS.md`, `EXHAUSTIVE_NPU_APPROACHES.md`
- **Tests**: `tests/test_tooling.py` (5/5 passing)
- **Examples**: `examples/example_generate.py`, `examples/example_client.py`

## Known Limitations

### 1. CPU Server Quality Degradation After Extended Uptime

The CPU server (ONNX Runtime) shows output quality degradation after extended uptime (repetitive output, conversation loops, garbled text). Restarting the server fixes the issue. The NPU server (FastFlowLM) maintains consistent quality over extended uptime.

### 2. VitisAI EP Cannot Run Full-Quality LFM3

The AMD `libvaiml.so` crashes with SIGSEGV in protobuf descriptor registration during `dlopen()`. This requires AMD to fix `libvaiml.so` or provide an LFM3-compiled xclbin artifact for Strix Halo.

## Files

### Key Runtime Files
| File | Purpose |
|------|---------|
| `/tmp/flm` | FLM v1.0.12 binary (LFM3 NPU support) |
| `/tmp/lib/liblfm3_npu.so` | LFM3 NPU engine library |
| `/home/scott/.config/flm/models/LFM3-1.7B-NPU2/` | Downloaded model files |
| `/media/scott/data/lfm3-npu-tool/lib/lfm3_npu/cpu_batched_server.py` | CPU server script (port 1238) |

### Documentation
| File | Purpose |
|------|---------|
| `README.md` | Updated with FLM NPU breakthrough |
| `FINAL_SUMMARY.md` | This file |
| `BENCHMARK_RESULTS.md` | Comprehensive CPU vs NPU benchmark results |
| `EXHAUSTIVE_NPU_APPROACHES.md` | All VitisAI EP approaches documented (historical) |

## Verified Status

- [x] CPU server running on port 1238 (~44k seconds uptime), full-quality when freshly started
- [x] NPU server running on port 1239 (FastFlowLM v1.0.12), full-quality LFM3-1.7B on NPU
- [x] All 24 layers + LM head on NPU
- [x] NPU produces clean output with reasoning traces (~27-32 tok/s decode)
- [x] CPU server quality degrades after extended uptime (restart fixes it)
- [x] CLI tooling: 12 commands, all verified working
- [x] Systemd service files created (requires sudo to install)
- [x] Benchmark results documented in BENCHMARK_RESULTS.md
- [x] Auto-finetune pipeline: 1089 passed, 3 skipped
- [x] All documentation and installation scripts complete

## Bottom Line

**LFM3-1.7B is now running full-quality inference on the AMD XDNA2 NPU** via FastFlowLM v1.0.12, alongside the CPU server. All original project requirements are met.

- **NPU (port 1239)**: Recommended for production — full quality, ~27-32 tok/s, ~0.2s TTFT, prefix cache, reasoning traces
- **CPU (port 1238)**: Available as fallback — full quality when fresh, but degrades after extended uptime

Repository: https://github.com/scottjoyner/lfm3-npu-tool
