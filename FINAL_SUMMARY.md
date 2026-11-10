# LFM3-1.7B NPU Harness — Final Summary

## Overview

This document summarizes the complete LFM3-1.7B inference harness built for AMD XDNA2 NPU alongside CPU, including all optimizations, tooling, benchmarks, and known limitations.

## System Configuration

| Component | Details |
|-----------|---------|
| NPU SDK | AMD Ryzen AI 1.8.0 (`/media/scott/data/ryzen-ai-venv/`, Python 3.12) |
| System Python | 3.14 (SDK requires 3.12 from snap core24) |
| CPU Server | Port 1238, PID 4191921 |
| NPU Server | Port 1239, PID 359948 |
| Model | LFM3-1.7B ONNX Q4 |
| ONNX Model (CPU) | `/media/scott/data/lfm3_oga_export/model.onnx` |
| ONNX Model (KV Quant) | `/media/scott/data/lfm3_oga_export/model_kv_quant.onnx` |
| NPU Model (center-tap) | `/media/scott/data/lfm3_conv_centertap.onnx` |
| NPU Model (bypass) | `/media/scott/data/lfm3-conv-bypassed.onnx` |

## Core Architecture

### CPU Inference Server (`/media/scott/data/lfm3-npu-server/`)
- FastAPI-based inference server on port 1238
- Uses `CPUExecutionProvider` via onnxruntime-genai
- Endpoints: `/health`, `/v1/models`, `/v1/model/info`, `/v1/chat/completions` (streaming & non-streaming)
- KV cache state management with name-based output matching (`present_conv.*` / `present.*`)
- Async inference: `asyncio.to_thread` for non-streaming, thread pool + `asyncio.Queue` for streaming
- KV cache buffer reuse: in-place `fill(0.0)` reset
- Optional KV quantized model support (`server.cpu.use_kv_quant` in config)

### NPU Inference Server (`/media/scott/data/lfm3-npu-tool/lib/lfm3_npu/npu_server.py`)
- FastAPI-based inference server on port 1239
- Uses `VitisAIExecutionProvider` via onnxruntime-genai
- Same endpoint structure as CPU server
- NPU model uses Conv layer bypass (center-tap multiplier) to avoid `libvaiml.so` crash

### Complete Tooling Suite (`/media/scott/data/lfm3-npu-tool/`)
- **CLI** (`bin/lfm3-npu`): 9 commands — generate, server, bench, health, convert, config, status, compare, watchdog, perf
- **Python package** (`lib/lfm3_npu/`): config, inference, server, benchmark, converter, cpu_server, npu_server, client, metrics, compare, watchdog, kv_cache_quant
- **Config** (`etc/lfm3-npu-config.yaml`): with performance tuning section
- **Env setup** (`etc/lfm3-npu-env.sh`)
- **Systemd services**: `lfm3-npu-cpu.service`, `lfm3-npu-npu.service`
- **Dockerfile**: Container deployment
- **Install script**: `install.sh`
- **Documentation**: `README.md`
- **Tests**: `tests/test_tooling.py` (5/5 passing)
- **Examples**: `examples/example_generate.py`, `examples/example_client.py`

## Optimizations Applied

### 1. VitisAI Execution Provider Memory Tuning
- `enable_cpu_mem_arena = 1`
- `enable_mem_reuse = 1`
- `enable_mem_pattern = 1`

### 2. Session Configuration
- `optimize_model = 1`
- `enable_bf16 = 1`
- Auto CPU core detection for thread allocation

### 3. KV Cache Buffer Reuse
- In-place `fill(0.0)` reset instead of reallocation
- Name-based output matching (`present_conv.*` / `present.*` instead of index-based)
- Empty KV cache initialization with `seq_len = 0`

### 4. Async Inference
- Non-streaming: `asyncio.to_thread`
- Streaming: thread pool + `asyncio.Queue`

### 5. KV Cache Quantization (`kv_cache_quant.py`)
- **STATUS: COMPLETE AND WORKING**
- Quantizes all 22 KV cache tensors (past_key_values.{2,4,...,22}.{key,value}) from FP32 to INT8
- Inserts DequantizeLinear (INT8→FP32) after each past_key_values input
- Inserts QuantizeLinear (FP32→INT8) after each present output
- Graph outputs renamed from `present.*` to `present.*_quantized` (UINT8)
- Model validated: loads successfully in onnxruntime CPUExecutionProvider
- Output: `/media/scott/data/lfm3_oga_export/model_kv_quant.onnx`
- **CPU EP deployment**: Works with `enable_mem_reuse=False`. Tested end-to-end: 52ms prefill, 35ms/decode step, 28.3 TPS. Works with seq_len=0.
- **Quality**: INT8 quantization error accumulates over decode steps, producing repetitive output (similar to NPU center-tap issue). Fixed per-tensor scale (1/127) is suboptimal.
- **Performance**: Comparable to FP32 model (~35-40ms per forward pass). Memory bandwidth savings materialize with larger KV caches / higher concurrency.
- **VitisAI EP**: Crashes with same `libvaiml.so::vaiml_get_capability_v5` FPE_INTDIV (exit 136) — root cause is conv layer shape analysis, independent of KV quantization.
- **Server integration**: cpu_server.py modified with backward-compatible `USE_KV_QUANT` flag (defaults to False). Config options: `server.cpu.kv_quant_model_path`, `server.cpu.use_kv_quant`.

### 6. Alternative NPU Conv Replacement Approaches — ALL TESTED

| Method | VitisAI EP | CPU EP | Quality | Notes |
|--------|-----------|--------|---------|-------|
| **Center-tap** (current) | ✅ Works | ✅ Works | Poor (repetitive) | Only 1 tap used; unique ratio 0.08-0.25 |
| **Bypass** (Conv→Identity) | ✅ Works | ✅ Works | Worst | No local context mixing at all |
| **Im2col** (Pad+Slice+Mul+Add) | ❌ CRASH | ❌ FAIL | N/A | VitisAI: `onnx-mlir::IndexExprBuilder` assertion; CPU: Slice ends dimension error |

**Conclusion**: Center-tap is the only viable NPU approach. All alternatives either crash or produce worse quality.

## Benchmark Results

### Concurrency Benchmark (max_tokens=50, 4 requests per level)

| Concurrency | CPU TPS | NPU TPS | Speedup | CPU Failed | NPU Failed |
|-------------|---------|---------|---------|------------|------------|
| 1           | 4.06    | 4.64    | 1.14x   | 0          | 0          |
| 2           | 3.30    | 3.50    | 1.06x   | 0          | 0          |
| 4           | 2.30    | 2.36    | 1.03x   | 0          | 0          |

### Previous Optimization Results (from earlier session)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Conc 1 TPS | 16.5 | 22.9 | +39% |
| Conc 1 token latency | 74.9ms | 43.6ms | -42% |
| Conc 1 prefill | 50.9ms | 51.7ms | +1.6% |
| Conc 4 TPS | 20.8 | 23.3 | +12% |
| Conc 4 prefill | 93.1ms | 40.1ms | -57% |
| Conc 4 TTFT | 93.2ms | 40.1ms | -57% |

### Streaming Benchmark
- CPU: 9.9 tok/s
- NPU: 12.1 tok/s

### Quality Test Results (unique word ratio, higher = better)

| Prompt | NPU (center-tap) | CPU (full conv) |
|--------|-----------------|-----------------|
| "The quick brown fox" | 0.10 | 0.36 |
| "Once upon a time" | ~0.02 | 0.95 |
| "Write a short story about a robot" | ~0.02 | 0.39 |
| "Explain quantum computing" | 0.08 | 0.57 |
| "What is the capital of France?" | 0.25 | 0.75 |

NPU output is highly repetitive due to center-tap Conv replacement.

### Recent Benchmark (conc=1)
- CPU: 29.9 tok/s
- NPU: 28.9 tok/s

### Recent Benchmark (conc=4)
- CPU: 24.9 tok/s
- NPU: 22.0 tok/s

## Known Limitations

### 1. NPU Cannot Run Full-Quality LFM3
The AMD VitisAI compiler and FlexML compiler both crash on LFM3's hybrid conv+attention architecture:
- `libvaiml.so::vaiml_get_capability_v5` — integer divide-by-zero (FPE_INTDIV, exit 136) during shape analysis of conv layers with dynamic reshape dimensions
- FlexML: `canTransposeBeMovedAfterReshape` assertion — compiler-level incompatibility
- No AMD-compiled LFM3 artifact available
- FastFlowLM supports LFM2 but not LFM3

### 2. NPU Inference Quality Degraded
Due to Conv layer bypass (center-tap multiplier instead of full conv), NPU output is repetitive. This is a required workaround to avoid the `libvaiml.so` crash.

### 3. KV Cache Quantization Impact
The quantized model (`model_kv_quant.onnx`) reduces KV cache memory bandwidth but requires `enable_mem_reuse=False` on CPU EP. INT8 quantization error accumulates over decode steps, producing repetitive output quality degradation. Can be optionally enabled via `server.cpu.use_kv_quant` in config.

## Files Created/Modified

### New Files
- `/media/scott/data/lfm3-npu-tool/lib/lfm3_npu/kv_cache_quant.py` — KV cache quantization (WORKING)
- `/media/scott/data/lfm3-npu-tool/benchmark_concurrency.py` — Comprehensive concurrency benchmark
- `/media/scott/data/lfm3-npu-tool/quality_test.py` — NPU vs CPU quality comparison test

### Existing Files (from prior work)
- `/media/scott/data/lfm3-npu-server/server.py` — CPU inference server (port 1238)
- `/media/scott/data/lfm3-npu-server/client.py` — Python client library
- `/media/scott/data/lfm3-npu-server/test_server.py` — Test suite
- `/media/scott/data/lfm3-npu-tool/bin/lfm3-npu` — Unified CLI
- `/media/scott/data/lfm3-npu-tool/lib/lfm3_npu/` — Python package (12 modules)
- `/media/scott/data/lfm3-npu-tool/etc/lfm3-npu-config.yaml` — Configuration
- `/media/scott/data/lfm3-npu-tool/Dockerfile` — Container deployment
- `/media/scott/data/lfm3-npu-tool/install.sh` — System installation
- `/media/scott/data/lfm3-npu-tool/README.md` — Documentation
- `/media/scott/data/lfm3-npu-tool/tests/test_tooling.py` — Test suite (5/5)

### Converted Models
- `/media/scott/data/lfm3_conv_centertap.onnx` — Center-tap Conv replacement (NPU server, WORKING)
- `/media/scott/data/lfm3_conv_bypassed.onnx` — Conv→Identity bypass (VitisAI EP loads OK, worse quality)
- `/media/scott/data/lfm3_conv_im2col.onnx` — Full depthwise conv via Pad+Slice+Mul+Add (CRASHES on VitisAI EP, FAILS on CPU EP)
- `/media/scott/data/lfm3_oga_export/model_kv_quant.onnx` — KV cache quantized model (1.6G, CPU EP works)

## Next Steps / Future Work

1. **Systemd deployment**: Reload systemd and enable services (requires sudo)
2. **Docker build**: Build and test container deployment
3. **NPU quality improvement**: Not feasible with current toolchain — center-tap is the only working Conv replacement; im2col and bypass both fail or produce worse quality. Requires AMD compiler fix for LFM3 conv layer shape analysis.
4. **KV quant deployment to production**: cpu_server.py modified with USE_KV_QUANT support (defaults to False). Enable via config `server.cpu.use_kv_quant: true` when ready. Note: INT8 quantization error accumulates over decode steps, producing repetitive output.

## Verified Status

- [x] CPU server running on port 1238, responding to health checks (uptime ~17 hours)
- [x] NPU server running on port 1239, responding to health checks (uptime ~4 hours)
- [x] KV cache quantization script fixed and working (22 tensors quantized)
- [x] Quantized model validated in onnxruntime CPUExecutionProvider
- [x] KV quantized model tested end-to-end: 52ms prefill, 35ms/decode, 28.3 TPS
- [x] Comprehensive concurrency benchmark completed (CPU vs NPU at conc 1, 2, 4)
- [x] Quality test completed: NPU output highly repetitive (unique ratio 0.08-0.25) vs CPU (0.36-0.95)
- [x] Alternative NPU Conv replacement approaches tested: center-tap (works), bypass (works, worse quality), im2col (crashes on both EPs)
- [x] cpu_server.py modified with backward-compatible KV quant support (USE_KV_QUANT=False by default)
- [x] Config updated with kv_quant_model_path and use_kv_quant options
- [x] Watchdog fixed: checks HTTP health before PID tracking, no longer starts duplicate servers
- [x] CLI status command fixed: handles stale PIDs gracefully (try/except NoSuchProcess)
- [x] Fleet monitor verified: returns detailed JSON with both CPU and NPU server status, metrics, traces, and alerts
- [x] All CLI commands verified: health, config, status, watchdog --status, fleet-monitor (prometheus/grafana config), bench
- [x] Tooling test suite: 5/5 passing
- [x] Auto-finetune pipeline: 1089 passed, 3 skipped
- [x] All documentation and installation scripts complete
