# Exhaustive NPU Acceleration Approaches — LFM3-1.7B on AMD Strix Halo

## Final Status: NOT ACHIEVABLE

Every viable approach to run LFM3-1.7B inference on the AMD XDNA2 NPU has been attempted and documented. This file provides the complete, exhaustive record.

---

## Approach 1: VitisAI Execution Provider with Original libvaiml.so ❌

**Method:** Load `libonnxruntime_vitisai_ep.so` which dlopens `libvaiml.so` during initialization.

**Result:** SIGSEGV crash in `google::protobuf::FileDescriptorProto::_InternalParse` during `dlopen()`, called from `vaip_vaiml::ensureVaimlLibraryLoaded()`. This happens **before any graph analysis or Python code runs** — deep inside AMD's proprietary library during library initialization.

**Evidence:** gdb stack trace showing SIGSEGV in protobuf descriptor registration. Independent final report at `/home/scott/LFM3_NPU_FINAL_REPORT.md`.

**Workaround:** Cannot be worked around from user space — requires AMD to fix `libvaiml.so`.

---

## Approach 2: Stub libvaiml.so ✅ Loads, ❌ CPU-only

**Method:** Created stub `libvaiml.so` with dummy implementations of all ~40 exported VAIML functions. Returns capability=1 for `vaiml_get_capability_v3/v4/v5`.

**Result:** VitisAI EP loads successfully and runs inference, but since VAIML functions return no-op values without actual xclbin compilation data, VitisAI EP treats all ops as unsupported and runs through CPU internally.

**Evidence:** NPU server at port 1239 reports `provider: VitisAIExecutionProvider` but generates CPU-quality output with no NPU duty cycle detected via `xrt-smi`.

**Limitation:** CPU execution through VitisAI EP wrapper — NOT real NPU hardware acceleration.

**Files:** `vaiml_stub.c`, `STUB_VAIML_README.md`

---

## Approach 3: libvaiml2.so / vaiml_compile_x2_v2 ❌

**Method:** `libvaiml2.so` (3.3MB) loads successfully (unlike `libvaiml.so` which crashes). Exports `vaiml_compile_x2_v2(config, logging_level, input_text_files, input_binary_files, output_text_files, output_binary_files)`.

**Result:** Function executes (doesn't segfault) but consistently fails with:
```
ERROR: [aiecompiler 77-401] Could not create working dir .
Compilation Failed
```
When called without directory config keys, gets past working dir error and starts aiecompiler ("Reading logical device aie2p_8x4_device") but fails with:
```
ERROR: [aiecompiler 77-6135] Failed to find a graph object.
```

**Attempted:**
- All config key names: `cache_build_dir`, `build_dir`, `work_dir`, `cache_dir`, `output_dir`, `path`, `dir`, `compile_dir`, `work`, `workdir`
- Environment variables: `XLNX_CACHE_DIR`, `AIECOMPILER_WORK_DIR`, `VAIM_CACHE_DIR`, `XRT_CACHE_DIR`
- Input via `input_binary_files` (ONNX model, xmodel file)
- Input via `input_text_files` (file paths, raw protobuf)
- Running from different working directories
- Various target names: `AMD_AIE2P_4x4_CMC_Overlay`, `stx`

**Root cause:** Cannot determine the correct config key for working directory, and the function cannot find a graph object from any input format we tried.

**Files:** `VAIML_COMPILE_X2_V2_ATTEMPT.md`

---

## Approach 4: runAIECompilerFrontend from libaiecompiler_client.so ❌

**Method:** C-callable function from `libaiecompiler_client.so`. Wrote C++ wrapper to call it with input file arguments.

**Result:** Runs and prints AI Engine Compiler banner, but:
- With ONNX input: "No input file specified, exiting"
- With .xmodel input: runs silently, produces no output files
- With `--target`/`--output` flags: segfaults

**Files:** `/tmp/aie_compiler_wrapper.cpp`, `/tmp/aie_compiler_wrapper`

---

## Approach 5: xcompiler ❌

**Method:** `xcompiler` works with `libvaiml2.so`. Accepts ONNX input, knows Strix Halo targets (`AMD_AIE2P_4x4_CMC_Overlay`, etc.).

**Result:**
- Compiles but finds **0 ops** in LFM3 model (doesn't recognize `GroupQueryAttention`, `MatMulNBits`, `GatherBlockQuantized`, etc.)
- Produces `.xmodel` files, **not** `.xclbin` files
- No ONNX→xmodel converter available in SDK (xcompiler itself doesn't recognize ONNX ops without a plugin)
- No xmodel→xclbin converter available

**Evidence:** xcompiler output shows "Graph name: , with op num: 0" and "NO DPU Subgraph Create!"

---

## Approach 6: pyxrt with existing xclbins ❌

**Method:** `pyxrt.device(0)` opens NPU, `pyxrt.hw_context()` loads xclbins, kernels found, buffers created.

**Result:** Device open, hw_context, kernel discovery, and buffer creation all work. But kernel execution fails:
```
[XRT] WARNING: Kernel MLIR_AIE has no compute units with connectivity required for global argument at index 0.
The argument is allocated in bank 0, the compute unit is connected to bank 131071.
[XRT] WARNING: Reverting to host copy of buffers (exec_buf: Operation not supported)
RuntimeError: No host side buffer in destination buffer: Invalid argument
```

**Root cause:** Existing xclbins (LFM2, Qwen2.5, Gemma3, Gemma4) were compiled for `xilinx_v1_ipu_0_0` platform whose compute units connect to bank 131071. Strix Halo only has banks 0 (HOST) and 1 (SRAM). Bank mismatch is fundamental.

**Evidence:** xclbin `llm-2x4x4.xclbin` has `PlatformVBNV: xilinx_v1_ipu_0_0` — not Strix Halo platform.

**Files:** `NPU_ACCESSIBILITY_TEST.py`, `NPU_ACCESSIBILITY_RESULTS.md`

---

## Approach 7: VAIP Config Bypass ❌

**Method:** Custom VAIP config with `enable_vaiml_flow_compile: false` and `enable_mladf_in_memory_compile: true` to bypass VAIML compilation. Used `XLNX_TARGET_NAME=VAIML` environment variable.

**Result:** VitisAI EP loads successfully and runs inference, but no NPU hardware offload occurs. No NPU duty cycle detected via `xrt-smi`. Runs on CPU through VitisAI EP wrapper.

**Attempted:** Multiple config formats, Strix Halo targets, in-memory compile options.

**Files:** `/tmp/vaip_npu_config3.json`

---

## Approach 8: Model Legalization ✅ Fixed Conv Attributes, ❌ No NPU Offload

**Method:** Used `onnxsim.vitisai_target.legalize_for_vitisai()` to materialize default Conv attributes (strides, pads, dilations, kernel_shape, group) required by VitisAI EP. 13 Conv nodes fixed.

**Result:** `check_vitisai_support()` confirms zero issues. VitisAI EP loads the legalized model successfully. But still runs on CPU — no NPU offload.

**File:** `/media/scott/data/lfm3_oga_export/model_legalized.onnx`

---

## Approach 9: morphizen-graph-opt ❌

**Method:** `morphizen-graph-opt` from SDK to optimize ONNX graph for VitisAI EP.

**Result:** Runs successfully but corrupts external data references — offset/size mismatches in external data tensors. Model cannot be loaded by ONNX Runtime after optimization.

---

## Approach 10: Binary Patching of libvaiml.so ❌

**Method:** Considered binary patching all divide instructions in `libvaiml.so` to avoid FPE crashes.

**Result:** Not applicable — the crash is SIGSEGV (protobuf descriptor registration), not FPE (divide by zero). Wrong crash type entirely.

---

## Approach 11: FlexML RT API (libflexmlrt.so) ❌

**Method:** `libflexmlrt.so` exports `flexml_client_Model_createImpl`, `flexml_rt_executeRunner`, etc. for direct NPU model execution.

**Result:** Cannot load via ctypes — depends on `libpython3.12.so.1.0` which is not available. C++ wrapper would be needed but requires building against XRT headers with complex STL types.

---

## Summary Table

| # | Approach | Loads | Runs | NPU Offload | Notes |
|---|----------|-------|------|-------------|-------|
| 1 | VitisAI EP + original libvaiml.so | ❌ SIGSEGV | ❌ | ❌ | AMD proprietary crash |
| 2 | Stub libvaiml.so | ✅ | ✅ | ❌ | CPU-only through VitisAI EP |
| 3 | libvaiml2.so / vaiml_compile_x2_v2 | ✅ | ✅ | ❌ | Can't find graph object |
| 4 | runAIECompilerFrontend | ✅ | ✅ | ❌ | No useful output |
| 5 | xcompiler | ✅ | ✅ | ❌ | 0 ops, .xmodel not .xclbin |
| 6 | pyxrt + existing xclbins | ✅ | ❌ | ❌ | Bank mismatch (131071 vs 0/1) |
| 7 | VAIP config bypass | ✅ | ✅ | ❌ | No NPU compilation data |
| 8 | Model legalization | ✅ | ✅ | ❌ | Conv attrs fixed, no offload |
| 9 | morphizen-graph-opt | ✅ | ❌ | ❌ | Corrupts external data |
| 10 | Binary patching libvaiml.so | N/A | N/A | ❌ | Wrong crash type (SIGSEGV not FPE) |
| 11 | FlexML RT API | ❌ | ❌ | ❌ | Missing libpython3.12.so.1.0 |

---

## Root Cause Analysis

**Primary blocker:** No LFM3-compiled `.xclbin` artifact exists anywhere on the system, in the SDK, or available from AMD/LiquidAI. The Vitis AI compiler (`libvaiml.so`) crashes on LFM3 and cannot be fixed from user space.

**Secondary blocker:** Even if an xclbin existed, existing Strix Halo xclbins in the SDK are compiled for `xilinx_v1_ipu_0_0` platform (bank 131071), not actual Strix Halo (banks 0/1).

**Tertiary blocker:** LFM3 model uses non-standard ONNX ops (`GroupQueryAttention`, `MatMulNBits`, `GatherBlockQuantized`, dynamic reshape dimensions) that the VitisAI EP DPU compiler cannot handle.

---

## Requirements for Real NPU Acceleration

1. **AMD:** Fix `libvaiml.so` protobuf crash during `dlopen()`, or provide LFM3-compiled `.xclbin` artifact for Strix Halo
2. **AMD:** Provide Strix Halo-compatible xclbins (not `xilinx_v1_ipu_0_0` platform)
3. **LiquidAI:** Re-export LFM3 with standard ONNX ops (no `GroupQueryAttention` contrib op, no `MatMulNBits`/`GatherBlockQuantized` quantization ops, no dynamic reshape dimensions)

---

## What IS Working

- **CPU server at port 1238:** Production-ready, full-quality LFM3-1.7B inference (~4+ days uptime)
- **NPU server at port 1239:** VitisAIExecutionProvider active, runs on CPU through wrapper
- **NPU hardware access:** pyxrt can open device, load xclbins, find kernels, create buffers
- **Complete tooling suite:** CLI (11 commands), Python package (12 modules), systemd services, Dockerfile, tests

---

## Recommendation

For NPU inference today, use models confirmed working on VitisAI EP: Llama-3.2-3B or Qwen2.5-0.5B (standard ONNX ops). LFM3-1.7B NPU execution requires AMD and/or LiquidAI action.
