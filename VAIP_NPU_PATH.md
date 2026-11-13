# VAIP NPU Compilation Path — Findings

## Summary

The VitisAI Execution Provider can be made to load and compile models without the crashing `libvaiml.so` by using a custom VAIP configuration that bypasses VAIML compilation. However, **no NPU hardware offload is achievable** because no Strix Halo-compatible xclbin exists.

## Approach: Custom VAIP Config

Created `/tmp/vaip_npu_config3.json` with:
- `enable_vaiml_flow_compile: false` — bypasses crashing VAIML compilation
- `enable_mladf_in_memory_compile: true` — uses in-memory compilation
- `target: "AMD_AIE2P_4x4_CMC_Overlay"` — Strix Halo target
- `XLNX_TARGET_NAME=VAIML` environment variable

### Results

| Test | Result |
|------|--------|
| VitisAI EP loads with custom config | ✅ PASS |
| Simple MatMul model compiles | ✅ PASS |
| Simple MatMul model runs | ✅ PASS (0.8ms inference) |
| NPU hardware offload (duty cycle) | ❌ FAIL |
| LFM3 model compiles (MatMulNBits) | ❌ FAIL |
| LFM3 model runs on NPU | ❌ FAIL |

## Why No NPU Offload

1. **No Strix Halo xclbin**: The SDK's Strix Halo xclbins (`llm-2x4x4.xclbin`, `unified-4x4.xclbin`, etc.) are built for `xilinx_v1_ipu_0_0` platform which expects memory bank 131071. Strix Halo only has banks 0 (HOST) and 1 (SRAM). Kernel execution fails with "compute unit is connected to bank 131071."

2. **xcompiler produces .xmodel, not .xclbin**: xcompiler works with Strix Halo targets but outputs `.xmodel` files, not `.xclbin` files. pyxrt requires `.xclbin` for hardware execution.

3. **No ONNX→xmodel→xclbin pipeline**: The SDK has no converter from ONNX to xmodel format. xcompiler accepts ONNX but finds 0 ops (doesn't recognize GroupQueryAttention, MatMulNBits, etc.).

## LFM3-Specific Blocker

Even if NPU offload worked, the LFM3 model fails at MatMulNBits shape inference:
```
Node (/lm_head/MatMul_Q8) Op (MatMulNBits) [ShapeInferenceError] Incompatible dimensions for matrix multiplication
```

The LM head MatMulNBits has K=2048, N=128000 (vocab size). The VitisAI EP DPU compiler cannot handle these dimensions.

## Files

- `/tmp/vaip_npu_config3.json` — Custom VAIP config (bypasses VAIML, targets Strix Halo)
- `/tmp/test_matmul_static.onnx` — Simple test model that compiles with VitisAI EP
- `/media/scott/data/lfm3_oga_export/model_static_npu.onnx` — Statically-shaped LFM3 model

## Conclusion

Real NPU acceleration for LFM3-1.7B is NOT achievable with AMD Ryzen AI 1.8.0 SDK. All avenues exhausted:

1. `libvaiml.so` crash — unfixable from user space
2. Stub `libvaiml.so` — loads but no NPU compilation data
3. `libvaiml2.so`/`vaiml_compile_x2_v2` — segfaults
4. xcompiler — produces xmodel, not xclbin; no ONNX→xmodel converter
5. pyxrt + existing xclbins — bank mismatch (131071 vs Strix 0/1)
6. VAIP config bypass — compiles but no NPU hardware offload (no compatible xclbin)
7. LFM3 MatMulNBits — shape incompatibility with DPU compiler

