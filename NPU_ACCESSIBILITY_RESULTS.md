# NPU Accessibility Results

## Test Results

| Test | Result | Details |
|------|--------|---------|
| Open NPU device | ✅ PASS | `pyxrt.device(0)` opens successfully |
| Load xclbin via hw_context | ✅ PASS | `pyxrt.hw_context(device, xclbin_path)` works |
| Find kernels in xclbin | ✅ PASS | Found 1 kernel: `MLIR_AIE` |
| Create buffers | ✅ PASS | `cacheable` flag works for buffer allocation |
| Run kernel on NPU | ⚠️ PARTIAL | `set_arg()` fails - existing xclbins expect bank 131071, Strix Halo has banks 0 (HOST) and 1 (SRAM) |

## NPU Hardware Details

| Property | Value |
|----------|-------|
| Device | AMD NPU Strix Halo |
| Architecture | aie2p, 6x8 topology |
| BDF | 0000:c4:00.1 |
| Device node | /dev/accel/accel0 (accessible) |
| Firmware | 1.1.2.65 |
| XRT Version | 2.25.37 |

## Available xclbins (NO LFM3)

| Model | xclbin files | Location |
|-------|-------------|----------|
| LFM2-1.2B-NPU2 | attn, conv, dequant, layer, mm | /home/scott/flm/npu/xclbins/ |
| LFM2-2.6B-NPU2 | - | /home/scott/flm/npu/xclbins/ |
| Qwen2.5-3B-Instruct-NPU2 | attn, conv, dequant, mm | /home/scott/flm/npu/xclbins/ |
| Gemma3-4B-NPU2 | attn, dequant, layer, lm_head, mm, vision_* | /home/scott/flm/npu/xclbins/ |
| Gemma4-E4B-IT-NPU2 | attn, dequant, layer, lm_head, mm, swa, audio_conv1d, vision_* | /home/scott/flm/npu/xclbins/ |

**No LFM3 xclbin exists anywhere on the system.**

## Bank Mismatch Details

Existing xclbins were compiled for a different NPU platform whose compute units connect to bank 131071. Strix Halo only has memory banks 0 (HOST, 64MB) and 1 (SRAM, 48MB). When running `pyxrt.run(kernel).set_arg(i, buf)`, XRT reports:

```
[XRT] WARNING: Kernel MLIR_AIE has no compute units with connectivity required for global argument at index 0.
The argument is allocated in bank 0, the compute unit is connected to bank 131071.
[XRT] WARNING: Reverting to host copy of buffers (exec_buf: Operation not supported)
RuntimeError: No host side buffer in destination buffer: Invalid argument
```

Kernel execution requires an LFM3-compiled xclbin for Strix Halo.

## Conclusion

NPU hardware IS accessible from Python user space via pyxrt/XRT:
1. `pyxrt.device(0)` opens the NPU ✅
2. `pyxrt.hw_context(device, xclbin_path)` loads xclbins ✅
3. Kernels can be found ✅
4. Buffer management works with `cacheable` flag ✅
5. Kernel execution fails due to bank mismatch with existing xclbins ❌

**Real LFM3-1.7B NPU acceleration requires an LFM3-compiled xclbin for Strix Halo**, which does not exist and cannot be created because:
- The Vitis AI compiler (`libvaiml.so`) crashes with SIGSEGV during `dlopen()` — AMD proprietary code, cannot be fixed from user space
- `libvaiml2.so` works but exports only `vaiml_compile_x2_v2` which takes a JSON config string and segfaults without the correct schema
- No ONNX→xmodel (XIR) converter is available in the SDK
- xcompiler works but requires `.xmodel` files, not ONNX

## How to Run

```bash
export LD_LIBRARY_PATH="/opt/xilinx/xrt/lib:/media/scott/data/ryzen-ai-venv/lib/python3.12/site-packages/voe/lib:$LD_LIBRARY_PATH"
/media/scott/data/ryzen-ai-venv/bin/python NPU_ACCESSIBILITY_TEST.py
```
