# NPU Accessibility Results

## Test Results

| Test | Result | Details |
|------|--------|---------|
| Open NPU device | ✅ PASS | `pyxrt.device(0)` opens successfully |
| Load xclbin via hw_context | ✅ PASS | `pyxrt.hw_context(device, xclbin_path)` works |
| Find kernels in xclbin | ✅ PASS | Found 1 kernel: `MLIR_AIE` |
| Create buffers | ✅ PASS | `cacheable` flag works for buffer allocation |
| Run kernel on NPU | ✅ PASS | `run.start()` + `run.wait()` completes |
| Read back results | ⚠️ PARTIAL | Buffer sync/read works with cacheable flag |

## NPU Hardware Details

- **Device:** AMD NPU Strix Halo
- **Architecture:** aie2p, 6x8 topology
- **BDF:** 0000:c4:00.1
- **Device node:** /dev/accel/accel0 (crw-rw-rw-)
- **Firmware:** 1.1.2.65
- **XRT Version:** 2.25.37

## Available xclbins (NO LFM3)

| Model | xclbin files | Location |
|-------|-------------|----------|
| LFM2-1.2B-NPU2 | attn, conv, dequant, layer, mm | /home/scott/flm/npu/xclbins/ |
| LFM2-2.6B-NPU2 | - | /home/scott/flm/npu/xclbins/ |
| Qwen2.5-3B-Instruct-NPU2 | attn, conv, dequant, mm | /home/scott/flm/npu/xclbins/ |
| Gemma3-4B-NPU2 | attn, dequant, layer, lm_head, mm, vision_* | /home/scott/flm/npu/xclbins/ |
| Gemma4-E4B-IT-NPU2 | attn, dequant, layer, lm_head, mm, swa, audio_conv1d, vision_* | /home/scott/flm/npu/xclbins/ |

**No LFM3 xclbin exists anywhere on the system.**

## Conclusion

NPU hardware IS accessible from Python user space via pyxrt/XRT:
1. `pyxrt.device(0)` opens the NPU
2. `pyxrt.hw_context(device, xclbin_path)` loads xclbins
3. Kernels can be found and `run.start()`/`run.wait()` completes
4. Buffer management works with `cacheable` flag

**However:** Real LFM3-1.7B NPU acceleration requires an LFM3-compiled xclbin, which does not exist and cannot be created because:
- The Vitis AI compiler crashes on LFM3 (`libvaiml.so` SIGSEGV in `dlopen`)
- `libflexml.so` crashes on LFM3 (`canTransposeBeMovedAfterReshape` assertion)
- No pre-compiled LFM3 xclbin is available from AMD or LiquidAI

## How to Run

```bash
export LD_LIBRARY_PATH="/opt/xilinx/xrt/lib:/media/scott/data/ryzen-ai-venv/lib/python3.12/site-packages/voe/lib:$LD_LIBRARY_PATH"
/media/scott/data/ryzen-ai-venv/bin/python NPU_ACCESSIBILITY_TEST.py
```
