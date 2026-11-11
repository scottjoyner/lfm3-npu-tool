# Stub libvaiml.so Build Instructions

## Problem
`libvaiml.so` crashes with SIGSEGV in protobuf descriptor registration during `dlopen()`,
called from `vaip_vaiml::ensureVaimlLibraryLoaded()` in `libonnxruntime_vitisai_ep.so`.
This happens BEFORE any graph analysis or Python code runs — deep inside AMD's proprietary
library during library initialization.

## Solution: Stub libvaiml.so
A stub library with dummy implementations of all ~40 exported VAIML functions allows
`dlopen(libvaiml.so)` and `dlsym()` to succeed. VitisAI EP loads and runs inference,
but since all VAIML functions return no-op values (NULL, 0, -1), VitisAI EP treats all
ops as unsupported and runs through CPU internally. This is CPU execution through the
VitisAI EP wrapper — NOT real NPU hardware acceleration.

## Build
```bash
gcc -shared -fPIC -o libvaiml.so vaiml_stub.c
```

## Install
```bash
cp libvaiml.so /media/scott/data/ryzen-ai-venv/lib/python3.12/site-packages/flexml/flexml_extras/lib/libvaiml.so
# Keep original as backup:
cp libvaiml.so.orig /media/scott/data/ryzen-ai-venv/lib/python3.12/site-packages/flexml/flexml_extras/lib/libvaiml.so.orig
```

## Source
See `vaiml_stub.c` for the complete stub implementation.

## Real NPU Acceleration Requires AMD
| What's Needed | Who Can Provide |
|---------------|-----------------|
| Fix `libvaiml.so` protobuf crash during `dlopen()` | AMD |
| LFM3-compiled `.xclbin` artifact | AMD |
| LiquidAI re-export LFM3 with standard ONNX ops | LiquidAI |

## NPU Hardware Status
- Device: AMD NPU Strix Halo (aie2p, 6x8 topology)
- BDF: 0000:c4:00.1
- Device node: /dev/accel/accel0 (accessible)
- Firmware: 1.1.2.65
- XRT Version: 2.25.37

## Current Performance (stub libvaiml.so, VitisAI EP on CPU)
| Prompt | CPU EP | VitisAI EP (stub) | Speedup |
|--------|--------|-------------------|---------|
| "The quick brown fox" | 12.9s | 5.2s | ~2.5x |
| "Once upon a time" | ~11s | ~5s | ~2.2x |
| "Say hello" | ~13s | ~5s | ~2.6x |
| "What is 2+2?" | ~14s | ~9s | ~1.6x |

Speedup comes from VitisAI EP internal optimizations, not NPU hardware.