#!/usr/bin/env python3
"""
NPU Accessibility Test - Prove NPU hardware is accessible from user space via pyxrt/XRT.

This test demonstrates that:
1. The AMD XDNA2 NPU device is accessible (/dev/accel/accel0)
2. pyxrt can open the device
3. hw_context can load xclbin files
4. Kernels can be found in xclbins
5. Kernel runs can be started and completed

NOTE: This does NOT prove LFM3-1.7B runs on NPU. No LFM3 xclbin exists.
The LFM2 xclbin is used only to prove the infrastructure works.

Run: python NPU_ACCESSIBILITY_TEST.py
"""

import pyxrt
import numpy as np
import sys

def test_npu_accessibility():
    print("=" * 60)
    print("NPU Accessibility Test via pyxrt/XRT")
    print("=" * 60)
    
    # Test 1: Open NPU device
    print("\n[1] Opening NPU device...")
    try:
        device = pyxrt.device(0)
        print(f"   Device opened: {device}")
        device_ok = True
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {e}")
        device_ok = False
    
    if not device_ok:
        print("\nRESULT: NPU device not accessible")
        return False
    
    # Test 2: Load xclbin via hw_context
    print("\n[2] Loading xclbin via hw_context...")
    xclbin_path = '/home/scott/flm/npu/xclbins/LFM2-1.2B-NPU2/mm.xclbin'
    try:
        hw_ctx = pyxrt.hw_context(device, xclbin_path)
        print(f"   hw_context created successfully")
        hw_ctx_ok = True
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {e}")
        hw_ctx_ok = False
    
    if not hw_ctx_ok:
        print("\nRESULT: hw_context creation failed")
        return False
    
    # Test 3: Find kernels in xclbin
    print("\n[3] Finding kernels in xclbin...")
    xclbin = pyxrt.xclbin(xclbin_path)
    kernels = xclbin.get_kernels()
    print(f"   Found {len(kernels)} kernel(s)")
    for k in kernels:
        print(f"   - {k.get_name()}")
    kernels_ok = len(kernels) > 0
    
    if not kernels_ok:
        print("\nRESULT: No kernels found in xclbin")
        return False
    
    # Test 4: Create kernel and buffers
    print("\n[4] Creating kernel and buffers...")
    kname = kernels[0].get_name()
    # Get num_args from xclbin kernel object (hw_ctx kernel doesn't have get_num_args)
    num_args = kernels[0].get_num_args()
    kernel = pyxrt.kernel(hw_ctx, kname)
    print(f"   Kernel: {kname}, args: {num_args}")
    
    try:
        buffers = []
        for i in range(num_args):
            buf = pyxrt.bo(device, 4096, pyxrt.bo.flags.cacheable, 0)
            data = np.arange(1024, dtype=np.int32) + i * 256
            buf.write(data, 0)
            buf.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            buffers.append(buf)
        print(f"   Created {len(buffers)} cacheable buffers")
        buffers_ok = True
    except Exception as e:
        print(f"   Buffer creation failed: {type(e).__name__}: {e}")
        buffers_ok = False
    
    if not buffers_ok:
        print("\nRESULT: Buffer creation failed")
        return False
    
    # Test 5: Run kernel (may fail due to bank mismatch with non-Strix xclbin)
    print("\n[5] Running kernel on NPU...")
    run = pyxrt.run(kernel)
    for i, buf in enumerate(buffers):
        try:
            run.set_arg(i, buf)
        except Exception as e:
            print(f"   set_arg({i}) failed: {type(e).__name__}: {e}")
            print("   NOTE: Existing xclbins (LFM2, Qwen2.5, etc.) were compiled for")
            print("         a different NPU platform expecting bank 131071. Strix Halo")
            print("         only has banks 0 (HOST) and 1 (SRAM). Kernel execution requires")
            print("         an LFM3-compiled xclbin for Strix Halo.")
            return False  # Partial success: device, hw_context, kernels, buffers all work
    
    try:
        run.start()
        run.wait()
        print(f"   Kernel started and completed!")
        
        # Read back buffer 0
        buffers[0].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        data = np.frombuffer(buffers[0].read(4096), dtype=np.int32)
        print(f"   Buffer 0 first 8 values: {data[:8]}")
        print("   RESULT: NPU kernel executed successfully!")
        return True
    except Exception as e:
        print(f"   Kernel run failed: {type(e).__name__}: {e}")
        return False  # Partial success

if __name__ == "__main__":
    success = test_npu_accessibility()
    print("\n" + "=" * 60)
    if success:
        print("OVERALL: NPU hardware is accessible from user space via pyxrt/XRT")
        print("         xclbins can be loaded via hw_context")
        print("         Kernels can be found and executed")
        print("         BUT: No LFM3 xclbin exists - only LFM2, Qwen2.5, Gemma3, Gemma4")
        print("         Real LFM3 NPU acceleration requires AMD to provide an xclbin")
    else:
        print("OVERALL: NPU accessibility test failed")
    print("=" * 60)
    sys.exit(0 if success else 1)
