/*
 * LFM3-1.7B NPU Inference - C++ XRT Host Program
 * 
 * Bypasses ONNX Runtime + VitisAI EP entirely.
 * Talks directly to the AMD XDNA2 NPU via XRT.
 *
 * BUILD:
 *   g++ -std=c++17 -O2 -o lfm3_npu_xrt lfm3_npu_xrt.cpp \
 *       -I/opt/xilinx/xrt/include \
 *       -L/opt/xilinx/xrt/lib \
 *       -lxrt_coreutil -lxrt_core -lstdc++
 *
 * RUN:
 *   sudo ./lfm3_npu_xrt <xclbin_file>
 *
 * NOTE: Requires an xclbin compiled for LFM3 by the Vitis AI compiler.
 * The Vitis AI compiler currently crashes on LFM3 (vaiml_get_capability_v5 FPE).
 * Once AMD provides an xclbin or fixes the compiler, this will run directly
 * on the NPU without ONNX Runtime.
 */

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_uuid.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <exception>
#include <map>

using namespace std;
using namespace xrt;

class LFM3NPU {
public:
    LFM3NPU(const string& xclbin_path) {
        // Discover NPU device
        cout << "[LFM3-NPU] Discovering XDNA2 NPU device..." << endl;
        dev_ = xrt::device(0);  // First NPU device
        cout << "[LFM3-NPU] Device: " << dev_.get_info<xrt::info::device::name>() << endl;
        cout << "[LFM3-NPU] Device BDF: " << dev_.get_info<xrt::info::device::bdf>() << endl;

        // Load xclbin from file
        cout << "[LFM3-NPU] Loading xclbin: " << xclbin_path << endl;
        xclbin_uuid_ = dev_.load_xclbin(xclbin_path);
        cout << "[LFM3-NPU] xclbin loaded, UUID: " << xclbin_uuid_ << endl;

        // Get kernel handles from the xclbin
        for (const auto& kname : {"prefill", "decode", "layer_norm", "attention", "mlp"}) {
            try {
                auto k = xrt::kernel(dev_, xclbin_uuid_, kname, xrt::kernel::cu_access_mode::shared);
                kernels_[kname] = k;
                cout << "[LFM3-NPU] Found kernel: " << kname << endl;
            } catch (...) {
                // Kernel may not exist in this xclbin
            }
        }

        if (kernels_.empty()) {
            throw runtime_error("No kernels found in xclbin");
        }
    }

    // Allocate buffer on NPU device memory
    xrt::bo allocate_buffer(size_t size_bytes, xrt::bo::flags flags = xrt::bo::flags::normal) {
        return xrt::bo(dev_, size_bytes, flags, 0);
    }

    // Run prefill kernel
    void run_prefill(xrt::bo& input_ids, xrt::bo& attention_mask,
                     xrt::bo& position_ids, xrt::bo& logits_out,
                     size_t num_tokens) {
        auto& k = kernels_.at("prefill");
        auto run = xrt::run(k);

        run.set_arg(0, input_ids);
        run.set_arg(1, attention_mask);
        run.set_arg(2, position_ids);
        run.set_arg(3, logits_out);
        run.set_arg(4, static_cast<uint32_t>(num_tokens));

        run.start();
        run.wait();
    }

    // Run decode step
    void run_decode(xrt::bo& input_ids, xrt::bo& attention_mask,
                    xrt::bo& position_ids, xrt::bo& past_kv,
                    xrt::bo& logits_out, xrt::bo& new_kv,
                    size_t seq_len) {
        auto& k = kernels_.at("decode");
        auto run = xrt::run(k);

        run.set_arg(0, input_ids);
        run.set_arg(1, attention_mask);
        run.set_arg(2, position_ids);
        run.set_arg(3, past_kv);
        run.set_arg(4, logits_out);
        run.set_arg(5, new_kv);
        run.set_arg(6, static_cast<uint32_t>(seq_len));

        run.start();
        run.wait();
    }

    // Synchronize buffer from device to host
    void sync_from_device(xrt::bo& buf) {
        buf.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }

    // Synchronize buffer from host to device
    void sync_to_device(xrt::bo& buf) {
        buf.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

private:
    xrt::device dev_;
    xrt::uuid xclbin_uuid_;
    map<string, xrt::kernel> kernels_;
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <xclbin_file>" << endl;
        cerr << endl;
        cerr << "NOTE: Requires xclbin compiled for LFM3-1.7B by Vitis AI compiler." << endl;
        cerr << "The Vitis AI compiler currently crashes on LFM3." << endl;
        cerr << "Once AMD provides an xclbin or fixes the compiler, this runs directly" << endl;
        cerr << "on the NPU without ONNX Runtime or libvaiml.so." << endl;
        return 1;
    }

    string xclbin_path = argv[1];

    try {
        LFM3NPU npu(xclbin_path);
        cout << "[LFM3-NPU] NPU inference engine ready" << endl;

        // Example: allocate buffers for LFM3-1.7B
        size_t hidden_size = 2048;
        size_t seq_len = 128;
        size_t kv_size = seq_len * hidden_size * sizeof(float);

        auto input_ids = npu.allocate_buffer(seq_len * sizeof(int32_t));
        auto attention_mask = npu.allocate_buffer(seq_len * sizeof(int32_t));
        auto position_ids = npu.allocate_buffer(seq_len * sizeof(int32_t));
        auto logits = npu.allocate_buffer(seq_len * hidden_size * sizeof(float));
        auto past_kv = npu.allocate_buffer(kv_size);
        auto new_kv = npu.allocate_buffer(kv_size);

        cout << "[LFM3-NPU] Buffers allocated on NPU device memory" << endl;
        cout << "[LFM3-NPU] Ready for inference" << endl;

        return 0;
    } catch (const std::exception& e) {
        cerr << "[LFM3-NPU] Error: " << e.what() << endl;
        return 1;
    }
}
