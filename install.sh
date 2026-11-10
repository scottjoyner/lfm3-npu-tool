#!/bin/bash
# =============================================================================
# LFM3-1.7B NPU Tooling - Installation Script
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLING_DIR="$SCRIPT_DIR"
INSTALL_DIR="/usr/local/bin"
CONFIG_DIR="$HOME/.config/lfm3-npu"

echo "============================================================"
echo "  LFM3-1.7B NPU Tooling - Installation"
echo "============================================================"
echo ""

# Check prerequisites
echo "[1/6] Checking prerequisites..."

RYZEN_VENV="/media/scott/data/ryzen-ai-venv"
if [ ! -d "$RYZEN_VENV" ]; then
    echo "WARNING: Ryzen AI venv not found at $RYZEN_VENV"
    echo "  NPU features may not work without it."
else
    echo "  Ryzen AI venv: $RYZEN_VENV ✓"
fi

if [ ! -c /dev/accel0 ]; then
    echo "WARNING: /dev/accel0 not found. NPU may not be available."
else
    echo "  NPU device: /dev/accel0 ✓"
fi

if command -v xrt-smi &>/dev/null; then
    echo "  XRT: available ✓"
else
    echo "WARNING: xrt-smi not found."
fi

# Create directories
echo ""
echo "[2/6] Creating directories..."
mkdir -p "$CONFIG_DIR"
mkdir -p "${TOOLING_DIR}/var/log"
mkdir -p "${TOOLING_DIR}/var/run"
echo "  Config: $CONFIG_DIR"
echo "  Logs:   ${TOOLING_DIR}/var/log"
echo "  PIDs:   ${TOOLING_DIR}/var/run"

# Install configuration
echo ""
echo "[3/6] Installing configuration..."
if [ ! -f "$CONFIG_DIR/config.yaml" ]; then
    cp "${TOOLING_DIR}/etc/lfm3-npu-config.yaml" "$CONFIG_DIR/config.yaml"
    echo "  Config installed: $CONFIG_DIR/config.yaml"
else
    echo "  Config already exists, skipping."
fi

# Install executables
echo ""
echo "[4/6] Installing executables..."
for tool in lfm3-npu lfm3-npu-server-mgr lfm3-npu-health; do
    cp "${TOOLING_DIR}/bin/$tool" "$INSTALL_DIR/$tool"
    chmod +x "$INSTALL_DIR/$tool"
    echo "  Installed: $INSTALL_DIR/$tool"
done

# Create symlinks
echo ""
echo "[5/6] Creating convenience symlinks..."
ln -sf "$INSTALL_DIR/lfm3-npu" "$INSTALL_DIR/lfm3-npu-gen" 2>/dev/null || true
ln -sf "$INSTALL_DIR/lfm3-npu" "$INSTALL_DIR/lfm3-npu-bench" 2>/dev/null || true
ln -sf "$INSTALL_DIR/lfm3-npu" "$INSTALL_DIR/lfm3-npu-convert" 2>/dev/null || true
echo "  Symlinks created."

# Verify
echo ""
echo "[6/6] Verifying installation..."
echo ""
"$INSTALL_DIR/lfm3-npu" status 2>&1 || true

echo ""
echo "============================================================"
echo "  Installation Complete!"
echo "============================================================"
echo ""
echo "Usage examples:"
echo ""
echo "  # Start NPU server (requires root for /dev/accel0 access)"
echo "  sudo lfm3-npu-server-mgr start npu"
echo ""
echo "  # Generate text on NPU"
echo "  lfm3-npu generate \"The quick brown fox\" --max-tokens 30"
echo ""
echo "  # Check server health"
echo "  lfm3-npu-health npu"
echo ""
echo "  # Benchmark performance"
echo "  lfm3-npu bench --tokens 50 --iterations 3"
echo ""
echo "  # Convert a model to NPU format"
echo "  lfm3-npu convert --model model.onnx --output model_npu.onnx --method centertap"
echo ""
echo "  # Show configuration"
echo "  lfm3-npu config"
echo ""
echo "Configuration file: $CONFIG_DIR/config.yaml"
echo "Logs:               ${TOOLING_DIR}/var/log/"
echo "PIDs:               ${TOOLING_DIR}/var/run/"
echo ""
echo "NOTE: NPU server commands require root for /dev/accel0 access."
echo "      CPU server commands do not require root."
echo ""