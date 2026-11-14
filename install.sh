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
echo "[1/7] Checking prerequisites..."

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

if command -v crontab &>/dev/null; then
    echo "  crontab: available ✓"
else
    echo "WARNING: crontab not found (watchdog automation unavailable)."
fi

# Create directories
echo ""
echo "[2/7] Creating directories..."
mkdir -p "$CONFIG_DIR"
mkdir -p "${TOOLING_DIR}/var/log"
mkdir -p "${TOOLING_DIR}/var/run"
mkdir -p "${TOOLING_DIR}/scripts"
echo "  Config: $CONFIG_DIR"
echo "  Logs:   ${TOOLING_DIR}/var/log"
echo "  PIDs:   ${TOOLING_DIR}/var/run"
echo "  Scripts: ${TOOLING_DIR}/scripts"

# Install configuration
echo ""
echo "[3/7] Installing configuration..."
if [ ! -f "$CONFIG_DIR/config.yaml" ]; then
    cp "${TOOLING_DIR}/etc/lfm3-npu-config.yaml" "$CONFIG_DIR/config.yaml"
    echo "  Config installed: $CONFIG_DIR/config.yaml"
else
    echo "  Config already exists, skipping."
fi

# Install executables
echo ""
echo "[4/7] Installing executables..."
for tool in lfm3-npu lfm3-monitor lfm3-npu-server-mgr lfm3-npu-health; do
    if [ -f "${TOOLING_DIR}/bin/$tool" ]; then
        cp "${TOOLING_DIR}/bin/$tool" "$INSTALL_DIR/$tool"
        chmod +x "$INSTALL_DIR/$tool"
        echo "  Installed: $INSTALL_DIR/$tool"
    else
        echo "  Skipping (not found): $tool"
    fi
done

# Install scripts
echo ""
echo "[5/7] Installing scripts..."
for script in watchdog-cron.sh; do
    if [ -f "${TOOLING_DIR}/scripts/$script" ]; then
        cp "${TOOLING_DIR}/scripts/$script" "${TOOLING_DIR}/scripts/$script"
        chmod +x "${TOOLING_DIR}/scripts/$script"
        echo "  Installed: ${TOOLING_DIR}/scripts/$script"
    fi
done

# Install start-all.sh
if [ -f "${TOOLING_DIR}/start-all.sh" ]; then
    cp "${TOOLING_DIR}/start-all.sh" "${TOOLING_DIR}/start-all.sh"
    chmod +x "${TOOLING_DIR}/start-all.sh"
    echo "  Installed: ${TOOLING_DIR}/start-all.sh"
fi

# Create symlinks
echo ""
echo "[6/7] Creating convenience symlinks..."
for alias_name in lfm3-npu-gen lfm3-npu-bench lfm3-npu-convert lfm3-npu-doctor lfm3-npu-monitor; do
    ln -sf "$INSTALL_DIR/lfm3-npu" "$INSTALL_DIR/$alias_name" 2>/dev/null || true
done
echo "  Symlinks created."

# Setup cron watchdog
echo ""
echo "[7/7] Setting up automated watchdog (cron)..."
if command -v crontab &>/dev/null; then
    # Check if already installed
    if crontab -l 2>/dev/null | grep -q "watchdog-cron.sh"; then
        echo "  Cron watchdog already configured, skipping."
    else
        (crontab -l 2>/dev/null; echo ""; echo "# LFM3-1.7B NPU inference servers - watchdog every 5 min"; echo "*/5 * * * * /media/scott/data/lfm3-npu-tool/scripts/watchdog-cron.sh >> /media/scott/data/lfm3-npu-tool/var/log/watchdog-cron.log 2>&1"; echo "# LFM3-1.7B NPU inference servers - doctor every hour"; echo "0 * * * * /media/scott/data/lfm3-npu-tool/bin/lfm3-npu doctor >> /media/scott/data/lfm3-npu-tool/var/log/doctor-cron.log 2>&1") | crontab -
        echo "  Cron watchdog configured (every 5 min) + doctor (hourly)."
    fi
else
    echo "  Skipped (crontab not available)."
fi

# Verify
echo ""
echo "Verifying installation..."
echo ""
"$INSTALL_DIR/lfm3-npu" status 2>&1 || true

echo ""
echo "============================================================"
echo "  Installation Complete!"
echo "============================================================"
echo ""
echo "Usage examples:"
echo ""
echo "  # Start both servers"
echo "  ${TOOLING_DIR}/start-all.sh start"
echo ""
echo "  # Generate text on NPU"
echo "  lfm3-npu generate \"The quick brown fox\" --max-tokens 30 --server npu"
echo ""
echo "  # Check server health"
echo "  lfm3-npu status"
echo "  lfm3-npu doctor"
echo ""
echo "  # Benchmark performance"
echo "  lfm3-npu bench --server npu --tokens 50 --iterations 3"
echo ""
echo "  # Compare CPU vs NPU live"
echo "  lfm3-npu compare-servers --prompts \"What is 2+2?\" \"The capital of France is\""
echo ""
echo "  # Monitor servers (real-time dashboard)"
echo "  lfm3-monitor"
echo ""
echo "  # Run watchdog (auto-restart CPU if uptime > 24h)"
echo "  lfm3-npu watchdog --once --max-uptime-hours 24"
echo ""
echo "Configuration file: $CONFIG_DIR/config.yaml"
echo "Logs:               ${TOOLING_DIR}/var/log/"
echo "PIDs:               ${TOOLING_DIR}/var/run/"
echo ""
echo "NOTE: NPU server requires /dev/accel0 access and FastFlowLM v1.0.12."
echo "      CPU server does not require root."
echo ""
