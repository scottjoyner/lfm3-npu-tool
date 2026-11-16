#!/bin/bash
# =============================================================================
# LFM3-1.7B NPU Tooling - One-Command Deploy Script
# Sets up the complete LFM3-1.7B NPU inference deployment on a fresh machine.
# =============================================================================

set -euo pipefail

TOOL_DIR="/media/scott/data/lfm3-npu-tool"

echo "============================================================"
echo "LFM3-1.7B NPU Tooling - One-Command Deploy"
echo "============================================================"
echo ""

# Step 1: Clone or update the repository
echo "[1/6] Ensuring repository is available..."
if [ ! -d "${TOOL_DIR}/.git" ]; then
    echo "  Cloning repository..."
    git clone https://github.com/scottjoyner/lfm3-npu-tool.git "${TOOL_DIR}"
else
    echo "  Repository already exists, pulling latest..."
    cd "${TOOL_DIR}"
    git pull origin master 2>/dev/null || echo "  (pull failed, using local version)"
fi

# Step 2: Install dependencies
echo ""
echo "[2/6] Installing dependencies..."
if command -v pip3 &>/dev/null; then
    pip3 install --user psutil pyyaml 2>/dev/null || pip3 install psutil pyyaml 2>/dev/null || true
elif command -v pip &>/dev/null; then
    pip install --user psutil pyyaml 2>/dev/null || pip install psutil pyyaml 2>/dev/null || true
fi

# Step 3: Create required directories
echo ""
echo "[3/6] Creating required directories..."
mkdir -p "${TOOL_DIR}/var/log"
mkdir -p "${TOOL_DIR}/var/run"
mkdir -p /media/scott/data/tmp
mkdir -p /home/scott/.config/systemd/user
mkdir -p /home/scott/.config/flm/models

# Step 4: Run the installation script
echo ""
echo "[4/6] Running installation script..."
bash "${TOOL_DIR}/install.sh"

# Step 5: Enable and start all services
echo ""
echo "[5/6] Enabling and starting systemd services..."
for svc in lfm3-npu-cpu.service lfm3-npu-npu.service lfm3-npu-metrics.service \
           lfm3-npu-quality-monitor.service; do
    systemctl --user enable "${svc}" 2>/dev/null || true
    systemctl --user start "${svc}" 2>/dev/null || true
    echo "  Started: ${svc}"
done

# Step 6: Enable all timers
echo ""
echo "[6/6] Enabling systemd timers..."
for timer in lfm3-npu-health.timer lfm3-npu-watchdog.timer \
             lfm3-npu-doctor.timer lfm3-npu-quality-monitor.timer; do
    systemctl --user enable "${timer}" 2>/dev/null || true
    systemctl --user start "${timer}" 2>/dev/null || true
    echo "  Started: ${timer}"
done

echo ""
echo "============================================================"
echo "Deploy complete!"
echo "============================================================"
echo ""
echo "Services:"
echo "  - CPU Server:    http://localhost:1238/health"
echo "  - NPU Server:    http://localhost:1239/v1/models"
echo "  - Metrics:       http://localhost:9190/metrics"
echo ""
echo "CLI commands:"
echo "  lfm3-npu smoke-test        # Validate all servers"
echo "  lfm3-npu quality-monitor   # Monitor quality with auto-restart"
echo "  lfm3-npu doctor            # Full system diagnosis"
echo "  lfm3-npu watchdog --once   # Single watchdog check"
echo ""
echo "To check status:  systemctl --user status 'lfm3-npu-*'"
echo "To view logs:     journalctl --user -u lfm3-npu-npu.service"
echo ""
