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
echo "[2/8] Creating directories..."
mkdir -p "$CONFIG_DIR"
mkdir -p "${TOOLING_DIR}/var/log"
mkdir -p "${TOOLING_DIR}/var/run"
mkdir -p "${TOOLING_DIR}/scripts"
mkdir -p "${HOME}/.config/systemd/user"
echo "  Config: $CONFIG_DIR"
echo "  Logs:   ${TOOLING_DIR}/var/log"
echo "  PIDs:   ${TOOLING_DIR}/var/run"
echo "  Scripts: ${TOOLING_DIR}/scripts"
echo "  Systemd user: ${HOME}/.config/systemd/user"

# Install configuration
echo ""
echo "[3/8] Installing configuration..."
if [ ! -f "$CONFIG_DIR/config.yaml" ]; then
    cp "${TOOLING_DIR}/etc/lfm3-npu-config.yaml" "$CONFIG_DIR/config.yaml"
    echo "  Config installed: $CONFIG_DIR/config.yaml"
else
    echo "  Config already exists, skipping."
fi

# Install executables
echo ""
echo "[4/8] Installing executables..."
for tool in lfm3-npu lfm3-monitor lfm3-fleet-monitor lfm3-npu-server-mgr lfm3-npu-health; do
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
echo "[5/8] Installing scripts..."
for script in watchdog-cron.sh quality_benchmark.py concurrency_stress.py; do
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

# Install user systemd services
echo ""
echo "[6/8] Installing user systemd services..."
for svc in lfm3-npu-cpu.service lfm3-npu-npu.service lfm3-npu-metrics.service \
           lfm3-npu-watchdog.service lfm3-npu-watchdog.timer \
           lfm3-npu-doctor.service lfm3-npu-doctor.timer; do
    if [ -f "${TOOLING_DIR}/etc/$svc" ]; then
        cp "${TOOLING_DIR}/etc/$svc" "${HOME}/.config/systemd/user/$svc"
        echo "  Installed: ${HOME}/.config/systemd/user/$svc"
    fi
done
if command -v systemctl &>/dev/null; then
    systemctl --user daemon-reload 2>/dev/null || true
    echo "  systemd user daemon reloaded."
fi

# Install logrotate config
if [ -f "${TOOLING_DIR}/etc/logrotate-lfm3-npu.conf" ]; then
    cp "${TOOLING_DIR}/etc/logrotate-lfm3-npu.conf" "${HOME}/.config/logrotate-lfm3-npu.conf"
    echo "  Logrotate config: ${HOME}/.config/logrotate-lfm3-npu.conf"
fi

# Create symlinks
echo ""
echo "[7/8] Creating convenience symlinks..."
for alias_name in lfm3-npu-gen lfm3-npu-bench lfm3-npu-convert lfm3-npu-doctor lfm3-npu-monitor; do
    ln -sf "$INSTALL_DIR/lfm3-npu" "$INSTALL_DIR/$alias_name" 2>/dev/null || true
done
echo "  Symlinks created."

# Setup cron watchdog
echo ""
echo "[8/8] Setting up automated watchdog (cron)..."
if command -v crontab &>/dev/null; then
    # Check if already installed
    if crontab -l 2>/dev/null | grep -q "watchdog-cron.sh"; then
        echo "  Cron watchdog already configured, skipping."
    else
        (crontab -l 2>/dev/null; echo ""; echo "# LFM3-1.7B NPU inference servers - watchdog every 5 min"; echo "*/5 * * * * /media/scott/data/lfm3-npu-tool/scripts/watchdog-cron.sh >> /media/scott/data/lfm3-npu-tool/var/log/watchdog-cron.log 2>&1"; echo "# LFM3-1.7B NPU inference servers - doctor every hour"; echo "0 * * * * /media/scott/data/lfm3-npu-tool/bin/lfm3-npu doctor >> /media/scott/data/lfm3-npu-tool/var/log/doctor-cron.log 2>&1"; echo "# LFM3-1.7B NPU inference servers - auto-start on reboot"; echo "@reboot /media/scott/data/lfm3-npu-tool/start-all.sh start >> /media/scott/data/lfm3-npu-tool/var/log/boot-start.log 2>&1") | crontab -
        echo "  Cron watchdog configured (every 5 min) + doctor (hourly) + auto-start on reboot."
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
echo "  # Start both servers (via script)"
echo "  ${TOOLING_DIR}/start-all.sh start"
echo ""
echo "  # Start both servers (via user systemd)"
echo "  systemctl --user start lfm3-npu-cpu.service"
echo "  systemctl --user start lfm3-npu-npu.service"
echo ""
echo "  # Check server health"
echo "  lfm3-npu status"
echo "  lfm3-npu doctor"
echo ""
echo "  # Generate text on NPU"
echo "  lfm3-npu generate \"The quick brown fox\" --max-tokens 30 --server npu"
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
echo "  # Fleet status (with NPU device metrics)"
echo "  lfm3-fleet-monitor --once"
echo ""
echo "  # NPU Prometheus metrics exporter"
echo "  systemctl --user start lfm3-npu-metrics.service"
echo "  curl http://localhost:9190/metrics"
echo ""
echo "  # Enable systemd timers (auto-start on boot)"
echo "  systemctl --user enable --now lfm3-npu-watchdog.timer   # Watchdog every 5 min"
echo "  systemctl --user enable --now lfm3-npu-doctor.timer      # Doctor hourly"
echo ""
echo "  # Run watchdog (auto-restart CPU if uptime > 24h)"
echo "  lfm3-npu watchdog --once --max-uptime-hours 24"
echo ""
echo "  # Quality benchmark (0-100 scoring)"
echo "  python3 ${TOOLING_DIR}/scripts/quality_benchmark.py --server both"
echo ""
echo "  # Concurrency stress test"
echo "  python3 ${TOOLING_DIR}/scripts/concurrency_stress.py --server both --concurrency 4 --requests 20"
echo ""
echo "Configuration file: $CONFIG_DIR/config.yaml"
echo "Logs:               ${TOOLING_DIR}/var/log/"
echo "PIDs:               ${TOOLING_DIR}/var/run/"
echo ""
echo "NOTE: NPU server requires /dev/accel0 access and FastFlowLM v1.0.12."
echo "      CPU server does not require root."
echo "      User systemd services require no sudo (systemctl --user)."
echo ""
