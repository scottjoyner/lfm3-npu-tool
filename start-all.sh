#!/bin/bash
# =============================================================================
# LFM3-1.7B Dual-Server Startup Script
# Starts both CPU (port 1238) and NPU (port 1239) inference servers via systemd --user.
# =============================================================================

set -euo pipefail

TOOL_DIR="/media/scott/data/lfm3-npu-tool"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

wait_for_port() {
    local port=$1
    local host=$2
    local timeout=${3:-60}
    local elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        if curl -s --max-time 2 "http://${host}:${port}/health" > /dev/null 2>&1 || \
           curl -s --max-time 2 "http://${host}:${port}/v1/models" > /dev/null 2>&1; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    return 1
}

start_cpu() {
    log "Starting CPU server on port 1238 via systemd..."
    if systemctl --user is-active --quiet lfm3-npu-cpu.service 2>/dev/null; then
        log "CPU server already running"
        return 0
    fi
    systemctl --user start lfm3-npu-cpu.service 2>/dev/null || {
        log "ERROR: Failed to start CPU server via systemd"
        return 1
    }
    if wait_for_port 1238 "0.0.0.0"; then
        log "CPU server ready on port 1238"
    else
        log "WARNING: CPU server may still be initializing. Check: journalctl --user -u lfm3-npu-cpu.service"
    fi
}

start_npu() {
    log "Starting NPU (FastFlowLM) server on port 1239 via systemd..."
    if [[ ! -x /tmp/flm ]]; then
        log "ERROR: FLM binary not found or not executable: /tmp/flm"
        log "  Download from: https://github.com/ROCm/FastFlowLM/releases/tag/v1.0.12"
        return 1
    fi
    if systemctl --user is-active --quiet lfm3-npu-npu.service 2>/dev/null; then
        log "NPU server already running"
        return 0
    fi
    systemctl --user start lfm3-npu-npu.service 2>/dev/null || {
        log "ERROR: Failed to start NPU server via systemd"
        return 1
    }
    if wait_for_port 1239 "127.0.0.1"; then
        log "NPU server ready on port 1239"
    else
        log "WARNING: NPU server may still be initializing. Check: journalctl --user -u lfm3-npu-npu.service"
    fi
}

start_metrics() {
    log "Starting NPU metrics exporter on port 9190 via systemd..."
    if systemctl --user is-active --quiet lfm3-npu-metrics.service 2>/dev/null; then
        log "Metrics exporter already running"
        return 0
    fi
    systemctl --user start lfm3-npu-metrics.service 2>/dev/null || {
        log "ERROR: Failed to start metrics exporter via systemd"
        return 1
    }
    if wait_for_port 9190 "0.0.0.0"; then
        log "Metrics exporter ready on port 9190"
    else
        log "WARNING: Metrics exporter may still be initializing. Check: journalctl --user -u lfm3-npu-metrics.service"
    fi
}

status() {
    log "=== Server Status ==="
    for svc in lfm3-npu-cpu.service lfm3-npu-npu.service lfm3-npu-metrics.service; do
        local name=$(basename "${svc}" .service)
        if systemctl --user is-active --quiet "${svc}" 2>/dev/null; then
            log "${name}: RUNNING"
        else
            log "${name}: NOT RUNNING"
        fi
    done
    log ""
    log "CPU health: $(curl -s --max-time 3 http://0.0.0.0:1238/health 2>/dev/null || echo 'unreachable')"
    local npu_models=$(curl -s --max-time 3 "http://127.0.0.1:1239/v1/models" 2>/dev/null || echo '')
    if [[ -n "${npu_models}" ]]; then
        local npu_lfm=$(echo "${npu_models}" | python3 -c "import sys,json; d=json.load(sys.stdin); lfm=[m for m in d['data'] if 'lfm3' in m['id']]; print(lfm[0]['id'] if lfm else 'not found')" 2>/dev/null || echo 'parse error')
        log "NPU models: ${npu_lfm}"
    else
        log "NPU models: unreachable"
    fi
    local metrics=$(curl -s --max-time 3 http://0.0.0.0:9190/metrics 2>/dev/null | head -1 || echo 'unreachable')
    log "Metrics: ${metrics}"
}

case "${1:-start}" in
    start)
        start_cpu
        start_npu
        start_metrics
        log ""
        log "=== All servers started ==="
        log "  CPU server:   http://0.0.0.0:1238"
        log "  NPU server:   http://127.0.0.1:1239"
        log "  Metrics:      http://0.0.0.0:9190/metrics"
        ;;
    stop)
        log "Stopping CPU server..."
        systemctl --user stop lfm3-npu-cpu.service 2>/dev/null || true
        log "Stopping NPU server..."
        systemctl --user stop lfm3-npu-npu.service 2>/dev/null || true
        log "Stopping metrics exporter..."
        systemctl --user stop lfm3-npu-metrics.service 2>/dev/null || true
        log "All servers stopped."
        ;;
    restart)
        "$0" stop
        sleep 3
        "$0" start
        ;;
    status)
        status
        ;;
    enable)
        log "Enabling services for boot..."
        systemctl --user enable lfm3-npu-cpu.service lfm3-npu-npu.service lfm3-npu-metrics.service 2>/dev/null || true
        systemctl --user enable lfm3-npu-watchdog.timer lfm3-npu-doctor.timer lfm3-npu-health.timer 2>/dev/null || true
        log "Services enabled for boot."
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|enable}"
        exit 1
        ;;
esac
