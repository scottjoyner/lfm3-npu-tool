#!/bin/bash
# =============================================================================
# LFM3-1.7B Dual-Server Startup Script
# Starts both CPU (port 1238) and NPU (port 1239) inference servers.
# =============================================================================

set -euo pipefail

TOOL_DIR="/media/scott/data/lfm3-npu-tool"
ENV_FILE="${TOOL_DIR}/etc/lfm3-npu-env.sh"

# Source environment if available
if [[ -f "${ENV_FILE}" ]]; then
    source "${ENV_FILE}"
fi

# Default environment (overrides from env file)
export LD_LIBRARY_PATH="/tmp/lib:/opt/xilinx/xrt/lib:${LD_LIBRARY_PATH:-}"
export XILINX_XRT="${XILINX_XRT:-/opt/xilinx/xrt}"
export FLM_MODEL_PATH="${FLM_MODEL_PATH:-/home/scott/.config/flm/models}"
export TMPDIR="${TMPDIR:-/media/scott/data/tmp}"
export XLNX_CACHE_DIR="${XLNX_CACHE_DIR:-/media/scott/data/vitisai_cache}"

CPU_PYTHON="/media/scott/data/ryzen-ai-venv/bin/python"
CPU_SERVER="${TOOL_DIR}/lib/lfm3_npu/cpu_batched_server.py"
NPU_BINARY="/tmp/flm"

CPU_PORT="${CPU_PORT:-1238}"
NPU_PORT="${NPU_PORT:-1239}"
NPU_HOST="${NPU_HOST:-127.0.0.1}"

PID_DIR="${TOOL_DIR}/var/run"
LOG_DIR="${TOOL_DIR}/var/log"
mkdir -p "${PID_DIR}" "${LOG_DIR}"

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
    log "Starting CPU server on port ${CPU_PORT}..."
    if [[ -f "${PID_DIR}/cpu.pid" ]]; then
        local old_pid=$(cat "${PID_DIR}/cpu.pid" 2>/dev/null || echo "")
        if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
            log "CPU server already running (PID ${old_pid})"
            return 0
        fi
    fi

    nohup "${CPU_PYTHON}" -B "${CPU_SERVER}" --host 0.0.0.0 --port "${CPU_PORT}" \
        > "${LOG_DIR}/cpu.log" 2>&1 &
    echo $! > "${PID_DIR}/cpu.pid"

    if wait_for_port "${CPU_PORT}" "0.0.0.0"; then
        log "CPU server ready on port ${CPU_PORT} (PID $(cat ${PID_DIR}/cpu.pid))"
    else
        log "WARNING: CPU server may still be initializing. Check ${LOG_DIR}/cpu.log"
    fi
}

start_npu() {
    log "Starting NPU (FastFlowLM) server on port ${NPU_PORT}..."
    if [[ ! -x "${NPU_BINARY}" ]]; then
        log "ERROR: FLM binary not found or not executable: ${NPU_BINARY}"
        log "  Download from: https://github.com/ROCm/FastFlowLM/releases/tag/v1.0.12"
        return 1
    fi

    if [[ -f "${PID_DIR}/npu.pid" ]]; then
        local old_pid=$(cat "${PID_DIR}/npu.pid" 2>/dev/null || echo "")
        if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
            log "NPU server already running (PID ${old_pid})"
            return 0
        fi
    fi

    nohup env LD_LIBRARY_PATH="/tmp/lib:/opt/xilinx/xrt/lib:${LD_LIBRARY_PATH:-}" \
        XILINX_XRT="/opt/xilinx/xrt" \
        FLM_MODEL_PATH="/home/scott/.config/flm/models" \
        "${NPU_BINARY}" serve lfm3:1.7b -p "${NPU_PORT}" --host "${NPU_HOST}" \
        > "${LOG_DIR}/npu.log" 2>&1 &
    echo $! > "${PID_DIR}/npu.pid"

    if wait_for_port "${NPU_PORT}" "${NPU_HOST}"; then
        log "NPU server ready on port ${NPU_PORT} (PID $(cat ${PID_DIR}/npu.pid))"
    else
        log "WARNING: NPU server may still be initializing. Check ${LOG_DIR}/npu.log"
    fi
}

status() {
    log "=== Server Status ==="
    for srv in cpu npu; do
        if [[ -f "${PID_DIR}/${srv}.pid" ]]; then
            local pid=$(cat "${PID_DIR}/${srv}.pid")
            if kill -0 "${pid}" 2>/dev/null; then
                log "${srv^} server: RUNNING (PID ${pid})"
            else
                log "${srv^} server: NOT RUNNING (stale PID file)"
            fi
        else
            log "${srv^} server: NOT RUNNING"
        fi
    done
    log ""
    log "CPU health: $(curl -s --max-time 3 http://0.0.0.0:${CPU_PORT}/health 2>/dev/null || echo 'unreachable')"
    local npu_models=$(curl -s --max-time 3 "http://${NPU_HOST}:${NPU_PORT}/v1/models" 2>/dev/null || echo '')
    if [[ -n "${npu_models}" ]]; then
        local npu_lfm=$(echo "${npu_models}" | python3 -c "import sys,json; d=json.load(sys.stdin); lfm=[m for m in d['data'] if 'lfm3' in m['id']]; print(lfm[0]['id'] if lfm else 'not found')" 2>/dev/null || echo 'parse error')
        log "NPU models: ${npu_lfm}"
    else
        log "NPU models: unreachable"
    fi
}

case "${1:-start}" in
    start)
        start_cpu
        start_npu
        log ""
        log "=== Both servers started ==="
        log "  CPU server:  http://0.0.0.0:${CPU_PORT}"
        log "  NPU server:  http://${NPU_HOST}:${NPU_PORT}"
        ;;
    stop)
        log "Stopping CPU server..."
        if [[ -f "${PID_DIR}/cpu.pid" ]]; then
            local pid=$(cat "${PID_DIR}/cpu.pid")
            kill "${pid}" 2>/dev/null || true
            rm -f "${PID_DIR}/cpu.pid"
            log "CPU server stopped."
        fi
        log "Stopping NPU server..."
        if [[ -f "${PID_DIR}/npu.pid" ]]; then
            local pid=$(cat "${PID_DIR}/npu.pid")
            kill "${pid}" 2>/dev/null || true
            rm -f "${PID_DIR}/npu.pid"
            log "NPU server stopped."
        fi
        ;;
    restart)
        "$0" stop
        sleep 3
        "$0" start
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
