#!/bin/bash
# =============================================================================
# LFM3-1.7B Watchdog Cron Wrapper
# Runs the watchdog health check and auto-restarts servers as needed.
# Intended for cron invocation (e.g., every 5 minutes).
# =============================================================================

set -uo pipefail

TOOL_DIR="/media/scott/data/lfm3-npu-tool"
LOG_FILE="${TOOL_DIR}/var/log/watchdog-cron.log"
MAX_UPTIME_HOURS="${LFM3_WATCHDOG_MAX_UPTIME_HOURS:-2}"
MAX_GARBLE_RATIO="${LFM3_WATCHDOG_MAX_GARBLE_RATIO:-0.2}"

mkdir -p "${TOOL_DIR}/var/log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log "=== Watchdog cron run ==="

# Source environment if available
if [[ -f "${TOOL_DIR}/etc/lfm3-npu-env.sh" ]]; then
    source "${TOOL_DIR}/etc/lfm3-npu-env.sh"
fi

export LD_LIBRARY_PATH="/tmp/lib:/opt/xilinx/xrt/lib:${LD_LIBRARY_PATH:-}"
export XILINX_XRT="${XILINX_XRT:-/opt/xilinx/xrt}"
export FLM_MODEL_PATH="${FLM_MODEL_PATH:-/home/scott/.config/flm/models}"
export TMPDIR="${TMPDIR:-/media/scott/data/tmp}"

# Run watchdog once
python3 "${TOOL_DIR}/bin/lfm3-npu" watchdog --once --max-uptime-hours "${MAX_UPTIME_HOURS}" --max-garble-ratio "${MAX_GARBLE_RATIO}" 2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}
log "=== Watchdog cron run complete (exit code: ${EXIT_CODE}) ==="
exit "${EXIT_CODE}"
