#!/bin/bash
# Manual fallback only; never override a healthy server or systemd ownership.
set -eo pipefail
LFM3_CPU_LEGACY_START_AUTHORIZED="${LFM3_CPU_LEGACY_START_AUTHORIZED:-0}"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
TOOL_DIR="/media/scott/data/lfm3-npu-tool"
PYTHON_BIN="/media/scott/data/ryzen-ai-venv/bin/python"
PORT=1238

cpu_healthy() {
  curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | python3 -c '
import json,sys
try:
    d=json.load(sys.stdin)
    ok=(d.get("status")=="healthy" and d.get("provider")=="CPUExecutionProvider"
        and "lfm3" in str(d.get("model","")).lower() and int(d.get("port",0))==1238)
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
'
}
# An existing healthy endpoint always wins. Do not launch a competitor.
if cpu_healthy; then
  echo "CPU endpoint already healthy; no new process started"
  exit 0
fi
if ! command -v ss >/dev/null 2>&1; then
  echo "HOLD: missing port-inspection tool" >&2; exit 3
fi
if ss -H -ltn '( sport = :1238 )' | grep -q .; then
  echo "HOLD: port 1238 occupied by an unverified listener" >&2; exit 3
fi
if ! command -v systemctl >/dev/null 2>&1; then
  echo "HOLD: cannot establish systemd ownership" >&2; exit 3
fi
if systemctl --user is-active --quiet lfm3-npu-cpu.service ||
   systemctl --user is-enabled --quiet lfm3-npu-cpu.service; then
  echo "HOLD: CPU service is managed by systemd" >&2; exit 3
fi
# Explicit manual recovery approval required, never implicit from cron.
if [ "$LFM3_CPU_LEGACY_START_AUTHORIZED" != "1" ]; then
  echo "HOLD: LFM3_CPU_LEGACY_START_AUTHORIZED=1 required" >&2; exit 3
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "HOLD: cannot serialize starts" >&2; exit 3
fi
RUNTIME_DIR="/run/user/$(id -u)"
if [ ! -d "$RUNTIME_DIR" ] || [ ! -w "$RUNTIME_DIR" ]; then
  echo "HOLD: private runtime lock dir unavailable" >&2; exit 3
fi
exec 9>"$RUNTIME_DIR/lfm3-cpu-start.lock"
if ! flock -n 9; then
  echo "HOLD: concurrent manual start" >&2; exit 3
fi
# Double-check after locking. This does not override any independent owner.
if cpu_healthy; then
  echo "CPU endpoint became healthy; no new process started"; exit 0
fi
if ss -H -ltn '( sport = :1238 )' | grep -q . ||
   systemctl --user is-active --quiet lfm3-npu-cpu.service ||
   systemctl --user is-enabled --quiet lfm3-npu-cpu.service; then
  echo "HOLD: CPU port or unit ownership changed" >&2; exit 3
fi
if [ ! -x "$PYTHON_BIN" ] || [ ! -f "$TOOL_DIR/lib/lfm3_npu/cpu_batched_server.py" ]; then
  echo "HOLD: pinned runtime missing" >&2; exit 3
fi
export TMPDIR="/media/scott/data/tmp"
export LD_LIBRARY_PATH="/media/scott/data/ryzen-ai-venv/lib/python3.12/site-packages/flexml/flexml_extras/lib:/media/scott/data/ryzen-ai-venv/lib/python3.12/site-packages/voe/lib:/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH"
mkdir -p "$TOOL_DIR/var/log"
cd "$TOOL_DIR"
nohup "$PYTHON_BIN" -B lib/lfm3_npu/cpu_batched_server.py --host 0.0.0.0 --port 1238 > "$TOOL_DIR/var/log/cpu_batched.log" 2>&1 &
child=$!
for n in 1 2 3 4 5 6; do
  if cpu_healthy; then
    echo "Legacy CPU server confirmed healthy"; exit 0
  fi
  if ! kill -0 "$child" 2>/dev/null; then
    echo "HOLD: CPU server child exited" >&2; exit 3
  fi
  sleep 2
done
echo "HOLD: startup not yet healthy; no process was killed" >&2
exit 3
