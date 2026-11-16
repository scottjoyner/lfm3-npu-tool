# LFM3-1.7B NPU Inference Tooling

Complete tooling suite for running LFM3-1.7B inference on the AMD XDNA2 NPU.

## Status: NPU Inference Fully Working ✅

**LFM3-1.7B runs fully on the NPU via FastFlowLM v1.0.12+** (released Nov 5, 2026).
All 24 layers + LM head execute on the AMD Strix Halo NPU. No VitisAI EP workaround needed.

| Component | Port | Provider | Quality | Status |
|-----------|------|----------|---------|--------|
| FLM NPU Server | 1239 | liblfm3_npu.so | **Full quality** | Production-ready ✅ |
| VitisAI EP (stub) | 1239 | VitisAIExecutionProvider | Degraded | Deprecated |

## Quick Start

### Install FastFlowLM (v1.0.12+)

```bash
# Download and extract FLM v1.0.12 for Linux
cd /tmp
curl -sL -o fastflowlm_linux.tar.gz \
  https://github.com/ROCm/FastFlowLM/releases/download/v1.0.12/fastflowlm_1.0.12_linux.tar.gz
tar xzf fastflowlm_linux.tar.gz

# Run LFM3-1.7B on NPU (interactive)
export LD_LIBRARY_PATH=/tmp/lib:/opt/xilinx/xrt/lib
export XILINX_XRT=/opt/xilinx/xrt
/tmp/flm run lfm3:1.7b

# Start NPU server (OpenAI-compatible API on port 1239)
/tmp/flm serve lfm3:1.7b -p 1239 --host 127.0.0.1
```

### CPU Server (Full Quality, Port 1238)

The existing CPU inference server remains available at port 1238:

```bash
cd /media/scott/data/lfm3-npu-server
# Server is already running (~4.7 days uptime)
curl http://localhost:1238/health
```

## NPU Server API Example

```bash
curl -s -X POST http://localhost:1239/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"lfm3:1.7b","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":100}'
```

**Expected output:**
```
"2 + 2 equals **4**."
```
- Prefill: ~69 tok/s
- Decode: ~27 tok/s
- TTFT: ~0.25s

## Performance (LFM3-1.7B on Strix Halo NPU)

| Metric | Value |
|--------|-------|
| Prefill speed | ~69 tok/s |
| Decode speed | ~27 tok/s |
| Time to first token | ~0.25s |
| Model load time | ~2.9s |
| Layers on NPU | All 24 + LM head |
| Quantization | Q4_0 |

## Directory Layout

```
lfm3-npu-tool/
├── bin/                          # CLI tools
├── etc/                          # Config files
│   ├── lfm3-npu-config.yaml      # Tooling configuration
│   ├── lfm3-npu-env.sh           # Environment setup
│   ├── lfm3-npu-cpu.service      # Systemd service (CPU)
│   ├── lfm3-npu-npu.service      # Systemd service (NPU)
│   ├── lfm3-npu-metrics.service  # Systemd service (metrics exporter)
│   └── logrotate-lfm3-npu.conf   # Log rotation config
├── examples/                     # Example scripts
├── scripts/                      # Utility scripts
│   ├── quality_benchmark.py      # Quality-scored benchmark (0-100)
│   ├── concurrency_stress.py     # Concurrent stress test
│   ├── watchdog-cron.sh          # Cron wrapper for watchdog
│   └── npu_metrics_exporter.py   # Prometheus metrics exporter (port 9190)
├── tests/                        # Tooling test suite (5/5 passing)
├── var/                          # Logs and PID files
├── start-all.sh                  # Unified startup script (both servers)
├── install.sh                    # Installation script
├── OPERATIONS_RUNBOOK.md         # Operations runbook
├── README.md                     # This file
├── FINAL_SUMMARY.md              # Overall project summary
├── EXHAUSTIVE_NPU_APPROACHES.md  # All VitisAI EP approaches attempted
├── STUB_VAIML_README.md          # Stub libvaiml.so documentation
├── NPU_ACCESSIBILITY_RESULTS.md  # NPU hardware accessibility proof
├── VAIP_NPU_PATH.md              # VAIP config bypass investigation
└── VAIML_COMPILE_X2_V2_ATTEMPT.md # libvaiml2.so compilation attempt
```

## Key Files

| File | Purpose |
|------|---------|
| `/tmp/flm` | FLM v1.0.12 binary (LFM3 NPU support) |
| `/tmp/lib/liblfm3_npu.so` | LFM3 NPU engine library |
| `/home/scott/.config/flm/models/LFM3-1.7B-NPU2/` | Downloaded model files |
| `/media/scott/data/lfm3-npu-server/` | CPU server (port 1238) |
| Port 1239 | NPU server (FLM LFM3) |
| Port 9190 | NPU Prometheus metrics exporter |

## Background: VitisAI EP Approaches Exhausted

All 13+ approaches to run LFM3 via VitisAI ExecutionProvider were attempted and documented in `EXHAUSTIVE_NPU_APPROACHES.md`. All were blocked by:

1. **`libvaiml.so` SIGSEGV** — Protobuf crash during `dlopen()` (AMD proprietary, unfixable from user space)
2. **No Strix Halo xclbin** — Existing SDK xclbins target `xilinx_v1_ipu_0_0` (bank 131071), not Strix (banks 0/1)
3. **No Strix Halo platform file** — No `.xpfm`/`.xsa` exists for `v++ --package`
4. **No ONNX→xclbin pipeline** — xcompiler finds 0 ops in LFM3

FastFlowLM v1.0.12 bypasses all of these by using its own `liblfm3_npu.so` engine directly with XRT.

## Automation & Monitoring

### Cron-based Watchdog (Auto-restart)

The watchdog runs every 5 minutes via cron to auto-restart unresponsive servers and prevent CPU quality degradation after extended uptime:

```bash
# View crontab
crontab -l

# Manual watchdog run (restarts CPU if uptime > 24h)
lfm3-npu watchdog --once --max-uptime-hours 24

# Manual system diagnosis
lfm3-npu doctor
```

Cron jobs:
- `*/5 * * * *` — Watchdog (server health + CPU max-uptime restart)
- `0 * * * *` — Doctor (system health diagnosis)
- `@reboot` — Auto-start both servers on boot

### Start All Servers

```bash
./start-all.sh start   # Start both CPU (1238) and NPU (1239) servers
./start-all.sh stop    # Stop both servers
./start-all.sh status  # Check both servers
./start-all.sh restart # Restart both servers
```

### User Systemd Services (No Sudo Required)

Services installed at `~/.config/systemd/user/`:

```bash
# Enable on boot
systemctl --user enable lfm3-npu-cpu.service
systemctl --user enable lfm3-npu-npu.service
systemctl --user enable lfm3-npu-metrics.service

# Start
systemctl --user start lfm3-npu-cpu.service
systemctl --user start lfm3-npu-npu.service
systemctl --user start lfm3-npu-metrics.service

# Status
systemctl --user status lfm3-npu-cpu.service
systemctl --user status lfm3-npu-npu.service
systemctl --user status lfm3-npu-metrics.service

# Logs
journalctl --user -u lfm3-npu-cpu.service
journalctl --user -u lfm3-npu-npu.service
journalctl --user -u lfm3-npu-metrics.service
```

### NPU Prometheus Metrics Exporter (port 9190)

```bash
# Start via systemd
systemctl --user start lfm3-npu-metrics.service

# Or run manually
python3 scripts/npu_metrics_exporter.py --port 9190 --npu-port 1239

# Scrape metrics
curl http://localhost:9190/metrics
```

Exposes: `lfm3_npu_up`, `lfm3_npu_total_models`, `lfm3_npu_lfm3_model_available`,
`lfm3_npu_process_cpu_percent`, `lfm3_npu_process_memory_rss_bytes`,
`lfm3_npu_process_memory_vms_bytes`, `lfm3_npu_process_threads`,
`lfm3_npu_request_duration_seconds`, `lfm3_npu_requests_total`,
`lfm3_npu_requests_errors_total`, `lfm3_npu_ttft_seconds`,
`lfm3_npu_decode_tokens_per_second`, `lfm3_npu_uptime_seconds`,
`lfm3_npu_in_warmup`, `lfm3_npu_last_request_duration`,
`lfm3_npu_last_request_success`.

Configure Prometheus to scrape: `http://localhost:9190/metrics`

### Prometheus Alerting Rules

Installed at `~/.config/prometheus-rules-lfm3-npu.yml`. Includes alerts for:
- NPU server unreachable
- LFM3 model unavailable
- Request errors
- Decode speed degraded
- Memory high
- Server restarted

### Grafana Dashboard

Installed at `~/.config/grafana-dashboard-lfm3-npu.json`. Includes panels for:
- Server status, process CPU/memory/threads
- Request count, errors, TTFT, decode speed
- Request duration histogram
- Exporter uptime, warmup status, last request duration/success

### Prometheus Scrape Config

Installed at `~/.config/prometheus-lfm3.yml`. Scrapes `localhost:9190/metrics`.

### Alertmanager Config

Installed at `~/.config/alertmanager-lfm3.yml`. Email and Slack alert routing.

### Smoke Test & CI Validation

Quick validation of all servers:

```bash
# CLI smoke test
lfm3-npu smoke-test --verbose

# Or run directly
python3 scripts/smoke_test.py --verbose

# CI test suite (for automated validation)
python3 scripts/ci_test.py
```

The smoke test checks:
- CPU server `/health` endpoint
- NPU server `/v1/models` endpoint
- Metrics exporter on port 9190
- Generation quality on both servers (correct answers, no degradation)
- Watchdog can run successfully

### Quality Monitor (Auto-Restart on Degradation)

The CPU server (ONNX Runtime) suffers from output quality degradation after
extended uptime (~20 min). The quality monitor continuously checks generation
quality and auto-restarts the CPU server when degradation is detected:

```bash
# CLI quality monitor (runs continuously with auto-restart)
lfm3-npu quality-monitor --fix --verbose

# One-shot check (for CI/cron)
lfm3-npu quality-monitor --once --fix

# Or run directly
python3 scripts/quality_monitor.py --once --fix --verbose
```

The quality monitor:
- Tests generation quality on both CPU and NPU servers every interval
- Detects conversation loops, numbered list explosions, repeated characters, wrong math
- Auto-restarts CPU server when quality falls below threshold
- Respects a restart cooldown (default 10 min) to prevent restart loops

### Systemd Services (Quality Monitor)

```bash
# Enable and start quality monitor
systemctl --user enable lfm3-npu-quality-monitor.service
systemctl --user enable lfm3-npu-quality-monitor.timer
systemctl --user start lfm3-npu-quality-monitor.timer

# Status
systemctl --user status lfm3-npu-quality-monitor.service
systemctl --user status lfm3-npu-quality-monitor.timer
```

### Operations Runbook

See `OPERATIONS_RUNBOOK.md` for complete deployment, monitoring,
troubleshooting, and maintenance procedures.

### Real-time Monitoring Dashboard

```bash
lfm3-monitor
```

Refreshes every 5 seconds with color-coded server health, uptime, and decode metrics.

### Quality Benchmark

```bash
python3 scripts/quality_benchmark.py --server both
```

Scores output quality 0-100 across arithmetic, factual, creative, explanation, and instruction test cases. NPU consistently scores ~97/100 vs CPU ~61/100.

### Concurrency Stress Test

```bash
python3 scripts/concurrency_stress.py --server both --concurrency 4 --requests 20
```

Sends N concurrent requests and measures success rate, TTFT, decode speed, and degradation.

### Log Rotation

```bash
# Copy logrotate config (requires root)
sudo cp etc/logrotate-lfm3-npu.conf /etc/logrotate.d/lfm3-npu
# Or run manually:
logrotate -f etc/logrotate-lfm3-npu.conf
```

## License

Internal use only.