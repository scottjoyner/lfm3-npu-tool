# LFM3-1.7B NPU Inference — Operations Runbook

This runbook covers deployment, monitoring, troubleshooting, and maintenance of
the LFM3-1.7B dual-server (CPU + NPU) inference setup.

## Architecture

| Component | Port | Technology | Purpose |
|-----------|------|------------|---------|
| CPU Server | 1238 | ONNX Runtime CPUExecutionProvider | Fallback / full-quality when fresh |
| NPU Server | 1239 | FastFlowLM v1.0.12 (liblfm3_npu.so) | Primary — full quality on AMD Strix Halo NPU |
| Metrics Exporter | 9190 | Prometheus text format | NPU server metrics for monitoring |

## Quick Start

### Start both servers

```bash
# Option 1: Unified script
./start-all.sh start

# Option 2: User systemd (no sudo required)
systemctl --user start lfm3-npu-cpu.service
systemctl --user start lfm3-npu-npu.service
systemctl --user start lfm3-npu-metrics.service

# Option 3: Manual
export LD_LIBRARY_PATH=/tmp/lib:/opt/xilinx/xrt/lib
export XILINX_XRT=/opt/xilinx/xrt
export FLM_MODEL_PATH=/home/scott/.config/flm/models
/tmp/flm serve lfm3:1.7b -p 1239 --host 127.0.0.1 &
```

### Verify servers

```bash
# CPU health
curl http://localhost:1238/health

# NPU models
curl http://localhost:1239/v1/models

# NPU metrics (Prometheus)
curl http://localhost:9190/metrics

# Generate text via NPU
lfm3-npu generate "What is 2+2?" --server npu --max-tokens 30
```

### Stop both servers

```bash
./start-all.sh stop
# or
systemctl --user stop lfm3-npu-cpu.service lfm3-npu-npu.service lfm3-npu-metrics.service
```

## Daily Operations

### Check system health

```bash
# Full system diagnosis
lfm3-npu doctor

# Fleet status (includes NPU device metrics)
lfm3-fleet-monitor --once

# Server status with process metrics
lfm3-npu status
```

### Real-time monitoring dashboard

```bash
lfm3-monitor
```

Refreshes every 5 seconds with color-coded server health, uptime, and decode metrics.

### Prometheus metrics

The metrics exporter runs on port 9190 and exposes:
- `lfm3_npu_up` — whether NPU server is reachable
- `lfm3_npu_total_models` — total models available
- `lfm3_npu_lfm3_model_available` — whether lfm3:1.7b is available
- `lfm3_npu_process_cpu_percent` — NPU server CPU usage
- `lfm3_npu_process_memory_rss_bytes` — NPU server RSS memory
- `lfm3_npu_process_memory_vms_bytes` — NPU server VMS memory
- `lfm3_npu_process_threads` — NPU server thread count
- `lfm3_npu_request_duration_seconds` — request duration histogram

Configure Prometheus to scrape: `http://localhost:9190/metrics`

## Benchmarking & Testing

### Quality benchmark (0-100 scoring)

```bash
python3 scripts/quality_benchmark.py --server both
```

Tests arithmetic, factual, creative, explanation, and instruction prompts.
NPU typically scores ~97/100 vs CPU ~61/100.

### Concurrency stress test

```bash
python3 scripts/concurrency_stress.py --server both --concurrency 4 --requests 20
```

Measures success rate, TTFT, decode speed, and degradation under concurrent load.

### Watchdog (auto-restart)

```bash
# Single check — restarts CPU if uptime > 24h
lfm3-npu watchdog --once --max-uptime-hours 24

# Run as daemon
lfm3-npu watchdog --daemon --interval 300 --max-uptime-hours 24
```

Watchdog runs automatically via cron every 5 minutes.

## Troubleshooting

### NPU server unreachable

1. Check process: `ps aux | grep flm`
2. Check logs: `cat /media/scott/data/lfm3-npu-tool/var/log/npu.log`
3. Restart: `lfm3-npu server restart npu`
4. Verify FLM binary: `ls -la /tmp/flm /tmp/lib/liblfm3_npu.so`
5. Run doctor: `lfm3-npu doctor`

### CPU server unreachable

1. Check process: `ps aux | grep cpu_batched_server`
2. Check logs: `cat /media/scott/data/lfm3-npu-tool/var/log/cpu.log`
3. Restart: `lfm3-npu server restart cpu`
4. Verify Python venv: `/media/scott/data/ryzen-ai-venv/bin/python --version`

### CPU quality degradation

The CPU server (ONNX Runtime) degrades after extended uptime (~24h+).
Symptoms: repetitive output, conversation loops, garbled text.

Fix: Restart the CPU server.
```bash
lfm3-npu server restart cpu
```
Or let the watchdog handle it automatically (cron every 5 min).

### NPU server high memory

NPU server typically uses ~2.5-3GB RSS. Alert triggers at >4GB.
If memory grows unbounded, restart: `lfm3-npu server restart npu`

### Metrics exporter not running

1. Check process: `ps aux | grep npu_metrics_exporter`
2. Check logs: `journalctl --user -u lfm3-npu-metrics.service`
3. Restart: `systemctl --user restart lfm3-npu-metrics.service`
4. Verify endpoint: `curl http://localhost:9190/metrics`

### Disk space issues

Use `/media/scott/data` for large files (avoid /tmp which has 15G quota).
Check: `df -h /media/scott/data /tmp`

## Systemd User Services

Enable on boot:
```bash
systemctl --user enable lfm3-npu-cpu.service
systemctl --user enable lfm3-npu-npu.service
systemctl --user enable lfm3-npu-metrics.service
```

Check status:
```bash
systemctl --user status lfm3-npu-cpu.service
systemctl --user status lfm3-npu-npu.service
systemctl --user status lfm3-npu-metrics.service
```

View logs:
```bash
journalctl --user -u lfm3-npu-cpu.service
journalctl --user -u lfm3-npu-npu.service
journalctl --user -u lfm3-npu-metrics.service
```

## Cron Jobs

View: `crontab -l`

| Schedule | Command |
|----------|---------|
| `*/5 * * * *` | Watchdog (server health + CPU max-uptime restart) |
| `0 * * * *` | Doctor (system health diagnosis) |
| `@reboot` | Auto-start both servers via start-all.sh |

Logs:
- `/media/scott/data/lfm3-npu-tool/var/log/watchdog-cron.log`
- `/media/scott/data/lfm3-npu-tool/var/log/doctor-cron.log`
- `/media/scott/data/lfm3-npu-tool/var/log/boot-start.log`

## Log Rotation

Logrotate config: `etc/logrotate-lfm3-npu.conf`

Run manually: `logrotate -f etc/logrotate-lfm3-npu.conf`

Rotates: `var/log/*.log`, `watchdog-cron.log`, `doctor-cron.log`, `boot-start.log`
Keeps 7 daily rotations, 4 weekly rotations, compressed.

## File Locations

| Path | Purpose |
|------|---------|
| `/tmp/flm` | FLM v1.0.12 binary |
| `/tmp/lib/liblfm3_npu.so` | LFM3 NPU engine library |
| `/home/scott/.config/flm/models/LFM3-1.7B-NPU2/` | FLM model files |
| `/media/scott/data/lfm3_oga_export/model.onnx` | CPU ONNX model |
| `/media/scott/data/lfm3-npu-tool/` | Tooling repository |
| `/media/scott/data/lfm3-npu-tool/var/log/` | Server logs |
| `/media/scott/data/lfm3-npu-tool/var/run/` | PID files |
| `/home/scott/.config/systemd/user/` | User systemd services |
| `/home/scott/.config/lfm3-npu/config.yaml` | Tooling configuration |

## Emergency Contacts / Escalation

1. CPU server degraded → `lfm3-npu server restart cpu`
2. NPU server down → `lfm3-npu server restart npu`
3. Both down → `./start-all.sh start`
4. NPU hardware issues → Check `xrt-smi examine`, verify `/dev/accel0`
5. Persistent issues → Check `lfm3-npu doctor` output
