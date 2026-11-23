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

## Enterprise Modules Operations

### Model Promotion Pipeline (CI/CD)

Staging → Canary → Production promotion with automated gates:

```bash
# Request a promotion
lfm3-npu promotion request --model-version 2.0.0 --from-stage staging --to-stage canary --auto-approve

# Approve and deploy
lfm3-npu promotion approve <request-id> --approved-by admin
lfm3-npu promotion deploy <request-id>

# Rollback a production model
lfm3-npu promotion rollback 2.0.0 --reason "regression detected"

# View current stages
lfm3-npu promotion stage
lfm3-npu promotion stage 2.0.0

# List all requests
lfm3-npu promotion list
```

Storage: `var/promotion.json`

### Disaster Recovery (Multi-Region)

Region management with health checking, failover, and failback:

```bash
# Add regions
lfm3-npu dr add-region us-east-1 --status primary --endpoint http://use1:1239
lfm3-npu dr add-region us-west-2 --status secondary --endpoint http://usw2:1239

# Check health
lfm3-npu dr check-health
lfm3-npu dr check-health us-east-1

# Set replication state
lfm3-npu dr replication us-west-2 --models --config --keys

# Failover to secondary
lfm3-npu dr failover us-west-2 --reason "primary outage"

# Failback to primary
lfm3-npu dr failback us-east-1

# Status and history
lfm3-npu dr status
lfm3-npu dr history
```

Storage: `var/dr.json`
Service: `lfm3-npu-dr-health.service` / `lfm3-npu-dr-health.timer` (every 1 min)

### Auto-Scaling

Metrics-based scaling with configurable cooldowns:

```bash
# Evaluate scaling decision
lfm3-npu autoscale evaluate

# View status and current policy
lfm3-npu autoscale status

# Update policy
lfm3-npu autoscale policy --min-replicas 2 --max-replicas 20 --target-cpu 70

# Manual scaling
lfm3-npu autoscale scale-up
lfm3-npu autoscale scale-down

# View history
lfm3-npu autoscale history --limit 20
```

Storage: `var/autoscaler.json`
Service: `lfm3-npu-autoscale.service` / `lfm3-npu-autoscale.timer` (every 5 min)

### A/B Testing

Create and manage A/B tests with traffic splitting:

```bash
# Create a test
lfm3-npu abtest create test-001 "v2 vs v1" lfm3-1.7b-v1 lfm3-1.7b-v2 --traffic-split 0.2

# Start the test
lfm3-npu abtest start test-001

# Route a user to a variant (consistent hash-based)
lfm3-npu abtest variant test-001 user-123

# Record metrics
lfm3-npu abtest record test-001 treatment --requests 100 --conversions 15 --latency 55

# Get results
lfm3-npu abtest results test-001

# Analyze and get recommendation
lfm3-npu abtest analyze test-001

# Complete and promote winner
lfm3-npu abtest complete test-001 --winner treatment
lfm3-npu abtest promote test-001
```

Storage: `var/ab_tests.json`

### Cost Tracking & Billing

```bash
# Calculate cost for a tenant
lfm3-npu cost calculate --tenant tenant-1 --input-tokens 1000 --output-tokens 500

# Record cost
lfm3-npu cost record --tenant tenant-1 --tokens 1500 --usd 0.03

# Generate report
lfm3-npu cost report --tenant tenant-1

# Generate invoice
lfm3-npu cost invoice --tenant tenant-1 --month 2026-11
```

Storage: `var/cost_tracking.json`

### API Key Rotation

Configurable rotation policies (age, usage, cost):

```bash
# View rotation policy
lfm3-npu rotation policy

# Check if rotation is needed
lfm3-npu rotation check

# Force rotate a key
lfm3-npu rotation rotate <key_id>

# View rotation history
lfm3-npu rotation history
```

Storage: `var/key_rotation.json`

### Security Scanning

SAST, secrets detection, and dependency vulnerability scanning:

```bash
# Run security scan
lfm3-npu security scan --report-format json

# View results
lfm3-npu security report
```

Storage: `var/security_scan.json`
Service: `lfm3-npu-security-scan.service` / `lfm3-npu-security-scan.timer` (daily at 02:00)

### Self-Service API Key Management

```bash
# Create a key
lfm3-npu keys create --role operator --label "team-a"

# List keys
lfm3-npu keys list

# Get key info
lfm3-npu keys get <key_id>

# Update a key
lfm3-npu keys update <key_id> --role viewer

# Revoke a key
lfm3-npu keys revoke <key_id>

# Rotate a key
lfm3-npu keys rotate <key_id>

# View key usage
lfm3-npu keys usage <key_id>
```

### Multi-Tenancy

Per-tenant quotas, rate limits, and usage tracking:

```bash
# List tenants
lfm3-npu tenant list

# Get tenant details
lfm3-npu tenant get <tenant_id>

# Set tenant quota
lfm3-npu tenant quota <tenant_id> --max-tokens 1000000

# View tenant usage
lfm3-npu tenant usage <tenant_id>
```

### Load Balancer (port 1240)

Health-based failover between CPU and NPU backends:

```bash
# Start load balancer
lfm3-npu lb start --port 1240 --primary npu --secondary cpu

# Check status
lfm3-npu lb status

# Stop load balancer
lfm3-npu lb stop
```

### Configuration Validation & Hot-Reload

```bash
# Validate configuration against schema
lfm3-npu config-validate

# Hot-reload configuration
lfm3-npu config-reload

# Feature flags
lfm3-npu feature-flags list
lfm3-npu feature-flags enable <flag>
lfm3-npu feature-flags disable <flag>
```

### Audit Logs

SIEM-ready audit logs with risk scoring:

```bash
# View audit logs
lfm3-npu audit list

# Export audit log
lfm3-npu audit export --format json
```

### Backup & Restore

```bash
# Create backup
lfm3-npu backup create --name weekly-2026-11-23

# List backups
lfm3-npu backup list

# Restore from backup
lfm3-npu backup restore weekly-2026-11-23

# Delete backup
lfm3-npu backup delete weekly-2026-11-23
```

All backups include SHA-256 integrity verification and automatic pruning.

### SLO (Service-Level Objective) Tracking

Monitor availability and latency SLOs with error budget tracking:

```bash
# Take an SLO snapshot
lfm3-npu slo snapshot

# SLO snapshot with custom provider/model labels
lfm3-npu slo snapshot --provider npu --model lfm3-1.7b-npu
```

The SLO tracker records:
- Availability percentage vs. target (default 99.9%)
- P95 latency vs. target (default 5000ms)
- Error budget consumed and remaining
- Whether the overall SLO is met

Metrics are exposed via Prometheus at the `/metrics` endpoint.

### OpenTelemetry Tracing

Distributed tracing for prefill/decode phases:

```bash
# Check tracing status
lfm3-npu tracing status

# Initialize tracing with OTLP endpoint (e.g., Jaeger)
lfm3-npu tracing init --service-name lfm3-inference --otlp-endpoint http://localhost:4318

# Initialize without OTLP (no-op mode for local testing)
lfm3-npu tracing init
```

Traces are exported via OTLP to compatible collectors (Jaeger, Tempo,
OpenTelemetry Collector). Spans are created for:
- Full inference requests (`trace_request`)
- Prefill phase (`trace_prefill`)
- Decode phase (`trace_decode`)


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

## Recovery Procedures

### Full Server Recovery

If both servers are down or unhealthy, use the recover command:

```bash
# Stop all services, start them, wait for health, and validate
lfm3-npu recover
```

This will:
1. Stop all LFM3 services (CPU, NPU, metrics, quality monitor)
2. Kill any remaining processes
3. Start all services in order
4. Wait up to 60 seconds for servers to become healthy
5. Run startup validation

### Individual Server Recovery

```bash
# Recover CPU server only
lfm3-npu server restart cpu

# Recover NPU server only
lfm3-npu server restart npu

# Recover metrics exporter
systemctl --user restart lfm3-npu-metrics.service

# Recover quality monitor
systemctl --user restart lfm3-npu-quality-monitor.service
```

## Quality Monitor

The quality monitor continuously checks generation quality on both servers
and auto-restarts when degradation is detected.

### Features

- **Quality degradation detection**: Conversation loops, numbered list explosions,
  repeated characters, wrong math answers
- **NPU reasoning trace detection**: Verifies NPU produces reasoning_content (Qwen XML format)
- **Server unreachable detection**: Restarts servers that stop responding to health endpoints
- **Restart cooldown**: Prevents restart loops (default 120s between restarts)
- **State file**: `/media/scott/data/lfm3-npu-tool/var/log/quality-monitor-state.json`
  tracks restart counts and check statistics

### Manual Quality Check

```bash
# One-shot check with auto-restart
lfm3-npu quality-monitor --once --fix --verbose

# Run continuously with auto-restart
lfm3-npu quality-monitor --fix --verbose
```

## Disk Monitoring

Disk usage is monitored via Prometheus metrics:

- `lfm3_disk_used_percent` — overall disk usage percentage
- Alerts trigger at 85% (warning) and 95% (critical)

Use `/media/scott/data` for large files (avoid /tmp which has 15G quota).

## Startup Validation

Validates all services after boot:

```bash
lfm3-npu validate
```

Checks systemd services, timers, server health, metrics exporter, NPU generation
quality, and disk space. Runs automatically via systemd timer at boot (1min delay).

## Prometheus Alerts

The following alerts are configured in `etc/prometheus-rules-lfm3-npu.yml`:

| Alert | Condition | Severity |
|-------|-----------|----------|
| LFM3NPUServerDown | NPU unreachable 30s | critical |
| LFM3NPUModelUnavailable | lfm3 model missing 60s | critical |
| LFM3NPURequestErrors | Error rate > 0 for 5m | warning |
| LFM3NPUDecodeSlow | Decode < 15 tok/s for 5m | warning |
| LFM3NPUMemoryHigh | RSS > 3500 MB for 5m | warning |
| LFM3NPUServerRestarted | Uptime reset in 1h | warning |
| LFM3CPUServerDown | CPU unreachable 30s | critical |
| LFM3CPUDecodeSlow | Avg decode > 200ms for 5m | warning |
| LFM3CPUUptimeHigh | Uptime > 7200s | warning |
| LFM3CPUOverloaded | Free slots = 0 for 5m | warning |
| LFM3DiskUsageHigh | Disk > 85% for 5m | warning |
| LFM3DiskUsageCritical | Disk > 95% for 2m | critical |
| LFM3NPUUnresponsiveRestarts | NPU restarted by QM in 1h | warning |
| LFM3CPUUnresponsiveRestarts | CPU restarted by QM in 1h | warning |

## Systemd User Services

| Service | Description |
|---------|-------------|
| lfm3-npu-cpu.service | CPU inference server (port 1238) |
| lfm3-npu-npu.service | NPU inference server (port 1239) |
| lfm3-npu-metrics.service | Prometheus metrics exporter (port 9190) |
| lfm3-npu-quality-monitor.service | Quality monitor with auto-restart |

| Timer | Schedule | Purpose |
|-------|----------|---------|
| lfm3-npu-health.timer | Every 1 min | Unified health check |
| lfm3-npu-watchdog.timer | Every 5 min | Server responsiveness + CPU uptime |
| lfm3-npu-doctor.timer | Hourly | Full system diagnosis |
| lfm3-npu-quality-monitor.timer | Every 5 min | Quality degradation detection |
| lfm3-npu-benchmark.timer | Hourly | Performance benchmarking |
| lfm3-npu-startup-validate.timer | Boot + 1min | Post-boot validation |

### Enable all services:

```bash
systemctl --user enable lfm3-npu-cpu.service
systemctl --user enable lfm3-npu-npu.service
systemctl --user enable lfm3-npu-metrics.service
systemctl --user enable lfm3-npu-quality-monitor.service
systemctl --user enable lfm3-npu-health.timer
systemctl --user enable lfm3-npu-watchdog.timer
systemctl --user enable lfm3-npu-doctor.timer
systemctl --user enable lfm3-npu-quality-monitor.timer
systemctl --user enable lfm3-npu-benchmark.timer
systemctl --user enable lfm3-npu-startup-validate.timer
```

## Metrics Exporter (Port 9190)

Exposes Prometheus metrics:

- `lfm3_npu_up` — NPU server reachable
- `lfm3_npu_total_models` — total models available
- `lfm3_npu_lfm3_model_available` — lfm3:1.7b available
- `lfm3_npu_process_cpu_percent` — NPU server CPU %
- `lfm3_npu_process_memory_rss_bytes` — NPU server RSS memory
- `lfm3_npu_process_memory_vms_bytes` — NPU server VMS memory
- `lfm3_npu_process_threads` — NPU server thread count
- `lfm3_npu_request_duration_seconds` — request duration histogram
- `lfm3_npu_ttft_seconds` — time to first token
- `lfm3_npu_decode_tokens_per_second` — decode speed
- `lfm3_npu_uptime_seconds` — exporter uptime
- `lfm3_cpu_up` — CPU server reachable
- `lfm3_cpu_uptime_seconds` — CPU server uptime
- `lfm3_cpu_avg_decode_ms` — CPU avg decode time
- `lfm3_cpu_active_slots` — CPU active request slots
- `lfm3_cpu_free_slots` — CPU free request slots
- `lfm3_disk_total_bytes` — total disk space
- `lfm3_disk_used_bytes` — used disk space
- `lfm3_disk_free_bytes` — free disk space
- `lfm3_disk_used_percent` — disk usage %
- `lfm3_qm_cpu_restarts` — CPU restarts by quality monitor
- `lfm3_qm_npu_restarts` — NPU restarts by quality monitor
- `lfm3_qm_npu_unresponsive_restarts` — NPU unresponsive restarts
- `lfm3_qm_last_check` — last QM check timestamp
- `lfm3_qm_cpu_passed` — CPU quality checks passed
- `lfm3_qm_cpu_total` — total CPU quality checks
- `lfm3_qm_npu_passed` — NPU quality checks passed
- `lfm3_qm_npu_total` — total NPU quality checks

Configure Prometheus to scrape: `http://localhost:9190/metrics`

## CLI Commands

| Command | Description |
|---------|-------------|
| `lfm3-npu generate` | Generate text via HTTP API |
| `lfm3-npu server` | Start/stop/restart/check servers |
| `lfm3-npu bench` | Benchmark inference performance |
| `lfm3-npu perf` | Performance profiling across concurrency |
| `lfm3-npu health` | Check server health |
| `lfm3-npu convert` | Convert model to NPU-compatible format |
| `lfm3-npu config` | Show configuration |
| `lfm3-npu status` | Show system status |
| `lfm3-npu compare` | Compare CPU vs NPU inference |
| `lfm3-npu compare-servers` | Compare live CPU vs NPU via HTTP |
| `lfm3-npu doctor` | Diagnose system health (--fix to auto-repair) |
| `lfm3-npu watchdog` | Run server watchdog |
| `lfm3-npu fleet-monitor` | Fleet monitoring |
| `lfm3-npu smoke-test` | Validate all servers (--verbose, --fix) |
| `lfm3-npu quality-monitor` | Monitor quality with auto-restart (--once, --fix, --verbose) |
| `lfm3-npu benchmark-cron` | Run periodic benchmark |
| `lfm3-npu version-check` | Check FastFlowLM updates (--upgrade to install) |
| `lfm3-npu validate` | Run startup validation |
| `lfm3-npu recover` | Recover all servers (stop, start, verify) |

## File Locations

| Path | Purpose |
|------|---------|
| `/tmp/flm` | FLM v1.0.12 binary |
| `/tmp/lib/liblfm3_npu.so` | LFM3 NPU engine library |
| `/home/scott/.config/flm/models/LFM3-1.7B-NPU2/` | FLM model files |
| `/media/scott/data/lfm3_oga_export/model.onnx` | CPU ONNX model |
| `/media/scott/data/lfm3-npu-tool/` | Tooling repository |
| `/media/scott/data/lfm3-npu-tool/var/log/` | Server logs |
| `/media/scott/data/lfm3-npu-tool/var/log/quality-monitor-state.json` | QM state for metrics |
| `/media/scott/data/lfm3-npu-tool/var/run/` | PID files |
| `/home/scott/.config/systemd/user/` | User systemd services |
| `/home/scott/.config/lfm3-npu/config.yaml` | Tooling configuration |

## Emergency Contacts / Escalation

1. CPU server degraded → `lfm3-npu server restart cpu` (or wait for quality monitor)
2. NPU server down → `lfm3-npu server restart npu`
3. Both down → `lfm3-npu recover`
4. NPU hardware issues → Check `xrt-smi examine`, verify `/dev/accel0`
5. Persistent issues → Check `lfm3-npu doctor` output
6. Disk full → Free space, use `/media/scott/data` for large files
7. NPU upgrade available → `lfm3-npu version-check --upgrade`
