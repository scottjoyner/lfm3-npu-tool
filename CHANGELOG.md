# Changelog

All notable changes to the LFM3-1.7B NPU Inference Tooling project are documented in this file.

## [Unreleased]

### Added
- `lfm3-npu recover` command — stop all services, kill stale processes, start all services, wait for health, run validation
- `lfm3-npu version-check --upgrade` — download and install latest FastFlowLM release with auto-rollback
- `lfm3-npu doctor --fix` — auto-fix common issues (restart unreachable servers)
- Disk usage metrics (`lfm3_disk_*`) for `/`, `/media/scott/data`, `/tmp`
- Quality monitor metrics (`lfm3_qm_*`) — restart counts, check statistics
- Quality monitor unresponsive server detection — restarts servers that stop responding
- CPU Prometheus alerts: decode slow, uptime high, overloaded, unresponsive restarts
- NPU Prometheus alerts: unresponsive restarts
- Grafana CPU panels: status, decode time, active/free slots (4 panels)
- Grafana disk panels: usage %, free GB (2 panels)
- Quality monitor state file for Prometheus metrics export
- Startup validation script + systemd service/timer (runs at boot + 1min)
- Benchmark cron script + systemd service/timer (runs hourly)
- Deploy script (`deploy.sh`) — one-command setup for fresh machines
- CI test suite (`scripts/ci_test.py`)
- Dockerfile for containerized deployment

### Changed
- Watchdog defaults: max_uptime_hours 24→2, max_garble_ratio 0.3→0.2
- Quality monitor default restart cooldown: 600s→120s
- Metrics exporter: `Restart=always` instead of `Restart=on-failure`
- Metrics exporter: added `allow_reuse_address` to TCPServer
- NPU service: added `ExecStartPre` to kill stale FLM processes
- Metrics service: added CPU port/host args, disk paths arg
- Watchdog: fixed NPU health check to use `/v1/models` instead of `/health`
- Dockerfile: removed broken multi-stage build reference

### Fixed
- Grafana dashboard JSON syntax error (extra comma/comment)
- Watchdog timer was disabled — now enabled and started
- Metrics service port conflict on restart

## [1.0.0] - 2026-11-15

### Added
- Complete LFM3-1.7B NPU inference deployment on AMD XDNA2 NPU via FastFlowLM v1.0.12
- CPU server (ONNX Runtime, port 1238) and NPU server (FastFlowLM, port 1239)
- Prometheus metrics exporter (port 9190) with 15+ metrics
- Grafana dashboard with 13 panels
- Prometheus alerting rules (7 alerts)
- Alertmanager config with email/Slack routing
- Quality benchmark (0-100 scoring): NPU 97/100 vs CPU 61/100
- Concurrency stress testing
- Unified CLI tooling (14 commands)
- Systemd user services and timers (health, watchdog, doctor)
- Watchdog with CPU quality degradation detection and auto-restart
- Operations runbook
- Logrotate config
- Installation script

### Performance
- NPU: ~28 tok/s decode, ~0.2s TTFT, quality 97/100
- CPU: ~142ms avg decode, quality 61/100
- NPU is ~5x faster end-to-end, ~50x faster TTFT, ~13x faster decode
