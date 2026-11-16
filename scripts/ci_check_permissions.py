#!/usr/bin/env python3
"""Check that executable scripts have the +x permission bit set."""
import os
import stat
import sys

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(SERVER_DIR)

executable_scripts = [
    'bin/lfm3-npu',
    'scripts/smoke_test.py',
    'scripts/ci_test.py',
    'scripts/quality_monitor.py',
    'scripts/benchmark_cron.py',
    'scripts/startup-validate.py',
    'scripts/health_check.py',
    'scripts/npu_metrics_exporter.py',
    'scripts/watchdog-cron.sh',
    'scripts/quality_benchmark.py',
    'scripts/concurrency_stress.py',
    'scripts/ci_validate_json.py',
    'scripts/ci_validate_yaml.py',
    'scripts/streamlit_chat.py',
    'install.sh',
    'start-all.sh',
    'deploy.sh',
]

all_ok = True
for f in executable_scripts:
    if os.path.exists(f):
        mode = os.stat(f).st_mode
        if not (mode & stat.S_IXUSR):
            print(f'  FAIL: {f} is not executable')
            all_ok = False
        else:
            print(f'  OK: {f}')
    else:
        print(f'  WARN: {f} does not exist')
        all_ok = False

if all_ok:
    print('All executable scripts have +x permission')
else:
    print('Some scripts are missing +x permission')
    sys.exit(1)
