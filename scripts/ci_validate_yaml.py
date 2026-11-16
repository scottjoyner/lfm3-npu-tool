#!/usr/bin/env python3
"""Validate YAML files in the repository."""
import yaml
import sys
import os

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(SERVER_DIR)

yaml_files = [
    '.github/workflows/ci.yml',
    'etc/prometheus-rules-lfm3-npu.yml',
    'etc/prometheus-lfm3.yml',
    'etc/alertmanager-lfm3.yml',
    'docker/otel-collector-config.yaml',
    'docker/observability.yml',
]

all_ok = True
for f in yaml_files:
    try:
        with open(f) as fh:
            yaml.safe_load(fh)
        print(f'  OK: {f}')
    except Exception as e:
        print(f'  FAIL: {f}: {e}')
        all_ok = False

if all_ok:
    print('All YAML files valid')
else:
    print('Some YAML files are invalid')
    sys.exit(1)
