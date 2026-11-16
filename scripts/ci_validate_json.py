#!/usr/bin/env python3
"""Validate JSON files in the repository."""
import json
import sys
import os

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(SERVER_DIR)

json_files = [
    'etc/grafana-dashboard-lfm3-npu.json',
]

all_ok = True
for f in json_files:
    try:
        with open(f) as fh:
            json.load(fh)
        print(f'  OK: {f}')
    except Exception as e:
        print(f'  FAIL: {f}: {e}')
        all_ok = False

if all_ok:
    print('All JSON files valid')
else:
    print('Some JSON files are invalid')
    sys.exit(1)
