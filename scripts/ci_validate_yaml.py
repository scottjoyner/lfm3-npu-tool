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

# Semantic transport checks: parsing YAML alone does not validate OTLP ports.
try:
    with open('docker/observability.yml') as fh:
        compose = yaml.safe_load(fh)
    with open('docker/otel-collector-config.yaml') as fh:
        collector = yaml.safe_load(fh)

    jaeger = compose['services']['jaeger']
    ports = jaeger['ports']
    if len(ports) != len(set(ports)):
        raise ValueError('Jaeger publishes duplicate host port mappings')
    if ports.count('14318:4318') != 1:
        raise ValueError('Expected one Jaeger OTLP HTTP mapping 14318:4318')

    exporter = collector['exporters']['otlp']
    if exporter['endpoint'] != 'jaeger:4317':
        raise ValueError('OTLP gRPC exporter must use jaeger:4317 (no HTTP path)')
    if exporter.get('tls', {}).get('insecure') is not True:
        raise ValueError('In-network OTLP gRPC exporter must declare its TLS policy')
    if 'otlp' not in collector['service']['pipelines']['traces']['exporters']:
        raise ValueError('Traces pipeline is not connected to OTLP exporter')
    print('  OK: Jaeger / OTLP exporter transport and unique port mappings')
except (KeyError, TypeError, ValueError, OSError, yaml.YAMLError) as e:
    print(f'  FAIL: Jaeger / OTLP transport: {e}')
    all_ok = False

if all_ok:
    print('All YAML files valid and OTLP transport checks passed')
else:
    print('Some YAML or OTLP transport checks failed')
    sys.exit(1)
