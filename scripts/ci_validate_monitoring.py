#!/usr/bin/env python3
"""Fail-closed, no-deployment validator for the LFM3 monitoring Compose source."""
from pathlib import Path
import os
import subprocess
import yaml

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "docker" / "prometheus-grafana.yml"


def main():
    with FILE.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    services = config["services"]
    assert {"grafana", "prometheus"} <= services.keys()
    grafana, prometheus = services["grafana"], services["prometheus"]

    assert grafana["ports"] == ["127.0.0.1:3000:3000"], "Grafana must bind loopback only"
    assert prometheus["ports"] == ["127.0.0.1:9090:9090"], "Prometheus must bind loopback only"

    settings = grafana["environment"]
    assert any(
        entry.startswith("GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:?")
        for entry in settings
    ), "Grafana password must be supplied externally with required interpolation"
    assert "GF_SECURITY_ADMIN_PASSWORD=admin" not in settings
    assert "GF_SECURITY_ALLOW_EMBEDDING=true" not in settings
    assert "host.docker.internal:host-gateway" in prometheus.get("extra_hosts", []), (
        "Prometheus needs an explicit Linux host-gateway mapping"
    )

    for service_name in ("grafana", "prometheus"):
        for mount in services[service_name].get("volumes", []):
            source = mount.split(":", 1)[0]
            if source.startswith("./"):
                path = (FILE.parent / source).resolve()
                assert path.is_file(), f"Missing bind-mount source: {source}"

    cmd = ["docker", "compose", "-f", str(FILE), "config", "--quiet"]
    no_secret_env = dict(os.environ)
    no_secret_env.pop("GRAFANA_ADMIN_PASSWORD", None)
    no_secret_env.pop("COMPOSE_ENV_FILES", None)
    no_secret = subprocess.run(cmd, cwd=ROOT, env=no_secret_env,
                               capture_output=True, text=True, timeout=15)
    assert no_secret.returncode != 0, "Compose must refuse missing password"
    assert "GRAFANA_ADMIN_PASSWORD" in no_secret.stderr, (
        "Compose did not refuse missing password for the expected reason"
    )

    dummy_env = dict(no_secret_env)
    dummy_env["GRAFANA_ADMIN_PASSWORD"] = "local-validation-only-placeholder"
    good = subprocess.run(cmd, cwd=ROOT, env=dummy_env,
                          capture_output=True, text=True, timeout=15)
    assert good.returncode == 0, "Safe Compose source configuration failed"
    print("PASS: loopback-only monitoring, required password, valid mounts and Compose")


if __name__ == "__main__":
    main()
