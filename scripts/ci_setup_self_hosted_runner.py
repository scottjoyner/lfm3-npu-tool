#!/usr/bin/env python3
"""
Self-hosted GitHub Actions runner setup for LFM3-1.7B NPU inference.

This script sets up a self-hosted runner on a machine with AMD XDNA2 NPU hardware,
so that GitHub Actions integration tests can run against real NPU hardware.

Prerequisites:
- AMD Strix Halo NPU (aie2p, 6x8) with XRT 2.25.37+
- FastFlowLM v1.0.12+ installed at /tmp/flm and /tmp/lib/liblfm3_npu.so
- LFM3-1.7B model files at /home/scott/.config/flm/models/LFM3-1.7B-NPU2/
- Ryzen AI SDK Python 3.12 venv at /media/scott/data/ryzen-ai-venv
- GitHub token with repo scope (for runner registration)

Usage:
    # 1. Create a GitHub Personal Access Token with 'repo' scope
    # 2. Get the registration token:
       curl -X POST -H "Authorization: token <PAT>" \
         https://api.github.com/repos/scottjoyner/lfm3-npu-tool/actions/runners/registration-token
    # 3. Run this script:
       python3 scripts/ci_setup_self_hosted_runner.py \
         --token <registration-token> \
         --name <runner-name> \
         --labels npu,amd-strix-halo,linux
"""

import argparse
import os
import subprocess
import sys
import time


RUNNER_DIR = "/media/scott/data/actions-runner"
RUNNER_VERSION = "2.322.0"  # Check https://github.com/actions/runner/releases for latest


def check_prerequisites():
    """Verify NPU hardware and required software are available."""
    print("Checking prerequisites...")

    checks = []

    # Check NPU device
    npu_dev = "/dev/accel/accel0"
    checks.append(("NPU device /dev/accel/accel0", os.path.exists(npu_dev)))

    # Check XRT
    xrt_smi = subprocess.run(
        ["which", "xrt-smi"], capture_output=True, text=True
    )
    checks.append(("xrt-smi available", xrt_smi.returncode == 0))

    # Check FLM binary
    flm_bin = "/tmp/flm"
    checks.append(("FLM binary /tmp/flm", os.path.exists(flm_bin)))

    # Check NPU engine
    npu_engine = "/tmp/lib/liblfm3_npu.so"
    checks.append(("NPU engine /tmp/lib/liblfm3_npu.so", os.path.exists(npu_engine)))

    # Check model files
    model_dir = "/home/scott/.config/flm/models/LFM3-1.7B-NPU2"
    model_files = ["config.json", "model.q4nx", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
    model_ok = all(os.path.exists(os.path.join(model_dir, f)) for f in model_files)
    checks.append(("FLM model files", model_ok))

    # Check Ryzen AI venv
    ryzen_python = "/media/scott/data/ryzen-ai-venv/bin/python"
    checks.append(("Ryzen AI Python venv", os.path.exists(ryzen_python)))

    all_ok = True
    for name, ok in checks:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")
        if not ok:
            all_ok = False

    return all_ok


def install_runner(token: str, runner_name: str, labels: str):
    """Download, configure, and start the self-hosted runner."""
    print(f"\nSetting up self-hosted runner: {runner_name}")

    runner_dir = RUNNER_DIR
    os.makedirs(runner_dir, exist_ok=True)

    # Download runner
    runner_archive = f"actions-runner-linux-x64-{RUNNER_VERSION}.tar.gz"
    runner_url = (
        f"https://github.com/actions/runner/releases/download/v{RUNNER_VERSION}/{runner_archive}"
    )
    archive_path = os.path.join(runner_dir, runner_archive)

    if not os.path.exists(archive_path):
        print(f"Downloading runner from {runner_url}...")
        subprocess.run(
            ["curl", "-o", archive_path, "-L", runner_url],
            check=True,
            cwd=runner_dir,
        )
    else:
        print(f"Runner archive already exists, skipping download.")

    # Extract
    print("Extracting runner...")
    subprocess.run(
        ["tar", "xzf", runner_archive],
        cwd=runner_dir,
        check=True,
    )

    # Configure runner
    print(f"Configuring runner '{runner_name}' with labels '{labels}'...")
    env = os.environ.copy()
    env["RUNNER_TOKEN"] = token

    configure_cmd = [
        "./config.sh",
        "--url", "https://github.com/scottjoyner/lfm3-npu-tool",
        "--token", token,
        "--name", runner_name,
        "--labels", labels,
        "--work", "_work",
        "--unattended",
        "--replace",
    ]
    subprocess.run(configure_cmd, cwd=runner_dir, env=env, check=True)

    print("\nRunner configured successfully.")
    print("\nTo start the runner, run:")
    print(f"  cd {runner_dir} && ./svc.sh install && ./svc.sh start")
    print("\nOr run interactively:")
    print(f"  cd {runner_dir} && ./run.sh")


def check_runner_status():
    """Check if the runner is already configured."""
    runner_dir = RUNNER_DIR
    config_path = os.path.join(runner_dir, ".runner")
    return os.path.exists(config_path)


def main():
    parser = argparse.ArgumentParser(
        description="Set up GitHub Actions self-hosted runner for NPU inference"
    )
    parser.add_argument(
        "--token", required=True,
        help="GitHub Actions runner registration token"
    )
    parser.add_argument(
        "--name", default="lfm3-npu-runner",
        help="Runner name (default: lfm3-npu-runner)"
    )
    parser.add_argument(
        "--labels", default="npu,amd-strix-halo,linux,self-hosted",
        help="Runner labels (default: npu,amd-strix-halo,linux,self-hosted)"
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only check prerequisites, don't install"
    )
    args = parser.parse_args()

    print("=== LFM3-1.7B NPU Self-Hosted Runner Setup ===\n")

    if not check_prerequisites():
        print("\n✗ Prerequisites not met. Please install missing components first.")
        sys.exit(1)

    print("\n✓ All prerequisites met.")

    if args.check_only:
        print("\nPrerequisite check complete.")
        return

    if check_runner_status():
        print(f"\nRunner already configured at {RUNNER_DIR}.")
        print("Remove .runner file to reconfigure.")
        return

    install_runner(args.token, args.name, args.labels)


if __name__ == "__main__":
    main()
