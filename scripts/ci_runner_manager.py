#!/usr/bin/env python3
"""
GitHub Actions self-hosted runner management for LFM3-1.7B NPU.

Manage the runner lifecycle: install, start, stop, status, uninstall.
Intended for use on the NPU host machine.

Usage:
    python3 scripts/ci_runner_manager.py install --token <token> [--name <name>] [--labels <labels>]
    python3 scripts/ci_runner_manager.py start
    python3 scripts/ci_runner_manager.py stop
    python3 scripts/ci_runner_manager.py restart
    python3 scripts/ci_runner_manager.py status
    python3 scripts/ci_runner_manager.py uninstall
"""

import argparse
import os
import subprocess
import sys

RUNNER_DIR = "/media/scott/data/actions-runner"
SERVICE_NAME = "lfm3-npu-github-runner"
RUNNER_SERVICE_FILE = "/media/scott/data/lfm3-npu-tool/etc/lfm3-npu-github-runner.service"
SYSTEMD_SERVICE_PATH = f"/home/scott/.config/systemd/user/{SERVICE_NAME}.service"


def run(cmd, check=True, capture=False):
    """Run a shell command."""
    result = subprocess.run(
        cmd, capture_output=capture, text=True, check=check
    )
    return result


def runner_configured():
    """Check if the runner is configured."""
    return os.path.exists(os.path.join(RUNNER_DIR, ".runner"))


def cmd_install(args):
    """Install and configure the runner."""
    import importlib.util

    # Import the setup script
    setup_path = "/media/scott/data/lfm3-npu-tool/scripts/ci_setup_self_hosted_runner.py"
    spec = importlib.util.spec_from_file_location("setup_runner", setup_path)
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)

    if not setup.check_prerequisites():
        print("Prerequisites not met. Aborting.")
        sys.exit(1)

    setup.install_runner(args.token, args.name, args.labels)

    # Install systemd service
    print(f"\nInstalling systemd service: {SERVICE_NAME}")
    os.makedirs("/home/scott/.config/systemd/user", exist_ok=True)
    subprocess.run(
        ["cp", RUNNER_SERVICE_FILE, SYSTEMD_SERVICE_PATH],
        check=True,
    )
    subprocess.run(
        ["systemctl", "--user", "daemon-reload"],
        check=True,
    )
    print(f"Service installed. Start with: systemctl --user start {SERVICE_NAME}")


def cmd_start(args):
    """Start the runner service."""
    if not runner_configured():
        print("Runner not configured. Run 'install' first.")
        sys.exit(1)

    subprocess.run(
        ["systemctl", "--user", "start", SERVICE_NAME],
        check=True,
    )
    print(f"{SERVICE_NAME} started.")


def cmd_stop(args):
    """Stop the runner service."""
    subprocess.run(
        ["systemctl", "--user", "stop", SERVICE_NAME],
        check=False,
    )
    print(f"{SERVICE_NAME} stopped.")


def cmd_restart(args):
    """Restart the runner service."""
    subprocess.run(
        ["systemctl", "--user", "restart", SERVICE_NAME],
        check=True,
    )
    print(f"{SERVICE_NAME} restarted.")


def cmd_status(args):
    """Show runner status."""
    # systemd status
    result = subprocess.run(
        ["systemctl", "--user", "status", SERVICE_NAME],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Runner status
    if runner_configured():
        result = subprocess.run(
            [os.path.join(RUNNER_DIR, "./svc.sh"), "status"],
            capture_output=True, text=True, cwd=RUNNER_DIR,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    else:
        print("Runner not configured.")


def cmd_uninstall(args):
    """Uninstall the runner."""
    cmd_stop(args)

    subprocess.run(
        ["systemctl", "--user", "disable", SERVICE_NAME],
        check=False,
    )
    os.remove(SYSTEMD_SERVICE_PATH) if os.path.exists(SYSTEMD_SERVICE_PATH) else None
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)

    if runner_configured():
        subprocess.run(
            [os.path.join(RUNNER_DIR, "./config.sh"), "remove"],
            cwd=RUNNER_DIR,
            check=False,
        )
    print(f"{SERVICE_NAME} uninstalled.")


def main():
    parser = argparse.ArgumentParser(
        description="Manage GitHub Actions self-hosted runner for LFM3-1.7B NPU"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # install
    install_parser = subparsers.add_parser("install", help="Install and configure runner")
    install_parser.add_argument("--token", required=True, help="GitHub registration token")
    install_parser.add_argument("--name", default="lfm3-npu-runner", help="Runner name")
    install_parser.add_argument(
        "--labels", default="npu,amd-strix-halo,linux,self-hosted",
        help="Runner labels"
    )
    install_parser.set_defaults(func=cmd_install)

    # start
    start_parser = subparsers.add_parser("start", help="Start runner service")
    start_parser.set_defaults(func=cmd_start)

    # stop
    stop_parser = subparsers.add_parser("stop", help="Stop runner service")
    stop_parser.set_defaults(func=cmd_stop)

    # restart
    restart_parser = subparsers.add_parser("restart", help="Restart runner service")
    restart_parser.set_defaults(func=cmd_restart)

    # status
    status_parser = subparsers.add_parser("status", help="Show runner status")
    status_parser.set_defaults(func=cmd_status)

    # uninstall
    uninstall_parser = subparsers.add_parser("uninstall", help="Uninstall runner")
    uninstall_parser.set_defaults(func=cmd_uninstall)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
