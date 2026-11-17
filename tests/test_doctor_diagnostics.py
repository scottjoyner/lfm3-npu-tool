"""Doctor checks are read-only and must honor the effective systemd environment."""
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
import subprocess

import pytest


@pytest.fixture(scope="module")
def cli():
    script = Path(__file__).resolve().parents[1] / "bin" / "lfm3-npu"
    loader = SourceFileLoader("_lfm3_doctor_test", str(script))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_env_file_cannot_override_effective_systemd_path(cli, tmp_path, monkeypatch):
    path = tmp_path / "env.sh"
    path.write_text(
        "export LD_LIBRARY_PATH=/old/xrt/lib:$LD_LIBRARY_PATH\n"
        "export XILINX_XRT=/opt/xilinx/xrt\n"
        "export EXPANSION=$UNRESOLVED\n"
    )
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/lib:/opt/xilinx/xrt/lib")
    monkeypatch.delenv("XILINX_XRT", raising=False)
    monkeypatch.delenv("EXPANSION", raising=False)
    cli._load_doctor_env_fallback(str(path))
    assert cli.os.environ["LD_LIBRARY_PATH"] == "/tmp/lib:/opt/xilinx/xrt/lib"
    assert cli.os.environ["XILINX_XRT"] == "/opt/xilinx/xrt"
    assert "EXPANSION" not in cli.os.environ


def test_missing_path_stays_missing_instead_of_faking_shell_expansion(cli, tmp_path, monkeypatch):
    path = tmp_path / "env.sh"
    path.write_text('export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/xilinx/xrt/lib"\n')
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    cli._load_doctor_env_fallback(str(path))
    assert "LD_LIBRARY_PATH" not in cli.os.environ


def test_xrt_supported_version_flag_and_parsing(cli, monkeypatch):
    def fake_run(command, **kwargs):
        assert command == ["xrt-smi", "--version"]
        assert kwargs["timeout"] == 10
        return subprocess.CompletedProcess(command, 0, "  Version : 2.25.37\n  amdxdna Version : 2.25\n", "")
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert cli._doctor_xrt_version() == "2.25.37"


@pytest.mark.parametrize(
    ("code", "stdout"), [(1, "Version : 2.25.37\n"), (0, "No XRT version returned\n")]
)
def test_xrt_bad_status_or_malformed_output_fail_closed(cli, monkeypatch, code, stdout):
    monkeypatch.setattr(
        subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, code, stdout, ""),
    )
    assert cli._doctor_xrt_version() is None
