"""Hermetic safety regression for legacy CPU startup (no live services/models)."""
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "start_cpu.sh"


class CpuLegacyStartSafetyTests(unittest.TestCase):
    def run_helper(self, *, healthy=False, occupied=False, unit_enabled=False):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            files = {
                "curl": "#!/bin/sh\n" + (
                    "printf '%s\\n' '{\"status\":\"healthy\",\"provider\":\"CPUExecutionProvider\",\"model\":\"lfm3-1.7b-cpu-cpu\",\"port\":1238}'\n"
                    if healthy else "exit 7\n"
                ),
                "ss": "#!/bin/sh\n" + (
                    "echo 'LISTEN 0 128 0.0.0.0:1238'\n" if occupied else "exit 0\n"
                ),
                "systemctl": "#!/bin/sh\n" + ("exit 0\n" if unit_enabled else "exit 1\n"),
            }
            for name, content in files.items():
                path = folder / name
                path.write_text(content)
                path.chmod(path.stat().st_mode | stat.S_IXUSR)
            env = {**os.environ, "PATH": f"{folder}:/usr/bin:/bin",
                   "LFM3_CPU_LEGACY_START_AUTHORIZED": "0"}
            return subprocess.run(
                ["bash", str(SCRIPT)], env=env, capture_output=True,
                text=True, timeout=8,
            )

    def test_bash_syntax(self):
        result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_healthy_listener_is_not_restarted(self):
        r = self.run_helper(healthy=True, occupied=True, unit_enabled=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("no new process", r.stdout)

    def test_unknown_listener_fails_closed(self):
        r = self.run_helper(healthy=False, occupied=True)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("unverified listener", r.stderr)

    def test_systemd_owner_blocks_legacy_start(self):
        r = self.run_helper(healthy=False, occupied=False, unit_enabled=True)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("managed by systemd", r.stderr)

    def test_no_automatic_new_start_without_approval(self):
        r = self.run_helper(healthy=False, occupied=False, unit_enabled=False)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("LFM3_CPU_LEGACY_START_AUTHORIZED", r.stderr)

    def test_no_destructive_process_or_cache_actions(self):
        source = SCRIPT.read_text()
        self.assertNotIn("pkill", source)
        self.assertNotIn("rm -rf", source)
        self.assertNotIn("kill -9", source)
        self.assertIn("flock -n", source)



    def run_port_guard(self, *, listen=False, ss_error=False, missing_ss=False):
        import sys
        guard = SCRIPT.parent / "scripts" / "cpu_port_guard.py"
        with tempfile.TemporaryDirectory() as dirname:
            tmp = Path(dirname)
            if not missing_ss:
                tool = tmp / "ss"
                if ss_error:
                    tool.write_text("#!/bin/sh\\nexit 3\\n".replace("\\n", "\n"))
                elif listen:
                    tool.write_text("#!/bin/sh\\necho 'LISTEN 0 128 0.0.0.0:1238'\\n".replace("\\n", "\n"))
                else:
                    tool.write_text("#!/bin/sh\\nexit 0\\n".replace("\\n", "\n"))
                tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
            env = {**os.environ, "PATH": str(tmp)}
            return subprocess.run(
                [sys.executable, str(guard)], env=env,
                capture_output=True, text=True, timeout=5,
            )

    def test_free_port_execcondition_allows_start(self):
        result = self.run_port_guard()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_occupied_port_execcondition_skips_start(self):
        result = self.run_port_guard(listen=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("occupied", result.stdout)

    def test_failed_ss_execcondition_skips_start(self):
        result = self.run_port_guard(ss_error=True)
        self.assertEqual(result.returncode, 1, result.stderr)

    def test_missing_ss_execcondition_skips_start(self):
        result = self.run_port_guard(missing_ss=True)
        self.assertEqual(result.returncode, 1, result.stderr)

    def test_unit_registers_port_condition_and_prevents_bind_exit_loop(self):
        unit = (SCRIPT.parent / "etc" / "lfm3-npu-cpu.service").read_text()
        self.assertIn("ExecCondition=/usr/bin/python3 -B /media/scott/data/lfm3-npu-tool/scripts/cpu_port_guard.py", unit)
        self.assertIn("RestartPreventExitStatus=3", unit)

if __name__ == "__main__":
    unittest.main()
