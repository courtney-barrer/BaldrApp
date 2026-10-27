"""Packaging regressions; stand-in servers never touch instrument shared memory."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from baldrapp.paranal_simulator import cli, paths


class InstalledRuntimeTests(unittest.TestCase):
    def test_resource_paths_preserve_user_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            self.assertEqual(paths.resource_path("custom.json", config), config.parent / "custom.json")
            absolute = config.parent / "missing.json"
            self.assertEqual(paths.resource_path(absolute, config), absolute)
            self.assertEqual(paths.resource_path("src/baldrapp/paranal_simulator/fake_configs/cred1_split.json", config), paths.CONFIG_DIR / "cred1_split.json")
            self.assertEqual(paths.resource_path("baldrapp/apps/paranal_simulator/fake_configs/phasemask_properties.json", config), paths.CONFIG_DIR / "phasemask_properties.json")

    def test_help_and_unsupported_platform_do_not_start_processes(self):
        with patch.object(sys, "argv", ["baldr", "--help"]), patch.object(cli.subprocess, "run") as run:
            self.assertEqual(cli.run_heimbal(), 0)
            run.assert_not_called()
        with patch.object(sys, "argv", ["baldr", "start"]), patch.object(cli.platform, "system", return_value="Darwin"), patch.object(cli.subprocess, "run") as run:
            self.assertEqual(cli.run_heimbal(), 1)
            run.assert_not_called()

    @unittest.skipUnless(sys.platform == "linux", "Bash process lifecycle is Linux-specific")
    def test_launcher_with_stand_in_servers_outside_checkout(self):
        script = paths.CONFIG_DIR.parent / "scripts/heimbal_simulation_servers.sh"
        with tempfile.TemporaryDirectory(prefix="baldr runtime test ") as directory:
            root = Path(directory)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            config = root / "config.json"
            split = root / "custom split.json"
            split.write_text('{}')
            config.write_text(json.dumps({"simulator_runtime": {"shared_memory": {"split_json": split.name}}}))
            def executable(name, body):
                file = bin_dir / name
                file.write_text(body)
                file.chmod(0o755)
                return file
            python_proxy = executable("python-proxy", '#!/usr/bin/env python3\nimport os,sys\nif sys.argv[1] == "-u":\n os.execlp("sleep", "sleep", "120")\nif sys.argv[1:] == ["-c", "import xaosim.shmlib, zmq, PyQt5, pyqtgraph"]:\n sys.exit(0)\nos.execv('+repr(sys.executable)+', ['+repr(sys.executable)+'] + sys.argv[1:])\n')
            executable("shm_creator_sim", '#!/usr/bin/env python3\nimport os,sys\nfrom pathlib import Path\nPath(os.environ["BALDR_TEST_ARGUMENT"]).write_text(sys.argv[1])\n')
            executable("sim_mdm_server", '#!/usr/bin/env bash\ncat >/dev/null\n')
            for name in ("lab-MDM-control", "shmview"):
                executable(name, '#!/usr/bin/env bash\nexec sleep 120\n')
            env = dict(os.environ, BALDR_PYTHON=str(python_proxy), BALDR_NATIVE_BIN_DIR=str(bin_dir),
                       BALDR_CONFIG_PATH=str(config), BALDR_STATE_DIR=str(root / "state"),
                       BALDR_TEST_ARGUMENT=str(root / "argument"), PATH=str(bin_dir)+os.pathsep+os.environ['PATH'])
            def launch(action):
                return subprocess.run(["bash", str(script), action], cwd=root, env=env, text=True, capture_output=True, timeout=30)
            try:
                result = launch("start")
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                self.assertEqual((root / "argument").read_text(), str(split))
                status = launch("status")
                self.assertEqual(status.stdout.count("[UP]"), 6, status.stdout+status.stderr)
                stopped = launch("stop")
                self.assertEqual(stopped.returncode, 0, stopped.stdout+stopped.stderr)
                self.assertFalse((root / "state/run/pids").exists())
            finally:
                pidfile = root / "state/run/pids"
                if pidfile.exists():
                    for line in pidfile.read_text().splitlines():
                        pgid = line.split(":")[-1]
                        if pgid:
                            try:
                                os.killpg(int(pgid), signal.SIGKILL)
                            except ProcessLookupError:
                                pass


if __name__ == "__main__":
    unittest.main()
