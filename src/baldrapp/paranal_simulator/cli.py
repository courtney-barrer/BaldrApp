"""Installed entry point for the Linux Paranal simulator."""
import os
import platform
import subprocess
import sys
from pathlib import Path

from .paths import configuration_path


def run_heimbal():
    if len(sys.argv) != 2 or sys.argv[1] not in {"start", "stop", "status", "tail"}:
        print("Usage: baldr {start|stop|status|tail}")
        return 0 if sys.argv[1:] in (["--help"], ["-h"]) else 2
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        print("The Paranal simulator requires Linux x86-64. Python optics modules remain available.", file=sys.stderr)
        return 1
    env = os.environ.copy()
    env["BALDR_PYTHON"] = sys.executable
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["BALDR_CONFIG_PATH"] = str(configuration_path())
    # Keep relative user overrides meaningful before the launcher changes directory.
    for variable in ("BALDR_STATE_DIR", "BALDR_NATIVE_BIN_DIR"):
        if env.get(variable):
            env[variable] = str(Path(env[variable]).expanduser().resolve())
    script = Path(__file__).resolve().parent / "scripts/heimbal_simulation_servers.sh"
    return subprocess.run(["bash", str(script), *sys.argv[1:]], env=env).returncode
