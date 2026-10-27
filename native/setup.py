"""Package existing executables; do not compile or silently relabel them."""
from pathlib import Path
import platform
import shutil
from setuptools import setup, Distribution
from setuptools.command.build_py import build_py
from setuptools.command.bdist_wheel import bdist_wheel

class NativeDistribution(Distribution):
    def has_ext_modules(self):
        return True

class NativeBuild(build_py):
    def run(self):
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("Build these precompiled helpers on Linux x86-64")
        super().run()
        source = Path(__file__).resolve().parents[1] / "src/baldrapp/paranal_simulator/scripts"
        destination = Path(self.build_lib) / "baldrapp_simulator_native/bin"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("shm_creator_sim", "sim_mdm_server"):
            binary = source / name
            header = binary.read_bytes()[:20]
            if header[:5] != b"\x7fELF\x02" or header[5] != 1 or header[18:20] != b"\x3e\x00":
                raise RuntimeError(f"Not a Linux x86-64 ELF executable: {binary}")
            shutil.copy2(binary, destination / name)
            (destination / name).chmod(0o755)

class NativeWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        # auditwheel determines the supported manylinux tag after this build.
        return "py3", "none", "linux_x86_64"

setup(distclass=NativeDistribution, cmdclass={"build_py": NativeBuild, "bdist_wheel": NativeWheel})
