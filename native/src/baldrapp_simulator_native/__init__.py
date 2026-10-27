"""Location of the bundled Paranal simulator executables."""
from pathlib import Path

def binary_directory():
    return Path(__file__).resolve().parent / "bin"
