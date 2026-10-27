"""Resolve packaged defaults and user configuration without a Git checkout."""
import os
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent / "fake_configs"


def configuration_path():
    override = os.environ.get("BALDR_CONFIG_PATH")
    return Path(override).expanduser().resolve() if override else CONFIG_DIR / "baldr_config.json"


def resource_path(value, config_path=None):
    """Custom relative resources are relative to the selected configuration.

    Recognize old repository-relative default paths for existing configurations.
    Absolute paths are always preserved, including missing paths (no fallback).
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    for prefix in ("src/baldrapp/paranal_simulator/fake_configs/",
                   "baldrapp/apps/paranal_simulator/fake_configs/"):
        if path.as_posix().startswith(prefix):
            return CONFIG_DIR / path.as_posix()[len(prefix):]
    return (Path(config_path).parent if config_path else CONFIG_DIR) / path
