# BaldrApp

Simulating Baldr - the Zernike Wavefront Sensor (ZWFS) for VLTI/Asgard

This repository contains modules and examples for end-to-end testing of a ZWFS. The primary application is a full end-to-end simulator for Baldr, which models optical propagation, the deformable mirror (DM), the camera, and multi-device servers via a ZMQ server-client architecture and shared memory (SHM).
The optical propagation includes polychromatic Fresnel models accounting for a configurable atmosphere, first-stage AO, internal misalignment, and misconjugation.

Because the simulator consumes DM shared memory to update the optical model and produces camera frames to shared memory, it operates transparently to the real-time computer (RTC). Any RTC architecture that reads camera frames and writes to the DM via SHM can be tested seamlessly. 

*Note: Shared memory functionality relies on Linux `/dev/shm` (tested on Ubuntu 20.04 and 24.04).*

![Baldr Simulation Architecture](BaldrApp_schematic.jpeg)

## Installation with pip

For Python optics and simulation modules:

```bash
pip install BaldrApp
```

For the full Paranal simulator, including precompiled native helpers:

```bash
pip install "BaldrApp[simulator]"
baldr start
baldr status
baldr stop
```

The base package does not install the Linux shared-memory or GUI dependencies.
The `simulator` extra adds `baldrapp-simulator-native` and the simulator's Python
and GUI dependencies. The bundled executables require Linux x86-64 with the
`manylinux_2_39` compatibility level and the standard C++ runtime. Older Linux
systems, macOS, Windows, and ARM cannot install this native wheel. Core Python
functionality remains available subject to its scientific Python dependencies.
A graphical session is required for the simulator GUIs.

Logs and process IDs are stored under `${XDG_STATE_HOME:-~/.local/state}/baldrapp`
(in `logs` and `run`). Override the location using `BALDR_STATE_DIR`.
Before switching from the older checkout launcher, stop any running simulator
with that older launcher: the new launcher does not read the old `.run/pids`.

Use `BALDR_CONFIG_PATH=/absolute/path/to/baldr_config.json` to select a custom
configuration. Relative resource paths in that file are resolved beside the
configuration. Old repository-relative paths for bundled defaults remain
supported. To copy the installed default for editing:

```bash
python -c 'from baldrapp.paranal_simulator.paths import CONFIG_DIR; import shutil; shutil.copytree(CONFIG_DIR, "baldr-config")'
BALDR_CONFIG_PATH="$PWD/baldr-config/baldr_config.json" baldr start
```

To use edited split or phasemask files in that copy, set their configuration
values to `cred1_split.json` and `phasemask_properties.json`, respectively.
The launcher passes the selected split file to the native shared-memory creator.

For development, clone the repository and use `pip install -e .` for the core.
To build native helpers yourself, use the existing `scripts/build_cpp.sh` with
ImageStreamIO and nlohmann_json installed, then set `BALDR_NATIVE_BIN_DIR` to
the directory containing both executables. This is an alternative to installing
the precompiled native package; the simulator Python dependencies are still
required. See [RELEASING.md](RELEASING.md) for the two-package release procedure.

## Installation with Nix `flake`s
If you are using Nix for package management, you can initialise a `devShell` with all the required dependencies, and installing the python package and GUIs using:
```bash
nix develop
```

This will also give you a shell alias to allow you to start and stop the simulator, e.g.:
```bash
heimbal-sim start  # to start the simulator
```

## Troubleshooting
If you are running this as another user on your linux you might need to give yourself permission to read/write to shared memory which can be done by 
```
sudo chown <user>:<user> /dev/shm/*some_shms* 
```
obviously using your username and whatever shared memory addresses (SHMs) you need to use
