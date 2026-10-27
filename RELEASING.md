# Releasing BaldrApp and the optional simulator

The main distribution is portable Python. Its `simulator` extra depends on
`baldrapp-simulator-native==0.1.9` without a platform marker: unsupported systems
must reject the requested extra rather than silently omit its essential helpers.
The native project in `native/` packages the two existing tracked ELF binaries.
It does not compile them or make unsupported compatibility claims.

## Build and inspect

Use a clean Linux x86-64 checkout and a separate build environment. Install
`build`, `twine`, `auditwheel`, and `patchelf` there. The build tool installs the
backend requirements automatically unless using an already prepared environment.
Keep the main project, native project, extra pin, and release checker versions
in sync. Use a fresh output directory for each release.

```bash
python -m build --outdir release-artifacts
python -m build --wheel native
# Put the build environment's bin directory on PATH for patchelf.
auditwheel show native/dist/baldrapp_simulator_native-0.1.9-py3-none-linux_x86_64.whl
auditwheel repair --plat manylinux_2_39_x86_64 --wheel-dir release-artifacts native/dist/baldrapp_simulator_native-0.1.9-py3-none-linux_x86_64.whl
python tools/check_release.py
python -m twine check --strict release-artifacts/*.whl release-artifacts/*.tar.gz
```

The current binaries audit to `manylinux_2_39_x86_64`. Retagging them for older
Linux would not make them compatible. Rebuild against an older supported runtime
if older Linux support is required. Check auditwheel again whenever binaries
change. The validator also verifies that both binaries remain byte-for-byte
identical to the tracked originals after packaging and auditing.

Only the core gets an sdist. Do not upload a native sdist containing precompiled
Linux files: a source fallback could bypass the wheel platform restriction. The
native wheel is built from the full checkout; its setup copies the original
tracked executables so duplicate binary copies need not be maintained.

## Validate installed behavior

Install the core wheel in a clean Python environment outside any Git checkout.
Check imports, default configuration resources, and absence of simulator-only
dependencies. In a second environment, install the core wheel with its extra,
using the local artifacts as a package source:

```bash
pip install --find-links release-artifacts "BaldrApp[simulator]==0.1.9"
```

Run `tests/test_installed_runtime.py` with that environment's Python from an
unrelated directory. Its lifecycle test uses stand-in processes and temporary
state; it does not start real servers or touch instrument shared memory.
Run the existing full test suite against the candidate source too.

Before publication, perform a real `baldr start/status/stop` acceptance test on a
dedicated Linux simulator host with a graphical session and no conflicting
shared-memory streams. Do not run this against an active instrument session.
Record the OS/runtime and results. Cross-platform Python availability is a
packaging design; macOS/Windows runtime support still needs testing on those OSes.

## Publish after review and explicit approval

Two PyPI projects are involved. Confirm the publisher can create/use
`baldrapp-simulator-native` and owns `BaldrApp`. The companion name is provisional
until PyPI accepts it; a name collision requires updating the extra dependency.
Use locally configured PyPI credentials or trusted publishing; never paste tokens
into a chat or repository.

Upload the audited native wheel first. Then upload the core wheel and sdist, using
explicit reviewed filenames and the recorded SHA256 hashes. The old `dist/`
candidates and the native `linux_x86_64` intermediate are not release files.
Do not publish until the installed-runtime acceptance test is complete.
Commit the release changes and tag the exact source without rewriting history.

The `tool.uv.sources` mapping is for development in a full checkout only. It points
to the local companion while that package is unpublished; it is not part of the
wheel's dependency metadata. `uv lock` updates the development lock without
requiring the new native package to exist on PyPI first.
