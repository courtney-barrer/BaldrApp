# BaldrApp simulator native helpers

Precompiled Linux x86-64 executables used by `pip install "BaldrApp[simulator]"`.
This package is released as an audited platform wheel only. It contains
`shm_creator_sim` and `sim_mdm_server`, built from the C++ sources in BaldrApp.
The portable BaldrApp package does not depend on this package unless the
simulator extra is requested. The existing binaries require recent glibc and
libstdc++; consult the audited wheel tag and BaldrApp release notes.

Build from the full BaldrApp checkout: `python -m build --wheel native`.
Then use auditwheel to check and repair the Linux wheel before publication.
Do not publish an unaudited linux_x86_64 wheel or a source distribution of
this packaging directory. Source and CMake build instructions live in BaldrApp.
