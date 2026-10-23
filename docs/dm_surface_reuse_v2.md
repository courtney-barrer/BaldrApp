# Per-frame DM OPD reuse in the Paranal simulator

## Motivation

The simulator formerly evaluated the same physical DM surface twice in each
beam/frame: once inside `get_frame_configured` (Fresnel/polychromatic generation)
and once for OPD diagnostics (or for telemetry, which already shared the
diagnostic evaluation). `CompiledDM.eval()` is a dense contraction over all
actuator influence functions, so avoiding the redundant call can help.

## Exact behaviour

1. Read a **fresh** command from the beam's DM SHM channel.
2. Assign a copy to `zwfs.dm.current_cmd`, as before.
3. Generate atmosphere/source fields for the current frame, as before.
4. Evaluate `compiled_dm.eval(current_cmd * opd_per_cmd)` **once**.
5. Pass the same metre-valued 2D OPD as `dm_opd=` to `get_frame_configured()`.
6. Reuse it for `update_opd_diagnostics(..., opd_dm=...)` and telemetry.

The DM contribution is **not** included in `opd_input` or `opd_internal`.
The optical frame functions still add it exactly once. In scintillation mode,
`incident_fields` still represents the complex atmospheric field *before* DM,
and the same DM phase is applied exactly once at every wavelength.

This is intentionally a **per-frame local** snapshot, **not** a cache on the
instrument namespace: frame N+1 must use its own newly sampled DM command.
Source switching, atmospheric evolution, phase-mask settings and the physical
DM influence-function implementation are unaffected.

## API and compatibility

`get_frame_configured(..., dm_opd=None)` accepts an optional precomputed DM OPD
map, in **metres**, on the wave-space pupil grid. It routes this argument to
all four optical modes (monochromatic/polychromatic, Fresnel/non-Fresnel).
Passing no map retains the original behaviour, with the original DM evaluation
inside the optical functions. Direct callers remain source compatible.

The runtime checks the supplied OPD shape against the pupil grid. Passing a
DM command vector instead of a 2D physical OPD map is an error.

The optional keyword is intended for tightly coupled callers that sample and
pass the **current** command. It cannot automatically detect a stale 2D array
provided by an external caller; the live runtime explicitly reevaluates every
frame to prevent this.

## Validation

`tests/test_dm_reuse_v2.py` uses actual production optical propagation
functions with nonzero, spatially varying DM surfaces to compare supplied and
internally evaluated OPDs in all four dispatcher modes. It also checks
wavelength-dependent complex atmospheric fields, command changes between
frames, the real diagnostics helper, and the main runtime call graph.

Run: `python -m unittest discover -s tests -p 'test_*.py' -v`

For a performance measurement, repeat the **same** SHM simulator configuration
and py-spy capture as the baseline, with the same beams, wavelengths, FFT
workers, atmospheric layers and telemetry settings. The removed contraction
was approximately 9% of sampled CPU in that baseline, but profiler samples do
not guarantee a corresponding wall-clock speedup.

The headless automated tests cannot replace one live SHM/telemetry smoke test
on the instrument computer.
