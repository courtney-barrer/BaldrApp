# BaldrApp v2 — Stage 3: multilayer, phase-only atmosphere

## Scope and physical conventions

The v2 `atmosphere` block describes one integrated von Kármán turbulence profile:

- `r0_500nm_m`: total Fried parameter at 500 nm, independent of the ZWFS central wavelength.
- `L0_m`: common von Kármán outer scale (metres) for every layer.
- `layers`: one or more layers, each with geometric `altitude_m`, `cn2_fraction`, `wind_speed_m_per_s`, and `wind_direction_deg`.
- `random_seed`: the single seed used to derive **independent** layer seeds.
- `scintillation.enabled: false`: preserves the Stage 3 phase-only path; `true` now enables Stage 4 full-field Fresnel propagation through those same layers.
- `simulator_runtime.time_step_s`: physical simulated time increment for each on-sky camera frame. **Not** `sleep_time_s`, which only throttles wall-clock execution.

Cn² fractions must sum to unity. For each layer, the code derives

\[
 r_{0,i}(500\,\mathrm{nm}) = r_{0,\mathrm{total}}f_i^{-3/5}
\]

and winds

\[
 v_x=v\cos\theta,\qquad v_y=v\sin\theta.
\]

**Wind directions follow the array-coordinate convention**: 0° moves the
*pattern* toward increasing columns (+x); 90° moves it toward increasing rows
(+y). These are mathematical direction-of-motion angles, **not** meteorological
FROM directions, nor automatically sky position angles.

## Numerical implementation

`baldrapp.utils.atmosphere_v2.MultilayerAtmosphere` instantiates **the same
`phasescreens.PhaseScreenKolmogorov` class** used by the original Paranal
simulator, once for every nonzero-Cn2 layer. This is BaldrApp's Fried-style
infinite-screen method, extended using its unmodified `add_row()` method
(Assemat--Wilson/Fried covariance technique). Its implementation accepts
`L0_m` despite the Kolmogorov class name. There is **no periodized FFT tile**
and no configurable choice of phase-screen algorithms.

The number of new rows follows the physical distance traveled along each
layer's wind vector divided by `grid.D/grid.N` (pixel size). The fractional
row displacement is sampled by linear interpolation. Wind direction rotates
the **sampling coordinates**, not the phase-screen generator: this maps the
native +row direction into the configured 2-D pupil motion. As in the old
simulator, extension is one-sided and the turbulent field has bounded
transverse width. Therefore the telescope pupil must fit within the center
of the `grid.dim` screen for all angles; the resolver now requires at least
three pixels of padding around `grid.N`.

Every layer is generated with its `r0_i` **at 500 nm**; all phase screens are
converted to physical OPD and summed:

\[
 \mathrm{OPD}(x,y,t) = \frac{500\,\mathrm{nm}}{2\pi} \sum_i \phi_i(x,y,t).
\]

Here each `phi_i` is the wind-advected sample of its own original infinite
screen, not a periodically wrapped FFT array. The `opd_at(t)` diagnostic
method can inspect future frames without advancing generator state, but a
committed screen cannot be queried at an earlier time after advancing.

This OPD is passed to the existing ZWFS propagation and acquires wavelength-
dependent phase through `2*pi*OPD/wavelength`; it must **not** be treated as a
phase array in radians at the ZWFS central wavelength.

For phase-only operation, layer altitudes are **metadata**, not propagation
distances: an on-axis source sees the summed OPD at the entrance pupil.
With Stage 4 scintillation enabled, they are Fresnel propagation separations.
See `docs/multilayer_scintillation_v2.md`.

First-stage AO acts on the **summed physical OPD** using the existing
Zernike basis: least-squares modal reconstruction, a configured correction
gain in `[0,1]`, optional piston exclusion, and an integer `lag_frames` delay.
Pupil masking is applied after AO; the Baldr DM contribution is still added
by `get_frame_configured()` and must not be included twice. The existing
`phase_scaling_factor` multiplies the atmospheric OPD before the AO fit when
first-stage AO is enabled.

## Runtime source selection

The v1 atmosphere path remains unchanged. For v2, on-sky camera frames draw
the next sum of physical layer OPDs; disabling phase from the runtime control
returns zero atmospheric OPD but continues to advance simulated time.
Switching to `internal` bypasses atmospheric generation entirely; the
internal-source spectrum and amplitude are retained. Disabling AO returns
uncorrected turbulence and clears delayed corrections. Enabling scintillation under v2 now uses Fresnel propagation through the
same atmospheric layers, not the independent v1 high-altitude screen.

## Limitations requiring explicit acknowledgement

**Infinite-screen limitations:** extending rows avoids periodic wrapping
along the wind direction and retains the original generator's long-time
statistics. However, fractional-pixel/rotated bilinear sampling mildly
attenuates high spatial frequencies. The available transverse screen width
is finite; this version evaluates the entrance-pupil footprint rather than
asserting that the full FFT array always lies in the rotated screen. For
long-duration simulations, runtime and numerical accuracy should still be
benchmarked against the original single-screen implementation.

The current `first_stage_ao` model is intentionally an idealized modal
projection, not a closed-loop predictive AO controller. Integer frame lag is
included, not detector-exposure integration or WFS/RTC latency models.

## Local tests

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m baldrapp.utils.configuration \
  src/baldrapp/paranal_simulator/fake_configs/baldr_config_v2.json
```

For phase-only operation set `atmosphere.enabled=true` and
`atmosphere.scintillation.enabled=false`. As before, launch the actual simulator
with the v2 file selected through `BALDR_CONFIG_PATH`; the SHM and camera
servers must be configured separately. No change to the v1 JSON is required.
