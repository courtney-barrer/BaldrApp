# BaldrApp v2 — Stage 3: multilayer, phase-only atmosphere

## Scope and physical conventions

The v2 `atmosphere` block describes one integrated von Kármán turbulence profile:

- `r0_500nm_m`: total Fried parameter at 500 nm, independent of the ZWFS central wavelength.
- `L0_m`: common von Kármán outer scale (metres) for every layer.
- `layers`: one or more layers, each with geometric `altitude_m`, `cn2_fraction`, `wind_speed_m_per_s`, and `wind_direction_deg`.
- `random_seed`: the single seed used to derive **independent** layer seeds.
- `scintillation.enabled: false`: required for Stage 3; `true` fails explicitly pending the Stage-4 complex-field implementation.
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

`baldrapp.utils.atmosphere_v2.MultilayerAtmosphere` creates a seeded,
independent von Kármán phase tile per layer using the existing
`phasescreens.ft_phase_screen()` spectral generator. A tile is four times
`grid.dim` in each axis. The telescope footprint samples this tile at
continuous positions displaced by `wind * simulated_time`, using periodic
bilinear interpolation. The total **physical OPD** is the sum of layer
OPDs, with each layer's phase generated at the 500-nm reference wavelength:

\[
 \mathrm{OPD}(x,y,t) = \frac{500\,\mathrm{nm}}{2\pi}
   \sum_i \phi_i(x-v_{x,i}t, y-v_{y,i}t).
\]

This OPD is passed to the existing ZWFS propagation and acquires wavelength-
dependent phase through `2*pi*OPD/wavelength`; it must **not** be treated as a
phase array in radians at the ZWFS central wavelength.

For the phase-only mode, layer altitudes are **metadata**, not propagation
distances: an on-axis source sees the summed OPD at the entrance pupil.
They will become Fresnel propagation separations in Stage 4.

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
uncorrected turbulence and clears delayed corrections. Enabling scintillation
under v2 fails explicitly rather than invoking the independent legacy
high-altitude screen.

## Limitations requiring explicit acknowledgement

**Finite periodic frozen-flow screens**: FFT tiles repeat after a displacement
of `4 * grid.dim * grid.D / grid.N` along either axis. At 20 m/s, for the
example `grid.D=1.8`, `grid.N=64`, `grid.dim=256`, the period along that axis
is approximately **1.44 seconds**. The model is suitable for short
phase-only sequences and code validation, but *not* multi-second or minute-
scale temporal-statistics studies without upgrading the frozen-flow screen
extension. Bilinear interpolation slightly attenuates high spatial
frequencies at fractional-pixel translations; high-accuracy temporal-PSD
studies should replace it with a better long-duration screen generator.

The FFT tile also omits explicit subharmonic augmentation; low-frequency power
is consequently limited by the tile size. Testing full on-sky AO performance
will require careful attention to low-frequency statistics and modal power.

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
