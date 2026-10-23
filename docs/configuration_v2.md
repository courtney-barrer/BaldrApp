# BaldrApp configuration v2 — Stage 1

**Status (Stage 4):** Schema v2 is supported by the current initializer and
simulator runtime. The v1 configuration and optical simulation path remain
separate. To simulate atmospheric scintillation, set
`atmosphere.scintillation.enabled=true` and use the existing layered profile.
See `docs/multilayer_scintillation_v2.md`.

## Example and validation

The reference configuration is
`src/baldrapp/paranal_simulator/fake_configs/baldr_config_v2.json`.
It is **strict JSON**, unlike the older commented `baldr_config_goal.json`.

With BaldrApp installed in editable mode, or `PYTHONPATH=src` set:

```bash
python -m baldrapp.utils.configuration \
  src/baldrapp/paranal_simulator/fake_configs/baldr_config_v2.json

python -m baldrapp.utils.configuration \
  src/baldrapp/paranal_simulator/fake_configs/baldr_config_v2.json \
  --dump-resolved /tmp/baldr_resolved.json

python -m unittest discover -s tests -p 'test_configuration_v2.py' -v
```

The JSON written by `--dump-resolved` separates the original
`configuration` from the `derived` values. It is a *diagnostic/derived
configuration*, not an instrument simulator product.

## Source and bandpass conventions

- `bandpass.wavelength_min_m` and `.wavelength_max_m` define the common
  Baldr instrument wavelength interval in **metres**.
- `bandpass.n_wavelengths` defines the number of equally spaced wavelength
  nodes. A single wavelength uses `optics.wvl0`, which must be within the band.
- Each `source_profiles.{internal,onsky}` has an independent **pupil**,
  **spectrum** and **flux**. Supported pupil models: `SOLARSTEIN`, `AT`, `UT`,
  and `DISC`. Supported spectrum models: `blackbody` (`temperature_K`) and
  `flat` (constant **photon spectral density**, not constant energy density).
- `flux.photons_per_second` means **total photon rate integrated over this
  bandpass at the telescope-conjugate illuminated pupil before downstream
  optical throughput, QE and detector exposure**. It is not photons per
  wave-space pixel, nanometre, or detector pixel. Zero photon rate is valid.
  This convention is implemented in the v2 source path.
- Normalized photon fractions are computed from the blackbody photon spectral
  density, proportional to `lambda**(-4) / (exp(hc/(lambda*k*T))-1)`, using
  trapezoidal integration on the shared wavelength grid. Fractions sum to
  one. The single-wavelength case receives fraction one of the *integrated*
  photon rate, not a 1 nm approximation.
- `optics.wvl0` retains its role as the reference wavelength for optical
  properties. It is not a competing bandwidth or source spectrum.
- The current switching code can change a pupil mask but not `grid.D`.
  The validator rejects obvious AT/UT physical-diameter mismatches. Full
  physical rescaling is not part of Stage 1.

## Atmosphere and time conventions

- `atmosphere.r0_500nm_m` describes the **total** Fried parameter at 500 nm.
  `atmosphere.L0_m` is one global von Karman outer scale in metres.
- `atmosphere.layers` is a nonempty list. Each layer contains altitude in
  metres, fraction of the **integrated Cn2** strength, and a flow velocity.
  Fractions sum to 1 within 1e-6 absolute tolerance. Zero-weight layers are
  allowed and have a null/infinite layer r0 in the derived JSON.
- Wind direction is the **direction of motion toward** +x for zero degrees,
  increasing counterclockwise toward +y for 90 degrees in the simulation
  pupil coordinate axes. Speeds are metres per second.
- The resolved report computes per-layer
  `r0_i = r0_total * cn2_fraction_i**(-3/5)`, and
  `v_eff = [sum(cn2_fraction_i * wind_speed_i**(5/3))]**(3/5)`;
  `tau0_500nm = 0.314 * r0_total / v_eff`. When all winds are zero, `tau0`
  is reported as JSON `null` to represent infinity. These resolved values are used to set independent layer strengths and wind evolution.
- `simulator_runtime.time_step_s` advances **physical simulation time**.
  `sleep_time_s` only throttles the simulator's wall-clock execution speed.
- The `scintillation.enabled` flag has no separate r0, high-layer distance,
  or phase screen configuration. `false` sums layer OPDs at the pupil; `true`
  Fresnel-propagates wavelength-specific complex fields through the layers.

## Migration / intentionally deferred code changes

V1's `source`, `stellar`, `stellar.spectrum`, and `atmosphere.phase` input
scheme are **not automatically translated**, as doing so could falsely
preserve a different photon-normalisation convention. The v2 validator
rejects those keys clearly. The v1 example is left in place; compare its
reference outputs before implementing v2 source integration.

Stage 2 now connects the v2 source spectra and integrated photon rates to
`baldr_core.py` and the runtime source-selection logic, without changing the
physics of the Fresnel/ZWFS kernels. See `docs/source_photon_budget_v2.md`
for the v2 amplitude/weight convention and the explicit remaining Stage 3
(multilayer atmosphere) limitation. No hardware-dependent simulator run has
been performed in this development environment.

The validator strictly checks all newly introduced source/bandpass/atmosphere
fields and their values. Unchanged existing instrument sections are
preserved as-is and only checked to be JSON objects at this stage; their
complete field-level constraints continue to belong to the corresponding
legacy instrument initializers.
