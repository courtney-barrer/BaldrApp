# BaldrApp v2 — Stage 4: atmospheric Fresnel scintillation

## User-facing switch

Use the **same** multi-layer `atmosphere.layers` configuration developed in Stage 3:

```json
"scintillation": { "enabled": true }
```

Set `enabled: false` for the existing faster **phase-only** mode, which sums
independent layer OPDs at the entrance pupil. The v1 simulation remains unchanged.
There is no second turbulence model, phase-screen algorithm, or independently
configured scintillation r0. Both paths use BaldrApp's original
`phasescreens.PhaseScreenKolmogorov` infinite-screen generator and its
unmodified `add_row()` evolution.

## Wave-optics model

For each physical frame, one OPD map per atmospheric layer is advanced from
its own infinite screen. The *same OPDs* are used at all wavelengths. For each
wavelength in the fixed source bandpass, a unit-amplitude on-axis plane wave
enters the topmost layer, and:

1. Multiply by `exp(2πi OPD_layer / wavelength)` at the layer altitude.
2. Angular-spectrum propagate to the next lower layer, using the actual
   altitude separation. Repeat down to the ground (propagate the lowest
   layer to ground if it is above 0 m).
3. Apply the **telescope pupil once, at the entrance pupil**, not aloft.
4. Scale with the currently active source amplitude, which contains the
   configured **band-integrated photons/s** at the aperture *in the absence*
   of scintillation. Do not renormalize pupil flux frame by frame.
5. Optionally apply the first-stage AO correction as wavelength-independent
   OPD to the complex fields (phase correction exp(-2πi OPD/λ)). AO uses a
   2D-unwrapped phase at the nearest simulated wavelength to `optics.wvl0`,
   followed by the existing idealized modal-projection and lag model.
6. Pass the full per-wavelength complex pupil fields into the existing
   monochromatic ZWFS/relay functions. The v2 spectral wrapper alone applies
   the **unit-sum photon fractions**, and detection happens once, after
   intensity (not amplitude) integration over the bandpass.

The interface is `get_frame_configured(..., incident_fields=fields)` where
`fields.shape = (n_wavelengths, grid.dim, grid.dim)` and `fields` is **complex**.
The fields already include atmospheric phase/amplitude and the telescope
pupil, but **not** the Baldr DM or internal OPD: those are still applied once
by the existing ZWFS functions. When using `incident_fields`, supply
`opd_input=0` to avoid double-counting atmospheric phase.

Both non-Fresnel and Fresnel-relay paths are supported. The latter is a
**separate downstream relay**: atmosphere-to-pupil propagation is followed by
the same ZWFS/relay as before.

## Original infinite phase screen and screen size

Phase-only mode uses the Stage 3 pupil-footprint sampling unchanged. In
scintillation mode, each instance of the **same original** infinite generator
has a larger *native transverse grid*, `ceil(sqrt(2) * grid.dim) + 6`, so that
rotation by arbitrary wind angles can sample an unbroken atmospheric OPD
across the **entire propagation plane**. The optical propagation array still
has shape `(grid.dim, grid.dim)`; its physical pixel scale is `grid.D/grid.N`.
This increases the memory and initialization cost of scintillation mode but
introduces no alternative generator or user configuration option.

## Numerical and physical limitations

- Angular-spectrum FFT propagation assumes periodic numerical boundary
  conditions. It conserves *full-grid* power, but wrap-around artefacts can
  contaminate the pupil if the propagation grid is insufficiently padded.
  Test convergence against a larger `grid.dim` at fixed physical pixel scale.
- Resolution must also capture the atmospheric Fresnel scale
  `sqrt(lambda * altitude)` and the turbulent spatial bandwidth. Grid
  sampling/convergence must be checked for the chosen layers and wavelength.
- Applying the telescope pupil after propagation allows aperture-integrated
  photon flux to fluctuate from frame to frame. Re-normalizing the pupil
  after scintillation would destroy these fluctuations.
- The first-stage AO estimate uses a two-axis 2D phase unwrap at the nearest
  simulated reference wavelength. This is an **approximate idealized modal
  correction**, not a wavefront-sensor simulation. Near amplitude zeros,
  phase unwrapping can be unreliable and should be verified separately.
- Each atmosphere is on-axis, plane-wave illumination. Finite-distance
  sources, anisoplanatic sightlines, chromatic refraction, and source angular
  extent are outside the current model.
- Infinite-screen evolution keeps a finite *transverse* width, while adding
  new rows indefinitely along the physical wind direction.

## Validation

Run after applying the Stage 4 patch:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Tests check zero-turbulence propagation, exact zero-altitude phase-only limit,
chromatic scintillation, full-grid photon-energy conservation, unchanged
infinite-screen type, real non-Fresnel and downstream Fresnel ZWFS outputs,
detector scaling, and source pupil-normalisation boundaries.

The suite does **not** constitute a full hardware/SHM execution or a
convergence study for an arbitrary science case.
