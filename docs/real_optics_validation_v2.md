# Stage 2.5 — numerical optical regression for BaldrApp configuration v2

This stage validates the *physical ZWFS propagation code* following the
configuration and photon-budget refactor. It does **not** change the optical
model, the JSON schema, the runtime server, or the atmospheric simulator.

## Run the tests

From the BaldrApp repository root (with your normal `.venv` active):

```bash
python -m unittest discover -s tests -p 'test_real_optics_v2.py' -v
python -m unittest discover -s tests -p 'test_*.py' -v
```

Run the optional numerical diagnostic, writing PNG figures and a JSON report:

```bash
python src/baldrapp/paranal_simulator/scripts/diagnose_real_optics_v2.py \
    --output-dir stage25_validation
```

To omit Matplotlib plots, add `--no-plots`. Generated reports are not source
code and should usually be kept **out of Git** (or saved outside the repo).

## What actually runs?

`tests/test_real_optics_v2.py` uses the unmodified, actual Python *function
bodies* of `baldr_core.py` and `spectrum.py`, compiled from their AST, along
with an actual import of `baldrapp.utils.fresnel`. This deliberate isolation
avoids importing unrelated optional GUI, SHM and astronomy dependencies that
are not installed on every computer. It does **not** fake `get_frame`,
`get_zwfs_output_field`, either spectral wrapper, `detect`, or any Fresnel
propagation. If these source functions are renamed or removed, the suite fails.

The simplified instrument namespace has:

- A deterministic, circular 40-pixel-diameter entrance pupil sampled on a
  64-by-64 array and a 129-pixel focal-plane FFT grid.
- Bandpass 1500–1800 nm and the **actual** resolved v2 blackbody spectra.
- Zero OPD from a synthetic zero-stroke DM. This isolates physical optics,
  and is **not** a test of the BMC DM, DM registration, or AO loop.
- Real relay dimensions from the v2 example configuration, resolved through
  `config_helper.derive_fresnel_relay_ns`.
- Disabled Poisson and read noise, so the expected photon budget is exact.

The test uses a synthetic on-axis pupil for both source profiles to isolate
spectral effects. It does not reproduce SOLARSTEIN or AT/UT spiders and
secondary obscurations. Source-profile geometry selection was separately
checked in Stage 2, but a hardware/runtime end-to-end test is still required.

## Assertions covered

1. **Mask-out conservation, non-Fresnel:** with phase shift zero and no cold
   stop, the field is unchanged, including for 1, 3, 7 and 15 wavelength
   samples, and summed photons/s equal the entrance rate.
2. **Mask-in finite energy:** with the phase mask and cold stop, the
   non-Fresnel model produces finite nonnegative intensity and no net
   photon creation. The field can lose photons outside the cropped output.
3. **Detector normalization:** actual `detect` bins pixels and scales by
   `QE * DIT` **once**. Changing brightness multiplies counts linearly;
   there is no extra factor of the 300-nm bandpass.
4. **Chromatic non-Fresnel convergence:** actual polychromatic images with a
   nonzero OPD converge with increasing wavelength samples.
5. **Physical Fresnel energy ledger:** compare fields before/after the mirror,
   lens, cold stop and detector. At differently sampled physical planes, use
   `sum(abs(E)**2) * (dx * dy / dx_input**2)` to express a consistent photon
   rate. Expect conservation across lossless propagation and decreases at
   apertures. The initial analytic ZWFS exit field is cropped to its array.
6. **Fresnel detector and spectral convergence:** photon accounting remains
   consistent when the full Fresnel relay is enabled; the chromatic image
   approaches a 31/51-node reference as the sample count increases.
7. **Internal versus on-sky spectrum:** different resolved blackbody photon
   fractions produce different physical ZWFS images with the same entrance
   total photon rate and pupil.
8. **Fresnel amplitude linearity:** multiplying the field by `sqrt(k)`
   multiplies the image by `k`, with no stochastic detector terms.

## Reference result (from 23 October 2026 test environment)

Test scene: sinusoidal OPD of peak scale ~1.1 micrometres; synthetic circular
pupil; internal 3000 K blackbody; v2 example relay; 1e9 entrance photons/s.
Reference spectral image computed with 51 wavelength samples.

| Samples | Non-Fresnel relative L2 | Fresnel relative L2 |
| ---: | ---: | ---: |
| 1 | 1.5730% | 9.1524% |
| 3 | 0.7572% | 3.8573% |
| 7 | 0.0825% | 1.4281% |
| 15 | 0.0142% | 0.4498% |
| 31 | 0.0021% | 0.4003% |

The 7-sample Fresnel error is not negligible. Wavelength convergence must be
assessed for the actual pupil, OPD, relay geometry, and desired accuracy; a
single `n_wavelengths` should not be assumed accurate for all observations.
Convergence need not be strictly monotonic on finite spatial grids.

For the *single* 1.65-micrometre Fresnel example without imposed OPD:

| Plane | Photon rate / entrance rate |
| --- | ---: |
| Entrance pupil | 1.0000 |
| Analytic ZWFS exit pupil (cropped) | 0.8799 |
| After mirror | 0.8799 |
| Before physical cold stop | 0.8799 |
| After physical cold stop | 0.8492 |
| Detector pupil (pre-QE) | 0.8492 |

These are expected losses from a finite cropped ZWFS field and the physical
stop, **not** a failure of photon conservation. They are geometry dependent.
The non-Fresnel `coldstop_diam` is dimensionless and the physical Fresnel
`D_coldstop` is in metres; they do **not** necessarily represent equivalent
stops, so do not compare their throughput numbers as though they did.

## Limitations / remaining validation

- The tests do **not** exercise `init_zwfs_from_json` as a fully constructed
  installed instrument, runtime source switching through ZMQ, SHM publication,
  real DM surfaces, detector crop/offset/noise, or actual calibration pupils.
- The 51-node curve is a numerical reference, not an analytic ground truth.
- FFT array wraparound, crop regions, resampling and angular-spectrum
  bandlimits can depend on the simulation grid; stronger spatial-sampling
  convergence tests will be useful when implementing multilayer Fresnel
  scintillation.
- Source flux denotes band-integrated photons/s in the *entrance pupil*,
  before the downstream optical losses above, detector QE or DIT.
- Stage 3 (multilayer phase-only atmosphere) and Stage 4 (chromatic layered
  scintillation) remain to be implemented.

Keep the previous Stage 1/2 tests; this suite complements rather than
replaces them.
