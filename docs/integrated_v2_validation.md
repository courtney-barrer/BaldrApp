# Stage 5: integrated v2 physics validation

This stage validates the interactions of **source selection, multi-layer frozen-flow atmosphere, phase-only AO correction, Fresnel scintillation, ZWFS propagation and detector photon accounting** without starting the Paranal simulator's SHM or ZMQ server loop.

## Execute

From the BaldrApp repository root, in the project environment:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python src/baldrapp/paranal_simulator/scripts/diagnose_integrated_v2.py --output-dir stage5_validation
```

Output (written to the chosen directory, not to the source tree):

- `stage5_metrics.json`: source/entrance/optical rates, throughput, detected electrons, residual OPD RMS and temporal diagnostics for every scenario.
- `stage5_detector_frames.png`: example detector images (log intensity scale).
- `stage5_temporal_metrics.png`: relative pupil photon rate, optical throughput and OPD RMS across frames.

For a faster non-Fresnel check, use `--non-fresnel`. For spectral sampling comparisons, try `--n-wavelengths 7` (the baseline uses 3 for speed). Numerical tests disable shot/read noise and use real production FFT/Fresnel frame generators.

## Six reproducible scenarios

1. `internal`: SOLARSTEIN calibration profile, no atmospheric evolution.
2. `onsky_clear`: AT profile, atmosphere disabled.
3. `phase_only`: AT with original infinite `PhaseScreenKolmogorov` layers and no AO/scintillation.
4. `phase_ao`: same phase layers, piston/tip/tilt modal correction.
5. `scintillation`: AT, wavelength-dependent propagation through all layers, no AO.
6. `scintillation_ao`: same propagation with modal phase correction.

The **source wavelength grid remains fixed** when switching profiles. The nominal source rate is band-integrated photons/s before atmospheric pupil scintillation. The *intercepted* rate is allowed to fluctuate when scintillation is enabled. Detector QE and DIT are each applied exactly once, and real optical losses are not divided out.

`residual_opd_rms_nm` is a geometric pupil-masked RMS, **not** a Strehl estimator. Under scintillation it uses the approximate unwrapped phase diagnostic computed by the runtime, so it should not be interpreted as the full turbulent phase or an absolute AO performance prediction. The pupil scintillation index is the variance of wavelength-integrated relative pupil irradiance divided by the square of its mean.

## Scope and limitations

The tests use the numerical implementations in `atmosphere_v2.py`, `baldr_core.py`, `source_v2.py`, `pupils.py`, and the **actual runtime helper function bodies** in `baldr_sim_updated_runtime_control.py`. Those helpers are extracted without importing the runtime module because it immediately opens SHM and ZMQ connections at module import time.

The headless instrument has a **synthetic zero-stroke DM** and a three-mode analytical piston/tip/tilt basis (rather than the complete HCIPy initialized Zernike basis). The small default pupil grid (64 px with a 40 px aperture), three wavelengths, and four frames are smoke-test settings; they are not quantitative science-grade AO or chromatic-scintillation convergence tests.

These checks **do not establish** that actual full instrument initialization, DM registration, control sockets, frame cropping, ADU offset/background, shared-memory frame formats, telemetry publishing or the GUI work together. Those still require the installed Paranal simulator and related hardware/SHM dependencies. The full live loop must be tested separately before claiming that v2 works end to end in the operational environment.

Do not apply the reference images or short-run photon throughput as on-sky predictions; the FFT-grid sampling, atmospheric Fresnel padding and modal phase-unwrapping approximation still merit science-level convergence checks.
