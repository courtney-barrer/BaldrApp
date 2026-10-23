# BaldrApp configuration v2 — Stage 2 spectral integration

Stage 2 connects the validated v2 `bandpass` and `source_profiles` to
`init_zwfs_from_json`, `get_frame_configured` and the simulator source switch.
The v1 JSON and legacy propagation conventions remain supported unchanged.

## Physical convention

- `source_profiles.<name>.flux.photons_per_second` is the *band-integrated*
  rate at the entrance pupil, before downstream losses and detector QE.
- The entrance amplitude is `pupil_mask * sqrt(rate / sum(pupil_mask**2))`,
  so `sum(abs(amp_input)**2) == rate` regardless of obscuration or simulation
  sampling. The pupil mask represents **amplitude** transmission.
- `zwfs.spectrum.photon_fractions[k]` are dimensionless non-negative photon
  fractions with sum one, calculated by v2 configuration resolution.
- Each monochromatic optical field is propagated with that same entrance
  amplitude. Detected pre-detector rate is `sum_k fraction[k] * intensity[k]`.
  Detector QE, DIT, and read noise are applied once after this sum.
- For `n_wavelengths=1`, the representative node carries the *whole* band-
  integrated photon rate. There is no implicit one-nm bin or extra bandwidth.
- The two source profiles share one wavelength grid, but have independent
  photon fractions, pupil shapes, and photon rates. Changing source does not
  rebuild the fixed spectral wavelength grid.

**V1 is different:** its `amp_input**2` means photons/s/pixel/nm, integrated
using `zwfs.spectrum.weights_nm`. Never use a v1 amplitude with the v2
`photon_fractions` namespace or vice versa.

## Simulator status and boundaries

The existing simulator script defaults to the original v1 configuration.
To select v2, use `BALDR_CONFIG_PATH`:

```bash
BALDR_CONFIG_PATH=src/baldrapp/paranal_simulator/fake_configs/baldr_config_v2.json \
    python -m baldrapp.paranal_simulator.scripts.baldr_sim_updated_runtime_control
```

Stage 3 adds physically evolved infinite turbulence screens with independent
Cn2 fractions and wind vectors. Stage 4 adds an optional Fresnel-propagated
complex atmospheric field at each wavelength. Set
`atmosphere.scintillation.enabled=true` to enable the latter, using the
same layers and the original infinite-screen generator. See
`docs/multilayer_scintillation_v2.md` for photon normalization,
AO and numerical sampling details. The legacy v1 atmosphere and source path
remains unchanged.

## Tests

```bash
python -m unittest discover -s tests -p 'test_configuration_v2.py' -v
python -m unittest discover -s tests -p 'test_source_v2.py' -v
```

`test_source_v2.py` tests the real source normalization and the actual optical
wrapper/dispatcher bodies against deterministic **mock optical propagators**.
The additional Stage 2.5 `test_real_optics_v2.py` tests the **real** Fourier,
Fresnel, detector and wavelength-integration implementations on a synthetic
optical instrument with a zero-stroke DM (without SHM/hardware dependencies).
Run `python -m unittest discover -s tests -p 'test_*.py' -v`. See
`docs/real_optics_validation_v2.md` for results and diagnostic figures.
Full installed-instrument, SHM and on-sky integration testing remains to be
done in the BaldrApp hardware environment.
