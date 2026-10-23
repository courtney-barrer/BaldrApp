# Faster polychromatic Fresnel propagation (v2)

This patch preserves the equations, optical sampling, spectral weights, and
public call signatures. It replaces repeated numerical work and switches
2D FFT execution from NumPy to SciPy pocketfft, already a project dependency.

## Changes

- `fresnel.fft2()` and `fresnel.ifft2()` wrap `scipy.fft` with the existing
  NumPy-compatible FFT normalization (`norm='ortho'` still works).
- `BALDR_FFT_WORKERS` sets the number of workers **per transform**. Default is
  `1`. Set the variable **before** starting Python, for example:
  `BALDR_FFT_WORKERS=4 python ...`. This is a process-wide setting in
  `fresnel.py`, not a new JSON property.
- Cached D-mirror and cold-stop aperture masks use keys containing the grid,
  physical scales, aperture diameter and offsets/angle. The cold-stop scale
  depends on wavelength, so masks remain chromatic. These caches use small
  LRU limits and return read-only arrays. New geometry uses new keys.
- The analytic ZWFS computation reuses its centered focal-plane radius grid
  and phase-disc masks. Pupil support is remeasured each call to preserve
  source switching and wavelength-dependent scintillation support.
- The polychromatic Fresnel wrapper evaluates the DM OPD **once per frame**
  and, for phase-only incident light, forms the combined pupil OPD once.
  Chromatic `incident_fields` still compute per-wavelength phase correctly.
- No change to detector photon accounting, propagation distances, AO, phase
  screen generators, or Fresnel transfer functions.

## Validation

From the project root:

```sh
python -m unittest discover -s tests -p 'test_*.py' -v
BALDR_FFT_WORKERS=1 python tests/benchmark_fresnel_v2.py
BALDR_FFT_WORKERS=4 python tests/benchmark_fresnel_v2.py
```

Benchmarks exclude the live SHM runtime and detector noise. In the development
container, using a 256x256 wave-space grid and 7 wavelengths, warm-cache
median times for one synthetic Fresnel frame were:

| Original NumPy code | Optimized / 1 worker | Optimized / 4 workers |
|---:|---:|---:|
| 88.0 ms | 76.8 ms | 61.8 ms |

The original and optimized frames agreed to floating-point precision, with
identical integrated photon rates (~959,343,050.928 photons/s for that fixture).
These are **machine-specific** timings, not promised runtime improvements.
Additional runtime DM, SHM, AO and atmosphere work is excluded.

## Operational cautions

- Four beams running in separate processes with 4 FFT workers each may contend
  for CPU cores. Benchmark `BALDR_FFT_WORKERS=1`, `2` and `4` with your actual
  beam count and load before choosing a production setting.
- The cached relay masks use geometry as part of the key. Do not mutate the
  returned read-only masks. Standard instrument configuration updates are fine:
  new values yield a new entry.
- Spectral sample count is unchanged. Optimizing FFT throughput does not make
  under-sampled chromatic Fresnel simulations more accurate.
