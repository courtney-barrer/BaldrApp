"""Schema-v2 source photon accounting, independent of optical propagation.

The source amplitude is the square root of *band-integrated* photons/s/pixel;
photon_fractions weight monochromatic intensity responses exactly once.
No numerical bandwidth factor is needed. The reference plane is the sampled
entrance pupil before downstream optical losses.
"""

from types import SimpleNamespace

import numpy as np


def spectrum_for_profile(resolved, name):
    """A fixed instrument wavelength grid and independent source weights."""
    derived = resolved["derived"]
    profile = derived["source_profiles"][name]
    wavelengths = np.asarray(derived["bandpass"]["wavelengths_m"], dtype=float)
    fractions = np.asarray(profile["photon_fractions"], dtype=float)
    if wavelengths.ndim != 1 or wavelengths.size == 0 or fractions.shape != wavelengths.shape:
        raise ValueError("v2 spectral wavelengths and photon fractions must be matching 1D arrays")
    if not np.all(np.isfinite(fractions)) or np.any(fractions < 0):
        raise ValueError("Invalid photon fractions")
    if not np.isclose(fractions.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("v2 photon fractions must sum to one")
    return SimpleNamespace(
        enabled=True,
        integration_mode="photon_fractions_v2",
        wavelengths=wavelengths.copy(),
        photon_fractions=fractions.copy(),
        active_source_profile=name,
    )


def source_input_amplitude(pupil_mask, photons_per_second):
    """Allocate the band-integrated photon rate across a real pupil mask.

    If the amplitude transmission is P(x,y), the resulting input satisfies
    sum |E|^2 = photons_per_second even for obscured, apodized or resampled
    pupils. Zero flux gives a zero field; zero illuminated area is an error.
    """
    pupil = np.asarray(pupil_mask, dtype=float)
    rate = float(photons_per_second)
    if pupil.ndim != 2 or not np.all(np.isfinite(pupil)) or np.any(pupil < 0):
        raise ValueError("pupil_mask must be a finite, nonnegative 2D transmission array")
    if not np.isfinite(rate) or rate < 0:
        raise ValueError("photons_per_second must be finite and nonnegative")
    norm = float(np.sum(pupil**2))
    if norm <= 0:
        raise ValueError("pupil_mask has no illuminated pixels")
    return np.sqrt(rate / norm) * pupil


def integration_weights(spectrum):
    """Return (weights, unit label) for either spectral convention.

    Schema v1 uses nm integration weights and a photon spectral-density input.
    Schema v2 uses unit-sum photon fractions and a band-integrated rate input.
    """
    if getattr(spectrum, "integration_mode", None) == "photon_fractions_v2":
        return np.asarray(spectrum.photon_fractions, dtype=float), "photon_fraction"
    if hasattr(spectrum, "weights_nm"):
        return np.asarray(spectrum.weights_nm, dtype=float), "nm"
    if hasattr(spectrum, "weights_normalized"):
        normalized = np.asarray(spectrum.weights_normalized, dtype=float)
    else:
        normalized = np.ones(len(spectrum.wavelengths), dtype=float) / len(spectrum.wavelengths)
    return normalized * float(getattr(spectrum, "bandwidth_nm", 1.0)), "nm"
