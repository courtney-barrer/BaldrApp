"""
Spectral sampling utilities for BaldrApp.

This module provides lightweight machinery for polychromatic simulations.
It intentionally does not perform ZWFS propagation, Fresnel propagation,
detector modelling, or DM modelling. Those remain in baldr_core.py and
fresnel.py.

Main responsibilities:

    1. Build a wavelength grid.
    2. Build relative spectral weights.
    3. Normalize spectral weights.
    4. Provide theta(lambda) helpers for chromatic phase-mask behaviour.

The preferred JSON config layout is:

    "stellar": {
        "bandwidth": 300.0,
        "spectrum": {
            "enabled": true,
            "mode": "blackbody",
            "temperature_K": 3500.0,
            "weighting": "photon",
            "wvl_min": 1.5e-6,
            "wvl_max": 1.8e-6,
            "n_wvl": 7
        }
    }

This module only consumes the nested stellar.spectrum dictionary. The scalar
stellar.bandwidth and weights_nm convention are handled in config_helper.py.

The expected use is:

    spectrum_cfg = cfghelp.get_spectrum_config_from_cfg(cfg)

    zwfs_ns.spectrum = spec.derive_spectrum(
        spectrum_cfg,
        default_wvl0=zwfs_ns.optics.wvl0,
    )

    zwfs_ns.spectrum = cfghelp.complete_spectrum_integration_fields(
        zwfs_ns.spectrum
    )

Then Baldr frame-generation functions can loop over:

    zwfs_ns.spectrum.wavelengths
    zwfs_ns.spectrum.weights_normalized
    zwfs_ns.spectrum.weights_nm
"""

from types import SimpleNamespace

import numpy as np

from baldrapp.common import utilities as util


# ============================================================
# Small conversion helpers
# ============================================================

def _as_namespace(obj):
    """
    Convert dictionaries to SimpleNamespace recursively.

    If obj is already a SimpleNamespace, it is returned unchanged.
    Lists are preserved, with any dictionaries inside converted.
    """
    if isinstance(obj, SimpleNamespace):
        return obj

    if isinstance(obj, dict):
        return SimpleNamespace(
            **{k: _as_namespace(v) for k, v in obj.items()}
        )

    if isinstance(obj, list):
        return [_as_namespace(v) for v in obj]

    return obj


def default_monochromatic_spectrum(default_wvl0):
    """
    Return the backwards-compatible monochromatic spectrum.

    This is the default when no spectrum section is provided in the config.
    """
    return SimpleNamespace(
        enabled=False,
        mode="monochromatic",
        weighting="none",
        normalize="sum",
        wavelengths=np.array([float(default_wvl0)], dtype=float),
        weights=np.array([1.0], dtype=float),
        weights_normalized=np.array([1.0], dtype=float),
    )


# ============================================================
# Spectral weighting
# ============================================================

def normalize_weights(weights, normalize="sum"):
    """
    Normalize spectral weights.

    Parameters
    ----------
    weights : array-like
        Non-negative spectral weights.
    normalize : str
        Supported options:

        "sum"
            Normalize so sum(weights) = 1. This is the recommended default
            for polychromatic intensity summation.

        "peak"
            Normalize by peak first, then normalize to unit sum. This gives
            the same final result as "sum" for positive weights, but is kept
            for clarity and future use.

        "none" or "raw"
            Return raw weights unchanged.

    Returns
    -------
    weights_normalized : ndarray
    """
    weights = np.asarray(weights, dtype=float)

    if np.any(weights < 0):
        raise ValueError("Spectrum weights must be non-negative.")

    if np.sum(weights) <= 0:
        raise ValueError("Spectrum weights must have positive sum.")

    normalize = str(normalize).lower()

    if normalize == "sum":
        return weights / np.sum(weights)

    if normalize == "peak":
        w = weights / np.max(weights)
        return w / np.sum(w)

    if normalize in ["none", "raw"]:
        return weights.copy()

    raise ValueError(f"Unsupported spectrum normalization mode: {normalize}")


def blackbody_weights(wavelengths_m, temperature_K, weighting="photon"):
    """
    Compute relative blackbody spectral weights.

    This uses util.planck_law if available in BaldrApp utilities. The returned
    values are relative weights only; absolute units are not required because
    the weights are normally normalized afterwards.

    Parameters
    ----------
    wavelengths_m : array-like
        Wavelengths in metres.
    temperature_K : float
        Blackbody temperature in Kelvin.
    weighting : str
        "energy"
            Use B_lambda.

        "photon"
            Use photon-counting weights proportional to B_lambda * lambda.
            This is usually the better choice for detector photon counts.

    Returns
    -------
    weights : ndarray
        Relative spectral weights.
    """
    wavelengths_m = np.asarray(wavelengths_m, dtype=float)

    if np.any(wavelengths_m <= 0):
        raise ValueError("All wavelengths must be positive.")

    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive.")

    # utilities.planck_law in BaldrApp has historically been used with SI-like
    # wavelength values. If your local implementation expects different units,
    # adjust here rather than in the frame-generation code.
    weights_energy = np.asarray(
        util.planck_law(wavelengths_m, temperature_K),
        dtype=float,
    )

    weighting = str(weighting).lower()

    if weighting == "energy":
        return weights_energy

    if weighting == "photon":
        # Photon number is proportional to energy flux divided by photon energy,
        # so relative photon weighting gains a factor lambda.
        return weights_energy * wavelengths_m

    raise ValueError(f"Unsupported blackbody weighting mode: {weighting}")


def derive_spectrum(spectrum_config, default_wvl0):
    """
    Derive wavelength and weight arrays from a spectrum config section.

    Parameters
    ----------
    spectrum_config : dict or SimpleNamespace or None
        Spectrum configuration. If None, a backwards-compatible monochromatic
        spectrum is returned.
    default_wvl0 : float
        Default central wavelength in metres, usually zwfs_ns.optics.wvl0.

    Supported modes
    ---------------
    monochromatic / single
        One wavelength. Uses spectrum.wvl0 if supplied, otherwise default_wvl0.

    flat
        Uniform weights over a wavelength grid.

    blackbody
        Blackbody spectral weights over a wavelength grid.

    table
        User-supplied wavelengths and weights.

    Example JSON
    ------------
    "spectrum": {
      "enabled": true,
      "mode": "blackbody",
      "temperature_K": 3500,
      "weighting": "photon",
      "wvl_min": 1.50e-6,
      "wvl_max": 1.80e-6,
      "n_wvl": 7,
      "normalize": "sum"
    }

    Returns
    -------
    spectrum_ns : SimpleNamespace
        Contains:

            enabled
            mode
            wavelengths
            weights
            weights_normalized
    """
    if spectrum_config is None:
        return default_monochromatic_spectrum(default_wvl0)

    spectrum_ns = _as_namespace(spectrum_config)

    if not hasattr(spectrum_ns, "enabled"):
        spectrum_ns.enabled = True

    if not spectrum_ns.enabled:
        return default_monochromatic_spectrum(default_wvl0)

    if not hasattr(spectrum_ns, "mode"):
        spectrum_ns.mode = "flat"

    if not hasattr(spectrum_ns, "normalize"):
        spectrum_ns.normalize = "sum"

    mode = str(spectrum_ns.mode).lower()

    if mode in ["monochromatic", "single"]:
        wvl = float(getattr(spectrum_ns, "wvl0", default_wvl0))
        wavelengths = np.array([wvl], dtype=float)
        weights = np.array([1.0], dtype=float)

    elif mode == "flat":
        wvl_min = float(getattr(spectrum_ns, "wvl_min", default_wvl0))
        wvl_max = float(getattr(spectrum_ns, "wvl_max", default_wvl0))
        n_wvl = int(getattr(spectrum_ns, "n_wvl", 1))

        if n_wvl < 1:
            raise ValueError("spectrum.n_wvl must be >= 1.")

        wavelengths = np.linspace(wvl_min, wvl_max, n_wvl)
        weights = np.ones_like(wavelengths)

    elif mode == "blackbody":
        if not hasattr(spectrum_ns, "temperature_K"):
            raise ValueError(
                "spectrum.temperature_K is required for blackbody mode."
            )

        wvl_min = float(getattr(spectrum_ns, "wvl_min", default_wvl0))
        wvl_max = float(getattr(spectrum_ns, "wvl_max", default_wvl0))
        n_wvl = int(getattr(spectrum_ns, "n_wvl", 5))
        weighting = str(getattr(spectrum_ns, "weighting", "photon")).lower()

        if n_wvl < 1:
            raise ValueError("spectrum.n_wvl must be >= 1.")

        wavelengths = np.linspace(wvl_min, wvl_max, n_wvl)
        weights = blackbody_weights(
            wavelengths,
            temperature_K=float(spectrum_ns.temperature_K),
            weighting=weighting,
        )

    elif mode == "table":
        if not hasattr(spectrum_ns, "wavelengths_m"):
            raise ValueError(
                "spectrum.wavelengths_m is required for table mode."
            )
        if not hasattr(spectrum_ns, "weights"):
            raise ValueError(
                "spectrum.weights is required for table mode."
            )

        wavelengths = np.asarray(spectrum_ns.wavelengths_m, dtype=float)
        weights = np.asarray(spectrum_ns.weights, dtype=float)

        if wavelengths.shape != weights.shape:
            raise ValueError(
                "spectrum.wavelengths_m and spectrum.weights must have "
                "the same shape."
            )

    else:
        raise ValueError(f"Unsupported spectrum.mode: {spectrum_ns.mode}")

    if np.any(wavelengths <= 0):
        raise ValueError("All spectrum wavelengths must be positive.")

    if np.any(weights < 0):
        raise ValueError("All spectrum weights must be non-negative.")

    if np.sum(weights) <= 0:
        raise ValueError("Spectrum weights must have positive sum.")

    weights_normalized = normalize_weights(
        weights,
        normalize=spectrum_ns.normalize,
    )

    spectrum_ns.wavelengths = wavelengths
    spectrum_ns.weights = weights
    spectrum_ns.weights_normalized = weights_normalized

    return spectrum_ns


# ============================================================
# Phase-mask chromaticity helpers
# ============================================================

def theta_at_wavelength(optics, wavelength_m):
    """
    Return the ZWFS phase-mask phase shift at a given wavelength.

    Backwards-compatible default:

        theta_mode = "constant"

    so the function simply returns optics.theta.

    Optional physical-depth mode:

        theta_mode = "physical_depth"

    then the phase shift is calculated using util.get_phasemask_phaseshift.
    This is useful when the mask depth and material are known and the phase
    shift should vary with wavelength.

    Expected optics fields for physical_depth mode
    ----------------------------------------------
    optics.theta_mode = "physical_depth"
    optics.mask_depth = mask depth in metres
    optics.dot_material = material name accepted by util.get_phasemask_phaseshift

    Returns
    -------
    theta : float
        Phase shift in radians.
    """
    theta_mode = str(getattr(optics, "theta_mode", "constant")).lower()

    if theta_mode == "constant":
        return float(optics.theta)

    if theta_mode == "physical_depth":
        if not hasattr(optics, "mask_depth"):
            raise ValueError(
                "optics.mask_depth is required when theta_mode is "
                "'physical_depth'."
            )

        dot_material = getattr(optics, "dot_material", "N_1405")

        # util.get_phasemask_phaseshift expects wavelength and depth in um
        # in the current BaldrApp utilities implementation.
        wavelength_um = float(wavelength_m) * 1e6
        depth_um = float(optics.mask_depth) * 1e6

        return float(
            util.get_phasemask_phaseshift(
                wvl=wavelength_um,
                depth=depth_um,
                dot_material=dot_material,
            )
        )

    raise ValueError(f"Unsupported optics.theta_mode: {theta_mode}")


def phasemask_diameter_at_wavelength(optics, wavelength_m, default_wvl0):
    """
    Return the phase-mask diameter parameter at a given wavelength.

    Backwards-compatible default:

        mask_diam_mode = "lambda_over_D"

    In this mode, optics.mask_diam is interpreted exactly as current BaldrApp
    code interprets it: a diameter in lambda/D-like units. Therefore the same
    numerical value is returned at every wavelength.

    Optional physical mode:

        mask_diam_mode = "physical"

    In this mode, optics.mask_diam_phys_m is interpreted as a physical mask
    diameter. The returned lambda/D-like diameter is scaled relative to wvl0:

        mask_diam(lambda) = mask_diam(wvl0) * default_wvl0 / lambda

    This assumes the current optics.mask_diam corresponds to the physical
    mask size expressed in lambda/D units at default_wvl0.

    Returns
    -------
    mask_diam : float
        Phase-mask diameter in the convention expected by
        get_zwfs_output_field.
    """
    mode = str(getattr(optics, "mask_diam_mode", "lambda_over_D")).lower()

    if mode in ["lambda_over_d", "lambda/d", "lambda_over_D".lower()]:
        return float(optics.mask_diam)

    if mode == "physical":
        # Keep this deliberately simple and backwards compatible. We do not
        # require a physical mask diameter unless later code needs it. The
        # existing optics.mask_diam is treated as the value at default_wvl0.
        return float(optics.mask_diam) * float(default_wvl0) / float(wavelength_m)

    raise ValueError(f"Unsupported optics.mask_diam_mode: {mode}")


# ============================================================
# Convenience iteration
# ============================================================

def iter_spectrum(spectrum_ns):
    """
    Iterate over normalized spectral samples.

    Yields
    ------
    wavelength_m : float
    weight : float
    """
    for wavelength_m, weight in zip(
        spectrum_ns.wavelengths,
        spectrum_ns.weights_normalized,
    ):
        yield float(wavelength_m), float(weight)