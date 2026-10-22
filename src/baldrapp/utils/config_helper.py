"""
Configuration helpers for BaldrApp.

This is BaldrApp specific and does not include the pyZelda configuration
machinery, which is internal to the pyZelda fork that BaldrApp uses.

This module contains JSON parsing, namespace conversion, config validation,
and derived-quantity helpers. It intentionally does not import baldr_core.py,
so that baldr_core.py can safely import this module without circular imports.

The public initializer should remain in baldr_core.py, for example:

    zwfs_ns = bldr.init_zwfs_from_json(config_path)

baldr_core.py can call the helpers here to parse the config, derive Fresnel
relay quantities, derive spectral integration weights, and attach optional
runtime metadata.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


# ============================================================
# Basic JSON / namespace helpers
# ============================================================

def load_json_config(json_config):
    """
    Load a JSON configuration file.

    The file extension is not checked. A file named .toml can still be parsed
    if its contents are valid JSON, although using .json is recommended.
    """
    json_config = Path(json_config)

    with open(json_config, "r") as f:
        return json.load(f)


def dict_to_namespace_recursive(d):
    """
    Recursively convert dictionaries to SimpleNamespace objects.

    Lists are preserved as lists, except dictionaries inside lists are also
    converted to SimpleNamespace.
    """
    if isinstance(d, dict):
        return SimpleNamespace(
            **{k: dict_to_namespace_recursive(v) for k, v in d.items()}
        )

    if isinstance(d, list):
        return [dict_to_namespace_recursive(v) for v in d]

    return d


def namespace_to_dict_recursive(ns):
    """
    Recursively convert SimpleNamespace objects to dictionaries.
    """
    if isinstance(ns, SimpleNamespace):
        return {
            k: namespace_to_dict_recursive(v)
            for k, v in vars(ns).items()
        }

    if isinstance(ns, list):
        return [namespace_to_dict_recursive(v) for v in ns]

    return ns


# ============================================================
# Optics defaults
# ============================================================

def ensure_optics_defaults(optics_ns):
    """
    Add backwards-compatible optional optics defaults.

    Existing BaldrApp behaviour is:

        theta is constant with wavelength
        mask_diam is interpreted in the current lambda/D-like convention
    """
    if not hasattr(optics_ns, "theta_mode"):
        optics_ns.theta_mode = "constant"

    if not hasattr(optics_ns, "mask_diam_mode"):
        optics_ns.mask_diam_mode = "lambda_over_D"

    return optics_ns


# ============================================================
# Fresnel relay config
# ============================================================

def validate_fresnel_relay_ns(fresnel_ns):
    """
    Validate and complete a Fresnel relay namespace.
    """
    required = [
        "D_entrance",
        "f_oap",
        "f_coll",
        "z_to_mirror",
        "D_mirror",
        "f_imaging",
        "D_coldstop",
        "D_detector_pupil",
    ]

    missing = [k for k in required if not hasattr(fresnel_ns, k)]
    if missing:
        raise ValueError(
            "Missing required fresnel_relay fields: " + ", ".join(missing)
        )

    for k in required:
        value = getattr(fresnel_ns, k)
        if value is None or value <= 0:
            raise ValueError(f"fresnel_relay.{k} must be positive.")

    defaults = {
        "enabled": True,
        "edge_offset": 0.0,
        "edge_angle": 0.0,
        "coldstop_x_offset": 0.0,
        "coldstop_y_offset": 0.0,
        "pupil_misconjugation": 0.0,
        "use_nominal_pupil_conjugation": True,
        "z_mirror_to_lens": None,
    }

    for key, value in defaults.items():
        if not hasattr(fresnel_ns, key):
            setattr(fresnel_ns, key, value)

    return fresnel_ns


def derive_fresnel_relay_ns(fresnel_ns):
    """
    Add derived quantities to a generic Fresnel relay namespace.

    The relay assumes:

        D_phys = D_entrance * f_coll / f_oap

    where D_phys is the collimated beam diameter after the post-mask
    collimating lens.
    """
    fresnel_ns = validate_fresnel_relay_ns(fresnel_ns)

    fresnel_ns.D_phys = (
        fresnel_ns.D_entrance * fresnel_ns.f_coll / fresnel_ns.f_oap
    )

    fresnel_ns.M_nominal = (
        fresnel_ns.D_detector_pupil / fresnel_ns.D_phys
    )

    fresnel_ns.s_object_nominal = (
        fresnel_ns.f_imaging * (1.0 + 1.0 / fresnel_ns.M_nominal)
    )

    fresnel_ns.s_image_nominal = (
        fresnel_ns.f_imaging * (1.0 + fresnel_ns.M_nominal)
    )

    if fresnel_ns.use_nominal_pupil_conjugation:
        fresnel_ns.z_mirror_to_lens = (
            fresnel_ns.s_object_nominal - fresnel_ns.z_to_mirror
        )
    else:
        if fresnel_ns.z_mirror_to_lens is None:
            raise ValueError(
                "fresnel_relay.z_mirror_to_lens must be provided when "
                "use_nominal_pupil_conjugation is false."
            )

    fresnel_ns.s_object = (
        fresnel_ns.z_to_mirror + fresnel_ns.z_mirror_to_lens
    )

    if fresnel_ns.s_object <= fresnel_ns.f_imaging:
        raise ValueError(
            "Invalid Fresnel relay geometry: s_object must be greater "
            "than f_imaging."
        )

    fresnel_ns.s_image = 1.0 / (
        1.0 / fresnel_ns.f_imaging - 1.0 / fresnel_ns.s_object
    )

    fresnel_ns.z_focus_to_detector_nominal = (
        fresnel_ns.s_image - fresnel_ns.f_imaging
    )

    fresnel_ns.z_focus_to_detector = (
        fresnel_ns.z_focus_to_detector_nominal
        + fresnel_ns.pupil_misconjugation
    )

    if fresnel_ns.z_focus_to_detector <= 0:
        raise ValueError(
            "Invalid Fresnel relay geometry: detector is before or at "
            "the cold-stop plane."
        )

    fresnel_ns.M_pupil = fresnel_ns.s_image / fresnel_ns.s_object

    fresnel_ns.D_detector_predicted = (
        fresnel_ns.M_pupil * fresnel_ns.D_phys
    )

    return fresnel_ns


# ============================================================
# Optional metadata/runtime config sections
# ============================================================

def attach_optional_config_sections(zwfs_ns, cfg):
    """
    Attach optional non-derived configuration sections to zwfs_ns.

    Sections attached here are not interpreted by the initializer. They are
    simply made available to higher-level simulation code.
    """
    optional_sections = [
        "meta",
        "source",
        "stellar",
        "throughput",
        "internal_aberrations",
        "atmosphere",
        "first_stage_ao",
        "simulator_runtime",
    ]

    for section in optional_sections:
        if section in cfg:
            setattr(
                zwfs_ns,
                section,
                dict_to_namespace_recursive(cfg[section]),
            )

    return zwfs_ns


# ============================================================
# Stellar / spectrum config helpers
# ============================================================

def get_stellar_config_from_cfg(cfg):
    """
    Return the stellar/source configuration dictionary.

    If no stellar section is present, return an empty dictionary. This allows
    the initializer to create a minimal zwfs_ns.stellar namespace when needed.
    """
    if "stellar" in cfg and isinstance(cfg["stellar"], dict):
        return cfg["stellar"]

    return {}


def get_spectrum_config_from_cfg(cfg):
    """
    Return the user-facing spectrum configuration from a generic BaldrApp
    config dictionary.

    Preferred format
    ----------------
    The preferred format is:

        cfg["stellar"]["spectrum"]

    because the spectrum is a property of the source/filter model.

    Backwards-compatible format
    ---------------------------
    Older experimental configs may contain a top-level:

        cfg["spectrum"]

    This is still accepted as a fallback.
    """
    if "stellar" in cfg and isinstance(cfg["stellar"], dict):
        if "spectrum" in cfg["stellar"]:
            return cfg["stellar"]["spectrum"]

    if "spectrum" in cfg:
        return cfg["spectrum"]

    return None


def infer_bandwidth_from_spectrum_config(spectrum_cfg, default_nm=1.0):
    """
    Infer a scalar bandwidth in nm from a spectrum configuration dictionary.

    Priority:
        1. wvl_min and wvl_max, interpreted in metres.
        2. wavelengths or wavelengths_m, interpreted in metres.
        3. default_nm.

    Returns
    -------
    bandwidth_nm : float
        Inferred bandwidth in nm.

    source : str
        Human-readable source of the inferred bandwidth.
    """
    if spectrum_cfg is None:
        return float(default_nm), "default"

    if "wvl_min" in spectrum_cfg and "wvl_max" in spectrum_cfg:
        bandwidth_nm = (
            float(spectrum_cfg["wvl_max"]) - float(spectrum_cfg["wvl_min"])
        ) * 1e9

        if bandwidth_nm <= 0:
            raise ValueError(
                "Invalid spectrum config: wvl_max must be greater than wvl_min."
            )

        return float(bandwidth_nm), "wvl_min/wvl_max"

    wavelength_key = None

    if "wavelengths" in spectrum_cfg:
        wavelength_key = "wavelengths"
    elif "wavelengths_m" in spectrum_cfg:
        wavelength_key = "wavelengths_m"

    if wavelength_key is not None:
        wavelengths = np.asarray(spectrum_cfg[wavelength_key], dtype=float)

        if wavelengths.ndim != 1:
            raise ValueError(
                f"Invalid spectrum config: {wavelength_key} must be one-dimensional."
            )

        if len(wavelengths) > 1:
            bandwidth_nm = (np.max(wavelengths) - np.min(wavelengths)) * 1e9

            if bandwidth_nm <= 0:
                raise ValueError(
                    f"Invalid spectrum config: {wavelength_key} range must be positive."
                )

            return float(bandwidth_nm), wavelength_key

        if len(wavelengths) == 1:
            return float(default_nm), "single wavelength default"

    return float(default_nm), "default"


def ensure_stellar_bandwidth(
    stellar_ns,
    spectrum_cfg=None,
    default_nm=1.0,
    verbose=True,
):
    """
    Ensure a stellar namespace has a scalar bandwidth field in nm.

    Design convention
    -----------------
    stellar.bandwidth is the scalar effective bandwidth in nm used by legacy
    flat-spectrum detector paths.

    A detailed stellar.spectrum section is optional and is only required for
    polychromatic propagation.

    Rules
    -----
    If stellar.bandwidth exists:
        Use it.

    Else if a spectrum wavelength range exists:
        Infer bandwidth from that range and attach it to stellar.bandwidth.

    Else:
        Attach a default bandwidth, usually 1 nm for monochromatic fallback.
    """
    if stellar_ns is None:
        stellar_ns = SimpleNamespace()

    if hasattr(stellar_ns, "bandwidth"):
        bandwidth_nm = float(stellar_ns.bandwidth)

        if bandwidth_nm <= 0:
            raise ValueError("stellar.bandwidth must be positive.")

        if not hasattr(stellar_ns, "bandwidth_unit"):
            stellar_ns.bandwidth_unit = "nm"

        if verbose:
            print(
                "Config: using stellar.bandwidth = "
                f"{bandwidth_nm:.6g} nm as scalar effective bandwidth."
            )

        return stellar_ns, "stellar.bandwidth"

    bandwidth_nm, source = infer_bandwidth_from_spectrum_config(
        spectrum_cfg,
        default_nm=default_nm,
    )

    stellar_ns.bandwidth = float(bandwidth_nm)
    stellar_ns.bandwidth_unit = "nm"

    if verbose:
        print(
            "Config: inferred stellar.bandwidth = "
            f"{bandwidth_nm:.6g} nm from {source}."
        )

    return stellar_ns, source


def check_stellar_bandwidth_consistency(
    stellar_ns,
    spectrum_cfg=None,
    relative_tolerance=1e-3,
    verbose=True,
):
    """
    Check whether stellar.bandwidth is consistent with the wavelength range
    implied by the spectrum config.

    This is a warning-level check by default because there are valid cases
    where the integration support is wider than the effective filter bandwidth.
    """
    if stellar_ns is None or not hasattr(stellar_ns, "bandwidth"):
        return True

    if spectrum_cfg is None:
        return True

    stellar_bandwidth_nm = float(stellar_ns.bandwidth)

    inferred_bandwidth_nm, source = infer_bandwidth_from_spectrum_config(
        spectrum_cfg,
        default_nm=stellar_bandwidth_nm,
    )

    if source in ["default", "single wavelength default"]:
        return True

    rel_diff = abs(inferred_bandwidth_nm - stellar_bandwidth_nm) / stellar_bandwidth_nm

    if rel_diff > relative_tolerance:
        if verbose:
            print(
                "Warning: stellar.bandwidth is inconsistent with the spectrum "
                "wavelength range. "
                f"stellar.bandwidth = {stellar_bandwidth_nm:.6g} nm, "
                f"inferred range from {source} = {inferred_bandwidth_nm:.6g} nm. "
                "Using stellar.bandwidth as the authoritative effective "
                "bandwidth for detector scaling and weights_nm normalization."
            )

        return False

    if verbose:
        print(
            "Config: stellar.bandwidth is consistent with spectrum wavelength "
            f"range ({stellar_bandwidth_nm:.6g} nm)."
        )

    return True


def complete_spectrum_integration_fields(spectrum_ns):
    """
    Ensure a derived spectrum namespace has integration weights in nm.

    Required/expected fields:

        wavelengths
        weights_normalized
        weights_nm
        bandwidth_nm

    Conventions:

        weights_normalized is dimensionless and sums to 1.

        weights_nm has units of nm and should be used for integrating
        spectral-density images.
    """
    if not hasattr(spectrum_ns, "wavelengths"):
        raise ValueError("Derived spectrum namespace must contain wavelengths.")

    wavelengths_m = np.asarray(spectrum_ns.wavelengths, dtype=float)

    if wavelengths_m.ndim != 1:
        raise ValueError("spectrum.wavelengths must be one-dimensional.")

    if len(wavelengths_m) < 1:
        raise ValueError("spectrum.wavelengths must contain at least one sample.")

    if not hasattr(spectrum_ns, "weights_normalized"):
        if hasattr(spectrum_ns, "weights"):
            weights = np.asarray(spectrum_ns.weights, dtype=float)

            if np.sum(weights) <= 0:
                raise ValueError("spectrum.weights must have positive sum.")

            spectrum_ns.weights_normalized = weights / np.sum(weights)
        else:
            spectrum_ns.weights_normalized = (
                np.ones_like(wavelengths_m) / len(wavelengths_m)
            )

    weights_normalized = np.asarray(spectrum_ns.weights_normalized, dtype=float)

    if weights_normalized.shape != wavelengths_m.shape:
        raise ValueError(
            "spectrum.weights_normalized must have the same shape as wavelengths."
        )

    if np.sum(weights_normalized) <= 0:
        raise ValueError("spectrum.weights_normalized must have positive sum.")

    weights_normalized = weights_normalized / np.sum(weights_normalized)

    if not hasattr(spectrum_ns, "bandwidth_nm"):
        if len(wavelengths_m) == 1:
            spectrum_ns.bandwidth_nm = 1.0
        else:
            spectrum_ns.bandwidth_nm = (
                np.max(wavelengths_m) - np.min(wavelengths_m)
            ) * 1e9

    if not hasattr(spectrum_ns, "weights_nm"):
        spectrum_ns.weights_nm = weights_normalized * float(spectrum_ns.bandwidth_nm)

    spectrum_ns.wavelengths = wavelengths_m
    spectrum_ns.weights_normalized = weights_normalized
    spectrum_ns.weights_nm = np.asarray(spectrum_ns.weights_nm, dtype=float)

    return spectrum_ns


def apply_stellar_bandwidth_to_spectrum(
    spectrum_ns,
    stellar_ns=None,
    warn=True,
    relative_tolerance=1e-3,
):
    """
    Normalize spectrum_ns.weights_nm to the scalar stellar bandwidth.

    If stellar.bandwidth exists, this function makes it authoritative:

        spectrum_ns.bandwidth_nm = stellar.bandwidth
        spectrum_ns.weights_nm = weights_normalized * stellar.bandwidth

    If stellar.bandwidth does not exist, spectrum_ns is returned unchanged.
    """
    if stellar_ns is None:
        return spectrum_ns

    if not hasattr(stellar_ns, "bandwidth"):
        return spectrum_ns

    stellar_bandwidth_nm = float(stellar_ns.bandwidth)

    if stellar_bandwidth_nm <= 0:
        raise ValueError("stellar.bandwidth must be positive when provided.")

    if not hasattr(spectrum_ns, "weights_normalized"):
        if hasattr(spectrum_ns, "weights_nm"):
            weights_nm = np.asarray(spectrum_ns.weights_nm, dtype=float)
            total = np.sum(weights_nm)

            if total <= 0:
                raise ValueError("spectrum.weights_nm must have positive sum.")

            spectrum_ns.weights_normalized = weights_nm / total
        else:
            raise ValueError(
                "Cannot apply stellar.bandwidth: spectrum has neither "
                "weights_normalized nor weights_nm."
            )

    weights_normalized = np.asarray(spectrum_ns.weights_normalized, dtype=float)

    if weights_normalized.ndim != 1:
        raise ValueError("spectrum.weights_normalized must be one-dimensional.")

    if np.sum(weights_normalized) <= 0:
        raise ValueError("spectrum.weights_normalized must have positive sum.")

    weights_normalized = weights_normalized / np.sum(weights_normalized)

    if hasattr(spectrum_ns, "bandwidth_nm"):
        old_bandwidth_nm = float(spectrum_ns.bandwidth_nm)

        if old_bandwidth_nm > 0:
            rel_diff = abs(old_bandwidth_nm - stellar_bandwidth_nm) / stellar_bandwidth_nm

            if warn and rel_diff > relative_tolerance:
                print(
                    "Warning: derived spectrum.bandwidth_nm "
                    f"({old_bandwidth_nm:.6g} nm) differs from "
                    f"stellar.bandwidth ({stellar_bandwidth_nm:.6g} nm). "
                    "Renormalizing spectrum.weights_nm to stellar.bandwidth."
                )

    spectrum_ns.weights_normalized = weights_normalized
    spectrum_ns.bandwidth_nm = stellar_bandwidth_nm
    spectrum_ns.weights_nm = weights_normalized * stellar_bandwidth_nm

    return spectrum_ns
