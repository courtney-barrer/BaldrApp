"""Namespace and Fresnel-relay helpers for the Baldr optical initializer.

JSON validation and derived source/atmospheric properties live in
configuration.py. This module has no dependency on baldr_core.py.
"""

from types import SimpleNamespace

import numpy as np


# ============================================================
# Basic JSON / namespace helpers
# ============================================================

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
