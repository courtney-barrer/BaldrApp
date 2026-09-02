#!/usr/bin/env python3
"""Generate deterministic clear-pupil and Baldr reference FITS images.

The input JSON supplies the spectrum, source flux, pupil, Fresnel-relay
alignment, phase-mask selection/properties, DM model, and detector sampling.
Atmosphere, first-stage AO, random DM-flat error, and detector noise are
disabled because the output is a theoretical reference rather than a noisy
measurement.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Baldr JSON configuration")
    parser.add_argument("output", type=Path, help="Output FITS filename")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists",
    )
    return parser.parse_args()


def find_repo_root(config_path: Path) -> Path:
    """Find the checkout containing the baldrapp package."""
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "baldrapp" / "common" / "baldr_core.py").is_file():
            return candidate

    script_root = Path(__file__).resolve().parents[3]
    if (script_root / "baldrapp" / "common" / "baldr_core.py").is_file():
        return script_root

    raise RuntimeError(
        "Could not locate the BaldrApp repository. Place this script inside "
        "the checkout or pass a configuration located inside it."
    )


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def deterministic_config(config: dict) -> tuple[dict, str]:
    """Return a copy configured for a noise-free reference exposure."""
    result = copy.deepcopy(config)

    source_name = result.get("reference_source", {}).get("profile", "configured")
    if source_name != "configured":
        try:
            source_profile = result["source_profiles"][source_name]
        except KeyError as error:
            raise ValueError(
                f"Unknown reference source profile {source_name!r}; expected one of "
                f"{sorted(result.get('source_profiles', {}))}"
            ) from error

        flux_model = source_profile.get("flux_model", "photon_density")
        if flux_model != "photon_density":
            raise ValueError(
                f"Unsupported flux model {flux_model!r} in source profile "
                f"{source_name!r}"
            )
        result["source"]["flux_mode"] = "photons_per_second_per_pixel_per_nm"
        result["source"]["photons_per_second_per_pixel_per_nm"] = float(
            source_profile["photons_per_second_per_pixel_per_nm"]
        )
        result["stellar"]["spectrum"]["enabled"] = True
        result["stellar"]["spectrum"]["mode"] = "blackbody"
        result["stellar"]["spectrum"]["temperature_K"] = float(
            source_profile["temperature_K"]
        )

    for section in ("internal_aberrations", "atmosphere", "first_stage_ao"):
        if section in result:
            result[section]["enabled"] = False

    if "dm" in result:
        result["dm"]["flat_rmse"] = 0.0

    detector = result.setdefault("detector", {})
    detector["enabled"] = True
    detector["include_shotnoise"] = False
    detector["include_readnoise"] = False
    detector["ron"] = 0.0
    detector["adu_offset"] = 0.0
    detector["noise_std_adu"] = 0.0
    return result, source_name


def fits_header(
    config: dict, mask_name: str, source_name: str, wavelengths: np.ndarray
) -> fits.Header:
    header = fits.Header()
    header["BUNIT"] = ("adu", "Detector output units")
    header["SOURCE"] = (source_name, "Selected source profile")
    header["MASKNAME"] = (mask_name, "Active focal-plane phase mask")
    header["NWAVE"] = (len(wavelengths), "Number of wavelength samples")
    header["WAVEMIN"] = (float(wavelengths.min()), "Minimum wavelength [m]")
    header["WAVEMAX"] = (float(wavelengths.max()), "Maximum wavelength [m]")
    header["WAVE0"] = (float(config["optics"]["wvl0"]), "Reference wavelength [m]")

    relay = config.get("fresnel_relay", {})
    header["FRESNEL"] = (bool(relay.get("enabled", False)), "Fresnel relay enabled")
    for keyword, key, comment in (
        ("EDGEOFF", "edge_offset", "D-mirror edge offset [m]"),
        ("EDGEANGL", "edge_angle", "D-mirror edge angle [rad]"),
        ("CSTOPX", "coldstop_x_offset", "Cold-stop x offset [m]"),
        ("CSTOPY", "coldstop_y_offset", "Cold-stop y offset [m]"),
        ("PUPMIS", "pupil_misconjugation", "Pupil misconjugation [m]"),
    ):
        if key in relay:
            header[keyword] = (float(relay[key]), comment)
    return header


def crop_detector_image(image: np.ndarray, crop_to_pixels: list[int]) -> np.ndarray:
    """Apply the configured centred post-detection crop.

    The direct configured-frame helpers currently return the binned detector
    plane without applying ``crop_after_detection`` in every propagation path.
    Applying it here makes this standalone product follow the detector config.
    """
    if len(crop_to_pixels) != 2:
        raise ValueError("detector.crop_to_pixels must contain [height, width].")

    crop_height, crop_width = (int(value) for value in crop_to_pixels)
    image_height, image_width = image.shape
    if crop_height <= 0 or crop_width <= 0:
        raise ValueError("detector.crop_to_pixels values must be positive.")
    if crop_height > image_height or crop_width > image_width:
        raise ValueError(
            f"Requested detector crop {(crop_height, crop_width)} exceeds "
            f"the binned image shape {image.shape}."
        )

    row_start = (image_height - crop_height) // 2
    column_start = (image_width - crop_width) // 2
    return image[
        row_start : row_start + crop_height,
        column_start : column_start + crop_width,
    ]


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    with config_path.open() as stream:
        original_config = json.load(stream)

    repo_root = find_repo_root(config_path)
    sys.path.insert(0, str(repo_root))

    from baldrapp.common import baldr_core as bldr
    from baldrapp.common import spectrum as spec

    config, source_name = deterministic_config(original_config)
    with tempfile.TemporaryDirectory() as temporary_directory:
        model_config_path = Path(temporary_directory) / "reference_config.json"
        with model_config_path.open("w") as stream:
            json.dump(config, stream)
        zwfs = bldr.init_zwfs_from_json(model_config_path)

    mask_runtime = original_config["simulator_runtime"]["phasemask"]
    mask_name = mask_runtime["default_mask"]
    properties_path = resolve_repo_path(repo_root, mask_runtime["properties_file"])
    with properties_path.open() as stream:
        properties = json.load(stream)
    mask_entry = properties["phasemask"]["masks"][mask_name]
    zwfs.optics.active_phasemask = spec.normalise_phasemask_entry(
        mask_name, mask_entry, zwfs.optics
    )

    flux_density = float(config["source"]["photons_per_second_per_pixel_per_nm"])
    amplitude = np.sqrt(flux_density) * zwfs.grid.pupil_mask.astype(float)
    zero_opd = np.zeros_like(amplitude)
    zwfs.dm.current_cmd = zwfs.dm.dm_flat.copy()

    clear_image = bldr.get_N0_configured(
        opd_input=zero_opd,
        amp_input=amplitude,
        opd_internal=zero_opd,
        zwfs_ns=zwfs,
        detector=zwfs.detector,
        include_shotnoise=False,
        force_fresnel=True,
        force_polychromatic=True,
    )
    masked_image = bldr.get_frame_configured(
        opd_input=zero_opd,
        amp_input=amplitude,
        opd_internal=zero_opd,
        zwfs_ns=zwfs,
        detector=zwfs.detector,
        include_shotnoise=False,
        force_fresnel=True,
        force_polychromatic=True,
    )

    clear_image = np.asarray(clear_image, dtype=np.float64)
    masked_image = np.asarray(masked_image, dtype=np.float64)
    for name, image in (("clear-pupil", clear_image), ("phase-mask", masked_image)):
        if image.ndim != 2 or not np.all(np.isfinite(image)):
            raise RuntimeError(f"The {name} model did not return a finite 2-D image.")

    crop_shape = config["detector"].get("crop_to_pixels", [32, 32])
    if config["detector"].get("crop_after_detection", True):
        clear_image = crop_detector_image(clear_image, crop_shape)
        masked_image = crop_detector_image(masked_image, crop_shape)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    common_header = fits_header(
        config,
        mask_name,
        source_name,
        np.asarray(zwfs.spectrum.wavelengths, dtype=float),
    )
    clear_header = common_header.copy()
    clear_header["EXTNAME"] = ("CLEAR_PUPIL", "No focal-plane phase mask")
    masked_header = common_header.copy()
    masked_header["EXTNAME"] = ("PHASE_MASK", "Configured phase mask inserted")
    fits.HDUList(
        [
            fits.PrimaryHDU(data=clear_image, header=clear_header),
            fits.ImageHDU(data=masked_image, header=masked_header, name="PHASE_MASK"),
        ]
    ).writeto(output_path, overwrite=args.overwrite)
    print(output_path)


if __name__ == "__main__":
    main()
