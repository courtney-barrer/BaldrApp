#!/usr/bin/env python3

from pathlib import Path
import subprocess
import numpy as np

import pyzelda.ztools as ztools

from baldrapp.common import baldr_core as bldr
from baldrapp.common import utilities as util
from baldrapp.common import phasescreens as ps


# ============================================================
# User knobs
# ============================================================

N_FRAMES = 15
RANDOM_SEED = 10

INCLUDE_SHOTNOISE_FROM_CONFIG = True

DM_MODE = "flat"          # "flat" or "random_static"
DM_RANDOM_RMS_CMD = 0.005

OVERRIDE_DM_COUPLING = True
DM_COUPLING_FACTOR = 0.9

LOG_FLOOR = 1e-8


# ============================================================
# Helpers
# ============================================================

def get_git_root() -> Path:
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
        ).strip()
    )


def get_cfg_value(ns, name, default=None):
    return getattr(ns, name, default) if ns is not None else default


def scale_r0_to_wavelength(r0_ref_m, wavelength_m, reference_wavelength_m):
    return float(r0_ref_m) * (
        float(wavelength_m) / float(reference_wavelength_m)
    ) ** (6.0 / 5.0)


def get_photon_flux_density_from_config(zwfs):
    source_cfg = getattr(zwfs, "source", None)
    return float(
        get_cfg_value(
            source_cfg,
            "photons_per_second_per_pixel_per_nm",
            1.0e5,
        )
    )


def get_internal_opd_from_config(zwfs):
    opd_internal = np.zeros_like(zwfs.grid.pupil_mask, dtype=float)
    dx = zwfs.grid.D / zwfs.grid.N

    ia = getattr(zwfs, "internal_aberrations", None)
    if ia is None or not bool(get_cfg_value(ia, "enabled", False)):
        return opd_internal

    scratches = getattr(ia, "parabolic_scratches", None)
    if scratches is not None and bool(get_cfg_value(scratches, "enabled", False)):
        width_list = None

        if hasattr(scratches, "width_list"):
            width_list = list(scratches.width_list)
        elif hasattr(scratches, "width_list_dx_factor"):
            width_list = [float(v) * dx for v in scratches.width_list_dx_factor]

        if width_list is None:
            width_list = [2.0 * dx]

        opd_internal = util.apply_parabolic_scratches(
            opd_internal,
            dx=dx,
            dy=dx,
            list_a=list(get_cfg_value(scratches, "list_a", [0.1])),
            list_b=list(get_cfg_value(scratches, "list_b", [0.0])),
            list_c=list(get_cfg_value(scratches, "list_c", [-2.0])),
            width_list=width_list,
            depth_list=list(get_cfg_value(scratches, "depth_list_m", [100e-9])),
        )

    return zwfs.grid.pupil_mask * opd_internal


def make_first_stage_ao_basis(zwfs, n_modes_removed):
    nterms = max(50, int(n_modes_removed) + 5)

    basis_cropped = ztools.zernike.zernike_basis(
        nterms=nterms,
        npix=zwfs.grid.N,
    )

    basis_template = np.zeros_like(zwfs.grid.pupil_mask, dtype=float)

    basis = np.array([
        util.insert_concentric(np.nan_to_num(b, 0.0), basis_template)
        for b in basis_cropped
    ])

    return basis


def upsample_by_factor(arr, factor):
    arr = np.asarray(arr)
    return np.repeat(np.repeat(arr, int(factor), axis=0), int(factor), axis=1)


def upsample_to_size(arr, target_size):
    arr = np.asarray(arr)

    if arr.shape[0] == target_size:
        return arr

    factor = max(1, int(target_size) // int(arr.shape[0]))
    out = upsample_by_factor(arr, factor)

    if out.shape[0] < target_size:
        pad = target_size - out.shape[0]
        out = np.pad(out, ((0, pad), (0, pad)), mode="edge")

    if out.shape[0] > target_size:
        out = out[:target_size, :target_size]

    return out


def center_crop_or_pad_to_shape(arr, target_shape, fill_value=0.0):
    arr = np.asarray(arr)

    if arr.ndim != 2:
        raise ValueError("center_crop_or_pad_to_shape expects a 2D array.")

    ty, tx = map(int, target_shape)
    sy, sx = arr.shape

    out = np.full((ty, tx), fill_value, dtype=arr.dtype)

    src_y0 = max(0, (sy - ty) // 2)
    src_x0 = max(0, (sx - tx) // 2)
    src_y1 = min(sy, src_y0 + ty)
    src_x1 = min(sx, src_x0 + tx)

    dst_y0 = max(0, (ty - sy) // 2)
    dst_x0 = max(0, (tx - sx) // 2)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    out[dst_y0:dst_y1, dst_x0:dst_x1] = arr[src_y0:src_y1, src_x0:src_x1]

    return out


def update_scintillation_amplitude(
    scint_screen,
    zwfs,
    pxl_scale,
    wavelength_m,
    rows_per_frame=1,
    propagation_distance_m=10000.0,
    renormalize_mean_intensity=True,
):
    for _ in range(int(rows_per_frame)):
        scint_screen.add_row()

    wavefront = np.exp(1j * scint_screen.scrn)

    propagated = ps.angularSpectrum(
        inputComplexAmp=wavefront,
        z=float(propagation_distance_m),
        wvl=float(wavelength_m),
        inputSpacing=float(pxl_scale),
        outputSpacing=float(pxl_scale),
    )

    amp = np.abs(propagated)
    amp = upsample_to_size(amp, zwfs.grid.pupil_mask.shape[0])

    if renormalize_mean_intensity:
        pupil = zwfs.grid.pupil_mask.astype(bool)
        mean_intensity = np.mean(amp[pupil] ** 2)
        if mean_intensity > 0:
            amp = amp / np.sqrt(mean_intensity)

    return amp


def init_dynamic_atmosphere_for_beam(zwfs):
    wvl0 = float(zwfs.optics.wvl0)
    dx = float(zwfs.grid.D) / float(zwfs.grid.N)

    atm_cfg = getattr(zwfs, "atmosphere", None)
    phase_cfg = getattr(atm_cfg, "phase", None)
    scint_cfg = getattr(atm_cfg, "scintillation", None)
    ao_cfg = getattr(zwfs, "first_stage_ao", None)

    phase_enabled = bool(get_cfg_value(phase_cfg, "enabled", False))
    ao_enabled = bool(get_cfg_value(ao_cfg, "enabled", False))
    scint_enabled = bool(get_cfg_value(scint_cfg, "enabled", False))

    phase_screen = None
    if phase_enabled:
        r0_500_m = float(get_cfg_value(phase_cfg, "r0_500_m", 0.126))
        reference_wavelength_m = float(
            get_cfg_value(phase_cfg, "reference_wavelength_m", 500e-9)
        )
        r0_wvl_m = scale_r0_to_wavelength(
            r0_500_m,
            wvl0,
            reference_wavelength_m,
        )

        phase_screen = ps.PhaseScreenKolmogorov(
            nx_size=int(zwfs.grid.dim),
            pixel_scale=dx,
            r0=r0_wvl_m,
            L0=float(get_cfg_value(phase_cfg, "L0_m", 25.0)),
            random_seed=get_cfg_value(phase_cfg, "random_seed", None),
        )

    n_modes_removed = int(get_cfg_value(ao_cfg, "Nmodes_removed", 0))
    ao_basis = None
    if phase_enabled and ao_enabled and n_modes_removed > 0:
        ao_basis = make_first_stage_ao_basis(zwfs, n_modes_removed)

    scint_screen = None
    if scint_enabled:
        scint_r0_500_m = float(get_cfg_value(scint_cfg, "r0_500_m", 0.126))
        scint_ref_wvl_m = float(
            get_cfg_value(scint_cfg, "reference_wavelength_m", 500e-9)
        )
        scint_r0_wvl_m = scale_r0_to_wavelength(
            scint_r0_500_m,
            wvl0,
            scint_ref_wvl_m,
        )

        scint_screen = ps.PhaseScreenVonKarman(
            nx_size=int(zwfs.grid.dim),
            pixel_scale=dx,
            r0=scint_r0_wvl_m,
            L0=float(get_cfg_value(scint_cfg, "L0_m", 25.0)),
            random_seed=get_cfg_value(scint_cfg, "random_seed", None),
        )

    return {
        "dx": dx,
        "phase_enabled": phase_enabled,
        "phase_screen": phase_screen,
        "phase_rows_per_frame": int(get_cfg_value(phase_cfg, "rows_per_frame", 1)),
        "phase_scaling_factor": float(
            get_cfg_value(phase_cfg, "phase_scaling_factor", 1.0)
        ),
        "ao_enabled": ao_enabled,
        "ao_basis": ao_basis,
        "n_modes_removed": n_modes_removed,
        "ao_phase_scaling_factor": float(
            get_cfg_value(ao_cfg, "phase_scaling_factor", 1.0)
        ),
        "scint_enabled": scint_enabled,
        "scint_screen": scint_screen,
        "scint_rows_per_frame": int(get_cfg_value(scint_cfg, "rows_per_frame", 1)),
        "scint_propagation_distance_m": float(
            get_cfg_value(scint_cfg, "propagation_distance_m", 10000.0)
        ),
        "scint_renormalize_mean_intensity": bool(
            get_cfg_value(scint_cfg, "renormalize_mean_intensity", True)
        ),
    }


def generate_atmosphere_and_source_for_beam_with_diagnostics(
    zwfs,
    amp_input_0,
    dynamic_state,
):
    pupil = zwfs.grid.pupil_mask
    wvl0 = float(zwfs.optics.wvl0)

    if dynamic_state["phase_enabled"]:
        scrn = dynamic_state["phase_screen"]

        for _ in range(int(dynamic_state["phase_rows_per_frame"])):
            scrn.add_row()

        phase_scaling = float(dynamic_state["phase_scaling_factor"])

        raw_phase = pupil * phase_scaling * scrn.scrn
        opd_raw = pupil * (wvl0 / (2.0 * np.pi)) * raw_phase

        if (
            dynamic_state["ao_enabled"]
            and dynamic_state["ao_basis"] is not None
            and dynamic_state["n_modes_removed"] > 0
        ):
            phase_after_ao = bldr.first_stage_ao(
                atm_scrn=scrn,
                Nmodes_removed=dynamic_state["n_modes_removed"],
                basis=dynamic_state["ao_basis"],
                phase_scaling_factor=(
                    phase_scaling
                    * float(dynamic_state["ao_phase_scaling_factor"])
                ),
                return_reconstructor=False,
            )
        else:
            phase_after_ao = raw_phase

        opd_after_ao = pupil * (wvl0 / (2.0 * np.pi)) * phase_after_ao

    else:
        opd_raw = np.zeros_like(pupil, dtype=float)
        opd_after_ao = np.zeros_like(pupil, dtype=float)

    if dynamic_state["scint_enabled"]:
        amp_scint = update_scintillation_amplitude(
            scint_screen=dynamic_state["scint_screen"],
            zwfs=zwfs,
            pxl_scale=dynamic_state["dx"],
            wavelength_m=wvl0,
            rows_per_frame=dynamic_state["scint_rows_per_frame"],
            propagation_distance_m=dynamic_state["scint_propagation_distance_m"],
            renormalize_mean_intensity=dynamic_state["scint_renormalize_mean_intensity"],
        )
        amp_input = amp_input_0 * amp_scint
    else:
        amp_scint = np.ones_like(pupil, dtype=float)
        amp_input = amp_input_0

    return opd_after_ao, amp_input, opd_raw, amp_scint


def log10_normalized(im, floor=1e-8):
    im = np.asarray(im, dtype=float)
    scale = np.nanmax(np.abs(im))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return np.log10(np.abs(im) / scale + floor)


def rms_nm(opd, pupil):
    mask = pupil.astype(bool)
    return 1e9 * np.std(opd[mask])


def get_poly_fresnel_intermediate_intensities(
    zwfs,
    opd_input,
    amp_input,
    opd_internal,
):
    wavelengths = np.asarray(zwfs.spectrum.wavelengths, dtype=float)

    if hasattr(zwfs.spectrum, "weights_nm"):
        weights_nm = np.asarray(zwfs.spectrum.weights_nm, dtype=float)
    else:
        weights_nm = np.ones_like(wavelengths, dtype=float)

    zwfs_output_intensity = None
    detector_pre_intensity = None

    for wavelength_m, weight_nm in zip(wavelengths, weights_nm):
        out = bldr.get_frame_fresnel(
            opd_input=opd_input,
            amp_input=amp_input,
            opd_internal=opd_internal,
            zwfs_ns=zwfs,
            detector=None,
            include_shotnoise=False,
            spectral_bandwidth=None,
            wavelength=float(wavelength_m),
            return_intermediates=True,
        )

        psi_c_int = np.abs(out["psi_C"]) ** 2
        det_pre_int = np.abs(out["field_detector_pupil"]) ** 2

        if zwfs_output_intensity is None:
            zwfs_output_intensity = weight_nm * psi_c_int
            detector_pre_intensity = weight_nm * det_pre_int
        else:
            zwfs_output_intensity += weight_nm * psi_c_int
            detector_pre_intensity += weight_nm * det_pre_int

    return zwfs_output_intensity, detector_pre_intensity


# ============================================================
# Main
# ============================================================

np.random.seed(RANDOM_SEED)

repo_root = get_git_root()
config_path = repo_root / "baldrapp/apps/paranal_simulator/fake_configs/baldr_config.json"

zwfs = bldr.init_zwfs_from_json(config_path)

if OVERRIDE_DM_COUPLING:
    zwfs.dm.actuator_coupling_factor = DM_COUPLING_FACTOR
    zwfs = bldr.update_dm_registration_wavespace(
        zwfs.dm2wavespace_registration.dm_to_wavesp_transform_matrix,
        zwfs,
    )

pupil = zwfs.grid.pupil_mask
detector = zwfs.detector
opd_internal = get_internal_opd_from_config(zwfs)

flux_density = get_photon_flux_density_from_config(zwfs)
amp_input_0 = np.sqrt(flux_density) * pupil

dynamic_state = init_dynamic_atmosphere_for_beam(zwfs)

det_cfg = getattr(zwfs, "detector_config", None)
adu_offset = float(get_cfg_value(det_cfg, "adu_offset", 1000.0))
noise_std_adu = float(get_cfg_value(det_cfg, "noise_std_adu", 100.0))
include_readnoise = bool(get_cfg_value(det_cfg, "include_readnoise", True))
include_shotnoise = bool(get_cfg_value(det_cfg, "include_shotnoise", True))
include_shotnoise = include_shotnoise and INCLUDE_SHOTNOISE_FROM_CONFIG

crop_after_detection = bool(get_cfg_value(det_cfg, "crop_after_detection", False))
crop_to_pixels = get_cfg_value(det_cfg, "crop_to_pixels", None)

if DM_MODE == "flat":
    dm_cmd = zwfs.dm.dm_flat.copy()
elif DM_MODE == "random_static":
    dm_cmd = zwfs.dm.dm_flat + DM_RANDOM_RMS_CMD * np.random.randn(len(zwfs.dm.dm_flat))
else:
    raise ValueError(f"Unknown DM_MODE: {DM_MODE}")

zwfs.dm.current_cmd = dm_cmd.copy()

print("\n=== Single-beam polychromatic Fresnel diagnostic ===")
print("config_path:", config_path)
print("grid N:", zwfs.grid.N)
print("grid dim:", zwfs.grid.dim)
print("detector binning:", detector.binning)
print("crop_after_detection:", crop_after_detection)
print("crop_to_pixels:", crop_to_pixels)
print("stellar.bandwidth [nm]:", getattr(zwfs.stellar, "bandwidth", None))
print("spectrum wavelengths [um]:", zwfs.spectrum.wavelengths * 1e6)
print("spectrum weights_nm:", zwfs.spectrum.weights_nm)
print("sum spectrum weights_nm:", np.sum(zwfs.spectrum.weights_nm))
print("fresnel enabled:", getattr(zwfs.fresnel_relay, "enabled", None))
print("phase enabled:", dynamic_state["phase_enabled"])
print("AO enabled:", dynamic_state["ao_enabled"])
print("AO modes removed:", dynamic_state["n_modes_removed"])
print("scintillation enabled:", dynamic_state["scint_enabled"])
print("flux density [phot/s/wave-pix/nm]:", flux_density)
print("DM mode:", DM_MODE)
print("DM RMS from flat:", np.std(zwfs.dm.current_cmd - zwfs.dm.dm_flat))


# ============================================================
# Generate frames
# ============================================================

raw_opd_nm_list = []
ao_opd_nm_list = []
scint_amp_list = []
dm_cmd_list = []

zwfs_output_log_list = []
det_pre_log_list = []
det_counts_list = []
camera_adu_list = []

rms_raw_list = []
rms_ao_list = []
flux_counts_list = []
flux_camera_list = []

for k in range(N_FRAMES):

    zwfs.dm.current_cmd = dm_cmd.copy()

    opd_after_ao, amp_input, opd_raw, amp_scint = (
        generate_atmosphere_and_source_for_beam_with_diagnostics(
            zwfs=zwfs,
            amp_input_0=amp_input_0,
            dynamic_state=dynamic_state,
        )
    )

    zwfs_output_intensity, detector_pre_intensity = (
        get_poly_fresnel_intermediate_intensities(
            zwfs=zwfs,
            opd_input=opd_after_ao,
            amp_input=amp_input,
            opd_internal=opd_internal,
        )
    )

    detected_counts = bldr.get_frame_configured(
        opd_input=opd_after_ao,
        amp_input=amp_input,
        opd_internal=opd_internal,
        zwfs_ns=zwfs,
        detector=detector,
        include_shotnoise=include_shotnoise,
        use_pyZelda=False,
    ).astype(float)

    if crop_after_detection and crop_to_pixels is not None:
        detected_counts = center_crop_or_pad_to_shape(
            detected_counts,
            tuple(crop_to_pixels),
        )

    camera_adu = adu_offset + detected_counts
    if include_readnoise and noise_std_adu > 0:
        camera_adu = camera_adu + noise_std_adu * np.random.randn(*camera_adu.shape)

    raw_opd_nm_list.append(1e9 * opd_raw)
    ao_opd_nm_list.append(1e9 * opd_after_ao)
    scint_amp_list.append(amp_scint)
    dm_cmd_list.append(util.get_DM_command_in_2D(zwfs.dm.current_cmd))

    zwfs_output_log_list.append(log10_normalized(zwfs_output_intensity, floor=LOG_FLOOR))
    det_pre_log_list.append(log10_normalized(detector_pre_intensity, floor=LOG_FLOOR))
    det_counts_list.append(detected_counts)
    camera_adu_list.append(camera_adu)

    rms_raw = rms_nm(opd_raw, pupil)
    rms_ao = rms_nm(opd_after_ao, pupil)
    flux_counts = float(np.nansum(detected_counts))
    flux_camera = float(np.nansum(camera_adu))

    rms_raw_list.append(rms_raw)
    rms_ao_list.append(rms_ao)
    flux_counts_list.append(flux_counts)
    flux_camera_list.append(flux_camera)

    print(
        f"frame {k:02d}: "
        f"raw OPD RMS={rms_raw:8.2f} nm, "
        f"AO OPD RMS={rms_ao:8.2f} nm, "
        f"det counts sum={flux_counts:.3e}, "
        f"camera ADU sum={flux_camera:.3e}"
    )


# ============================================================
# Slider display
# ============================================================

image_lists = [
    raw_opd_nm_list,
    ao_opd_nm_list,
    scint_amp_list,
    dm_cmd_list,
    zwfs_output_log_list,
    det_pre_log_list,
    det_counts_list,
    camera_adu_list,
]

plot_titles = [
    "Raw atmosphere OPD [nm]",
    "After first-stage AO OPD [nm]",
    "Scintillation amplitude",
    "Manual DM command",
    "ZWFS output |psi_C|²\nlog10 normalized",
    "Detector pupil intensity before detection\nlog10 normalized",
    "Detected counts\npost detector + crop",
    "Camera image [ADU]\ncounts + offset/read noise",
]

cbar_labels = [
    "nm",
    "nm",
    "amplitude",
    "cmd",
    "log10 norm.",
    "log10 norm.",
    "counts",
    "ADU",
]

util.display_images_with_slider(
    image_lists=image_lists,
    plot_titles=plot_titles,
    cbar_labels=cbar_labels,
    row_col=(2, 4),
)

print("\nDone. Slider diagnostic generated.")