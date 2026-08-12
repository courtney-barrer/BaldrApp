#!/usr/bin/env python3
"""
Paper-grade chromatic Baldr I0 reference-bias analysis for the physical H-band masks.

This keeps the original analysis unchanged, but improves the plotting:
- legend labels use physical mask size at wvl0 in lambda/D (and um);
- traces are coloured by mask diameter with a continuous colormap;
- the original 2x2 summary figure is retained;
- each panel is also saved as a standalone figure;
- the delta-T scan uses a fixed internal-source I2M(T1).
"""

import copy
import csv
import hashlib
import json
import platform
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.ndimage import binary_erosion

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from baldrapp.common import DM_basis
from baldrapp.common import baldr_core as bldr
from baldrapp.common import spectrum as spec


# ============================================================
# Strict local-import check
# ============================================================

for module in (bldr, spec, DM_basis):
    module_path = Path(module.__file__).resolve()

    if REPO_ROOT not in module_path.parents:
        raise ImportError(
            "BaldrApp import shadowing detected.\n"
            f"Expected modules below: {REPO_ROOT}\n"
            f"Imported instead: {module_path}\n"
            "Run from the repository root, or use PYTHONPATH=$PWD."
        )

print(f"Python executable: {Path(sys.executable).resolve()}")
print(f"Python version: {platform.python_version()}")
print(f"baldr_core: {Path(bldr.__file__).resolve()}")
print(f"spectrum: {Path(spec.__file__).resolve()}")
print(f"DM_basis: {Path(DM_basis.__file__).resolve()}")


# ============================================================
# Settings
# ============================================================

CONFIG_PATH = (
    REPO_ROOT
    / "baldrapp/playground/publication/baldr_config.json"#"baldrapp/apps/paranal_simulator/fake_configs/baldr_config.json"
)

T_INTERNAL_K = 1900.0
T_ONSKY_K = 10000.0

MASK_NAMES = ["H1", "H2", "H3", "H4", "H5"]

N_ZERNIKE_MODES = 20
LINEAR_POKE_NM = 10.0
SVD_RELATIVE_CUTOFF = 1e-3

N_RADIAL_BINS = 10
CENTER_MAX_RHO = 0.4
EDGE_MIN_RHO = 0.8

# Temperature-difference scan. T1 is fixed at T_INTERNAL_K and
# T2 = T1 + delta_T. Zero is included explicitly; positive values are
# logarithmically spaced to 20,000 K.
DELTA_T_MAX_K = 20000.0
N_DELTA_T_LOG_SAMPLES = 18
DELTA_T_VALUES_K = np.concatenate(
    (
        np.array([0.0]),
        np.geomspace(1.0, DELTA_T_MAX_K, N_DELTA_T_LOG_SAMPLES),
    )
)

OUTPUT_TABLE_CSV = Path("chromatic_I0_bias_table.csv")
OUTPUT_TABLE_TEX = Path("chromatic_I0_bias_table.tex")
OUTPUT_DATA = Path("chromatic_I0_bias_results.npz")
OUTPUT_PROVENANCE = Path("chromatic_I0_bias_provenance.json")

OUTPUT_COMBINED_FIGURE = Path("chromatic_I0_bias_radial_profiles_paper.png")
OUTPUT_SUMMARY_FIGURE = Path("chromatic_I0_bias_summary_paper.png")

OUTPUT_SINGLE_MEAN_SIGNAL = Path("chromatic_I0_bias_signal_mean_radial.png")
OUTPUT_SINGLE_RMS_SIGNAL = Path("chromatic_I0_bias_signal_rms_radial.png")
OUTPUT_SINGLE_MEAN_WFE = Path("chromatic_I0_bias_wfe_mean_radial.png")
OUTPUT_SINGLE_RMS_WFE = Path("chromatic_I0_bias_wfe_rms_radial.png")

OUTPUT_DELTA_T_FIGURE = Path(
    "chromatic_I0_peak_WFE_bias_vs_deltaT.png"
)
OUTPUT_DELTA_T_CSV = Path(
    "chromatic_I0_peak_WFE_bias_vs_deltaT.csv"
)


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 2.2,
        "lines.markersize": 6.0,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.7,
        "mathtext.default": "regular",
    }
)


# ============================================================
# Initialize identical aligned states with different spectra
# ============================================================

with open(CONFIG_PATH, "r") as f:
    base_cfg = json.load(f)

cfg_internal = copy.deepcopy(base_cfg)
cfg_onsky = copy.deepcopy(base_cfg)

for cfg, temperature_K in (
    (cfg_internal, T_INTERNAL_K),
    (cfg_onsky, T_ONSKY_K),
):
    cfg["stellar"]["spectrum"]["enabled"] = True
    cfg["stellar"]["spectrum"]["mode"] = "blackbody"
    cfg["stellar"]["spectrum"]["temperature_K"] = temperature_K

    cfg["fresnel_relay"]["coldstop_x_offset"] = 0.0
    cfg["fresnel_relay"]["coldstop_y_offset"] = 0.0
    cfg["fresnel_relay"]["pupil_misconjugation"] = 0.0
    cfg["fresnel_relay"]["edge_offset"] = 0.0
    cfg["fresnel_relay"]["edge_angle"] = 0.0
    cfg["fresnel_relay"]["use_nominal_pupil_conjugation"] = True

    cfg["internal_aberrations"]["enabled"] = False

    cfg["detector"]["enabled"] = True
    cfg["detector"]["ron"] = 0.0
    cfg["detector"]["include_shotnoise"] = False
    cfg["detector"]["include_readnoise"] = False
    cfg["detector"]["adu_offset"] = 0.0
    cfg["detector"]["noise_std_adu"] = 0.0

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    internal_path = tmp / "internal_1900K.json"
    onsky_path = tmp / "onsky_10000K.json"

    with open(internal_path, "w") as f:
        json.dump(cfg_internal, f, indent=2)

    with open(onsky_path, "w") as f:
        json.dump(cfg_onsky, f, indent=2)

    zwfs_internal = bldr.init_zwfs_from_json(internal_path)
    zwfs_onsky = bldr.init_zwfs_from_json(onsky_path)


# ============================================================
# Physical masks, pupil masks, and exact matched N0 references
# ============================================================

phasemask_path = (
    REPO_ROOT
    / base_cfg["simulator_runtime"]["phasemask"]["properties_file"]
)

with open(phasemask_path, "r") as f:
    phasemask_cfg = json.load(f)

mask_entries = phasemask_cfg["phasemask"]["masks"]

amp_internal = zwfs_internal.grid.pupil_mask.astype(float)
amp_onsky = zwfs_onsky.grid.pupil_mask.astype(float)

zero_internal = np.zeros_like(amp_internal)
zero_onsky = np.zeros_like(amp_onsky)

N0_internal = bldr.get_N0_configured(
    opd_input=zero_internal,
    amp_input=amp_internal,
    opd_internal=zero_internal,
    zwfs_ns=zwfs_internal,
    detector=zwfs_internal.detector,
    include_shotnoise=False,
)

N0_onsky = bldr.get_N0_configured(
    opd_input=zero_onsky,
    amp_input=amp_onsky,
    opd_internal=zero_onsky,
    zwfs_ns=zwfs_onsky,
    detector=zwfs_onsky.detector,
    include_shotnoise=False,
)

binning = int(base_cfg["detector"]["binning"])

pupil_detector = (
    bldr.sum_subarrays(
        zwfs_onsky.grid.pupil_mask,
        block_size=(binning, binning),
    )
    > 0.5 * binning**2
)

analysis_mask = binary_erosion(pupil_detector, iterations=1)

if not np.any(analysis_mask):
    raise RuntimeError("The eroded detector pupil mask is empty.")

N0_internal_mean = np.mean(N0_internal[analysis_mask])
N0_onsky_mean = np.mean(N0_onsky[analysis_mask])

yy_det, xx_det = np.indices(pupil_detector.shape)
cy_det = np.mean(yy_det[pupil_detector])
cx_det = np.mean(xx_det[pupil_detector])

r_det = np.sqrt((xx_det - cx_det) ** 2 + (yy_det - cy_det) ** 2)
outer_radius_det = np.percentile(r_det[pupil_detector], 99.5)
rho_detector = r_det / outer_radius_det

pupil_wave = zwfs_onsky.grid.pupil_mask.astype(bool)

yy_wave, xx_wave = np.indices(pupil_wave.shape)
cy_wave = np.mean(yy_wave[pupil_wave])
cx_wave = np.mean(xx_wave[pupil_wave])

r_wave = np.sqrt((xx_wave - cx_wave) ** 2 + (yy_wave - cy_wave) ** 2)
outer_radius_wave = np.percentile(r_wave[pupil_wave], 99.5)
rho_wave = r_wave / outer_radius_wave
theta_wave = np.arctan2(yy_wave - cy_wave, xx_wave - cx_wave)

radial_edges = np.linspace(0.0, 1.0, N_RADIAL_BINS + 1)
radial_centres = 0.5 * (radial_edges[:-1] + radial_edges[1:])


# ============================================================
# Common phase basis: piston removed, each mode = 1 nm RMS OPD
# ============================================================

raw_zernikes = DM_basis.zernike_basis(
    nterms=N_ZERNIKE_MODES + 1,
    rho=rho_wave,
    theta=theta_wave,
    outside=0.0,
)

modes = []

for mode_index in range(1, N_ZERNIKE_MODES + 1):
    mode = np.nan_to_num(raw_zernikes[mode_index], nan=0.0)
    mode *= pupil_wave
    mode -= np.mean(mode[pupil_wave])

    mode_rms = np.sqrt(np.mean(mode[pupil_wave] ** 2))

    if mode_rms <= 0:
        raise RuntimeError(f"Zernike mode {mode_index + 1} has zero RMS.")

    modes.append(mode / mode_rms)

modes = np.asarray(modes)


# ============================================================
# Per-mask signal bias and on-sky linear reconstruction bias
# ============================================================

table_rows = []

signal_bias_maps = []
reconstructed_opd_maps_nm = []
fitted_signal_maps = []
I0_internal_norm_by_mask = []

radial_signal_mean = np.full((len(MASK_NAMES), N_RADIAL_BINS), np.nan)
radial_signal_rms = np.full((len(MASK_NAMES), N_RADIAL_BINS), np.nan)
radial_wfe_mean_nm = np.full((len(MASK_NAMES), N_RADIAL_BINS), np.nan)
radial_wfe_rms_nm = np.full((len(MASK_NAMES), N_RADIAL_BINS), np.nan)

mask_diam_um_list = []
mask_diam_lambdaD_list = []
mask_labels = []

for mask_number, mask_name in enumerate(MASK_NAMES):
    print(f"Processing {mask_name} ({mask_number + 1}/{len(MASK_NAMES)})")

    mask_entry = mask_entries[mask_name]

    active_mask_internal = spec.normalise_phasemask_entry(
        mask_name,
        mask_entry,
        zwfs_internal.optics,
    )

    active_mask_onsky = spec.normalise_phasemask_entry(
        mask_name,
        mask_entry,
        zwfs_onsky.optics,
    )

    zwfs_internal.optics.active_phasemask = active_mask_internal
    zwfs_onsky.optics.active_phasemask = active_mask_onsky

    mask_diam_um = float(active_mask_onsky.mask_diam_um)
    mask_diam_lambdaD = float(active_mask_onsky.mask_diam_lambdaD_wvl0)

    mask_diam_um_list.append(mask_diam_um)
    mask_diam_lambdaD_list.append(mask_diam_lambdaD)
    mask_labels.append(
        rf"{mask_diam_lambdaD:.2f}$\,\lambda/D$ ({mask_diam_um:.0f}$\,\mu$m)"
    )

    I0_internal = bldr.get_I0_configured(
        opd_input=zero_internal,
        amp_input=amp_internal,
        opd_internal=zero_internal,
        zwfs_ns=zwfs_internal,
        detector=zwfs_internal.detector,
        include_shotnoise=False,
    )

    I0_onsky = bldr.get_I0_configured(
        opd_input=zero_onsky,
        amp_input=amp_onsky,
        opd_internal=zero_onsky,
        zwfs_ns=zwfs_onsky,
        detector=zwfs_onsky.detector,
        include_shotnoise=False,
    )

    I0_internal_norm = I0_internal / N0_internal_mean
    I0_onsky_norm = I0_onsky / N0_onsky_mean

    I0_internal_norm_by_mask.append(I0_internal_norm.copy())

    signal_bias = I0_onsky_norm - I0_internal_norm
    signal_vector = signal_bias[analysis_mask]

    signal_peak_abs = np.max(np.abs(signal_vector))
    signal_rmse = np.sqrt(np.mean(signal_vector**2))

    response_columns = []

    for mode in modes:
        opd_plus = LINEAR_POKE_NM * 1e-9 * mode
        opd_minus = -LINEAR_POKE_NM * 1e-9 * mode

        I_plus = bldr.get_I0_configured(
            opd_input=opd_plus,
            amp_input=amp_onsky,
            opd_internal=zero_onsky,
            zwfs_ns=zwfs_onsky,
            detector=zwfs_onsky.detector,
            include_shotnoise=False,
        )

        I_minus = bldr.get_I0_configured(
            opd_input=opd_minus,
            amp_input=amp_onsky,
            opd_internal=zero_onsky,
            zwfs_ns=zwfs_onsky,
            detector=zwfs_onsky.detector,
            include_shotnoise=False,
        )

        I_plus_norm = I_plus / N0_onsky_mean
        I_minus_norm = I_minus / N0_onsky_mean

        derivative_per_nm = (
            I_plus_norm - I_minus_norm
        ) / (2.0 * LINEAR_POKE_NM)

        response_columns.append(derivative_per_nm[analysis_mask])

    interaction_matrix = np.column_stack(response_columns)

    U, singular_values, Vt = np.linalg.svd(
        interaction_matrix,
        full_matrices=False,
    )

    keep = singular_values > (
        SVD_RELATIVE_CUTOFF * singular_values[0]
    )

    if not np.any(keep):
        raise RuntimeError(f"No singular values retained for {mask_name}.")

    reconstructor = (
        Vt[keep].T
        @ np.diag(1.0 / singular_values[keep])
        @ U[:, keep].T
    )

    coefficients_nm = reconstructor @ signal_vector

    reconstructed_opd_nm = np.sum(
        coefficients_nm[:, None, None] * modes,
        axis=0,
    )

    fitted_signal_vector = interaction_matrix @ coefficients_nm
    fitted_signal_map = np.full_like(signal_bias, np.nan, dtype=float)
    fitted_signal_map[analysis_mask] = fitted_signal_vector

    signal_power = np.sum(signal_vector**2)
    residual_power = np.sum(
        (signal_vector - fitted_signal_vector) ** 2
    )

    explained_fraction = (
        1.0 - residual_power / signal_power
        if signal_power > 0
        else np.nan
    )

    wfe_values_nm = reconstructed_opd_nm[pupil_wave]
    equivalent_wfe_rms_nm = np.sqrt(np.mean(wfe_values_nm**2))
    peak_abs_wfe_nm = np.max(np.abs(wfe_values_nm))

    centre_mask_wave = pupil_wave & (rho_wave < CENTER_MAX_RHO)
    edge_mask_wave = (
        pupil_wave
        & (rho_wave >= EDGE_MIN_RHO)
        & (rho_wave <= 1.0)
    )

    centre_wfe_rms_nm = np.sqrt(
        np.mean(reconstructed_opd_nm[centre_mask_wave] ** 2)
    )

    edge_wfe_rms_nm = np.sqrt(
        np.mean(reconstructed_opd_nm[edge_mask_wave] ** 2)
    )

    for radial_index in range(N_RADIAL_BINS):
        detector_annulus = (
            analysis_mask
            & (rho_detector >= radial_edges[radial_index])
            & (rho_detector < radial_edges[radial_index + 1])
        )

        wave_annulus = (
            pupil_wave
            & (rho_wave >= radial_edges[radial_index])
            & (rho_wave < radial_edges[radial_index + 1])
        )

        if np.any(detector_annulus):
            radial_values = signal_bias[detector_annulus]

            radial_signal_mean[mask_number, radial_index] = np.mean(
                radial_values
            )

            radial_signal_rms[mask_number, radial_index] = np.sqrt(
                np.mean(radial_values**2)
            )

        if np.any(wave_annulus):
            radial_values_nm = reconstructed_opd_nm[wave_annulus]

            radial_wfe_mean_nm[mask_number, radial_index] = np.mean(
                radial_values_nm
            )

            radial_wfe_rms_nm[mask_number, radial_index] = np.sqrt(
                np.mean(radial_values_nm**2)
            )

    condition_number_retained = (
        singular_values[keep][0] / singular_values[keep][-1]
    )

    table_rows.append(
        {
            "mask": mask_name,
            "mask_diameter_um": float(mask_diam_um),
            "mask_diameter_lambda_over_D": float(mask_diam_lambdaD),
            "peak_abs_signal_bias": float(signal_peak_abs),
            "signal_bias_rmse": float(signal_rmse),
            "equivalent_wfe_rms_nm": float(equivalent_wfe_rms_nm),
            "peak_abs_wfe_nm": float(peak_abs_wfe_nm),
            "centre_wfe_rms_nm": float(centre_wfe_rms_nm),
            "edge_wfe_rms_nm": float(edge_wfe_rms_nm),
            "explained_signal_percent": float(
                100.0 * explained_fraction
            ),
            "retained_rank": int(np.sum(keep)),
            "retained_condition_number": float(
                condition_number_retained
            ),
        }
    )

    signal_bias_maps.append(signal_bias)
    reconstructed_opd_maps_nm.append(reconstructed_opd_nm)
    fitted_signal_maps.append(fitted_signal_map)

signal_bias_maps = np.asarray(signal_bias_maps)
reconstructed_opd_maps_nm = np.asarray(reconstructed_opd_maps_nm)
fitted_signal_maps = np.asarray(fitted_signal_maps)
I0_internal_norm_by_mask = np.asarray(I0_internal_norm_by_mask)

mask_diam_um_array = np.asarray(mask_diam_um_list)
mask_diam_lambdaD_array = np.asarray(mask_diam_lambdaD_list)

# Preserve the exact H1--H5 calculation order. The physical diameters are
# already monotonic, so no numerical arrays need to be reordered for plotting.
if not np.all(np.diff(mask_diam_lambdaD_array) > 0):
    raise RuntimeError(
        "MASK_NAMES are not ordered by increasing physical mask diameter."
    )

cmap = plt.get_cmap("viridis")
norm = Normalize(
    vmin=float(np.min(mask_diam_lambdaD_array)),
    vmax=float(np.max(mask_diam_lambdaD_array)),
)
trace_colours = [cmap(norm(v)) for v in mask_diam_lambdaD_array]


# ============================================================
# Fixed internal-source reconstructors for the delta-T scan
# ============================================================

I2M_T1_by_mask = []

for mask_number, mask_name in enumerate(MASK_NAMES):
    print(
        f"Building fixed internal-source I2M for "
        f"{mask_name} ({mask_number + 1}/{len(MASK_NAMES)})"
    )

    mask_entry = mask_entries[mask_name]

    zwfs_internal.optics.active_phasemask = (
        spec.normalise_phasemask_entry(
            mask_name,
            mask_entry,
            zwfs_internal.optics,
        )
    )

    response_columns_T1 = []

    for mode in modes:
        opd_plus = LINEAR_POKE_NM * 1e-9 * mode
        opd_minus = -LINEAR_POKE_NM * 1e-9 * mode

        I_plus = bldr.get_I0_configured(
            opd_input=opd_plus,
            amp_input=amp_internal,
            opd_internal=zero_internal,
            zwfs_ns=zwfs_internal,
            detector=zwfs_internal.detector,
            include_shotnoise=False,
        )

        I_minus = bldr.get_I0_configured(
            opd_input=opd_minus,
            amp_input=amp_internal,
            opd_internal=zero_internal,
            zwfs_ns=zwfs_internal,
            detector=zwfs_internal.detector,
            include_shotnoise=False,
        )

        response_columns_T1.append(
            (
                (I_plus - I_minus)
                / N0_internal_mean
                / (2.0 * LINEAR_POKE_NM)
            )[analysis_mask]
        )

    interaction_matrix_T1 = np.column_stack(
        response_columns_T1
    )

    U_T1, singular_values_T1, Vt_T1 = np.linalg.svd(
        interaction_matrix_T1,
        full_matrices=False,
    )

    keep_T1 = singular_values_T1 > (
        SVD_RELATIVE_CUTOFF * singular_values_T1[0]
    )

    if not np.any(keep_T1):
        raise RuntimeError(
            f"No singular values retained for {mask_name} at T1."
        )

    I2M_T1 = (
        Vt_T1[keep_T1].T
        @ np.diag(1.0 / singular_values_T1[keep_T1])
        @ U_T1[:, keep_T1].T
    )

    I2M_T1_by_mask.append(I2M_T1)

I2M_T1_by_mask = tuple(I2M_T1_by_mask)


# ============================================================
# Worst-case peak WFE bias versus source-temperature difference
# ============================================================

# Operational result:
#   fixed internal calibration, I2M(T1)
#
# Diagnostic comparison:
#   perfectly matched on-sky interaction matrix, I2M(T2)

peak_wfe_bias_fixed_I2M_nm = np.zeros(
    (len(MASK_NAMES), len(DELTA_T_VALUES_K)),
    dtype=float,
)

peak_wfe_bias_matched_I2M_nm = np.zeros(
    (len(MASK_NAMES), len(DELTA_T_VALUES_K)),
    dtype=float,
)

with tempfile.TemporaryDirectory() as scan_tmp:
    scan_tmp = Path(scan_tmp)

    for delta_index, delta_T_K in enumerate(DELTA_T_VALUES_K):
        T2_K = T_INTERNAL_K + delta_T_K

        print(
            f"Delta-T scan {delta_index + 1}/{len(DELTA_T_VALUES_K)}: "
            f"delta_T={delta_T_K:.6g} K, T2={T2_K:.6g} K"
        )

        # Identical spectra give exactly zero bias in both cases.
        if delta_T_K == 0.0:
            continue

        cfg_T2 = copy.deepcopy(base_cfg)
        cfg_T2["stellar"]["spectrum"]["enabled"] = True
        cfg_T2["stellar"]["spectrum"]["mode"] = "blackbody"
        cfg_T2["stellar"]["spectrum"]["temperature_K"] = float(T2_K)

        cfg_T2["fresnel_relay"]["coldstop_x_offset"] = 0.0
        cfg_T2["fresnel_relay"]["coldstop_y_offset"] = 0.0
        cfg_T2["fresnel_relay"]["pupil_misconjugation"] = 0.0
        cfg_T2["fresnel_relay"]["edge_offset"] = 0.0
        cfg_T2["fresnel_relay"]["edge_angle"] = 0.0
        cfg_T2["fresnel_relay"]["use_nominal_pupil_conjugation"] = True

        cfg_T2["internal_aberrations"]["enabled"] = False

        cfg_T2["detector"]["enabled"] = True
        cfg_T2["detector"]["ron"] = 0.0
        cfg_T2["detector"]["include_shotnoise"] = False
        cfg_T2["detector"]["include_readnoise"] = False
        cfg_T2["detector"]["adu_offset"] = 0.0
        cfg_T2["detector"]["noise_std_adu"] = 0.0

        T2_path = scan_tmp / f"T2_{delta_index:03d}.json"

        with open(T2_path, "w") as f:
            json.dump(cfg_T2, f, indent=2)

        zwfs_T2 = bldr.init_zwfs_from_json(T2_path)

        amp_T2 = zwfs_T2.grid.pupil_mask.astype(float)
        zero_T2 = np.zeros_like(amp_T2)

        if amp_T2.shape != amp_internal.shape:
            raise RuntimeError(
                "T2 pupil shape differs from the T1 pupil shape."
            )

        N0_T2 = bldr.get_N0_configured(
            opd_input=zero_T2,
            amp_input=amp_T2,
            opd_internal=zero_T2,
            zwfs_ns=zwfs_T2,
            detector=zwfs_T2.detector,
            include_shotnoise=False,
        )

        if N0_T2.shape != analysis_mask.shape:
            raise RuntimeError(
                "T2 detector image shape differs from analysis_mask."
            )

        N0_T2_mean = np.mean(N0_T2[analysis_mask])

        for mask_number, mask_name in enumerate(MASK_NAMES):
            mask_entry = mask_entries[mask_name]

            zwfs_T2.optics.active_phasemask = (
                spec.normalise_phasemask_entry(
                    mask_name,
                    mask_entry,
                    zwfs_T2.optics,
                )
            )

            I0_T2 = bldr.get_I0_configured(
                opd_input=zero_T2,
                amp_input=amp_T2,
                opd_internal=zero_T2,
                zwfs_ns=zwfs_T2,
                detector=zwfs_T2.detector,
                include_shotnoise=False,
            )

            I0_T2_norm = I0_T2 / N0_T2_mean

            reference_difference = (
                I0_internal_norm_by_mask[mask_number]
                - I0_T2_norm
            )

            # ------------------------------------------------
            # Operational case: fixed internal I2M(T1)
            # ------------------------------------------------

            coefficients_fixed_nm = (
                I2M_T1_by_mask[mask_number]
                @ reference_difference[analysis_mask]
            )

            reconstructed_fixed_nm = np.sum(
                coefficients_fixed_nm[:, None, None] * modes,
                axis=0,
            )

            peak_wfe_bias_fixed_I2M_nm[
                mask_number,
                delta_index,
            ] = np.max(
                np.abs(reconstructed_fixed_nm[pupil_wave])
            )

            # ------------------------------------------------
            # Diagnostic case: matched I2M(T2)
            # ------------------------------------------------

            response_columns_T2 = []

            for mode in modes:
                opd_plus = LINEAR_POKE_NM * 1e-9 * mode
                opd_minus = -LINEAR_POKE_NM * 1e-9 * mode

                I_plus = bldr.get_I0_configured(
                    opd_input=opd_plus,
                    amp_input=amp_T2,
                    opd_internal=zero_T2,
                    zwfs_ns=zwfs_T2,
                    detector=zwfs_T2.detector,
                    include_shotnoise=False,
                )

                I_minus = bldr.get_I0_configured(
                    opd_input=opd_minus,
                    amp_input=amp_T2,
                    opd_internal=zero_T2,
                    zwfs_ns=zwfs_T2,
                    detector=zwfs_T2.detector,
                    include_shotnoise=False,
                )

                response_columns_T2.append(
                    (
                        (I_plus - I_minus)
                        / N0_T2_mean
                        / (2.0 * LINEAR_POKE_NM)
                    )[analysis_mask]
                )

            interaction_matrix_T2 = np.column_stack(
                response_columns_T2
            )

            U_T2, singular_values_T2, Vt_T2 = np.linalg.svd(
                interaction_matrix_T2,
                full_matrices=False,
            )

            keep_T2 = singular_values_T2 > (
                SVD_RELATIVE_CUTOFF * singular_values_T2[0]
            )

            if not np.any(keep_T2):
                raise RuntimeError(
                    f"No singular values retained for {mask_name} "
                    f"at T2={T2_K:.3f} K."
                )

            I2M_T2 = (
                Vt_T2[keep_T2].T
                @ np.diag(1.0 / singular_values_T2[keep_T2])
                @ U_T2[:, keep_T2].T
            )

            coefficients_matched_nm = (
                I2M_T2
                @ reference_difference[analysis_mask]
            )

            reconstructed_matched_nm = np.sum(
                coefficients_matched_nm[:, None, None] * modes,
                axis=0,
            )

            peak_wfe_bias_matched_I2M_nm[
                mask_number,
                delta_index,
            ] = np.max(
                np.abs(reconstructed_matched_nm[pupil_wave])
            )


with open(OUTPUT_DELTA_T_CSV, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow(
        [
            "mask",
            "mask_diameter_um",
            "mask_diameter_lambda_over_D",
            "delta_T_K",
            "T1_K",
            "T2_K",
            "peak_abs_WFE_fixed_I2M_T1_nm",
            "peak_abs_WFE_matched_I2M_T2_nm",
        ]
    )

    for mask_number, mask_name in enumerate(MASK_NAMES):
        for delta_index, delta_T_K in enumerate(DELTA_T_VALUES_K):
            writer.writerow(
                [
                    mask_name,
                    mask_diam_um_array[mask_number],
                    mask_diam_lambdaD_array[mask_number],
                    delta_T_K,
                    T_INTERNAL_K,
                    T_INTERNAL_K + delta_T_K,
                    peak_wfe_bias_fixed_I2M_nm[
                        mask_number,
                        delta_index,
                    ],
                    peak_wfe_bias_matched_I2M_nm[
                        mask_number,
                        delta_index,
                    ],
                ]
            )


# Immutable snapshots and numerical fingerprints before any plotting.
analysis_arrays_before_plotting = {
    "radial_signal_mean": radial_signal_mean.copy(),
    "radial_signal_rms": radial_signal_rms.copy(),
    "radial_wfe_mean_nm": radial_wfe_mean_nm.copy(),
    "radial_wfe_rms_nm": radial_wfe_rms_nm.copy(),
    "peak_wfe_bias_fixed_I2M_nm": (
        peak_wfe_bias_fixed_I2M_nm.copy()
    ),
    "peak_wfe_bias_matched_I2M_nm": (
        peak_wfe_bias_matched_I2M_nm.copy()
    ),
}

analysis_fingerprints = {
    name: hashlib.sha256(
        np.ascontiguousarray(values).view(np.uint8)
    ).hexdigest()
    for name, values in analysis_arrays_before_plotting.items()
}

print("Numerical result fingerprints:")
for name, digest in analysis_fingerprints.items():
    print(f"  {name}: {digest}")


# ============================================================
# CSV and LaTeX tables
# ============================================================

fieldnames = list(table_rows[0].keys())

with open(OUTPUT_TABLE_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(table_rows)

with open(OUTPUT_TABLE_TEX, "w") as f:
    f.write("\\begin{tabular}{lrrrrrrrr}\n")
    f.write("\\hline\n")
    f.write(
        "Mask & Diam. & Diam. & Peak $|\\Delta s|$ & "
        "RMSE $\\Delta s$ & WFE RMS & Peak $|\\phi|$ & "
        "Centre RMS & Edge RMS \\\\\n"
    )
    f.write(
        " & [$\\mu$m] & [$\\lambda/D$] & & & [nm] & [nm] & [nm] & [nm] \\\\\n"
    )
    f.write("\\hline\n")

    for row in table_rows:
        f.write(
            f"{row['mask']} & "
            f"{row['mask_diameter_um']:.0f} & "
            f"{row['mask_diameter_lambda_over_D']:.2f} & "
            f"{row['peak_abs_signal_bias']:.3e} & "
            f"{row['signal_bias_rmse']:.3e} & "
            f"{row['equivalent_wfe_rms_nm']:.2f} & "
            f"{row['peak_abs_wfe_nm']:.2f} & "
            f"{row['centre_wfe_rms_nm']:.2f} & "
            f"{row['edge_wfe_rms_nm']:.2f} \\\\\n"
        )

    f.write("\\hline\n")
    f.write("\\end{tabular}\n")


# ============================================================
# Plot helpers
# ============================================================

def decorate_axis(ax, title, ylabel, yzero=False, xlab=False):
    # 'title' is retained in the call signature for backwards readability,
    # but publication figures intentionally contain no axes titles.
    ax.set_ylabel(ylabel)
    if xlab:
        ax.set_xlabel("Normalized pupil radius")
    if yzero:
        ax.axhline(0.0, color="0.25", linewidth=1.0, zorder=0)
    ax.axvline(
        EDGE_MIN_RHO,
        linestyle=":",
        linewidth=1.2,
        color="0.4",
        zorder=0,
    )
    ax.set_xlim(radial_edges[0], radial_edges[-1])


def add_colourbar(fig, axs):
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axs,
        shrink=0.94,
        pad=0.02,
    )
    cbar.set_label(
        r"Mask diameter [$\lambda/D$]",
        fontsize=12.5,
    )
    cbar.ax.tick_params(labelsize=10.5)
    return cbar


def draw_panel(ax, ydata, yzero, ylabel, title, show_legend=False):
    for i, label in enumerate(mask_labels):
        ax.plot(
            radial_centres,
            ydata[i],
            marker="o",
            color=trace_colours[i],
            label=label,
        )
    decorate_axis(ax, title=title, ylabel=ylabel, yzero=yzero, xlab=True)
    if show_legend:
        ax.legend(frameon=True, ncol=1, loc="best")


def save_standalone_figure(filename, ydata, yzero, ylabel, title):
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for i, label in enumerate(mask_labels):
        ax.plot(
            radial_centres,
            ydata[i],
            marker="o",
            color=trace_colours[i],
            label=label,
        )
    decorate_axis(ax, title=title, ylabel=ylabel, yzero=yzero, xlab=True)
    ax.legend(
        frameon=True,
        loc="best",
        title="Phase mask",
        fontsize=12,
        title_fontsize=12,
    )
    add_colourbar(fig, ax)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Combined radial profile figure
# ============================================================

fig, ax = plt.subplots(2, 2, figsize=(12.6, 9.2), constrained_layout=True)

for i, label in enumerate(mask_labels):
    ax[0, 0].plot(
        radial_centres,
        radial_signal_mean[i],
        marker="o",
        color=trace_colours[i],
        label=label,
    )
    ax[0, 1].plot(
        radial_centres,
        radial_signal_rms[i],
        marker="o",
        color=trace_colours[i],
        label=label,
    )
    ax[1, 0].plot(
        radial_centres,
        radial_wfe_mean_nm[i],
        marker="o",
        color=trace_colours[i],
        label=label,
    )
    ax[1, 1].plot(
        radial_centres,
        radial_wfe_rms_nm[i],
        marker="o",
        color=trace_colours[i],
        label=label,
    )

decorate_axis(
    ax[0, 0],
    title="Signed radial signal bias",
    ylabel="Mean normalized-signal bias",
    yzero=True,
    xlab=False,
)
decorate_axis(
    ax[0, 1],
    title="Radial signal-bias magnitude",
    ylabel="RMS normalized-signal bias",
    yzero=False,
    xlab=False,
)
decorate_axis(
    ax[1, 0],
    title="Signed radial reconstruction bias",
    ylabel="Mean reconstructed OPD bias [nm]",
    yzero=True,
    xlab=True,
)
decorate_axis(
    ax[1, 1],
    title="Radial reconstruction-bias magnitude",
    ylabel="RMS reconstructed OPD bias [nm]",
    yzero=False,
    xlab=True,
)

legend = ax[0, 0].legend(
    title="Phase mask",
    frameon=True,
    ncol=1,
    loc="best",
)

add_colourbar(fig, ax)
fig.savefig(OUTPUT_COMBINED_FIGURE, bbox_inches="tight")


# ============================================================
# Standalone radial profile figures
# ============================================================

save_standalone_figure(
    OUTPUT_SINGLE_MEAN_SIGNAL,
    radial_signal_mean,
    True,
    "Mean normalized-signal bias",
    "Signed radial signal bias",
)

save_standalone_figure(
    OUTPUT_SINGLE_RMS_SIGNAL,
    radial_signal_rms,
    False,
    "RMS normalized-signal bias",
    "Radial signal-bias magnitude",
)

save_standalone_figure(
    OUTPUT_SINGLE_MEAN_WFE,
    radial_wfe_mean_nm,
    True,
    "Mean reconstructed OPD bias [nm]",
    "Signed radial reconstruction bias",
)

save_standalone_figure(
    OUTPUT_SINGLE_RMS_WFE,
    radial_wfe_rms_nm,
    False,
    "RMS reconstructed OPD bias [nm]",
    "Radial reconstruction-bias magnitude",
)


# ============================================================
# Peak WFE bias versus delta T
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

for mask_number in range(len(MASK_NAMES)):
    ax.plot(
        DELTA_T_VALUES_K,
        peak_wfe_bias_fixed_I2M_nm[mask_number],
        marker="o",
        linestyle="-",
        color=trace_colours[mask_number],
    )

    ax.plot(
        DELTA_T_VALUES_K,
        peak_wfe_bias_matched_I2M_nm[mask_number],
        linestyle="--",
        color=trace_colours[mask_number],
    )

# A pure logarithmic axis cannot contain delta_T=0. Symlog preserves the
# exact zero point and is logarithmic above the small linear region.
ax.set_xscale(
    "symlog",
    linthresh=1.0,
    linscale=1.0,
    base=10,
)

ax.set_xticks(
    [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0]
)
ax.set_xticklabels(
    ["0", "1", "10", r"$10^2$", r"$10^3$", r"$10^4$"]
)

ax.set_xlabel(r"Source-temperature difference $\Delta T=T_2-T_1$ [K]")
ax.set_ylabel("Peak absolute reconstructed WFE bias [nm]")
ax.set_xlim(0.0, DELTA_T_MAX_K)
ax.set_ylim(bottom=0.0)
ax.grid(True, which="both", alpha=0.22)

calibration_legend = [
    Line2D(
        [0],
        [0],
        color="0.2",
        linewidth=2.2,
        marker="o",
        linestyle="-",
        label=r"Fixed internal $I2M(T_1)$",
    ),
    Line2D(
        [0],
        [0],
        color="0.2",
        linewidth=2.2,
        linestyle="--",
        label=r"Matched $I2M(T_2)$",
    ),
]

ax.legend(
    handles=calibration_legend,
    frameon=True,
    loc="upper left",
    fontsize=8.8,
)

add_colourbar(fig, ax)
fig.savefig(OUTPUT_DELTA_T_FIGURE, bbox_inches="tight")



# ============================================================
# Global summary figure
# ============================================================

signal_peak_values = [
    row["peak_abs_signal_bias"] for row in table_rows
]
signal_rmse_values = [
    row["signal_bias_rmse"] for row in table_rows
]
wfe_rms_values = [
    row["equivalent_wfe_rms_nm"] for row in table_rows
]
centre_values = [
    row["centre_wfe_rms_nm"] for row in table_rows
]
edge_values = [
    row["edge_wfe_rms_nm"] for row in table_rows
]

x = np.arange(len(mask_labels))
width = 0.36

fig, ax = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)

bar_colours = trace_colours

ax[0].bar(
    x - width / 2,
    signal_peak_values,
    width,
    color=bar_colours,
    label=r"Peak $|\Delta s|$",
)

ax[0].bar(
    x + width / 2,
    signal_rmse_values,
    width,
    color=bar_colours,
    alpha=0.45,
    label=r"RMSE $(\Delta s)$",
)

ax[0].set_xticks(x, [rf"{v:.2f}$\,\lambda/D$" for v in mask_diam_lambdaD_array])
ax[0].set_ylabel("Normalized-signal bias")
ax[0].legend(frameon=True)
ax[0].grid(axis="y", alpha=0.22)

ax[1].bar(
    x - width,
    wfe_rms_values,
    width,
    color=bar_colours,
    label="Full pupil",
)

ax[1].bar(
    x,
    centre_values,
    width,
    color=bar_colours,
    alpha=0.70,
    label=rf"Centre, $\rho<{CENTER_MAX_RHO}$",
)

ax[1].bar(
    x + width,
    edge_values,
    width,
    color=bar_colours,
    alpha=0.40,
    label=rf"Edge, $\rho\geq{EDGE_MIN_RHO}$",
)

ax[1].set_xticks(x, [rf"{v:.2f}$\,\lambda/D$" for v in mask_diam_lambdaD_array])
ax[1].set_ylabel("Equivalent WFE bias [nm RMS OPD]")
ax[1].legend(frameon=True)
ax[1].grid(axis="y", alpha=0.22)

add_colourbar(fig, ax)
fig.savefig(OUTPUT_SUMMARY_FIGURE, bbox_inches="tight")


# ============================================================
# Save numerical products and print table
# ============================================================

np.savez_compressed(
    OUTPUT_DATA,
    mask_names=np.asarray(MASK_NAMES),
    mask_labels=np.asarray(mask_labels),
    mask_diam_um=mask_diam_um_array,
    mask_diam_lambdaD_wvl0=mask_diam_lambdaD_array,
    radial_edges=radial_edges,
    radial_centres=radial_centres,
    pupil_detector=pupil_detector,
    analysis_mask=analysis_mask,
    pupil_wave=pupil_wave,
    rho_detector=rho_detector,
    rho_wave=rho_wave,
    N0_internal=N0_internal,
    N0_onsky=N0_onsky,
    signal_bias_maps=signal_bias_maps,
    reconstructed_opd_maps_nm=reconstructed_opd_maps_nm,
    fitted_signal_maps=fitted_signal_maps,
    radial_signal_mean=radial_signal_mean,
    radial_signal_rms=radial_signal_rms,
    radial_wfe_mean_nm=radial_wfe_mean_nm,
    radial_wfe_rms_nm=radial_wfe_rms_nm,
    delta_T_values_K=DELTA_T_VALUES_K,
    T1_scan_K=np.array(T_INTERNAL_K),
    T2_scan_K=T_INTERNAL_K + DELTA_T_VALUES_K,
    peak_wfe_bias_fixed_I2M_nm=peak_wfe_bias_fixed_I2M_nm,
    peak_wfe_bias_matched_I2M_nm=peak_wfe_bias_matched_I2M_nm,
)

print()
print(
    "Mask  Diam[um]  Diam[lambda/D]  Peak|ds|    RMSE(ds)    "
    "WFE_RMS[nm]  Centre[nm]  Edge[nm]  Explained[%]"
)

for row in table_rows:
    print(
        f"{row['mask']:4s}  "
        f"{row['mask_diameter_um']:8.0f}  "
        f"{row['mask_diameter_lambda_over_D']:14.2f}  "
        f"{row['peak_abs_signal_bias']:9.3e}  "
        f"{row['signal_bias_rmse']:9.3e}  "
        f"{row['equivalent_wfe_rms_nm']:11.2f}  "
        f"{row['centre_wfe_rms_nm']:10.2f}  "
        f"{row['edge_wfe_rms_nm']:8.2f}  "
        f"{row['explained_signal_percent']:12.1f}"
    )

# Verify that the plotting section did not alter any analysis arrays.
analysis_arrays_after_plotting = {
    "radial_signal_mean": radial_signal_mean,
    "radial_signal_rms": radial_signal_rms,
    "radial_wfe_mean_nm": radial_wfe_mean_nm,
    "radial_wfe_rms_nm": radial_wfe_rms_nm,
    "peak_wfe_bias_fixed_I2M_nm": (
        peak_wfe_bias_fixed_I2M_nm
    ),
    "peak_wfe_bias_matched_I2M_nm": (
        peak_wfe_bias_matched_I2M_nm
    ),
}

for name, before in analysis_arrays_before_plotting.items():
    after = analysis_arrays_after_plotting[name]

    if not np.array_equal(before, after, equal_nan=True):
        raise RuntimeError(
            f"Plotting unexpectedly modified numerical array: {name}"
        )

def file_sha256(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()

provenance = {
    "python_executable": str(Path(sys.executable).resolve()),
    "python_version": platform.python_version(),
    "repo_root": str(REPO_ROOT),
    "imports": {
        "baldr_core": str(Path(bldr.__file__).resolve()),
        "spectrum": str(Path(spec.__file__).resolve()),
        "DM_basis": str(Path(DM_basis.__file__).resolve()),
    },
    "input_hashes_sha256": {
        "config": file_sha256(CONFIG_PATH),
        "phasemask_properties": file_sha256(phasemask_path),
        "baldr_core": file_sha256(bldr.__file__),
        "spectrum": file_sha256(spec.__file__),
        "DM_basis": file_sha256(DM_basis.__file__),
    },
    "analysis_fingerprints_sha256": analysis_fingerprints,
    "mask_order": MASK_NAMES,
    "mask_diameter_um": mask_diam_um_array.tolist(),
    "mask_diameter_lambda_over_D_at_wvl0": (
        mask_diam_lambdaD_array.tolist()
    ),
    "temperature_scan": {
        "T1_K": T_INTERNAL_K,
        "delta_T_values_K": DELTA_T_VALUES_K.tolist(),
        "T2_values_K": (
            T_INTERNAL_K + DELTA_T_VALUES_K
        ).tolist(),
        "I2M_fixed_at_T1_for_each_mask": True,
        "matched_I2M_rebuilt_for_every_T2_and_mask": True,
        "internal_I2M_calibration_temperature_K": T_INTERNAL_K,
        "metric": "peak absolute reconstructed pupil OPD bias [nm]",
        "fixed_bias_definition": (
            "I2M(T1) @ [I0_norm(T1) - I0_norm(T2)]"
        ),
        "matched_bias_definition": (
            "I2M(T2) @ [I0_norm(T1) - I0_norm(T2)]"
        ),
    },
}

with open(OUTPUT_PROVENANCE, "w") as f:
    json.dump(provenance, f, indent=2)

print()
for path in [
    OUTPUT_TABLE_CSV,
    OUTPUT_TABLE_TEX,
    OUTPUT_COMBINED_FIGURE,
    OUTPUT_SUMMARY_FIGURE,
    OUTPUT_SINGLE_MEAN_SIGNAL,
    OUTPUT_SINGLE_RMS_SIGNAL,
    OUTPUT_SINGLE_MEAN_WFE,
    OUTPUT_SINGLE_RMS_WFE,
    OUTPUT_DELTA_T_FIGURE,
    OUTPUT_DELTA_T_CSV,
    OUTPUT_DATA,
    OUTPUT_PROVENANCE,
]:
    print(f"Saved: {path.resolve()}")

plt.show()


# #!/usr/bin/env python3
# """
# Paper-grade chromatic Baldr I0 reference-bias analysis for the physical H-band masks.

# This keeps the original analysis unchanged, but improves the plotting:
# - legend labels use physical mask size at wvl0 in lambda/D (and um);
# - traces are coloured by mask diameter with a continuous colormap;
# - the original 2x2 summary figure is retained;
# - each panel is also saved as a standalone figure;
# - the delta-T scan uses a fixed internal-source I2M(T1).
# """

# import copy
# import csv
# import hashlib
# import json
# import platform
# import sys
# import tempfile
# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.colors import Normalize
# from matplotlib.lines import Line2D
# from scipy.ndimage import binary_erosion

# REPO_ROOT = Path(__file__).resolve().parents[3]
# sys.path.insert(0, str(REPO_ROOT))

# from baldrapp.common import DM_basis
# from baldrapp.common import baldr_core as bldr
# from baldrapp.common import spectrum as spec


# # ============================================================
# # Strict local-import check
# # ============================================================

# for module in (bldr, spec, DM_basis):
#     module_path = Path(module.__file__).resolve()

#     if REPO_ROOT not in module_path.parents:
#         raise ImportError(
#             "BaldrApp import shadowing detected.\n"
#             f"Expected modules below: {REPO_ROOT}\n"
#             f"Imported instead: {module_path}\n"
#             "Run from the repository root, or use PYTHONPATH=$PWD."
#         )

# print(f"Python executable: {Path(sys.executable).resolve()}")
# print(f"Python version: {platform.python_version()}")
# print(f"baldr_core: {Path(bldr.__file__).resolve()}")
# print(f"spectrum: {Path(spec.__file__).resolve()}")
# print(f"DM_basis: {Path(DM_basis.__file__).resolve()}")


# # ============================================================
# # Settings
# # ============================================================

# CONFIG_PATH = (
#     REPO_ROOT
#     / "baldrapp/apps/paranal_simulator/fake_configs/baldr_config.json"
# )

# T_INTERNAL_K = 1900.0
# T_ONSKY_K = 10000.0

# MASK_NAMES = ["H1", "H2", "H3", "H4", "H5"]

# N_ZERNIKE_MODES = 20
# LINEAR_POKE_NM = 10.0
# SVD_RELATIVE_CUTOFF = 1e-3

# N_RADIAL_BINS = 10
# CENTER_MAX_RHO = 0.4
# EDGE_MIN_RHO = 0.8

# # Temperature-difference scan. T1 is fixed at T_INTERNAL_K and
# # T2 = T1 + delta_T. Zero is included explicitly; positive values are
# # logarithmically spaced to 20,000 K.
# DELTA_T_MAX_K = 20000.0
# N_DELTA_T_LOG_SAMPLES = 18
# DELTA_T_VALUES_K = np.concatenate(
#     (
#         np.array([0.0]),
#         np.geomspace(1.0, DELTA_T_MAX_K, N_DELTA_T_LOG_SAMPLES),
#     )
# )

# OUTPUT_TABLE_CSV = Path("chromatic_I0_bias_table.csv")
# OUTPUT_TABLE_TEX = Path("chromatic_I0_bias_table.tex")
# OUTPUT_DATA = Path("chromatic_I0_bias_results.npz")
# OUTPUT_PROVENANCE = Path("chromatic_I0_bias_provenance.json")

# OUTPUT_COMBINED_FIGURE = Path("chromatic_I0_bias_radial_profiles_paper.png")
# OUTPUT_SUMMARY_FIGURE = Path("chromatic_I0_bias_summary_paper.png")

# OUTPUT_SINGLE_MEAN_SIGNAL = Path("chromatic_I0_bias_signal_mean_radial.png")
# OUTPUT_SINGLE_RMS_SIGNAL = Path("chromatic_I0_bias_signal_rms_radial.png")
# OUTPUT_SINGLE_MEAN_WFE = Path("chromatic_I0_bias_wfe_mean_radial.png")
# OUTPUT_SINGLE_RMS_WFE = Path("chromatic_I0_bias_wfe_rms_radial.png")

# OUTPUT_DELTA_T_FIGURE = Path(
#     "chromatic_I0_peak_WFE_bias_vs_deltaT.png"
# )
# OUTPUT_DELTA_T_CSV = Path(
#     "chromatic_I0_peak_WFE_bias_vs_deltaT.csv"
# )


# # ============================================================
# # Plot style
# # ============================================================

# plt.rcParams.update(
#     {
#         "figure.dpi": 140,
#         "savefig.dpi": 300,
#         "font.size": 11,
#         "axes.titlesize": 13,
#         "axes.labelsize": 12,
#         "legend.fontsize": 10,
#         "xtick.labelsize": 10,
#         "ytick.labelsize": 10,
#         "lines.linewidth": 2.2,
#         "lines.markersize": 6.0,
#         "axes.grid": True,
#         "grid.alpha": 0.22,
#         "grid.linewidth": 0.7,
#         "mathtext.default": "regular",
#     }
# )


# # ============================================================
# # Initialize identical aligned states with different spectra
# # ============================================================

# with open(CONFIG_PATH, "r") as f:
#     base_cfg = json.load(f)

# cfg_internal = copy.deepcopy(base_cfg)
# cfg_onsky = copy.deepcopy(base_cfg)

# for cfg, temperature_K in (
#     (cfg_internal, T_INTERNAL_K),
#     (cfg_onsky, T_ONSKY_K),
# ):
#     cfg["stellar"]["spectrum"]["enabled"] = True
#     cfg["stellar"]["spectrum"]["mode"] = "blackbody"
#     cfg["stellar"]["spectrum"]["temperature_K"] = temperature_K

#     cfg["fresnel_relay"]["coldstop_x_offset"] = 0.0
#     cfg["fresnel_relay"]["coldstop_y_offset"] = 0.0
#     cfg["fresnel_relay"]["pupil_misconjugation"] = 0.0
#     cfg["fresnel_relay"]["edge_offset"] = 0.0
#     cfg["fresnel_relay"]["edge_angle"] = 0.0
#     cfg["fresnel_relay"]["use_nominal_pupil_conjugation"] = True

#     cfg["internal_aberrations"]["enabled"] = False

#     cfg["detector"]["enabled"] = True
#     cfg["detector"]["ron"] = 0.0
#     cfg["detector"]["include_shotnoise"] = False
#     cfg["detector"]["include_readnoise"] = False
#     cfg["detector"]["adu_offset"] = 0.0
#     cfg["detector"]["noise_std_adu"] = 0.0

# with tempfile.TemporaryDirectory() as tmp:
#     tmp = Path(tmp)
#     internal_path = tmp / "internal_1900K.json"
#     onsky_path = tmp / "onsky_10000K.json"

#     with open(internal_path, "w") as f:
#         json.dump(cfg_internal, f, indent=2)

#     with open(onsky_path, "w") as f:
#         json.dump(cfg_onsky, f, indent=2)

#     zwfs_internal = bldr.init_zwfs_from_json(internal_path)
#     zwfs_onsky = bldr.init_zwfs_from_json(onsky_path)


# # ============================================================
# # Physical masks, pupil masks, and exact matched N0 references
# # ============================================================

# phasemask_path = (
#     REPO_ROOT
#     / base_cfg["simulator_runtime"]["phasemask"]["properties_file"]
# )

# with open(phasemask_path, "r") as f:
#     phasemask_cfg = json.load(f)

# mask_entries = phasemask_cfg["phasemask"]["masks"]

# amp_internal = zwfs_internal.grid.pupil_mask.astype(float)
# amp_onsky = zwfs_onsky.grid.pupil_mask.astype(float)

# zero_internal = np.zeros_like(amp_internal)
# zero_onsky = np.zeros_like(amp_onsky)

# N0_internal = bldr.get_N0_configured(
#     opd_input=zero_internal,
#     amp_input=amp_internal,
#     opd_internal=zero_internal,
#     zwfs_ns=zwfs_internal,
#     detector=zwfs_internal.detector,
#     include_shotnoise=False,
# )

# N0_onsky = bldr.get_N0_configured(
#     opd_input=zero_onsky,
#     amp_input=amp_onsky,
#     opd_internal=zero_onsky,
#     zwfs_ns=zwfs_onsky,
#     detector=zwfs_onsky.detector,
#     include_shotnoise=False,
# )

# binning = int(base_cfg["detector"]["binning"])

# pupil_detector = (
#     bldr.sum_subarrays(
#         zwfs_onsky.grid.pupil_mask,
#         block_size=(binning, binning),
#     )
#     > 0.5 * binning**2
# )

# analysis_mask = binary_erosion(pupil_detector, iterations=1)

# if not np.any(analysis_mask):
#     raise RuntimeError("The eroded detector pupil mask is empty.")

# N0_internal_mean = np.mean(N0_internal[analysis_mask])
# N0_onsky_mean = np.mean(N0_onsky[analysis_mask])

# yy_det, xx_det = np.indices(pupil_detector.shape)
# cy_det = np.mean(yy_det[pupil_detector])
# cx_det = np.mean(xx_det[pupil_detector])

# r_det = np.sqrt((xx_det - cx_det) ** 2 + (yy_det - cy_det) ** 2)
# outer_radius_det = np.percentile(r_det[pupil_detector], 99.5)
# rho_detector = r_det / outer_radius_det

# pupil_wave = zwfs_onsky.grid.pupil_mask.astype(bool)

# yy_wave, xx_wave = np.indices(pupil_wave.shape)
# cy_wave = np.mean(yy_wave[pupil_wave])
# cx_wave = np.mean(xx_wave[pupil_wave])

# r_wave = np.sqrt((xx_wave - cx_wave) ** 2 + (yy_wave - cy_wave) ** 2)
# outer_radius_wave = np.percentile(r_wave[pupil_wave], 99.5)
# rho_wave = r_wave / outer_radius_wave
# theta_wave = np.arctan2(yy_wave - cy_wave, xx_wave - cx_wave)

# radial_edges = np.linspace(0.0, 1.0, N_RADIAL_BINS + 1)
# radial_centres = 0.5 * (radial_edges[:-1] + radial_edges[1:])


# # ============================================================
# # Common phase basis: piston removed, each mode = 1 nm RMS OPD
# # ============================================================

# raw_zernikes = DM_basis.zernike_basis(
#     nterms=N_ZERNIKE_MODES + 1,
#     rho=rho_wave,
#     theta=theta_wave,
#     outside=0.0,
# )

# modes = []

# for mode_index in range(1, N_ZERNIKE_MODES + 1):
#     mode = np.nan_to_num(raw_zernikes[mode_index], nan=0.0)
#     mode *= pupil_wave
#     mode -= np.mean(mode[pupil_wave])

#     mode_rms = np.sqrt(np.mean(mode[pupil_wave] ** 2))

#     if mode_rms <= 0:
#         raise RuntimeError(f"Zernike mode {mode_index + 1} has zero RMS.")

#     modes.append(mode / mode_rms)

# modes = np.asarray(modes)


# # ============================================================
# # Per-mask signal bias and on-sky linear reconstruction bias
# # ============================================================

# table_rows = []

# signal_bias_maps = []
# reconstructed_opd_maps_nm = []
# fitted_signal_maps = []
# I0_internal_norm_by_mask = []

# radial_signal_mean = np.full((len(MASK_NAMES), N_RADIAL_BINS), np.nan)
# radial_signal_rms = np.full((len(MASK_NAMES), N_RADIAL_BINS), np.nan)
# radial_wfe_mean_nm = np.full((len(MASK_NAMES), N_RADIAL_BINS), np.nan)
# radial_wfe_rms_nm = np.full((len(MASK_NAMES), N_RADIAL_BINS), np.nan)

# mask_diam_um_list = []
# mask_diam_lambdaD_list = []
# mask_labels = []

# for mask_number, mask_name in enumerate(MASK_NAMES):
#     print(f"Processing {mask_name} ({mask_number + 1}/{len(MASK_NAMES)})")

#     mask_entry = mask_entries[mask_name]

#     active_mask_internal = spec.normalise_phasemask_entry(
#         mask_name,
#         mask_entry,
#         zwfs_internal.optics,
#     )

#     active_mask_onsky = spec.normalise_phasemask_entry(
#         mask_name,
#         mask_entry,
#         zwfs_onsky.optics,
#     )

#     zwfs_internal.optics.active_phasemask = active_mask_internal
#     zwfs_onsky.optics.active_phasemask = active_mask_onsky

#     mask_diam_um = float(active_mask_onsky.mask_diam_um)
#     mask_diam_lambdaD = float(active_mask_onsky.mask_diam_lambdaD_wvl0)

#     mask_diam_um_list.append(mask_diam_um)
#     mask_diam_lambdaD_list.append(mask_diam_lambdaD)
#     mask_labels.append(
#         rf"{mask_diam_lambdaD:.2f}$\,\lambda/D$ ({mask_diam_um:.0f}$\,\mu$m)"
#     )

#     I0_internal = bldr.get_I0_configured(
#         opd_input=zero_internal,
#         amp_input=amp_internal,
#         opd_internal=zero_internal,
#         zwfs_ns=zwfs_internal,
#         detector=zwfs_internal.detector,
#         include_shotnoise=False,
#     )

#     I0_onsky = bldr.get_I0_configured(
#         opd_input=zero_onsky,
#         amp_input=amp_onsky,
#         opd_internal=zero_onsky,
#         zwfs_ns=zwfs_onsky,
#         detector=zwfs_onsky.detector,
#         include_shotnoise=False,
#     )

#     I0_internal_norm = I0_internal / N0_internal_mean
#     I0_onsky_norm = I0_onsky / N0_onsky_mean

#     I0_internal_norm_by_mask.append(I0_internal_norm.copy())

#     signal_bias = I0_onsky_norm - I0_internal_norm
#     signal_vector = signal_bias[analysis_mask]

#     signal_peak_abs = np.max(np.abs(signal_vector))
#     signal_rmse = np.sqrt(np.mean(signal_vector**2))

#     response_columns = []

#     for mode in modes:
#         opd_plus = LINEAR_POKE_NM * 1e-9 * mode
#         opd_minus = -LINEAR_POKE_NM * 1e-9 * mode

#         I_plus = bldr.get_I0_configured(
#             opd_input=opd_plus,
#             amp_input=amp_onsky,
#             opd_internal=zero_onsky,
#             zwfs_ns=zwfs_onsky,
#             detector=zwfs_onsky.detector,
#             include_shotnoise=False,
#         )

#         I_minus = bldr.get_I0_configured(
#             opd_input=opd_minus,
#             amp_input=amp_onsky,
#             opd_internal=zero_onsky,
#             zwfs_ns=zwfs_onsky,
#             detector=zwfs_onsky.detector,
#             include_shotnoise=False,
#         )

#         I_plus_norm = I_plus / N0_onsky_mean
#         I_minus_norm = I_minus / N0_onsky_mean

#         derivative_per_nm = (
#             I_plus_norm - I_minus_norm
#         ) / (2.0 * LINEAR_POKE_NM)

#         response_columns.append(derivative_per_nm[analysis_mask])

#     interaction_matrix = np.column_stack(response_columns)

#     U, singular_values, Vt = np.linalg.svd(
#         interaction_matrix,
#         full_matrices=False,
#     )

#     keep = singular_values > (
#         SVD_RELATIVE_CUTOFF * singular_values[0]
#     )

#     if not np.any(keep):
#         raise RuntimeError(f"No singular values retained for {mask_name}.")

#     reconstructor = (
#         Vt[keep].T
#         @ np.diag(1.0 / singular_values[keep])
#         @ U[:, keep].T
#     )

#     coefficients_nm = reconstructor @ signal_vector

#     reconstructed_opd_nm = np.sum(
#         coefficients_nm[:, None, None] * modes,
#         axis=0,
#     )

#     fitted_signal_vector = interaction_matrix @ coefficients_nm
#     fitted_signal_map = np.full_like(signal_bias, np.nan, dtype=float)
#     fitted_signal_map[analysis_mask] = fitted_signal_vector

#     signal_power = np.sum(signal_vector**2)
#     residual_power = np.sum(
#         (signal_vector - fitted_signal_vector) ** 2
#     )

#     explained_fraction = (
#         1.0 - residual_power / signal_power
#         if signal_power > 0
#         else np.nan
#     )

#     wfe_values_nm = reconstructed_opd_nm[pupil_wave]
#     equivalent_wfe_rms_nm = np.sqrt(np.mean(wfe_values_nm**2))
#     peak_abs_wfe_nm = np.max(np.abs(wfe_values_nm))

#     centre_mask_wave = pupil_wave & (rho_wave < CENTER_MAX_RHO)
#     edge_mask_wave = (
#         pupil_wave
#         & (rho_wave >= EDGE_MIN_RHO)
#         & (rho_wave <= 1.0)
#     )

#     centre_wfe_rms_nm = np.sqrt(
#         np.mean(reconstructed_opd_nm[centre_mask_wave] ** 2)
#     )

#     edge_wfe_rms_nm = np.sqrt(
#         np.mean(reconstructed_opd_nm[edge_mask_wave] ** 2)
#     )

#     for radial_index in range(N_RADIAL_BINS):
#         detector_annulus = (
#             analysis_mask
#             & (rho_detector >= radial_edges[radial_index])
#             & (rho_detector < radial_edges[radial_index + 1])
#         )

#         wave_annulus = (
#             pupil_wave
#             & (rho_wave >= radial_edges[radial_index])
#             & (rho_wave < radial_edges[radial_index + 1])
#         )

#         if np.any(detector_annulus):
#             radial_values = signal_bias[detector_annulus]

#             radial_signal_mean[mask_number, radial_index] = np.mean(
#                 radial_values
#             )

#             radial_signal_rms[mask_number, radial_index] = np.sqrt(
#                 np.mean(radial_values**2)
#             )

#         if np.any(wave_annulus):
#             radial_values_nm = reconstructed_opd_nm[wave_annulus]

#             radial_wfe_mean_nm[mask_number, radial_index] = np.mean(
#                 radial_values_nm
#             )

#             radial_wfe_rms_nm[mask_number, radial_index] = np.sqrt(
#                 np.mean(radial_values_nm**2)
#             )

#     condition_number_retained = (
#         singular_values[keep][0] / singular_values[keep][-1]
#     )

#     table_rows.append(
#         {
#             "mask": mask_name,
#             "mask_diameter_um": float(mask_diam_um),
#             "mask_diameter_lambda_over_D": float(mask_diam_lambdaD),
#             "peak_abs_signal_bias": float(signal_peak_abs),
#             "signal_bias_rmse": float(signal_rmse),
#             "equivalent_wfe_rms_nm": float(equivalent_wfe_rms_nm),
#             "peak_abs_wfe_nm": float(peak_abs_wfe_nm),
#             "centre_wfe_rms_nm": float(centre_wfe_rms_nm),
#             "edge_wfe_rms_nm": float(edge_wfe_rms_nm),
#             "explained_signal_percent": float(
#                 100.0 * explained_fraction
#             ),
#             "retained_rank": int(np.sum(keep)),
#             "retained_condition_number": float(
#                 condition_number_retained
#             ),
#         }
#     )

#     signal_bias_maps.append(signal_bias)
#     reconstructed_opd_maps_nm.append(reconstructed_opd_nm)
#     fitted_signal_maps.append(fitted_signal_map)

# signal_bias_maps = np.asarray(signal_bias_maps)
# reconstructed_opd_maps_nm = np.asarray(reconstructed_opd_maps_nm)
# fitted_signal_maps = np.asarray(fitted_signal_maps)
# I0_internal_norm_by_mask = np.asarray(I0_internal_norm_by_mask)

# mask_diam_um_array = np.asarray(mask_diam_um_list)
# mask_diam_lambdaD_array = np.asarray(mask_diam_lambdaD_list)

# # Preserve the exact H1--H5 calculation order. The physical diameters are
# # already monotonic, so no numerical arrays need to be reordered for plotting.
# if not np.all(np.diff(mask_diam_lambdaD_array) > 0):
#     raise RuntimeError(
#         "MASK_NAMES are not ordered by increasing physical mask diameter."
#     )

# cmap = plt.get_cmap("viridis")
# norm = Normalize(
#     vmin=float(np.min(mask_diam_lambdaD_array)),
#     vmax=float(np.max(mask_diam_lambdaD_array)),
# )
# trace_colours = [cmap(norm(v)) for v in mask_diam_lambdaD_array]


# # ============================================================
# # Fixed internal-source reconstructors for the delta-T scan
# # ============================================================

# I2M_T1_by_mask = []

# for mask_number, mask_name in enumerate(MASK_NAMES):
#     print(
#         f"Building fixed internal-source I2M for "
#         f"{mask_name} ({mask_number + 1}/{len(MASK_NAMES)})"
#     )

#     mask_entry = mask_entries[mask_name]

#     zwfs_internal.optics.active_phasemask = (
#         spec.normalise_phasemask_entry(
#             mask_name,
#             mask_entry,
#             zwfs_internal.optics,
#         )
#     )

#     response_columns_T1 = []

#     for mode in modes:
#         opd_plus = LINEAR_POKE_NM * 1e-9 * mode
#         opd_minus = -LINEAR_POKE_NM * 1e-9 * mode

#         I_plus = bldr.get_I0_configured(
#             opd_input=opd_plus,
#             amp_input=amp_internal,
#             opd_internal=zero_internal,
#             zwfs_ns=zwfs_internal,
#             detector=zwfs_internal.detector,
#             include_shotnoise=False,
#         )

#         I_minus = bldr.get_I0_configured(
#             opd_input=opd_minus,
#             amp_input=amp_internal,
#             opd_internal=zero_internal,
#             zwfs_ns=zwfs_internal,
#             detector=zwfs_internal.detector,
#             include_shotnoise=False,
#         )

#         response_columns_T1.append(
#             (
#                 (I_plus - I_minus)
#                 / N0_internal_mean
#                 / (2.0 * LINEAR_POKE_NM)
#             )[analysis_mask]
#         )

#     interaction_matrix_T1 = np.column_stack(
#         response_columns_T1
#     )

#     U_T1, singular_values_T1, Vt_T1 = np.linalg.svd(
#         interaction_matrix_T1,
#         full_matrices=False,
#     )

#     keep_T1 = singular_values_T1 > (
#         SVD_RELATIVE_CUTOFF * singular_values_T1[0]
#     )

#     if not np.any(keep_T1):
#         raise RuntimeError(
#             f"No singular values retained for {mask_name} at T1."
#         )

#     I2M_T1 = (
#         Vt_T1[keep_T1].T
#         @ np.diag(1.0 / singular_values_T1[keep_T1])
#         @ U_T1[:, keep_T1].T
#     )

#     I2M_T1_by_mask.append(I2M_T1)

# I2M_T1_by_mask = tuple(I2M_T1_by_mask)


# # ============================================================
# # Worst-case peak WFE bias versus source-temperature difference
# # ============================================================

# # Operational result:
# #   fixed internal calibration, I2M(T1)
# #
# # Diagnostic comparison:
# #   perfectly matched on-sky interaction matrix, I2M(T2)

# peak_wfe_bias_fixed_I2M_nm = np.zeros(
#     (len(MASK_NAMES), len(DELTA_T_VALUES_K)),
#     dtype=float,
# )

# peak_wfe_bias_matched_I2M_nm = np.zeros(
#     (len(MASK_NAMES), len(DELTA_T_VALUES_K)),
#     dtype=float,
# )

# with tempfile.TemporaryDirectory() as scan_tmp:
#     scan_tmp = Path(scan_tmp)

#     for delta_index, delta_T_K in enumerate(DELTA_T_VALUES_K):
#         T2_K = T_INTERNAL_K + delta_T_K

#         print(
#             f"Delta-T scan {delta_index + 1}/{len(DELTA_T_VALUES_K)}: "
#             f"delta_T={delta_T_K:.6g} K, T2={T2_K:.6g} K"
#         )

#         # Identical spectra give exactly zero bias in both cases.
#         if delta_T_K == 0.0:
#             continue

#         cfg_T2 = copy.deepcopy(base_cfg)
#         cfg_T2["stellar"]["spectrum"]["enabled"] = True
#         cfg_T2["stellar"]["spectrum"]["mode"] = "blackbody"
#         cfg_T2["stellar"]["spectrum"]["temperature_K"] = float(T2_K)

#         cfg_T2["fresnel_relay"]["coldstop_x_offset"] = 0.0
#         cfg_T2["fresnel_relay"]["coldstop_y_offset"] = 0.0
#         cfg_T2["fresnel_relay"]["pupil_misconjugation"] = 0.0
#         cfg_T2["fresnel_relay"]["edge_offset"] = 0.0
#         cfg_T2["fresnel_relay"]["edge_angle"] = 0.0
#         cfg_T2["fresnel_relay"]["use_nominal_pupil_conjugation"] = True

#         cfg_T2["internal_aberrations"]["enabled"] = False

#         cfg_T2["detector"]["enabled"] = True
#         cfg_T2["detector"]["ron"] = 0.0
#         cfg_T2["detector"]["include_shotnoise"] = False
#         cfg_T2["detector"]["include_readnoise"] = False
#         cfg_T2["detector"]["adu_offset"] = 0.0
#         cfg_T2["detector"]["noise_std_adu"] = 0.0

#         T2_path = scan_tmp / f"T2_{delta_index:03d}.json"

#         with open(T2_path, "w") as f:
#             json.dump(cfg_T2, f, indent=2)

#         zwfs_T2 = bldr.init_zwfs_from_json(T2_path)

#         amp_T2 = zwfs_T2.grid.pupil_mask.astype(float)
#         zero_T2 = np.zeros_like(amp_T2)

#         if amp_T2.shape != amp_internal.shape:
#             raise RuntimeError(
#                 "T2 pupil shape differs from the T1 pupil shape."
#             )

#         N0_T2 = bldr.get_N0_configured(
#             opd_input=zero_T2,
#             amp_input=amp_T2,
#             opd_internal=zero_T2,
#             zwfs_ns=zwfs_T2,
#             detector=zwfs_T2.detector,
#             include_shotnoise=False,
#         )

#         if N0_T2.shape != analysis_mask.shape:
#             raise RuntimeError(
#                 "T2 detector image shape differs from analysis_mask."
#             )

#         N0_T2_mean = np.mean(N0_T2[analysis_mask])

#         for mask_number, mask_name in enumerate(MASK_NAMES):
#             mask_entry = mask_entries[mask_name]

#             zwfs_T2.optics.active_phasemask = (
#                 spec.normalise_phasemask_entry(
#                     mask_name,
#                     mask_entry,
#                     zwfs_T2.optics,
#                 )
#             )

#             I0_T2 = bldr.get_I0_configured(
#                 opd_input=zero_T2,
#                 amp_input=amp_T2,
#                 opd_internal=zero_T2,
#                 zwfs_ns=zwfs_T2,
#                 detector=zwfs_T2.detector,
#                 include_shotnoise=False,
#             )

#             I0_T2_norm = I0_T2 / N0_T2_mean

#             reference_difference = (
#                 I0_internal_norm_by_mask[mask_number]
#                 - I0_T2_norm
#             )

#             # ------------------------------------------------
#             # Operational case: fixed internal I2M(T1)
#             # ------------------------------------------------

#             coefficients_fixed_nm = (
#                 I2M_T1_by_mask[mask_number]
#                 @ reference_difference[analysis_mask]
#             )

#             reconstructed_fixed_nm = np.sum(
#                 coefficients_fixed_nm[:, None, None] * modes,
#                 axis=0,
#             )

#             peak_wfe_bias_fixed_I2M_nm[
#                 mask_number,
#                 delta_index,
#             ] = np.max(
#                 np.abs(reconstructed_fixed_nm[pupil_wave])
#             )

#             # ------------------------------------------------
#             # Diagnostic case: matched I2M(T2)
#             # ------------------------------------------------

#             response_columns_T2 = []

#             for mode in modes:
#                 opd_plus = LINEAR_POKE_NM * 1e-9 * mode
#                 opd_minus = -LINEAR_POKE_NM * 1e-9 * mode

#                 I_plus = bldr.get_I0_configured(
#                     opd_input=opd_plus,
#                     amp_input=amp_T2,
#                     opd_internal=zero_T2,
#                     zwfs_ns=zwfs_T2,
#                     detector=zwfs_T2.detector,
#                     include_shotnoise=False,
#                 )

#                 I_minus = bldr.get_I0_configured(
#                     opd_input=opd_minus,
#                     amp_input=amp_T2,
#                     opd_internal=zero_T2,
#                     zwfs_ns=zwfs_T2,
#                     detector=zwfs_T2.detector,
#                     include_shotnoise=False,
#                 )

#                 response_columns_T2.append(
#                     (
#                         (I_plus - I_minus)
#                         / N0_T2_mean
#                         / (2.0 * LINEAR_POKE_NM)
#                     )[analysis_mask]
#                 )

#             interaction_matrix_T2 = np.column_stack(
#                 response_columns_T2
#             )

#             U_T2, singular_values_T2, Vt_T2 = np.linalg.svd(
#                 interaction_matrix_T2,
#                 full_matrices=False,
#             )

#             keep_T2 = singular_values_T2 > (
#                 SVD_RELATIVE_CUTOFF * singular_values_T2[0]
#             )

#             if not np.any(keep_T2):
#                 raise RuntimeError(
#                     f"No singular values retained for {mask_name} "
#                     f"at T2={T2_K:.3f} K."
#                 )

#             I2M_T2 = (
#                 Vt_T2[keep_T2].T
#                 @ np.diag(1.0 / singular_values_T2[keep_T2])
#                 @ U_T2[:, keep_T2].T
#             )

#             coefficients_matched_nm = (
#                 I2M_T2
#                 @ reference_difference[analysis_mask]
#             )

#             reconstructed_matched_nm = np.sum(
#                 coefficients_matched_nm[:, None, None] * modes,
#                 axis=0,
#             )

#             peak_wfe_bias_matched_I2M_nm[
#                 mask_number,
#                 delta_index,
#             ] = np.max(
#                 np.abs(reconstructed_matched_nm[pupil_wave])
#             )


# with open(OUTPUT_DELTA_T_CSV, "w", newline="") as f:
#     writer = csv.writer(f)

#     writer.writerow(
#         [
#             "mask",
#             "mask_diameter_um",
#             "mask_diameter_lambda_over_D",
#             "delta_T_K",
#             "T1_K",
#             "T2_K",
#             "peak_abs_WFE_fixed_I2M_T1_nm",
#             "peak_abs_WFE_matched_I2M_T2_nm",
#         ]
#     )

#     for mask_number, mask_name in enumerate(MASK_NAMES):
#         for delta_index, delta_T_K in enumerate(DELTA_T_VALUES_K):
#             writer.writerow(
#                 [
#                     mask_name,
#                     mask_diam_um_array[mask_number],
#                     mask_diam_lambdaD_array[mask_number],
#                     delta_T_K,
#                     T_INTERNAL_K,
#                     T_INTERNAL_K + delta_T_K,
#                     peak_wfe_bias_fixed_I2M_nm[
#                         mask_number,
#                         delta_index,
#                     ],
#                     peak_wfe_bias_matched_I2M_nm[
#                         mask_number,
#                         delta_index,
#                     ],
#                 ]
#             )


# # Immutable snapshots and numerical fingerprints before any plotting.
# analysis_arrays_before_plotting = {
#     "radial_signal_mean": radial_signal_mean.copy(),
#     "radial_signal_rms": radial_signal_rms.copy(),
#     "radial_wfe_mean_nm": radial_wfe_mean_nm.copy(),
#     "radial_wfe_rms_nm": radial_wfe_rms_nm.copy(),
#     "peak_wfe_bias_fixed_I2M_nm": (
#         peak_wfe_bias_fixed_I2M_nm.copy()
#     ),
#     "peak_wfe_bias_matched_I2M_nm": (
#         peak_wfe_bias_matched_I2M_nm.copy()
#     ),
# }

# analysis_fingerprints = {
#     name: hashlib.sha256(
#         np.ascontiguousarray(values).view(np.uint8)
#     ).hexdigest()
#     for name, values in analysis_arrays_before_plotting.items()
# }

# print("Numerical result fingerprints:")
# for name, digest in analysis_fingerprints.items():
#     print(f"  {name}: {digest}")


# # ============================================================
# # CSV and LaTeX tables
# # ============================================================

# fieldnames = list(table_rows[0].keys())

# with open(OUTPUT_TABLE_CSV, "w", newline="") as f:
#     writer = csv.DictWriter(f, fieldnames=fieldnames)
#     writer.writeheader()
#     writer.writerows(table_rows)

# with open(OUTPUT_TABLE_TEX, "w") as f:
#     f.write("\\begin{tabular}{lrrrrrrrr}\n")
#     f.write("\\hline\n")
#     f.write(
#         "Mask & Diam. & Diam. & Peak $|\\Delta s|$ & "
#         "RMSE $\\Delta s$ & WFE RMS & Peak $|\\phi|$ & "
#         "Centre RMS & Edge RMS \\\\\n"
#     )
#     f.write(
#         " & [$\\mu$m] & [$\\lambda/D$] & & & [nm] & [nm] & [nm] & [nm] \\\\\n"
#     )
#     f.write("\\hline\n")

#     for row in table_rows:
#         f.write(
#             f"{row['mask']} & "
#             f"{row['mask_diameter_um']:.0f} & "
#             f"{row['mask_diameter_lambda_over_D']:.2f} & "
#             f"{row['peak_abs_signal_bias']:.3e} & "
#             f"{row['signal_bias_rmse']:.3e} & "
#             f"{row['equivalent_wfe_rms_nm']:.2f} & "
#             f"{row['peak_abs_wfe_nm']:.2f} & "
#             f"{row['centre_wfe_rms_nm']:.2f} & "
#             f"{row['edge_wfe_rms_nm']:.2f} \\\\\n"
#         )

#     f.write("\\hline\n")
#     f.write("\\end{tabular}\n")


# # ============================================================
# # Plot helpers
# # ============================================================

# def decorate_axis(ax, title, ylabel, yzero=False, xlab=False):
#     # 'title' is retained in the call signature for backwards readability,
#     # but publication figures intentionally contain no axes titles.
#     ax.set_ylabel(ylabel)
#     if xlab:
#         ax.set_xlabel("Normalized pupil radius")
#     if yzero:
#         ax.axhline(0.0, color="0.25", linewidth=1.0, zorder=0)
#     ax.axvline(
#         EDGE_MIN_RHO,
#         linestyle=":",
#         linewidth=1.2,
#         color="0.4",
#         zorder=0,
#     )
#     ax.set_xlim(radial_edges[0], radial_edges[-1])


# def add_colourbar(fig, axs):
#     sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
#     sm.set_array([])
#     cbar = fig.colorbar(
#         sm,
#         ax=axs,
#         shrink=0.94,
#         pad=0.02,
#     )
#     cbar.set_label(r"Mask diameter [$\lambda/D$]")
#     return cbar


# def draw_panel(ax, ydata, yzero, ylabel, title, show_legend=False):
#     for i, label in enumerate(mask_labels):
#         ax.plot(
#             radial_centres,
#             ydata[i],
#             marker="o",
#             color=trace_colours[i],
#             label=label,
#         )
#     decorate_axis(ax, title=title, ylabel=ylabel, yzero=yzero, xlab=True)
#     if show_legend:
#         ax.legend(frameon=True, ncol=1, loc="best")


# def save_standalone_figure(filename, ydata, yzero, ylabel, title):
#     fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
#     for i, label in enumerate(mask_labels):
#         ax.plot(
#             radial_centres,
#             ydata[i],
#             marker="o",
#             color=trace_colours[i],
#             label=label,
#         )
#     decorate_axis(ax, title=title, ylabel=ylabel, yzero=yzero, xlab=True)
#     ax.legend(frameon=True, loc="best", title="Phase mask")
#     add_colourbar(fig, ax)
#     fig.savefig(filename, bbox_inches="tight")
#     plt.close(fig)


# # ============================================================
# # Combined radial profile figure
# # ============================================================

# fig, ax = plt.subplots(2, 2, figsize=(12.6, 9.2), constrained_layout=True)

# for i, label in enumerate(mask_labels):
#     ax[0, 0].plot(
#         radial_centres,
#         radial_signal_mean[i],
#         marker="o",
#         color=trace_colours[i],
#         label=label,
#     )
#     ax[0, 1].plot(
#         radial_centres,
#         radial_signal_rms[i],
#         marker="o",
#         color=trace_colours[i],
#         label=label,
#     )
#     ax[1, 0].plot(
#         radial_centres,
#         radial_wfe_mean_nm[i],
#         marker="o",
#         color=trace_colours[i],
#         label=label,
#     )
#     ax[1, 1].plot(
#         radial_centres,
#         radial_wfe_rms_nm[i],
#         marker="o",
#         color=trace_colours[i],
#         label=label,
#     )

# decorate_axis(
#     ax[0, 0],
#     title="Signed radial signal bias",
#     ylabel="Mean normalized-signal bias",
#     yzero=True,
#     xlab=False,
# )
# decorate_axis(
#     ax[0, 1],
#     title="Radial signal-bias magnitude",
#     ylabel="RMS normalized-signal bias",
#     yzero=False,
#     xlab=False,
# )
# decorate_axis(
#     ax[1, 0],
#     title="Signed radial reconstruction bias",
#     ylabel="Mean reconstructed OPD bias [nm]",
#     yzero=True,
#     xlab=True,
# )
# decorate_axis(
#     ax[1, 1],
#     title="Radial reconstruction-bias magnitude",
#     ylabel="RMS reconstructed OPD bias [nm]",
#     yzero=False,
#     xlab=True,
# )

# legend = ax[0, 0].legend(
#     title="Phase mask",
#     frameon=True,
#     ncol=1,
#     loc="best",
# )

# add_colourbar(fig, ax)
# fig.savefig(OUTPUT_COMBINED_FIGURE, bbox_inches="tight")


# # ============================================================
# # Standalone radial profile figures
# # ============================================================

# save_standalone_figure(
#     OUTPUT_SINGLE_MEAN_SIGNAL,
#     radial_signal_mean,
#     True,
#     "Mean normalized-signal bias",
#     "Signed radial signal bias",
# )

# save_standalone_figure(
#     OUTPUT_SINGLE_RMS_SIGNAL,
#     radial_signal_rms,
#     False,
#     "RMS normalized-signal bias",
#     "Radial signal-bias magnitude",
# )

# save_standalone_figure(
#     OUTPUT_SINGLE_MEAN_WFE,
#     radial_wfe_mean_nm,
#     True,
#     "Mean reconstructed OPD bias [nm]",
#     "Signed radial reconstruction bias",
# )

# save_standalone_figure(
#     OUTPUT_SINGLE_RMS_WFE,
#     radial_wfe_rms_nm,
#     False,
#     "RMS reconstructed OPD bias [nm]",
#     "Radial reconstruction-bias magnitude",
# )


# # ============================================================
# # Peak WFE bias versus delta T
# # ============================================================

# fig, ax = plt.subplots(figsize=(7.1, 5.2), constrained_layout=True)

# for mask_number in range(len(MASK_NAMES)):
#     ax.plot(
#         DELTA_T_VALUES_K,
#         peak_wfe_bias_fixed_I2M_nm[mask_number],
#         marker="o",
#         linestyle="-",
#         color=trace_colours[mask_number],
#     )

#     ax.plot(
#         DELTA_T_VALUES_K,
#         peak_wfe_bias_matched_I2M_nm[mask_number],
#         linestyle="--",
#         color=trace_colours[mask_number],
#     )

# # A pure logarithmic axis cannot contain delta_T=0. Symlog preserves the
# # exact zero point and is logarithmic above the small linear region.
# ax.set_xscale(
#     "symlog",
#     linthresh=1.0,
#     linscale=1.0,
#     base=10,
# )

# ax.set_xticks(
#     [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0, 20000.0]
# )
# ax.set_xticklabels(
#     ["0", "1", "10", "100", r"$10^3$", r"$10^4$", r"$2\times10^4$"]
# )

# ax.set_xlabel(r"Source-temperature difference $\Delta T=T_2-T_1$ [K]")
# ax.set_ylabel("Peak absolute reconstructed WFE bias [nm]")
# ax.set_xlim(0.0, DELTA_T_MAX_K)
# ax.set_ylim(bottom=0.0)
# ax.grid(True, which="both", alpha=0.22)

# calibration_legend = [
#     Line2D(
#         [0],
#         [0],
#         color="0.2",
#         linewidth=2.2,
#         marker="o",
#         linestyle="-",
#         label=r"Fixed internal $I2M(T_1)$",
#     ),
#     Line2D(
#         [0],
#         [0],
#         color="0.2",
#         linewidth=2.2,
#         linestyle="--",
#         label=r"Matched $I2M(T_2)$",
#     ),
# ]

# ax.legend(
#     handles=calibration_legend,
#     frameon=True,
#     loc="best",
# )

# add_colourbar(fig, ax)
# fig.savefig(OUTPUT_DELTA_T_FIGURE, bbox_inches="tight")



# # ============================================================
# # Global summary figure
# # ============================================================

# signal_peak_values = [
#     row["peak_abs_signal_bias"] for row in table_rows
# ]
# signal_rmse_values = [
#     row["signal_bias_rmse"] for row in table_rows
# ]
# wfe_rms_values = [
#     row["equivalent_wfe_rms_nm"] for row in table_rows
# ]
# centre_values = [
#     row["centre_wfe_rms_nm"] for row in table_rows
# ]
# edge_values = [
#     row["edge_wfe_rms_nm"] for row in table_rows
# ]

# x = np.arange(len(mask_labels))
# width = 0.36

# fig, ax = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)

# bar_colours = trace_colours

# ax[0].bar(
#     x - width / 2,
#     signal_peak_values,
#     width,
#     color=bar_colours,
#     label=r"Peak $|\Delta s|$",
# )

# ax[0].bar(
#     x + width / 2,
#     signal_rmse_values,
#     width,
#     color=bar_colours,
#     alpha=0.45,
#     label=r"RMSE $(\Delta s)$",
# )

# ax[0].set_xticks(x, [rf"{v:.2f}$\,\lambda/D$" for v in mask_diam_lambdaD_array])
# ax[0].set_ylabel("Normalized-signal bias")
# ax[0].legend(frameon=True)
# ax[0].grid(axis="y", alpha=0.22)

# ax[1].bar(
#     x - width,
#     wfe_rms_values,
#     width,
#     color=bar_colours,
#     label="Full pupil",
# )

# ax[1].bar(
#     x,
#     centre_values,
#     width,
#     color=bar_colours,
#     alpha=0.70,
#     label=rf"Centre, $\rho<{CENTER_MAX_RHO}$",
# )

# ax[1].bar(
#     x + width,
#     edge_values,
#     width,
#     color=bar_colours,
#     alpha=0.40,
#     label=rf"Edge, $\rho\geq{EDGE_MIN_RHO}$",
# )

# ax[1].set_xticks(x, [rf"{v:.2f}$\,\lambda/D$" for v in mask_diam_lambdaD_array])
# ax[1].set_ylabel("Equivalent WFE bias [nm RMS OPD]")
# ax[1].legend(frameon=True)
# ax[1].grid(axis="y", alpha=0.22)

# add_colourbar(fig, ax)
# fig.savefig(OUTPUT_SUMMARY_FIGURE, bbox_inches="tight")


# # ============================================================
# # Save numerical products and print table
# # ============================================================

# np.savez_compressed(
#     OUTPUT_DATA,
#     mask_names=np.asarray(MASK_NAMES),
#     mask_labels=np.asarray(mask_labels),
#     mask_diam_um=mask_diam_um_array,
#     mask_diam_lambdaD_wvl0=mask_diam_lambdaD_array,
#     radial_edges=radial_edges,
#     radial_centres=radial_centres,
#     pupil_detector=pupil_detector,
#     analysis_mask=analysis_mask,
#     pupil_wave=pupil_wave,
#     rho_detector=rho_detector,
#     rho_wave=rho_wave,
#     N0_internal=N0_internal,
#     N0_onsky=N0_onsky,
#     signal_bias_maps=signal_bias_maps,
#     reconstructed_opd_maps_nm=reconstructed_opd_maps_nm,
#     fitted_signal_maps=fitted_signal_maps,
#     radial_signal_mean=radial_signal_mean,
#     radial_signal_rms=radial_signal_rms,
#     radial_wfe_mean_nm=radial_wfe_mean_nm,
#     radial_wfe_rms_nm=radial_wfe_rms_nm,
#     delta_T_values_K=DELTA_T_VALUES_K,
#     T1_scan_K=np.array(T_INTERNAL_K),
#     T2_scan_K=T_INTERNAL_K + DELTA_T_VALUES_K,
#     peak_wfe_bias_fixed_I2M_nm=peak_wfe_bias_fixed_I2M_nm,
#     peak_wfe_bias_matched_I2M_nm=peak_wfe_bias_matched_I2M_nm,
# )

# print()
# print(
#     "Mask  Diam[um]  Diam[lambda/D]  Peak|ds|    RMSE(ds)    "
#     "WFE_RMS[nm]  Centre[nm]  Edge[nm]  Explained[%]"
# )

# for row in table_rows:
#     print(
#         f"{row['mask']:4s}  "
#         f"{row['mask_diameter_um']:8.0f}  "
#         f"{row['mask_diameter_lambda_over_D']:14.2f}  "
#         f"{row['peak_abs_signal_bias']:9.3e}  "
#         f"{row['signal_bias_rmse']:9.3e}  "
#         f"{row['equivalent_wfe_rms_nm']:11.2f}  "
#         f"{row['centre_wfe_rms_nm']:10.2f}  "
#         f"{row['edge_wfe_rms_nm']:8.2f}  "
#         f"{row['explained_signal_percent']:12.1f}"
#     )

# # Verify that the plotting section did not alter any analysis arrays.
# analysis_arrays_after_plotting = {
#     "radial_signal_mean": radial_signal_mean,
#     "radial_signal_rms": radial_signal_rms,
#     "radial_wfe_mean_nm": radial_wfe_mean_nm,
#     "radial_wfe_rms_nm": radial_wfe_rms_nm,
#     "peak_wfe_bias_fixed_I2M_nm": (
#         peak_wfe_bias_fixed_I2M_nm
#     ),
#     "peak_wfe_bias_matched_I2M_nm": (
#         peak_wfe_bias_matched_I2M_nm
#     ),
# }

# for name, before in analysis_arrays_before_plotting.items():
#     after = analysis_arrays_after_plotting[name]

#     if not np.array_equal(before, after, equal_nan=True):
#         raise RuntimeError(
#             f"Plotting unexpectedly modified numerical array: {name}"
#         )

# def file_sha256(path):
#     path = Path(path).resolve()
#     digest = hashlib.sha256()

#     with open(path, "rb") as f:
#         for block in iter(lambda: f.read(1024 * 1024), b""):
#             digest.update(block)

#     return digest.hexdigest()

# provenance = {
#     "python_executable": str(Path(sys.executable).resolve()),
#     "python_version": platform.python_version(),
#     "repo_root": str(REPO_ROOT),
#     "imports": {
#         "baldr_core": str(Path(bldr.__file__).resolve()),
#         "spectrum": str(Path(spec.__file__).resolve()),
#         "DM_basis": str(Path(DM_basis.__file__).resolve()),
#     },
#     "input_hashes_sha256": {
#         "config": file_sha256(CONFIG_PATH),
#         "phasemask_properties": file_sha256(phasemask_path),
#         "baldr_core": file_sha256(bldr.__file__),
#         "spectrum": file_sha256(spec.__file__),
#         "DM_basis": file_sha256(DM_basis.__file__),
#     },
#     "analysis_fingerprints_sha256": analysis_fingerprints,
#     "mask_order": MASK_NAMES,
#     "mask_diameter_um": mask_diam_um_array.tolist(),
#     "mask_diameter_lambda_over_D_at_wvl0": (
#         mask_diam_lambdaD_array.tolist()
#     ),
#     "temperature_scan": {
#         "T1_K": T_INTERNAL_K,
#         "delta_T_values_K": DELTA_T_VALUES_K.tolist(),
#         "T2_values_K": (
#             T_INTERNAL_K + DELTA_T_VALUES_K
#         ).tolist(),
#         "I2M_fixed_at_T1_for_each_mask": True,
#         "matched_I2M_rebuilt_for_every_T2_and_mask": True,
#         "internal_I2M_calibration_temperature_K": T_INTERNAL_K,
#         "metric": "peak absolute reconstructed pupil OPD bias [nm]",
#         "fixed_bias_definition": (
#             "I2M(T1) @ [I0_norm(T1) - I0_norm(T2)]"
#         ),
#         "matched_bias_definition": (
#             "I2M(T2) @ [I0_norm(T1) - I0_norm(T2)]"
#         ),
#     },
# }

# with open(OUTPUT_PROVENANCE, "w") as f:
#     json.dump(provenance, f, indent=2)

# print()
# for path in [
#     OUTPUT_TABLE_CSV,
#     OUTPUT_TABLE_TEX,
#     OUTPUT_COMBINED_FIGURE,
#     OUTPUT_SUMMARY_FIGURE,
#     OUTPUT_SINGLE_MEAN_SIGNAL,
#     OUTPUT_SINGLE_RMS_SIGNAL,
#     OUTPUT_SINGLE_MEAN_WFE,
#     OUTPUT_SINGLE_RMS_WFE,
#     OUTPUT_DELTA_T_FIGURE,
#     OUTPUT_DELTA_T_CSV,
#     OUTPUT_DATA,
#     OUTPUT_PROVENANCE,
# ]:
#     print(f"Saved: {path.resolve()}")

# plt.show()


# # #!/usr/bin/env python3
# # """
# # Paper-grade chromatic Baldr I0 reference-bias analysis for the physical H-band masks.

# # This keeps the original analysis unchanged, but improves the plotting:
# # - legend labels use physical mask size at wvl0 in lambda/D (and um);
# # - traces are coloured by mask diameter with a continuous colormap;
# # - the original 2x2 summary figure is retained;
# # - each panel is also saved as a standalone figure.
# # """

# # import copy
# # import csv
# # import hashlib
# # import json
# # import platform
# # import sys
# # import tempfile
# # from pathlib import Path

# # import matplotlib.pyplot as plt
# # import numpy as np
# # from matplotlib.colors import Normalize
# # from scipy.ndimage import binary_erosion

# # REPO_ROOT = Path(__file__).resolve().parents[3]
# # sys.path.insert(0, str(REPO_ROOT))

# # from baldrapp.common import DM_basis
# # from baldrapp.common import baldr_core as bldr
# # from baldrapp.common import spectrum as spec


# # # ============================================================
# # # Strict local-import check
# # # ============================================================

# # for module in (bldr, spec, DM_basis):
# #     module_path = Path(module.__file__).resolve()

# #     if REPO_ROOT not in module_path.parents:
# #         raise ImportError(
# #             "BaldrApp import shadowing detected.\n"
# #             f"Expected modules below: {REPO_ROOT}\n"
# #             f"Imported instead: {module_path}\n"
# #             "Run from the repository root, or use PYTHONPATH=$PWD."
# #         )

# # print(f"Python executable: {Path(sys.executable).resolve()}")
# # print(f"Python version: {platform.python_version()}")
# # print(f"baldr_core: {Path(bldr.__file__).resolve()}")
# # print(f"spectrum: {Path(spec.__file__).resolve()}")
# # print(f"DM_basis: {Path(DM_basis.__file__).resolve()}")


# # # ============================================================
# # # Settings
# # # ============================================================

# # CONFIG_PATH = (
# #     REPO_ROOT
# #     / "baldrapp/apps/paranal_simulator/fake_configs/baldr_config.json"
# # )

# # T_INTERNAL_K = 1900.0
# # T_ONSKY_K = 10000.0

# # MASK_NAMES = ["H1", "H2", "H3", "H4", "H5"]

# # N_ZERNIKE_MODES = 20
# # LINEAR_POKE_NM = 10.0
# # SVD_RELATIVE_CUTOFF = 1e-3

# # N_RADIAL_BINS = 10
# # CENTER_MAX_RHO = 0.4
# # EDGE_MIN_RHO = 0.8

# # # Temperature-difference scan. T1 is fixed at T_INTERNAL_K and
# # # T2 = T1 + delta_T. Zero is included explicitly; positive values are
# # # logarithmically spaced to 20,000 K.
# # DELTA_T_MAX_K = 20000.0
# # N_DELTA_T_LOG_SAMPLES = 18
# # DELTA_T_VALUES_K = np.concatenate(
# #     (
# #         np.array([0.0]),
# #         np.geomspace(1.0, DELTA_T_MAX_K, N_DELTA_T_LOG_SAMPLES),
# #     )
# # )

# # OUTPUT_TABLE_CSV = Path("chromatic_I0_bias_table.csv")
# # OUTPUT_TABLE_TEX = Path("chromatic_I0_bias_table.tex")
# # OUTPUT_DATA = Path("chromatic_I0_bias_results.npz")
# # OUTPUT_PROVENANCE = Path("chromatic_I0_bias_provenance.json")

# # OUTPUT_COMBINED_FIGURE = Path("chromatic_I0_bias_radial_profiles_paper.png")
# # OUTPUT_SUMMARY_FIGURE = Path("chromatic_I0_bias_summary_paper.png")

# # OUTPUT_SINGLE_MEAN_SIGNAL = Path("chromatic_I0_bias_signal_mean_radial.png")
# # OUTPUT_SINGLE_RMS_SIGNAL = Path("chromatic_I0_bias_signal_rms_radial.png")
# # OUTPUT_SINGLE_MEAN_WFE = Path("chromatic_I0_bias_wfe_mean_radial.png")
# # OUTPUT_SINGLE_RMS_WFE = Path("chromatic_I0_bias_wfe_rms_radial.png")

# # OUTPUT_DELTA_T_FIGURE = Path(
# #     "chromatic_I0_peak_WFE_bias_vs_deltaT.png"
# # )
# # OUTPUT_DELTA_T_CSV = Path(
# #     "chromatic_I0_peak_WFE_bias_vs_deltaT.csv"
# # )


# # # ============================================================
# # # Plot style
# # # ============================================================

# # plt.rcParams.update(
# #     {
# #         "figure.dpi": 140,
# #         "savefig.dpi": 300,
# #         "font.size": 11,
# #         "axes.titlesize": 13,
# #         "axes.labelsize": 12,
# #         "legend.fontsize": 10,
# #         "xtick.labelsize": 10,
# #         "ytick.labelsize": 10,
# #         "lines.linewidth": 2.2,
# #         "lines.markersize": 6.0,
# #         "axes.grid": True,
# #         "grid.alpha": 0.22,
# #         "grid.linewidth": 0.7,
# #         "mathtext.default": "regular",
# #     }
# # )


# # # ============================================================
# # # Initialize identical aligned states with different spectra
# # # ============================================================

# # with open(CONFIG_PATH, "r") as f:
# #     base_cfg = json.load(f)

# # cfg_internal = copy.deepcopy(base_cfg)
# # cfg_onsky = copy.deepcopy(base_cfg)

# # for cfg, temperature_K in (
# #     (cfg_internal, T_INTERNAL_K),
# #     (cfg_onsky, T_ONSKY_K),
# # ):
# #     cfg["stellar"]["spectrum"]["enabled"] = True
# #     cfg["stellar"]["spectrum"]["mode"] = "blackbody"
# #     cfg["stellar"]["spectrum"]["temperature_K"] = temperature_K

# #     cfg["fresnel_relay"]["coldstop_x_offset"] = 0.0
# #     cfg["fresnel_relay"]["coldstop_y_offset"] = 0.0
# #     cfg["fresnel_relay"]["pupil_misconjugation"] = 0.0
# #     cfg["fresnel_relay"]["edge_offset"] = 0.0
# #     cfg["fresnel_relay"]["edge_angle"] = 0.0
# #     cfg["fresnel_relay"]["use_nominal_pupil_conjugation"] = True

# #     cfg["internal_aberrations"]["enabled"] = False

# #     cfg["detector"]["enabled"] = True
# #     cfg["detector"]["ron"] = 0.0
# #     cfg["detector"]["include_shotnoise"] = False
# #     cfg["detector"]["include_readnoise"] = False
# #     cfg["detector"]["adu_offset"] = 0.0
# #     cfg["detector"]["noise_std_adu"] = 0.0

# # with tempfile.TemporaryDirectory() as tmp:
# #     tmp = Path(tmp)
# #     internal_path = tmp / "internal_1900K.json"
# #     onsky_path = tmp / "onsky_10000K.json"

# #     with open(internal_path, "w") as f:
# #         json.dump(cfg_internal, f, indent=2)

# #     with open(onsky_path, "w") as f:
# #         json.dump(cfg_onsky, f, indent=2)

# #     zwfs_internal = bldr.init_zwfs_from_json(internal_path)
# #     zwfs_onsky = bldr.init_zwfs_from_json(onsky_path)


# # # ============================================================
# # # Physical masks, pupil masks, and exact matched N0 references
# # # ============================================================

# # phasemask_path = (
# #     REPO_ROOT
# #     / base_cfg["simulator_runtime"]["phasemask"]["properties_file"]
# # )

# # with open(phasemask_path, "r") as f:
# #     phasemask_cfg = json.load(f)

# # mask_entries = phasemask_cfg["phasemask"]["masks"]

# # amp_internal = zwfs_internal.grid.pupil_mask.astype(float)
# # amp_onsky = zwfs_onsky.grid.pupil_mask.astype(float)

# # zero_internal = np.zeros_like(amp_internal)
# # zero_onsky = np.zeros_like(amp_onsky)

# # N0_internal = bldr.get_N0_configured(
# #     opd_input=zero_internal,
# #     amp_input=amp_internal,
# #     opd_internal=zero_internal,
# #     zwfs_ns=zwfs_internal,
# #     detector=zwfs_internal.detector,
# #     include_shotnoise=False,
# # )

# # N0_onsky = bldr.get_N0_configured(
# #     opd_input=zero_onsky,
# #     amp_input=amp_onsky,
# #     opd_internal=zero_onsky,
# #     zwfs_ns=zwfs_onsky,
# #     detector=zwfs_onsky.detector,
# #     include_shotnoise=False,
# # )

# # binning = int(base_cfg["detector"]["binning"])

# # pupil_detector = (
# #     bldr.sum_subarrays(
# #         zwfs_onsky.grid.pupil_mask,
# #         block_size=(binning, binning),
# #     )
# #     > 0.5 * binning**2
# # )

# # analysis_mask = binary_erosion(pupil_detector, iterations=1)

# # if not np.any(analysis_mask):
# #     raise RuntimeError("The eroded detector pupil mask is empty.")

# # N0_internal_mean = np.mean(N0_internal[analysis_mask])
# # N0_onsky_mean = np.mean(N0_onsky[analysis_mask])

# # yy_det, xx_det = np.indices(pupil_detector.shape)
# # cy_det = np.mean(yy_det[pupil_detector])
# # cx_det = np.mean(xx_det[pupil_detector])

# # r_det = np.sqrt((xx_det - cx_det) ** 2 + (yy_det - cy_det) ** 2)
# # outer_radius_det = np.percentile(r_det[pupil_detector], 99.5)
# # rho_detector = r_det / outer_radius_det

# # pupil_wave = zwfs_onsky.grid.pupil_mask.astype(bool)

# # yy_wave, xx_wave = np.indices(pupil_wave.shape)
# # cy_wave = np.mean(yy_wave[pupil_wave])
# # cx_wave = np.mean(xx_wave[pupil_wave])

# # r_wave = np.sqrt((xx_wave - cx_wave) ** 2 + (yy_wave - cy_wave) ** 2)
# # outer_radius_wave = np.percentile(r_wave[pupil_wave], 99.5)
# # rho_wave = r_wave / outer_radius_wave
# # theta_wave = np.arctan2(yy_wave - cy_wave, xx_wave - cx_wave)

# # radial_edges = np.linspace(0.0, 1.0, N_RADIAL_BINS + 1)
# # radial_centres = 0.5 * (radial_edges[:-1] + radial_edges[1:])


# # # ============================================================
# # # Common phase basis: piston removed, each mode = 1 nm RMS OPD
# # # ============================================================

# # raw_zernikes = DM_basis.zernike_basis(
# #     nterms=N_ZERNIKE_MODES + 1,
# #     rho=rho_wave,
# #     theta=theta_wave,
# #     outside=0.0,
# # )

# # modes = []

# # for mode_index in range(1, N_ZERNIKE_MODES + 1):
# #     mode = np.nan_to_num(raw_zernikes[mode_index], nan=0.0)
# #     mode *= pupil_wave
# #     mode -= np.mean(mode[pupil_wave])

# #     mode_rms = np.sqrt(np.mean(mode[pupil_wave] ** 2))

# #     if mode_rms <= 0:
# #         raise RuntimeError(f"Zernike mode {mode_index + 1} has zero RMS.")

# #     modes.append(mode / mode_rms)

# # modes = np.asarray(modes)


# # # ============================================================
# # # Per-mask signal bias and on-sky linear reconstruction bias
# # # ============================================================

# # table_rows = []

# # signal_bias_maps = []
# # reconstructed_opd_maps_nm = []
# # fitted_signal_maps = []
# # I0_internal_norm_by_mask = []

# # radial_signal_mean = np.full((len(MASK_NAMES), N_RADIAL_BINS), np.nan)
# # radial_signal_rms = np.full((len(MASK_NAMES), N_RADIAL_BINS), np.nan)
# # radial_wfe_mean_nm = np.full((len(MASK_NAMES), N_RADIAL_BINS), np.nan)
# # radial_wfe_rms_nm = np.full((len(MASK_NAMES), N_RADIAL_BINS), np.nan)

# # mask_diam_um_list = []
# # mask_diam_lambdaD_list = []
# # mask_labels = []

# # for mask_number, mask_name in enumerate(MASK_NAMES):
# #     print(f"Processing {mask_name} ({mask_number + 1}/{len(MASK_NAMES)})")

# #     mask_entry = mask_entries[mask_name]

# #     active_mask_internal = spec.normalise_phasemask_entry(
# #         mask_name,
# #         mask_entry,
# #         zwfs_internal.optics,
# #     )

# #     active_mask_onsky = spec.normalise_phasemask_entry(
# #         mask_name,
# #         mask_entry,
# #         zwfs_onsky.optics,
# #     )

# #     zwfs_internal.optics.active_phasemask = active_mask_internal
# #     zwfs_onsky.optics.active_phasemask = active_mask_onsky

# #     mask_diam_um = float(active_mask_onsky.mask_diam_um)
# #     mask_diam_lambdaD = float(active_mask_onsky.mask_diam_lambdaD_wvl0)

# #     mask_diam_um_list.append(mask_diam_um)
# #     mask_diam_lambdaD_list.append(mask_diam_lambdaD)
# #     mask_labels.append(
# #         rf"{mask_diam_lambdaD:.2f}$\,\lambda/D$ ({mask_diam_um:.0f}$\,\mu$m)"
# #     )

# #     I0_internal = bldr.get_I0_configured(
# #         opd_input=zero_internal,
# #         amp_input=amp_internal,
# #         opd_internal=zero_internal,
# #         zwfs_ns=zwfs_internal,
# #         detector=zwfs_internal.detector,
# #         include_shotnoise=False,
# #     )

# #     I0_onsky = bldr.get_I0_configured(
# #         opd_input=zero_onsky,
# #         amp_input=amp_onsky,
# #         opd_internal=zero_onsky,
# #         zwfs_ns=zwfs_onsky,
# #         detector=zwfs_onsky.detector,
# #         include_shotnoise=False,
# #     )

# #     I0_internal_norm = I0_internal / N0_internal_mean
# #     I0_onsky_norm = I0_onsky / N0_onsky_mean

# #     I0_internal_norm_by_mask.append(I0_internal_norm.copy())

# #     signal_bias = I0_onsky_norm - I0_internal_norm
# #     signal_vector = signal_bias[analysis_mask]

# #     signal_peak_abs = np.max(np.abs(signal_vector))
# #     signal_rmse = np.sqrt(np.mean(signal_vector**2))

# #     response_columns = []

# #     for mode in modes:
# #         opd_plus = LINEAR_POKE_NM * 1e-9 * mode
# #         opd_minus = -LINEAR_POKE_NM * 1e-9 * mode

# #         I_plus = bldr.get_I0_configured(
# #             opd_input=opd_plus,
# #             amp_input=amp_onsky,
# #             opd_internal=zero_onsky,
# #             zwfs_ns=zwfs_onsky,
# #             detector=zwfs_onsky.detector,
# #             include_shotnoise=False,
# #         )

# #         I_minus = bldr.get_I0_configured(
# #             opd_input=opd_minus,
# #             amp_input=amp_onsky,
# #             opd_internal=zero_onsky,
# #             zwfs_ns=zwfs_onsky,
# #             detector=zwfs_onsky.detector,
# #             include_shotnoise=False,
# #         )

# #         I_plus_norm = I_plus / N0_onsky_mean
# #         I_minus_norm = I_minus / N0_onsky_mean

# #         derivative_per_nm = (
# #             I_plus_norm - I_minus_norm
# #         ) / (2.0 * LINEAR_POKE_NM)

# #         response_columns.append(derivative_per_nm[analysis_mask])

# #     interaction_matrix = np.column_stack(response_columns)

# #     U, singular_values, Vt = np.linalg.svd(
# #         interaction_matrix,
# #         full_matrices=False,
# #     )

# #     keep = singular_values > (
# #         SVD_RELATIVE_CUTOFF * singular_values[0]
# #     )

# #     if not np.any(keep):
# #         raise RuntimeError(f"No singular values retained for {mask_name}.")

# #     reconstructor = (
# #         Vt[keep].T
# #         @ np.diag(1.0 / singular_values[keep])
# #         @ U[:, keep].T
# #     )

# #     coefficients_nm = reconstructor @ signal_vector

# #     reconstructed_opd_nm = np.sum(
# #         coefficients_nm[:, None, None] * modes,
# #         axis=0,
# #     )

# #     fitted_signal_vector = interaction_matrix @ coefficients_nm
# #     fitted_signal_map = np.full_like(signal_bias, np.nan, dtype=float)
# #     fitted_signal_map[analysis_mask] = fitted_signal_vector

# #     signal_power = np.sum(signal_vector**2)
# #     residual_power = np.sum(
# #         (signal_vector - fitted_signal_vector) ** 2
# #     )

# #     explained_fraction = (
# #         1.0 - residual_power / signal_power
# #         if signal_power > 0
# #         else np.nan
# #     )

# #     wfe_values_nm = reconstructed_opd_nm[pupil_wave]
# #     equivalent_wfe_rms_nm = np.sqrt(np.mean(wfe_values_nm**2))
# #     peak_abs_wfe_nm = np.max(np.abs(wfe_values_nm))

# #     centre_mask_wave = pupil_wave & (rho_wave < CENTER_MAX_RHO)
# #     edge_mask_wave = (
# #         pupil_wave
# #         & (rho_wave >= EDGE_MIN_RHO)
# #         & (rho_wave <= 1.0)
# #     )

# #     centre_wfe_rms_nm = np.sqrt(
# #         np.mean(reconstructed_opd_nm[centre_mask_wave] ** 2)
# #     )

# #     edge_wfe_rms_nm = np.sqrt(
# #         np.mean(reconstructed_opd_nm[edge_mask_wave] ** 2)
# #     )

# #     for radial_index in range(N_RADIAL_BINS):
# #         detector_annulus = (
# #             analysis_mask
# #             & (rho_detector >= radial_edges[radial_index])
# #             & (rho_detector < radial_edges[radial_index + 1])
# #         )

# #         wave_annulus = (
# #             pupil_wave
# #             & (rho_wave >= radial_edges[radial_index])
# #             & (rho_wave < radial_edges[radial_index + 1])
# #         )

# #         if np.any(detector_annulus):
# #             radial_values = signal_bias[detector_annulus]

# #             radial_signal_mean[mask_number, radial_index] = np.mean(
# #                 radial_values
# #             )

# #             radial_signal_rms[mask_number, radial_index] = np.sqrt(
# #                 np.mean(radial_values**2)
# #             )

# #         if np.any(wave_annulus):
# #             radial_values_nm = reconstructed_opd_nm[wave_annulus]

# #             radial_wfe_mean_nm[mask_number, radial_index] = np.mean(
# #                 radial_values_nm
# #             )

# #             radial_wfe_rms_nm[mask_number, radial_index] = np.sqrt(
# #                 np.mean(radial_values_nm**2)
# #             )

# #     condition_number_retained = (
# #         singular_values[keep][0] / singular_values[keep][-1]
# #     )

# #     table_rows.append(
# #         {
# #             "mask": mask_name,
# #             "mask_diameter_um": float(mask_diam_um),
# #             "mask_diameter_lambda_over_D": float(mask_diam_lambdaD),
# #             "peak_abs_signal_bias": float(signal_peak_abs),
# #             "signal_bias_rmse": float(signal_rmse),
# #             "equivalent_wfe_rms_nm": float(equivalent_wfe_rms_nm),
# #             "peak_abs_wfe_nm": float(peak_abs_wfe_nm),
# #             "centre_wfe_rms_nm": float(centre_wfe_rms_nm),
# #             "edge_wfe_rms_nm": float(edge_wfe_rms_nm),
# #             "explained_signal_percent": float(
# #                 100.0 * explained_fraction
# #             ),
# #             "retained_rank": int(np.sum(keep)),
# #             "retained_condition_number": float(
# #                 condition_number_retained
# #             ),
# #         }
# #     )

# #     signal_bias_maps.append(signal_bias)
# #     reconstructed_opd_maps_nm.append(reconstructed_opd_nm)
# #     fitted_signal_maps.append(fitted_signal_map)

# # signal_bias_maps = np.asarray(signal_bias_maps)
# # reconstructed_opd_maps_nm = np.asarray(reconstructed_opd_maps_nm)
# # fitted_signal_maps = np.asarray(fitted_signal_maps)
# # I0_internal_norm_by_mask = np.asarray(I0_internal_norm_by_mask)

# # mask_diam_um_array = np.asarray(mask_diam_um_list)
# # mask_diam_lambdaD_array = np.asarray(mask_diam_lambdaD_list)

# # # Preserve the exact H1--H5 calculation order. The physical diameters are
# # # already monotonic, so no numerical arrays need to be reordered for plotting.
# # if not np.all(np.diff(mask_diam_lambdaD_array) > 0):
# #     raise RuntimeError(
# #         "MASK_NAMES are not ordered by increasing physical mask diameter."
# #     )

# # cmap = plt.get_cmap("viridis")
# # norm = Normalize(
# #     vmin=float(np.min(mask_diam_lambdaD_array)),
# #     vmax=float(np.max(mask_diam_lambdaD_array)),
# # )
# # trace_colours = [cmap(norm(v)) for v in mask_diam_lambdaD_array]


# # # ============================================================
# # # Worst-case peak WFE bias versus source-temperature difference
# # # ============================================================

# # peak_wfe_bias_vs_deltaT_nm = np.zeros(
# #     (len(MASK_NAMES), len(DELTA_T_VALUES_K)),
# #     dtype=float,
# # )

# # with tempfile.TemporaryDirectory() as scan_tmp:
# #     scan_tmp = Path(scan_tmp)

# #     for delta_index, delta_T_K in enumerate(DELTA_T_VALUES_K):
# #         T2_K = T_INTERNAL_K + delta_T_K

# #         print(
# #             f"Delta-T scan {delta_index + 1}/{len(DELTA_T_VALUES_K)}: "
# #             f"delta_T={delta_T_K:.6g} K, T2={T2_K:.6g} K"
# #         )

# #         # Identical spectra give exactly zero bias.
# #         if delta_T_K == 0.0:
# #             continue

# #         cfg_T2 = copy.deepcopy(base_cfg)
# #         cfg_T2["stellar"]["spectrum"]["enabled"] = True
# #         cfg_T2["stellar"]["spectrum"]["mode"] = "blackbody"
# #         cfg_T2["stellar"]["spectrum"]["temperature_K"] = float(T2_K)

# #         cfg_T2["fresnel_relay"]["coldstop_x_offset"] = 0.0
# #         cfg_T2["fresnel_relay"]["coldstop_y_offset"] = 0.0
# #         cfg_T2["fresnel_relay"]["pupil_misconjugation"] = 0.0
# #         cfg_T2["fresnel_relay"]["edge_offset"] = 0.0
# #         cfg_T2["fresnel_relay"]["edge_angle"] = 0.0
# #         cfg_T2["fresnel_relay"]["use_nominal_pupil_conjugation"] = True

# #         cfg_T2["internal_aberrations"]["enabled"] = False

# #         cfg_T2["detector"]["enabled"] = True
# #         cfg_T2["detector"]["ron"] = 0.0
# #         cfg_T2["detector"]["include_shotnoise"] = False
# #         cfg_T2["detector"]["include_readnoise"] = False
# #         cfg_T2["detector"]["adu_offset"] = 0.0
# #         cfg_T2["detector"]["noise_std_adu"] = 0.0

# #         T2_path = scan_tmp / f"T2_{delta_index:03d}.json"

# #         with open(T2_path, "w") as f:
# #             json.dump(cfg_T2, f, indent=2)

# #         zwfs_T2 = bldr.init_zwfs_from_json(T2_path)

# #         amp_T2 = zwfs_T2.grid.pupil_mask.astype(float)
# #         zero_T2 = np.zeros_like(amp_T2)

# #         if amp_T2.shape != amp_onsky.shape:
# #             raise RuntimeError(
# #                 "T2 pupil shape differs from the reference pupil shape."
# #             )

# #         N0_T2 = bldr.get_N0_configured(
# #             opd_input=zero_T2,
# #             amp_input=amp_T2,
# #             opd_internal=zero_T2,
# #             zwfs_ns=zwfs_T2,
# #             detector=zwfs_T2.detector,
# #             include_shotnoise=False,
# #         )

# #         if N0_T2.shape != analysis_mask.shape:
# #             raise RuntimeError(
# #                 "T2 detector image shape differs from analysis_mask."
# #             )

# #         N0_T2_mean = np.mean(N0_T2[analysis_mask])

# #         for mask_number, mask_name in enumerate(MASK_NAMES):
# #             mask_entry = mask_entries[mask_name]

# #             zwfs_T2.optics.active_phasemask = (
# #                 spec.normalise_phasemask_entry(
# #                     mask_name,
# #                     mask_entry,
# #                     zwfs_T2.optics,
# #                 )
# #             )

# #             I0_T2 = bldr.get_I0_configured(
# #                 opd_input=zero_T2,
# #                 amp_input=amp_T2,
# #                 opd_internal=zero_T2,
# #                 zwfs_ns=zwfs_T2,
# #                 detector=zwfs_T2.detector,
# #                 include_shotnoise=False,
# #             )

# #             I0_T2_norm = I0_T2 / N0_T2_mean

# #             # Requested sign convention. The peak absolute WFE is independent
# #             # of reversing this difference.
# #             reference_difference = (
# #                 I0_internal_norm_by_mask[mask_number]
# #                 - I0_T2_norm
# #             )

# #             response_columns_T2 = []

# #             for mode in modes:
# #                 opd_plus = LINEAR_POKE_NM * 1e-9 * mode
# #                 opd_minus = -LINEAR_POKE_NM * 1e-9 * mode

# #                 I_plus = bldr.get_I0_configured(
# #                     opd_input=opd_plus,
# #                     amp_input=amp_T2,
# #                     opd_internal=zero_T2,
# #                     zwfs_ns=zwfs_T2,
# #                     detector=zwfs_T2.detector,
# #                     include_shotnoise=False,
# #                 )

# #                 I_minus = bldr.get_I0_configured(
# #                     opd_input=opd_minus,
# #                     amp_input=amp_T2,
# #                     opd_internal=zero_T2,
# #                     zwfs_ns=zwfs_T2,
# #                     detector=zwfs_T2.detector,
# #                     include_shotnoise=False,
# #                 )

# #                 response_columns_T2.append(
# #                     (
# #                         (I_plus - I_minus)
# #                         / N0_T2_mean
# #                         / (2.0 * LINEAR_POKE_NM)
# #                     )[analysis_mask]
# #                 )

# #             interaction_matrix_T2 = np.column_stack(
# #                 response_columns_T2
# #             )

# #             U_T2, singular_values_T2, Vt_T2 = np.linalg.svd(
# #                 interaction_matrix_T2,
# #                 full_matrices=False,
# #             )

# #             keep_T2 = singular_values_T2 > (
# #                 SVD_RELATIVE_CUTOFF * singular_values_T2[0]
# #             )

# #             if not np.any(keep_T2):
# #                 raise RuntimeError(
# #                     f"No singular values retained for {mask_name} "
# #                     f"at T2={T2_K:.3f} K."
# #                 )

# #             I2M_T2 = (
# #                 Vt_T2[keep_T2].T
# #                 @ np.diag(1.0 / singular_values_T2[keep_T2])
# #                 @ U_T2[:, keep_T2].T
# #             )

# #             coefficients_bias_nm = (
# #                 I2M_T2
# #                 @ reference_difference[analysis_mask]
# #             )

# #             reconstructed_bias_nm = np.sum(
# #                 coefficients_bias_nm[:, None, None] * modes,
# #                 axis=0,
# #             )

# #             peak_wfe_bias_vs_deltaT_nm[
# #                 mask_number,
# #                 delta_index,
# #             ] = np.max(
# #                 np.abs(reconstructed_bias_nm[pupil_wave])
# #             )


# # with open(OUTPUT_DELTA_T_CSV, "w", newline="") as f:
# #     writer = csv.writer(f)

# #     writer.writerow(
# #         [
# #             "mask",
# #             "mask_diameter_um",
# #             "mask_diameter_lambda_over_D",
# #             "delta_T_K",
# #             "T1_K",
# #             "T2_K",
# #             "peak_abs_reconstructed_WFE_bias_nm",
# #         ]
# #     )

# #     for mask_number, mask_name in enumerate(MASK_NAMES):
# #         for delta_index, delta_T_K in enumerate(DELTA_T_VALUES_K):
# #             writer.writerow(
# #                 [
# #                     mask_name,
# #                     mask_diam_um_array[mask_number],
# #                     mask_diam_lambdaD_array[mask_number],
# #                     delta_T_K,
# #                     T_INTERNAL_K,
# #                     T_INTERNAL_K + delta_T_K,
# #                     peak_wfe_bias_vs_deltaT_nm[
# #                         mask_number,
# #                         delta_index,
# #                     ],
# #                 ]
# #             )


# # # Immutable snapshots and numerical fingerprints before any plotting.
# # analysis_arrays_before_plotting = {
# #     "radial_signal_mean": radial_signal_mean.copy(),
# #     "radial_signal_rms": radial_signal_rms.copy(),
# #     "radial_wfe_mean_nm": radial_wfe_mean_nm.copy(),
# #     "radial_wfe_rms_nm": radial_wfe_rms_nm.copy(),
# #     "peak_wfe_bias_vs_deltaT_nm": (
# #         peak_wfe_bias_vs_deltaT_nm.copy()
# #     ),
# # }

# # analysis_fingerprints = {
# #     name: hashlib.sha256(
# #         np.ascontiguousarray(values).view(np.uint8)
# #     ).hexdigest()
# #     for name, values in analysis_arrays_before_plotting.items()
# # }

# # print("Numerical result fingerprints:")
# # for name, digest in analysis_fingerprints.items():
# #     print(f"  {name}: {digest}")


# # # ============================================================
# # # CSV and LaTeX tables
# # # ============================================================

# # fieldnames = list(table_rows[0].keys())

# # with open(OUTPUT_TABLE_CSV, "w", newline="") as f:
# #     writer = csv.DictWriter(f, fieldnames=fieldnames)
# #     writer.writeheader()
# #     writer.writerows(table_rows)

# # with open(OUTPUT_TABLE_TEX, "w") as f:
# #     f.write("\\begin{tabular}{lrrrrrrrr}\n")
# #     f.write("\\hline\n")
# #     f.write(
# #         "Mask & Diam. & Diam. & Peak $|\\Delta s|$ & "
# #         "RMSE $\\Delta s$ & WFE RMS & Peak $|\\phi|$ & "
# #         "Centre RMS & Edge RMS \\\\\n"
# #     )
# #     f.write(
# #         " & [$\\mu$m] & [$\\lambda/D$] & & & [nm] & [nm] & [nm] & [nm] \\\\\n"
# #     )
# #     f.write("\\hline\n")

# #     for row in table_rows:
# #         f.write(
# #             f"{row['mask']} & "
# #             f"{row['mask_diameter_um']:.0f} & "
# #             f"{row['mask_diameter_lambda_over_D']:.2f} & "
# #             f"{row['peak_abs_signal_bias']:.3e} & "
# #             f"{row['signal_bias_rmse']:.3e} & "
# #             f"{row['equivalent_wfe_rms_nm']:.2f} & "
# #             f"{row['peak_abs_wfe_nm']:.2f} & "
# #             f"{row['centre_wfe_rms_nm']:.2f} & "
# #             f"{row['edge_wfe_rms_nm']:.2f} \\\\\n"
# #         )

# #     f.write("\\hline\n")
# #     f.write("\\end{tabular}\n")


# # # ============================================================
# # # Plot helpers
# # # ============================================================

# # def decorate_axis(ax, title, ylabel, yzero=False, xlab=False):
# #     ax.set_title(title, pad=8)
# #     ax.set_ylabel(ylabel)
# #     if xlab:
# #         ax.set_xlabel("Normalized pupil radius")
# #     if yzero:
# #         ax.axhline(0.0, color="0.25", linewidth=1.0, zorder=0)
# #     ax.axvline(
# #         EDGE_MIN_RHO,
# #         linestyle=":",
# #         linewidth=1.2,
# #         color="0.4",
# #         zorder=0,
# #     )
# #     ax.set_xlim(radial_edges[0], radial_edges[-1])


# # def add_colourbar(fig, axs):
# #     sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
# #     sm.set_array([])
# #     cbar = fig.colorbar(
# #         sm,
# #         ax=axs,
# #         shrink=0.94,
# #         pad=0.02,
# #     )
# #     cbar.set_label(r"Mask diameter at $wvl_0$ [$\lambda/D$]")
# #     return cbar


# # def draw_panel(ax, ydata, yzero, ylabel, title, show_legend=False):
# #     for i, label in enumerate(mask_labels):
# #         ax.plot(
# #             radial_centres,
# #             ydata[i],
# #             marker="o",
# #             color=trace_colours[i],
# #             label=label,
# #         )
# #     decorate_axis(ax, title=title, ylabel=ylabel, yzero=yzero, xlab=True)
# #     if show_legend:
# #         ax.legend(frameon=True, ncol=1, loc="best")


# # def save_standalone_figure(filename, ydata, yzero, ylabel, title):
# #     fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
# #     for i, label in enumerate(mask_labels):
# #         ax.plot(
# #             radial_centres,
# #             ydata[i],
# #             marker="o",
# #             color=trace_colours[i],
# #             label=label,
# #         )
# #     decorate_axis(ax, title=title, ylabel=ylabel, yzero=yzero, xlab=True)
# #     ax.legend(frameon=True, loc="best", title="Phase mask")
# #     add_colourbar(fig, ax)
# #     fig.savefig(filename, bbox_inches="tight")
# #     plt.close(fig)


# # # ============================================================
# # # Combined radial profile figure
# # # ============================================================

# # fig, ax = plt.subplots(2, 2, figsize=(12.6, 9.2), constrained_layout=True)

# # for i, label in enumerate(mask_labels):
# #     ax[0, 0].plot(
# #         radial_centres,
# #         radial_signal_mean[i],
# #         marker="o",
# #         color=trace_colours[i],
# #         label=label,
# #     )
# #     ax[0, 1].plot(
# #         radial_centres,
# #         radial_signal_rms[i],
# #         marker="o",
# #         color=trace_colours[i],
# #         label=label,
# #     )
# #     ax[1, 0].plot(
# #         radial_centres,
# #         radial_wfe_mean_nm[i],
# #         marker="o",
# #         color=trace_colours[i],
# #         label=label,
# #     )
# #     ax[1, 1].plot(
# #         radial_centres,
# #         radial_wfe_rms_nm[i],
# #         marker="o",
# #         color=trace_colours[i],
# #         label=label,
# #     )

# # decorate_axis(
# #     ax[0, 0],
# #     title="Signed radial signal bias",
# #     ylabel="Mean normalized-signal bias",
# #     yzero=True,
# #     xlab=False,
# # )
# # decorate_axis(
# #     ax[0, 1],
# #     title="Radial signal-bias magnitude",
# #     ylabel="RMS normalized-signal bias",
# #     yzero=False,
# #     xlab=False,
# # )
# # decorate_axis(
# #     ax[1, 0],
# #     title="Signed radial reconstruction bias",
# #     ylabel="Mean reconstructed OPD bias [nm]",
# #     yzero=True,
# #     xlab=True,
# # )
# # decorate_axis(
# #     ax[1, 1],
# #     title="Radial reconstruction-bias magnitude",
# #     ylabel="RMS reconstructed OPD bias [nm]",
# #     yzero=False,
# #     xlab=True,
# # )

# # legend = ax[0, 0].legend(
# #     title="Phase mask",
# #     frameon=True,
# #     ncol=1,
# #     loc="best",
# # )

# # fig.suptitle(
# #     rf"Chromatic $I_0$ bias: {T_INTERNAL_K:.0f} K internal reference used on "
# #     rf"{T_ONSKY_K:.0f} K source",
# #     y=1.01,
# #     fontsize=15,
# # )

# # add_colourbar(fig, ax)
# # fig.savefig(OUTPUT_COMBINED_FIGURE, bbox_inches="tight")


# # # ============================================================
# # # Standalone radial profile figures
# # # ============================================================

# # save_standalone_figure(
# #     OUTPUT_SINGLE_MEAN_SIGNAL,
# #     radial_signal_mean,
# #     True,
# #     "Mean normalized-signal bias",
# #     "Signed radial signal bias",
# # )

# # save_standalone_figure(
# #     OUTPUT_SINGLE_RMS_SIGNAL,
# #     radial_signal_rms,
# #     False,
# #     "RMS normalized-signal bias",
# #     "Radial signal-bias magnitude",
# # )

# # save_standalone_figure(
# #     OUTPUT_SINGLE_MEAN_WFE,
# #     radial_wfe_mean_nm,
# #     True,
# #     "Mean reconstructed OPD bias [nm]",
# #     "Signed radial reconstruction bias",
# # )

# # save_standalone_figure(
# #     OUTPUT_SINGLE_RMS_WFE,
# #     radial_wfe_rms_nm,
# #     False,
# #     "RMS reconstructed OPD bias [nm]",
# #     "Radial reconstruction-bias magnitude",
# # )


# # # ============================================================
# # # Peak WFE bias versus delta T
# # # ============================================================

# # fig, ax = plt.subplots(figsize=(7.1, 5.2), constrained_layout=True)

# # for mask_number, label in enumerate(mask_labels):
# #     ax.plot(
# #         DELTA_T_VALUES_K,
# #         peak_wfe_bias_vs_deltaT_nm[mask_number],
# #         marker="o",
# #         color=trace_colours[mask_number],
# #         label=label,
# #     )

# # # A pure logarithmic axis cannot contain delta_T=0. Symlog preserves the
# # # exact zero point and is logarithmic above the small linear region.
# # ax.set_xscale(
# #     "symlog",
# #     linthresh=1.0,
# #     linscale=1.0,
# #     base=10,
# # )

# # ax.set_xticks(
# #     [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0, 20000.0]
# # )
# # ax.set_xticklabels(
# #     ["0", "1", "10", "100", r"$10^3$", r"$10^4$", r"$2\times10^4$"]
# # )

# # ax.set_xlabel(r"Source-temperature difference $\Delta T=T_2-T_1$ [K]")
# # ax.set_ylabel("Peak absolute reconstructed WFE bias [nm]")
# # ax.set_title(
# #     rf"Worst-case chromatic reference bias, $T_1={T_INTERNAL_K:.0f}$ K"
# # )
# # ax.set_xlim(0.0, DELTA_T_MAX_K)
# # ax.set_ylim(bottom=0.0)
# # ax.legend(title="Phase mask", frameon=True)
# # ax.grid(True, which="both", alpha=0.22)

# # add_colourbar(fig, ax)
# # fig.savefig(OUTPUT_DELTA_T_FIGURE, bbox_inches="tight")


# # # ============================================================
# # # Global summary figure
# # # ============================================================

# # signal_peak_values = [
# #     row["peak_abs_signal_bias"] for row in table_rows
# # ]
# # signal_rmse_values = [
# #     row["signal_bias_rmse"] for row in table_rows
# # ]
# # wfe_rms_values = [
# #     row["equivalent_wfe_rms_nm"] for row in table_rows
# # ]
# # centre_values = [
# #     row["centre_wfe_rms_nm"] for row in table_rows
# # ]
# # edge_values = [
# #     row["edge_wfe_rms_nm"] for row in table_rows
# # ]

# # x = np.arange(len(mask_labels))
# # width = 0.36

# # fig, ax = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)

# # bar_colours = trace_colours

# # ax[0].bar(
# #     x - width / 2,
# #     signal_peak_values,
# #     width,
# #     color=bar_colours,
# #     label=r"Peak $|\Delta s|$",
# # )

# # ax[0].bar(
# #     x + width / 2,
# #     signal_rmse_values,
# #     width,
# #     color=bar_colours,
# #     alpha=0.45,
# #     label=r"RMSE $(\Delta s)$",
# # )

# # ax[0].set_xticks(x, [rf"{v:.2f}$\,\lambda/D$" for v in mask_diam_lambdaD_array])
# # ax[0].set_ylabel("Normalized-signal bias")
# # ax[0].set_title("Reference signal bias")
# # ax[0].legend(frameon=True)
# # ax[0].grid(axis="y", alpha=0.22)

# # ax[1].bar(
# #     x - width,
# #     wfe_rms_values,
# #     width,
# #     color=bar_colours,
# #     label="Full pupil",
# # )

# # ax[1].bar(
# #     x,
# #     centre_values,
# #     width,
# #     color=bar_colours,
# #     alpha=0.70,
# #     label=rf"Centre, $\rho<{CENTER_MAX_RHO}$",
# # )

# # ax[1].bar(
# #     x + width,
# #     edge_values,
# #     width,
# #     color=bar_colours,
# #     alpha=0.40,
# #     label=rf"Edge, $\rho\geq{EDGE_MIN_RHO}$",
# # )

# # ax[1].set_xticks(x, [rf"{v:.2f}$\,\lambda/D$" for v in mask_diam_lambdaD_array])
# # ax[1].set_ylabel("Equivalent WFE bias [nm RMS OPD]")
# # ax[1].set_title("Linear reconstruction bias")
# # ax[1].legend(frameon=True)
# # ax[1].grid(axis="y", alpha=0.22)

# # add_colourbar(fig, ax)
# # fig.savefig(OUTPUT_SUMMARY_FIGURE, bbox_inches="tight")


# # # ============================================================
# # # Save numerical products and print table
# # # ============================================================

# # np.savez_compressed(
# #     OUTPUT_DATA,
# #     mask_names=np.asarray(MASK_NAMES),
# #     mask_labels=np.asarray(mask_labels),
# #     mask_diam_um=mask_diam_um_array,
# #     mask_diam_lambdaD_wvl0=mask_diam_lambdaD_array,
# #     radial_edges=radial_edges,
# #     radial_centres=radial_centres,
# #     pupil_detector=pupil_detector,
# #     analysis_mask=analysis_mask,
# #     pupil_wave=pupil_wave,
# #     rho_detector=rho_detector,
# #     rho_wave=rho_wave,
# #     N0_internal=N0_internal,
# #     N0_onsky=N0_onsky,
# #     signal_bias_maps=signal_bias_maps,
# #     reconstructed_opd_maps_nm=reconstructed_opd_maps_nm,
# #     fitted_signal_maps=fitted_signal_maps,
# #     radial_signal_mean=radial_signal_mean,
# #     radial_signal_rms=radial_signal_rms,
# #     radial_wfe_mean_nm=radial_wfe_mean_nm,
# #     radial_wfe_rms_nm=radial_wfe_rms_nm,
# #     delta_T_values_K=DELTA_T_VALUES_K,
# #     T1_scan_K=np.array(T_INTERNAL_K),
# #     T2_scan_K=T_INTERNAL_K + DELTA_T_VALUES_K,
# #     peak_wfe_bias_vs_deltaT_nm=peak_wfe_bias_vs_deltaT_nm,
# # )

# # print()
# # print(
# #     "Mask  Diam[um]  Diam[lambda/D]  Peak|ds|    RMSE(ds)    "
# #     "WFE_RMS[nm]  Centre[nm]  Edge[nm]  Explained[%]"
# # )

# # for row in table_rows:
# #     print(
# #         f"{row['mask']:4s}  "
# #         f"{row['mask_diameter_um']:8.0f}  "
# #         f"{row['mask_diameter_lambda_over_D']:14.2f}  "
# #         f"{row['peak_abs_signal_bias']:9.3e}  "
# #         f"{row['signal_bias_rmse']:9.3e}  "
# #         f"{row['equivalent_wfe_rms_nm']:11.2f}  "
# #         f"{row['centre_wfe_rms_nm']:10.2f}  "
# #         f"{row['edge_wfe_rms_nm']:8.2f}  "
# #         f"{row['explained_signal_percent']:12.1f}"
# #     )

# # # Verify that the plotting section did not alter any analysis arrays.
# # analysis_arrays_after_plotting = {
# #     "radial_signal_mean": radial_signal_mean,
# #     "radial_signal_rms": radial_signal_rms,
# #     "radial_wfe_mean_nm": radial_wfe_mean_nm,
# #     "radial_wfe_rms_nm": radial_wfe_rms_nm,
# #     "peak_wfe_bias_vs_deltaT_nm": peak_wfe_bias_vs_deltaT_nm,
# # }

# # for name, before in analysis_arrays_before_plotting.items():
# #     after = analysis_arrays_after_plotting[name]

# #     if not np.array_equal(before, after, equal_nan=True):
# #         raise RuntimeError(
# #             f"Plotting unexpectedly modified numerical array: {name}"
# #         )

# # def file_sha256(path):
# #     path = Path(path).resolve()
# #     digest = hashlib.sha256()

# #     with open(path, "rb") as f:
# #         for block in iter(lambda: f.read(1024 * 1024), b""):
# #             digest.update(block)

# #     return digest.hexdigest()

# # provenance = {
# #     "python_executable": str(Path(sys.executable).resolve()),
# #     "python_version": platform.python_version(),
# #     "repo_root": str(REPO_ROOT),
# #     "imports": {
# #         "baldr_core": str(Path(bldr.__file__).resolve()),
# #         "spectrum": str(Path(spec.__file__).resolve()),
# #         "DM_basis": str(Path(DM_basis.__file__).resolve()),
# #     },
# #     "input_hashes_sha256": {
# #         "config": file_sha256(CONFIG_PATH),
# #         "phasemask_properties": file_sha256(phasemask_path),
# #         "baldr_core": file_sha256(bldr.__file__),
# #         "spectrum": file_sha256(spec.__file__),
# #         "DM_basis": file_sha256(DM_basis.__file__),
# #     },
# #     "analysis_fingerprints_sha256": analysis_fingerprints,
# #     "mask_order": MASK_NAMES,
# #     "mask_diameter_um": mask_diam_um_array.tolist(),
# #     "mask_diameter_lambda_over_D_at_wvl0": (
# #         mask_diam_lambdaD_array.tolist()
# #     ),
# #     "temperature_scan": {
# #         "T1_K": T_INTERNAL_K,
# #         "delta_T_values_K": DELTA_T_VALUES_K.tolist(),
# #         "T2_values_K": (
# #             T_INTERNAL_K + DELTA_T_VALUES_K
# #         ).tolist(),
# #         "I2M_rebuilt_for_every_T2_and_mask": True,
# #         "metric": "peak absolute reconstructed pupil OPD bias [nm]",
# #     },
# # }

# # with open(OUTPUT_PROVENANCE, "w") as f:
# #     json.dump(provenance, f, indent=2)

# # print()
# # for path in [
# #     OUTPUT_TABLE_CSV,
# #     OUTPUT_TABLE_TEX,
# #     OUTPUT_COMBINED_FIGURE,
# #     OUTPUT_SUMMARY_FIGURE,
# #     OUTPUT_SINGLE_MEAN_SIGNAL,
# #     OUTPUT_SINGLE_RMS_SIGNAL,
# #     OUTPUT_SINGLE_MEAN_WFE,
# #     OUTPUT_SINGLE_RMS_WFE,
# #     OUTPUT_DELTA_T_FIGURE,
# #     OUTPUT_DELTA_T_CSV,
# #     OUTPUT_DATA,
# #     OUTPUT_PROVENANCE,
# # ]:
# #     print(f"Saved: {path.resolve()}")

# # plt.show()


# # # #!/usr/bin/env python3
# # # """
# # # Paper-grade chromatic Baldr I0 reference-bias analysis for the physical H-band masks.

# # # This keeps the original analysis unchanged, but improves the plotting:
# # # - legend labels use physical mask size at wvl0 in lambda/D (and um);
# # # - traces are coloured by mask diameter with a continuous colormap;
# # # - the original 2x2 summary figure is retained;
# # # - each panel is also saved as a standalone figure.
# # # """

# # # import copy
# # # import csv
# # # import hashlib
# # # import json
# # # import platform
# # # import sys
# # # import tempfile
# # # from pathlib import Path

# # # import matplotlib.pyplot as plt
# # # import numpy as np
# # # from matplotlib.colors import Normalize
# # # from scipy.ndimage import binary_erosion

# # # REPO_ROOT = Path(__file__).resolve().parents[3]
# # # sys.path.insert(0, str(REPO_ROOT))

# # # from baldrapp.common import DM_basis
# # # from baldrapp.common import baldr_core as bldr
# # # from baldrapp.common import spectrum as spec


# # # # ============================================================
# # # # Strict local-import check
# # # # ============================================================

# # # for module in (bldr, spec, DM_basis):
# # #     module_path = Path(module.__file__).resolve()

# # #     if REPO_ROOT not in module_path.parents:
# # #         raise ImportError(
# # #             "BaldrApp import shadowing detected.\n"
# # #             f"Expected modules below: {REPO_ROOT}\n"
# # #             f"Imported instead: {module_path}\n"
# # #             "Run from the repository root, or use PYTHONPATH=$PWD."
# # #         )

# # # print(f"Python executable: {Path(sys.executable).resolve()}")
# # # print(f"Python version: {platform.python_version()}")
# # # print(f"baldr_core: {Path(bldr.__file__).resolve()}")
# # # print(f"spectrum: {Path(spec.__file__).resolve()}")
# # # print(f"DM_basis: {Path(DM_basis.__file__).resolve()}")


# # # # ============================================================
# # # # Settings
# # # # ============================================================

# # # CONFIG_PATH = (
# # #     REPO_ROOT
# # #     / "baldrapp/apps/paranal_simulator/fake_configs/baldr_config.json"
# # # )

# # # T_INTERNAL_K = 1900.0
# # # T_ONSKY_K = 10000.0

# # # MASK_NAMES = ["H1", "H2", "H3", "H4", "H5"]

# # # N_ZERNIKE_MODES = 20
# # # LINEAR_POKE_NM = 10.0
# # # SVD_RELATIVE_CUTOFF = 1e-3

# # # N_RADIAL_BINS = 10
# # # CENTER_MAX_RHO = 0.4
# # # EDGE_MIN_RHO = 0.8

# # # OUTPUT_TABLE_CSV = Path("chromatic_I0_bias_table.csv")
# # # OUTPUT_TABLE_TEX = Path("chromatic_I0_bias_table.tex")
# # # OUTPUT_DATA = Path("chromatic_I0_bias_results.npz")
# # # OUTPUT_PROVENANCE = Path("chromatic_I0_bias_provenance.json")

# # # OUTPUT_COMBINED_FIGURE = Path("chromatic_I0_bias_radial_profiles_paper.png")
# # # OUTPUT_SUMMARY_FIGURE = Path("chromatic_I0_bias_summary_paper.png")

# # # OUTPUT_SINGLE_MEAN_SIGNAL = Path("chromatic_I0_bias_signal_mean_radial.png")
# # # OUTPUT_SINGLE_RMS_SIGNAL = Path("chromatic_I0_bias_signal_rms_radial.png")
# # # OUTPUT_SINGLE_MEAN_WFE = Path("chromatic_I0_bias_wfe_mean_radial.png")
# # # OUTPUT_SINGLE_RMS_WFE = Path("chromatic_I0_bias_wfe_rms_radial.png")


# # # # ============================================================
# # # # Plot style
# # # # ============================================================

# # # plt.rcParams.update(
# # #     {
# # #         "figure.dpi": 140,
# # #         "savefig.dpi": 300,
# # #         "font.size": 11,
# # #         "axes.titlesize": 13,
# # #         "axes.labelsize": 12,
# # #         "legend.fontsize": 10,
# # #         "xtick.labelsize": 10,
# # #         "ytick.labelsize": 10,
# # #         "lines.linewidth": 2.2,
# # #         "lines.markersize": 6.0,
# # #         "axes.grid": True,
# # #         "grid.alpha": 0.22,
# # #         "grid.linewidth": 0.7,
# # #         "mathtext.default": "regular",
# # #     }
# # # )


# # # # ============================================================
# # # # Initialize identical aligned states with different spectra
# # # # ============================================================

# # # with open(CONFIG_PATH, "r") as f:
# # #     base_cfg = json.load(f)

# # # cfg_internal = copy.deepcopy(base_cfg)
# # # cfg_onsky = copy.deepcopy(base_cfg)

# # # for cfg, temperature_K in (
# # #     (cfg_internal, T_INTERNAL_K),
# # #     (cfg_onsky, T_ONSKY_K),
# # # ):
# # #     cfg["stellar"]["spectrum"]["enabled"] = True
# # #     cfg["stellar"]["spectrum"]["mode"] = "blackbody"
# # #     cfg["stellar"]["spectrum"]["temperature_K"] = temperature_K

# # #     cfg["fresnel_relay"]["coldstop_x_offset"] = 0.0
# # #     cfg["fresnel_relay"]["coldstop_y_offset"] = 0.0
# # #     cfg["fresnel_relay"]["pupil_misconjugation"] = 0.0
# # #     cfg["fresnel_relay"]["edge_offset"] = 0.0
# # #     cfg["fresnel_relay"]["edge_angle"] = 0.0
# # #     cfg["fresnel_relay"]["use_nominal_pupil_conjugation"] = True

# # #     cfg["internal_aberrations"]["enabled"] = False

# # #     cfg["detector"]["enabled"] = True
# # #     cfg["detector"]["ron"] = 0.0
# # #     cfg["detector"]["include_shotnoise"] = False
# # #     cfg["detector"]["include_readnoise"] = False
# # #     cfg["detector"]["adu_offset"] = 0.0
# # #     cfg["detector"]["noise_std_adu"] = 0.0

# # # with tempfile.TemporaryDirectory() as tmp:
# # #     tmp = Path(tmp)
# # #     internal_path = tmp / "internal_1900K.json"
# # #     onsky_path = tmp / "onsky_10000K.json"

# # #     with open(internal_path, "w") as f:
# # #         json.dump(cfg_internal, f, indent=2)

# # #     with open(onsky_path, "w") as f:
# # #         json.dump(cfg_onsky, f, indent=2)

# # #     zwfs_internal = bldr.init_zwfs_from_json(internal_path)
# # #     zwfs_onsky = bldr.init_zwfs_from_json(onsky_path)


# # # # ============================================================
# # # # Physical masks, pupil masks, and exact matched N0 references
# # # # ============================================================

# # # phasemask_path = (
# # #     REPO_ROOT
# # #     / base_cfg["simulator_runtime"]["phasemask"]["properties_file"]
# # # )

# # # with open(phasemask_path, "r") as f:
# # #     phasemask_cfg = json.load(f)

# # # mask_entries = phasemask_cfg["phasemask"]["masks"]

# # # amp_internal = zwfs_internal.grid.pupil_mask.astype(float)
# # # amp_onsky = zwfs_onsky.grid.pupil_mask.astype(float)

# # # zero_internal = np.zeros_like(amp_internal)
# # # zero_onsky = np.zeros_like(amp_onsky)

# # # N0_internal = bldr.get_N0_configured(
# # #     opd_input=zero_internal,
# # #     amp_input=amp_internal,
# # #     opd_internal=zero_internal,
# # #     zwfs_ns=zwfs_internal,
# # #     detector=zwfs_internal.detector,
# # #     include_shotnoise=False,
# # # )

# # # N0_onsky = bldr.get_N0_configured(
# # #     opd_input=zero_onsky,
# # #     amp_input=amp_onsky,
# # #     opd_internal=zero_onsky,
# # #     zwfs_ns=zwfs_onsky,
# # #     detector=zwfs_onsky.detector,
# # #     include_shotnoise=False,
# # # )

# # # binning = int(base_cfg["detector"]["binning"])

# # # pupil_detector = (
# # #     bldr.sum_subarrays(
# # #         zwfs_onsky.grid.pupil_mask,
# # #         block_size=(binning, binning),
# # #     )
# # #     > 0.5 * binning**2
# # # )

# # # analysis_mask = binary_erosion(pupil_detector, iterations=1)

# # # if not np.any(analysis_mask):
# # #     raise RuntimeError("The eroded detector pupil mask is empty.")

# # # N0_internal_mean = np.mean(N0_internal[analysis_mask])
# # # N0_onsky_mean = np.mean(N0_onsky[analysis_mask])

# # # yy_det, xx_det = np.indices(pupil_detector.shape)
# # # cy_det = np.mean(yy_det[pupil_detector])
# # # cx_det = np.mean(xx_det[pupil_detector])

# # # r_det = np.sqrt((xx_det - cx_det) ** 2 + (yy_det - cy_det) ** 2)
# # # outer_radius_det = np.percentile(r_det[pupil_detector], 99.5)
# # # rho_detector = r_det / outer_radius_det

# # # pupil_wave = zwfs_onsky.grid.pupil_mask.astype(bool)

# # # yy_wave, xx_wave = np.indices(pupil_wave.shape)
# # # cy_wave = np.mean(yy_wave[pupil_wave])
# # # cx_wave = np.mean(xx_wave[pupil_wave])

# # # r_wave = np.sqrt((xx_wave - cx_wave) ** 2 + (yy_wave - cy_wave) ** 2)
# # # outer_radius_wave = np.percentile(r_wave[pupil_wave], 99.5)
# # # rho_wave = r_wave / outer_radius_wave
# # # theta_wave = np.arctan2(yy_wave - cy_wave, xx_wave - cx_wave)

# # # radial_edges = np.linspace(0.0, 1.0, N_RADIAL_BINS + 1)
# # # radial_centres = 0.5 * (radial_edges[:-1] + radial_edges[1:])


# # # # ============================================================
# # # # Common phase basis: piston removed, each mode = 1 nm RMS OPD
# # # # ============================================================

# # # raw_zernikes = DM_basis.zernike_basis(
# # #     nterms=N_ZERNIKE_MODES + 1,
# # #     rho=rho_wave,
# # #     theta=theta_wave,
# # #     outside=0.0,
# # # )

# # # modes = []

# # # for mode_index in range(1, N_ZERNIKE_MODES + 1):
# # #     mode = np.nan_to_num(raw_zernikes[mode_index], nan=0.0)
# # #     mode *= pupil_wave
# # #     mode -= np.mean(mode[pupil_wave])

# # #     mode_rms = np.sqrt(np.mean(mode[pupil_wave] ** 2))

# # #     if mode_rms <= 0:
# # #         raise RuntimeError(f"Zernike mode {mode_index + 1} has zero RMS.")

# # #     modes.append(mode / mode_rms)

# # # modes = np.asarray(modes)


# # # # ============================================================
# # # # Per-mask signal bias and on-sky linear reconstruction bias
# # # # ============================================================

# # # table_rows = []

# # # signal_bias_maps = []
# # # reconstructed_opd_maps_nm = []
# # # fitted_signal_maps = []

# # # radial_signal_mean = np.full((len(MASK_NAMES), N_RADIAL_BINS), np.nan)
# # # radial_signal_rms = np.full((len(MASK_NAMES), N_RADIAL_BINS), np.nan)
# # # radial_wfe_mean_nm = np.full((len(MASK_NAMES), N_RADIAL_BINS), np.nan)
# # # radial_wfe_rms_nm = np.full((len(MASK_NAMES), N_RADIAL_BINS), np.nan)

# # # mask_diam_um_list = []
# # # mask_diam_lambdaD_list = []
# # # mask_labels = []

# # # for mask_number, mask_name in enumerate(MASK_NAMES):
# # #     print(f"Processing {mask_name} ({mask_number + 1}/{len(MASK_NAMES)})")

# # #     mask_entry = mask_entries[mask_name]

# # #     active_mask_internal = spec.normalise_phasemask_entry(
# # #         mask_name,
# # #         mask_entry,
# # #         zwfs_internal.optics,
# # #     )

# # #     active_mask_onsky = spec.normalise_phasemask_entry(
# # #         mask_name,
# # #         mask_entry,
# # #         zwfs_onsky.optics,
# # #     )

# # #     zwfs_internal.optics.active_phasemask = active_mask_internal
# # #     zwfs_onsky.optics.active_phasemask = active_mask_onsky

# # #     mask_diam_um = float(active_mask_onsky.mask_diam_um)
# # #     mask_diam_lambdaD = float(active_mask_onsky.mask_diam_lambdaD_wvl0)

# # #     mask_diam_um_list.append(mask_diam_um)
# # #     mask_diam_lambdaD_list.append(mask_diam_lambdaD)
# # #     mask_labels.append(
# # #         rf"{mask_diam_lambdaD:.2f}$\,\lambda/D$ ({mask_diam_um:.0f}$\,\mu$m)"
# # #     )

# # #     I0_internal = bldr.get_I0_configured(
# # #         opd_input=zero_internal,
# # #         amp_input=amp_internal,
# # #         opd_internal=zero_internal,
# # #         zwfs_ns=zwfs_internal,
# # #         detector=zwfs_internal.detector,
# # #         include_shotnoise=False,
# # #     )

# # #     I0_onsky = bldr.get_I0_configured(
# # #         opd_input=zero_onsky,
# # #         amp_input=amp_onsky,
# # #         opd_internal=zero_onsky,
# # #         zwfs_ns=zwfs_onsky,
# # #         detector=zwfs_onsky.detector,
# # #         include_shotnoise=False,
# # #     )

# # #     I0_internal_norm = I0_internal / N0_internal_mean
# # #     I0_onsky_norm = I0_onsky / N0_onsky_mean

# # #     signal_bias = I0_onsky_norm - I0_internal_norm
# # #     signal_vector = signal_bias[analysis_mask]

# # #     signal_peak_abs = np.max(np.abs(signal_vector))
# # #     signal_rmse = np.sqrt(np.mean(signal_vector**2))

# # #     response_columns = []

# # #     for mode in modes:
# # #         opd_plus = LINEAR_POKE_NM * 1e-9 * mode
# # #         opd_minus = -LINEAR_POKE_NM * 1e-9 * mode

# # #         I_plus = bldr.get_I0_configured(
# # #             opd_input=opd_plus,
# # #             amp_input=amp_onsky,
# # #             opd_internal=zero_onsky,
# # #             zwfs_ns=zwfs_onsky,
# # #             detector=zwfs_onsky.detector,
# # #             include_shotnoise=False,
# # #         )

# # #         I_minus = bldr.get_I0_configured(
# # #             opd_input=opd_minus,
# # #             amp_input=amp_onsky,
# # #             opd_internal=zero_onsky,
# # #             zwfs_ns=zwfs_onsky,
# # #             detector=zwfs_onsky.detector,
# # #             include_shotnoise=False,
# # #         )

# # #         I_plus_norm = I_plus / N0_onsky_mean
# # #         I_minus_norm = I_minus / N0_onsky_mean

# # #         derivative_per_nm = (
# # #             I_plus_norm - I_minus_norm
# # #         ) / (2.0 * LINEAR_POKE_NM)

# # #         response_columns.append(derivative_per_nm[analysis_mask])

# # #     interaction_matrix = np.column_stack(response_columns)

# # #     U, singular_values, Vt = np.linalg.svd(
# # #         interaction_matrix,
# # #         full_matrices=False,
# # #     )

# # #     keep = singular_values > (
# # #         SVD_RELATIVE_CUTOFF * singular_values[0]
# # #     )

# # #     if not np.any(keep):
# # #         raise RuntimeError(f"No singular values retained for {mask_name}.")

# # #     reconstructor = (
# # #         Vt[keep].T
# # #         @ np.diag(1.0 / singular_values[keep])
# # #         @ U[:, keep].T
# # #     )

# # #     coefficients_nm = reconstructor @ signal_vector

# # #     reconstructed_opd_nm = np.sum(
# # #         coefficients_nm[:, None, None] * modes,
# # #         axis=0,
# # #     )

# # #     fitted_signal_vector = interaction_matrix @ coefficients_nm
# # #     fitted_signal_map = np.full_like(signal_bias, np.nan, dtype=float)
# # #     fitted_signal_map[analysis_mask] = fitted_signal_vector

# # #     signal_power = np.sum(signal_vector**2)
# # #     residual_power = np.sum(
# # #         (signal_vector - fitted_signal_vector) ** 2
# # #     )

# # #     explained_fraction = (
# # #         1.0 - residual_power / signal_power
# # #         if signal_power > 0
# # #         else np.nan
# # #     )

# # #     wfe_values_nm = reconstructed_opd_nm[pupil_wave]
# # #     equivalent_wfe_rms_nm = np.sqrt(np.mean(wfe_values_nm**2))
# # #     peak_abs_wfe_nm = np.max(np.abs(wfe_values_nm))

# # #     centre_mask_wave = pupil_wave & (rho_wave < CENTER_MAX_RHO)
# # #     edge_mask_wave = (
# # #         pupil_wave
# # #         & (rho_wave >= EDGE_MIN_RHO)
# # #         & (rho_wave <= 1.0)
# # #     )

# # #     centre_wfe_rms_nm = np.sqrt(
# # #         np.mean(reconstructed_opd_nm[centre_mask_wave] ** 2)
# # #     )

# # #     edge_wfe_rms_nm = np.sqrt(
# # #         np.mean(reconstructed_opd_nm[edge_mask_wave] ** 2)
# # #     )

# # #     for radial_index in range(N_RADIAL_BINS):
# # #         detector_annulus = (
# # #             analysis_mask
# # #             & (rho_detector >= radial_edges[radial_index])
# # #             & (rho_detector < radial_edges[radial_index + 1])
# # #         )

# # #         wave_annulus = (
# # #             pupil_wave
# # #             & (rho_wave >= radial_edges[radial_index])
# # #             & (rho_wave < radial_edges[radial_index + 1])
# # #         )

# # #         if np.any(detector_annulus):
# # #             radial_values = signal_bias[detector_annulus]

# # #             radial_signal_mean[mask_number, radial_index] = np.mean(
# # #                 radial_values
# # #             )

# # #             radial_signal_rms[mask_number, radial_index] = np.sqrt(
# # #                 np.mean(radial_values**2)
# # #             )

# # #         if np.any(wave_annulus):
# # #             radial_values_nm = reconstructed_opd_nm[wave_annulus]

# # #             radial_wfe_mean_nm[mask_number, radial_index] = np.mean(
# # #                 radial_values_nm
# # #             )

# # #             radial_wfe_rms_nm[mask_number, radial_index] = np.sqrt(
# # #                 np.mean(radial_values_nm**2)
# # #             )

# # #     condition_number_retained = (
# # #         singular_values[keep][0] / singular_values[keep][-1]
# # #     )

# # #     table_rows.append(
# # #         {
# # #             "mask": mask_name,
# # #             "mask_diameter_um": float(mask_diam_um),
# # #             "mask_diameter_lambda_over_D": float(mask_diam_lambdaD),
# # #             "peak_abs_signal_bias": float(signal_peak_abs),
# # #             "signal_bias_rmse": float(signal_rmse),
# # #             "equivalent_wfe_rms_nm": float(equivalent_wfe_rms_nm),
# # #             "peak_abs_wfe_nm": float(peak_abs_wfe_nm),
# # #             "centre_wfe_rms_nm": float(centre_wfe_rms_nm),
# # #             "edge_wfe_rms_nm": float(edge_wfe_rms_nm),
# # #             "explained_signal_percent": float(
# # #                 100.0 * explained_fraction
# # #             ),
# # #             "retained_rank": int(np.sum(keep)),
# # #             "retained_condition_number": float(
# # #                 condition_number_retained
# # #             ),
# # #         }
# # #     )

# # #     signal_bias_maps.append(signal_bias)
# # #     reconstructed_opd_maps_nm.append(reconstructed_opd_nm)
# # #     fitted_signal_maps.append(fitted_signal_map)

# # # signal_bias_maps = np.asarray(signal_bias_maps)
# # # reconstructed_opd_maps_nm = np.asarray(reconstructed_opd_maps_nm)
# # # fitted_signal_maps = np.asarray(fitted_signal_maps)

# # # mask_diam_um_array = np.asarray(mask_diam_um_list)
# # # mask_diam_lambdaD_array = np.asarray(mask_diam_lambdaD_list)

# # # # Preserve the exact H1--H5 calculation order. The physical diameters are
# # # # already monotonic, so no numerical arrays need to be reordered for plotting.
# # # if not np.all(np.diff(mask_diam_lambdaD_array) > 0):
# # #     raise RuntimeError(
# # #         "MASK_NAMES are not ordered by increasing physical mask diameter."
# # #     )

# # # cmap = plt.get_cmap("viridis")
# # # norm = Normalize(
# # #     vmin=float(np.min(mask_diam_lambdaD_array)),
# # #     vmax=float(np.max(mask_diam_lambdaD_array)),
# # # )
# # # trace_colours = [cmap(norm(v)) for v in mask_diam_lambdaD_array]

# # # # Immutable snapshots and numerical fingerprints before any plotting.
# # # analysis_arrays_before_plotting = {
# # #     "radial_signal_mean": radial_signal_mean.copy(),
# # #     "radial_signal_rms": radial_signal_rms.copy(),
# # #     "radial_wfe_mean_nm": radial_wfe_mean_nm.copy(),
# # #     "radial_wfe_rms_nm": radial_wfe_rms_nm.copy(),
# # # }

# # # analysis_fingerprints = {
# # #     name: hashlib.sha256(
# # #         np.ascontiguousarray(values).view(np.uint8)
# # #     ).hexdigest()
# # #     for name, values in analysis_arrays_before_plotting.items()
# # # }

# # # print("Numerical result fingerprints:")
# # # for name, digest in analysis_fingerprints.items():
# # #     print(f"  {name}: {digest}")


# # # # ============================================================
# # # # CSV and LaTeX tables
# # # # ============================================================

# # # fieldnames = list(table_rows[0].keys())

# # # with open(OUTPUT_TABLE_CSV, "w", newline="") as f:
# # #     writer = csv.DictWriter(f, fieldnames=fieldnames)
# # #     writer.writeheader()
# # #     writer.writerows(table_rows)

# # # with open(OUTPUT_TABLE_TEX, "w") as f:
# # #     f.write("\\begin{tabular}{lrrrrrrrr}\n")
# # #     f.write("\\hline\n")
# # #     f.write(
# # #         "Mask & Diam. & Diam. & Peak $|\\Delta s|$ & "
# # #         "RMSE $\\Delta s$ & WFE RMS & Peak $|\\phi|$ & "
# # #         "Centre RMS & Edge RMS \\\\\n"
# # #     )
# # #     f.write(
# # #         " & [$\\mu$m] & [$\\lambda/D$] & & & [nm] & [nm] & [nm] & [nm] \\\\\n"
# # #     )
# # #     f.write("\\hline\n")

# # #     for row in table_rows:
# # #         f.write(
# # #             f"{row['mask']} & "
# # #             f"{row['mask_diameter_um']:.0f} & "
# # #             f"{row['mask_diameter_lambda_over_D']:.2f} & "
# # #             f"{row['peak_abs_signal_bias']:.3e} & "
# # #             f"{row['signal_bias_rmse']:.3e} & "
# # #             f"{row['equivalent_wfe_rms_nm']:.2f} & "
# # #             f"{row['peak_abs_wfe_nm']:.2f} & "
# # #             f"{row['centre_wfe_rms_nm']:.2f} & "
# # #             f"{row['edge_wfe_rms_nm']:.2f} \\\\\n"
# # #         )

# # #     f.write("\\hline\n")
# # #     f.write("\\end{tabular}\n")


# # # # ============================================================
# # # # Plot helpers
# # # # ============================================================

# # # def decorate_axis(ax, title, ylabel, yzero=False, xlab=False):
# # #     ax.set_title(title, pad=8)
# # #     ax.set_ylabel(ylabel)
# # #     if xlab:
# # #         ax.set_xlabel("Normalized pupil radius")
# # #     if yzero:
# # #         ax.axhline(0.0, color="0.25", linewidth=1.0, zorder=0)
# # #     ax.axvline(
# # #         EDGE_MIN_RHO,
# # #         linestyle=":",
# # #         linewidth=1.2,
# # #         color="0.4",
# # #         zorder=0,
# # #     )
# # #     ax.set_xlim(radial_edges[0], radial_edges[-1])


# # # def add_colourbar(fig, axs):
# # #     sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
# # #     sm.set_array([])
# # #     cbar = fig.colorbar(
# # #         sm,
# # #         ax=axs,
# # #         shrink=0.94,
# # #         pad=0.02,
# # #     )
# # #     cbar.set_label(r"Mask diameter at $wvl_0$ [$\lambda/D$]")
# # #     return cbar


# # # def draw_panel(ax, ydata, yzero, ylabel, title, show_legend=False):
# # #     for i, label in enumerate(mask_labels):
# # #         ax.plot(
# # #             radial_centres,
# # #             ydata[i],
# # #             marker="o",
# # #             color=trace_colours[i],
# # #             label=label,
# # #         )
# # #     decorate_axis(ax, title=title, ylabel=ylabel, yzero=yzero, xlab=True)
# # #     if show_legend:
# # #         ax.legend(frameon=True, ncol=1, loc="best")


# # # def save_standalone_figure(filename, ydata, yzero, ylabel, title):
# # #     fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
# # #     for i, label in enumerate(mask_labels):
# # #         ax.plot(
# # #             radial_centres,
# # #             ydata[i],
# # #             marker="o",
# # #             color=trace_colours[i],
# # #             label=label,
# # #         )
# # #     decorate_axis(ax, title=title, ylabel=ylabel, yzero=yzero, xlab=True)
# # #     ax.legend(frameon=True, loc="best", title="Phase mask")
# # #     add_colourbar(fig, ax)
# # #     fig.savefig(filename, bbox_inches="tight")
# # #     plt.close(fig)


# # # # ============================================================
# # # # Combined radial profile figure
# # # # ============================================================

# # # fig, ax = plt.subplots(2, 2, figsize=(12.6, 9.2), constrained_layout=True)

# # # for i, label in enumerate(mask_labels):
# # #     ax[0, 0].plot(
# # #         radial_centres,
# # #         radial_signal_mean[i],
# # #         marker="o",
# # #         color=trace_colours[i],
# # #         label=label,
# # #     )
# # #     ax[0, 1].plot(
# # #         radial_centres,
# # #         radial_signal_rms[i],
# # #         marker="o",
# # #         color=trace_colours[i],
# # #         label=label,
# # #     )
# # #     ax[1, 0].plot(
# # #         radial_centres,
# # #         radial_wfe_mean_nm[i],
# # #         marker="o",
# # #         color=trace_colours[i],
# # #         label=label,
# # #     )
# # #     ax[1, 1].plot(
# # #         radial_centres,
# # #         radial_wfe_rms_nm[i],
# # #         marker="o",
# # #         color=trace_colours[i],
# # #         label=label,
# # #     )

# # # decorate_axis(
# # #     ax[0, 0],
# # #     title="Signed radial signal bias",
# # #     ylabel="Mean normalized-signal bias",
# # #     yzero=True,
# # #     xlab=False,
# # # )
# # # decorate_axis(
# # #     ax[0, 1],
# # #     title="Radial signal-bias magnitude",
# # #     ylabel="RMS normalized-signal bias",
# # #     yzero=False,
# # #     xlab=False,
# # # )
# # # decorate_axis(
# # #     ax[1, 0],
# # #     title="Signed radial reconstruction bias",
# # #     ylabel="Mean reconstructed OPD bias [nm]",
# # #     yzero=True,
# # #     xlab=True,
# # # )
# # # decorate_axis(
# # #     ax[1, 1],
# # #     title="Radial reconstruction-bias magnitude",
# # #     ylabel="RMS reconstructed OPD bias [nm]",
# # #     yzero=False,
# # #     xlab=True,
# # # )

# # # legend = ax[0, 0].legend(
# # #     title="Phase mask",
# # #     frameon=True,
# # #     ncol=1,
# # #     loc="best",
# # # )

# # # fig.suptitle(
# # #     rf"Chromatic $I_0$ bias: {T_INTERNAL_K:.0f} K internal reference used on "
# # #     rf"{T_ONSKY_K:.0f} K source",
# # #     y=1.01,
# # #     fontsize=15,
# # # )

# # # add_colourbar(fig, ax)
# # # fig.savefig(OUTPUT_COMBINED_FIGURE, bbox_inches="tight")


# # # # ============================================================
# # # # Standalone radial profile figures
# # # # ============================================================

# # # save_standalone_figure(
# # #     OUTPUT_SINGLE_MEAN_SIGNAL,
# # #     radial_signal_mean,
# # #     True,
# # #     "Mean normalized-signal bias",
# # #     "Signed radial signal bias",
# # # )

# # # save_standalone_figure(
# # #     OUTPUT_SINGLE_RMS_SIGNAL,
# # #     radial_signal_rms,
# # #     False,
# # #     "RMS normalized-signal bias",
# # #     "Radial signal-bias magnitude",
# # # )

# # # save_standalone_figure(
# # #     OUTPUT_SINGLE_MEAN_WFE,
# # #     radial_wfe_mean_nm,
# # #     True,
# # #     "Mean reconstructed OPD bias [nm]",
# # #     "Signed radial reconstruction bias",
# # # )

# # # save_standalone_figure(
# # #     OUTPUT_SINGLE_RMS_WFE,
# # #     radial_wfe_rms_nm,
# # #     False,
# # #     "RMS reconstructed OPD bias [nm]",
# # #     "Radial reconstruction-bias magnitude",
# # # )


# # # # ============================================================
# # # # Global summary figure
# # # # ============================================================

# # # signal_peak_values = [
# # #     row["peak_abs_signal_bias"] for row in table_rows
# # # ]
# # # signal_rmse_values = [
# # #     row["signal_bias_rmse"] for row in table_rows
# # # ]
# # # wfe_rms_values = [
# # #     row["equivalent_wfe_rms_nm"] for row in table_rows
# # # ]
# # # centre_values = [
# # #     row["centre_wfe_rms_nm"] for row in table_rows
# # # ]
# # # edge_values = [
# # #     row["edge_wfe_rms_nm"] for row in table_rows
# # # ]

# # # x = np.arange(len(mask_labels))
# # # width = 0.36

# # # fig, ax = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)

# # # bar_colours = trace_colours

# # # ax[0].bar(
# # #     x - width / 2,
# # #     signal_peak_values,
# # #     width,
# # #     color=bar_colours,
# # #     label=r"Peak $|\Delta s|$",
# # # )

# # # ax[0].bar(
# # #     x + width / 2,
# # #     signal_rmse_values,
# # #     width,
# # #     color=bar_colours,
# # #     alpha=0.45,
# # #     label=r"RMSE $(\Delta s)$",
# # # )

# # # ax[0].set_xticks(x, [rf"{v:.2f}$\,\lambda/D$" for v in mask_diam_lambdaD_array])
# # # ax[0].set_ylabel("Normalized-signal bias")
# # # ax[0].set_title("Reference signal bias")
# # # ax[0].legend(frameon=True)
# # # ax[0].grid(axis="y", alpha=0.22)

# # # ax[1].bar(
# # #     x - width,
# # #     wfe_rms_values,
# # #     width,
# # #     color=bar_colours,
# # #     label="Full pupil",
# # # )

# # # ax[1].bar(
# # #     x,
# # #     centre_values,
# # #     width,
# # #     color=bar_colours,
# # #     alpha=0.70,
# # #     label=rf"Centre, $\rho<{CENTER_MAX_RHO}$",
# # # )

# # # ax[1].bar(
# # #     x + width,
# # #     edge_values,
# # #     width,
# # #     color=bar_colours,
# # #     alpha=0.40,
# # #     label=rf"Edge, $\rho\geq{EDGE_MIN_RHO}$",
# # # )

# # # ax[1].set_xticks(x, [rf"{v:.2f}$\,\lambda/D$" for v in mask_diam_lambdaD_array])
# # # ax[1].set_ylabel("Equivalent WFE bias [nm RMS OPD]")
# # # ax[1].set_title("Linear reconstruction bias")
# # # ax[1].legend(frameon=True)
# # # ax[1].grid(axis="y", alpha=0.22)

# # # add_colourbar(fig, ax)
# # # fig.savefig(OUTPUT_SUMMARY_FIGURE, bbox_inches="tight")


# # # # ============================================================
# # # # Save numerical products and print table
# # # # ============================================================

# # # np.savez_compressed(
# # #     OUTPUT_DATA,
# # #     mask_names=np.asarray(MASK_NAMES),
# # #     mask_labels=np.asarray(mask_labels),
# # #     mask_diam_um=mask_diam_um_array,
# # #     mask_diam_lambdaD_wvl0=mask_diam_lambdaD_array,
# # #     radial_edges=radial_edges,
# # #     radial_centres=radial_centres,
# # #     pupil_detector=pupil_detector,
# # #     analysis_mask=analysis_mask,
# # #     pupil_wave=pupil_wave,
# # #     rho_detector=rho_detector,
# # #     rho_wave=rho_wave,
# # #     N0_internal=N0_internal,
# # #     N0_onsky=N0_onsky,
# # #     signal_bias_maps=signal_bias_maps,
# # #     reconstructed_opd_maps_nm=reconstructed_opd_maps_nm,
# # #     fitted_signal_maps=fitted_signal_maps,
# # #     radial_signal_mean=radial_signal_mean,
# # #     radial_signal_rms=radial_signal_rms,
# # #     radial_wfe_mean_nm=radial_wfe_mean_nm,
# # #     radial_wfe_rms_nm=radial_wfe_rms_nm,
# # # )

# # # print()
# # # print(
# # #     "Mask  Diam[um]  Diam[lambda/D]  Peak|ds|    RMSE(ds)    "
# # #     "WFE_RMS[nm]  Centre[nm]  Edge[nm]  Explained[%]"
# # # )

# # # for row in table_rows:
# # #     print(
# # #         f"{row['mask']:4s}  "
# # #         f"{row['mask_diameter_um']:8.0f}  "
# # #         f"{row['mask_diameter_lambda_over_D']:14.2f}  "
# # #         f"{row['peak_abs_signal_bias']:9.3e}  "
# # #         f"{row['signal_bias_rmse']:9.3e}  "
# # #         f"{row['equivalent_wfe_rms_nm']:11.2f}  "
# # #         f"{row['centre_wfe_rms_nm']:10.2f}  "
# # #         f"{row['edge_wfe_rms_nm']:8.2f}  "
# # #         f"{row['explained_signal_percent']:12.1f}"
# # #     )

# # # # Verify that the plotting section did not alter any analysis arrays.
# # # analysis_arrays_after_plotting = {
# # #     "radial_signal_mean": radial_signal_mean,
# # #     "radial_signal_rms": radial_signal_rms,
# # #     "radial_wfe_mean_nm": radial_wfe_mean_nm,
# # #     "radial_wfe_rms_nm": radial_wfe_rms_nm,
# # # }

# # # for name, before in analysis_arrays_before_plotting.items():
# # #     after = analysis_arrays_after_plotting[name]

# # #     if not np.array_equal(before, after, equal_nan=True):
# # #         raise RuntimeError(
# # #             f"Plotting unexpectedly modified numerical array: {name}"
# # #         )

# # # def file_sha256(path):
# # #     path = Path(path).resolve()
# # #     digest = hashlib.sha256()

# # #     with open(path, "rb") as f:
# # #         for block in iter(lambda: f.read(1024 * 1024), b""):
# # #             digest.update(block)

# # #     return digest.hexdigest()

# # # provenance = {
# # #     "python_executable": str(Path(sys.executable).resolve()),
# # #     "python_version": platform.python_version(),
# # #     "repo_root": str(REPO_ROOT),
# # #     "imports": {
# # #         "baldr_core": str(Path(bldr.__file__).resolve()),
# # #         "spectrum": str(Path(spec.__file__).resolve()),
# # #         "DM_basis": str(Path(DM_basis.__file__).resolve()),
# # #     },
# # #     "input_hashes_sha256": {
# # #         "config": file_sha256(CONFIG_PATH),
# # #         "phasemask_properties": file_sha256(phasemask_path),
# # #         "baldr_core": file_sha256(bldr.__file__),
# # #         "spectrum": file_sha256(spec.__file__),
# # #         "DM_basis": file_sha256(DM_basis.__file__),
# # #     },
# # #     "analysis_fingerprints_sha256": analysis_fingerprints,
# # #     "mask_order": MASK_NAMES,
# # #     "mask_diameter_um": mask_diam_um_array.tolist(),
# # #     "mask_diameter_lambda_over_D_at_wvl0": (
# # #         mask_diam_lambdaD_array.tolist()
# # #     ),
# # # }

# # # with open(OUTPUT_PROVENANCE, "w") as f:
# # #     json.dump(provenance, f, indent=2)

# # # print()
# # # for path in [
# # #     OUTPUT_TABLE_CSV,
# # #     OUTPUT_TABLE_TEX,
# # #     OUTPUT_COMBINED_FIGURE,
# # #     OUTPUT_SUMMARY_FIGURE,
# # #     OUTPUT_SINGLE_MEAN_SIGNAL,
# # #     OUTPUT_SINGLE_RMS_SIGNAL,
# # #     OUTPUT_SINGLE_MEAN_WFE,
# # #     OUTPUT_SINGLE_RMS_WFE,
# # #     OUTPUT_DATA,
# # #     OUTPUT_PROVENANCE,
# # # ]:
# # #     print(f"Saved: {path.resolve()}")

# # # plt.show()


# # # # #!/usr/bin/env python3
# # # # """
# # # # Chromatic Baldr I0 reference bias for the physical H-band phase masks.

# # # # Idealized experiment:
# # # # - identical internal/on-sky pupil and optics;
# # # # - all explicit alignment offsets set to zero;
# # # # - 1900 K internal blackbody and 10000 K on-sky blackbody;
# # # # - perfect, noise-free matched N0 for each source;
# # # # - true on-sky wavefront is zero;
# # # # - the internal-source I0 is used as the incorrect on-sky reference.

# # # # Outputs:
# # # # - chromatic_I0_bias_table.csv
# # # # - chromatic_I0_bias_table.tex
# # # # - chromatic_I0_bias_radial_profiles.png
# # # # - chromatic_I0_bias_summary.png
# # # # - chromatic_I0_bias_results.npz
# # # # """

# # # # import copy
# # # # import csv
# # # # import json
# # # # import sys
# # # # import tempfile
# # # # from pathlib import Path

# # # # import matplotlib.pyplot as plt
# # # # import numpy as np
# # # # from scipy.ndimage import binary_erosion

# # # # REPO_ROOT = Path(__file__).resolve().parents[3]
# # # # sys.path.insert(0, str(REPO_ROOT))

# # # # from baldrapp.common import DM_basis
# # # # from baldrapp.common import baldr_core as bldr
# # # # from baldrapp.common import spectrum as spec


# # # # # ============================================================
# # # # # Settings
# # # # # ============================================================

# # # # CONFIG_PATH = (
# # # #     REPO_ROOT
# # # #     / "baldrapp/apps/paranal_simulator/fake_configs/baldr_config.json"
# # # # )

# # # # T_INTERNAL_K = 1900.0
# # # # T_ONSKY_K = 10000.0

# # # # MASK_NAMES = ["H1", "H2", "H3", "H4", "H5"]

# # # # N_ZERNIKE_MODES = 20
# # # # LINEAR_POKE_NM = 10.0
# # # # SVD_RELATIVE_CUTOFF = 1e-3

# # # # N_RADIAL_BINS = 10
# # # # CENTER_MAX_RHO = 0.4
# # # # EDGE_MIN_RHO = 0.8

# # # # OUTPUT_TABLE_CSV = Path("chromatic_I0_bias_table.csv")
# # # # OUTPUT_TABLE_TEX = Path("chromatic_I0_bias_table.tex")
# # # # OUTPUT_RADIAL_FIGURE = Path("chromatic_I0_bias_radial_profiles.png")
# # # # OUTPUT_SUMMARY_FIGURE = Path("chromatic_I0_bias_summary.png")
# # # # OUTPUT_DATA = Path("chromatic_I0_bias_results.npz")


# # # # # ============================================================
# # # # # Initialize identical aligned states with different spectra
# # # # # ============================================================

# # # # with open(CONFIG_PATH, "r") as f:
# # # #     base_cfg = json.load(f)

# # # # cfg_internal = copy.deepcopy(base_cfg)
# # # # cfg_onsky = copy.deepcopy(base_cfg)

# # # # for cfg, temperature_K in (
# # # #     (cfg_internal, T_INTERNAL_K),
# # # #     (cfg_onsky, T_ONSKY_K),
# # # # ):
# # # #     cfg["stellar"]["spectrum"]["enabled"] = True
# # # #     cfg["stellar"]["spectrum"]["mode"] = "blackbody"
# # # #     cfg["stellar"]["spectrum"]["temperature_K"] = temperature_K

# # # #     cfg["fresnel_relay"]["coldstop_x_offset"] = 0.0
# # # #     cfg["fresnel_relay"]["coldstop_y_offset"] = 0.0
# # # #     cfg["fresnel_relay"]["pupil_misconjugation"] = 0.0
# # # #     cfg["fresnel_relay"]["edge_offset"] = -20.0
# # # #     cfg["fresnel_relay"]["edge_angle"] = 0.0
# # # #     cfg["fresnel_relay"]["use_nominal_pupil_conjugation"] = True

# # # #     cfg["internal_aberrations"]["enabled"] = False

# # # #     cfg["detector"]["enabled"] = True
# # # #     cfg["detector"]["ron"] = 0.0
# # # #     cfg["detector"]["include_shotnoise"] = False
# # # #     cfg["detector"]["include_readnoise"] = False
# # # #     cfg["detector"]["adu_offset"] = 0.0
# # # #     cfg["detector"]["noise_std_adu"] = 0.0

# # # # with tempfile.TemporaryDirectory() as tmp:
# # # #     tmp = Path(tmp)
# # # #     internal_path = tmp / "internal_1900K.json"
# # # #     onsky_path = tmp / "onsky_10000K.json"

# # # #     with open(internal_path, "w") as f:
# # # #         json.dump(cfg_internal, f, indent=2)

# # # #     with open(onsky_path, "w") as f:
# # # #         json.dump(cfg_onsky, f, indent=2)

# # # #     zwfs_internal = bldr.init_zwfs_from_json(internal_path)
# # # #     zwfs_onsky = bldr.init_zwfs_from_json(onsky_path)


# # # # # ============================================================
# # # # # Physical masks, pupil masks, and exact matched N0 references
# # # # # ============================================================

# # # # phasemask_path = (
# # # #     REPO_ROOT
# # # #     / base_cfg["simulator_runtime"]["phasemask"]["properties_file"]
# # # # )

# # # # with open(phasemask_path, "r") as f:
# # # #     phasemask_cfg = json.load(f)

# # # # mask_entries = phasemask_cfg["phasemask"]["masks"]

# # # # amp_internal = zwfs_internal.grid.pupil_mask.astype(float)
# # # # amp_onsky = zwfs_onsky.grid.pupil_mask.astype(float)

# # # # zero_internal = np.zeros_like(amp_internal)
# # # # zero_onsky = np.zeros_like(amp_onsky)

# # # # N0_internal = bldr.get_N0_configured(
# # # #     opd_input=zero_internal,
# # # #     amp_input=amp_internal,
# # # #     opd_internal=zero_internal,
# # # #     zwfs_ns=zwfs_internal,
# # # #     detector=zwfs_internal.detector,
# # # #     include_shotnoise=False,
# # # # )

# # # # N0_onsky = bldr.get_N0_configured(
# # # #     opd_input=zero_onsky,
# # # #     amp_input=amp_onsky,
# # # #     opd_internal=zero_onsky,
# # # #     zwfs_ns=zwfs_onsky,
# # # #     detector=zwfs_onsky.detector,
# # # #     include_shotnoise=False,
# # # # )

# # # # binning = int(base_cfg["detector"]["binning"])

# # # # pupil_detector = (
# # # #     bldr.sum_subarrays(
# # # #         zwfs_onsky.grid.pupil_mask,
# # # #         block_size=(binning, binning),
# # # #     )
# # # #     > 0.5 * binning**2
# # # # )

# # # # analysis_mask = binary_erosion(pupil_detector, iterations=1)

# # # # if not np.any(analysis_mask):
# # # #     raise RuntimeError("The eroded detector pupil mask is empty.")

# # # # N0_internal_mean = np.mean(N0_internal[analysis_mask])
# # # # N0_onsky_mean = np.mean(N0_onsky[analysis_mask])

# # # # yy_det, xx_det = np.indices(pupil_detector.shape)
# # # # cy_det = np.mean(yy_det[pupil_detector])
# # # # cx_det = np.mean(xx_det[pupil_detector])

# # # # r_det = np.sqrt((xx_det - cx_det) ** 2 + (yy_det - cy_det) ** 2)
# # # # outer_radius_det = np.percentile(r_det[pupil_detector], 99.5)
# # # # rho_detector = r_det / outer_radius_det

# # # # pupil_wave = zwfs_onsky.grid.pupil_mask.astype(bool)

# # # # yy_wave, xx_wave = np.indices(pupil_wave.shape)
# # # # cy_wave = np.mean(yy_wave[pupil_wave])
# # # # cx_wave = np.mean(xx_wave[pupil_wave])

# # # # r_wave = np.sqrt((xx_wave - cx_wave) ** 2 + (yy_wave - cy_wave) ** 2)
# # # # outer_radius_wave = np.percentile(r_wave[pupil_wave], 99.5)
# # # # rho_wave = r_wave / outer_radius_wave
# # # # theta_wave = np.arctan2(yy_wave - cy_wave, xx_wave - cx_wave)

# # # # radial_edges = np.linspace(0.0, 1.0, N_RADIAL_BINS + 1)
# # # # radial_centres = 0.5 * (radial_edges[:-1] + radial_edges[1:])


# # # # # ============================================================
# # # # # Common phase basis: piston removed, each mode = 1 nm RMS OPD
# # # # # ============================================================

# # # # raw_zernikes = DM_basis.zernike_basis(
# # # #     nterms=N_ZERNIKE_MODES + 1,
# # # #     rho=rho_wave,
# # # #     theta=theta_wave,
# # # #     outside=0.0,
# # # # )

# # # # modes = []

# # # # for mode_index in range(1, N_ZERNIKE_MODES + 1):
# # # #     mode = np.nan_to_num(raw_zernikes[mode_index], nan=0.0)
# # # #     mode *= pupil_wave
# # # #     mode -= np.mean(mode[pupil_wave])

# # # #     mode_rms = np.sqrt(np.mean(mode[pupil_wave] ** 2))

# # # #     if mode_rms <= 0:
# # # #         raise RuntimeError(f"Zernike mode {mode_index + 1} has zero RMS.")

# # # #     modes.append(mode / mode_rms)

# # # # modes = np.asarray(modes)


# # # # # ============================================================
# # # # # Per-mask signal bias and on-sky linear reconstruction bias
# # # # # ============================================================

# # # # table_rows = []

# # # # signal_bias_maps = []
# # # # reconstructed_opd_maps_nm = []
# # # # fitted_signal_maps = []

# # # # radial_signal_mean = np.full((len(MASK_NAMES), N_RADIAL_BINS), np.nan)
# # # # radial_signal_rms = np.full((len(MASK_NAMES), N_RADIAL_BINS), np.nan)
# # # # radial_wfe_mean_nm = np.full((len(MASK_NAMES), N_RADIAL_BINS), np.nan)
# # # # radial_wfe_rms_nm = np.full((len(MASK_NAMES), N_RADIAL_BINS), np.nan)

# # # # for mask_number, mask_name in enumerate(MASK_NAMES):
# # # #     print(f"Processing {mask_name} ({mask_number + 1}/{len(MASK_NAMES)})")

# # # #     mask_entry = mask_entries[mask_name]

# # # #     zwfs_internal.optics.active_phasemask = spec.normalise_phasemask_entry(
# # # #         mask_name,
# # # #         mask_entry,
# # # #         zwfs_internal.optics,
# # # #     )

# # # #     zwfs_onsky.optics.active_phasemask = spec.normalise_phasemask_entry(
# # # #         mask_name,
# # # #         mask_entry,
# # # #         zwfs_onsky.optics,
# # # #     )

# # # #     I0_internal = bldr.get_I0_configured(
# # # #         opd_input=zero_internal,
# # # #         amp_input=amp_internal,
# # # #         opd_internal=zero_internal,
# # # #         zwfs_ns=zwfs_internal,
# # # #         detector=zwfs_internal.detector,
# # # #         include_shotnoise=False,
# # # #     )

# # # #     I0_onsky = bldr.get_I0_configured(
# # # #         opd_input=zero_onsky,
# # # #         amp_input=amp_onsky,
# # # #         opd_internal=zero_onsky,
# # # #         zwfs_ns=zwfs_onsky,
# # # #         detector=zwfs_onsky.detector,
# # # #         include_shotnoise=False,
# # # #     )

# # # #     I0_internal_norm = I0_internal / N0_internal_mean
# # # #     I0_onsky_norm = I0_onsky / N0_onsky_mean

# # # #     # Perfect zero-aberration on-sky signal, reduced using the internal I0.
# # # #     signal_bias = I0_onsky_norm - I0_internal_norm
# # # #     signal_vector = signal_bias[analysis_mask]

# # # #     signal_peak_abs = np.max(np.abs(signal_vector))
# # # #     signal_rmse = np.sqrt(np.mean(signal_vector**2))

# # # #     # On-sky interaction matrix for this mask and source spectrum.
# # # #     response_columns = []

# # # #     for mode in modes:
# # # #         opd_plus = LINEAR_POKE_NM * 1e-9 * mode
# # # #         opd_minus = -LINEAR_POKE_NM * 1e-9 * mode

# # # #         I_plus = bldr.get_I0_configured(
# # # #             opd_input=opd_plus,
# # # #             amp_input=amp_onsky,
# # # #             opd_internal=zero_onsky,
# # # #             zwfs_ns=zwfs_onsky,
# # # #             detector=zwfs_onsky.detector,
# # # #             include_shotnoise=False,
# # # #         )

# # # #         I_minus = bldr.get_I0_configured(
# # # #             opd_input=opd_minus,
# # # #             amp_input=amp_onsky,
# # # #             opd_internal=zero_onsky,
# # # #             zwfs_ns=zwfs_onsky,
# # # #             detector=zwfs_onsky.detector,
# # # #             include_shotnoise=False,
# # # #         )

# # # #         I_plus_norm = I_plus / N0_onsky_mean
# # # #         I_minus_norm = I_minus / N0_onsky_mean

# # # #         derivative_per_nm = (
# # # #             I_plus_norm - I_minus_norm
# # # #         ) / (2.0 * LINEAR_POKE_NM)

# # # #         response_columns.append(derivative_per_nm[analysis_mask])

# # # #     interaction_matrix = np.column_stack(response_columns)

# # # #     U, singular_values, Vt = np.linalg.svd(
# # # #         interaction_matrix,
# # # #         full_matrices=False,
# # # #     )

# # # #     keep = singular_values > (
# # # #         SVD_RELATIVE_CUTOFF * singular_values[0]
# # # #     )

# # # #     if not np.any(keep):
# # # #         raise RuntimeError(f"No singular values retained for {mask_name}.")

# # # #     reconstructor = (
# # # #         Vt[keep].T
# # # #         @ np.diag(1.0 / singular_values[keep])
# # # #         @ U[:, keep].T
# # # #     )

# # # #     coefficients_nm = reconstructor @ signal_vector

# # # #     reconstructed_opd_nm = np.sum(
# # # #         coefficients_nm[:, None, None] * modes,
# # # #         axis=0,
# # # #     )

# # # #     fitted_signal_vector = interaction_matrix @ coefficients_nm
# # # #     fitted_signal_map = np.full_like(signal_bias, np.nan, dtype=float)
# # # #     fitted_signal_map[analysis_mask] = fitted_signal_vector

# # # #     signal_power = np.sum(signal_vector**2)
# # # #     residual_power = np.sum(
# # # #         (signal_vector - fitted_signal_vector) ** 2
# # # #     )

# # # #     explained_fraction = (
# # # #         1.0 - residual_power / signal_power
# # # #         if signal_power > 0
# # # #         else np.nan
# # # #     )

# # # #     wfe_values_nm = reconstructed_opd_nm[pupil_wave]
# # # #     equivalent_wfe_rms_nm = np.sqrt(np.mean(wfe_values_nm**2))
# # # #     peak_abs_wfe_nm = np.max(np.abs(wfe_values_nm))

# # # #     centre_mask_wave = pupil_wave & (rho_wave < CENTER_MAX_RHO)
# # # #     edge_mask_wave = (
# # # #         pupil_wave
# # # #         & (rho_wave >= EDGE_MIN_RHO)
# # # #         & (rho_wave <= 1.0)
# # # #     )

# # # #     centre_wfe_rms_nm = np.sqrt(
# # # #         np.mean(reconstructed_opd_nm[centre_mask_wave] ** 2)
# # # #     )

# # # #     edge_wfe_rms_nm = np.sqrt(
# # # #         np.mean(reconstructed_opd_nm[edge_mask_wave] ** 2)
# # # #     )

# # # #     for radial_index in range(N_RADIAL_BINS):
# # # #         detector_annulus = (
# # # #             analysis_mask
# # # #             & (rho_detector >= radial_edges[radial_index])
# # # #             & (rho_detector < radial_edges[radial_index + 1])
# # # #         )

# # # #         wave_annulus = (
# # # #             pupil_wave
# # # #             & (rho_wave >= radial_edges[radial_index])
# # # #             & (rho_wave < radial_edges[radial_index + 1])
# # # #         )

# # # #         if np.any(detector_annulus):
# # # #             radial_values = signal_bias[detector_annulus]

# # # #             radial_signal_mean[mask_number, radial_index] = np.mean(
# # # #                 radial_values
# # # #             )

# # # #             radial_signal_rms[mask_number, radial_index] = np.sqrt(
# # # #                 np.mean(radial_values**2)
# # # #             )

# # # #         if np.any(wave_annulus):
# # # #             radial_values_nm = reconstructed_opd_nm[wave_annulus]

# # # #             radial_wfe_mean_nm[mask_number, radial_index] = np.mean(
# # # #                 radial_values_nm
# # # #             )

# # # #             radial_wfe_rms_nm[mask_number, radial_index] = np.sqrt(
# # # #                 np.mean(radial_values_nm**2)
# # # #             )

# # # #     condition_number_retained = (
# # # #         singular_values[keep][0] / singular_values[keep][-1]
# # # #     )

# # # #     table_rows.append(
# # # #         {
# # # #             "mask": mask_name,
# # # #             "mask_diameter_um": float(mask_entry["mask_diam_um"]),
# # # #             "peak_abs_signal_bias": float(signal_peak_abs),
# # # #             "signal_bias_rmse": float(signal_rmse),
# # # #             "equivalent_wfe_rms_nm": float(equivalent_wfe_rms_nm),
# # # #             "peak_abs_wfe_nm": float(peak_abs_wfe_nm),
# # # #             "centre_wfe_rms_nm": float(centre_wfe_rms_nm),
# # # #             "edge_wfe_rms_nm": float(edge_wfe_rms_nm),
# # # #             "explained_signal_percent": float(
# # # #                 100.0 * explained_fraction
# # # #             ),
# # # #             "retained_rank": int(np.sum(keep)),
# # # #             "retained_condition_number": float(
# # # #                 condition_number_retained
# # # #             ),
# # # #         }
# # # #     )

# # # #     signal_bias_maps.append(signal_bias)
# # # #     reconstructed_opd_maps_nm.append(reconstructed_opd_nm)
# # # #     fitted_signal_maps.append(fitted_signal_map)


# # # # signal_bias_maps = np.asarray(signal_bias_maps)
# # # # reconstructed_opd_maps_nm = np.asarray(reconstructed_opd_maps_nm)
# # # # fitted_signal_maps = np.asarray(fitted_signal_maps)


# # # # # ============================================================
# # # # # CSV and LaTeX tables
# # # # # ============================================================

# # # # fieldnames = list(table_rows[0].keys())

# # # # with open(OUTPUT_TABLE_CSV, "w", newline="") as f:
# # # #     writer = csv.DictWriter(f, fieldnames=fieldnames)
# # # #     writer.writeheader()
# # # #     writer.writerows(table_rows)

# # # # with open(OUTPUT_TABLE_TEX, "w") as f:
# # # #     f.write("\\begin{tabular}{lrrrrrrr}\n")
# # # #     f.write("\\hline\n")
# # # #     f.write(
# # # #         "Mask & Diam. & Peak $|\\Delta s|$ & "
# # # #         "RMSE $\\Delta s$ & WFE RMS & Peak $|\\phi|$ & "
# # # #         "Centre RMS & Edge RMS \\\\\n"
# # # #     )
# # # #     f.write(
# # # #         " & [$\\mu$m] & & & [nm] & [nm] & [nm] & [nm] \\\\\n"
# # # #     )
# # # #     f.write("\\hline\n")

# # # #     for row in table_rows:
# # # #         f.write(
# # # #             f"{row['mask']} & "
# # # #             f"{row['mask_diameter_um']:.0f} & "
# # # #             f"{row['peak_abs_signal_bias']:.3e} & "
# # # #             f"{row['signal_bias_rmse']:.3e} & "
# # # #             f"{row['equivalent_wfe_rms_nm']:.2f} & "
# # # #             f"{row['peak_abs_wfe_nm']:.2f} & "
# # # #             f"{row['centre_wfe_rms_nm']:.2f} & "
# # # #             f"{row['edge_wfe_rms_nm']:.2f} \\\\\n"
# # # #         )

# # # #     f.write("\\hline\n")
# # # #     f.write("\\end{tabular}\n")


# # # # # ============================================================
# # # # # Radial profile figure
# # # # # ============================================================

# # # # fig, ax = plt.subplots(2, 2, figsize=(12, 9), sharex=True)

# # # # for mask_number, mask_name in enumerate(MASK_NAMES):
# # # #     ax[0, 0].plot(
# # # #         radial_centres,
# # # #         radial_signal_mean[mask_number],
# # # #         "o-",
# # # #         label=mask_name,
# # # #     )

# # # #     ax[0, 1].plot(
# # # #         radial_centres,
# # # #         radial_signal_rms[mask_number],
# # # #         "o-",
# # # #         label=mask_name,
# # # #     )

# # # #     ax[1, 0].plot(
# # # #         radial_centres,
# # # #         radial_wfe_mean_nm[mask_number],
# # # #         "o-",
# # # #         label=mask_name,
# # # #     )

# # # #     ax[1, 1].plot(
# # # #         radial_centres,
# # # #         radial_wfe_rms_nm[mask_number],
# # # #         "o-",
# # # #         label=mask_name,
# # # #     )

# # # # ax[0, 0].axhline(0.0, linewidth=0.8)
# # # # ax[1, 0].axhline(0.0, linewidth=0.8)

# # # # ax[0, 0].set_ylabel("Mean normalized-signal bias")
# # # # ax[0, 1].set_ylabel("RMS normalized-signal bias")
# # # # ax[1, 0].set_ylabel("Mean reconstructed OPD bias [nm]")
# # # # ax[1, 1].set_ylabel("RMS reconstructed OPD bias [nm]")

# # # # ax[0, 0].set_title("Signed radial signal bias")
# # # # ax[0, 1].set_title("Radial signal-bias magnitude")
# # # # ax[1, 0].set_title("Signed radial reconstruction bias")
# # # # ax[1, 1].set_title("Radial reconstruction-bias magnitude")

# # # # for a in ax[1]:
# # # #     a.set_xlabel("Normalized pupil radius")

# # # # for a in ax.flat:
# # # #     a.axvline(EDGE_MIN_RHO, linestyle=":", linewidth=1)
# # # #     a.grid(alpha=0.25)

# # # # ax[0, 0].legend(ncol=2)
# # # # ax[0, 1].legend(ncol=2)

# # # # fig.suptitle(
# # # #     f"Chromatic I0 bias: {T_INTERNAL_K:.0f} K internal reference "
# # # #     f"used on {T_ONSKY_K:.0f} K source"
# # # # )

# # # # plt.tight_layout()
# # # # plt.savefig(
# # # #     OUTPUT_RADIAL_FIGURE,
# # # #     dpi=220,
# # # #     bbox_inches="tight",
# # # # )


# # # # # ============================================================
# # # # # Global summary figure
# # # # # ============================================================

# # # # mask_labels = [row["mask"] for row in table_rows]
# # # # signal_peak_values = [
# # # #     row["peak_abs_signal_bias"] for row in table_rows
# # # # ]
# # # # signal_rmse_values = [
# # # #     row["signal_bias_rmse"] for row in table_rows
# # # # ]
# # # # wfe_rms_values = [
# # # #     row["equivalent_wfe_rms_nm"] for row in table_rows
# # # # ]
# # # # centre_values = [
# # # #     row["centre_wfe_rms_nm"] for row in table_rows
# # # # ]
# # # # edge_values = [
# # # #     row["edge_wfe_rms_nm"] for row in table_rows
# # # # ]

# # # # x = np.arange(len(mask_labels))
# # # # width = 0.36

# # # # fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# # # # ax[0].bar(
# # # #     x - width / 2,
# # # #     signal_peak_values,
# # # #     width,
# # # #     label="Peak |bias|",
# # # # )

# # # # ax[0].bar(
# # # #     x + width / 2,
# # # #     signal_rmse_values,
# # # #     width,
# # # #     label="RMSE",
# # # # )

# # # # ax[0].set_xticks(x, mask_labels)
# # # # ax[0].set_ylabel("Normalized-signal bias")
# # # # ax[0].set_title("Reference signal bias")
# # # # ax[0].legend()
# # # # ax[0].grid(axis="y", alpha=0.25)

# # # # ax[1].bar(
# # # #     x - width,
# # # #     wfe_rms_values,
# # # #     width,
# # # #     label="Full pupil",
# # # # )

# # # # ax[1].bar(
# # # #     x,
# # # #     centre_values,
# # # #     width,
# # # #     label=f"Centre, ρ < {CENTER_MAX_RHO}",
# # # # )

# # # # ax[1].bar(
# # # #     x + width,
# # # #     edge_values,
# # # #     width,
# # # #     label=f"Edge, ρ ≥ {EDGE_MIN_RHO}",
# # # # )

# # # # ax[1].set_xticks(x, mask_labels)
# # # # ax[1].set_ylabel("Equivalent WFE bias [nm RMS OPD]")
# # # # ax[1].set_title("Linear reconstruction bias")
# # # # ax[1].legend()
# # # # ax[1].grid(axis="y", alpha=0.25)

# # # # plt.tight_layout()
# # # # plt.savefig(
# # # #     OUTPUT_SUMMARY_FIGURE,
# # # #     dpi=220,
# # # #     bbox_inches="tight",
# # # # )


# # # # # ============================================================
# # # # # Save numerical products and print table
# # # # # ============================================================

# # # # np.savez_compressed(
# # # #     OUTPUT_DATA,
# # # #     mask_names=np.asarray(MASK_NAMES),
# # # #     radial_edges=radial_edges,
# # # #     radial_centres=radial_centres,
# # # #     pupil_detector=pupil_detector,
# # # #     analysis_mask=analysis_mask,
# # # #     pupil_wave=pupil_wave,
# # # #     rho_detector=rho_detector,
# # # #     rho_wave=rho_wave,
# # # #     N0_internal=N0_internal,
# # # #     N0_onsky=N0_onsky,
# # # #     signal_bias_maps=signal_bias_maps,
# # # #     reconstructed_opd_maps_nm=reconstructed_opd_maps_nm,
# # # #     fitted_signal_maps=fitted_signal_maps,
# # # #     radial_signal_mean=radial_signal_mean,
# # # #     radial_signal_rms=radial_signal_rms,
# # # #     radial_wfe_mean_nm=radial_wfe_mean_nm,
# # # #     radial_wfe_rms_nm=radial_wfe_rms_nm,
# # # # )

# # # # print()
# # # # print(
# # # #     "Mask  Diam[um]  Peak|ds|    RMSE(ds)    "
# # # #     "WFE_RMS[nm]  Centre[nm]  Edge[nm]  Explained[%]"
# # # # )

# # # # for row in table_rows:
# # # #     print(
# # # #         f"{row['mask']:4s}  "
# # # #         f"{row['mask_diameter_um']:8.0f}  "
# # # #         f"{row['peak_abs_signal_bias']:9.3e}  "
# # # #         f"{row['signal_bias_rmse']:9.3e}  "
# # # #         f"{row['equivalent_wfe_rms_nm']:11.2f}  "
# # # #         f"{row['centre_wfe_rms_nm']:10.2f}  "
# # # #         f"{row['edge_wfe_rms_nm']:8.2f}  "
# # # #         f"{row['explained_signal_percent']:12.1f}"
# # # #     )

# # # # print()
# # # # print(f"Saved: {OUTPUT_TABLE_CSV.resolve()}")
# # # # print(f"Saved: {OUTPUT_TABLE_TEX.resolve()}")
# # # # print(f"Saved: {OUTPUT_RADIAL_FIGURE.resolve()}")
# # # # print(f"Saved: {OUTPUT_SUMMARY_FIGURE.resolve()}")
# # # # print(f"Saved: {OUTPUT_DATA.resolve()}")

# # # # plt.show()


# # # # # #!/usr/bin/env python3

# # # # # import copy
# # # # # import json
# # # # # import sys
# # # # # import tempfile
# # # # # from pathlib import Path

# # # # # import matplotlib.pyplot as plt
# # # # # import numpy as np
# # # # # from scipy.ndimage import binary_erosion

# # # # # REPO_ROOT = Path(__file__).resolve().parents[3]
# # # # # sys.path.insert(0, str(REPO_ROOT))

# # # # # from baldrapp.common import baldr_core as bldr
# # # # # from baldrapp.common import DM_basis
# # # # # from baldrapp.common import spectrum as spec


# # # # # # ============================================================
# # # # # # Settings
# # # # # # ============================================================

# # # # # CONFIG_PATH = REPO_ROOT / "baldrapp/apps/paranal_simulator/fake_configs/baldr_config.json"

# # # # # T_INTERNAL_K = 1900.0
# # # # # T_ONSKY_K = 10000.0

# # # # # LINEAR_POKE_NM = 10.0
# # # # # INJECTION_RMS_NM = np.array([0, 5, 10, 20, 40, 80, 120], dtype=float)

# # # # # # Set this to the relevant total or reference-allocation WFE budget to print
# # # # # # the fraction consumed by the temperature mismatch.
# # # # # WFE_BUDGET_NM = None

# # # # # OUTPUT_TEMPERATURE_FIGURE = Path("I0_temperature_comparison.png")
# # # # # OUTPUT_WFE_FIGURE = Path("I0_temperature_WFE_interpretation.png")


# # # # # # ============================================================
# # # # # # Same instrument state; only source temperature changes
# # # # # # ============================================================

# # # # # with open(CONFIG_PATH, "r") as f:
# # # # #     base_cfg = json.load(f)

# # # # # cfg_internal = copy.deepcopy(base_cfg)
# # # # # cfg_onsky = copy.deepcopy(base_cfg)

# # # # # for cfg, temperature_K in (
# # # # #     (cfg_internal, T_INTERNAL_K),
# # # # #     (cfg_onsky, T_ONSKY_K),
# # # # # ):
# # # # #     cfg["stellar"]["spectrum"]["enabled"] = True
# # # # #     cfg["stellar"]["spectrum"]["mode"] = "blackbody"
# # # # #     cfg["stellar"]["spectrum"]["temperature_K"] = temperature_K

# # # # #     # No internal alignment differences between calibration and sky.
# # # # #     cfg["fresnel_relay"]["coldstop_x_offset"] = 0.0
# # # # #     cfg["fresnel_relay"]["coldstop_y_offset"] = 0.0
# # # # #     cfg["fresnel_relay"]["pupil_misconjugation"] = 0.0
# # # # #     cfg["fresnel_relay"]["edge_angle"] = 0.0
# # # # #     cfg["fresnel_relay"]["use_nominal_pupil_conjugation"] = True

# # # # #     # D mirror (knife edge) offset (vignetting)
# # # # #     cfg["fresnel_relay"]["edge_offset"] = -20.0
# # # # #     # position, not a differential internal/sky misalignment.

# # # # #     cfg["internal_aberrations"]["enabled"] = False

# # # # #     cfg["detector"]["enabled"] = True
# # # # #     cfg["detector"]["ron"] = 0.0
# # # # #     cfg["detector"]["include_shotnoise"] = False
# # # # #     cfg["detector"]["include_readnoise"] = False
# # # # #     cfg["detector"]["adu_offset"] = 0.0
# # # # #     cfg["detector"]["noise_std_adu"] = 0.0

# # # # # with tempfile.TemporaryDirectory() as tmp:
# # # # #     tmp = Path(tmp)
# # # # #     internal_path = tmp / "internal.json"
# # # # #     onsky_path = tmp / "onsky.json"

# # # # #     with open(internal_path, "w") as f:
# # # # #         json.dump(cfg_internal, f, indent=2)

# # # # #     with open(onsky_path, "w") as f:
# # # # #         json.dump(cfg_onsky, f, indent=2)

# # # # #     zwfs_internal = bldr.init_zwfs_from_json(internal_path)
# # # # #     zwfs_onsky = bldr.init_zwfs_from_json(onsky_path)


# # # # # # ============================================================
# # # # # # Same physical phase mask
# # # # # # ============================================================

# # # # # phasemask_path = REPO_ROOT / base_cfg["simulator_runtime"]["phasemask"]["properties_file"]
# # # # # mask_name = base_cfg["simulator_runtime"]["phasemask"]["default_mask"]

# # # # # with open(phasemask_path, "r") as f:
# # # # #     phasemask_cfg = json.load(f)

# # # # # mask_entry = phasemask_cfg["phasemask"]["masks"][mask_name]

# # # # # zwfs_internal.optics.active_phasemask = spec.normalise_phasemask_entry(
# # # # #     mask_name, mask_entry, zwfs_internal.optics
# # # # # )
# # # # # zwfs_onsky.optics.active_phasemask = spec.normalise_phasemask_entry(
# # # # #     mask_name, mask_entry, zwfs_onsky.optics
# # # # # )


# # # # # # ============================================================
# # # # # # Zero-aberration references
# # # # # # ============================================================

# # # # # amp_internal = zwfs_internal.grid.pupil_mask.astype(float)
# # # # # amp_onsky = zwfs_onsky.grid.pupil_mask.astype(float)

# # # # # zero_internal = np.zeros_like(amp_internal)
# # # # # zero_onsky = np.zeros_like(amp_onsky)

# # # # # I0_internal = bldr.get_I0_configured(
# # # # #     opd_input=zero_internal,
# # # # #     amp_input=amp_internal,
# # # # #     opd_internal=zero_internal,
# # # # #     zwfs_ns=zwfs_internal,
# # # # #     detector=zwfs_internal.detector,
# # # # #     include_shotnoise=False,
# # # # # )

# # # # # N0_internal = bldr.get_N0_configured(
# # # # #     opd_input=zero_internal,
# # # # #     amp_input=amp_internal,
# # # # #     opd_internal=zero_internal,
# # # # #     zwfs_ns=zwfs_internal,
# # # # #     detector=zwfs_internal.detector,
# # # # #     include_shotnoise=False,
# # # # # )

# # # # # I0_onsky = bldr.get_I0_configured(
# # # # #     opd_input=zero_onsky,
# # # # #     amp_input=amp_onsky,
# # # # #     opd_internal=zero_onsky,
# # # # #     zwfs_ns=zwfs_onsky,
# # # # #     detector=zwfs_onsky.detector,
# # # # #     include_shotnoise=False,
# # # # # )

# # # # # N0_onsky = bldr.get_N0_configured(
# # # # #     opd_input=zero_onsky,
# # # # #     amp_input=amp_onsky,
# # # # #     opd_internal=zero_onsky,
# # # # #     zwfs_ns=zwfs_onsky,
# # # # #     detector=zwfs_onsky.detector,
# # # # #     include_shotnoise=False,
# # # # # )

# # # # # binning = int(base_cfg["detector"]["binning"])
# # # # # pupil_detector = (
# # # # #     bldr.sum_subarrays(
# # # # #         zwfs_internal.grid.pupil_mask,
# # # # #         block_size=(binning, binning),
# # # # #     )
# # # # #     > 0.5 * binning**2
# # # # # )

# # # # # analysis_mask = binary_erosion(pupil_detector, iterations=1)

# # # # # I0_internal_norm = I0_internal / np.mean(N0_internal[analysis_mask])
# # # # # I0_onsky_norm = I0_onsky / np.mean(N0_onsky[analysis_mask])

# # # # # temperature_signal = I0_onsky_norm - I0_internal_norm

# # # # # rms_temperature_signal = np.sqrt(
# # # # #     np.mean(temperature_signal[analysis_mask] ** 2)
# # # # # )


# # # # # # ============================================================
# # # # # # Temperature comparison plot
# # # # # # ============================================================

# # # # # relative_difference_percent = np.zeros_like(temperature_signal)
# # # # # relative_difference_percent[analysis_mask] = (
# # # # #     100.0
# # # # #     * temperature_signal[analysis_mask]
# # # # #     / np.maximum(np.abs(I0_internal_norm[analysis_mask]), 1e-12)
# # # # # )

# # # # # wavelength_um = zwfs_internal.spectrum.wavelengths * 1e6

# # # # # fig, ax = plt.subplots(2, 3, figsize=(13, 8))

# # # # # ax[0, 0].plot(
# # # # #     wavelength_um,
# # # # #     zwfs_internal.spectrum.weights_normalized,
# # # # #     "o-",
# # # # #     label=f"{T_INTERNAL_K:.0f} K",
# # # # # )
# # # # # ax[0, 0].plot(
# # # # #     wavelength_um,
# # # # #     zwfs_onsky.spectrum.weights_normalized,
# # # # #     "o-",
# # # # #     label=f"{T_ONSKY_K:.0f} K",
# # # # # )
# # # # # ax[0, 0].set_xlabel("Wavelength [µm]")
# # # # # ax[0, 0].set_ylabel("Normalized photon weight")
# # # # # ax[0, 0].set_title("Configured bandpass weights")
# # # # # ax[0, 0].legend()

# # # # # im = ax[0, 1].imshow(I0_internal_norm)
# # # # # fig.colorbar(im, ax=ax[0, 1])
# # # # # ax[0, 1].set_title("Normalized internal I0")

# # # # # im = ax[0, 2].imshow(I0_onsky_norm)
# # # # # fig.colorbar(im, ax=ax[0, 2])
# # # # # ax[0, 2].set_title("Normalized on-sky I0")

# # # # # limit = np.max(np.abs(temperature_signal[analysis_mask]))
# # # # # im = ax[1, 0].imshow(
# # # # #     temperature_signal,
# # # # #     vmin=-limit,
# # # # #     vmax=limit,
# # # # #     cmap="RdBu_r",
# # # # # )
# # # # # fig.colorbar(im, ax=ax[1, 0])
# # # # # ax[1, 0].set_title("On-sky − internal")

# # # # # limit_percent = np.nanpercentile(
# # # # #     np.abs(relative_difference_percent[analysis_mask]), 99
# # # # # )
# # # # # im = ax[1, 1].imshow(
# # # # #     relative_difference_percent,
# # # # #     vmin=-limit_percent,
# # # # #     vmax=limit_percent,
# # # # #     cmap="RdBu_r",
# # # # # )
# # # # # fig.colorbar(im, ax=ax[1, 1], label="%")
# # # # # ax[1, 1].set_title("Relative difference")

# # # # # ax[1, 2].axis("off")
# # # # # ax[1, 2].text(
# # # # #     0.0,
# # # # #     0.95,
# # # # #     f"Mask: {mask_name}\n"
# # # # #     f"Band: {wavelength_um.min():.3f}–{wavelength_um.max():.3f} µm\n"
# # # # #     f"RMS temperature signal: {rms_temperature_signal:.6g}",
# # # # #     va="top",
# # # # #     family="monospace",
# # # # # )

# # # # # for a in ax.flat:
# # # # #     if a is not ax[0, 0] and a is not ax[1, 2]:
# # # # #         a.set_xticks([])
# # # # #         a.set_yticks([])

# # # # # plt.tight_layout()
# # # # # plt.savefig(OUTPUT_TEMPERATURE_FIGURE, dpi=200, bbox_inches="tight")


# # # # # # ============================================================
# # # # # # Small-aberration Zernike responses
# # # # # # ============================================================

# # # # # pupil_wave = zwfs_onsky.grid.pupil_mask.astype(bool)
# # # # # yy, xx = np.indices(pupil_wave.shape)
# # # # # cy = np.mean(yy[pupil_wave])
# # # # # cx = np.mean(xx[pupil_wave])
# # # # # radius = np.percentile(
# # # # #     np.sqrt((xx[pupil_wave] - cx) ** 2 + (yy[pupil_wave] - cy) ** 2),
# # # # #     99.5,
# # # # # )

# # # # # rho = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius
# # # # # theta = np.arctan2(yy - cy, xx - cx)

# # # # # zernikes = DM_basis.zernike_basis(
# # # # #     nterms=8,
# # # # #     rho=rho,
# # # # #     theta=theta,
# # # # #     outside=0.0,
# # # # # )

# # # # # mode_indices = [1, 2, 3, 4, 5, 6, 7]
# # # # # mode_names = [
# # # # #     "Tip",
# # # # #     "Tilt",
# # # # #     "Defocus",
# # # # #     "Astig 1",
# # # # #     "Astig 2",
# # # # #     "Coma 1",
# # # # #     "Coma 2",
# # # # # ]

# # # # # modes = []

# # # # # for index in mode_indices:
# # # # #     mode = np.nan_to_num(zernikes[index], nan=0.0)
# # # # #     mode *= pupil_wave
# # # # #     mode -= np.mean(mode[pupil_wave])
# # # # #     mode /= np.std(mode[pupil_wave])
# # # # #     modes.append(mode)

# # # # # response_columns = []
# # # # # sweep_signals = np.zeros((len(modes), len(INJECTION_RMS_NM)))

# # # # # for mode_index, mode in enumerate(modes):
# # # # #     opd_plus = LINEAR_POKE_NM * 1e-9 * mode
# # # # #     opd_minus = -LINEAR_POKE_NM * 1e-9 * mode

# # # # #     I_plus = bldr.get_I0_configured(
# # # # #         opd_input=opd_plus,
# # # # #         amp_input=amp_onsky,
# # # # #         opd_internal=zero_onsky,
# # # # #         zwfs_ns=zwfs_onsky,
# # # # #         detector=zwfs_onsky.detector,
# # # # #         include_shotnoise=False,
# # # # #     )
# # # # #     N_plus = bldr.get_N0_configured(
# # # # #         opd_input=opd_plus,
# # # # #         amp_input=amp_onsky,
# # # # #         opd_internal=zero_onsky,
# # # # #         zwfs_ns=zwfs_onsky,
# # # # #         detector=zwfs_onsky.detector,
# # # # #         include_shotnoise=False,
# # # # #     )

# # # # #     I_minus = bldr.get_I0_configured(
# # # # #         opd_input=opd_minus,
# # # # #         amp_input=amp_onsky,
# # # # #         opd_internal=zero_onsky,
# # # # #         zwfs_ns=zwfs_onsky,
# # # # #         detector=zwfs_onsky.detector,
# # # # #         include_shotnoise=False,
# # # # #     )
# # # # #     N_minus = bldr.get_N0_configured(
# # # # #         opd_input=opd_minus,
# # # # #         amp_input=amp_onsky,
# # # # #         opd_internal=zero_onsky,
# # # # #         zwfs_ns=zwfs_onsky,
# # # # #         detector=zwfs_onsky.detector,
# # # # #         include_shotnoise=False,
# # # # #     )

# # # # #     I_plus_norm = I_plus / np.mean(N_plus[analysis_mask])
# # # # #     I_minus_norm = I_minus / np.mean(N_minus[analysis_mask])

# # # # #     derivative_per_nm = (
# # # # #         I_plus_norm - I_minus_norm
# # # # #     ) / (2.0 * LINEAR_POKE_NM)

# # # # #     response_columns.append(derivative_per_nm[analysis_mask])

# # # # #     for amplitude_index, amplitude_nm in enumerate(INJECTION_RMS_NM):
# # # # #         if amplitude_nm == 0:
# # # # #             injected_norm = I0_onsky_norm
# # # # #         else:
# # # # #             opd_injected = amplitude_nm * 1e-9 * mode

# # # # #             I_injected = bldr.get_I0_configured(
# # # # #                 opd_input=opd_injected,
# # # # #                 amp_input=amp_onsky,
# # # # #                 opd_internal=zero_onsky,
# # # # #                 zwfs_ns=zwfs_onsky,
# # # # #                 detector=zwfs_onsky.detector,
# # # # #                 include_shotnoise=False,
# # # # #             )
# # # # #             N_injected = bldr.get_N0_configured(
# # # # #                 opd_input=opd_injected,
# # # # #                 amp_input=amp_onsky,
# # # # #                 opd_internal=zero_onsky,
# # # # #                 zwfs_ns=zwfs_onsky,
# # # # #                 detector=zwfs_onsky.detector,
# # # # #                 include_shotnoise=False,
# # # # #             )

# # # # #             injected_norm = I_injected / np.mean(N_injected[analysis_mask])

# # # # #         sweep_signals[mode_index, amplitude_index] = np.sqrt(
# # # # #             np.mean(
# # # # #                 (
# # # # #                     injected_norm[analysis_mask]
# # # # #                     - I0_onsky_norm[analysis_mask]
# # # # #                 )
# # # # #                 ** 2
# # # # #             )
# # # # #         )

# # # # # response_matrix = np.column_stack(response_columns)

# # # # # equivalent_coefficients_nm, _, _, _ = np.linalg.lstsq(
# # # # #     response_matrix,
# # # # #     temperature_signal[analysis_mask],
# # # # #     rcond=None,
# # # # # )

# # # # # equivalent_opd_nm = np.zeros_like(amp_onsky)

# # # # # for coefficient_nm, mode in zip(equivalent_coefficients_nm, modes):
# # # # #     equivalent_opd_nm += coefficient_nm * mode

# # # # # equivalent_wfe_rms_nm = np.std(equivalent_opd_nm[pupil_wave])

# # # # # fitted_temperature_signal = (
# # # # #     response_matrix @ equivalent_coefficients_nm
# # # # # )

# # # # # explained_fraction = 1.0 - (
# # # # #     np.sum(
# # # # #         (
# # # # #             temperature_signal[analysis_mask]
# # # # #             - fitted_temperature_signal
# # # # #         )
# # # # #         ** 2
# # # # #     )
# # # # #     / np.sum(temperature_signal[analysis_mask] ** 2)
# # # # # )


# # # # # # ============================================================
# # # # # # WFE interpretation plot
# # # # # # ============================================================

# # # # # fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# # # # # for mode_name, signal_curve in zip(mode_names, sweep_signals):
# # # # #     ax[0].plot(
# # # # #         INJECTION_RMS_NM,
# # # # #         signal_curve,
# # # # #         "o-",
# # # # #         label=mode_name,
# # # # #     )

# # # # # ax[0].axhline(
# # # # #     rms_temperature_signal,
# # # # #     linestyle="--",
# # # # #     label="Temperature mismatch",
# # # # # )
# # # # # ax[0].set_xlabel("Injected modal WFE [nm RMS OPD]")
# # # # # ax[0].set_ylabel("RMS normalized-intensity change")
# # # # # ax[0].set_title("Temperature signal versus aberration signals")
# # # # # ax[0].legend(fontsize=8)

# # # # # ax[1].bar(mode_names, equivalent_coefficients_nm)
# # # # # ax[1].axhline(0.0, linewidth=1)
# # # # # ax[1].set_ylabel("Equivalent coefficient [nm RMS OPD]")
# # # # # ax[1].set_title(
# # # # #     f"Equivalent WFE = {equivalent_wfe_rms_nm:.2f} nm RMS\n"
# # # # #     f"Linear modal fit explains {100 * explained_fraction:.1f}%"
# # # # # )
# # # # # ax[1].tick_params(axis="x", rotation=45)

# # # # # plt.tight_layout()
# # # # # plt.savefig(OUTPUT_WFE_FIGURE, dpi=200, bbox_inches="tight")
# # # # # plt.show()


# # # # # # ============================================================
# # # # # # Prints
# # # # # # ============================================================

# # # # # print(f"Saved: {OUTPUT_TEMPERATURE_FIGURE.resolve()}")
# # # # # print(f"Saved: {OUTPUT_WFE_FIGURE.resolve()}")
# # # # # print()
# # # # # print(f"Temperature-only false signal RMS: {rms_temperature_signal:.6g}")
# # # # # print("Equivalent modal coefficients:")
# # # # # for mode_name, coefficient_nm in zip(mode_names, equivalent_coefficients_nm):
# # # # #     print(f"  {mode_name:8s}: {coefficient_nm:+8.3f} nm RMS OPD")

# # # # # print(f"Combined equivalent WFE: {equivalent_wfe_rms_nm:.3f} nm RMS OPD")
# # # # # print(f"Temperature signal explained by these modes: {100 * explained_fraction:.1f}%")
# # # # # print()
# # # # # print(
# # # # #     "Interpretation: this is a reference zero-point bias, not photon noise. "
# # # # #     "If the source-temperature mismatch is not corrected, treat the equivalent "
# # # # #     "WFE as a systematic reference-calibration allocation."
# # # # # )

# # # # # if WFE_BUDGET_NM is not None:
# # # # #     fraction_percent = 100.0 * equivalent_wfe_rms_nm / WFE_BUDGET_NM
# # # # #     remaining_rss_nm = np.sqrt(
# # # # #         max(WFE_BUDGET_NM**2 - equivalent_wfe_rms_nm**2, 0.0)
# # # # #     )

# # # # #     print(
# # # # #         f"It consumes {fraction_percent:.1f}% of a "
# # # # #         f"{WFE_BUDGET_NM:.1f} nm RMS budget."
# # # # #     )
# # # # #     print(
# # # # #         f"Remaining RSS allocation: {remaining_rss_nm:.2f} nm RMS."
# # # # #     )

# # # # # # #!/usr/bin/env python3

# # # # # # import copy
# # # # # # import json
# # # # # # import sys
# # # # # # import tempfile
# # # # # # from pathlib import Path

# # # # # # import matplotlib.pyplot as plt
# # # # # # import numpy as np

# # # # # # # Run this script from inside the BaldrApp repository.
# # # # # # REPO_ROOT = Path(__file__).resolve().parents[3]
# # # # # # sys.path.insert(0, str(REPO_ROOT))

# # # # # # from baldrapp.common import baldr_core as bldr
# # # # # # from baldrapp.common import spectrum as spec


# # # # # # # ============================================================
# # # # # # # Settings
# # # # # # # ============================================================

# # # # # # CONFIG_PATH = REPO_ROOT / "baldrapp/apps/paranal_simulator/fake_configs/baldr_config.json"
# # # # # # T_INTERNAL_K = 1900.0
# # # # # # T_ONSKY_K = 10000.0   # "10 K" here means 10,000 K
# # # # # # OUTPUT_FIGURE = Path("I0_temperature_comparison.png")


# # # # # # # ============================================================
# # # # # # # Load one instrument configuration and change only temperature
# # # # # # # ============================================================

# # # # # # with open(CONFIG_PATH, "r") as f:
# # # # # #     base_cfg = json.load(f)

# # # # # # cfg_internal = copy.deepcopy(base_cfg)
# # # # # # cfg_onsky = copy.deepcopy(base_cfg)

# # # # # # cfg_internal["stellar"]["spectrum"]["enabled"] = True
# # # # # # cfg_internal["stellar"]["spectrum"]["mode"] = "blackbody"
# # # # # # cfg_internal["stellar"]["spectrum"]["temperature_K"] = T_INTERNAL_K

# # # # # # cfg_onsky["stellar"]["spectrum"]["enabled"] = True
# # # # # # cfg_onsky["stellar"]["spectrum"]["mode"] = "blackbody"
# # # # # # cfg_onsky["stellar"]["spectrum"]["temperature_K"] = T_ONSKY_K

# # # # # # # Noise-free detector sampling.
# # # # # # for cfg in (cfg_internal, cfg_onsky):
# # # # # #     cfg["detector"]["enabled"] = True
# # # # # #     cfg["detector"]["ron"] = 0.0
# # # # # #     cfg["detector"]["include_shotnoise"] = False
# # # # # #     cfg["detector"]["include_readnoise"] = False
# # # # # #     cfg["detector"]["adu_offset"] = 0.0
# # # # # #     cfg["detector"]["noise_std_adu"] = 0.0

# # # # # # # NO MISALIGNMENT 
# # # # # # for cfg in (cfg_internal, cfg_onsky):
# # # # # #     cfg["fresnel_relay"]["coldstop_x_offset"] = 0.0
# # # # # #     cfg["fresnel_relay"]["coldstop_y_offset"] = 0.0
# # # # # #     cfg["fresnel_relay"]["pupil_misconjugation"] = 0.0
# # # # # #     cfg["fresnel_relay"]["edge_angle"] = 0.0
# # # # # #     cfg["fresnel_relay"]["edge_offset"] = -20.0
# # # # # #     cfg["fresnel_relay"]["use_nominal_pupil_conjugation"] = True
# # # # # #     cfg["internal_aberrations"]["enabled"] = False

# # # # # # with tempfile.TemporaryDirectory() as tmp:
# # # # # #     tmp = Path(tmp)
# # # # # #     internal_path = tmp / "internal_1900K.json"
# # # # # #     onsky_path = tmp / "onsky_10000K.json"

# # # # # #     with open(internal_path, "w") as f:
# # # # # #         json.dump(cfg_internal, f, indent=2)

# # # # # #     with open(onsky_path, "w") as f:
# # # # # #         json.dump(cfg_onsky, f, indent=2)

# # # # # #     zwfs_internal = bldr.init_zwfs_from_json(internal_path)
# # # # # #     zwfs_onsky = bldr.init_zwfs_from_json(onsky_path)

# # # # # # # Use the same physical phase mask for both temperatures.
# # # # # # phasemask_path = REPO_ROOT / base_cfg["simulator_runtime"]["phasemask"]["properties_file"]
# # # # # # mask_name = base_cfg["simulator_runtime"]["phasemask"]["default_mask"]

# # # # # # with open(phasemask_path, "r") as f:
# # # # # #     phasemask_cfg = json.load(f)

# # # # # # mask_entry = phasemask_cfg["phasemask"]["masks"][mask_name]

# # # # # # zwfs_internal.optics.active_phasemask = spec.normalise_phasemask_entry(
# # # # # #     mask_name, mask_entry, zwfs_internal.optics
# # # # # # )
# # # # # # zwfs_onsky.optics.active_phasemask = spec.normalise_phasemask_entry(
# # # # # #     mask_name, mask_entry, zwfs_onsky.optics
# # # # # # )

# # # # # # # Same pupil, optics, total spectral-density amplitude, and zero OPD.
# # # # # # amp_internal = zwfs_internal.grid.pupil_mask.astype(float)
# # # # # # amp_onsky = zwfs_onsky.grid.pupil_mask.astype(float)

# # # # # # opd_internal = np.zeros_like(amp_internal)
# # # # # # opd_onsky = np.zeros_like(amp_onsky)

# # # # # # I0_internal = bldr.get_I0_configured(
# # # # # #     opd_input=opd_internal,
# # # # # #     amp_input=amp_internal,
# # # # # #     opd_internal=np.zeros_like(opd_internal),
# # # # # #     zwfs_ns=zwfs_internal,
# # # # # #     detector=zwfs_internal.detector,
# # # # # #     include_shotnoise=False,
# # # # # # )

# # # # # # N0_internal = bldr.get_N0_configured(
# # # # # #     opd_input=opd_internal,
# # # # # #     amp_input=amp_internal,
# # # # # #     opd_internal=np.zeros_like(opd_internal),
# # # # # #     zwfs_ns=zwfs_internal,
# # # # # #     detector=zwfs_internal.detector,
# # # # # #     include_shotnoise=False,
# # # # # # )

# # # # # # I0_onsky = bldr.get_I0_configured(
# # # # # #     opd_input=opd_onsky,
# # # # # #     amp_input=amp_onsky,
# # # # # #     opd_internal=np.zeros_like(opd_onsky),
# # # # # #     zwfs_ns=zwfs_onsky,
# # # # # #     detector=zwfs_onsky.detector,
# # # # # #     include_shotnoise=False,
# # # # # # )

# # # # # # N0_onsky = bldr.get_N0_configured(
# # # # # #     opd_input=opd_onsky,
# # # # # #     amp_input=amp_onsky,
# # # # # #     opd_internal=np.zeros_like(opd_onsky),
# # # # # #     zwfs_ns=zwfs_onsky,
# # # # # #     detector=zwfs_onsky.detector,
# # # # # #     include_shotnoise=False,
# # # # # # )

# # # # # # # Detector-space pupil mask from the known input pupil.
# # # # # # binning = int(base_cfg["detector"]["binning"])
# # # # # # pupil_detector = (
# # # # # #     bldr.sum_subarrays(
# # # # # #         zwfs_internal.grid.pupil_mask,
# # # # # #         block_size=(binning, binning),
# # # # # #     )
# # # # # #     > 0.5 * binning**2
# # # # # # )

# # # # # # # Paper convention: divide by the mean clear-pupil intensity, not pixel-by-pixel N0.
# # # # # # I0_internal_norm = I0_internal / np.mean(N0_internal[pupil_detector])
# # # # # # I0_onsky_norm = I0_onsky / np.mean(N0_onsky[pupil_detector])

# # # # # # difference = I0_onsky_norm - I0_internal_norm
# # # # # # relative_difference_percent = np.zeros_like(difference)
# # # # # # relative_difference_percent[pupil_detector] = (
# # # # # #     100.0
# # # # # #     * difference[pupil_detector]
# # # # # #     / np.maximum(np.abs(I0_internal_norm[pupil_detector]), 1e-12)
# # # # # # )

# # # # # # rms_difference = np.sqrt(np.mean(difference[pupil_detector] ** 2))
# # # # # # mean_difference = np.mean(difference[pupil_detector])
# # # # # # max_abs_difference = np.max(np.abs(difference[pupil_detector]))

# # # # # # wavelength_um = zwfs_internal.spectrum.wavelengths * 1e6

# # # # # # fig, ax = plt.subplots(2, 3, figsize=(13, 8))

# # # # # # ax[0, 0].plot(
# # # # # #     wavelength_um,
# # # # # #     zwfs_internal.spectrum.weights_normalized,
# # # # # #     "o-",
# # # # # #     label=f"{T_INTERNAL_K:.0f} K",
# # # # # # )
# # # # # # ax[0, 0].plot(
# # # # # #     wavelength_um,
# # # # # #     zwfs_onsky.spectrum.weights_normalized,
# # # # # #     "o-",
# # # # # #     label=f"{T_ONSKY_K:.0f} K",
# # # # # # )
# # # # # # ax[0, 0].set_xlabel("Wavelength [µm]")
# # # # # # ax[0, 0].set_ylabel("Normalized photon weight")
# # # # # # ax[0, 0].set_title("Configured bandpass weights")
# # # # # # ax[0, 0].legend()

# # # # # # im = ax[0, 1].imshow(I0_internal_norm)
# # # # # # fig.colorbar(im, ax=ax[0, 1])
# # # # # # ax[0, 1].set_title(r"$I_0(1900\,\mathrm{K})/\langle N_0(1900\,\mathrm{K})\rangle_P$")

# # # # # # im = ax[0, 2].imshow(I0_onsky_norm)
# # # # # # fig.colorbar(im, ax=ax[0, 2])
# # # # # # ax[0, 2].set_title(r"$I_0(10000\,\mathrm{K})/\langle N_0(10000\,\mathrm{K})\rangle_P$")

# # # # # # limit = np.max(np.abs(difference[pupil_detector]))
# # # # # # im = ax[1, 0].imshow(difference, vmin=-limit, vmax=limit, cmap="RdBu_r")
# # # # # # fig.colorbar(im, ax=ax[1, 0])
# # # # # # ax[1, 0].set_title("10000 K − 1900 K")

# # # # # # limit_percent = np.nanpercentile(
# # # # # #     np.abs(relative_difference_percent[pupil_detector]), 99
# # # # # # )
# # # # # # im = ax[1, 1].imshow(
# # # # # #     relative_difference_percent,
# # # # # #     vmin=-limit_percent,
# # # # # #     vmax=limit_percent,
# # # # # #     cmap="RdBu_r",
# # # # # # )
# # # # # # fig.colorbar(im, ax=ax[1, 1], label="%")
# # # # # # ax[1, 1].set_title("Relative difference")

# # # # # # ax[1, 2].axis("off")
# # # # # # ax[1, 2].text(
# # # # # #     0.0,
# # # # # #     0.95,
# # # # # #     f"Mask: {mask_name}\n"
# # # # # #     f"Band: {wavelength_um.min():.3f}–{wavelength_um.max():.3f} µm\n"
# # # # # #     f"RMS difference in pupil: {rms_difference:.6g}\n"
# # # # # #     f"Mean difference in pupil: {mean_difference:.6g}\n"
# # # # # #     f"Max |difference| in pupil: {max_abs_difference:.6g}",
# # # # # #     va="top",
# # # # # #     family="monospace",
# # # # # # )

# # # # # # for a in ax.flat:
# # # # # #     if a is not ax[0, 0] and a is not ax[1, 2]:
# # # # # #         a.set_xticks([])
# # # # # #         a.set_yticks([])

# # # # # # plt.tight_layout()
# # # # # # plt.savefig(OUTPUT_FIGURE, dpi=200, bbox_inches="tight")
# # # # # # plt.show()

# # # # # # print(f"Saved: {OUTPUT_FIGURE.resolve()}")
# # # # # # print(f"RMS normalized-I0 difference in pupil: {rms_difference:.6g}")
# # # # # # print(f"Mean normalized-I0 difference in pupil: {mean_difference:.6g}")
# # # # # # print(f"Max absolute normalized-I0 difference in pupil: {max_abs_difference:.6g}")
