#!/usr/bin/env python3
"""
Fit Baldr Fresnel-relay alignment parameters from a synthetic DM-probe cube.

Known quantities
----------------
- source temperature selected from a small LUT;
- spectral bandwidth and wavelength sampling from baldr_config.json;
- pupil, phase-mask properties, detector, DM model, and DM commands.

Fitted quantities
-----------------
1. D-mirror edge offset
2. D-mirror edge angle
3. cold-stop x offset
4. cold-stop y offset
5. pupil misconjugation

The script first generates noise-free or optionally noisy synthetic data at a
known parameter vector, then recovers those parameters with bounded nonlinear
least squares.

The code intentionally stays sequential. Only the forward-cube evaluator,
residual, and finite-difference Jacobian are factored into functions because
the optimiser must call them repeatedly.
"""

import copy
import json
import platform
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.optimize import least_squares


# ============================================================
# Use the current BaldrApp repository, not an installed package
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from baldrapp.common import DM_basis
from baldrapp.common import baldr_core as bldr
from baldrapp.common import spectrum as spec

for module in (bldr, spec, DM_basis):
    module_path = Path(module.__file__).resolve()

    if REPO_ROOT not in module_path.parents:
        raise ImportError(
            "BaldrApp import shadowing detected.\n"
            f"Expected modules below: {REPO_ROOT}\n"
            f"Imported instead: {module_path}"
        )

print(f"Python executable: {Path(sys.executable).resolve()}")
print(f"Python version: {platform.python_version()}")
print(f"baldr_core: {Path(bldr.__file__).resolve()}")
print(f"spectrum: {Path(spec.__file__).resolve()}")
print(f"DM_basis: {Path(DM_basis.__file__).resolve()}")


# ============================================================
# User settings
# ============================================================

CONFIG_PATH = (
    REPO_ROOT
    / "baldrapp/apps/paranal_simulator/fake_configs/baldr_config.json"
)

OUTPUT_DIRECTORY = Path("baldr_relay_parameter_fit")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

STAR_TEMPERATURE_LUT_K = {
    "internal": 1900.0,
    "M2V": 3500.0,
    "K5V": 4400.0,
    "G2V": 5770.0,
    "A0V": 9500.0,
}

SOURCE_TYPE = "K5V"
SOURCE_TEMPERATURE_K = STAR_TEMPERATURE_LUT_K[SOURCE_TYPE]

PHASE_MASK_NAME = "H3"

N_LOW_ORDER_PROBES = 4
N_FOURIER_PROBES = 2
PROBE_RMS_NM = 50.0

# Synthetic truth. The fit vector uses:
# [edge offset mm, edge angle deg, cold-stop x um,
#  cold-stop y um, pupil misconjugation mm]
TRUE_PARAMETERS = np.array(
    [
        -0.80,
        1.50,
        50.0,
        -35.0,
        6.0,
    ],
    dtype=float,
)

INITIAL_PARAMETERS = np.array(
    [
        -1.00,
        0.00,
        0.0,
        0.0,
        0.0,
    ],
    dtype=float,
)

LOWER_BOUNDS = np.array(
    [
        -1.35,
        -5.0,
        -180.0,
        -180.0,
        -20.0,
    ],
    dtype=float,
)

UPPER_BOUNDS = np.array(
    [
        -0.35,
        5.0,
        180.0,
        180.0,
        20.0,
    ],
    dtype=float,
)

# Absolute finite-difference steps in the same units as the fit vector.
# The D-edge and cold stop are binary masks in the current forward model, so
# these steps are deliberately comparable to their sampled physical pixel size.
JACOBIAN_STEPS = np.array(
    [
        0.040,
        0.25,
        40.0,
        40.0,
        1.0,
    ],
    dtype=float,
)

MAX_FUNCTION_EVALUATIONS = 30

# Set to zero for an exact synthetic recovery test.
# A value such as 2e-3 adds Gaussian noise with RMS equal to 0.2 per cent of
# the median normalized pupil intensity.
MEASUREMENT_NOISE_FRACTION = 0.0
RANDOM_SEED = 5

# Scale used only to keep least-squares residuals near order unity.
FIT_SIGMA_NORMALIZED = 2e-3


# ============================================================
# Load and fix all known optical and spectral quantities
# ============================================================

with open(CONFIG_PATH, "r") as f:
    base_cfg = json.load(f)

cfg = copy.deepcopy(base_cfg)

cfg["stellar"]["spectrum"]["enabled"] = True
cfg["stellar"]["spectrum"]["mode"] = "blackbody"
cfg["stellar"]["spectrum"]["temperature_K"] = SOURCE_TEMPERATURE_K

# The bandwidth, wavelength range, wavelength sampling, pupil, mask depth,
# detector, and DM model remain fixed at their configured values.

cfg["internal_aberrations"]["enabled"] = False
cfg["atmosphere"]["enabled"] = False
cfg["first_stage_ao"]["enabled"] = False

cfg["dm"]["flat_rmse"] = 0.0

cfg["detector"]["enabled"] = True
cfg["detector"]["include_shotnoise"] = False
cfg["detector"]["include_readnoise"] = False
cfg["detector"]["ron"] = 0.0
cfg["detector"]["adu_offset"] = 0.0
cfg["detector"]["noise_std_adu"] = 0.0

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    true_config_path = tmp / "synthetic_truth.json"
    fit_config_path = tmp / "fit_model.json"

    with open(true_config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    with open(fit_config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    zwfs_true = bldr.init_zwfs_from_json(true_config_path)
    zwfs_fit = bldr.init_zwfs_from_json(fit_config_path)


# ============================================================
# Use one known physical phase mask
# ============================================================

phasemask_path = (
    REPO_ROOT
    / base_cfg["simulator_runtime"]["phasemask"]["properties_file"]
)

with open(phasemask_path, "r") as f:
    phasemask_cfg = json.load(f)

mask_entry = phasemask_cfg["phasemask"]["masks"][PHASE_MASK_NAME]

zwfs_true.optics.active_phasemask = spec.normalise_phasemask_entry(
    PHASE_MASK_NAME,
    mask_entry,
    zwfs_true.optics,
)

zwfs_fit.optics.active_phasemask = spec.normalise_phasemask_entry(
    PHASE_MASK_NAME,
    mask_entry,
    zwfs_fit.optics,
)

print()
print(f"Source profile: {SOURCE_TYPE}")
print(f"Source temperature: {SOURCE_TEMPERATURE_K:.1f} K")
print(
    "Wavelengths [um]:",
    np.asarray(zwfs_true.spectrum.wavelengths) * 1e6,
)
print(
    "Spectral integration weights [nm]:",
    np.asarray(zwfs_true.spectrum.weights_nm),
)
print(f"Phase mask: {PHASE_MASK_NAME}")


# ============================================================
# Fixed pupil amplitude, zero external OPD, and zero internal OPD
# ============================================================

flux_density = float(
    cfg["source"]["photons_per_second_per_pixel_per_nm"]
)

amp_input = (
    np.sqrt(flux_density)
    * zwfs_true.grid.pupil_mask.astype(float)
)

zero_opd = np.zeros_like(amp_input)

binning = int(cfg["detector"]["binning"])

pupil_detector = (
    bldr.sum_subarrays(
        zwfs_true.grid.pupil_mask,
        block_size=(binning, binning),
    )
    > 0.5 * binning**2
)

normalization_mask = binary_erosion(
    pupil_detector,
    iterations=1,
)

if not np.any(normalization_mask):
    raise RuntimeError("The normalization pupil mask is empty.")


# ============================================================
# Build a fixed sequence of real DM commands
# ============================================================

low_order_basis = DM_basis.construct_command_basis(
    basis="Zernike_pinned_edges",
    number_of_modes=N_LOW_ORDER_PROBES,
    Nx_act_DM=12,
    Nx_act_basis=12,
    without_piston=True,
)

fourier_basis = DM_basis.construct_command_basis(
    basis="fourier_pinned_edges",
    number_of_modes=max(N_FOURIER_PROBES, 4),
    Nx_act_DM=12,
    Nx_act_basis=12,
    without_piston=True,
)

probe_basis = np.column_stack(
    (
        low_order_basis[:, :N_LOW_ORDER_PROBES],
        fourier_basis[:, :N_FOURIER_PROBES],
    )
)

command_vectors = [zwfs_true.dm.dm_flat.copy()]
command_labels = ["flat"]

for probe_index in range(probe_basis.shape[1]):
    raw_command_mode = probe_basis[:, probe_index]

    opd_mode = bldr.get_dm_displacement(
        command_vector=raw_command_mode,
        gain=zwfs_true.dm.opd_per_cmd,
        sigma=zwfs_true.grid.dm_coord.act_sigma_wavesp,
        X=zwfs_true.grid.wave_coord.X,
        Y=zwfs_true.grid.wave_coord.Y,
        x0=zwfs_true.grid.dm_coord.act_x0_list_wavesp,
        y0=zwfs_true.grid.dm_coord.act_y0_list_wavesp,
    )

    opd_mode_values = opd_mode[
        zwfs_true.grid.pupil_mask.astype(bool)
    ]

    opd_mode_values = (
        opd_mode_values - np.mean(opd_mode_values)
    )

    mode_rms_m = np.sqrt(
        np.mean(opd_mode_values**2)
    )

    if mode_rms_m <= 0:
        raise RuntimeError(
            f"Probe mode {probe_index} has zero OPD RMS."
        )

    command_scale = PROBE_RMS_NM * 1e-9 / mode_rms_m

    command_vectors.append(
        zwfs_true.dm.dm_flat
        + command_scale * raw_command_mode
    )

    command_vectors.append(
        zwfs_true.dm.dm_flat
        - command_scale * raw_command_mode
    )

    if probe_index < N_LOW_ORDER_PROBES:
        probe_name = f"zernike_{probe_index + 1}"
    else:
        probe_name = (
            f"fourier_"
            f"{probe_index - N_LOW_ORDER_PROBES + 1}"
        )

    command_labels.extend(
        [f"+{probe_name}", f"-{probe_name}"]
    )

command_vectors = np.asarray(command_vectors)

print()
print(f"Number of phase-mask-in frames: {len(command_vectors)}")
print(f"Probe RMS: {PROBE_RMS_NM:.1f} nm OPD")


# ============================================================
# Forward evaluator used for both truth and fitting
# ============================================================

def evaluate_normalized_cube(zwfs, parameters):
    edge_offset_mm = float(parameters[0])
    edge_angle_deg = float(parameters[1])
    coldstop_x_um = float(parameters[2])
    coldstop_y_um = float(parameters[3])
    pupil_misconjugation_mm = float(parameters[4])

    fr = zwfs.fresnel_relay

    fr.edge_offset = edge_offset_mm * 1e-3
    fr.edge_angle = np.deg2rad(edge_angle_deg)
    fr.coldstop_x_offset = coldstop_x_um * 1e-6
    fr.coldstop_y_offset = coldstop_y_um * 1e-6

    fr.pupil_misconjugation = (
        pupil_misconjugation_mm * 1e-3
    )

    fr.z_focus_to_detector = (
        fr.z_focus_to_detector_nominal
        + fr.pupil_misconjugation
    )

    if fr.z_focus_to_detector <= 0:
        raise ValueError(
            "The fitted pupil misconjugation places the detector "
            "before the cold-stop plane."
        )

    zwfs.dm.current_cmd = zwfs.dm.dm_flat.copy()

    N0 = bldr.get_N0_configured(
        opd_input=zero_opd,
        amp_input=amp_input,
        opd_internal=zero_opd,
        zwfs_ns=zwfs,
        detector=zwfs.detector,
        include_shotnoise=False,
        force_fresnel=True,
        force_polychromatic=True,
    )

    normalization = np.mean(
        N0[normalization_mask]
    )

    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("Invalid model N0 normalization.")

    normalized_frames = [N0 / normalization]

    for command_vector in command_vectors:
        zwfs.dm.current_cmd = command_vector.copy()

        frame = bldr.get_frame_configured(
            opd_input=zero_opd,
            amp_input=amp_input,
            opd_internal=zero_opd,
            zwfs_ns=zwfs,
            detector=zwfs.detector,
            include_shotnoise=False,
            force_fresnel=True,
            force_polychromatic=True,
        )

        normalized_frames.append(
            frame / normalization
        )

    zwfs.dm.current_cmd = zwfs.dm.dm_flat.copy()

    return np.asarray(normalized_frames)


# ============================================================
# Generate synthetic measurements at the known true parameters
# ============================================================

print()
print("Generating synthetic measurement cube...")

measured_cube = evaluate_normalized_cube(
    zwfs_true,
    TRUE_PARAMETERS,
)

# Include the illuminated pupil and a small surrounding boundary region.
support_mask = (
    (measured_cube[0] > 5e-3 * np.max(measured_cube[0]))
    | (measured_cube[1] > 5e-3 * np.max(measured_cube[1]))
)

fit_mask = binary_dilation(
    support_mask,
    iterations=2,
)

rng = np.random.default_rng(RANDOM_SEED)

if MEASUREMENT_NOISE_FRACTION > 0:
    reference_level = np.median(
        measured_cube[0][normalization_mask]
    )

    measurement_noise_rms = (
        MEASUREMENT_NOISE_FRACTION * reference_level
    )

    measured_cube = (
        measured_cube
        + rng.normal(
            scale=measurement_noise_rms,
            size=measured_cube.shape,
        )
    )
else:
    measurement_noise_rms = 0.0

print(f"Detector image shape: {measured_cube.shape[1:]}")
print(f"Fitted detector pixels per frame: {np.sum(fit_mask)}")
print(
    "Synthetic normalized-intensity noise RMS:",
    measurement_noise_rms,
)


# ============================================================
# Residual and finite-difference Jacobian
# ============================================================

def residual_vector(parameters):
    model_cube = evaluate_normalized_cube(
        zwfs_fit,
        parameters,
    )

    return (
        (
            model_cube[:, fit_mask]
            - measured_cube[:, fit_mask]
        )
        / FIT_SIGMA_NORMALIZED
    ).reshape(-1)


def finite_difference_jacobian(parameters):
    residual_at_parameters = residual_vector(parameters)

    jacobian = np.zeros(
        (
            residual_at_parameters.size,
            len(parameters),
        ),
        dtype=float,
    )

    for parameter_index in range(len(parameters)):
        trial = parameters.copy()

        step = JACOBIAN_STEPS[parameter_index]

        if (
            parameters[parameter_index] + step
            <= UPPER_BOUNDS[parameter_index]
        ):
            trial[parameter_index] += step
        else:
            trial[parameter_index] -= step

        actual_step = (
            trial[parameter_index]
            - parameters[parameter_index]
        )

        jacobian[:, parameter_index] = (
            residual_vector(trial)
            - residual_at_parameters
        ) / actual_step

    return jacobian


# ============================================================
# Fit the alignment parameters
# ============================================================

print()
print("Parameter order:")
print(
    "[edge offset mm, edge angle deg, cold-stop x um, "
    "cold-stop y um, pupil misconjugation mm]"
)
print("Truth:       ", TRUE_PARAMETERS)
print("Initial:     ", INITIAL_PARAMETERS)
print("Lower bound:", LOWER_BOUNDS)
print("Upper bound:", UPPER_BOUNDS)
print()
print("Starting nonlinear fit...")

fit_result = least_squares(
    residual_vector,
    INITIAL_PARAMETERS,
    jac=finite_difference_jacobian,
    bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
    max_nfev=MAX_FUNCTION_EVALUATIONS,
    x_scale=np.maximum(
        np.abs(JACOBIAN_STEPS),
        1e-12,
    ),
    ftol=1e-8,
    xtol=1e-8,
    gtol=1e-8,
    verbose=2,
)

fitted_parameters = fit_result.x

initial_cube = evaluate_normalized_cube(
    zwfs_fit,
    INITIAL_PARAMETERS,
)

fitted_cube = evaluate_normalized_cube(
    zwfs_fit,
    fitted_parameters,
)

initial_frame_rms = np.sqrt(
    np.mean(
        (
            initial_cube[:, fit_mask]
            - measured_cube[:, fit_mask]
        )
        ** 2,
        axis=1,
    )
)

fitted_frame_rms = np.sqrt(
    np.mean(
        (
            fitted_cube[:, fit_mask]
            - measured_cube[:, fit_mask]
        )
        ** 2,
        axis=1,
    )
)


# ============================================================
# Print and save parameter recovery
# ============================================================

parameter_names = [
    "edge_offset_mm",
    "edge_angle_deg",
    "coldstop_x_um",
    "coldstop_y_um",
    "pupil_misconjugation_mm",
]

print()
print(
    f"{'Parameter':28s} "
    f"{'True':>12s} "
    f"{'Initial':>12s} "
    f"{'Fitted':>12s} "
    f"{'Error':>12s}"
)

for name, truth, initial, fitted in zip(
    parameter_names,
    TRUE_PARAMETERS,
    INITIAL_PARAMETERS,
    fitted_parameters,
):
    print(
        f"{name:28s} "
        f"{truth:12.5g} "
        f"{initial:12.5g} "
        f"{fitted:12.5g} "
        f"{fitted - truth:12.5g}"
    )

summary = {
    "source_type": SOURCE_TYPE,
    "source_temperature_K": SOURCE_TEMPERATURE_K,
    "phase_mask": PHASE_MASK_NAME,
    "probe_rms_nm": PROBE_RMS_NM,
    "command_labels": command_labels,
    "parameter_names": parameter_names,
    "true_parameters": TRUE_PARAMETERS.tolist(),
    "initial_parameters": INITIAL_PARAMETERS.tolist(),
    "fitted_parameters": fitted_parameters.tolist(),
    "fit_error": (
        fitted_parameters - TRUE_PARAMETERS
    ).tolist(),
    "lower_bounds": LOWER_BOUNDS.tolist(),
    "upper_bounds": UPPER_BOUNDS.tolist(),
    "jacobian_steps": JACOBIAN_STEPS.tolist(),
    "measurement_noise_fraction": (
        MEASUREMENT_NOISE_FRACTION
    ),
    "fit_success": bool(fit_result.success),
    "fit_status": int(fit_result.status),
    "fit_message": str(fit_result.message),
    "fit_cost": float(fit_result.cost),
    "fit_optimality": float(fit_result.optimality),
    "fit_nfev": int(fit_result.nfev),
    "fit_njev": (
        int(fit_result.njev)
        if fit_result.njev is not None
        else None
    ),
    "initial_global_rms": float(
        np.sqrt(
            np.mean(
                (
                    initial_cube[:, fit_mask]
                    - measured_cube[:, fit_mask]
                )
                ** 2
            )
        )
    ),
    "fitted_global_rms": float(
        np.sqrt(
            np.mean(
                (
                    fitted_cube[:, fit_mask]
                    - measured_cube[:, fit_mask]
                )
                ** 2
            )
        )
    ),
}

with open(
    OUTPUT_DIRECTORY / "fit_summary.json",
    "w",
) as f:
    json.dump(summary, f, indent=2)

np.savez_compressed(
    OUTPUT_DIRECTORY / "fit_products.npz",
    measured_cube=measured_cube,
    initial_cube=initial_cube,
    fitted_cube=fitted_cube,
    fit_mask=fit_mask,
    normalization_mask=normalization_mask,
    command_vectors=command_vectors,
    command_labels=np.asarray(command_labels),
    true_parameters=TRUE_PARAMETERS,
    initial_parameters=INITIAL_PARAMETERS,
    fitted_parameters=fitted_parameters,
    initial_frame_rms=initial_frame_rms,
    fitted_frame_rms=fitted_frame_rms,
)


# ============================================================
# Diagnostic figures
# ============================================================

x = np.arange(len(parameter_names))

fig, ax = plt.subplots(figsize=(10, 4.8))

ax.plot(
    x,
    TRUE_PARAMETERS,
    "o-",
    label="True",
)

ax.plot(
    x,
    INITIAL_PARAMETERS,
    "s--",
    label="Initial",
)

ax.plot(
    x,
    fitted_parameters,
    "d-",
    label="Fitted",
)

ax.set_xticks(
    x,
    [
        "D-edge\n[mm]",
        "D-edge angle\n[deg]",
        "Cold stop x\n[um]",
        "Cold stop y\n[um]",
        "Misconjugation\n[mm]",
    ],
)

ax.set_ylabel("Parameter value in displayed units")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(
    OUTPUT_DIRECTORY / "parameter_recovery.png",
    dpi=220,
)
plt.close(fig)


fig, ax = plt.subplots(figsize=(10, 4.5))

frame_indices = np.arange(len(command_labels) + 1)

ax.semilogy(
    frame_indices,
    initial_frame_rms,
    "o--",
    label="Initial model",
)

ax.semilogy(
    frame_indices,
    fitted_frame_rms,
    "o-",
    label="Fitted model",
)

ax.set_xticks(
    frame_indices,
    ["N0"] + command_labels,
    rotation=45,
    ha="right",
)

ax.set_ylabel("Normalized-intensity residual RMS")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(
    OUTPUT_DIRECTORY / "residual_rms_by_frame.png",
    dpi=220,
)
plt.close(fig)


display_indices = [
    0,
    1,
    min(2, measured_cube.shape[0] - 1),
]

display_labels = [
    "N0",
    "I0 flat",
    command_labels[1]
    if len(command_labels) > 1
    else "probe",
]

fig, axes = plt.subplots(
    len(display_indices),
    3,
    figsize=(10.5, 9.0),
)

for row, (frame_index, frame_label) in enumerate(
    zip(display_indices, display_labels)
):
    measured_image = measured_cube[frame_index]
    fitted_image = fitted_cube[frame_index]
    residual_image = fitted_image - measured_image

    common_min = min(
        np.min(measured_image),
        np.min(fitted_image),
    )

    common_max = max(
        np.max(measured_image),
        np.max(fitted_image),
    )

    im0 = axes[row, 0].imshow(
        measured_image,
        origin="lower",
        vmin=common_min,
        vmax=common_max,
    )

    axes[row, 1].imshow(
        fitted_image,
        origin="lower",
        vmin=common_min,
        vmax=common_max,
    )

    residual_limit = np.max(
        np.abs(residual_image)
    )

    im2 = axes[row, 2].imshow(
        residual_image,
        origin="lower",
        cmap="RdBu_r",
        vmin=-residual_limit,
        vmax=residual_limit,
    )

    axes[row, 0].set_ylabel(frame_label)
    axes[row, 0].set_title("Synthetic measurement")
    axes[row, 1].set_title("Best-fit model")
    axes[row, 2].set_title("Model - measurement")

    fig.colorbar(
        im0,
        ax=axes[row, :2],
        shrink=0.75,
    )

    fig.colorbar(
        im2,
        ax=axes[row, 2],
        shrink=0.75,
    )

for axis in axes.flat:
    axis.set_xticks([])
    axis.set_yticks([])

fig.tight_layout()
fig.savefig(
    OUTPUT_DIRECTORY / "measured_fitted_residual_frames.png",
    dpi=220,
)
plt.close(fig)


print()
print(f"Saved outputs to: {OUTPUT_DIRECTORY.resolve()}")
print(
    "Note: recovery precision for the D-edge and cold-stop offsets is "
    "limited by the current hard binary aperture sampling. The deliberately "
    "finite Jacobian steps prevent the optimiser from seeing a zero derivative."
)
