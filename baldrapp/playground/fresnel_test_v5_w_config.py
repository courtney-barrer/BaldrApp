from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

import aotools
import pyzelda.ztools as ztools

from baldrapp.common import baldr_core as bldr
from baldrapp.common import utilities as util
from baldrapp.common import phasescreens as ps


# ============================================================
# 0. User paths / config
# ============================================================

repo_root = Path("/Users/bencb/Documents/ASGARD/BaldrApp")
config_path = repo_root / "baldrapp/apps/paranal_simulator/fake_configs/baldr_config.json"

zwfs_ns = bldr.init_zwfs_from_json(config_path)

pupil = zwfs_ns.grid.pupil_mask
pupil_bool = pupil.astype(bool)
detector = zwfs_ns.detector

use_pyZelda = False
include_shotnoise = False

np.random.seed(2)


# ============================================================
# 1. Small helper functions
# ============================================================

def get_cfg_value(ns, name, default=None):
    return getattr(ns, name, default) if ns is not None else default


def upsample_by_factor(ar, f):
    ar = np.asarray(ar)
    fy, fx = (f, f) if isinstance(f, int) else f
    return np.repeat(np.repeat(ar, fy, axis=0), fx, axis=1)


def pad_to_shape_edge(arr, target_shape):
    m, n = arr.shape
    M, N = target_shape
    if M < m or N < n:
        raise ValueError("target_shape must be >= current shape.")
    return np.pad(arr, ((0, M - m), (0, N - n)), mode="edge")


def upsample_to_size(ar, target_size):
    ar = np.asarray(ar)
    if ar.shape[0] == target_size:
        return ar

    factor = target_size // ar.shape[0]
    if factor < 1:
        raise ValueError("This helper only upsamples, not down-samples.")

    out = upsample_by_factor(ar, factor)

    if out.shape[0] != target_size:
        out = pad_to_shape_edge(out, (target_size, target_size))

    return out


def update_scintillation(
    high_alt_phasescreen,
    pxl_scale,
    wavelength,
    final_size=None,
    jumps=1,
    propagation_distance=10000.0,
    renormalize_mean_intensity=True,
):
    """
    Generate amplitude scintillation from a high-altitude phase screen.

    Returns amplitude, not intensity. Therefore amp_input should be multiplied
    by this returned map.
    """
    for _ in range(int(jumps)):
        high_alt_phasescreen.add_row()

    wavefront = np.exp(1j * high_alt_phasescreen.scrn)

    propagated = aotools.opticalpropagation.angularSpectrum(
        inputComplexAmp=wavefront,
        z=propagation_distance,
        wvl=wavelength,
        inputSpacing=pxl_scale,
        outputSpacing=pxl_scale,
    )

    amp = np.abs(propagated)

    if final_size is not None and amp.shape[0] != final_size:
        amp = upsample_to_size(amp, final_size)

    if renormalize_mean_intensity:
        mean_I = np.mean(amp[pupil_bool] ** 2)
        if mean_I > 0:
            amp = amp / np.sqrt(mean_I)

    return amp


def rms_nm(opd, mask):
    return 1e9 * np.std(opd[mask])


def strehl_from_opd(opd, wavelength, mask):
    phase = 2.0 * np.pi / wavelength * opd[mask]
    return np.exp(-np.var(phase))


# ============================================================
# 2. Pull atmosphere / AO / source settings from config
# ============================================================

wvl0 = zwfs_ns.optics.wvl0
dx = zwfs_ns.grid.D / zwfs_ns.grid.N

atm_cfg = getattr(zwfs_ns, "atmosphere", SimpleNamespace())
phase_cfg = getattr(atm_cfg, "phase", SimpleNamespace())
scint_cfg = getattr(atm_cfg, "scintillation", SimpleNamespace())
ao_cfg = getattr(zwfs_ns, "first_stage_ao", SimpleNamespace())

# Phase turbulence. Config convention: r0_500_m at 500 nm.
r0_500_m = float(get_cfg_value(phase_cfg, "r0_500_m", 0.126))
L0_m = float(get_cfg_value(phase_cfg, "L0_m", 25.0))
reference_wavelength_m = float(get_cfg_value(phase_cfg, "reference_wavelength_m", 500e-9))

# Convert r0 to the simulation wavelength because the phase screen produces
# phase in radians for the chosen r0 convention.
r0_wvl_m = r0_500_m * (wvl0 / reference_wavelength_m) ** (6.0 / 5.0)

phase_seed = get_cfg_value(phase_cfg, "random_seed", 2)
phase_rows_per_frame = int(get_cfg_value(phase_cfg, "rows_per_frame", 1))
phase_scaling_factor = float(get_cfg_value(phase_cfg, "phase_scaling_factor", 1.0))

# Scintillation settings.
scint_enabled = bool(get_cfg_value(scint_cfg, "enabled", False))
scint_r0_500_m = float(get_cfg_value(scint_cfg, "r0_500_m", r0_500_m))
scint_L0_m = float(get_cfg_value(scint_cfg, "L0_m", L0_m))
scint_reference_wavelength_m = float(
    get_cfg_value(scint_cfg, "reference_wavelength_m", reference_wavelength_m)
)
scint_r0_wvl_m = scint_r0_500_m * (wvl0 / scint_reference_wavelength_m) ** (6.0 / 5.0)

scint_seed = get_cfg_value(scint_cfg, "random_seed", 3)
scint_rows_per_frame = int(get_cfg_value(scint_cfg, "rows_per_frame", 1))
propagation_distance_m = float(get_cfg_value(scint_cfg, "propagation_distance_m", 10000.0))
renormalize_scint = bool(get_cfg_value(scint_cfg, "renormalize_mean_intensity", True))

# First-stage AO settings.
ao_enabled = bool(get_cfg_value(ao_cfg, "enabled", True))
Nmodes_removed = int(get_cfg_value(ao_cfg, "Nmodes_removed", 14))
ao_phase_scaling_factor = float(get_cfg_value(ao_cfg, "phase_scaling_factor", 1.0))

# Source flux.
source_cfg = getattr(zwfs_ns, "source", SimpleNamespace())
photons_per_second_per_pixel_per_nm = float(
    get_cfg_value(source_cfg, "photons_per_second_per_pixel_per_nm", 1.0e5)
)

amp_input_0 = np.sqrt(photons_per_second_per_pixel_per_nm) * pupil


print("\n=== Config-derived simulation parameters ===")
print("wavelength [um]:", 1e6 * wvl0)
print("dx [m/pixel]:", dx)
print("r0_500_m:", r0_500_m)
print("r0 at wvl0 [m]:", r0_wvl_m)
print("L0 [m]:", L0_m)
print("AO enabled:", ao_enabled)
print("Nmodes_removed:", Nmodes_removed)
print("scintillation enabled:", scint_enabled)
print("scint propagation distance [m]:", propagation_distance_m)
print("photons/s/pixel/nm:", photons_per_second_per_pixel_per_nm)
print("spectrum wavelengths [um]:", zwfs_ns.spectrum.wavelengths * 1e6)
print("sum weights_nm:", np.sum(zwfs_ns.spectrum.weights_nm))
print("fresnel enabled:", zwfs_ns.fresnel_relay.enabled)


# ============================================================
# 3. Initialise phase and scintillation screens
# ============================================================

scrn = ps.PhaseScreenKolmogorov(
    nx_size=zwfs_ns.grid.dim,
    pixel_scale=dx,
    r0=r0_wvl_m,
    L0=L0_m,
    random_seed=phase_seed,
)

if scint_enabled:
    scint_screen = aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman(
        nx_size=zwfs_ns.grid.dim,
        pixel_scale=dx,
        r0=scint_r0_wvl_m,
        L0=scint_L0_m,
        random_seed=scint_seed,
    )
else:
    scint_screen = None


# ============================================================
# 4. Build first-stage AO Zernike basis
# ============================================================

# Add a few extra modes so removing 14 is safely covered.
nterms_basis = max(50, Nmodes_removed + 5)

basis_cropped = ztools.zernike.zernike_basis(
    nterms=nterms_basis,
    npix=zwfs_ns.grid.N,
)

basis_template = np.zeros_like(pupil)
basis = np.array([
    util.insert_concentric(np.nan_to_num(b, 0.0), basis_template)
    for b in basis_cropped
])

# basis[0] is the disk used inside bldr.first_stage_ao(...)
print("AO basis shape:", basis.shape)


# ============================================================
# 5. Reference frames with configured propagation
# ============================================================

opd_internal = np.zeros_like(pupil)

zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy()

I0 = bldr.get_I0_configured(
    opd_input=np.zeros_like(pupil),
    amp_input=amp_input_0,
    opd_internal=opd_internal,
    zwfs_ns=zwfs_ns,
    detector=detector,
    include_shotnoise=False,
    use_pyZelda=use_pyZelda,
)

N0 = bldr.get_N0_configured(
    opd_input=np.zeros_like(pupil),
    amp_input=amp_input_0,
    opd_internal=opd_internal,
    zwfs_ns=zwfs_ns,
    detector=detector,
    include_shotnoise=False,
    use_pyZelda=use_pyZelda,
)

print("\n=== Reference frames ===")
print("I0 shape/sum:", I0.shape, np.sum(I0))
print("N0 shape/sum:", N0.shape, np.sum(N0))
print("I0/N0 sum ratio:", np.sum(I0) / np.sum(N0))


# ============================================================
# 6. Main open-loop time sequence
# ============================================================

n_frames = 40

frames = []
signals = []
opd_raw_list = []
opd_ao_list = []
amp_scint_list = []
strehl_raw_list = []
strehl_ao_list = []
rms_raw_nm_list = []
rms_ao_nm_list = []

for k in range(n_frames):

    # Evolve ground-layer phase screen.
    for _ in range(phase_rows_per_frame):
        scrn.add_row()

    phase_raw = phase_scaling_factor * scrn.scrn

    if ao_enabled and Nmodes_removed > 0:
        phase_after_ao = bldr.first_stage_ao(
            atm_scrn=scrn,
            Nmodes_removed=Nmodes_removed,
            basis=basis,
            phase_scaling_factor=phase_scaling_factor * ao_phase_scaling_factor,
            return_reconstructor=False,
        )
    else:
        phase_after_ao = phase_raw * pupil

    # Convert phase [rad] to OPD [m] at wvl0.
    opd_raw = pupil * (wvl0 / (2.0 * np.pi)) * phase_raw
    opd_after_ao = pupil * (wvl0 / (2.0 * np.pi)) * phase_after_ao

    # Scintillation amplitude.
    if scint_enabled:
        amp_scint = update_scintillation(
            high_alt_phasescreen=scint_screen,
            pxl_scale=dx,
            wavelength=wvl0,
            final_size=zwfs_ns.grid.dim,
            jumps=scint_rows_per_frame,
            propagation_distance=propagation_distance_m,
            renormalize_mean_intensity=renormalize_scint,
        )
        amp_input = amp_input_0 * amp_scint
    else:
        amp_scint = np.ones_like(pupil)
        amp_input = amp_input_0

    # Keep Baldr DM flat for this open-loop WFS simulation.
    zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy()

    I = bldr.get_frame_configured(
        opd_input=opd_after_ao,
        amp_input=amp_input,
        opd_internal=opd_internal,
        zwfs_ns=zwfs_ns,
        detector=detector,
        include_shotnoise=include_shotnoise,
        use_pyZelda=use_pyZelda,
    ).astype(float)

    # Standard-ish normalized signal diagnostic.
    N0_mean = np.mean(N0)
    signal = I / (N0_mean + 1e-12) - I0 / (N0_mean + 1e-12)

    frames.append(I)
    signals.append(signal)
    opd_raw_list.append(opd_raw)
    opd_ao_list.append(opd_after_ao)
    amp_scint_list.append(amp_scint)

    strehl_raw_list.append(strehl_from_opd(opd_raw, wvl0, pupil_bool))
    strehl_ao_list.append(strehl_from_opd(opd_after_ao, wvl0, pupil_bool))
    rms_raw_nm_list.append(rms_nm(opd_raw, pupil_bool))
    rms_ao_nm_list.append(rms_nm(opd_after_ao, pupil_bool))

    if k % 5 == 0:
        print(
            f"{k:03d}/{n_frames}: "
            f"raw rms={rms_raw_nm_list[-1]:.1f} nm, "
            f"AO rms={rms_ao_nm_list[-1]:.1f} nm, "
            f"Sraw={strehl_raw_list[-1]:.3f}, "
            f"Sao={strehl_ao_list[-1]:.3f}, "
            f"I sum={np.sum(I):.3e}"
        )

frames = np.asarray(frames)
signals = np.asarray(signals)
opd_raw_list = np.asarray(opd_raw_list)
opd_ao_list = np.asarray(opd_ao_list)
amp_scint_list = np.asarray(amp_scint_list)

print("\n=== Sequence complete ===")
print("frames shape:", frames.shape)
print("signals shape:", signals.shape)
print("mean frame sum:", np.mean(np.sum(frames, axis=(1, 2))))
print("raw rms nm median:", np.median(rms_raw_nm_list))
print("AO rms nm median:", np.median(rms_ao_nm_list))


# ============================================================
# 7. Summary plots
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

idx = min(n_frames // 2, n_frames - 1)

plot_items = [
    ("Scintillation amplitude", amp_scint_list[idx]),
    ("Raw phase OPD [nm]", 1e9 * opd_raw_list[idx]),
    ("After first-stage AO OPD [nm]", 1e9 * opd_ao_list[idx]),
    ("Detected frame", frames[idx]),
    ("Reference I0", I0),
    ("Signal: I/<N0> - I0/<N0>", signals[idx]),
]

for ax, (title, im) in zip(axes.ravel(), plot_items):
    im = np.asarray(im, dtype=float)

    if "OPD" in title or "Signal" in title:
        lim = np.nanpercentile(np.abs(im), 99)
        lim = max(lim, 1e-12)
        h = ax.imshow(im, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    else:
        p1, p99 = np.nanpercentile(im, [1, 99])
        h = ax.imshow(im, origin="lower", vmin=p1, vmax=p99)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(h, ax=ax, fraction=0.046)

fig.suptitle(f"Kolmogorov + {Nmodes_removed}-mode AO + scintillation + poly-Fresnel frame {idx}", fontsize=14)
plt.tight_layout()
plt.show()


# ============================================================
# 8. Time-series diagnostics
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

axes[0].plot(rms_raw_nm_list, label="raw atmosphere")
axes[0].plot(rms_ao_nm_list, label=f"after {Nmodes_removed}-mode AO")
axes[0].set_ylabel("OPD RMS [nm]")
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].plot(strehl_raw_list, label="raw atmosphere")
axes[1].plot(strehl_ao_list, label=f"after {Nmodes_removed}-mode AO")
axes[1].set_ylabel("Maréchal Strehl")
axes[1].grid(alpha=0.3)
axes[1].legend()

frame_sums = np.sum(frames, axis=(1, 2))
axes[2].plot(frame_sums / np.mean(frame_sums), label="frame sum / mean")
axes[2].set_xlabel("frame index")
axes[2].set_ylabel("relative flux")
axes[2].grid(alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.show()


# ============================================================
# 9. Optional slider display using your utility module
# ============================================================

try:
    util.display_images_with_slider(
        image_lists=[
            list(amp_scint_list),
            list(1e9 * opd_ao_list),
            list(frames),
            list(signals),
        ],
        plot_titles=[
            "scintillation amplitude",
            f"OPD after {Nmodes_removed}-mode AO [nm]",
            "detected poly-Fresnel frame",
            "normalized ZWFS signal",
        ],
        cbar_labels=[
            "amplitude",
            "nm",
            "detected counts",
            "normalized signal",
        ],
    )
except Exception as err:
    print("Slider display failed, but simulation completed.")
    print("Error:", repr(err))