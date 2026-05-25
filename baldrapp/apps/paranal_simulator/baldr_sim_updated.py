

# ### TO TEST 

## Run ./shm_creator_sim first to create the shm in the right format.
# Do not pass data here otherwise it will overwrite/recreate with the Python
# shmlib.py wrapper format, which may differ from the RTC-required format.

import numpy as np
import json
import zmq
import time
import os
from xaosim.shmlib import shm
import subprocess
from pathlib import Path

from baldrapp.common import baldr_core as bldr
from baldrapp.common import utilities as util
from baldrapp.common import phasescreens as ps
import pyzelda.ztools as ztools

try:
    import aotools
except ImportError:
    aotools = None


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


def convert_12x12_to_140(arr):
    """
    Convert a 12x12 DM command image with four unused corners into the
    140-element BMC multi-3.5 command vector.
    """
    arr = np.asarray(arr)

    if arr.shape != (12, 12):
        raise ValueError("Input must be a 12x12 array.")

    flat = arr.flatten()
    corner_indices = [0, 11, 132, 143]
    return np.delete(flat, corner_indices)


def get_cfg_value(ns, name, default=None):
    return getattr(ns, name, default) if ns is not None else default


def get_photon_flux_density_from_config(zwfs):
    """
    Return photons / second / wave-space-pixel / nm for the simulator source.

    Preferred new JSON config field:
        zwfs.source.photons_per_second_per_pixel_per_nm

    Fallback:
        1.0e5 photons / s / pixel / nm

    This keeps the simulator independent of the old pyZELDA/throughput/magnitude
    calculation path.
    """
    source_cfg = getattr(zwfs, "source", None)

    return float(
        get_cfg_value(
            source_cfg,
            "photons_per_second_per_pixel_per_nm",
            1.0e5,
        )
    )


def get_internal_opd_from_config(zwfs):
    """
    Build static/internal OPD from the new JSON config.

    Currently supports:
        internal_aberrations.parabolic_scratches

    If no internal-aberrations section is present, returns zeros.
    """
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


def set_phase_mask_state(zwfs, mask_inserted, original_optics_state):
    """
    Apply phase-mask in/out state to the configured analytic propagation path.

    For mask out, force theta=0 and theta_mode='constant'. This is robust even
    when the configured mask normally uses theta_mode='physical_depth'.
    """
    if mask_inserted:
        zwfs.optics.theta = original_optics_state["theta"]
        zwfs.optics.theta_mode = original_optics_state["theta_mode"]
    else:
        zwfs.optics.theta = 0.0
        zwfs.optics.theta_mode = "constant"


def read_dm_command(dm_shm):
    """
    Read the combined 12x12 simulated DM shared-memory image and convert it to
    the Baldr 140-vector command.

    The simulator currently treats this shared-memory command as the full DM
    command used by the optical model, preserving the behaviour of the previous
    script.
    """
    dmcmd_2d = dm_shm.get_data()
    return convert_12x12_to_140(dmcmd_2d)


def scale_r0_to_wavelength(r0_ref_m, wavelength_m, reference_wavelength_m):
    """Scale Fried parameter from reference_wavelength_m to wavelength_m."""
    return float(r0_ref_m) * (float(wavelength_m) / float(reference_wavelength_m)) ** (6.0 / 5.0)


def make_first_stage_ao_basis(zwfs, n_modes_removed):
    """Build a Zernike basis on the BaldrApp wave-space grid for first-stage AO."""
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
    """Nearest-neighbour upsample of a square 2D array by an integer factor."""
    arr = np.asarray(arr)
    return np.repeat(np.repeat(arr, int(factor), axis=0), int(factor), axis=1)


def upsample_to_size(arr, target_size):
    """Nearest-neighbour upsample/pad a square 2D array to target_size."""
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


def update_scintillation_amplitude(
    scint_screen,
    zwfs,
    pxl_scale,
    wavelength_m,
    rows_per_frame=1,
    propagation_distance_m=10000.0,
    renormalize_mean_intensity=True,
):
    """
    Evolve a high-altitude phase screen and return scintillation amplitude.

    The returned map is amplitude, not intensity, so the source amplitude is
    multiplied by this result before calling bldr.get_frame_configured(...).
    """
    if aotools is None:
        raise ImportError(
            "aotools is required for scintillation.enabled=true in the simulator."
        )

    for _ in range(int(rows_per_frame)):
        scint_screen.add_row()

    wavefront = np.exp(1j * scint_screen.scrn)

    propagated = aotools.opticalpropagation.angularSpectrum(
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
    """
    Initialise Kolmogorov phase, first-stage AO, and scintillation state from
    the JSON-derived zwfs namespace.
    """
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
        if aotools is None:
            raise ImportError(
                "aotools is required for scintillation.enabled=true in the simulator."
            )

        scint_r0_500_m = float(get_cfg_value(scint_cfg, "r0_500_m", 0.126))
        scint_ref_wvl_m = float(
            get_cfg_value(scint_cfg, "reference_wavelength_m", 500e-9)
        )
        scint_r0_wvl_m = scale_r0_to_wavelength(
            scint_r0_500_m,
            wvl0,
            scint_ref_wvl_m,
        )

        scint_screen = aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman(
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


def generate_atmosphere_and_source_for_beam(zwfs, amp_input_0, dynamic_state):
    """
    Evolve configured atmosphere/scintillation and return:
        opd_input : OPD in metres after optional first-stage AO
        amp_input : source amplitude after optional scintillation

    The Baldr DM is not included here. bldr.get_frame_configured(...) adds the
    current DM command internally.
    """
    pupil = zwfs.grid.pupil_mask
    wvl0 = float(zwfs.optics.wvl0)

    if dynamic_state["phase_enabled"]:
        scrn = dynamic_state["phase_screen"]
        for _ in range(int(dynamic_state["phase_rows_per_frame"])):
            scrn.add_row()

        phase_scaling = float(dynamic_state["phase_scaling_factor"])

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
            phase_after_ao = pupil * phase_scaling * scrn.scrn

        opd_input = pupil * (wvl0 / (2.0 * np.pi)) * phase_after_ao
    else:
        opd_input = np.zeros_like(pupil, dtype=float)

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
        amp_input = amp_input_0

    return opd_input, amp_input


# ============================================================
# Configuration / initialisation
# ============================================================

root_path = get_git_root()

# New generic BaldrApp JSON config. This should include:
#   grid, optics, dm, stellar.spectrum, fresnel_relay, detector, source, etc.
config_path = (
    root_path
    / "baldrapp/apps/paranal_simulator/fake_configs/baldr_config.json"
)

# Initialise one independent ZWFS namespace per beam.
# The configured frame dispatcher will automatically use:
#   - polychromatic propagation if zwfs.spectrum.enabled and n_wvl > 1
#   - Fresnel relay propagation if zwfs.fresnel_relay.enabled is True
zwfs_ns = {beam: None for beam in [1, 2, 3, 4]}
amp_input = {}
opd_internal = {}
dynamic_atmosphere_state = {}
original_optics_state = {}

for beam in [1, 2, 3, 4]:
    zwfs = bldr.init_zwfs_from_json(config_path)

    # Optional per-simulator override retained from the old script.
    zwfs.dm.actuator_coupling_factor = 0.9

    # Recompute DM registration after changing actuator coupling factor.
    # This keeps act_sigma_wavesp consistent with the updated coupling.
    zwfs = bldr.update_dm_registration_wavespace(
        zwfs.dm2wavespace_registration.dm_to_wavesp_transform_matrix,
        zwfs,
    )

    flux_density = get_photon_flux_density_from_config(zwfs)
    amp_input[beam] = np.sqrt(flux_density) * zwfs.grid.pupil_mask

    opd_internal[beam] = get_internal_opd_from_config(zwfs)

    original_optics_state[beam] = {
        "theta": float(zwfs.optics.theta),
        "theta_mode": str(getattr(zwfs.optics, "theta_mode", "constant")),
    }

    dynamic_atmosphere_state[beam] = init_dynamic_atmosphere_for_beam(zwfs)

    zwfs_ns[beam] = zwfs


default_tel = 1
use_pyZelda = False

print("\n=== Baldr simulator optical configuration ===")
print("config_path:", config_path)
print("grid N:", zwfs_ns[default_tel].grid.N)
print("grid dim:", zwfs_ns[default_tel].grid.dim)
print("detector binning:", zwfs_ns[default_tel].detector.binning)
print("stellar.bandwidth [nm]:", getattr(zwfs_ns[default_tel].stellar, "bandwidth", None))
print("spectrum wavelengths [um]:", zwfs_ns[default_tel].spectrum.wavelengths * 1e6)
print("sum spectrum weights_nm:", np.sum(zwfs_ns[default_tel].spectrum.weights_nm))
print("fresnel enabled:", getattr(zwfs_ns[default_tel].fresnel_relay, "enabled", None))
print("atmosphere phase enabled:", dynamic_atmosphere_state[default_tel]["phase_enabled"])
print("first-stage AO enabled:", dynamic_atmosphere_state[default_tel]["ao_enabled"])
print("first-stage AO modes removed:", dynamic_atmosphere_state[default_tel]["n_modes_removed"])
print("scintillation enabled:", dynamic_atmosphere_state[default_tel]["scint_enabled"])


# ============================================================
# Shared-memory layout
# ============================================================

# Prefer split JSON from simulator_runtime config if present.
runtime_cfg = getattr(zwfs_ns[default_tel], "simulator_runtime", None)
shared_memory_cfg = getattr(runtime_cfg, "shared_memory", None)

split_filename = get_cfg_value(
    shared_memory_cfg,
    "split_json",
    str(root_path / "baldrapp/apps/paranal_simulator/fake_configs/cred1_split.json"),
)
split_filename = Path(split_filename)
if not split_filename.is_absolute():
    split_filename = root_path / split_filename

with open(split_filename, "r") as file:
    split_dict = json.load(file)

# nrs should ideally be read from the camera server/config.
nrs = int(get_cfg_value(runtime_cfg, "nrs", 5))
global_frame_size = [256, 320, nrs]

baldr_frame_sizes = []
baldr_frame_corners = []

for beam in [1, 2, 3, 4]:
    nx = split_dict[f"baldr{beam}"]["xsz"]
    ny = split_dict[f"baldr{beam}"]["ysz"]
    baldr_frame_sizes.append([nx, ny])

    x0 = split_dict[f"baldr{beam}"]["x0"]
    y0 = split_dict[f"baldr{beam}"]["y0"]
    baldr_frame_corners.append([x0, y0])


# ============================================================
# SHM initialisation
# ============================================================

baldr_sub_shms = {beam: None for beam in [1, 2, 3, 4]}
dm_shms = {beam: None for beam in [1, 2, 3, 4]}

f_cred1_global = get_cfg_value(
    shared_memory_cfg,
    "cred1_global",
    "/dev/shm/cred1.im.shm",
)

global_frame_shm = shm(f_cred1_global, nosem=False)
global_frame_shm.set_data(np.zeros(global_frame_size).astype(dtype=np.uint16))

for ct, beam in enumerate([1, 2, 3, 4]):
    f_baldr = get_cfg_value(
        shared_memory_cfg,
        "baldr_template",
        "/dev/shm/baldr{beam}.im.shm",
    ).format(beam=beam)

    f_dm = get_cfg_value(
        shared_memory_cfg,
        "dm_template",
        "/dev/shm/dm{beam}.im.shm",
    ).format(beam=beam)

    ss = shm(f_baldr, nosem=False)
    ss.set_data(np.zeros(baldr_frame_sizes[ct]))
    baldr_sub_shms[beam] = ss

    # Should be running sim_mdm_server.
    # If not, initialise elsewhere as:
    #   shm(f_dm, data=np.zeros([12, 12]), nosem=False)
    dm_shms[beam] = shm(f_dm, nosem=False)


# ============================================================
# Camera / MDS server state
# ============================================================

ctx = zmq.Context()

cam_socket = ctx.socket(zmq.REQ)
camera_zmq = get_cfg_value(
    getattr(runtime_cfg, "servers", None),
    "camera_zmq",
    "tcp://127.0.0.1:6667",
)
cam_socket.connect(camera_zmq)
cam_socket.send_string('cli "gain"')
print("Camera reply:", cam_socket.recv_string())

# TODO: query camera server for offset/noise when available.
det_cfg = getattr(zwfs_ns[default_tel], "detector_config", None)
adu_offset = float(get_cfg_value(det_cfg, "adu_offset", 1000.0))
noise_std = float(get_cfg_value(det_cfg, "noise_std_adu", 100.0))
include_shotnoise = bool(get_cfg_value(det_cfg, "include_shotnoise", True))

mds_socket = ctx.socket(zmq.REQ)
mds_zmq = get_cfg_value(
    getattr(runtime_cfg, "servers", None),
    "mds_zmq",
    "tcp://127.0.0.1:5555",
)
mds_socket.connect(mds_zmq)

mds_socket.send_string("on SBB")
print("MDS reply (on):", mds_socket.recv_string())
mds_socket.send_string("off SBB")
print("MDS reply (off):", mds_socket.recv_string())

assert len(dm_shms) == len(baldr_sub_shms)

# Move to the configured phasemask in MDS state.
default_mask = get_cfg_value(
    getattr(runtime_cfg, "phasemask", None),
    "default_mask",
    "J3",
)

for beam in [1, 2, 3, 4]:
    mds_socket.send_string(f"fpm_move {beam} {default_mask}")
    print(mds_socket.recv_string())


# ============================================================
# Main simulator loop
# ============================================================

liveindex = global_frame_shm.mtdata["cnt0"]

zero_opd = {
    beam: np.zeros_like(zwfs_ns[beam].grid.pupil_mask, dtype=float)
    for beam in [1, 2, 3, 4]
}

sleep_time_s = float(get_cfg_value(runtime_cfg, "sleep_time_s", 0.01))

while True:

    # ------------------------------------------------------------
    # Read MDS phase-mask state and update each beam's optical config.
    # Empty fpm_whereami response is interpreted as mask out.
    # ------------------------------------------------------------
    for beam in [1, 2, 3, 4]:
        mds_socket.send_string(f"fpm_whereami {beam}")
        mask = mds_socket.recv_string()

        mask_inserted = (mask != "")
        set_phase_mask_state(
            zwfs_ns[beam],
            mask_inserted=mask_inserted,
            original_optics_state=original_optics_state[beam],
        )

    # ------------------------------------------------------------
    # Camera shared-memory counters.
    # ------------------------------------------------------------
    cnt0 = liveindex
    cnt1 = liveindex % global_frame_size[2]

    # ------------------------------------------------------------
    # Per-beam propagation and subframe SHM update.
    # ------------------------------------------------------------
    for ct, beam in enumerate([1, 2, 3, 4]):

        dmcmd = read_dm_command(dm_shms[beam])

        # Current BaldrApp get_frame_configured/get_frame_fresnel internally
        # adds the OPD from zwfs_ns.dm.current_cmd, so do not also pass the DM
        # OPD through opd_input. This avoids double-counting the DM.
        zwfs_ns[beam].dm.current_cmd = dmcmd.copy()

        opd_atmosphere, amp_input_dynamic = generate_atmosphere_and_source_for_beam(
            zwfs=zwfs_ns[beam],
            amp_input_0=amp_input[beam],
            dynamic_state=dynamic_atmosphere_state[beam],
        )

        intensity = bldr.get_frame_configured(
            opd_input=opd_atmosphere,
            amp_input=amp_input_dynamic,
            opd_internal=opd_internal[beam],
            zwfs_ns=zwfs_ns[beam],
            detector=zwfs_ns[beam].detector,
            include_shotnoise=include_shotnoise,
            use_pyZelda=use_pyZelda,
        )

        # Detector/background model in ADU-like units. The optical image is
        # inserted concentrically into the configured Baldr subframe.
        subim_tmp = (
            adu_offset
            + noise_std * np.random.randn(*baldr_frame_sizes[ct])
        )

        simImg = util.insert_concentric(
            intensity,
            np.zeros_like(subim_tmp),
        )
        subim_tmp += simImg

        baldr_sub_shms[beam].set_data(subim_tmp.astype(dtype=np.int32))
        baldr_sub_shms[beam].mtdata["cnt0"] = cnt0
        baldr_sub_shms[beam].mtdata["cnt1"] = cnt1
        baldr_sub_shms[beam].post_sems(1)

    # ------------------------------------------------------------
    # Global CRED1 frame SHM update for display/debugging.
    # ------------------------------------------------------------
    global_frame_shm.mtdata["cnt0"] = cnt0
    global_frame_shm.mtdata["cnt1"] = cnt1

    global_im_tmp = (
        adu_offset
        + noise_std * np.random.randn(*global_frame_size[:-1])
    )

    for ct, beam in enumerate([1, 2, 3, 4]):
        x0, y0 = baldr_frame_corners[ct]
        xsz, ysz = baldr_frame_sizes[ct]

        global_im_tmp[y0:y0 + ysz, x0:x0 + xsz] = (
            baldr_sub_shms[beam].get_data().copy()
        )

    gframe_now = global_frame_shm.get_data().copy()

    print(f"frame {cnt1}")

    gframe_now[cnt1, :, :] = global_im_tmp
    global_frame_shm.set_data(gframe_now.astype(np.uint16))
    time.sleep(sleep_time_s)
    global_frame_shm.post_sems(1)

    liveindex += 1







# ============================================================
# Usage notes
# ============================================================
#
# python3 -m venv venv
# git clone BaldrApp
# cd /to/cloned/directory
#
# git clone pyZelda fork
# cd /to/cloned/directory
# pip install -e .
#
# cd dcs/simulation
# run bash script to start servers:
#
# ./shm_creator_sim
# ./sim_mdm_server
# source venv/bin/activate
# python3 -i simulation/baldr_sim.py
#
# View:
#   shmview /dev/shm/cred1.im.shm
#
# DM GUI:
#   lab-MDM-control &



# ## Run ./shm_creator_sim first to create the shm in the right format.
# # Do not pass data here otherwise it will overwrite/recreate with the Python
# # shmlib.py wrapper format, which may differ from the RTC-required format.






# ## stand alone test 
# from pathlib import Path
# from types import SimpleNamespace

# import numpy as np
# import matplotlib.pyplot as plt

# from baldrapp.common import baldr_core as bldr
# from baldrapp.common import utilities as util
# from baldrapp.common import phasescreens as ps

# import pyzelda.ztools as ztools

# try:
#     import aotools
# except ImportError:
#     aotools = None


# # ============================================================
# # Helpers copied/simplified from paranal_sim.py
# # ============================================================

# def get_cfg_value(ns, name, default=None):
#     return getattr(ns, name, default) if ns is not None else default


# def scale_r0_to_wavelength(r0_ref_m, wavelength_m, reference_wavelength_m):
#     return float(r0_ref_m) * (float(wavelength_m) / float(reference_wavelength_m)) ** (6.0 / 5.0)


# def get_photon_flux_density_from_config(zwfs):
#     source_cfg = getattr(zwfs, "source", None)
#     return float(
#         get_cfg_value(
#             source_cfg,
#             "photons_per_second_per_pixel_per_nm",
#             1.0e5,
#         )
#     )


# def make_first_stage_ao_basis(zwfs, n_modes_removed):
#     nterms = max(50, int(n_modes_removed) + 5)

#     basis_cropped = ztools.zernike.zernike_basis(
#         nterms=nterms,
#         npix=zwfs.grid.N,
#     )

#     basis_template = np.zeros_like(zwfs.grid.pupil_mask, dtype=float)

#     basis = np.array([
#         util.insert_concentric(np.nan_to_num(b, 0.0), basis_template)
#         for b in basis_cropped
#     ])

#     return basis


# def upsample_by_factor(arr, factor):
#     arr = np.asarray(arr)
#     return np.repeat(np.repeat(arr, int(factor), axis=0), int(factor), axis=1)


# def upsample_to_size(arr, target_size):
#     arr = np.asarray(arr)

#     if arr.shape[0] == target_size:
#         return arr

#     factor = max(1, int(target_size) // int(arr.shape[0]))
#     out = upsample_by_factor(arr, factor)

#     if out.shape[0] < target_size:
#         pad = target_size - out.shape[0]
#         out = np.pad(out, ((0, pad), (0, pad)), mode="edge")

#     if out.shape[0] > target_size:
#         out = out[:target_size, :target_size]

#     return out


# def update_scintillation_amplitude(
#     scint_screen,
#     zwfs,
#     pxl_scale,
#     wavelength_m,
#     rows_per_frame=1,
#     propagation_distance_m=10000.0,
#     renormalize_mean_intensity=True,
# ):
#     if aotools is None:
#         raise ImportError("aotools is required for scintillation testing.")

#     for _ in range(int(rows_per_frame)):
#         scint_screen.add_row()

#     wavefront = np.exp(1j * scint_screen.scrn)

#     propagated = aotools.opticalpropagation.angularSpectrum(
#         inputComplexAmp=wavefront,
#         z=float(propagation_distance_m),
#         wvl=float(wavelength_m),
#         inputSpacing=float(pxl_scale),
#         outputSpacing=float(pxl_scale),
#     )

#     amp = np.abs(propagated)
#     amp = upsample_to_size(amp, zwfs.grid.pupil_mask.shape[0])

#     if renormalize_mean_intensity:
#         pupil = zwfs.grid.pupil_mask.astype(bool)
#         mean_intensity = np.mean(amp[pupil] ** 2)

#         if mean_intensity > 0:
#             amp = amp / np.sqrt(mean_intensity)

#     return amp


# def init_dynamic_atmosphere_for_beam(zwfs):
#     wvl0 = float(zwfs.optics.wvl0)
#     dx = float(zwfs.grid.D) / float(zwfs.grid.N)

#     atm_cfg = getattr(zwfs, "atmosphere", None)
#     phase_cfg = getattr(atm_cfg, "phase", None)
#     scint_cfg = getattr(atm_cfg, "scintillation", None)
#     ao_cfg = getattr(zwfs, "first_stage_ao", None)

#     phase_enabled = bool(get_cfg_value(phase_cfg, "enabled", False))
#     ao_enabled = bool(get_cfg_value(ao_cfg, "enabled", False))
#     scint_enabled = bool(get_cfg_value(scint_cfg, "enabled", False))

#     phase_screen = None
#     if phase_enabled:
#         r0_500_m = float(get_cfg_value(phase_cfg, "r0_500_m", 0.126))
#         reference_wavelength_m = float(
#             get_cfg_value(phase_cfg, "reference_wavelength_m", 500e-9)
#         )

#         r0_wvl_m = scale_r0_to_wavelength(
#             r0_500_m,
#             wvl0,
#             reference_wavelength_m,
#         )

#         phase_screen = ps.PhaseScreenKolmogorov(
#             nx_size=int(zwfs.grid.dim),
#             pixel_scale=dx,
#             r0=r0_wvl_m,
#             L0=float(get_cfg_value(phase_cfg, "L0_m", 25.0)),
#             random_seed=get_cfg_value(phase_cfg, "random_seed", None),
#         )

#     n_modes_removed = int(get_cfg_value(ao_cfg, "Nmodes_removed", 0))
#     ao_basis = None

#     if phase_enabled and ao_enabled and n_modes_removed > 0:
#         ao_basis = make_first_stage_ao_basis(zwfs, n_modes_removed)

#     scint_screen = None
#     if scint_enabled:
#         if aotools is None:
#             raise ImportError("aotools is required for scintillation.enabled=true.")

#         scint_r0_500_m = float(get_cfg_value(scint_cfg, "r0_500_m", 0.126))
#         scint_ref_wvl_m = float(
#             get_cfg_value(scint_cfg, "reference_wavelength_m", 500e-9)
#         )

#         scint_r0_wvl_m = scale_r0_to_wavelength(
#             scint_r0_500_m,
#             wvl0,
#             scint_ref_wvl_m,
#         )

#         scint_screen = aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman(
#             nx_size=int(zwfs.grid.dim),
#             pixel_scale=dx,
#             r0=scint_r0_wvl_m,
#             L0=float(get_cfg_value(scint_cfg, "L0_m", 25.0)),
#             random_seed=get_cfg_value(scint_cfg, "random_seed", None),
#         )

#     return {
#         "dx": dx,
#         "phase_enabled": phase_enabled,
#         "phase_screen": phase_screen,
#         "phase_rows_per_frame": int(get_cfg_value(phase_cfg, "rows_per_frame", 1)),
#         "phase_scaling_factor": float(
#             get_cfg_value(phase_cfg, "phase_scaling_factor", 1.0)
#         ),
#         "ao_enabled": ao_enabled,
#         "ao_basis": ao_basis,
#         "n_modes_removed": n_modes_removed,
#         "ao_phase_scaling_factor": float(
#             get_cfg_value(ao_cfg, "phase_scaling_factor", 1.0)
#         ),
#         "scint_enabled": scint_enabled,
#         "scint_screen": scint_screen,
#         "scint_rows_per_frame": int(get_cfg_value(scint_cfg, "rows_per_frame", 1)),
#         "scint_propagation_distance_m": float(
#             get_cfg_value(scint_cfg, "propagation_distance_m", 10000.0)
#         ),
#         "scint_renormalize_mean_intensity": bool(
#             get_cfg_value(scint_cfg, "renormalize_mean_intensity", True)
#         ),
#     }


# def generate_atmosphere_and_source_for_beam(zwfs, amp_input_0, dynamic_state):
#     pupil = zwfs.grid.pupil_mask
#     wvl0 = float(zwfs.optics.wvl0)

#     if dynamic_state["phase_enabled"]:
#         scrn = dynamic_state["phase_screen"]

#         for _ in range(int(dynamic_state["phase_rows_per_frame"])):
#             scrn.add_row()

#         phase_scaling = float(dynamic_state["phase_scaling_factor"])

#         if (
#             dynamic_state["ao_enabled"]
#             and dynamic_state["ao_basis"] is not None
#             and dynamic_state["n_modes_removed"] > 0
#         ):
#             phase_after_ao = bldr.first_stage_ao(
#                 atm_scrn=scrn,
#                 Nmodes_removed=dynamic_state["n_modes_removed"],
#                 basis=dynamic_state["ao_basis"],
#                 phase_scaling_factor=(
#                     phase_scaling
#                     * float(dynamic_state["ao_phase_scaling_factor"])
#                 ),
#                 return_reconstructor=False,
#             )
#         else:
#             phase_after_ao = pupil * phase_scaling * scrn.scrn

#         opd_input = pupil * (wvl0 / (2.0 * np.pi)) * phase_after_ao
#         opd_raw = pupil * (wvl0 / (2.0 * np.pi)) * phase_scaling * scrn.scrn
#     else:
#         opd_input = np.zeros_like(pupil, dtype=float)
#         opd_raw = np.zeros_like(pupil, dtype=float)

#     if dynamic_state["scint_enabled"]:
#         amp_scint = update_scintillation_amplitude(
#             scint_screen=dynamic_state["scint_screen"],
#             zwfs=zwfs,
#             pxl_scale=dynamic_state["dx"],
#             wavelength_m=wvl0,
#             rows_per_frame=dynamic_state["scint_rows_per_frame"],
#             propagation_distance_m=dynamic_state["scint_propagation_distance_m"],
#             renormalize_mean_intensity=dynamic_state["scint_renormalize_mean_intensity"],
#         )

#         amp_input = amp_input_0 * amp_scint
#     else:
#         amp_scint = np.ones_like(pupil, dtype=float)
#         amp_input = amp_input_0

#     return opd_input, amp_input, opd_raw, amp_scint


# def rms_nm(opd, pupil):
#     mask = pupil.astype(bool)
#     return 1e9 * np.std(opd[mask])


# def marechal_strehl(opd, pupil, wavelength_m):
#     mask = pupil.astype(bool)
#     phase = 2.0 * np.pi / wavelength_m * opd[mask]
#     return np.exp(-np.var(phase))


# def summarize(name, im):
#     print(f"\n{name}")
#     print("  shape:", im.shape)
#     print("  sum:", np.sum(im))
#     print("  min/max:", np.min(im), np.max(im))
#     print("  mean/std:", np.mean(im), np.std(im))


# # ============================================================
# # Main single-beam verification
# # ============================================================

# repo_root = Path("/Users/bencb/Documents/ASGARD/BaldrApp")
# config_path = repo_root / "baldrapp/apps/paranal_simulator/fake_configs/baldr_config.json"

# zwfs = bldr.init_zwfs_from_json(config_path)

# # Match simulator override, if you want to test exactly the live script.
# zwfs.dm.actuator_coupling_factor = 0.9
# zwfs = bldr.update_dm_registration_wavespace(
#     zwfs.dm2wavespace_registration.dm_to_wavesp_transform_matrix,
#     zwfs,
# )

# pupil = zwfs.grid.pupil_mask
# detector = zwfs.detector
# opd_internal = np.zeros_like(pupil, dtype=float)

# flux_density = get_photon_flux_density_from_config(zwfs)
# amp_input_0 = np.sqrt(flux_density) * pupil

# dynamic_state = init_dynamic_atmosphere_for_beam(zwfs)

# print("\n=== Single-beam verification config ===")
# print("config_path:", config_path)
# print("grid N:", zwfs.grid.N)
# print("grid dim:", zwfs.grid.dim)
# print("detector binning:", detector.binning)
# print("stellar.bandwidth [nm]:", getattr(zwfs.stellar, "bandwidth", None))
# print("spectrum wavelengths [um]:", zwfs.spectrum.wavelengths * 1e6)
# print("sum spectrum weights_nm:", np.sum(zwfs.spectrum.weights_nm))
# print("fresnel enabled:", getattr(zwfs.fresnel_relay, "enabled", None))
# print("phase enabled:", dynamic_state["phase_enabled"])
# print("AO enabled:", dynamic_state["ao_enabled"])
# print("AO modes removed:", dynamic_state["n_modes_removed"])
# print("scintillation enabled:", dynamic_state["scint_enabled"])
# print("flux density:", flux_density)


# # Reference frames, flat DM, no atmosphere
# zwfs.dm.current_cmd = zwfs.dm.dm_flat.copy()

# I0 = bldr.get_I0_configured(
#     opd_input=np.zeros_like(pupil),
#     amp_input=amp_input_0,
#     opd_internal=opd_internal,
#     zwfs_ns=zwfs,
#     detector=detector,
#     include_shotnoise=False,
#     use_pyZelda=False,
# )

# N0 = bldr.get_N0_configured(
#     opd_input=np.zeros_like(pupil),
#     amp_input=amp_input_0,
#     opd_internal=opd_internal,
#     zwfs_ns=zwfs,
#     detector=detector,
#     include_shotnoise=False,
#     use_pyZelda=False,
# )

# summarize("I0 configured", I0)
# summarize("N0 configured", N0)
# print("I0/N0 sum ratio:", np.sum(I0) / np.sum(N0))


# # A small non-flat DM command to verify DM is included by get_frame_configured.
# np.random.seed(4)
# zwfs.dm.current_cmd = zwfs.dm.dm_flat + 0.005 * np.random.randn(len(zwfs.dm.dm_flat))

# n_iter = 8

# frames = []
# signals = []
# opd_raw_list = []
# opd_ao_list = []
# amp_scint_list = []

# rms_raw_list = []
# rms_ao_list = []
# strehl_raw_list = []
# strehl_ao_list = []
# flux_sum_list = []

# for k in range(n_iter):

#     opd_ao, amp_input, opd_raw, amp_scint = generate_atmosphere_and_source_for_beam(
#         zwfs=zwfs,
#         amp_input_0=amp_input_0,
#         dynamic_state=dynamic_state,
#     )

#     I = bldr.get_frame_configured(
#         opd_input=opd_ao,
#         amp_input=amp_input,
#         opd_internal=opd_internal,
#         zwfs_ns=zwfs,
#         detector=detector,
#         include_shotnoise=False,
#         use_pyZelda=False,
#     ).astype(float)

#     signal = I / (np.mean(N0) + 1e-12) - I0 / (np.mean(N0) + 1e-12)

#     frames.append(I)
#     signals.append(signal)
#     opd_raw_list.append(opd_raw)
#     opd_ao_list.append(opd_ao)
#     amp_scint_list.append(amp_scint)

#     rms_raw_list.append(rms_nm(opd_raw, pupil))
#     rms_ao_list.append(rms_nm(opd_ao, pupil))
#     strehl_raw_list.append(marechal_strehl(opd_raw, pupil, zwfs.optics.wvl0))
#     strehl_ao_list.append(marechal_strehl(opd_ao, pupil, zwfs.optics.wvl0))
#     flux_sum_list.append(np.sum(I))

#     print(
#         f"iter {k:02d}: "
#         f"raw rms={rms_raw_list[-1]:8.2f} nm, "
#         f"AO rms={rms_ao_list[-1]:8.2f} nm, "
#         f"Sraw={strehl_raw_list[-1]:.4f}, "
#         f"Sao={strehl_ao_list[-1]:.4f}, "
#         f"flux={flux_sum_list[-1]:.3e}"
#     )

# frames = np.asarray(frames)
# signals = np.asarray(signals)
# opd_raw_list = np.asarray(opd_raw_list)
# opd_ao_list = np.asarray(opd_ao_list)
# amp_scint_list = np.asarray(amp_scint_list)

# print("\n=== Verification summary ===")
# print("frames shape:", frames.shape)
# print("signals shape:", signals.shape)
# print("median raw RMS [nm]:", np.median(rms_raw_list))
# print("median AO RMS [nm]:", np.median(rms_ao_list))
# print("median raw Strehl:", np.median(strehl_raw_list))
# print("median AO Strehl:", np.median(strehl_ao_list))
# print("relative flux std:", np.std(flux_sum_list) / np.mean(flux_sum_list))

# assert np.all(np.isfinite(frames))
# assert np.all(np.isfinite(signals))
# assert np.sum(frames) > 0

# if dynamic_state["phase_enabled"] and dynamic_state["ao_enabled"]:
#     assert np.median(rms_ao_list) < np.median(rms_raw_list), (
#         "AO did not reduce median OPD RMS. Check phase-screen convention/basis."
#     )

# print("\nPASS: single-beam configured atmosphere + AO + scintillation + poly-Fresnel propagation works.")


# # ============================================================
# # Diagnostic plots
# # ============================================================

# idx = min(3, n_iter - 1)

# fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# plot_items = [
#     ("Scintillation amplitude", amp_scint_list[idx]),
#     ("Raw atmosphere OPD [nm]", 1e9 * opd_raw_list[idx]),
#     ("After first-stage AO OPD [nm]", 1e9 * opd_ao_list[idx]),
#     ("Detected frame", frames[idx]),
#     ("I0 reference", I0),
#     ("Signal I/<N0> - I0/<N0>", signals[idx]),
# ]

# for ax, (title, im) in zip(axes.ravel(), plot_items):
#     im = np.asarray(im, dtype=float)

#     if "OPD" in title or "Signal" in title:
#         lim = np.nanpercentile(np.abs(im), 99)
#         lim = max(lim, 1e-12)
#         h = ax.imshow(im, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
#     else:
#         p1, p99 = np.nanpercentile(im, [1, 99])
#         h = ax.imshow(im, origin="lower", vmin=p1, vmax=p99)

#     ax.set_title(title)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     plt.colorbar(h, ax=ax, fraction=0.046)

# plt.tight_layout()
# plt.show()


# fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# axes[0].plot(rms_raw_list, label="raw atmosphere")
# axes[0].plot(rms_ao_list, label="after first-stage AO")
# axes[0].set_ylabel("OPD RMS [nm]")
# axes[0].grid(alpha=0.3)
# axes[0].legend()

# axes[1].plot(strehl_raw_list, label="raw atmosphere")
# axes[1].plot(strehl_ao_list, label="after first-stage AO")
# axes[1].set_ylabel("Marechal Strehl")
# axes[1].grid(alpha=0.3)
# axes[1].legend()

# flux_sum_arr = np.asarray(flux_sum_list)
# axes[2].plot(flux_sum_arr / np.mean(flux_sum_arr), label="frame sum / mean")
# axes[2].set_xlabel("iteration")
# axes[2].set_ylabel("relative flux")
# axes[2].grid(alpha=0.3)
# axes[2].legend()

# plt.tight_layout()
# plt.show()








# # import numpy as np
# # import json
# # import zmq
# # import time
# # import os
# # from xaosim.shmlib import shm
# # import subprocess
# # from pathlib import Path

# # from baldrapp.common import baldr_core as bldr
# # from baldrapp.common import utilities as util


# # # ============================================================
# # # Helpers
# # # ============================================================

# # def get_git_root() -> Path:
# #     return Path(
# #         subprocess.check_output(
# #             ["git", "rev-parse", "--show-toplevel"],
# #             text=True,
# #         ).strip()
# #     )


# # def convert_12x12_to_140(arr):
# #     """
# #     Convert a 12x12 DM command image with four unused corners into the
# #     140-element BMC multi-3.5 command vector.
# #     """
# #     arr = np.asarray(arr)

# #     if arr.shape != (12, 12):
# #         raise ValueError("Input must be a 12x12 array.")

# #     flat = arr.flatten()
# #     corner_indices = [0, 11, 132, 143]
# #     return np.delete(flat, corner_indices)


# # def get_cfg_value(ns, name, default=None):
# #     return getattr(ns, name, default) if ns is not None else default


# # def get_photon_flux_density_from_config(zwfs):
# #     """
# #     Return photons / second / wave-space-pixel / nm for the simulator source.

# #     Preferred new JSON config field:
# #         zwfs.source.photons_per_second_per_pixel_per_nm

# #     Fallback:
# #         1.0e5 photons / s / pixel / nm

# #     This keeps the simulator independent of the old pyZELDA/throughput/magnitude
# #     calculation path.
# #     """
# #     source_cfg = getattr(zwfs, "source", None)

# #     return float(
# #         get_cfg_value(
# #             source_cfg,
# #             "photons_per_second_per_pixel_per_nm",
# #             1.0e5,
# #         )
# #     )


# # def get_internal_opd_from_config(zwfs):
# #     """
# #     Build static/internal OPD from the new JSON config.

# #     Currently supports:
# #         internal_aberrations.parabolic_scratches

# #     If no internal-aberrations section is present, returns zeros.
# #     """
# #     opd_internal = np.zeros_like(zwfs.grid.pupil_mask, dtype=float)
# #     dx = zwfs.grid.D / zwfs.grid.N

# #     ia = getattr(zwfs, "internal_aberrations", None)
# #     if ia is None or not bool(get_cfg_value(ia, "enabled", False)):
# #         return opd_internal

# #     scratches = getattr(ia, "parabolic_scratches", None)
# #     if scratches is not None and bool(get_cfg_value(scratches, "enabled", False)):
# #         width_list = None

# #         if hasattr(scratches, "width_list"):
# #             width_list = list(scratches.width_list)
# #         elif hasattr(scratches, "width_list_dx_factor"):
# #             width_list = [float(v) * dx for v in scratches.width_list_dx_factor]

# #         if width_list is None:
# #             width_list = [2.0 * dx]

# #         opd_internal = util.apply_parabolic_scratches(
# #             opd_internal,
# #             dx=dx,
# #             dy=dx,
# #             list_a=list(get_cfg_value(scratches, "list_a", [0.1])),
# #             list_b=list(get_cfg_value(scratches, "list_b", [0.0])),
# #             list_c=list(get_cfg_value(scratches, "list_c", [-2.0])),
# #             width_list=width_list,
# #             depth_list=list(get_cfg_value(scratches, "depth_list_m", [100e-9])),
# #         )

# #     return zwfs.grid.pupil_mask * opd_internal


# # def set_phase_mask_state(zwfs, mask_inserted, original_optics_state):
# #     """
# #     Apply phase-mask in/out state to the configured analytic propagation path.

# #     For mask out, force theta=0 and theta_mode='constant'. This is robust even
# #     when the configured mask normally uses theta_mode='physical_depth'.
# #     """
# #     if mask_inserted:
# #         zwfs.optics.theta = original_optics_state["theta"]
# #         zwfs.optics.theta_mode = original_optics_state["theta_mode"]
# #     else:
# #         zwfs.optics.theta = 0.0
# #         zwfs.optics.theta_mode = "constant"


# # def read_dm_command(dm_shm):
# #     """
# #     Read the combined 12x12 simulated DM shared-memory image and convert it to
# #     the Baldr 140-vector command.

# #     The simulator currently treats this shared-memory command as the full DM
# #     command used by the optical model, preserving the behaviour of the previous
# #     script.
# #     """
# #     dmcmd_2d = dm_shm.get_data()
# #     return convert_12x12_to_140(dmcmd_2d)


# # # ============================================================
# # # Configuration / initialisation
# # # ============================================================

# # root_path = get_git_root()

# # # New generic BaldrApp JSON config. This should include:
# # #   grid, optics, dm, stellar.spectrum, fresnel_relay, detector, source, etc.
# # config_path = (
# #     root_path
# #     / "baldrapp/apps/paranal_simulator/fake_configs/baldr_config.json"
# # )

# # # Initialise one independent ZWFS namespace per beam.
# # # The configured frame dispatcher will automatically use:
# # #   - polychromatic propagation if zwfs.spectrum.enabled and n_wvl > 1
# # #   - Fresnel relay propagation if zwfs.fresnel_relay.enabled is True
# # zwfs_ns = {beam: None for beam in [1, 2, 3, 4]}
# # amp_input = {}
# # opd_internal = {}
# # original_optics_state = {}

# # for beam in [1, 2, 3, 4]:
# #     zwfs = bldr.init_zwfs_from_json(config_path)

# #     # Optional per-simulator override retained from the old script.
# #     zwfs.dm.actuator_coupling_factor = 0.9

# #     # Recompute DM registration after changing actuator coupling factor.
# #     # This keeps act_sigma_wavesp consistent with the updated coupling.
# #     zwfs = bldr.update_dm_registration_wavespace(
# #         zwfs.dm2wavespace_registration.dm_to_wavesp_transform_matrix,
# #         zwfs,
# #     )

# #     flux_density = get_photon_flux_density_from_config(zwfs)
# #     amp_input[beam] = np.sqrt(flux_density) * zwfs.grid.pupil_mask

# #     opd_internal[beam] = get_internal_opd_from_config(zwfs)

# #     original_optics_state[beam] = {
# #         "theta": float(zwfs.optics.theta),
# #         "theta_mode": str(getattr(zwfs.optics, "theta_mode", "constant")),
# #     }

# #     zwfs_ns[beam] = zwfs


# # default_tel = 1
# # use_pyZelda = False

# # print("\n=== Baldr simulator optical configuration ===")
# # print("config_path:", config_path)
# # print("grid N:", zwfs_ns[default_tel].grid.N)
# # print("grid dim:", zwfs_ns[default_tel].grid.dim)
# # print("detector binning:", zwfs_ns[default_tel].detector.binning)
# # print("stellar.bandwidth [nm]:", getattr(zwfs_ns[default_tel].stellar, "bandwidth", None))
# # print("spectrum wavelengths [um]:", zwfs_ns[default_tel].spectrum.wavelengths * 1e6)
# # print("sum spectrum weights_nm:", np.sum(zwfs_ns[default_tel].spectrum.weights_nm))
# # print("fresnel enabled:", getattr(zwfs_ns[default_tel].fresnel_relay, "enabled", None))


# # # ============================================================
# # # Shared-memory layout
# # # ============================================================

# # # Prefer split JSON from simulator_runtime config if present.
# # runtime_cfg = getattr(zwfs_ns[default_tel], "simulator_runtime", None)
# # shared_memory_cfg = getattr(runtime_cfg, "shared_memory", None)

# # split_filename = get_cfg_value(
# #     shared_memory_cfg,
# #     "split_json",
# #     str(root_path / "baldrapp/apps/paranal_simulator/fake_configs/cred1_split.json"),
# # )
# # split_filename = Path(split_filename)
# # if not split_filename.is_absolute():
# #     split_filename = root_path / split_filename

# # with open(split_filename, "r") as file:
# #     split_dict = json.load(file)

# # # nrs should ideally be read from the camera server/config.
# # nrs = int(get_cfg_value(runtime_cfg, "nrs", 5))
# # global_frame_size = [256, 320, nrs]

# # baldr_frame_sizes = []
# # baldr_frame_corners = []

# # for beam in [1, 2, 3, 4]:
# #     nx = split_dict[f"baldr{beam}"]["xsz"]
# #     ny = split_dict[f"baldr{beam}"]["ysz"]
# #     baldr_frame_sizes.append([nx, ny])

# #     x0 = split_dict[f"baldr{beam}"]["x0"]
# #     y0 = split_dict[f"baldr{beam}"]["y0"]
# #     baldr_frame_corners.append([x0, y0])


# # # ============================================================
# # # SHM initialisation
# # # ============================================================

# # baldr_sub_shms = {beam: None for beam in [1, 2, 3, 4]}
# # dm_shms = {beam: None for beam in [1, 2, 3, 4]}

# # f_cred1_global = get_cfg_value(
# #     shared_memory_cfg,
# #     "cred1_global",
# #     "/dev/shm/cred1.im.shm",
# # )

# # global_frame_shm = shm(f_cred1_global, nosem=False)
# # global_frame_shm.set_data(np.zeros(global_frame_size).astype(dtype=np.uint16))

# # for ct, beam in enumerate([1, 2, 3, 4]):
# #     f_baldr = get_cfg_value(
# #         shared_memory_cfg,
# #         "baldr_template",
# #         "/dev/shm/baldr{beam}.im.shm",
# #     ).format(beam=beam)

# #     f_dm = get_cfg_value(
# #         shared_memory_cfg,
# #         "dm_template",
# #         "/dev/shm/dm{beam}.im.shm",
# #     ).format(beam=beam)

# #     ss = shm(f_baldr, nosem=False)
# #     ss.set_data(np.zeros(baldr_frame_sizes[ct]))
# #     baldr_sub_shms[beam] = ss

# #     # Should be running sim_mdm_server.
# #     # If not, initialise elsewhere as:
# #     #   shm(f_dm, data=np.zeros([12, 12]), nosem=False)
# #     dm_shms[beam] = shm(f_dm, nosem=False)


# # # ============================================================
# # # Camera / MDS server state
# # # ============================================================

# # ctx = zmq.Context()

# # cam_socket = ctx.socket(zmq.REQ)
# # camera_zmq = get_cfg_value(
# #     getattr(runtime_cfg, "servers", None),
# #     "camera_zmq",
# #     "tcp://127.0.0.1:6667",
# # )
# # cam_socket.connect(camera_zmq)
# # cam_socket.send_string('cli "gain"')
# # print("Camera reply:", cam_socket.recv_string())

# # # TODO: query camera server for offset/noise when available.
# # det_cfg = getattr(zwfs_ns[default_tel], "detector_config", None)
# # adu_offset = float(get_cfg_value(det_cfg, "adu_offset", 1000.0))
# # noise_std = float(get_cfg_value(det_cfg, "noise_std_adu", 100.0))
# # include_shotnoise = bool(get_cfg_value(det_cfg, "include_shotnoise", True))

# # mds_socket = ctx.socket(zmq.REQ)
# # mds_zmq = get_cfg_value(
# #     getattr(runtime_cfg, "servers", None),
# #     "mds_zmq",
# #     "tcp://127.0.0.1:5555",
# # )
# # mds_socket.connect(mds_zmq)

# # mds_socket.send_string("on SBB")
# # print("MDS reply (on):", mds_socket.recv_string())
# # mds_socket.send_string("off SBB")
# # print("MDS reply (off):", mds_socket.recv_string())

# # assert len(dm_shms) == len(baldr_sub_shms)

# # # Move to the configured phasemask in MDS state.
# # default_mask = get_cfg_value(
# #     getattr(runtime_cfg, "phasemask", None),
# #     "default_mask",
# #     "J3",
# # )

# # for beam in [1, 2, 3, 4]:
# #     mds_socket.send_string(f"fpm_move {beam} {default_mask}")
# #     print(mds_socket.recv_string())


# # # ============================================================
# # # Main simulator loop
# # # ============================================================

# # liveindex = global_frame_shm.mtdata["cnt0"]

# # zero_opd = {
# #     beam: np.zeros_like(zwfs_ns[beam].grid.pupil_mask, dtype=float)
# #     for beam in [1, 2, 3, 4]
# # }

# # sleep_time_s = float(get_cfg_value(runtime_cfg, "sleep_time_s", 0.01))

# # while True:

# #     # ------------------------------------------------------------
# #     # Read MDS phase-mask state and update each beam's optical config.
# #     # Empty fpm_whereami response is interpreted as mask out.
# #     # ------------------------------------------------------------
# #     for beam in [1, 2, 3, 4]:
# #         mds_socket.send_string(f"fpm_whereami {beam}")
# #         mask = mds_socket.recv_string()

# #         mask_inserted = (mask != "")
# #         set_phase_mask_state(
# #             zwfs_ns[beam],
# #             mask_inserted=mask_inserted,
# #             original_optics_state=original_optics_state[beam],
# #         )

# #     # ------------------------------------------------------------
# #     # Camera shared-memory counters.
# #     # ------------------------------------------------------------
# #     cnt0 = liveindex
# #     cnt1 = liveindex % global_frame_size[2]

# #     # ------------------------------------------------------------
# #     # Per-beam propagation and subframe SHM update.
# #     # ------------------------------------------------------------
# #     for ct, beam in enumerate([1, 2, 3, 4]):

# #         dmcmd = read_dm_command(dm_shms[beam])

# #         # Current BaldrApp get_frame_configured/get_frame_fresnel internally
# #         # adds the OPD from zwfs_ns.dm.current_cmd, so do not also pass the DM
# #         # OPD through opd_input. This avoids double-counting the DM.
# #         zwfs_ns[beam].dm.current_cmd = dmcmd.copy()

# #         intensity = bldr.get_frame_configured(
# #             opd_input=zero_opd[beam],
# #             amp_input=amp_input[beam],
# #             opd_internal=opd_internal[beam],
# #             zwfs_ns=zwfs_ns[beam],
# #             detector=zwfs_ns[beam].detector,
# #             include_shotnoise=include_shotnoise,
# #             use_pyZelda=use_pyZelda,
# #         )

# #         # Detector/background model in ADU-like units. The optical image is
# #         # inserted concentrically into the configured Baldr subframe.
# #         subim_tmp = (
# #             adu_offset
# #             + noise_std * np.random.randn(*baldr_frame_sizes[ct])
# #         )

# #         simImg = util.insert_concentric(
# #             intensity,
# #             np.zeros_like(subim_tmp),
# #         )
# #         subim_tmp += simImg

# #         baldr_sub_shms[beam].set_data(subim_tmp.astype(dtype=np.int32))
# #         baldr_sub_shms[beam].mtdata["cnt0"] = cnt0
# #         baldr_sub_shms[beam].mtdata["cnt1"] = cnt1
# #         baldr_sub_shms[beam].post_sems(1)

# #     # ------------------------------------------------------------
# #     # Global CRED1 frame SHM update for display/debugging.
# #     # ------------------------------------------------------------
# #     global_frame_shm.mtdata["cnt0"] = cnt0
# #     global_frame_shm.mtdata["cnt1"] = cnt1

# #     global_im_tmp = (
# #         adu_offset
# #         + noise_std * np.random.randn(*global_frame_size[:-1])
# #     )

# #     for ct, beam in enumerate([1, 2, 3, 4]):
# #         x0, y0 = baldr_frame_corners[ct]
# #         xsz, ysz = baldr_frame_sizes[ct]

# #         global_im_tmp[y0:y0 + ysz, x0:x0 + xsz] = (
# #             baldr_sub_shms[beam].get_data().copy()
# #         )

# #     gframe_now = global_frame_shm.get_data().copy()

# #     print(f"frame {cnt1}")

# #     gframe_now[cnt1, :, :] = global_im_tmp
# #     global_frame_shm.set_data(gframe_now.astype(np.uint16))
# #     time.sleep(sleep_time_s)
# #     global_frame_shm.post_sems(1)

# #     liveindex += 1


# # # ============================================================
# # # Usage notes
# # # ============================================================
# # #
# # # python3 -m venv venv
# # # git clone BaldrApp
# # # cd /to/cloned/directory
# # #
# # # git clone pyZelda fork
# # # cd /to/cloned/directory
# # # pip install -e .
# # #
# # # cd dcs/simulation
# # # run bash script to start servers:
# # #
# # # ./shm_creator_sim
# # # ./sim_mdm_server
# # # source venv/bin/activate
# # # python3 -i simulation/baldr_sim.py
# # #
# # # View:
# # #   shmview /dev/shm/cred1.im.shm
# # #
# # # DM GUI:
# # #   lab-MDM-control &
