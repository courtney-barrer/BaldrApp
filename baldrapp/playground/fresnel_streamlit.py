import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# If running from installed package this is fine, but this helps when running local source.
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from baldrapp.common import baldr_core as bldr
from baldrapp.common import fresnel



"""
Notess about mis-conjugation and relation to telescope diameter 
Convert detector-side pupil mis-conjugation to an equivalent telescope-scale
axial mis-conjugation.

In this model, the Streamlit "pupil mis-conjugation" parameter is an axial
detector displacement from the nominal detector pupil image plane, dz_det.
It is not a direct atmospheric-layer altitude.

The Baldr relay used here has an effective post-collimator pupil diameter:

    D_relay = 12 mm * (40 mm / 254 mm)
            = 1.890 mm

and a nominal detector pupil diameter:

    D_det = 288 um

The transverse pupil magnification is therefore:

    M = D_det / D_relay
      = 288e-6 / 1.890e-3
      = 0.1524

For paraxial imaging, axial defocus or mis-conjugation scales approximately
as the square of the transverse magnification:

    dz_det = M^2 * dz_relay

so the equivalent object-side displacement in the Baldr relay space is:

    dz_relay = dz_det / M^2

Since M^2 = 0.0232, a 1 mm detector-side displacement corresponds to:

    dz_relay = 1 mm / 0.1524^2
             = 43.1 mm

in the 1.89 mm Baldr relay space.

To express this as an equivalent telescope-scale longitudinal distance, scale
by the ratio of telescope pupil diameter to relay pupil diameter:

    dz_tel = dz_relay * (D_tel / D_relay)

For an AT-sized 1.8 m pupil:

    dz_AT = 43.1 mm * (1.8 / 1.890e-3)
          = 41 m

per 1 mm detector-side displacement.

For a UT-sized 8.0 m pupil:

    dz_UT = 43.1 mm * (8.0 / 1.890e-3)
          = 182 m

per 1 mm detector-side displacement.

Rule of thumb for this relay:

    1 mm detector-side pupil mis-conjugation
        is about 43 mm in Baldr relay object space
        is about 41 m at the 1.8 m AT scale
        is about 182 m at the 8.0 m UT scale

This is a geometric scaling of pupil-image defocus only. It should not be
interpreted as a full atmospheric-layer conjugation model, which would also
depend on field angle, atmospheric propagation, and the detailed instrument
conjugation geometry.
"""

# ============================================================
# Streamlit page setup
# ============================================================

st.set_page_config(
    page_title="Baldr ZWFS Fresnel Relay Explorer",
    layout="wide",
)

st.title("Baldr ZWFS Fresnel Relay Explorer")

st.markdown(
    """
This app evaluates:

`analytic ZWFS field` → `free-space to D/knife-edge mirror` → `D-edge vignetting`
→ `propagate to 184 mm re-imaging lens` → `thin-lens phase`
→ `physical cold stop / star-image plane` → `detector pupil plane`.

The simulation only updates when **Update simulation** is pressed.
"""
)


# ============================================================
# Hardcoded nominal physical parameters
# ============================================================

D_entrance = 12e-3          # m, Baldr input pupil diameter before OAP
f_oap = 254e-3              # m, Baldr OAP focal length
f_coll = 40e-3              # m, updated post-mask collimating lens
z_to_mirror = 0.50          # m, collimator output / ZWFS pupil plane to D/knife-edge mirror
D_mirror = 25.4e-3          # m, 1 inch mirror clear aperture
f_imaging = 184e-3          # m, Baldr re-imaging lens
D_coldstop_nominal = 2.145e-3
D_detector_pupil_nominal = 288e-6

# Physical beam diameter after 40 mm collimator.
D_phys = D_entrance * f_coll / f_oap

# Nominal pupil magnification and lens conjugates for ~288 um detector pupil.
M_nominal = D_detector_pupil_nominal / D_phys
s_object_nominal = f_imaging * (1.0 + 1.0 / M_nominal)
s_image_nominal = f_imaging * (1.0 + M_nominal)
z_mirror_to_lens_nominal = s_object_nominal - z_to_mirror
z_focus_to_detector_nominal = s_image_nominal - f_imaging

edge_offset_nominal = -1.0e-3
tip_amp_nominal_nm = 0.0
astig_amp_nominal_nm = 0.0
theta_nominal = 1.57079  # rad, approximately pi/2

coldstop_x_offset_nominal = 0.0      # m
pupil_misconjugation_nominal = 0.0   # m, detector offset from nominal pupil plane


# ============================================================
# Controls: form means no live update while sliding
# ============================================================

with st.sidebar:
    st.header("Controls")
    st.caption("Nominal values are shown in each slider label.")

    with st.form("simulation_controls"):

        theta_phase_shift_rad = st.slider(
            r"ZWFS phase shift θ [rad] — nominal π/2",
            min_value=0.0,
            max_value=float(np.pi),
            value=float(theta_nominal),
            step=0.01,
            help="Phase shift applied by the ZWFS focal-plane phase mask.",
        )

        st.caption(
            f"θ = {theta_phase_shift_rad:.3f} rad "
            f"= {theta_phase_shift_rad / np.pi:.3f}π"
        )

        edge_offset_mm = st.slider(
            "Knife/D-edge offset [mm] — nominal -1.0 mm",
            min_value=-3.0,
            max_value=1.5,
            value=1e3 * edge_offset_nominal,
            step=0.05,
            help=(
                "Half-plane mask keeps X >= edge_offset. "
                "More positive values clip more of the beam."
            ),
        )

        z_mirror_to_lens_mm = st.slider(
            f"Mirror → re-imaging lens distance [mm] — nominal {1e3 * z_mirror_to_lens_nominal:.1f} mm",
            min_value=300.0,
            max_value=1300.0,
            value=float(1e3 * z_mirror_to_lens_nominal),
            step=5.0,
            help=(
                "Controls the object distance for pupil imaging. "
                "The detector distance is recomputed from the thin-lens formula."
            ),
        )

        coldstop_diam_mm = st.slider(
            "Cold stop diameter [mm] — nominal 2.145 mm",
            min_value=0.5,
            max_value=5.0,
            value=1e3 * D_coldstop_nominal,
            step=0.025,
        )

        coldstop_x_offset_mm = st.slider(
            "Cold stop x-offset [mm] — nominal 0.0 mm",
            min_value=-1.5,
            max_value=1.5,
            value=1e3 * coldstop_x_offset_nominal,
            step=0.025,
            help="Physical lateral offset of the cold stop in the star-image/cold-stop plane.",
        )

        pupil_misconjugation_mm = st.slider(
            "Detector pupil mis-conjugation [mm] — nominal 0.0 mm",
            min_value=-20.0,
            max_value=20.0,
            value=1e3 * pupil_misconjugation_nominal,
            step=0.25,
            help=(
                "Axial detector offset from the nominal pupil conjugate plane. "
                "Positive means detector farther downstream from the cold stop."
            ),
        )

        tip_amp_nm = st.slider(
            "Tip amplitude [nm OPD] — nominal 0 nm",
            min_value=-300.0,
            max_value=300.0,
            value=tip_amp_nominal_nm,
            step=5.0,
        )

        astig_amp_nm = st.slider(
            "Astigmatism amplitude [nm OPD] — nominal 0 nm",
            min_value=-300.0,
            max_value=300.0,
            value=astig_amp_nominal_nm,
            step=5.0,
        )

        show_log_detector = st.checkbox(
            "Show detector intensity in log scale",
            value=False,
        )

        update = st.form_submit_button("Update simulation", type="primary")


# Run once on page load too.
if "has_run_once" not in st.session_state:
    st.session_state["has_run_once"] = True
    update = True


if update:

    # ============================================================
    # Convert controls to SI units
    # ============================================================

    edge_offset = edge_offset_mm * 1e-3
    z_mirror_to_lens = z_mirror_to_lens_mm * 1e-3
    D_coldstop = coldstop_diam_mm * 1e-3
    coldstop_x_offset = coldstop_x_offset_mm * 1e-3
    pupil_misconjugation = pupil_misconjugation_mm * 1e-3
    tip_amp = tip_amp_nm * 1e-9
    astig_amp = astig_amp_nm * 1e-9
    theta_phase_shift = theta_phase_shift_rad

    # ============================================================
    # 1. Initialise ZWFS from BaldrApp
    # ============================================================

    grid_dict = {
        "telescope": "solarstein",
        "D": 1.8,
        "N": 72,
        "dim": 72 * 4,
    }

    optics_dict = {
        "wvl0": 1.65e-6,
        "F_number": 21.2,
        "mask_diam": 1.06,
        "theta": theta_phase_shift,
        "coldstop_diam": 8.4,
        "coldstop_offset": (0, 0),
    }

    dm_dict = {
        "dm_model": "BMC-multi-3.5",
        "actuator_coupling_factor": 0.75,
        "dm_pitch": 1,
        "dm_aoi": 0,
        "opd_per_cmd": 3e-6,
        "flat_rmse": 0.0,
    }

    grid_ns = SimpleNamespace(**grid_dict)
    optics_ns = SimpleNamespace(**optics_dict)
    dm_ns = SimpleNamespace(**dm_dict)

    zwfs_ns = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)

    wavelength = zwfs_ns.optics.wvl0
    pupil = zwfs_ns.grid.pupil_mask
    pupil_bool = pupil.astype(bool)

    # ============================================================
    # 2. Input OPD: tip + astigmatism
    # ============================================================

    Xw = zwfs_ns.grid.wave_coord.X
    Yw = zwfs_ns.grid.wave_coord.Y

    x_norm = np.zeros_like(Xw, dtype=float)
    y_norm = np.zeros_like(Yw, dtype=float)

    x_norm[pupil_bool] = Xw[pupil_bool] / np.max(np.abs(Xw[pupil_bool]))
    y_norm[pupil_bool] = Yw[pupil_bool] / np.max(np.abs(Yw[pupil_bool]))

    tip_opd = pupil * tip_amp * x_norm
    astig_opd = pupil * astig_amp * (x_norm**2 - y_norm**2)
    opd = tip_opd + astig_opd

    phi = pupil * 2.0 * np.pi / wavelength * opd
    amp = pupil.copy()

    # ============================================================
    # 3. Analytic ZWFS output field
    # ============================================================

    zwfs_terms = bldr.get_zwfs_output_field(
        phi=phi,
        amp=amp,
        theta=zwfs_ns.optics.theta,
        phasemask_diameter=zwfs_ns.optics.mask_diam,
        phasemask_mask=None,
        pupil_diameter=zwfs_ns.grid.N,
        fplane_pixels=zwfs_ns.focal_plane.fplane_pixels,
        pixels_across_mask=zwfs_ns.focal_plane.pixels_across_mask,
        return_terms=True,
    )

    psi_A = zwfs_terms["psi_A"]
    b = zwfs_terms["b"]
    psi_C = zwfs_terms["psi_C"]

    # Flat reference for differences.
    zwfs_terms_flat = bldr.get_zwfs_output_field(
        phi=np.zeros_like(phi),
        amp=amp,
        theta=zwfs_ns.optics.theta,
        phasemask_diameter=zwfs_ns.optics.mask_diam,
        phasemask_mask=None,
        pupil_diameter=zwfs_ns.grid.N,
        fplane_pixels=zwfs_ns.focal_plane.fplane_pixels,
        pixels_across_mask=zwfs_ns.focal_plane.pixels_across_mask,
        return_terms=True,
    )

    psi_C_flat = zwfs_terms_flat["psi_C"]

    # ============================================================
    # 4. Physical propagation scale
    # ============================================================

    dx_phys = D_phys / zwfs_ns.grid.N

    X_phys, Y_phys = fresnel.make_coordinate_grid(psi_C.shape, dx_phys)
    R_phys = np.hypot(X_phys, Y_phys)

    # ============================================================
    # 5. Propagate to D/knife-edge mirror
    # ============================================================

    psi_mirror = fresnel.propagate(
        psi_C,
        wavelength=wavelength,
        dx=dx_phys,
        z=z_to_mirror,
        method="angular_spectrum",
    )

    psi_mirror_flat = fresnel.propagate(
        psi_C_flat,
        wavelength=wavelength,
        dx=dx_phys,
        z=z_to_mirror,
        method="angular_spectrum",
    )

    mirror_mask_full = fresnel.circular_aperture(
        psi_mirror.shape,
        dx=dx_phys,
        radius=D_mirror / 2,
    ).astype(float)

    edge_angle = 0.0
    Xr = X_phys * np.cos(edge_angle) + Y_phys * np.sin(edge_angle)
    mirror_mask_D = (Xr >= edge_offset).astype(float) * mirror_mask_full

    psi_after_mirror = psi_mirror * mirror_mask_D
    psi_after_mirror_flat = psi_mirror_flat * mirror_mask_D

    E_mirror = fresnel.energy(psi_mirror, dx_phys)
    E_after_mirror = fresnel.energy(psi_after_mirror, dx_phys)

    # ============================================================
    # 6. Re-imaging lens and cold stop
    # Correct physical relay:
    # mirror plane -> re-imaging lens -> cold-stop/star-image plane
    # ============================================================

    s_object = z_to_mirror + z_mirror_to_lens

    if s_object <= f_imaging:
        st.error(
            "Invalid lens geometry: object distance must be greater than f_imaging."
        )
        st.stop()

    s_image = 1.0 / (1.0 / f_imaging - 1.0 / s_object)

    z_focus_to_detector_nominal_current = s_image - f_imaging
    z_focus_to_detector = z_focus_to_detector_nominal_current + pupil_misconjugation

    if z_focus_to_detector <= 0:
        st.error("Invalid detector position: cold-stop to detector distance must be positive.")
        st.stop()

    M_pupil = s_image / s_object
    D_detector_predicted = M_pupil * D_phys

    # Propagate from D/knife-edge mirror plane to the 184 mm re-imaging lens.
    field_at_lens = fresnel.propagate(
        psi_after_mirror,
        wavelength=wavelength,
        dx=dx_phys,
        z=z_mirror_to_lens,
        method="angular_spectrum",
    )

    field_at_lens_flat = fresnel.propagate(
        psi_after_mirror_flat,
        wavelength=wavelength,
        dx=dx_phys,
        z=z_mirror_to_lens,
        method="angular_spectrum",
    )

    # Apply the 184 mm thin-lens phase.
    field_after_lens = fresnel.apply_thin_lens(
        field_at_lens,
        wavelength=wavelength,
        dx=dx_phys,
        focal_length=f_imaging,
    )

    field_after_lens_flat = fresnel.apply_thin_lens(
        field_at_lens_flat,
        wavelength=wavelength,
        dx=dx_phys,
        focal_length=f_imaging,
    )

    # Propagate from the re-imaging lens to the cold-stop/star-image plane.
    # Since the incoming field is close to collimated, the star image is near f_imaging.
    field_coldstop_plane, dx_cold, dy_cold = fresnel.fresnel_one_step_propagate(
        field_after_lens,
        wavelength=wavelength,
        dx=dx_phys,
        z=f_imaging,
        include_global_phase=False,
    )

    field_coldstop_plane_flat, _, _ = fresnel.fresnel_one_step_propagate(
        field_after_lens_flat,
        wavelength=wavelength,
        dx=dx_phys,
        z=f_imaging,
        include_global_phase=False,
    )

    X_cold, Y_cold = fresnel.make_coordinate_grid(
        field_coldstop_plane.shape,
        dx=dx_cold,
        dy=dy_cold,
    )

    R_cold = np.hypot(X_cold - coldstop_x_offset, Y_cold)
    coldstop_mask_phys = (R_cold <= D_coldstop / 2).astype(float)

    field_after_coldstop = field_coldstop_plane * coldstop_mask_phys
    field_after_coldstop_flat = field_coldstop_plane_flat * coldstop_mask_phys

    E_before_cold = np.sum(np.abs(field_coldstop_plane) ** 2)
    E_after_cold = np.sum(np.abs(field_after_coldstop) ** 2)

    # ============================================================
    # 7. Propagate from cold-stop/star-image plane to detector pupil plane
    # ============================================================

    field_detector_pupil, dx_detector, dy_detector = fresnel.fresnel_one_step_propagate(
        field_after_coldstop,
        wavelength=wavelength,
        dx=dx_cold,
        dy=dy_cold,
        z=z_focus_to_detector,
        include_global_phase=False,
    )

    field_detector_pupil_flat, _, _ = fresnel.fresnel_one_step_propagate(
        field_after_coldstop_flat,
        wavelength=wavelength,
        dx=dx_cold,
        dy=dy_cold,
        z=z_focus_to_detector,
        include_global_phase=False,
    )

    # ============================================================
    # 8. Intensities and normalizations
    # ============================================================

    I_zwfs = np.abs(psi_C) ** 2
    I_zwfs_flat = np.abs(psi_C_flat) ** 2

    I_det = np.abs(field_detector_pupil) ** 2
    I_det_flat = np.abs(field_detector_pupil_flat) ** 2

    I_zwfs_norm = I_zwfs / np.nanmax(I_zwfs_flat)
    I_zwfs_flat_norm = I_zwfs_flat / np.nanmax(I_zwfs_flat)
    dI_zwfs = I_zwfs_norm - I_zwfs_flat_norm

    I_det_norm = I_det / np.nanmax(I_det_flat)
    I_det_flat_norm = I_det_flat / np.nanmax(I_det_flat)
    dI_det = I_det_norm - I_det_flat_norm

    X_det, Y_det = fresnel.make_coordinate_grid(
        I_det.shape,
        dx=dx_detector,
        dy=dy_detector,
    )

    lambda_over_D_phys = wavelength * f_imaging / D_phys

    # ============================================================
    # Diagnostics
    # ============================================================

    st.subheader("Current geometry")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("D after 40 mm collimator", f"{1e3 * D_phys:.3f} mm")
    c2.metric("D-edge throughput", f"{E_after_mirror / E_mirror:.3f}")
    c3.metric("Cold-stop throughput", f"{E_after_cold / E_before_cold:.3f}")
    c4.metric("Predicted detector pupil", f"{1e6 * D_detector_predicted:.1f} µm")

    st.write(
        {
            "wavelength_um": 1e6 * wavelength,
            "theta_rad": theta_phase_shift,
            "theta_pi_units": theta_phase_shift / np.pi,
            "dx_phys_um_per_pix": 1e6 * dx_phys,
            "D_phys_mm": 1e3 * D_phys,
            "z_to_mirror_mm": 1e3 * z_to_mirror,
            "z_mirror_to_lens_mm": 1e3 * z_mirror_to_lens,
            "s_object_mm": 1e3 * s_object,
            "s_image_mm": 1e3 * s_image,
            "z_focus_to_detector_nominal_mm": 1e3 * z_focus_to_detector_nominal_current,
            "pupil_misconjugation_mm": 1e3 * pupil_misconjugation,
            "z_focus_to_detector_actual_mm": 1e3 * z_focus_to_detector,
            "M_pupil": M_pupil,
            "predicted_detector_pupil_um": 1e6 * D_detector_predicted,
            "dx_detector_um_per_pix": 1e6 * dx_detector,
            "predicted_detector_pupil_pix": D_detector_predicted / dx_detector,
            "coldstop_diam_lambda_over_D": D_coldstop / lambda_over_D_phys,
            "coldstop_x_offset_mm": 1e3 * coldstop_x_offset,
            "OPD_rms_nm": 1e9 * np.std(opd[pupil_bool]),
            "theta_zero_check_rel_norm_psiC_minus_psiA": (
                np.linalg.norm((psi_C - psi_A).ravel()) / np.linalg.norm(psi_A.ravel())
            ),
        }
    )

    # ============================================================
    # Big top plot: detector intensity only
    # ============================================================

    st.subheader("Detector pupil intensity")

    if show_log_detector:
        big_intensity_plot = np.log10(I_det_norm + 1e-8)
        big_title = "Detector pupil intensity, log10 scale"
        big_vmin = -8
        big_vmax = 0
        big_cbar = r"$\log_{10}$ normalized intensity"
    else:
        big_intensity_plot = I_det_norm
        big_title = "Detector pupil intensity"
        big_vmin = 0
        big_vmax = 1
        big_cbar = "Intensity / flat peak"

    fig_big, ax_big = plt.subplots(1, 1, figsize=(9, 8))

    im_big = ax_big.imshow(
        big_intensity_plot,
        origin="lower",
        extent=[
            1e6 * X_det.min(),
            1e6 * X_det.max(),
            1e6 * Y_det.min(),
            1e6 * Y_det.max(),
        ],
        vmin=big_vmin,
        vmax=big_vmax,
    )

    ax_big.set_title(big_title, fontsize=18)
    ax_big.set_xlabel("x [µm]", fontsize=14)
    ax_big.set_ylabel("y [µm]", fontsize=14)

    cbar_big = plt.colorbar(im_big, ax=ax_big, fraction=0.046, pad=0.04)
    cbar_big.set_label(big_cbar, fontsize=13)

    plt.tight_layout()
    st.pyplot(fig_big)

    # ============================================================
    # Plot 1: OPD, raw ZWFS response, detector response
    # ============================================================

    st.subheader("ZWFS and detector response")

    dmax = max(
        np.nanmax(np.abs(dI_zwfs)),
        np.nanmax(np.abs(dI_det)),
        1e-12,
    )

    if show_log_detector:
        det_plot = np.log10(I_det_norm + 1e-8)
        det_title = "Detector intensity, log10"
        det_vmin = -8
        det_vmax = 0
        det_cbar = "log10 normalized intensity"
    else:
        det_plot = I_det_norm
        det_title = "Detector intensity"
        det_vmin = 0
        det_vmax = 1
        det_cbar = "Intensity / flat peak"

    fig, axes = plt.subplots(1, 5, figsize=(21, 4.2))

    im0 = axes[0].imshow(1e9 * opd, origin="lower")
    axes[0].set_title("Input OPD [nm]")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(I_zwfs_norm, origin="lower", vmin=0)
    axes[1].set_title(r"Raw ZWFS $|\Psi_C|^2$")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(
        dI_zwfs,
        origin="lower",
        vmin=-dmax,
        vmax=dmax,
        cmap="RdBu_r",
    )
    axes[2].set_title("Raw ZWFS - flat")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(
        det_plot,
        origin="lower",
        extent=[
            1e6 * X_det.min(),
            1e6 * X_det.max(),
            1e6 * Y_det.min(),
            1e6 * Y_det.max(),
        ],
        vmin=det_vmin,
        vmax=det_vmax,
    )
    axes[3].set_title(det_title)
    axes[3].set_xlabel("x [µm]")
    axes[3].set_ylabel("y [µm]")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, label=det_cbar)

    im4 = axes[4].imshow(
        dI_det,
        origin="lower",
        extent=[
            1e6 * X_det.min(),
            1e6 * X_det.max(),
            1e6 * Y_det.min(),
            1e6 * Y_det.max(),
        ],
        vmin=-dmax,
        vmax=dmax,
        cmap="RdBu_r",
    )
    axes[4].set_title("Detector - flat")
    axes[4].set_xlabel("x [µm]")
    axes[4].set_ylabel("y [µm]")
    plt.colorbar(im4, ax=axes[4], fraction=0.046)

    for ax in axes[:3]:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    st.pyplot(fig)

    # ============================================================
    # Plot 2: D mirror and cold stop geometry
    # ============================================================

    st.subheader("Apertures and intermediate planes")

    I_mirror = np.abs(psi_mirror) ** 2
    I_lens = np.abs(field_at_lens) ** 2
    I_cold = np.abs(field_coldstop_plane) ** 2
    I_cold_norm = I_cold / np.nanmax(I_cold)

    X_cold_lamD = X_cold / lambda_over_D_phys
    Y_cold_lamD = Y_cold / lambda_over_D_phys

    fig2, axes2 = plt.subplots(1, 5, figsize=(22, 4.2))

    im0 = axes2[0].imshow(
        I_mirror / np.nanmax(I_mirror),
        origin="lower",
        extent=[
            1e3 * X_phys.min(),
            1e3 * X_phys.max(),
            1e3 * Y_phys.min(),
            1e3 * Y_phys.max(),
        ],
    )
    axes2[0].contour(
        1e3 * X_phys,
        1e3 * Y_phys,
        mirror_mask_D,
        levels=[0.5],
        colors="w",
    )
    axes2[0].set_title("Mirror plane")
    axes2[0].set_xlabel("x [mm]")
    axes2[0].set_ylabel("y [mm]")
    plt.colorbar(im0, ax=axes2[0], fraction=0.046)

    im1 = axes2[1].imshow(
        mirror_mask_D,
        origin="lower",
        extent=[
            1e3 * X_phys.min(),
            1e3 * X_phys.max(),
            1e3 * Y_phys.min(),
            1e3 * Y_phys.max(),
        ],
        vmin=0,
        vmax=1,
    )
    axes2[1].set_title("D/knife-edge mask")
    axes2[1].set_xlabel("x [mm]")
    axes2[1].set_ylabel("y [mm]")
    plt.colorbar(im1, ax=axes2[1], fraction=0.046)

    im2 = axes2[2].imshow(
        I_lens / np.nanmax(I_lens),
        origin="lower",
        extent=[
            1e3 * X_phys.min(),
            1e3 * X_phys.max(),
            1e3 * Y_phys.min(),
            1e3 * Y_phys.max(),
        ],
    )
    axes2[2].set_title("Field at re-imaging lens")
    axes2[2].set_xlabel("x [mm]")
    axes2[2].set_ylabel("y [mm]")
    plt.colorbar(im2, ax=axes2[2], fraction=0.046)

    im3 = axes2[3].imshow(
        np.log10(I_cold_norm + 1e-8),
        origin="lower",
        extent=[
            X_cold_lamD.min(),
            X_cold_lamD.max(),
            Y_cold_lamD.min(),
            Y_cold_lamD.max(),
        ],
        vmin=-8,
        vmax=0,
    )
    axes2[3].contour(
        X_cold_lamD,
        Y_cold_lamD,
        coldstop_mask_phys,
        levels=[0.5],
        colors="w",
    )
    axes2[3].set_xlim(-10, 10)
    axes2[3].set_ylim(-10, 10)
    axes2[3].set_title("Cold stop plane")
    axes2[3].set_xlabel(r"$\lambda/D$")
    axes2[3].set_ylabel(r"$\lambda/D$")
    plt.colorbar(im3, ax=axes2[3], fraction=0.046)

    im4 = axes2[4].imshow(
        coldstop_mask_phys,
        origin="lower",
        extent=[
            1e3 * X_cold.min(),
            1e3 * X_cold.max(),
            1e3 * Y_cold.min(),
            1e3 * Y_cold.max(),
        ],
        vmin=0,
        vmax=1,
    )
    axes2[4].set_title(f"Cold stop mask\nx offset={1e3 * coldstop_x_offset:+.2f} mm")
    axes2[4].set_xlabel("x [mm]")
    axes2[4].set_ylabel("y [mm]")
    plt.colorbar(im4, ax=axes2[4], fraction=0.046)

    plt.tight_layout()
    st.pyplot(fig2)

    plt.close("all")

else:
    st.info("Adjust sliders and press **Update simulation**.")


# import sys
# from pathlib import Path
# from types import SimpleNamespace

# import numpy as np
# import matplotlib.pyplot as plt
# import streamlit as st

# # If running from installed package this is fine, but this helps when running local source.
# repo_root = Path(__file__).resolve().parents[2]
# if str(repo_root) not in sys.path:
#     sys.path.insert(0, str(repo_root))

# from baldrapp.common import baldr_core as bldr
# from baldrapp.common import fresnel


# # ============================================================
# # Streamlit page setup
# # ============================================================

# st.set_page_config(
#     page_title="Baldr ZWFS Fresnel Relay Explorer",
#     layout="wide",
# )

# st.title("Baldr ZWFS Fresnel Relay Explorer")

# st.markdown(
#     """
# This app evaluates:

# `analytic ZWFS field` → `free-space to D/knife-edge mirror` → `D-edge vignetting`
# → `propagate to 184 mm re-imaging lens` → `thin-lens phase`
# → `physical cold stop / star-image plane` → `detector pupil plane`.

# The simulation only updates when **Update simulation** is pressed.
# """
# )


# # ============================================================
# # Hardcoded nominal physical parameters
# # ============================================================

# D_entrance = 12e-3          # m, Baldr input pupil diameter before OAP
# f_oap = 254e-3              # m, Baldr OAP focal length
# f_coll = 40e-3              # m, updated post-mask collimating lens
# z_to_mirror = 0.50          # m, collimator output / ZWFS pupil plane to D/knife-edge mirror
# D_mirror = 25.4e-3          # m, 1 inch mirror clear aperture
# f_imaging = 184e-3          # m, Baldr re-imaging lens
# D_coldstop_nominal = 2.145e-3
# D_detector_pupil_nominal = 288e-6

# # Physical beam diameter after 40 mm collimator.
# D_phys = D_entrance * f_coll / f_oap

# # Nominal pupil magnification and lens conjugates for ~288 um detector pupil.
# M_nominal = D_detector_pupil_nominal / D_phys
# s_object_nominal = f_imaging * (1.0 + 1.0 / M_nominal)
# s_image_nominal = f_imaging * (1.0 + M_nominal)
# z_mirror_to_lens_nominal = s_object_nominal - z_to_mirror
# z_focus_to_detector_nominal = s_image_nominal - f_imaging

# edge_offset_nominal = -1.0e-3
# tip_amp_nominal_nm = 0.0
# astig_amp_nominal_nm = 0.0
# theta_nominal = 1.57079  # rad, approximately pi/2


# # ============================================================
# # Controls: form means no live update while sliding
# # ============================================================

# with st.sidebar:
#     st.header("Controls")
#     st.caption("Nominal values are shown in each slider label.")

#     with st.form("simulation_controls"):

#         theta_phase_shift_rad = st.slider(
#             r"ZWFS phase shift θ [rad] — nominal π/2",
#             min_value=0.0,
#             max_value=float(np.pi),
#             value=float(theta_nominal),
#             step=0.01,
#             help="Phase shift applied by the ZWFS focal-plane phase mask.",
#         )

#         st.caption(
#             f"θ = {theta_phase_shift_rad:.3f} rad "
#             f"= {theta_phase_shift_rad / np.pi:.3f}π"
#         )

#         edge_offset_mm = st.slider(
#             "Knife/D-edge offset [mm] — nominal -1.0 mm",
#             min_value=-3.0,
#             max_value=1.5,
#             value=1e3 * edge_offset_nominal,
#             step=0.05,
#             help=(
#                 "Half-plane mask keeps X >= edge_offset. "
#                 "More positive values clip more of the beam."
#             ),
#         )

#         z_mirror_to_lens_mm = st.slider(
#             f"Mirror → re-imaging lens distance [mm] — nominal {1e3 * z_mirror_to_lens_nominal:.1f} mm",
#             min_value=300.0,
#             max_value=1300.0,
#             value=float(1e3 * z_mirror_to_lens_nominal),
#             step=5.0,
#             help=(
#                 "Controls the object distance for pupil imaging. "
#                 "The detector distance is recomputed from the thin-lens formula."
#             ),
#         )

#         coldstop_diam_mm = st.slider(
#             "Cold stop diameter [mm] — nominal 2.145 mm",
#             min_value=0.5,
#             max_value=5.0,
#             value=1e3 * D_coldstop_nominal,
#             step=0.025,
#         )

#         tip_amp_nm = st.slider(
#             "Tip amplitude [nm OPD] — nominal 0 nm",
#             min_value=-300.0,
#             max_value=300.0,
#             value=tip_amp_nominal_nm,
#             step=5.0,
#         )

#         astig_amp_nm = st.slider(
#             "Astigmatism amplitude [nm OPD] — nominal 0 nm",
#             min_value=-300.0,
#             max_value=300.0,
#             value=astig_amp_nominal_nm,
#             step=5.0,
#         )

#         show_log_detector = st.checkbox(
#             "Show detector intensity in log scale",
#             value=False,
#         )

#         update = st.form_submit_button("Update simulation", type="primary")


# # Run once on page load too.
# if "has_run_once" not in st.session_state:
#     st.session_state["has_run_once"] = True
#     update = True


# if update:

#     # ============================================================
#     # Convert controls to SI units
#     # ============================================================

#     edge_offset = edge_offset_mm * 1e-3
#     z_mirror_to_lens = z_mirror_to_lens_mm * 1e-3
#     D_coldstop = coldstop_diam_mm * 1e-3
#     tip_amp = tip_amp_nm * 1e-9
#     astig_amp = astig_amp_nm * 1e-9
#     theta_phase_shift = theta_phase_shift_rad

#     # ============================================================
#     # 1. Initialise ZWFS from BaldrApp
#     # ============================================================

#     grid_dict = {
#         "telescope": "solarstein",
#         "D": 1.8,
#         "N": 72,
#         "dim": 72 * 4,
#     }

#     optics_dict = {
#         "wvl0": 1.65e-6,
#         "F_number": 21.2,
#         "mask_diam": 1.06,
#         "theta": theta_phase_shift,
#         "coldstop_diam": 8.4,
#         "coldstop_offset": (0, 0),
#     }

#     dm_dict = {
#         "dm_model": "BMC-multi-3.5",
#         "actuator_coupling_factor": 0.75,
#         "dm_pitch": 1,
#         "dm_aoi": 0,
#         "opd_per_cmd": 3e-6,
#         "flat_rmse": 0.0,
#     }

#     grid_ns = SimpleNamespace(**grid_dict)
#     optics_ns = SimpleNamespace(**optics_dict)
#     dm_ns = SimpleNamespace(**dm_dict)

#     zwfs_ns = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)

#     wavelength = zwfs_ns.optics.wvl0
#     pupil = zwfs_ns.grid.pupil_mask
#     pupil_bool = pupil.astype(bool)

#     # ============================================================
#     # 2. Input OPD: tip + astigmatism
#     # ============================================================

#     Xw = zwfs_ns.grid.wave_coord.X
#     Yw = zwfs_ns.grid.wave_coord.Y

#     x_norm = np.zeros_like(Xw, dtype=float)
#     y_norm = np.zeros_like(Yw, dtype=float)

#     x_norm[pupil_bool] = Xw[pupil_bool] / np.max(np.abs(Xw[pupil_bool]))
#     y_norm[pupil_bool] = Yw[pupil_bool] / np.max(np.abs(Yw[pupil_bool]))

#     tip_opd = pupil * tip_amp * x_norm
#     astig_opd = pupil * astig_amp * (x_norm**2 - y_norm**2)
#     opd = tip_opd + astig_opd

#     phi = pupil * 2.0 * np.pi / wavelength * opd
#     amp = pupil.copy()

#     # ============================================================
#     # 3. Analytic ZWFS output field
#     # ============================================================

#     zwfs_terms = bldr.get_zwfs_output_field(
#         phi=phi,
#         amp=amp,
#         theta=zwfs_ns.optics.theta,
#         phasemask_diameter=zwfs_ns.optics.mask_diam,
#         phasemask_mask=None,
#         pupil_diameter=zwfs_ns.grid.N,
#         fplane_pixels=zwfs_ns.focal_plane.fplane_pixels,
#         pixels_across_mask=zwfs_ns.focal_plane.pixels_across_mask,
#         return_terms=True,
#     )

#     psi_A = zwfs_terms["psi_A"]
#     b = zwfs_terms["b"]
#     psi_C = zwfs_terms["psi_C"]

#     # Flat reference for differences.
#     zwfs_terms_flat = bldr.get_zwfs_output_field(
#         phi=np.zeros_like(phi),
#         amp=amp,
#         theta=zwfs_ns.optics.theta,
#         phasemask_diameter=zwfs_ns.optics.mask_diam,
#         phasemask_mask=None,
#         pupil_diameter=zwfs_ns.grid.N,
#         fplane_pixels=zwfs_ns.focal_plane.fplane_pixels,
#         pixels_across_mask=zwfs_ns.focal_plane.pixels_across_mask,
#         return_terms=True,
#     )

#     psi_C_flat = zwfs_terms_flat["psi_C"]

#     # ============================================================
#     # 4. Physical propagation scale
#     # ============================================================

#     dx_phys = D_phys / zwfs_ns.grid.N

#     X_phys, Y_phys = fresnel.make_coordinate_grid(psi_C.shape, dx_phys)
#     R_phys = np.hypot(X_phys, Y_phys)

#     # ============================================================
#     # 5. Propagate to D/knife-edge mirror
#     # ============================================================

#     psi_mirror = fresnel.propagate(
#         psi_C,
#         wavelength=wavelength,
#         dx=dx_phys,
#         z=z_to_mirror,
#         method="angular_spectrum",
#     )

#     psi_mirror_flat = fresnel.propagate(
#         psi_C_flat,
#         wavelength=wavelength,
#         dx=dx_phys,
#         z=z_to_mirror,
#         method="angular_spectrum",
#     )

#     mirror_mask_full = fresnel.circular_aperture(
#         psi_mirror.shape,
#         dx=dx_phys,
#         radius=D_mirror / 2,
#     ).astype(float)

#     edge_angle = 0.0
#     Xr = X_phys * np.cos(edge_angle) + Y_phys * np.sin(edge_angle)
#     mirror_mask_D = (Xr >= edge_offset).astype(float) * mirror_mask_full

#     psi_after_mirror = psi_mirror * mirror_mask_D
#     psi_after_mirror_flat = psi_mirror_flat * mirror_mask_D

#     E_mirror = fresnel.energy(psi_mirror, dx_phys)
#     E_after_mirror = fresnel.energy(psi_after_mirror, dx_phys)

#     # ============================================================
#     # 6. Re-imaging lens and cold stop
#     # Correct physical relay:
#     # mirror plane -> re-imaging lens -> cold-stop/star-image plane
#     # ============================================================

#     s_object = z_to_mirror + z_mirror_to_lens

#     if s_object <= f_imaging:
#         st.error(
#             "Invalid lens geometry: object distance must be greater than f_imaging."
#         )
#         st.stop()

#     s_image = 1.0 / (1.0 / f_imaging - 1.0 / s_object)
#     z_focus_to_detector = s_image - f_imaging
#     M_pupil = s_image / s_object
#     D_detector_predicted = M_pupil * D_phys

#     # Propagate from D/knife-edge mirror plane to the 184 mm re-imaging lens.
#     field_at_lens = fresnel.propagate(
#         psi_after_mirror,
#         wavelength=wavelength,
#         dx=dx_phys,
#         z=z_mirror_to_lens,
#         method="angular_spectrum",
#     )

#     field_at_lens_flat = fresnel.propagate(
#         psi_after_mirror_flat,
#         wavelength=wavelength,
#         dx=dx_phys,
#         z=z_mirror_to_lens,
#         method="angular_spectrum",
#     )

#     # Apply the 184 mm thin-lens phase.
#     field_after_lens = fresnel.apply_thin_lens(
#         field_at_lens,
#         wavelength=wavelength,
#         dx=dx_phys,
#         focal_length=f_imaging,
#     )

#     field_after_lens_flat = fresnel.apply_thin_lens(
#         field_at_lens_flat,
#         wavelength=wavelength,
#         dx=dx_phys,
#         focal_length=f_imaging,
#     )

#     # Propagate from the re-imaging lens to the cold-stop/star-image plane.
#     # Since the incoming field is close to collimated, the star image is near f_imaging.
#     field_coldstop_plane, dx_cold, dy_cold = fresnel.fresnel_one_step_propagate(
#         field_after_lens,
#         wavelength=wavelength,
#         dx=dx_phys,
#         z=f_imaging,
#         include_global_phase=False,
#     )

#     field_coldstop_plane_flat, _, _ = fresnel.fresnel_one_step_propagate(
#         field_after_lens_flat,
#         wavelength=wavelength,
#         dx=dx_phys,
#         z=f_imaging,
#         include_global_phase=False,
#     )

#     X_cold, Y_cold = fresnel.make_coordinate_grid(
#         field_coldstop_plane.shape,
#         dx=dx_cold,
#         dy=dy_cold,
#     )

#     R_cold = np.hypot(X_cold, Y_cold)
#     coldstop_mask_phys = (R_cold <= D_coldstop / 2).astype(float)

#     field_after_coldstop = field_coldstop_plane * coldstop_mask_phys
#     field_after_coldstop_flat = field_coldstop_plane_flat * coldstop_mask_phys

#     E_before_cold = np.sum(np.abs(field_coldstop_plane) ** 2)
#     E_after_cold = np.sum(np.abs(field_after_coldstop) ** 2)

#     # ============================================================
#     # 7. Propagate from cold-stop/star-image plane to detector pupil plane
#     # ============================================================

#     field_detector_pupil, dx_detector, dy_detector = fresnel.fresnel_one_step_propagate(
#         field_after_coldstop,
#         wavelength=wavelength,
#         dx=dx_cold,
#         dy=dy_cold,
#         z=z_focus_to_detector,
#         include_global_phase=False,
#     )

#     field_detector_pupil_flat, _, _ = fresnel.fresnel_one_step_propagate(
#         field_after_coldstop_flat,
#         wavelength=wavelength,
#         dx=dx_cold,
#         dy=dy_cold,
#         z=z_focus_to_detector,
#         include_global_phase=False,
#     )

#     # ============================================================
#     # 8. Intensities and normalizations
#     # ============================================================

#     I_zwfs = np.abs(psi_C) ** 2
#     I_zwfs_flat = np.abs(psi_C_flat) ** 2

#     I_det = np.abs(field_detector_pupil) ** 2
#     I_det_flat = np.abs(field_detector_pupil_flat) ** 2

#     I_zwfs_norm = I_zwfs / np.nanmax(I_zwfs_flat)
#     I_zwfs_flat_norm = I_zwfs_flat / np.nanmax(I_zwfs_flat)
#     dI_zwfs = I_zwfs_norm - I_zwfs_flat_norm

#     I_det_norm = I_det / np.nanmax(I_det_flat)
#     I_det_flat_norm = I_det_flat / np.nanmax(I_det_flat)
#     dI_det = I_det_norm - I_det_flat_norm

#     X_det, Y_det = fresnel.make_coordinate_grid(
#         I_det.shape,
#         dx=dx_detector,
#         dy=dy_detector,
#     )

#     lambda_over_D_phys = wavelength * f_imaging / D_phys

#     # ============================================================
#     # Diagnostics
#     # ============================================================

#     st.subheader("Current geometry")

#     c1, c2, c3, c4 = st.columns(4)

#     c1.metric("D after 40 mm collimator", f"{1e3 * D_phys:.3f} mm")
#     c2.metric("D-edge throughput", f"{E_after_mirror / E_mirror:.3f}")
#     c3.metric("Cold-stop throughput", f"{E_after_cold / E_before_cold:.3f}")
#     c4.metric("Predicted detector pupil", f"{1e6 * D_detector_predicted:.1f} µm")

#     st.write(
#         {
#             "wavelength_um": 1e6 * wavelength,
#             "theta_rad": theta_phase_shift,
#             "theta_pi_units": theta_phase_shift / np.pi,
#             "dx_phys_um_per_pix": 1e6 * dx_phys,
#             "D_phys_mm": 1e3 * D_phys,
#             "z_to_mirror_mm": 1e3 * z_to_mirror,
#             "z_mirror_to_lens_mm": 1e3 * z_mirror_to_lens,
#             "s_object_mm": 1e3 * s_object,
#             "s_image_mm": 1e3 * s_image,
#             "z_focus_to_detector_mm": 1e3 * z_focus_to_detector,
#             "M_pupil": M_pupil,
#             "predicted_detector_pupil_um": 1e6 * D_detector_predicted,
#             "dx_detector_um_per_pix": 1e6 * dx_detector,
#             "predicted_detector_pupil_pix": D_detector_predicted / dx_detector,
#             "coldstop_diam_lambda_over_D": D_coldstop / lambda_over_D_phys,
#             "OPD_rms_nm": 1e9 * np.std(opd[pupil_bool]),
#             "theta_zero_check_rel_norm_psiC_minus_psiA": (
#                 np.linalg.norm((psi_C - psi_A).ravel()) / np.linalg.norm(psi_A.ravel())
#             ),
#         }
#     )

#     # ============================================================
#     # Big top plot: detector intensity only
#     # ============================================================

#     st.subheader("Detector pupil intensity")

#     if show_log_detector:
#         big_intensity_plot = np.log10(I_det_norm + 1e-8)
#         big_title = "Detector pupil intensity, log10 scale"
#         big_vmin = -8
#         big_vmax = 0
#         big_cbar = r"$\log_{10}$ normalized intensity"
#     else:
#         big_intensity_plot = I_det_norm
#         big_title = "Detector pupil intensity"
#         big_vmin = 0
#         big_vmax = 1
#         big_cbar = "Intensity / flat peak"

#     fig_big, ax_big = plt.subplots(1, 1, figsize=(9, 8))

#     im_big = ax_big.imshow(
#         big_intensity_plot,
#         origin="lower",
#         extent=[
#             1e6 * X_det.min(),
#             1e6 * X_det.max(),
#             1e6 * Y_det.min(),
#             1e6 * Y_det.max(),
#         ],
#         vmin=big_vmin,
#         vmax=big_vmax,
#     )

#     ax_big.set_title(big_title, fontsize=18)
#     ax_big.set_xlabel("x [µm]", fontsize=14)
#     ax_big.set_ylabel("y [µm]", fontsize=14)

#     cbar_big = plt.colorbar(im_big, ax=ax_big, fraction=0.046, pad=0.04)
#     cbar_big.set_label(big_cbar, fontsize=13)

#     plt.tight_layout()
#     st.pyplot(fig_big)

#     # ============================================================
#     # Plot 1: OPD, raw ZWFS response, detector response
#     # ============================================================

#     st.subheader("ZWFS and detector response")

#     dmax = max(
#         np.nanmax(np.abs(dI_zwfs)),
#         np.nanmax(np.abs(dI_det)),
#         1e-12,
#     )

#     if show_log_detector:
#         det_plot = np.log10(I_det_norm + 1e-8)
#         det_title = "Detector intensity, log10"
#         det_vmin = -8
#         det_vmax = 0
#         det_cbar = "log10 normalized intensity"
#     else:
#         det_plot = I_det_norm
#         det_title = "Detector intensity"
#         det_vmin = 0
#         det_vmax = 1
#         det_cbar = "Intensity / flat peak"

#     fig, axes = plt.subplots(1, 5, figsize=(21, 4.2))

#     im0 = axes[0].imshow(1e9 * opd, origin="lower")
#     axes[0].set_title("Input OPD [nm]")
#     plt.colorbar(im0, ax=axes[0], fraction=0.046)

#     im1 = axes[1].imshow(I_zwfs_norm, origin="lower", vmin=0)
#     axes[1].set_title(r"Raw ZWFS $|\Psi_C|^2$")
#     plt.colorbar(im1, ax=axes[1], fraction=0.046)

#     im2 = axes[2].imshow(
#         dI_zwfs,
#         origin="lower",
#         vmin=-dmax,
#         vmax=dmax,
#         cmap="RdBu_r",
#     )
#     axes[2].set_title("Raw ZWFS - flat")
#     plt.colorbar(im2, ax=axes[2], fraction=0.046)

#     im3 = axes[3].imshow(
#         det_plot,
#         origin="lower",
#         extent=[
#             1e6 * X_det.min(),
#             1e6 * X_det.max(),
#             1e6 * Y_det.min(),
#             1e6 * Y_det.max(),
#         ],
#         vmin=det_vmin,
#         vmax=det_vmax,
#     )
#     axes[3].set_title(det_title)
#     axes[3].set_xlabel("x [µm]")
#     axes[3].set_ylabel("y [µm]")
#     plt.colorbar(im3, ax=axes[3], fraction=0.046, label=det_cbar)

#     im4 = axes[4].imshow(
#         dI_det,
#         origin="lower",
#         extent=[
#             1e6 * X_det.min(),
#             1e6 * X_det.max(),
#             1e6 * Y_det.min(),
#             1e6 * Y_det.max(),
#         ],
#         vmin=-dmax,
#         vmax=dmax,
#         cmap="RdBu_r",
#     )
#     axes[4].set_title("Detector - flat")
#     axes[4].set_xlabel("x [µm]")
#     axes[4].set_ylabel("y [µm]")
#     plt.colorbar(im4, ax=axes[4], fraction=0.046)

#     for ax in axes[:3]:
#         ax.set_xticks([])
#         ax.set_yticks([])

#     plt.tight_layout()
#     st.pyplot(fig)

#     # ============================================================
#     # Plot 2: D mirror and cold stop geometry
#     # ============================================================

#     st.subheader("Apertures and intermediate planes")

#     I_mirror = np.abs(psi_mirror) ** 2
#     I_lens = np.abs(field_at_lens) ** 2
#     I_cold = np.abs(field_coldstop_plane) ** 2
#     I_cold_norm = I_cold / np.nanmax(I_cold)

#     X_cold_lamD = X_cold / lambda_over_D_phys
#     Y_cold_lamD = Y_cold / lambda_over_D_phys

#     fig2, axes2 = plt.subplots(1, 5, figsize=(22, 4.2))

#     im0 = axes2[0].imshow(
#         I_mirror / np.nanmax(I_mirror),
#         origin="lower",
#         extent=[
#             1e3 * X_phys.min(),
#             1e3 * X_phys.max(),
#             1e3 * Y_phys.min(),
#             1e3 * Y_phys.max(),
#         ],
#     )
#     axes2[0].contour(
#         1e3 * X_phys,
#         1e3 * Y_phys,
#         mirror_mask_D,
#         levels=[0.5],
#         colors="w",
#     )
#     axes2[0].set_title("Mirror plane")
#     axes2[0].set_xlabel("x [mm]")
#     axes2[0].set_ylabel("y [mm]")
#     plt.colorbar(im0, ax=axes2[0], fraction=0.046)

#     im1 = axes2[1].imshow(
#         mirror_mask_D,
#         origin="lower",
#         extent=[
#             1e3 * X_phys.min(),
#             1e3 * X_phys.max(),
#             1e3 * Y_phys.min(),
#             1e3 * Y_phys.max(),
#         ],
#         vmin=0,
#         vmax=1,
#     )
#     axes2[1].set_title("D/knife-edge mask")
#     axes2[1].set_xlabel("x [mm]")
#     axes2[1].set_ylabel("y [mm]")
#     plt.colorbar(im1, ax=axes2[1], fraction=0.046)

#     im2 = axes2[2].imshow(
#         I_lens / np.nanmax(I_lens),
#         origin="lower",
#         extent=[
#             1e3 * X_phys.min(),
#             1e3 * X_phys.max(),
#             1e3 * Y_phys.min(),
#             1e3 * Y_phys.max(),
#         ],
#     )
#     axes2[2].set_title("Field at re-imaging lens")
#     axes2[2].set_xlabel("x [mm]")
#     axes2[2].set_ylabel("y [mm]")
#     plt.colorbar(im2, ax=axes2[2], fraction=0.046)

#     im3 = axes2[3].imshow(
#         np.log10(I_cold_norm + 1e-8),
#         origin="lower",
#         extent=[
#             X_cold_lamD.min(),
#             X_cold_lamD.max(),
#             Y_cold_lamD.min(),
#             Y_cold_lamD.max(),
#         ],
#         vmin=-8,
#         vmax=0,
#     )
#     axes2[3].contour(
#         X_cold_lamD,
#         Y_cold_lamD,
#         coldstop_mask_phys,
#         levels=[0.5],
#         colors="w",
#     )
#     axes2[3].set_xlim(-10, 10)
#     axes2[3].set_ylim(-10, 10)
#     axes2[3].set_title("Cold stop plane")
#     axes2[3].set_xlabel(r"$\lambda/D$")
#     axes2[3].set_ylabel(r"$\lambda/D$")
#     plt.colorbar(im3, ax=axes2[3], fraction=0.046)

#     im4 = axes2[4].imshow(
#         coldstop_mask_phys,
#         origin="lower",
#         extent=[
#             1e3 * X_cold.min(),
#             1e3 * X_cold.max(),
#             1e3 * Y_cold.min(),
#             1e3 * Y_cold.max(),
#         ],
#         vmin=0,
#         vmax=1,
#     )
#     axes2[4].set_title("Physical cold stop mask")
#     axes2[4].set_xlabel("x [mm]")
#     axes2[4].set_ylabel("y [mm]")
#     plt.colorbar(im4, ax=axes2[4], fraction=0.046)

#     plt.tight_layout()
#     st.pyplot(fig2)

#     plt.close("all")

# else:
#     st.info("Adjust sliders and press **Update simulation**.")

# # import sys
# # from pathlib import Path
# # from types import SimpleNamespace

# # import numpy as np
# # import matplotlib.pyplot as plt
# # import streamlit as st

# # # If running from installed package this is fine, but this helps when running local source.
# # repo_root = Path(__file__).resolve().parents[2]
# # if str(repo_root) not in sys.path:
# #     sys.path.insert(0, str(repo_root))

# # from baldrapp.common import baldr_core as bldr
# # from baldrapp.common import fresnel


# # # ============================================================
# # # Streamlit page setup
# # # ============================================================

# # st.set_page_config(
# #     page_title="Baldr ZWFS Fresnel Relay Explorer",
# #     layout="wide",
# # )

# # st.title("Baldr ZWFS Fresnel Relay Explorer")

# # st.markdown(
# #     """
# # This app evaluates:

# # `analytic ZWFS field` --> `free-space to D/knife-edge mirror` --> `D-edge vignetting`
# # --> `184 mm re-imaging lens` --> `physical cold stop` --> `detector pupil plane`.

# # The simulation only updates when **Update simulation** is pressed.
# # """
# # )


# # # ============================================================
# # # Hardcoded nominal physical parameters
# # # ============================================================

# # D_entrance = 12e-3          # m, Baldr input pupil diameter before OAP
# # f_oap = 254e-3              # m
# # f_coll = 40e-3              # m, updated collimating lens
# # z_to_mirror = 0.50          # m, collimator output to D/knife-edge mirror
# # D_mirror = 25.4e-3          # m, 1 inch mirror clear aperture
# # f_imaging = 184e-3          # m, re-imaging lens
# # D_coldstop_nominal = 2.145e-3
# # D_detector_pupil_nominal = 288e-6

# # D_phys = D_entrance * f_coll / f_oap
# # M_nominal = D_detector_pupil_nominal / D_phys
# # s_object_nominal = f_imaging * (1.0 + 1.0 / M_nominal)
# # s_image_nominal = f_imaging * (1.0 + M_nominal)
# # z_mirror_to_lens_nominal = s_object_nominal - z_to_mirror
# # z_focus_to_detector_nominal = s_image_nominal - f_imaging

# # edge_offset_nominal = -1.0e-3
# # tip_amp_nominal_nm = 0.0
# # astig_amp_nominal_nm = 0.0


# # theta_nominal = 1.57079  # rad, approximately pi/2

# # # ============================================================
# # # Controls: form means no live update while sliding
# # # ============================================================

# # with st.sidebar:
# #     st.header("Controls")

# #     st.caption("Nominal values are shown in each slider label.")

# #     with st.form("simulation_controls"):

# #         theta_phase_shift_rad = st.slider(
# #             r"ZWFS phase shift θ [rad] — nominal pi/2",
# #             min_value=0.0,
# #             max_value=float(np.pi),
# #             value=float(theta_nominal),
# #             step=0.01,
# #             help="Phase shift applied by the ZWFS focal-plane phase mask.",
# #         )
                
# #         edge_offset_mm = st.slider(
# #             "Knife/D-edge offset [mm] — nominal -1.0 mm",
# #             min_value=-3.0,
# #             max_value=1.5,
# #             value=1e3 * edge_offset_nominal,
# #             step=0.05,
# #             help=(
# #                 "Half-plane mask keeps X >= edge_offset. "
# #                 "More positive values clip more of the beam."
# #             ),
# #         )

# #         z_mirror_to_lens_mm = st.slider(
# #             f"Mirror --> re-imaging lens distance [mm] — nominal {1e3 * z_mirror_to_lens_nominal:.1f} mm",
# #             min_value=300.0,
# #             max_value=1300.0,
# #             value=float(1e3 * z_mirror_to_lens_nominal),
# #             step=5.0,
# #             help=(
# #                 "Controls the object distance for pupil imaging. "
# #                 "The detector distance is recomputed from the thin-lens formula."
# #             ),
# #         )

# #         coldstop_diam_mm = st.slider(
# #             "Cold stop diameter [mm] — nominal 2.145 mm",
# #             min_value=0.5,
# #             max_value=5.0,
# #             value=1e3 * D_coldstop_nominal,
# #             step=0.025,
# #         )

# #         tip_amp_nm = st.slider(
# #             "Tip amplitude [nm OPD] — nominal 0 nm",
# #             min_value=-300.0,
# #             max_value=300.0,
# #             value=tip_amp_nominal_nm,
# #             step=5.0,
# #         )

# #         astig_amp_nm = st.slider(
# #             "Astigmatism amplitude [nm OPD] — nominal 0 nm",
# #             min_value=-300.0,
# #             max_value=300.0,
# #             value=astig_amp_nominal_nm,
# #             step=5.0,
# #         )

# #         show_log_detector = st.checkbox(
# #             "Show detector intensity in log scale",
# #             value=False,
# #         )

# #         update = st.form_submit_button("Update simulation", type="primary")


# # # Run once on page load too
# # if "has_run_once" not in st.session_state:
# #     st.session_state["has_run_once"] = True
# #     update = True


# # if update:

# #     edge_offset = edge_offset_mm * 1e-3
# #     z_mirror_to_lens = z_mirror_to_lens_mm * 1e-3
# #     D_coldstop = coldstop_diam_mm * 1e-3
# #     tip_amp = tip_amp_nm * 1e-9
# #     astig_amp = astig_amp_nm * 1e-9
# #     theta_phase_shift = theta_phase_shift_rad
# #     # ============================================================
# #     # 1. Initialise ZWFS from BaldrApp
# #     # ============================================================

# #     grid_dict = {
# #         "telescope": "solarstein",
# #         "D": 1.8,
# #         "N": 72,
# #         "dim": 72 * 4,
# #     }

# #     optics_dict = {
# #         "wvl0": 1.65e-6,
# #         "F_number": 21.2,
# #         "mask_diam": 1.06,
# #         "theta": theta_phase_shift,
# #         "coldstop_diam": 8.4,
# #         "coldstop_offset": (0, 0),
# #     }

# #     dm_dict = {
# #         "dm_model": "BMC-multi-3.5",
# #         "actuator_coupling_factor": 0.75,
# #         "dm_pitch": 1,
# #         "dm_aoi": 0,
# #         "opd_per_cmd": 3e-6,
# #         "flat_rmse": 0.0,
# #     }

# #     grid_ns = SimpleNamespace(**grid_dict)
# #     optics_ns = SimpleNamespace(**optics_dict)
# #     dm_ns = SimpleNamespace(**dm_dict)

# #     zwfs_ns = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)

# #     wavelength = zwfs_ns.optics.wvl0
# #     pupil = zwfs_ns.grid.pupil_mask
# #     pupil_bool = pupil.astype(bool)

# #     # ============================================================
# #     # 2. Input OPD: tip + astigmatism
# #     # ============================================================

# #     Xw = zwfs_ns.grid.wave_coord.X
# #     Yw = zwfs_ns.grid.wave_coord.Y

# #     x_norm = np.zeros_like(Xw, dtype=float)
# #     y_norm = np.zeros_like(Yw, dtype=float)

# #     x_norm[pupil_bool] = Xw[pupil_bool] / np.max(np.abs(Xw[pupil_bool]))
# #     y_norm[pupil_bool] = Yw[pupil_bool] / np.max(np.abs(Yw[pupil_bool]))

# #     tip_opd = pupil * tip_amp * x_norm
# #     astig_opd = pupil * astig_amp * (x_norm**2 - y_norm**2)
# #     opd = tip_opd + astig_opd

# #     phi = pupil * 2.0 * np.pi / wavelength * opd
# #     amp = pupil.copy()

# #     # ============================================================
# #     # 3. Analytic ZWFS output field
# #     # ============================================================

# #     zwfs_terms = bldr.get_zwfs_output_field(
# #         phi=phi,
# #         amp=amp,
# #         theta=zwfs_ns.optics.theta,
# #         phasemask_diameter=zwfs_ns.optics.mask_diam,
# #         phasemask_mask=None,
# #         pupil_diameter=zwfs_ns.grid.N,
# #         fplane_pixels=zwfs_ns.focal_plane.fplane_pixels,
# #         pixels_across_mask=zwfs_ns.focal_plane.pixels_across_mask,
# #         return_terms=True,
# #     )

# #     psi_A = zwfs_terms["psi_A"]
# #     b = zwfs_terms["b"]
# #     psi_C = zwfs_terms["psi_C"]

# #     # Also compute flat reference for differences
# #     zwfs_terms_flat = bldr.get_zwfs_output_field(
# #         phi=np.zeros_like(phi),
# #         amp=amp,
# #         theta=zwfs_ns.optics.theta,
# #         phasemask_diameter=zwfs_ns.optics.mask_diam,
# #         phasemask_mask=None,
# #         pupil_diameter=zwfs_ns.grid.N,
# #         fplane_pixels=zwfs_ns.focal_plane.fplane_pixels,
# #         pixels_across_mask=zwfs_ns.focal_plane.pixels_across_mask,
# #         return_terms=True,
# #     )

# #     psi_C_flat = zwfs_terms_flat["psi_C"]

# #     # ============================================================
# #     # 4. Physical propagation scale
# #     # ============================================================

# #     dx_phys = D_phys / zwfs_ns.grid.N

# #     X_phys, Y_phys = fresnel.make_coordinate_grid(psi_C.shape, dx_phys)
# #     R_phys = np.hypot(X_phys, Y_phys)

# #     # ============================================================
# #     # 5. Propagate to D/knife-edge mirror
# #     # ============================================================

# #     psi_mirror = fresnel.propagate(
# #         psi_C,
# #         wavelength=wavelength,
# #         dx=dx_phys,
# #         z=z_to_mirror,
# #         method="angular_spectrum",
# #     )

# #     psi_mirror_flat = fresnel.propagate(
# #         psi_C_flat,
# #         wavelength=wavelength,
# #         dx=dx_phys,
# #         z=z_to_mirror,
# #         method="angular_spectrum",
# #     )

# #     mirror_mask_full = fresnel.circular_aperture(
# #         psi_mirror.shape,
# #         dx=dx_phys,
# #         radius=D_mirror / 2,
# #     ).astype(float)

# #     edge_angle = 0.0
# #     Xr = X_phys * np.cos(edge_angle) + Y_phys * np.sin(edge_angle)

# #     mirror_mask_D = (Xr >= edge_offset).astype(float) * mirror_mask_full

# #     psi_after_mirror = psi_mirror * mirror_mask_D
# #     psi_after_mirror_flat = psi_mirror_flat * mirror_mask_D

# #     E_mirror = fresnel.energy(psi_mirror, dx_phys)
# #     E_after_mirror = fresnel.energy(psi_after_mirror, dx_phys)

# #     # ============================================================
# #     # 6. Re-imaging lens and cold stop
# #     # ============================================================

# #     s_object = z_to_mirror + z_mirror_to_lens

# #     if s_object <= f_imaging:
# #         st.error(
# #             "Invalid lens geometry: object distance must be greater than f_imaging."
# #         )
# #         st.stop()

# #     s_image = 1.0 / (1.0 / f_imaging - 1.0 / s_object)
# #     z_focus_to_detector = s_image - f_imaging
# #     M_pupil = s_image / s_object
# #     D_detector_predicted = M_pupil * D_phys

# #     field_coldstop_plane, dx_cold, dy_cold = fresnel.lens_focal_plane_field(
# #         psi_after_mirror,
# #         wavelength=wavelength,
# #         dx=dx_phys,
# #         focal_length=f_imaging,
# #         include_global_phase=False,
# #     )

# #     field_coldstop_plane_flat, _, _ = fresnel.lens_focal_plane_field(
# #         psi_after_mirror_flat,
# #         wavelength=wavelength,
# #         dx=dx_phys,
# #         focal_length=f_imaging,
# #         include_global_phase=False,
# #     )

# #     X_cold, Y_cold = fresnel.make_coordinate_grid(
# #         field_coldstop_plane.shape,
# #         dx=dx_cold,
# #         dy=dy_cold,
# #     )

# #     R_cold = np.hypot(X_cold, Y_cold)
# #     coldstop_mask_phys = (R_cold <= D_coldstop / 2).astype(float)

# #     field_after_coldstop = field_coldstop_plane * coldstop_mask_phys
# #     field_after_coldstop_flat = field_coldstop_plane_flat * coldstop_mask_phys

# #     E_before_cold = np.sum(np.abs(field_coldstop_plane) ** 2)
# #     E_after_cold = np.sum(np.abs(field_after_coldstop) ** 2)

# #     # ============================================================
# #     # 7. Propagate from star-image/cold-stop plane to detector pupil plane
# #     # ============================================================

# #     field_detector_pupil, dx_detector, dy_detector = fresnel.fresnel_one_step_propagate(
# #         field_after_coldstop,
# #         wavelength=wavelength,
# #         dx=dx_cold,
# #         dy=dy_cold,
# #         z=z_focus_to_detector,
# #         include_global_phase=False,
# #     )

# #     field_detector_pupil_flat, _, _ = fresnel.fresnel_one_step_propagate(
# #         field_after_coldstop_flat,
# #         wavelength=wavelength,
# #         dx=dx_cold,
# #         dy=dy_cold,
# #         z=z_focus_to_detector,
# #         include_global_phase=False,
# #     )

# #     I_zwfs = np.abs(psi_C) ** 2
# #     I_zwfs_flat = np.abs(psi_C_flat) ** 2

# #     I_det = np.abs(field_detector_pupil) ** 2
# #     I_det_flat = np.abs(field_detector_pupil_flat) ** 2

# #     I_zwfs_norm = I_zwfs / np.nanmax(I_zwfs_flat)
# #     I_zwfs_flat_norm = I_zwfs_flat / np.nanmax(I_zwfs_flat)
# #     dI_zwfs = I_zwfs_norm - I_zwfs_flat_norm

# #     I_det_norm = I_det / np.nanmax(I_det_flat)
# #     I_det_flat_norm = I_det_flat / np.nanmax(I_det_flat)
# #     dI_det = I_det_norm - I_det_flat_norm

# #     X_det, Y_det = fresnel.make_coordinate_grid(
# #         I_det.shape,
# #         dx=dx_detector,
# #         dy=dy_detector,
# #     )

# #     lambda_over_D_phys = wavelength * f_imaging / D_phys

# #     # ============================================================
# #     # Diagnostics
# #     # ============================================================

# #     st.subheader("Current geometry")

# #     c1, c2, c3, c4 = st.columns(4)

# #     c1.metric("D after 40 mm collimator", f"{1e3 * D_phys:.3f} mm")
# #     c2.metric("D-edge throughput", f"{E_after_mirror / E_mirror:.3f}")
# #     c3.metric("Cold-stop throughput", f"{E_after_cold / E_before_cold:.3f}")
# #     c4.metric("Predicted detector pupil", f"{1e6 * D_detector_predicted:.1f} µm")

# #     st.write(
# #         {
# #             "wavelength_um": 1e6 * wavelength,
# #             "dx_phys_um_per_pix": 1e6 * dx_phys,
# #             "z_to_mirror_mm": 1e3 * z_to_mirror,
# #             "z_mirror_to_lens_mm": 1e3 * z_mirror_to_lens,
# #             "s_object_mm": 1e3 * s_object,
# #             "s_image_mm": 1e3 * s_image,
# #             "z_focus_to_detector_mm": 1e3 * z_focus_to_detector,
# #             "dx_detector_um_per_pix": 1e6 * dx_detector,
# #             "coldstop_diam_lambda_over_D": D_coldstop / lambda_over_D_phys,
# #             "OPD_rms_nm": 1e9 * np.std(opd[pupil_bool]),
# #         }
# #     )

# #     # ============================================================
# #     # Plot 1: OPD, raw ZWFS response, detector response
# #     # ============================================================


# #     # ============================================================
# #     # Big top plot: detector intensity only
# #     # ============================================================

# #     st.subheader("Detector pupil intensity")

# #     if show_log_detector:
# #         big_intensity_plot = np.log10(I_det_norm + 1e-8)
# #         big_title = "Detector pupil intensity, log10 scale"
# #         big_vmin = -8
# #         big_vmax = 0
# #         big_cbar = r"$\log_{10}$ normalized intensity"
# #     else:
# #         big_intensity_plot = I_det_norm
# #         big_title = "Detector pupil intensity"
# #         big_vmin = 0
# #         big_vmax = 1
# #         big_cbar = "Intensity / flat peak"

# #     fig_big, ax_big = plt.subplots(1, 1, figsize=(9, 8))

# #     im_big = ax_big.imshow(
# #         big_intensity_plot,
# #         origin="lower",
# #         extent=[
# #             1e6 * X_det.min(),
# #             1e6 * X_det.max(),
# #             1e6 * Y_det.min(),
# #             1e6 * Y_det.max(),
# #         ],
# #         vmin=big_vmin,
# #         vmax=big_vmax,
# #     )

# #     ax_big.set_title(big_title, fontsize=18)
# #     ax_big.set_xlabel("x [µm]", fontsize=14)
# #     ax_big.set_ylabel("y [µm]", fontsize=14)

# #     cbar_big = plt.colorbar(im_big, ax=ax_big, fraction=0.046, pad=0.04)
# #     cbar_big.set_label(big_cbar, fontsize=13)

# #     plt.tight_layout()
# #     st.pyplot(fig_big)



# #     st.subheader("ZWFS and detector response")

# #     dmax = max(
# #         np.nanmax(np.abs(dI_zwfs)),
# #         np.nanmax(np.abs(dI_det)),
# #         1e-12,
# #     )

# #     if show_log_detector:
# #         det_plot = np.log10(I_det_norm + 1e-8)
# #         det_title = "Detector intensity, log10"
# #         det_vmin = -8
# #         det_vmax = 0
# #         det_cbar = "log10 normalized intensity"
# #     else:
# #         det_plot = I_det_norm
# #         det_title = "Detector intensity"
# #         det_vmin = 0
# #         det_vmax = 1
# #         det_cbar = "Intensity / flat peak"

# #     fig, axes = plt.subplots(1, 5, figsize=(21, 4.2))

# #     im0 = axes[0].imshow(1e9 * opd, origin="lower")
# #     axes[0].set_title("Input OPD [nm]")
# #     plt.colorbar(im0, ax=axes[0], fraction=0.046)

# #     im1 = axes[1].imshow(I_zwfs_norm, origin="lower", vmin=0)
# #     axes[1].set_title(r"Raw ZWFS $|\Psi_C|^2$")
# #     plt.colorbar(im1, ax=axes[1], fraction=0.046)

# #     im2 = axes[2].imshow(
# #         dI_zwfs,
# #         origin="lower",
# #         vmin=-dmax,
# #         vmax=dmax,
# #         cmap="RdBu_r",
# #     )
# #     axes[2].set_title("Raw ZWFS - flat")
# #     plt.colorbar(im2, ax=axes[2], fraction=0.046)

# #     im3 = axes[3].imshow(
# #         det_plot,
# #         origin="lower",
# #         extent=[
# #             1e6 * X_det.min(),
# #             1e6 * X_det.max(),
# #             1e6 * Y_det.min(),
# #             1e6 * Y_det.max(),
# #         ],
# #         vmin=det_vmin,
# #         vmax=det_vmax,
# #     )
# #     axes[3].set_title(det_title)
# #     axes[3].set_xlabel("x [µm]")
# #     axes[3].set_ylabel("y [µm]")
# #     plt.colorbar(im3, ax=axes[3], fraction=0.046, label=det_cbar)

# #     im4 = axes[4].imshow(
# #         dI_det,
# #         origin="lower",
# #         extent=[
# #             1e6 * X_det.min(),
# #             1e6 * X_det.max(),
# #             1e6 * Y_det.min(),
# #             1e6 * Y_det.max(),
# #         ],
# #         vmin=-dmax,
# #         vmax=dmax,
# #         cmap="RdBu_r",
# #     )
# #     axes[4].set_title("Detector - flat")
# #     axes[4].set_xlabel("x [µm]")
# #     axes[4].set_ylabel("y [µm]")
# #     plt.colorbar(im4, ax=axes[4], fraction=0.046)

# #     for ax in axes[:3]:
# #         ax.set_xticks([])
# #         ax.set_yticks([])

# #     plt.tight_layout()
# #     st.pyplot(fig)

# #     # ============================================================
# #     # Plot 2: D mirror and cold stop geometry
# #     # ============================================================

# #     st.subheader("Apertures and intermediate planes")

# #     I_mirror = np.abs(psi_mirror) ** 2
# #     I_cold = np.abs(field_coldstop_plane) ** 2
# #     I_cold_norm = I_cold / np.nanmax(I_cold)

# #     X_cold_lamD = X_cold / lambda_over_D_phys
# #     Y_cold_lamD = Y_cold / lambda_over_D_phys

# #     fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4.2))

# #     im0 = axes2[0].imshow(
# #         I_mirror / np.nanmax(I_mirror),
# #         origin="lower",
# #         extent=[
# #             1e3 * X_phys.min(),
# #             1e3 * X_phys.max(),
# #             1e3 * Y_phys.min(),
# #             1e3 * Y_phys.max(),
# #         ],
# #     )
# #     axes2[0].contour(
# #         1e3 * X_phys,
# #         1e3 * Y_phys,
# #         mirror_mask_D,
# #         levels=[0.5],
# #         colors="w",
# #     )
# #     axes2[0].set_title("Mirror plane")
# #     axes2[0].set_xlabel("x [mm]")
# #     axes2[0].set_ylabel("y [mm]")
# #     plt.colorbar(im0, ax=axes2[0], fraction=0.046)

# #     im1 = axes2[1].imshow(
# #         mirror_mask_D,
# #         origin="lower",
# #         extent=[
# #             1e3 * X_phys.min(),
# #             1e3 * X_phys.max(),
# #             1e3 * Y_phys.min(),
# #             1e3 * Y_phys.max(),
# #         ],
# #         vmin=0,
# #         vmax=1,
# #     )
# #     axes2[1].set_title("D/knife-edge mask")
# #     axes2[1].set_xlabel("x [mm]")
# #     axes2[1].set_ylabel("y [mm]")
# #     plt.colorbar(im1, ax=axes2[1], fraction=0.046)

# #     im2 = axes2[2].imshow(
# #         np.log10(I_cold_norm + 1e-8),
# #         origin="lower",
# #         extent=[
# #             X_cold_lamD.min(),
# #             X_cold_lamD.max(),
# #             Y_cold_lamD.min(),
# #             Y_cold_lamD.max(),
# #         ],
# #         vmin=-8,
# #         vmax=0,
# #     )
# #     axes2[2].contour(
# #         X_cold_lamD,
# #         Y_cold_lamD,
# #         coldstop_mask_phys,
# #         levels=[0.5],
# #         colors="w",
# #     )
# #     axes2[2].set_xlim(-10, 10)
# #     axes2[2].set_ylim(-10, 10)
# #     axes2[2].set_title("Cold stop plane")
# #     axes2[2].set_xlabel(r"$\lambda/D$")
# #     axes2[2].set_ylabel(r"$\lambda/D$")
# #     plt.colorbar(im2, ax=axes2[2], fraction=0.046)

# #     im3 = axes2[3].imshow(
# #         coldstop_mask_phys,
# #         origin="lower",
# #         extent=[
# #             1e3 * X_cold.min(),
# #             1e3 * X_cold.max(),
# #             1e3 * Y_cold.min(),
# #             1e3 * Y_cold.max(),
# #         ],
# #         vmin=0,
# #         vmax=1,
# #     )
# #     axes2[3].set_title("Physical cold stop mask")
# #     axes2[3].set_xlabel("x [mm]")
# #     axes2[3].set_ylabel("y [mm]")
# #     plt.colorbar(im3, ax=axes2[3], fraction=0.046)

# #     plt.tight_layout()
# #     st.pyplot(fig2)

# #     plt.close("all")

# # else:
# #     st.info("Adjust sliders and press **Update simulation**.")