import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

# Ensure local source tree is used
# repo_root = Path(__file__).resolve().parents[2]
# if str(repo_root) not in sys.path:
#     sys.path.insert(0, str(repo_root))

from baldrapp.common import baldr_core as bldr
from baldrapp.common import fresnel


"""
Standalone single-case Baldr ZWFS Fresnel relay simulation.

Optical model:

    analytic ZWFS field
    -> free-space propagation to D/knife-edge mirror
    -> D-edge vignetting
    -> propagation to 184 mm re-imaging lens
    -> thin-lens phase
    -> physical cold stop / star-image plane
    -> detector pupil plane

All variables are hardcoded intentionally so this script can be used as a
transparent base for later exploration.


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
# 0. Hardcoded physical parameters
# ============================================================

D_entrance = 12e-3          # m, Baldr input pupil diameter before OAP
f_oap = 254e-3              # m, Baldr OAP focal length
f_coll = 40e-3              # m, post-mask collimating lens
z_to_mirror = 0.50          # m, collimator output / ZWFS pupil plane to D mirror
D_mirror = 25.4e-3          # m, 1 inch mirror clear aperture
f_imaging = 184e-3          # m, Baldr re-imaging lens
D_coldstop = 2.145e-3       # m, physical cold stop diameter
D_detector_pupil = 288e-6   # m, nominal detector pupil diameter

# ZWFS and aberration settings
theta_phase_shift = np.pi / 2     # rad; use 0.0 for clear-pupil sanity test
tip_amp_nm = 100.0                # nm OPD
astig_amp_nm = 0.0                # nm OPD

# Knife-edge / D mirror
edge_offset = -1.0e-3             # m; more positive clips more of the beam
edge_angle = 0.0                  # rad; 0 means vertical edge in x

# Cold stop alignment
coldstop_x_offset = 0.0           # m, lateral offset in cold-stop plane

# Detector axial offset from nominal pupil conjugate plane
pupil_misconjugation = 0.0        # m

# Display options
show_log_detector = False
save_figure = False
output_path = Path("baldr_zwfs_fresnel_single_case.png")


# ============================================================
# 1. Derived nominal geometry
# ============================================================

D_phys = D_entrance * f_coll / f_oap

M_nominal = D_detector_pupil / D_phys
s_object_nominal = f_imaging * (1.0 + 1.0 / M_nominal)
s_image_nominal = f_imaging * (1.0 + M_nominal)
z_mirror_to_lens = s_object_nominal - z_to_mirror
z_focus_to_detector_nominal = s_image_nominal - f_imaging
z_focus_to_detector = z_focus_to_detector_nominal + pupil_misconjugation

if z_focus_to_detector <= 0:
    raise ValueError("Invalid detector position: cold-stop to detector distance must be positive.")

print("\n=== Nominal physical geometry ===")
print("D_phys after 40 mm collimator [mm]:", 1e3 * D_phys)
print("M_nominal:", M_nominal)
print("s_object_nominal [mm]:", 1e3 * s_object_nominal)
print("s_image_nominal [mm]:", 1e3 * s_image_nominal)
print("z_to_mirror [mm]:", 1e3 * z_to_mirror)
print("z_mirror_to_lens [mm]:", 1e3 * z_mirror_to_lens)
print("z_focus_to_detector_nominal [mm]:", 1e3 * z_focus_to_detector_nominal)
print("pupil_misconjugation [mm]:", 1e3 * pupil_misconjugation)
print("z_focus_to_detector_actual [mm]:", 1e3 * z_focus_to_detector)


# ============================================================
# 2. Initialise BaldrApp ZWFS
# ============================================================

grid_dict = {
    "telescope": "solarstein",
    "D": 1.8,              # telescope-equivalent internal BaldrApp coordinate
    "N": 72,               # pixels across pupil diameter
    "dim": 72 * 4,         # full simulation grid
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

dx_phys = D_phys / zwfs_ns.grid.N

print("\n=== Simulation grid ===")
print("wavelength [um]:", 1e6 * wavelength)
print("pupil array shape:", pupil.shape)
print("dx_phys [um/pix]:", 1e6 * dx_phys)
print("full physical grid width [mm]:", 1e3 * dx_phys * pupil.shape[0])


# ============================================================
# 3. Input OPD: tip + astigmatism
# ============================================================

Xw = zwfs_ns.grid.wave_coord.X
Yw = zwfs_ns.grid.wave_coord.Y

x_norm = np.zeros_like(Xw, dtype=float)
y_norm = np.zeros_like(Yw, dtype=float)

x_norm[pupil_bool] = Xw[pupil_bool] / np.max(np.abs(Xw[pupil_bool]))
y_norm[pupil_bool] = Yw[pupil_bool] / np.max(np.abs(Yw[pupil_bool]))

tip_opd = pupil * (tip_amp_nm * 1e-9) * x_norm
astig_opd = pupil * (astig_amp_nm * 1e-9) * (x_norm**2 - y_norm**2)
opd = tip_opd + astig_opd

phi = pupil * 2.0 * np.pi / wavelength * opd
amp = pupil.copy()

print("\n=== Input aberration ===")
print("theta [rad]:", theta_phase_shift)
print("theta/pi:", theta_phase_shift / np.pi)
print("tip_amp_nm:", tip_amp_nm)
print("astig_amp_nm:", astig_amp_nm)
print("OPD rms in pupil [nm]:", 1e9 * np.std(opd[pupil_bool]))


# ============================================================
# 4. Analytic ZWFS output field and flat reference
# ============================================================

zwfs_terms = bldr.get_zwfs_output_field(
    phi=phi,
    amp=amp,
    theta=theta_phase_shift,
    phasemask_diameter=zwfs_ns.optics.mask_diam,
    phasemask_mask=None,
    pupil_diameter=zwfs_ns.grid.N,
    fplane_pixels=zwfs_ns.focal_plane.fplane_pixels,
    pixels_across_mask=zwfs_ns.focal_plane.pixels_across_mask,
    return_terms=True,
)

zwfs_terms_flat = bldr.get_zwfs_output_field(
    phi=np.zeros_like(phi),
    amp=amp,
    theta=theta_phase_shift,
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
psi_C_flat = zwfs_terms_flat["psi_C"]

theta_zero_check = np.linalg.norm((psi_C - psi_A).ravel()) / np.linalg.norm(psi_A.ravel())

print("\n=== ZWFS analytic field ===")
print("pix_per_wvld:", zwfs_terms["pix_per_wvld"])
print("sum |psi_A|^2:", np.sum(np.abs(psi_A) ** 2))
print("sum |psi_C|^2:", np.sum(np.abs(psi_C) ** 2))
print("relative ||psi_C - psi_A||:", theta_zero_check)


# ============================================================
# 5. Propagate to D/knife-edge mirror and apply mask
# ============================================================

X_phys, Y_phys = fresnel.make_coordinate_grid(psi_C.shape, dx_phys)
R_phys = np.hypot(X_phys, Y_phys)

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

Xr = X_phys * np.cos(edge_angle) + Y_phys * np.sin(edge_angle)
mirror_mask_D = (Xr >= edge_offset).astype(float) * mirror_mask_full

psi_after_mirror = psi_mirror * mirror_mask_D
psi_after_mirror_flat = psi_mirror_flat * mirror_mask_D

E_mirror = fresnel.energy(psi_mirror, dx_phys)
E_after_mirror = fresnel.energy(psi_after_mirror, dx_phys)

print("\n=== D/knife-edge mirror ===")
print("edge_offset [mm]:", 1e3 * edge_offset)
print("mirror throughput:", E_after_mirror / E_mirror)


# ============================================================
# 6. Thin-lens conjugation and explicit propagation to lens
# ============================================================

s_object = z_to_mirror + z_mirror_to_lens

if s_object <= f_imaging:
    raise RuntimeError("Invalid geometry: s_object must be greater than f_imaging.")

s_image = 1.0 / (1.0 / f_imaging - 1.0 / s_object)
z_focus_to_detector_nominal_current = s_image - f_imaging
z_focus_to_detector = z_focus_to_detector_nominal_current + pupil_misconjugation

if z_focus_to_detector <= 0:
    raise RuntimeError("Invalid detector position: cold-stop to detector distance must be positive.")

M_pupil = s_image / s_object
D_detector_predicted = M_pupil * D_phys

print("\n=== Thin-lens conjugation ===")
print("s_object [mm]:", 1e3 * s_object)
print("s_image [mm]:", 1e3 * s_image)
print("z_focus_to_detector_nominal [mm]:", 1e3 * z_focus_to_detector_nominal_current)
print("pupil_misconjugation [mm]:", 1e3 * pupil_misconjugation)
print("z_focus_to_detector_actual [mm]:", 1e3 * z_focus_to_detector)
print("M_pupil:", M_pupil)
print("predicted detector pupil [um]:", 1e6 * D_detector_predicted)

# Propagate from mirror to re-imaging lens
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

# Apply 184 mm thin lens
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


# ============================================================
# 7. Propagate to cold-stop / star-image plane and apply stop
# ============================================================

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

lambda_over_D_phys = wavelength * f_imaging / D_phys

print("\n=== Cold-stop plane ===")
print("dx_cold [um/pix]:", 1e6 * dx_cold)
print("full cold-stop grid [mm]:", 1e3 * dx_cold * field_coldstop_plane.shape[0])
print("lambda f / D [um]:", 1e6 * lambda_over_D_phys)
print("coldstop diameter [mm]:", 1e3 * D_coldstop)
print("coldstop x offset [mm]:", 1e3 * coldstop_x_offset)
print("coldstop diameter [lambda/D]:", D_coldstop / lambda_over_D_phys)
print("coldstop throughput:", E_after_cold / E_before_cold)


# ============================================================
# 8. Propagate from cold stop to detector pupil plane
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

print("\n=== Detector plane ===")
print("dx_detector [um/pix]:", 1e6 * dx_detector)
print("full detector grid [um]:", 1e6 * dx_detector * I_det.shape[0])
print("predicted detector pupil [um]:", 1e6 * D_detector_predicted)
print("predicted detector pupil [pixels]:", D_detector_predicted / dx_detector)


# ============================================================
# 9. Single consolidated subplot figure
# ============================================================

I_mirror = np.abs(psi_mirror) ** 2
I_lens = np.abs(field_at_lens) ** 2
I_cold = np.abs(field_coldstop_plane) ** 2

I_mirror_norm = I_mirror / np.nanmax(I_mirror)
I_lens_norm = I_lens / np.nanmax(I_lens)
I_cold_norm = I_cold / np.nanmax(I_cold)

X_cold_lamD = X_cold / lambda_over_D_phys
Y_cold_lamD = Y_cold / lambda_over_D_phys

if show_log_detector:
    I_detector_display = np.log10(I_det_norm + 1e-8)
    detector_vmin = -8
    detector_vmax = 0
    detector_label = "log10 detector intensity"
else:
    I_detector_display = I_det_norm
    detector_vmin = 0
    detector_vmax = 1
    detector_label = "detector intensity / flat peak"

dmax = max(
    np.nanmax(np.abs(dI_zwfs)),
    np.nanmax(np.abs(dI_det)),
    1e-12,
)

fig, axes = plt.subplots(3, 4, figsize=(22, 15))

# Row 1: input and ZWFS quantities
im = axes[0, 0].imshow(1e9 * opd, origin="lower")
axes[0, 0].set_title("Input OPD [nm]")
plt.colorbar(im, ax=axes[0, 0], fraction=0.046)

im = axes[0, 1].imshow(np.abs(psi_A) ** 2, origin="lower")
axes[0, 1].set_title("Input pupil intensity")
plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

im = axes[0, 2].imshow(np.abs(b) ** 2, origin="lower")
axes[0, 2].set_title("ZWFS reference intensity |b|^2")
plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

im = axes[0, 3].imshow(I_zwfs_norm, origin="lower", vmin=0)
axes[0, 3].set_title("Raw ZWFS output |Psi_C|^2")
plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

# Row 2: intermediate propagation and apertures
im = axes[1, 0].imshow(
    I_mirror_norm,
    origin="lower",
    extent=[
        1e3 * X_phys.min(),
        1e3 * X_phys.max(),
        1e3 * Y_phys.min(),
        1e3 * Y_phys.max(),
    ],
)
axes[1, 0].contour(
    1e3 * X_phys,
    1e3 * Y_phys,
    mirror_mask_D,
    levels=[0.5],
    colors="w",
)
axes[1, 0].set_title("Mirror plane intensity")
axes[1, 0].set_xlabel("x [mm]")
axes[1, 0].set_ylabel("y [mm]")
plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

im = axes[1, 1].imshow(
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
axes[1, 1].set_title("D / knife-edge mask")
axes[1, 1].set_xlabel("x [mm]")
axes[1, 1].set_ylabel("y [mm]")
plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

im = axes[1, 2].imshow(
    I_lens_norm,
    origin="lower",
    extent=[
        1e3 * X_phys.min(),
        1e3 * X_phys.max(),
        1e3 * Y_phys.min(),
        1e3 * Y_phys.max(),
    ],
)
axes[1, 2].set_title("Field at re-imaging lens")
axes[1, 2].set_xlabel("x [mm]")
axes[1, 2].set_ylabel("y [mm]")
plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

im = axes[1, 3].imshow(
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
axes[1, 3].contour(
    X_cold_lamD,
    Y_cold_lamD,
    coldstop_mask_phys,
    levels=[0.5],
    colors="w",
)
axes[1, 3].set_xlim(-10, 10)
axes[1, 3].set_ylim(-10, 10)
axes[1, 3].set_title("Cold stop / star-image plane")
axes[1, 3].set_xlabel("lambda / D")
axes[1, 3].set_ylabel("lambda / D")
plt.colorbar(im, ax=axes[1, 3], fraction=0.046)

# Row 3: detector and differential responses
im = axes[2, 0].imshow(
    I_detector_display,
    origin="lower",
    extent=[
        1e6 * X_det.min(),
        1e6 * X_det.max(),
        1e6 * Y_det.min(),
        1e6 * Y_det.max(),
    ],
    vmin=detector_vmin,
    vmax=detector_vmax,
)
axes[2, 0].set_title("Detector pupil intensity")
axes[2, 0].set_xlabel("x [um]")
axes[2, 0].set_ylabel("y [um]")
plt.colorbar(im, ax=axes[2, 0], fraction=0.046, label=detector_label)

im = axes[2, 1].imshow(
    dI_zwfs,
    origin="lower",
    vmin=-dmax,
    vmax=dmax,
    cmap="RdBu_r",
)
axes[2, 1].set_title("Raw ZWFS - flat")
plt.colorbar(im, ax=axes[2, 1], fraction=0.046)

im = axes[2, 2].imshow(
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
axes[2, 2].set_title("Detector - flat")
axes[2, 2].set_xlabel("x [um]")
axes[2, 2].set_ylabel("y [um]")
plt.colorbar(im, ax=axes[2, 2], fraction=0.046)

im = axes[2, 3].imshow(
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
axes[2, 3].set_title(f"Cold stop mask, x offset = {1e3 * coldstop_x_offset:+.2f} mm")
axes[2, 3].set_xlabel("x [mm]")
axes[2, 3].set_ylabel("y [mm]")
plt.colorbar(im, ax=axes[2, 3], fraction=0.046)

for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[0, 3], axes[2, 1]]:
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(
    (
        f"Baldr ZWFS Fresnel relay single case | "
        f"theta = {theta_phase_shift / np.pi:.2f} pi, "
        f"tip = {tip_amp_nm:.1f} nm, astig = {astig_amp_nm:.1f} nm, "
        f"edge = {1e3 * edge_offset:+.2f} mm"
    ),
    fontsize=16,
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

if save_figure:
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print("Saved figure:", output_path)

plt.show()


# ============================================================
# 10. Summary diagnostics
# ============================================================

print("\n=== Summary diagnostics ===")
print("Mirror throughput:", E_after_mirror / E_mirror)
print("Cold stop throughput:", E_after_cold / E_before_cold)
print("D_phys [mm]:", 1e3 * D_phys)
print("dx_phys [um/pix]:", 1e6 * dx_phys)
print("dx_cold [um/pix]:", 1e6 * dx_cold)
print("dx_detector [um/pix]:", 1e6 * dx_detector)
print("coldstop diameter [lambda/D]:", D_coldstop / lambda_over_D_phys)
print("predicted detector pupil [um]:", 1e6 * D_detector_predicted)
print("predicted detector pupil [pix]:", D_detector_predicted / dx_detector)
print("OPD rms [nm]:", 1e9 * np.std(opd[pupil_bool]))
print("Done.")