from pathlib import Path
from types import SimpleNamespace
import copy

import numpy as np
import matplotlib.pyplot as plt

from baldrapp.common import baldr_core as bldr


# ============================================================
# Config
# ============================================================

repo_root = Path("/Users/bencb/Documents/ASGARD/BaldrApp")
config_path = repo_root / "baldrapp/apps/paranal_simulator/fake_configs/baldr_config.json"

zwfs_ns = bldr.init_zwfs_from_json(config_path)

pupil = zwfs_ns.grid.pupil_mask
detector = zwfs_ns.detector

np.random.seed(3)

# Input source model.
# Convention: amp_input**2 is photons / s / wave-space-pixel / nm.
amp_input = np.sqrt(1.0e5) * pupil

# Add a small physically interpretable aberration so morphology is non-trivial.
X = zwfs_ns.grid.wave_coord.X
Y = zwfs_ns.grid.wave_coord.Y
pupil_bool = pupil.astype(bool)

x_norm = np.zeros_like(X, dtype=float)
y_norm = np.zeros_like(Y, dtype=float)

x_norm[pupil_bool] = X[pupil_bool] / np.max(np.abs(X[pupil_bool]))
y_norm[pupil_bool] = Y[pupil_bool] / np.max(np.abs(Y[pupil_bool]))

opd_tip = 50e-9 * pupil * x_norm
opd_astig = 30e-9 * pupil * (x_norm**2 - y_norm**2)
opd_input = opd_tip + opd_astig

# Internal static OPD.
opd_internal = 20e-9 * pupil * np.random.randn(*pupil.shape)

# Non-flat DM command so all paths include DM consistently.
dm_perturb = 0.01 * np.random.randn(len(zwfs_ns.dm.dm_flat))
test_dm_cmd = zwfs_ns.dm.dm_flat + dm_perturb
zwfs_ns.dm.current_cmd = test_dm_cmd.copy()


# ============================================================
# Helpers
# ============================================================

def make_mono_copy(zwfs):
    """
    Shallow copy with a monochromatic 1 nm spectrum.

    This is enough for this verification script because only the spectrum
    namespace is replaced. The DM object is shared, so we explicitly restore
    the test DM command after making the copy.
    """
    z = copy.copy(zwfs)
    z.spectrum = SimpleNamespace(
        enabled=False,
        mode="monochromatic",
        wavelengths=np.array([zwfs.optics.wvl0], dtype=float),
        weights=np.array([1.0], dtype=float),
        weights_normalized=np.array([1.0], dtype=float),
        weights_nm=np.array([1.0], dtype=float),
        bandwidth_nm=1.0,
    )
    z.dm.current_cmd = test_dm_cmd.copy()
    return z


def normalize_for_display(im):
    im = np.asarray(im, dtype=float)
    p99 = np.nanpercentile(im, 99)
    scale = p99 if p99 > 0 else np.nanmax(im)
    if scale <= 0:
        scale = 1.0
    return im / scale


def relative_difference(a, b, eps=1e-12):
    return (a - b) / (np.nanmax(np.abs(b)) + eps)


def summarize_image(name, im):
    print(f"\n{name}")
    print("  shape:", im.shape)
    print("  sum:", np.sum(im))
    print("  min/max:", np.min(im), np.max(im))
    print("  mean/std:", np.mean(im), np.std(im))


def assert_finite_positive_image(name, im):
    assert np.all(np.isfinite(im)), f"{name} contains non-finite values."
    assert np.sum(im) > 0, f"{name} has non-positive total flux."


# ============================================================
# Print configuration summary
# ============================================================

print("\n=== Configuration summary ===")
print("config_path:", config_path)
print("grid N:", zwfs_ns.grid.N)
print("grid dim:", zwfs_ns.grid.dim)
print("detector binning:", detector.binning)
print("stellar.bandwidth [nm]:", getattr(zwfs_ns.stellar, "bandwidth", None))

if hasattr(zwfs_ns, "spectrum"):
    print("spectrum wavelengths [um]:", zwfs_ns.spectrum.wavelengths * 1e6)
    print("spectrum weights_nm:", zwfs_ns.spectrum.weights_nm)
    print("sum weights_nm [nm]:", np.sum(zwfs_ns.spectrum.weights_nm))

if hasattr(zwfs_ns, "fresnel_relay"):
    print("fresnel enabled:", getattr(zwfs_ns.fresnel_relay, "enabled", None))
    print("D_phys [mm]:", 1e3 * zwfs_ns.fresnel_relay.D_phys)
    print("z_focus_to_detector [mm]:", 1e3 * zwfs_ns.fresnel_relay.z_focus_to_detector)

print("DM command RMS from flat:", np.std(zwfs_ns.dm.current_cmd - zwfs_ns.dm.dm_flat))


# ============================================================
# Generate science frames
# ============================================================

frames = {}

# 1. Non-Fresnel monochromatic, 1 nm.
zwfs_mono = make_mono_copy(zwfs_ns)
frames["NF mono"] = bldr.get_frame_configured(
    opd_input=opd_input,
    amp_input=amp_input,
    opd_internal=opd_internal,
    zwfs_ns=zwfs_mono,
    detector=detector,
    include_shotnoise=False,
    use_pyZelda=False,
    spectral_bandwidth=1.0,
    force_fresnel=False,
    force_polychromatic=False,
)

# 2. Non-Fresnel polychromatic.
zwfs_ns.dm.current_cmd = test_dm_cmd.copy()
frames["NF poly"] = bldr.get_frame_configured(
    opd_input=opd_input,
    amp_input=amp_input,
    opd_internal=opd_internal,
    zwfs_ns=zwfs_ns,
    detector=detector,
    include_shotnoise=False,
    use_pyZelda=False,
    force_fresnel=False,
    force_polychromatic=True,
)

# 3. Fresnel monochromatic, 1 nm.
zwfs_mono = make_mono_copy(zwfs_ns)
frames["F mono"] = bldr.get_frame_configured(
    opd_input=opd_input,
    amp_input=amp_input,
    opd_internal=opd_internal,
    zwfs_ns=zwfs_mono,
    detector=detector,
    include_shotnoise=False,
    use_pyZelda=False,
    spectral_bandwidth=1.0,
    force_fresnel=True,
    force_polychromatic=False,
)

# 4. Fresnel polychromatic.
zwfs_ns.dm.current_cmd = test_dm_cmd.copy()
frames["F poly"] = bldr.get_frame_configured(
    opd_input=opd_input,
    amp_input=amp_input,
    opd_internal=opd_internal,
    zwfs_ns=zwfs_ns,
    detector=detector,
    include_shotnoise=False,
    use_pyZelda=False,
    force_fresnel=True,
    force_polychromatic=True,
)

print("\n=== Science frame summaries ===")
for name, im in frames.items():
    summarize_image(name, im)
    assert_finite_positive_image(name, im)


# ============================================================
# Generate configured I0 and N0 references
# ============================================================

references = {}

# 1. Non-Fresnel monochromatic, 1 nm.
zwfs_mono = make_mono_copy(zwfs_ns)
references["NF mono"] = {
    "I0": bldr.get_I0_configured(
        opd_input=opd_input,
        amp_input=amp_input,
        opd_internal=opd_internal,
        zwfs_ns=zwfs_mono,
        detector=detector,
        include_shotnoise=False,
        use_pyZelda=False,
        spectral_bandwidth=1.0,
        force_fresnel=False,
        force_polychromatic=False,
    ),
    "N0": bldr.get_N0_configured(
        opd_input=opd_input,
        amp_input=amp_input,
        opd_internal=opd_internal,
        zwfs_ns=zwfs_mono,
        detector=detector,
        include_shotnoise=False,
        use_pyZelda=False,
        spectral_bandwidth=1.0,
        force_fresnel=False,
        force_polychromatic=False,
    ),
}

# 2. Non-Fresnel polychromatic.
zwfs_ns.dm.current_cmd = test_dm_cmd.copy()
references["NF poly"] = {
    "I0": bldr.get_I0_configured(
        opd_input=opd_input,
        amp_input=amp_input,
        opd_internal=opd_internal,
        zwfs_ns=zwfs_ns,
        detector=detector,
        include_shotnoise=False,
        use_pyZelda=False,
        force_fresnel=False,
        force_polychromatic=True,
    ),
    "N0": bldr.get_N0_configured(
        opd_input=opd_input,
        amp_input=amp_input,
        opd_internal=opd_internal,
        zwfs_ns=zwfs_ns,
        detector=detector,
        include_shotnoise=False,
        use_pyZelda=False,
        force_fresnel=False,
        force_polychromatic=True,
    ),
}

# 3. Fresnel monochromatic, 1 nm.
zwfs_mono = make_mono_copy(zwfs_ns)
references["F mono"] = {
    "I0": bldr.get_I0_configured(
        opd_input=opd_input,
        amp_input=amp_input,
        opd_internal=opd_internal,
        zwfs_ns=zwfs_mono,
        detector=detector,
        include_shotnoise=False,
        use_pyZelda=False,
        spectral_bandwidth=1.0,
        force_fresnel=True,
        force_polychromatic=False,
    ),
    "N0": bldr.get_N0_configured(
        opd_input=opd_input,
        amp_input=amp_input,
        opd_internal=opd_internal,
        zwfs_ns=zwfs_mono,
        detector=detector,
        include_shotnoise=False,
        use_pyZelda=False,
        spectral_bandwidth=1.0,
        force_fresnel=True,
        force_polychromatic=False,
    ),
}

# 4. Fresnel polychromatic.
zwfs_ns.dm.current_cmd = test_dm_cmd.copy()
references["F poly"] = {
    "I0": bldr.get_I0_configured(
        opd_input=opd_input,
        amp_input=amp_input,
        opd_internal=opd_internal,
        zwfs_ns=zwfs_ns,
        detector=detector,
        include_shotnoise=False,
        use_pyZelda=False,
        force_fresnel=True,
        force_polychromatic=True,
    ),
    "N0": bldr.get_N0_configured(
        opd_input=opd_input,
        amp_input=amp_input,
        opd_internal=opd_internal,
        zwfs_ns=zwfs_ns,
        detector=detector,
        include_shotnoise=False,
        use_pyZelda=False,
        force_fresnel=True,
        force_polychromatic=True,
    ),
}

# Restore science-frame DM state after reference generation.
zwfs_ns.dm.current_cmd = test_dm_cmd.copy()

print("\n=== I0 / N0 reference summaries ===")
for name, ref in references.items():
    I0 = ref["I0"]
    N0 = ref["N0"]
    dI = I0 - N0

    print(f"\n{name}")
    print("  I0 shape/sum/min/max:", I0.shape, np.sum(I0), np.min(I0), np.max(I0))
    print("  N0 shape/sum/min/max:", N0.shape, np.sum(N0), np.min(N0), np.max(N0))
    print("  I0/N0 sum ratio:", np.sum(I0) / (np.sum(N0) + 1e-12))
    print("  (I0-N0) rms:", np.std(dI))
    print("  frame/I0 sum ratio:", np.sum(frames[name]) / (np.sum(I0) + 1e-12))

    assert I0.shape == frames[name].shape
    assert N0.shape == frames[name].shape
    assert_finite_positive_image(f"{name} I0", I0)
    assert_finite_positive_image(f"{name} N0", N0)

print("\nPASS: configured I0/N0 references generated for all four propagation modes.")
print("current DM restored RMS from flat:", np.std(zwfs_ns.dm.current_cmd - zwfs_ns.dm.dm_flat))


# ============================================================
# 1. Absolute science frames
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(11, 10))
axes = axes.ravel()

for ax, (name, im) in zip(axes, frames.items()):
    p1, p99 = np.nanpercentile(im, [1, 99])
    h = ax.imshow(im, origin="lower", vmin=p1, vmax=p99)
    ax.set_title(f"{name}\nscience frame, absolute, 1-99 pct")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(h, ax=ax, fraction=0.046)

fig.suptitle("Detected science frames: absolute scaling", fontsize=14)
plt.tight_layout()
plt.show()


# ============================================================
# 2. Normalized science morphology comparison
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(11, 10))
axes = axes.ravel()

for ax, (name, im) in zip(axes, frames.items()):
    imn = normalize_for_display(im)
    h = ax.imshow(imn, origin="lower", vmin=0, vmax=1)
    ax.set_title(f"{name}\nscience frame normalized by 99th percentile")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(h, ax=ax, fraction=0.046)

fig.suptitle("Detected science frames: normalized morphology", fontsize=14)
plt.tight_layout()
plt.show()


# ============================================================
# 3. Polychromatic minus monochromatic science morphology
# ============================================================

diff_nf = relative_difference(
    normalize_for_display(frames["NF poly"]),
    normalize_for_display(frames["NF mono"]),
)

diff_f = relative_difference(
    normalize_for_display(frames["F poly"]),
    normalize_for_display(frames["F mono"]),
)

dlim = max(
    np.nanpercentile(np.abs(diff_nf), 99),
    np.nanpercentile(np.abs(diff_f), 99),
    1e-12,
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

h = axes[0].imshow(diff_nf, origin="lower", cmap="RdBu_r", vmin=-dlim, vmax=dlim)
axes[0].set_title("Non-Fresnel science: normalized poly - mono")
axes[0].set_xticks([])
axes[0].set_yticks([])
plt.colorbar(h, ax=axes[0], fraction=0.046)

h = axes[1].imshow(diff_f, origin="lower", cmap="RdBu_r", vmin=-dlim, vmax=dlim)
axes[1].set_title("Fresnel science: normalized poly - mono")
axes[1].set_xticks([])
axes[1].set_yticks([])
plt.colorbar(h, ax=axes[1], fraction=0.046)

fig.suptitle("Chromatic morphology change in science frames", fontsize=14)
plt.tight_layout()
plt.show()


# ============================================================
# 4. Fresnel minus non-Fresnel science morphology
# ============================================================

diff_mono = relative_difference(
    normalize_for_display(frames["F mono"]),
    normalize_for_display(frames["NF mono"]),
)

diff_poly = relative_difference(
    normalize_for_display(frames["F poly"]),
    normalize_for_display(frames["NF poly"]),
)

dlim = max(
    np.nanpercentile(np.abs(diff_mono), 99),
    np.nanpercentile(np.abs(diff_poly), 99),
    1e-12,
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

h = axes[0].imshow(diff_mono, origin="lower", cmap="RdBu_r", vmin=-dlim, vmax=dlim)
axes[0].set_title("Monochromatic science: normalized Fresnel - non-Fresnel")
axes[0].set_xticks([])
axes[0].set_yticks([])
plt.colorbar(h, ax=axes[0], fraction=0.046)

h = axes[1].imshow(diff_poly, origin="lower", cmap="RdBu_r", vmin=-dlim, vmax=dlim)
axes[1].set_title("Polychromatic science: normalized Fresnel - non-Fresnel")
axes[1].set_xticks([])
axes[1].set_yticks([])
plt.colorbar(h, ax=axes[1], fraction=0.046)

fig.suptitle("Fresnel relay morphology change in science frames", fontsize=14)
plt.tight_layout()
plt.show()


# ============================================================
# 5. I0 and N0 absolute reference frames
# ============================================================

fig, axes = plt.subplots(4, 3, figsize=(14, 16))

for row, (name, ref) in enumerate(references.items()):
    I0 = ref["I0"]
    N0 = ref["N0"]
    dI = I0 - N0

    images = [I0, N0, dI]
    titles = [
        f"{name}: I0 phasemask in",
        f"{name}: N0 clear pupil",
        f"{name}: I0 - N0",
    ]

    for col, (im, title) in enumerate(zip(images, titles)):
        ax = axes[row, col]

        if col < 2:
            p1, p99 = np.nanpercentile(im, [1, 99])
            h = ax.imshow(im, origin="lower", vmin=p1, vmax=p99)
        else:
            dlim = np.nanpercentile(np.abs(im), 99)
            dlim = max(dlim, 1e-12)
            h = ax.imshow(im, origin="lower", cmap="RdBu_r", vmin=-dlim, vmax=dlim)

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(h, ax=ax, fraction=0.046)

fig.suptitle("Configured reference frames: I0, N0, and I0 - N0", fontsize=14)
plt.tight_layout()
plt.show()


# ============================================================
# 6. Normalized references and science signal
# ============================================================

fig, axes = plt.subplots(4, 3, figsize=(14, 16))

for row, (name, ref) in enumerate(references.items()):
    I0 = ref["I0"]
    N0 = ref["N0"]
    I = frames[name]

    N0_mean = np.nanmean(N0) + 1e-12
    I0n = I0 / N0_mean
    N0n = N0 / N0_mean
    signal = I / N0_mean - I0n

    images = [I0n, N0n, signal]
    titles = [
        f"{name}: I0 / <N0>",
        f"{name}: N0 / <N0>",
        f"{name}: I/<N0> - I0/<N0>",
    ]

    for col, (im, title) in enumerate(zip(images, titles)):
        ax = axes[row, col]

        if col < 2:
            p1, p99 = np.nanpercentile(im, [1, 99])
            h = ax.imshow(im, origin="lower", vmin=p1, vmax=p99)
        else:
            dlim = np.nanpercentile(np.abs(im), 99)
            dlim = max(dlim, 1e-12)
            h = ax.imshow(im, origin="lower", cmap="RdBu_r", vmin=-dlim, vmax=dlim)

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(h, ax=ax, fraction=0.046)

fig.suptitle("Normalized references and ZWFS science signal", fontsize=14)
plt.tight_layout()
plt.show()


# ============================================================
# 7. Fresnel intermediates for visual diagnostics
# ============================================================

zwfs_ns.dm.current_cmd = test_dm_cmd.copy()

out_fresnel = bldr.get_frame_fresnel(
    opd_input=opd_input,
    amp_input=amp_input,
    opd_internal=opd_internal,
    zwfs_ns=zwfs_ns,
    detector=None,
    include_shotnoise=False,
    spectral_bandwidth=None,
    wavelength=zwfs_ns.optics.wvl0,
    return_intermediates=True,
)

intermediate_images = {
    "ZWFS output |psi_C|^2": np.abs(out_fresnel["psi_C"]) ** 2,
    "Mirror plane |field|^2": np.abs(out_fresnel["psi_mirror"]) ** 2,
    "At lens |field|^2": np.abs(out_fresnel["field_at_lens"]) ** 2,
    "Cold-stop plane |field|^2": np.abs(out_fresnel["field_coldstop_plane"]) ** 2,
    "After cold stop |field|^2": np.abs(out_fresnel["field_after_coldstop"]) ** 2,
    "Detector pupil |field|^2": np.abs(out_fresnel["field_detector_pupil"]) ** 2,
}

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.ravel()

for ax, (name, im) in zip(axes, intermediate_images.items()):
    im = np.asarray(im, dtype=float)
    imn = im / (np.nanmax(im) + 1e-12)
    h = ax.imshow(np.log10(imn + 1e-8), origin="lower", vmin=-8, vmax=0)
    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(h, ax=ax, fraction=0.046, label="log10 normalized intensity")

fig.suptitle("Fresnel relay intermediate planes", fontsize=14)
plt.tight_layout()
plt.show()


# ============================================================
# 8. Horizontal and vertical cuts through detected science frames
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for name, im in frames.items():
    imn = normalize_for_display(im)
    cy, cx = np.array(imn.shape) // 2
    axes[0].plot(imn[cy, :], label=name)
    axes[1].plot(imn[:, cx], label=name)

axes[0].set_title("Science horizontal center cut")
axes[0].set_xlabel("x pixel")
axes[0].set_ylabel("normalized intensity")
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].set_title("Science vertical center cut")
axes[1].set_xlabel("y pixel")
axes[1].set_ylabel("normalized intensity")
axes[1].grid(alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()


# ============================================================
# 9. Horizontal and vertical cuts through I0 and N0
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for name, ref in references.items():
    I0n = normalize_for_display(ref["I0"])
    N0n = normalize_for_display(ref["N0"])
    cy, cx = np.array(I0n.shape) // 2

    axes[0, 0].plot(I0n[cy, :], label=name)
    axes[0, 1].plot(I0n[:, cx], label=name)
    axes[1, 0].plot(N0n[cy, :], label=name)
    axes[1, 1].plot(N0n[:, cx], label=name)

axes[0, 0].set_title("I0 horizontal center cut")
axes[0, 1].set_title("I0 vertical center cut")
axes[1, 0].set_title("N0 horizontal center cut")
axes[1, 1].set_title("N0 vertical center cut")

for ax in axes.ravel():
    ax.set_xlabel("pixel")
    ax.set_ylabel("normalized intensity")
    ax.grid(alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()


print("\nPlotting verification complete.")
# from pathlib import Path
# from types import SimpleNamespace
# import copy

# import numpy as np
# import matplotlib.pyplot as plt

# from baldrapp.common import baldr_core as bldr


# # ============================================================
# # Config
# # ============================================================

# repo_root = Path("/Users/bencb/Documents/ASGARD/BaldrApp")
# config_path = repo_root / "baldrapp/apps/paranal_simulator/fake_configs/baldr_config.json"

# zwfs_ns = bldr.init_zwfs_from_json(config_path)

# pupil = zwfs_ns.grid.pupil_mask
# detector = zwfs_ns.detector

# np.random.seed(3)

# # Input source model.
# amp_input = np.sqrt(1.0e5) * pupil

# # Add a small physically interpretable aberration so morphology is non-trivial.
# X = zwfs_ns.grid.wave_coord.X
# Y = zwfs_ns.grid.wave_coord.Y
# pupil_bool = pupil.astype(bool)

# x_norm = np.zeros_like(X, dtype=float)
# y_norm = np.zeros_like(Y, dtype=float)

# x_norm[pupil_bool] = X[pupil_bool] / np.max(np.abs(X[pupil_bool]))
# y_norm[pupil_bool] = Y[pupil_bool] / np.max(np.abs(Y[pupil_bool]))

# opd_tip = 50e-9 * pupil * x_norm
# opd_astig = 30e-9 * pupil * (x_norm**2 - y_norm**2)
# opd_input = opd_tip + opd_astig

# # Internal static OPD.
# opd_internal = 20e-9 * pupil * np.random.randn(*pupil.shape)

# # Non-flat DM command so all paths include DM consistently.
# dm_perturb = 0.01 * np.random.randn(len(zwfs_ns.dm.dm_flat))
# test_dm_cmd = zwfs_ns.dm.dm_flat + dm_perturb
# zwfs_ns.dm.current_cmd = test_dm_cmd.copy()


# # ============================================================
# # Helpers
# # ============================================================

# def make_mono_copy(zwfs):
#     """
#     Shallow copy with a monochromatic 1 nm spectrum.
#     """
#     z = copy.copy(zwfs)
#     z.spectrum = SimpleNamespace(
#         enabled=False,
#         mode="monochromatic",
#         wavelengths=np.array([zwfs.optics.wvl0], dtype=float),
#         weights=np.array([1.0], dtype=float),
#         weights_normalized=np.array([1.0], dtype=float),
#         weights_nm=np.array([1.0], dtype=float),
#         bandwidth_nm=1.0,
#     )
#     z.dm.current_cmd = test_dm_cmd.copy()
#     return z


# def normalize_for_display(im):
#     im = np.asarray(im, dtype=float)
#     med = np.nanmedian(im)
#     p99 = np.nanpercentile(im, 99)
#     scale = p99 if p99 > 0 else np.nanmax(im)
#     if scale <= 0:
#         scale = 1.0
#     return im / scale


# def relative_difference(a, b, eps=1e-12):
#     return (a - b) / (np.nanmax(np.abs(b)) + eps)


# def summarize_image(name, im):
#     print(f"\n{name}")
#     print("  shape:", im.shape)
#     print("  sum:", np.sum(im))
#     print("  min/max:", np.min(im), np.max(im))
#     print("  mean/std:", np.mean(im), np.std(im))


# # ============================================================
# # Generate frames
# # ============================================================

# zwfs_mono = make_mono_copy(zwfs_ns)

# frames = {}

# # 1. Non-Fresnel monochromatic, 1 nm.
# frames["NF mono"] = bldr.get_frame_configured(
#     opd_input=opd_input,
#     amp_input=amp_input,
#     opd_internal=opd_internal,
#     zwfs_ns=zwfs_mono,
#     detector=detector,
#     include_shotnoise=False,
#     use_pyZelda=False,
#     spectral_bandwidth=1.0,
#     force_fresnel=False,
#     force_polychromatic=False,
# )

# # 2. Non-Fresnel polychromatic.
# zwfs_ns.dm.current_cmd = test_dm_cmd.copy()
# frames["NF poly"] = bldr.get_frame_configured(
#     opd_input=opd_input,
#     amp_input=amp_input,
#     opd_internal=opd_internal,
#     zwfs_ns=zwfs_ns,
#     detector=detector,
#     include_shotnoise=False,
#     use_pyZelda=False,
#     force_fresnel=False,
#     force_polychromatic=True,
# )

# # 3. Fresnel monochromatic, 1 nm.
# zwfs_mono = make_mono_copy(zwfs_ns)
# frames["F mono"] = bldr.get_frame_configured(
#     opd_input=opd_input,
#     amp_input=amp_input,
#     opd_internal=opd_internal,
#     zwfs_ns=zwfs_mono,
#     detector=detector,
#     include_shotnoise=False,
#     use_pyZelda=False,
#     spectral_bandwidth=1.0,
#     force_fresnel=True,
#     force_polychromatic=False,
# )

# # 4. Fresnel polychromatic.
# zwfs_ns.dm.current_cmd = test_dm_cmd.copy()
# frames["F poly"] = bldr.get_frame_configured(
#     opd_input=opd_input,
#     amp_input=amp_input,
#     opd_internal=opd_internal,
#     zwfs_ns=zwfs_ns,
#     detector=detector,
#     include_shotnoise=False,
#     use_pyZelda=False,
#     force_fresnel=True,
#     force_polychromatic=True,
# )

# for name, im in frames.items():
#     summarize_image(name, im)


# # ============================================================
# # 1. Absolute frames
# # ============================================================

# fig, axes = plt.subplots(2, 2, figsize=(11, 10))
# axes = axes.ravel()

# for ax, (name, im) in zip(axes, frames.items()):
#     p1, p99 = np.nanpercentile(im, [1, 99])
#     h = ax.imshow(im, origin="lower", vmin=p1, vmax=p99)
#     ax.set_title(f"{name}\nabsolute, 1-99 pct")
#     ax.set_xticks([])
#     ax.set_yticks([])
#     plt.colorbar(h, ax=ax, fraction=0.046)

# fig.suptitle("Detected frames: absolute scaling", fontsize=14)
# plt.tight_layout()
# plt.show()


# # ============================================================
# # 2. Normalized morphology comparison
# # ============================================================

# fig, axes = plt.subplots(2, 2, figsize=(11, 10))
# axes = axes.ravel()

# for ax, (name, im) in zip(axes, frames.items()):
#     imn = normalize_for_display(im)
#     h = ax.imshow(imn, origin="lower", vmin=0, vmax=1)
#     ax.set_title(f"{name}\nnormalized by 99th percentile")
#     ax.set_xticks([])
#     ax.set_yticks([])
#     plt.colorbar(h, ax=ax, fraction=0.046)

# fig.suptitle("Detected frames: normalized morphology", fontsize=14)
# plt.tight_layout()
# plt.show()


# # ============================================================
# # 3. Polychromatic minus monochromatic morphology
# # ============================================================

# diff_nf = relative_difference(
#     normalize_for_display(frames["NF poly"]),
#     normalize_for_display(frames["NF mono"]),
# )

# diff_f = relative_difference(
#     normalize_for_display(frames["F poly"]),
#     normalize_for_display(frames["F mono"]),
# )

# dlim = max(
#     np.nanpercentile(np.abs(diff_nf), 99),
#     np.nanpercentile(np.abs(diff_f), 99),
#     1e-12,
# )

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# h = axes[0].imshow(diff_nf, origin="lower", cmap="RdBu_r", vmin=-dlim, vmax=dlim)
# axes[0].set_title("Non-Fresnel: normalized poly - mono")
# axes[0].set_xticks([])
# axes[0].set_yticks([])
# plt.colorbar(h, ax=axes[0], fraction=0.046)

# h = axes[1].imshow(diff_f, origin="lower", cmap="RdBu_r", vmin=-dlim, vmax=dlim)
# axes[1].set_title("Fresnel: normalized poly - mono")
# axes[1].set_xticks([])
# axes[1].set_yticks([])
# plt.colorbar(h, ax=axes[1], fraction=0.046)

# fig.suptitle("Chromatic morphology change", fontsize=14)
# plt.tight_layout()
# plt.show()


# # ============================================================
# # 4. Fresnel minus non-Fresnel morphology
# # ============================================================

# diff_mono = relative_difference(
#     normalize_for_display(frames["F mono"]),
#     normalize_for_display(frames["NF mono"]),
# )

# diff_poly = relative_difference(
#     normalize_for_display(frames["F poly"]),
#     normalize_for_display(frames["NF poly"]),
# )

# dlim = max(
#     np.nanpercentile(np.abs(diff_mono), 99),
#     np.nanpercentile(np.abs(diff_poly), 99),
#     1e-12,
# )

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# h = axes[0].imshow(diff_mono, origin="lower", cmap="RdBu_r", vmin=-dlim, vmax=dlim)
# axes[0].set_title("Monochromatic: normalized Fresnel - non-Fresnel")
# axes[0].set_xticks([])
# axes[0].set_yticks([])
# plt.colorbar(h, ax=axes[0], fraction=0.046)

# h = axes[1].imshow(diff_poly, origin="lower", cmap="RdBu_r", vmin=-dlim, vmax=dlim)
# axes[1].set_title("Polychromatic: normalized Fresnel - non-Fresnel")
# axes[1].set_xticks([])
# axes[1].set_yticks([])
# plt.colorbar(h, ax=axes[1], fraction=0.046)

# fig.suptitle("Fresnel relay morphology change", fontsize=14)
# plt.tight_layout()
# plt.show()


# # ============================================================
# # 5. Fresnel intermediates for visual diagnostics
# # ============================================================

# zwfs_ns.dm.current_cmd = test_dm_cmd.copy()

# out_fresnel = bldr.get_frame_fresnel(
#     opd_input=opd_input,
#     amp_input=amp_input,
#     opd_internal=opd_internal,
#     zwfs_ns=zwfs_ns,
#     detector=None,
#     include_shotnoise=False,
#     spectral_bandwidth=None,
#     wavelength=zwfs_ns.optics.wvl0,
#     return_intermediates=True,
# )

# intermediate_images = {
#     "ZWFS output |psi_C|^2": np.abs(out_fresnel["psi_C"]) ** 2,
#     "Mirror plane |field|^2": np.abs(out_fresnel["psi_mirror"]) ** 2,
#     "At lens |field|^2": np.abs(out_fresnel["field_at_lens"]) ** 2,
#     "Cold-stop plane |field|^2": np.abs(out_fresnel["field_coldstop_plane"]) ** 2,
#     "After cold stop |field|^2": np.abs(out_fresnel["field_after_coldstop"]) ** 2,
#     "Detector pupil |field|^2": np.abs(out_fresnel["field_detector_pupil"]) ** 2,
# }

# fig, axes = plt.subplots(2, 3, figsize=(15, 9))
# axes = axes.ravel()

# for ax, (name, im) in zip(axes, intermediate_images.items()):
#     im = np.asarray(im, dtype=float)
#     imn = im / (np.nanmax(im) + 1e-12)
#     h = ax.imshow(np.log10(imn + 1e-8), origin="lower", vmin=-8, vmax=0)
#     ax.set_title(name)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     plt.colorbar(h, ax=ax, fraction=0.046, label="log10 normalized intensity")

# fig.suptitle("Fresnel relay intermediate planes", fontsize=14)
# plt.tight_layout()
# plt.show()


# # ============================================================
# # 6. Horizontal and vertical cuts through detected frames
# # ============================================================

# fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# for name, im in frames.items():
#     imn = normalize_for_display(im)
#     cy, cx = np.array(imn.shape) // 2
#     axes[0].plot(imn[cy, :], label=name)
#     axes[1].plot(imn[:, cx], label=name)

# axes[0].set_title("Horizontal center cut")
# axes[0].set_xlabel("x pixel")
# axes[0].set_ylabel("normalized intensity")
# axes[0].grid(alpha=0.3)
# axes[0].legend()

# axes[1].set_title("Vertical center cut")
# axes[1].set_xlabel("y pixel")
# axes[1].set_ylabel("normalized intensity")
# axes[1].grid(alpha=0.3)
# axes[1].legend()

# plt.tight_layout()
# plt.show()


# print("\nPlotting verification complete.")


# # from pathlib import Path
# # from types import SimpleNamespace
# # import copy

# # import numpy as np

# # from baldrapp.common import baldr_core as bldr


# # # ============================================================
# # # Config
# # # ============================================================

# # repo_root = Path("/Users/bencb/Documents/ASGARD/BaldrApp")
# # config_path = repo_root / "baldrapp/apps/paranal_simulator/fake_configs/baldr_config.json"

# # zwfs_ns = bldr.init_zwfs_from_json(config_path)

# # pupil = zwfs_ns.grid.pupil_mask
# # amp_input = np.sqrt(1.0e5) * pupil
# # opd_input = np.zeros_like(pupil)
# # opd_internal = np.zeros_like(pupil)

# # detector = zwfs_ns.detector

# # rtol = 1e-12


# # # ============================================================
# # # Helper
# # # ============================================================

# # def relerr(a, b):
# #     denom = np.linalg.norm(b)
# #     if denom == 0:
# #         return np.linalg.norm(a - b)
# #     return np.linalg.norm(a - b) / denom


# # def make_mono_copy(zwfs):
# #     """
# #     Shallow copy with a monochromatic 1 nm spectrum.
# #     This is enough because we only replace the spectrum namespace.
# #     """
# #     z = copy.copy(zwfs)
# #     z.spectrum = SimpleNamespace(
# #         enabled=False,
# #         mode="monochromatic",
# #         wavelengths=np.array([zwfs.optics.wvl0], dtype=float),
# #         weights=np.array([1.0], dtype=float),
# #         weights_normalized=np.array([1.0], dtype=float),
# #         weights_nm=np.array([1.0], dtype=float),
# #         bandwidth_nm=1.0,
# #     )
# #     return z


# # def print_check(name, configured, direct):
# #     err = relerr(configured, direct)

# #     print(f"\n=== {name} ===")
# #     print("configured shape:", configured.shape)
# #     print("direct shape:    ", direct.shape)
# #     print("configured sum:  ", np.sum(configured))
# #     print("direct sum:      ", np.sum(direct))
# #     print("relative error:  ", err)

# #     assert configured.shape == direct.shape
# #     assert err < rtol

# #     print(f"PASS: {name}")


# # # ============================================================
# # # 1. Non-Fresnel + monochromatic
# # # ============================================================

# # zwfs_mono = make_mono_copy(zwfs_ns)

# # I_config = bldr.get_frame_configured(
# #     opd_input=opd_input,
# #     amp_input=amp_input,
# #     opd_internal=opd_internal,
# #     zwfs_ns=zwfs_mono,
# #     detector=detector,
# #     include_shotnoise=False,
# #     use_pyZelda=False,
# #     spectral_bandwidth=1.0,
# #     force_fresnel=False,
# #     force_polychromatic=False,
# # )

# # I_direct = bldr.get_frame(
# #     opd_input=opd_input,
# #     amp_input=amp_input,
# #     opd_internal=opd_internal,
# #     zwfs_ns=zwfs_mono,
# #     detector=detector,
# #     include_shotnoise=False,
# #     spectral_bandwidth=1.0,
# #     use_pyZelda=False,
# # )

# # print_check("non-Fresnel + monochromatic", I_config, I_direct)


# # # ============================================================
# # # 2. Non-Fresnel + polychromatic
# # # ============================================================

# # I_config = bldr.get_frame_configured(
# #     opd_input=opd_input,
# #     amp_input=amp_input,
# #     opd_internal=opd_internal,
# #     zwfs_ns=zwfs_ns,
# #     detector=detector,
# #     include_shotnoise=False,
# #     use_pyZelda=False,
# #     force_fresnel=False,
# #     force_polychromatic=True,
# # )

# # I_direct = bldr.get_frame_polychromatic(
# #     opd_input=opd_input,
# #     amp_input=amp_input,
# #     opd_internal=opd_internal,
# #     zwfs_ns=zwfs_ns,
# #     detector=detector,
# #     include_shotnoise=False,
# #     use_pyZelda=False,
# # )

# # print_check("non-Fresnel + polychromatic", I_config, I_direct)


# # # ============================================================
# # # 3. Fresnel + monochromatic
# # # ============================================================

# # zwfs_mono = make_mono_copy(zwfs_ns)

# # I_config = bldr.get_frame_configured(
# #     opd_input=opd_input,
# #     amp_input=amp_input,
# #     opd_internal=opd_internal,
# #     zwfs_ns=zwfs_mono,
# #     detector=detector,
# #     include_shotnoise=False,
# #     use_pyZelda=False,
# #     spectral_bandwidth=1.0,
# #     force_fresnel=True,
# #     force_polychromatic=False,
# # )

# # I_direct = bldr.get_frame_fresnel(
# #     opd_input=opd_input,
# #     amp_input=amp_input,
# #     opd_internal=opd_internal,
# #     zwfs_ns=zwfs_mono,
# #     detector=detector,
# #     include_shotnoise=False,
# #     spectral_bandwidth=1.0,
# #     wavelength=zwfs_mono.optics.wvl0,
# # )

# # print_check("Fresnel + monochromatic", I_config, I_direct)


# # # ============================================================
# # # 4. Fresnel + polychromatic
# # # ============================================================

# # I_config = bldr.get_frame_configured(
# #     opd_input=opd_input,
# #     amp_input=amp_input,
# #     opd_internal=opd_internal,
# #     zwfs_ns=zwfs_ns,
# #     detector=detector,
# #     include_shotnoise=False,
# #     use_pyZelda=False,
# #     force_fresnel=True,
# #     force_polychromatic=True,
# # )

# # I_direct = bldr.get_frame_fresnel_polychromatic(
# #     opd_input=opd_input,
# #     amp_input=amp_input,
# #     opd_internal=opd_internal,
# #     zwfs_ns=zwfs_ns,
# #     detector=detector,
# #     include_shotnoise=False,
# # )

# # print_check("Fresnel + polychromatic", I_config, I_direct)


# # # ============================================================
# # # 5. Automatic routing check from config
# # # ============================================================

# # I_config_auto = bldr.get_frame_configured(
# #     opd_input=opd_input,
# #     amp_input=amp_input,
# #     opd_internal=opd_internal,
# #     zwfs_ns=zwfs_ns,
# #     detector=detector,
# #     include_shotnoise=False,
# #     use_pyZelda=False,
# # )

# # I_direct_auto = bldr.get_frame_fresnel_polychromatic(
# #     opd_input=opd_input,
# #     amp_input=amp_input,
# #     opd_internal=opd_internal,
# #     zwfs_ns=zwfs_ns,
# #     detector=detector,
# #     include_shotnoise=False,
# # )

# # print_check("automatic config routing", I_config_auto, I_direct_auto)


# # print("\nAll get_frame_configured routing tests passed.")


# # # import sys
# # # from pathlib import Path
# # # from types import SimpleNamespace

# # # import numpy as np
# # # import matplotlib.pyplot as plt

# # # repo_root = Path("/Users/bencb/Documents/ASGARD/BaldrApp")
# # # if str(repo_root) not in sys.path:
# # #     sys.path.insert(0, str(repo_root))

# # # from baldrapp.common import baldr_core as bldr
# # # from baldrapp.common import config_helper as cfghelp
# # # from baldrapp.common import spectrum as spec


# # # # ============================================================
# # # # 1. Initialise from JSON config
# # # # ============================================================

# # # config_path = (
# # #     repo_root
# # #     / "baldrapp/apps/paranal_simulator/fake_configs/baldr_config.json"
# # # )

# # # zwfs_json = bldr.init_zwfs_from_json(config_path)

# # # print("\n=== JSON init ===")
# # # print("grid.N:", zwfs_json.grid.N)
# # # print("grid.dim:", zwfs_json.grid.dim)
# # # print("detector.binning:", zwfs_json.detector.binning)
# # # print("detected pupil pixels:", zwfs_json.grid.N / zwfs_json.detector.binning)
# # # print("wavelengths [um]:", 1e6 * zwfs_json.spectrum.wavelengths)
# # # print("weights_nm:", zwfs_json.spectrum.weights_nm)

# # # assert zwfs_json.grid.N / zwfs_json.detector.binning == 16.0


# # # # ============================================================
# # # # 2. Manual equivalent initialisation
# # # # ============================================================

# # # grid_dict = {
# # #     "telescope": "solarstein",
# # #     "D": 1.8,
# # #     "N": 64,
# # #     "dim": 256,
# # # }

# # # optics_dict = {
# # #     "wvl0": 1.65e-6,
# # #     "F_number": 21.2,
# # #     "mask_diam": 1.06,
# # #     "mask_diam_mode": "lambda_over_D",
# # #     "theta": np.pi / 2,
# # #     "theta_mode": "constant",
# # #     "coldstop_diam": 8.4,
# # #     "coldstop_offset": (0, 0),
# # # }

# # # dm_dict = {
# # #     "dm_model": "BMC-multi-3.5",
# # #     "actuator_coupling_factor": 0.75,
# # #     "dm_pitch": 1.0,
# # #     "dm_aoi": 0.0,
# # #     "opd_per_cmd": 3.0e-6,
# # #     "flat_rmse": 0.0,
# # # }

# # # fresnel_relay_dict = {
# # #     "enabled": True,

# # #     "D_entrance": 0.012,
# # #     "f_oap": 0.254,
# # #     "f_coll": 0.040,

# # #     "z_to_mirror": 0.50,
# # #     "D_mirror": 0.0254,
# # #     "edge_offset": -0.001,
# # #     "edge_angle": 0.0,

# # #     "f_imaging": 0.184,
# # #     "D_coldstop": 0.002145,
# # #     "coldstop_x_offset": 0.0,
# # #     "coldstop_y_offset": 0.0,

# # #     "D_detector_pupil": 0.000288,
# # #     "pupil_misconjugation": 0.0,

# # #     "use_nominal_pupil_conjugation": True,
# # #     "z_mirror_to_lens": None,
# # # }

# # # spectrum_dict = {
# # #     "enabled": True,
# # #     "mode": "blackbody",
# # #     "temperature_K": 3500.0,
# # #     "weighting": "photon",
# # #     "wvl_min": 1.5e-6,
# # #     "wvl_max": 1.8e-6,
# # #     "n_wvl": 7,
# # #     "normalize": "sum",
# # # }

# # # detector_dict = {
# # #     "enabled": True,
# # #     "binning": 4,
# # #     "dit": 0.001,
# # #     "ron": 0.0,
# # #     "qe": 1.0,
# # # }


# # # grid_ns = SimpleNamespace(**grid_dict)
# # # optics_ns = SimpleNamespace(**optics_dict)
# # # dm_ns = SimpleNamespace(**dm_dict)

# # # optics_ns = cfghelp.ensure_optics_defaults(optics_ns)

# # # zwfs_manual = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)

# # # # Attach Fresnel relay manually
# # # fresnel_ns = cfghelp.dict_to_namespace_recursive(fresnel_relay_dict)
# # # fresnel_ns = cfghelp.derive_fresnel_relay_ns(fresnel_ns)
# # # zwfs_manual.fresnel_relay = fresnel_ns

# # # # Attach spectrum manually
# # # spectrum_ns = spec.derive_spectrum(
# # #     spectrum_dict,
# # #     default_wvl0=zwfs_manual.optics.wvl0,
# # # )
# # # spectrum_ns = cfghelp.complete_spectrum_integration_fields(spectrum_ns)
# # # zwfs_manual.spectrum = spectrum_ns

# # # # Attach detector manually
# # # zwfs_manual.detector = bldr.detector(
# # #     binning=detector_dict["binning"],
# # #     dit=detector_dict["dit"],
# # #     ron=detector_dict["ron"],
# # #     qe=detector_dict["qe"],
# # # )

# # # print("\n=== Manual init ===")
# # # print("grid.N:", zwfs_manual.grid.N)
# # # print("grid.dim:", zwfs_manual.grid.dim)
# # # print("detector.binning:", zwfs_manual.detector.binning)
# # # print("detected pupil pixels:", zwfs_manual.grid.N / zwfs_manual.detector.binning)
# # # print("wavelengths [um]:", 1e6 * zwfs_manual.spectrum.wavelengths)
# # # print("weights_nm:", zwfs_manual.spectrum.weights_nm)

# # # assert zwfs_manual.grid.N / zwfs_manual.detector.binning == 16.0


# # # # ============================================================
# # # # 3. Compare JSON and manual derived quantities
# # # # ============================================================

# # # print("\n=== Compare JSON vs manual ===")

# # # checks = {
# # #     "grid.N": (zwfs_json.grid.N, zwfs_manual.grid.N),
# # #     "grid.dim": (zwfs_json.grid.dim, zwfs_manual.grid.dim),
# # #     "optics.wvl0": (zwfs_json.optics.wvl0, zwfs_manual.optics.wvl0),
# # #     "optics.theta": (zwfs_json.optics.theta, zwfs_manual.optics.theta),
# # #     "fr.D_phys": (zwfs_json.fresnel_relay.D_phys, zwfs_manual.fresnel_relay.D_phys),
# # #     "fr.z_mirror_to_lens": (
# # #         zwfs_json.fresnel_relay.z_mirror_to_lens,
# # #         zwfs_manual.fresnel_relay.z_mirror_to_lens,
# # #     ),
# # #     "fr.z_focus_to_detector": (
# # #         zwfs_json.fresnel_relay.z_focus_to_detector,
# # #         zwfs_manual.fresnel_relay.z_focus_to_detector,
# # #     ),
# # #     "detector.binning": (
# # #         zwfs_json.detector.binning,
# # #         zwfs_manual.detector.binning,
# # #     ),
# # # }

# # # for name, (a, b) in checks.items():
# # #     print(f"{name}: json={a}, manual={b}")
# # #     assert np.allclose(a, b)

# # # assert np.allclose(
# # #     zwfs_json.spectrum.wavelengths,
# # #     zwfs_manual.spectrum.wavelengths,
# # # )

# # # assert np.allclose(
# # #     zwfs_json.spectrum.weights_nm,
# # #     zwfs_manual.spectrum.weights_nm,
# # # )

# # # print("\nPASS: JSON and manual initialisation agree for core derived quantities.")


# # # # ============================================================
# # # # 4. Generate polychromatic Fresnel frames from both
# # # # ============================================================

# # # photons_per_s_pix_nm = 1.0e5

# # # pupil_json = zwfs_json.grid.pupil_mask
# # # pupil_manual = zwfs_manual.grid.pupil_mask

# # # amp_json = np.sqrt(photons_per_s_pix_nm) * pupil_json
# # # amp_manual = np.sqrt(photons_per_s_pix_nm) * pupil_manual

# # # opd_json = np.zeros_like(pupil_json)
# # # opd_manual = np.zeros_like(pupil_manual)

# # # opd_internal_json = np.zeros_like(pupil_json)
# # # opd_internal_manual = np.zeros_like(pupil_manual)

# # # I_json = bldr.get_frame_fresnel_polychromatic(
# # #     opd_input=opd_json,
# # #     amp_input=amp_json,
# # #     opd_internal=opd_internal_json,
# # #     zwfs_ns=zwfs_json,
# # #     detector=zwfs_json.detector,
# # #     include_shotnoise=False,
# # #     return_intermediates=False,
# # # )

# # # I_manual = bldr.get_frame_fresnel_polychromatic(
# # #     opd_input=opd_manual,
# # #     amp_input=amp_manual,
# # #     opd_internal=opd_internal_manual,
# # #     zwfs_ns=zwfs_manual,
# # #     detector=zwfs_manual.detector,
# # #     include_shotnoise=False,
# # #     return_intermediates=False,
# # # )

# # # rel_err = np.linalg.norm(I_json - I_manual) / np.linalg.norm(I_json)

# # # print("\n=== Frame comparison ===")
# # # print("I_json shape:", I_json.shape)
# # # print("I_manual shape:", I_manual.shape)
# # # print("I_json sum:", np.sum(I_json))
# # # print("I_manual sum:", np.sum(I_manual))
# # # print("relative error:", rel_err)

# # # assert I_json.shape == I_manual.shape
# # # assert rel_err < 1e-12

# # # print("\nPASS: JSON and manual initialisation produce equivalent Fresnel-polychromatic detector frames.")


# # # # ============================================================
# # # # 5. Plot detector frame
# # # # ============================================================

# # # plt.figure(figsize=(6, 5))
# # # im = plt.imshow(I_json, origin="lower")
# # # plt.title("Detected polychromatic Fresnel frame, 16 pixels across pupil")
# # # plt.xlabel("detector x [pixels]")
# # # plt.ylabel("detector y [pixels]")
# # # plt.colorbar(im, label="detected counts")
# # # plt.tight_layout()
# # # plt.show()

# # # # import sys
# # # # from pathlib import Path
# # # # from types import SimpleNamespace
# # # # import copy

# # # # import numpy as np
# # # # import matplotlib.pyplot as plt

# # # # # ============================================================
# # # # # Local source tree
# # # # # ============================================================

# # # # repo_root = Path("/Users/bencb/Documents/ASGARD/BaldrApp")

# # # # if str(repo_root) not in sys.path:
# # # #     sys.path.insert(0, str(repo_root))

# # # # from baldrapp.common import baldr_core as bldr
# # # # from baldrapp.common import fresnel
# # # # from baldrapp.common import spectrum as spec


# # # # # ============================================================
# # # # # User settings
# # # # # ============================================================

# # # # config_path = (
# # # #     repo_root
# # # #     / "baldrapp/apps/paranal_simulator/fake_configs/baldr_config.json"
# # # # )

# # # # photons_per_s_pix_nm = 1.0e5

# # # # # Test aberration
# # # # tip_amp_nm = 50.0
# # # # astig_amp_nm = 80.0

# # # # rtol_strict = 1e-12
# # # # rtol_loose = 1e-9

# # # # make_plots = True


# # # # # ============================================================
# # # # # Init
# # # # # ============================================================

# # # # zwfs_ns = bldr.init_zwfs_from_json(config_path)

# # # # pupil = zwfs_ns.grid.pupil_mask
# # # # pupil_bool = pupil.astype(bool)

# # # # amp_input = np.sqrt(photons_per_s_pix_nm) * pupil
# # # # opd_internal = np.zeros_like(pupil)

# # # # Xw = zwfs_ns.grid.wave_coord.X
# # # # Yw = zwfs_ns.grid.wave_coord.Y

# # # # x_norm = np.zeros_like(Xw, dtype=float)
# # # # y_norm = np.zeros_like(Yw, dtype=float)

# # # # x_norm[pupil_bool] = Xw[pupil_bool] / np.max(np.abs(Xw[pupil_bool]))
# # # # y_norm[pupil_bool] = Yw[pupil_bool] / np.max(np.abs(Yw[pupil_bool]))

# # # # opd_tip = pupil * tip_amp_nm * 1e-9 * x_norm
# # # # opd_astig = pupil * astig_amp_nm * 1e-9 * (x_norm**2 - y_norm**2)
# # # # opd_input = opd_tip + opd_astig

# # # # sp = zwfs_ns.spectrum
# # # # fr = zwfs_ns.fresnel_relay
# # # # det = zwfs_ns.detector

# # # # wavelengths = np.asarray(sp.wavelengths, dtype=float)
# # # # weights_nm = np.asarray(sp.weights_nm, dtype=float)

# # # # print("\n=== Test setup ===")
# # # # print("config_path:", config_path)
# # # # print("wavelengths [um]:", 1e6 * wavelengths)
# # # # print("weights_nm:", weights_nm)
# # # # print("sum(weights_nm) [nm]:", np.sum(weights_nm))
# # # # print("bandwidth_nm:", sp.bandwidth_nm)
# # # # print("OPD rms [nm]:", 1e9 * np.std(opd_input[pupil_bool]))
# # # # print("D_phys [mm]:", 1e3 * fr.D_phys)
# # # # print("z_focus_to_detector [mm]:", 1e3 * fr.z_focus_to_detector)
# # # # print("detector binning:", det.binning)


# # # # # ============================================================
# # # # # 1. Spectrum consistency
# # # # # ============================================================

# # # # assert wavelengths.ndim == 1
# # # # assert weights_nm.ndim == 1
# # # # assert wavelengths.shape == weights_nm.shape
# # # # assert np.all(wavelengths > 0)
# # # # assert np.all(weights_nm >= 0)
# # # # assert np.sum(weights_nm) > 0

# # # # assert np.isclose(
# # # #     np.sum(weights_nm),
# # # #     sp.bandwidth_nm,
# # # #     rtol=rtol_strict,
# # # #     atol=1e-9,
# # # # )

# # # # print("\nPASS 1: spectrum wavelengths and weights_nm are valid.")
# # # # print("PASS 1: sum(weights_nm) equals spectrum.bandwidth_nm.")


# # # # # ============================================================
# # # # # 2. Wrapper pre-detector output
# # # # # ============================================================

# # # # wrapper_out = bldr.get_frame_fresnel_polychromatic(
# # # #     opd_input=opd_input,
# # # #     amp_input=amp_input,
# # # #     opd_internal=opd_internal,
# # # #     zwfs_ns=zwfs_ns,
# # # #     detector=None,
# # # #     include_shotnoise=False,
# # # #     return_intermediates=True,
# # # # )

# # # # I_poly_wrapper = wrapper_out["intensity_pre_detector"]

# # # # print("\n=== Wrapper output ===")
# # # # print("I_poly_wrapper shape:", I_poly_wrapper.shape)
# # # # print("I_poly_wrapper sum:", np.sum(I_poly_wrapper))
# # # # print("I_poly_wrapper min/max:", np.min(I_poly_wrapper), np.max(I_poly_wrapper))
# # # # print("wrapper weights_nm sum:", wrapper_out["weights_nm_sum"])

# # # # assert np.all(np.isfinite(I_poly_wrapper))
# # # # assert np.all(I_poly_wrapper >= 0)
# # # # assert np.isclose(wrapper_out["weights_nm_sum"], np.sum(weights_nm))

# # # # print("PASS 2: wrapper pre-detector output is finite and non-negative.")


# # # # # ============================================================
# # # # # 3. Manual integration equivalence
# # # # # ============================================================

# # # # I_poly_manual = None

# # # # for wavelength, weight_nm in zip(wavelengths, weights_nm):

# # # #     theta = spec.theta_at_wavelength(zwfs_ns.optics, wavelength)

# # # #     phasemask_diameter = spec.phasemask_diameter_at_wavelength(
# # # #         zwfs_ns.optics,
# # # #         wavelength_m=wavelength,
# # # #         default_wvl0=zwfs_ns.optics.wvl0,
# # # #     )

# # # #     I_density = bldr.get_frame_fresnel(
# # # #         opd_input=opd_input,
# # # #         amp_input=amp_input,
# # # #         opd_internal=opd_internal,
# # # #         zwfs_ns=zwfs_ns,
# # # #         detector=None,
# # # #         include_shotnoise=False,
# # # #         spectral_bandwidth=None,
# # # #         wavelength=wavelength,
# # # #         theta=theta,
# # # #         phasemask_diameter=phasemask_diameter,
# # # #         return_intermediates=False,
# # # #     )

# # # #     if I_poly_manual is None:
# # # #         I_poly_manual = weight_nm * I_density
# # # #     else:
# # # #         I_poly_manual += weight_nm * I_density

# # # # manual_wrapper_relerr = (
# # # #     np.linalg.norm(I_poly_manual - I_poly_wrapper)
# # # #     / np.linalg.norm(I_poly_manual)
# # # # )

# # # # print("\n=== Manual vs wrapper integration ===")
# # # # print("manual sum:", np.sum(I_poly_manual))
# # # # print("wrapper sum:", np.sum(I_poly_wrapper))
# # # # print("relative error:", manual_wrapper_relerr)

# # # # assert manual_wrapper_relerr < rtol_loose

# # # # print("PASS 3: get_frame_fresnel_polychromatic matches explicit manual wavelength loop.")


# # # # # ============================================================
# # # # # 4. Detector convention equivalence
# # # # # ============================================================

# # # # I_det_wrapper = bldr.get_frame_fresnel_polychromatic(
# # # #     opd_input=opd_input,
# # # #     amp_input=amp_input,
# # # #     opd_internal=opd_internal,
# # # #     zwfs_ns=zwfs_ns,
# # # #     detector=det,
# # # #     include_shotnoise=False,
# # # #     return_intermediates=False,
# # # # )

# # # # I_det_manual = det.detect(
# # # #     I_poly_manual,
# # # #     include_shotnoise=False,
# # # #     spectral_bandwidth=None,
# # # # )

# # # # det_relerr = (
# # # #     np.linalg.norm(I_det_wrapper - I_det_manual)
# # # #     / np.linalg.norm(I_det_manual)
# # # # )

# # # # print("\n=== Detector convention check ===")
# # # # print("det wrapper shape:", I_det_wrapper.shape)
# # # # print("det manual shape:", I_det_manual.shape)
# # # # print("det wrapper sum:", np.sum(I_det_wrapper))
# # # # print("det manual sum:", np.sum(I_det_manual))
# # # # print("relative error:", det_relerr)

# # # # assert I_det_wrapper.shape == I_det_manual.shape
# # # # assert det_relerr < rtol_loose

# # # # print("PASS 4: wrapper detector path equals manual detect(integrated_rate, spectral_bandwidth=None).")


# # # # # ============================================================
# # # # # 5. Single-wavelength compatibility check
# # # # # ============================================================

# # # # # Use a shallow copy of zwfs_ns so we do not mutate the real config.
# # # # zwfs_single = copy.copy(zwfs_ns)
# # # # zwfs_single.spectrum = SimpleNamespace(
# # # #     enabled=False,
# # # #     mode="monochromatic",
# # # #     wavelengths=np.array([zwfs_ns.optics.wvl0], dtype=float),
# # # #     weights=np.array([1.0], dtype=float),
# # # #     weights_normalized=np.array([1.0], dtype=float),
# # # #     weights_nm=np.array([1.0], dtype=float),
# # # #     bandwidth_nm=1.0,
# # # # )

# # # # I_mono = bldr.get_frame_fresnel(
# # # #     opd_input=opd_input,
# # # #     amp_input=amp_input,
# # # #     opd_internal=opd_internal,
# # # #     zwfs_ns=zwfs_single,
# # # #     detector=None,
# # # #     include_shotnoise=False,
# # # #     spectral_bandwidth=None,
# # # #     wavelength=zwfs_ns.optics.wvl0,
# # # #     return_intermediates=False,
# # # # )

# # # # I_single_poly = bldr.get_frame_fresnel_polychromatic(
# # # #     opd_input=opd_input,
# # # #     amp_input=amp_input,
# # # #     opd_internal=opd_internal,
# # # #     zwfs_ns=zwfs_single,
# # # #     detector=None,
# # # #     include_shotnoise=False,
# # # #     return_intermediates=False,
# # # # )

# # # # single_relerr = (
# # # #     np.linalg.norm(I_single_poly - I_mono)
# # # #     / np.linalg.norm(I_mono)
# # # # )

# # # # print("\n=== Single-wavelength compatibility ===")
# # # # print("mono sum:", np.sum(I_mono))
# # # # print("single-poly sum:", np.sum(I_single_poly))
# # # # print("relative error:", single_relerr)

# # # # assert single_relerr < rtol_loose

# # # # print("PASS 5: single-wavelength polychromatic wrapper equals monochromatic Fresnel frame.")


# # # # # ============================================================
# # # # # 6. Detector flat-band convention check in the simple limiting case
# # # # # ============================================================

# # # # # This does not test Fresnel chromaticity. It tests detector units.
# # # # I_density_flat = np.ones_like(I_mono) * 100.0

# # # # I_rate_explicit = I_density_flat * np.sum(weights_nm)

# # # # I_det_explicit = det.detect(
# # # #     I_rate_explicit,
# # # #     include_shotnoise=False,
# # # #     spectral_bandwidth=None,
# # # # )

# # # # I_det_legacy = det.detect(
# # # #     I_density_flat,
# # # #     include_shotnoise=False,
# # # #     spectral_bandwidth=np.sum(weights_nm),
# # # # )

# # # # flatband_relerr = (
# # # #     np.linalg.norm(I_det_explicit - I_det_legacy)
# # # #     / np.linalg.norm(I_det_legacy)
# # # # )

# # # # print("\n=== Detector flat-band limiting convention ===")
# # # # print("explicit sum:", np.sum(I_det_explicit))
# # # # print("legacy sum:", np.sum(I_det_legacy))
# # # # print("relative error:", flatband_relerr)

# # # # assert flatband_relerr < rtol_strict

# # # # print("PASS 6: detector convention remains compatible with legacy flat spectral bandwidth.")


# # # # # ============================================================
# # # # # 7. Per-wavelength physical sanity from intermediates
# # # # # ============================================================

# # # # print("\n=== Per-wavelength sanity ===")

# # # # for out in wrapper_out["per_wavelength_outputs"]:
# # # #     wavelength = out["wavelength"]
# # # #     weight_nm = out["weight_nm"]

# # # #     psi_mirror = out["psi_mirror"]
# # # #     mirror_mask_D = out["mirror_mask_D"]
# # # #     field_cold = out["field_coldstop_plane"]
# # # #     coldstop_mask = out["coldstop_mask_phys"]

# # # #     E_mirror_before = fresnel.energy(psi_mirror, out["dx_phys"])
# # # #     E_mirror_after = fresnel.energy(psi_mirror * mirror_mask_D, out["dx_phys"])

# # # #     E_cold_before = np.sum(np.abs(field_cold) ** 2)
# # # #     E_cold_after = np.sum(np.abs(field_cold * coldstop_mask) ** 2)

# # # #     mirror_throughput = E_mirror_after / E_mirror_before
# # # #     cold_throughput = E_cold_after / E_cold_before

# # # #     print(
# # # #         f"lambda={1e6*wavelength:.3f} um, "
# # # #         f"weight={weight_nm:.3f} nm, "
# # # #         f"D throughput={mirror_throughput:.3f}, "
# # # #         f"cold throughput={cold_throughput:.3f}, "
# # # #         f"dx_det={1e6*out['dx_detector']:.3f} um"
# # # #     )

# # # #     assert 0.0 <= mirror_throughput <= 1.0
# # # #     assert 0.0 <= cold_throughput <= 1.0
# # # #     assert out["dx_detector"] > 0
# # # #     assert np.all(np.isfinite(out["intensity_pre_detector"]))
# # # #     assert np.all(out["intensity_pre_detector"] >= 0)

# # # # print("PASS 7: per-wavelength relay throughputs and detector samplings are physically valid.")


# # # # # ============================================================
# # # # # 8. Shape and scaling diagnostics
# # # # # ============================================================

# # # # print("\n=== Final diagnostics ===")
# # # # print("I_poly_wrapper / I_mono sum ratio:", np.sum(I_poly_wrapper) / np.sum(I_mono))
# # # # print("Expected rough scale from weights_nm sum:", np.sum(weights_nm))
# # # # print(
# # # #     "Note: exact ratio need not equal bandwidth because the Fresnel relay, "
# # # #     "phase-to-OPD conversion, diffraction scale, cold stop, and detector-plane "
# # # #     "sampling are wavelength-dependent."
# # # # )


# # # # # ============================================================
# # # # # 9. Optional plots
# # # # # ============================================================

# # # # if make_plots:

# # # #     first_wvl_out = wrapper_out["per_wavelength_outputs"][0]
# # # #     mid_wvl_out = wrapper_out["per_wavelength_outputs"][len(wavelengths) // 2]
# # # #     last_wvl_out = wrapper_out["per_wavelength_outputs"][-1]

# # # #     I_first = first_wvl_out["intensity_pre_detector"]
# # # #     I_mid = mid_wvl_out["intensity_pre_detector"]
# # # #     I_last = last_wvl_out["intensity_pre_detector"]

# # # #     fig, axes = plt.subplots(3, 3, figsize=(16, 14), constrained_layout=True)

# # # #     axes[0, 0].plot(1e6 * wavelengths, weights_nm, "o-")
# # # #     axes[0, 0].set_xlabel("wavelength [um]")
# # # #     axes[0, 0].set_ylabel("weight [nm]")
# # # #     axes[0, 0].set_title("Spectral integration weights")
# # # #     axes[0, 0].grid(alpha=0.3)

# # # #     axes[0, 1].plot(1e6 * wavelengths, zwfs_ns.spectrum.weights_normalized, "o-")
# # # #     axes[0, 1].set_xlabel("wavelength [um]")
# # # #     axes[0, 1].set_ylabel("normalized weight")
# # # #     axes[0, 1].set_title("Normalized spectral shape")
# # # #     axes[0, 1].grid(alpha=0.3)

# # # #     axes[0, 2].plot(
# # # #         1e6 * wavelengths,
# # # #         [
# # # #             np.sum(o["intensity_pre_detector"])
# # # #             for o in wrapper_out["per_wavelength_outputs"]
# # # #         ],
# # # #         "o-",
# # # #     )
# # # #     axes[0, 2].set_xlabel("wavelength [um]")
# # # #     axes[0, 2].set_ylabel("sum intensity density")
# # # #     axes[0, 2].set_title("Per-wavelength intensity density")
# # # #     axes[0, 2].grid(alpha=0.3)

# # # #     im = axes[1, 0].imshow(I_first, origin="lower")
# # # #     axes[1, 0].set_title(f"{1e6*first_wvl_out['wavelength']:.2f} um density")
# # # #     plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

# # # #     im = axes[1, 1].imshow(I_mid, origin="lower")
# # # #     axes[1, 1].set_title(f"{1e6*mid_wvl_out['wavelength']:.2f} um density")
# # # #     plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

# # # #     im = axes[1, 2].imshow(I_last, origin="lower")
# # # #     axes[1, 2].set_title(f"{1e6*last_wvl_out['wavelength']:.2f} um density")
# # # #     plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

# # # #     im = axes[2, 0].imshow(I_poly_manual, origin="lower")
# # # #     axes[2, 0].set_title("Manual polychromatic rate")
# # # #     plt.colorbar(im, ax=axes[2, 0], fraction=0.046)

# # # #     im = axes[2, 1].imshow(I_poly_wrapper, origin="lower")
# # # #     axes[2, 1].set_title("Wrapper polychromatic rate")
# # # #     plt.colorbar(im, ax=axes[2, 1], fraction=0.046)

# # # #     diff = I_poly_wrapper - I_poly_manual
# # # #     diff_max = np.max(np.abs(diff))
# # # #     im = axes[2, 2].imshow(
# # # #         diff,
# # # #         origin="lower",
# # # #         cmap="RdBu_r",
# # # #         vmin=-diff_max if diff_max > 0 else None,
# # # #         vmax=diff_max if diff_max > 0 else None,
# # # #     )
# # # #     axes[2, 2].set_title("Wrapper - manual")
# # # #     plt.colorbar(im, ax=axes[2, 2], fraction=0.046)

# # # #     for ax in axes.ravel()[3:]:
# # # #         ax.set_xticks([])
# # # #         ax.set_yticks([])

# # # #     plt.show()


# # # # print("\nAll get_frame_fresnel_polychromatic tests passed.")



# # # # # import sys
# # # # # from pathlib import Path

# # # # # import numpy as np
# # # # # import matplotlib.pyplot as plt

# # # # # repo_root = Path("/Users/bencb/Documents/ASGARD/BaldrApp")
# # # # # if str(repo_root) not in sys.path:
# # # # #     sys.path.insert(0, str(repo_root))

# # # # # from baldrapp.common import baldr_core as bldr
# # # # # from baldrapp.common import fresnel
# # # # # from baldrapp.common import spectrum as spec


# # # # # # ============================================================
# # # # # # 0. Config
# # # # # # ============================================================

# # # # # config_path = (
# # # # #     repo_root
# # # # #     / "baldrapp/apps/paranal_simulator/fake_configs/baldr_config.json"
# # # # # )

# # # # # zwfs_ns = bldr.init_zwfs_from_json(config_path)

# # # # # fr = zwfs_ns.fresnel_relay
# # # # # sp = zwfs_ns.spectrum
# # # # # det = zwfs_ns.detector

# # # # # wavelengths = np.asarray(sp.wavelengths, dtype=float)
# # # # # weights_nm = np.asarray(sp.weights_nm, dtype=float)
# # # # # weights_norm = np.asarray(sp.weights_normalized, dtype=float)

# # # # # print("\n=== Config / spectrum ===")
# # # # # print("N:", zwfs_ns.grid.N)
# # # # # print("dim:", zwfs_ns.grid.dim)
# # # # # print("wvl0 [um]:", 1e6 * zwfs_ns.optics.wvl0)
# # # # # print("wavelengths [um]:", 1e6 * wavelengths)
# # # # # print("weights_normalized:", weights_norm)
# # # # # print("sum weights_normalized:", np.sum(weights_norm))
# # # # # print("weights_nm:", weights_nm)
# # # # # print("sum weights_nm [nm]:", np.sum(weights_nm))
# # # # # print("bandwidth_nm:", sp.bandwidth_nm)
# # # # # print("D_phys [mm]:", 1e3 * fr.D_phys)
# # # # # print("z_focus_to_detector [mm]:", 1e3 * fr.z_focus_to_detector)
# # # # # print("detector binning:", det.binning)


# # # # # # ============================================================
# # # # # # 1. Basic spectrum consistency
# # # # # # ============================================================

# # # # # assert np.all(wavelengths > 0)
# # # # # assert np.all(weights_norm >= 0)
# # # # # assert np.all(weights_nm >= 0)
# # # # # assert np.isclose(np.sum(weights_norm), 1.0, rtol=1e-12, atol=1e-12)
# # # # # assert np.isclose(np.sum(weights_nm), sp.bandwidth_nm, rtol=1e-12, atol=1e-9)

# # # # # print("\nPASS: spectrum weights are non-negative and normalized.")
# # # # # print("PASS: sum(weights_nm) equals bandwidth_nm.")


# # # # # # ============================================================
# # # # # # 2. Detector bandwidth convention
# # # # # # ============================================================

# # # # # # Make a simple flat spectral-density image.
# # # # # I_density = np.ones((72, 72), dtype=float) * 100.0  # photons/s/pix/nm

# # # # # # Legacy flat-spectrum convention.
# # # # # I_det_legacy = det.detect(
# # # # #     I_density,
# # # # #     include_shotnoise=False,
# # # # #     spectral_bandwidth=sp.bandwidth_nm,
# # # # # )

# # # # # # Explicit spectral integration convention.
# # # # # I_rate = I_density * np.sum(weights_nm)  # photons/s/pix
# # # # # I_det_integrated = det.detect(
# # # # #     I_rate,
# # # # #     include_shotnoise=False,
# # # # #     spectral_bandwidth=None,
# # # # # )

# # # # # rel_err_detector = (
# # # # #     np.linalg.norm(I_det_legacy - I_det_integrated)
# # # # #     / np.linalg.norm(I_det_legacy)
# # # # # )

# # # # # print("\n=== Detector convention check ===")
# # # # # print("legacy detected sum:", np.sum(I_det_legacy))
# # # # # print("integrated detected sum:", np.sum(I_det_integrated))
# # # # # print("relative difference:", rel_err_detector)

# # # # # assert rel_err_detector < 1e-12
# # # # # print("PASS: detector legacy flat-band and explicit integrated-rate conventions agree.")


# # # # # # ============================================================
# # # # # # 3. Prepare ZWFS field at one wavelength
# # # # # # ============================================================

# # # # # pupil = zwfs_ns.grid.pupil_mask
# # # # # pupil_bool = pupil.astype(bool)

# # # # # photons_per_s_pix_nm = 1.0e5
# # # # # amp_input = np.sqrt(photons_per_s_pix_nm) * pupil

# # # # # opd = np.zeros_like(pupil)
# # # # # opd_internal = np.zeros_like(pupil)
# # # # # opd_total = pupil * (opd + opd_internal)

# # # # # wvl = zwfs_ns.optics.wvl0
# # # # # theta = spec.theta_at_wavelength(zwfs_ns.optics, wvl)
# # # # # mask_diam = spec.phasemask_diameter_at_wavelength(
# # # # #     zwfs_ns.optics,
# # # # #     wavelength_m=wvl,
# # # # #     default_wvl0=zwfs_ns.optics.wvl0,
# # # # # )

# # # # # zwfs_terms = bldr.get_zwfs_output_field_from_opd(
# # # # #     opd=opd_total,
# # # # #     amp=amp_input,
# # # # #     wavelength=wvl,
# # # # #     zwfs_ns=zwfs_ns,
# # # # #     theta=theta,
# # # # #     phasemask_diameter=mask_diam,
# # # # #     return_terms=True,
# # # # # )

# # # # # psi_A = zwfs_terms["psi_A"]
# # # # # psi_C = zwfs_terms["psi_C"]

# # # # # dx_phys = fr.D_phys / zwfs_ns.grid.N

# # # # # E_A = fresnel.energy(psi_A, dx_phys)
# # # # # E_C = fresnel.energy(psi_C, dx_phys)

# # # # # print("\n=== Analytic ZWFS field energy ===")
# # # # # print("E_A:", E_A)
# # # # # print("E_C:", E_C)
# # # # # print("E_C / E_A:", E_C / E_A)

# # # # # # The ZWFS phase mask redistributes/interferes; depending on implementation
# # # # # # and finite numerical sampling this may not be exactly unit-energy.
# # # # # # Do not assert equality too tightly here.
# # # # # assert np.isfinite(E_A)
# # # # # assert np.isfinite(E_C)
# # # # # assert E_C > 0
# # # # # print("PASS: analytic ZWFS field has finite positive energy.")


# # # # # # ============================================================
# # # # # # 4. Free-space Fresnel propagation energy conservation
# # # # # # ============================================================

# # # # # psi_prop = fresnel.propagate(
# # # # #     psi_C,
# # # # #     wavelength=wvl,
# # # # #     dx=dx_phys,
# # # # #     z=fr.z_to_mirror,
# # # # #     method="angular_spectrum",
# # # # # )

# # # # # E_before = fresnel.energy(psi_C, dx_phys)
# # # # # E_after = fresnel.energy(psi_prop, dx_phys)
# # # # # rel_err_free_space = (E_after - E_before) / E_before

# # # # # print("\n=== Free-space propagation energy conservation ===")
# # # # # print("E_before:", E_before)
# # # # # print("E_after:", E_after)
# # # # # print("relative error:", rel_err_free_space)

# # # # # assert abs(rel_err_free_space) < 1e-10
# # # # # print("PASS: angular-spectrum free-space propagation conserves energy.")


# # # # # # ============================================================
# # # # # # 5. Fresnel relay throughput diagnostics
# # # # # # ============================================================

# # # # # out = bldr.get_frame_fresnel(
# # # # #     opd_input=opd,
# # # # #     amp_input=amp_input,
# # # # #     opd_internal=opd_internal,
# # # # #     zwfs_ns=zwfs_ns,
# # # # #     detector=None,
# # # # #     include_shotnoise=False,
# # # # #     spectral_bandwidth=None,
# # # # #     wavelength=wvl,
# # # # #     return_intermediates=True,
# # # # # )

# # # # # I_pre_detector = out["intensity_pre_detector"]
# # # # # psi_mirror = out["psi_mirror"]
# # # # # mirror_mask_D = out["mirror_mask_D"]
# # # # # field_cold = out["field_coldstop_plane"]
# # # # # coldstop_mask = out["coldstop_mask_phys"]

# # # # # E_mirror_before = fresnel.energy(psi_mirror, out["dx_phys"])
# # # # # E_mirror_after = fresnel.energy(psi_mirror * mirror_mask_D, out["dx_phys"])

# # # # # E_cold_before = np.sum(np.abs(field_cold) ** 2)
# # # # # E_cold_after = np.sum(np.abs(field_cold * coldstop_mask) ** 2)

# # # # # print("\n=== Relay aperture throughputs ===")
# # # # # print("D mirror throughput:", E_mirror_after / E_mirror_before)
# # # # # print("Cold stop throughput:", E_cold_after / E_cold_before)
# # # # # print("pre-detector intensity sum:", np.sum(I_pre_detector))

# # # # # assert 0.0 <= E_mirror_after / E_mirror_before <= 1.0
# # # # # assert 0.0 <= E_cold_after / E_cold_before <= 1.0
# # # # # assert np.all(np.isfinite(I_pre_detector))
# # # # # assert np.all(I_pre_detector >= 0)
# # # # # print("PASS: relay aperture throughputs are within [0, 1].")


# # # # # # ============================================================
# # # # # # 6. Polychromatic integration consistency without detector
# # # # # # ============================================================

# # # # # I_poly_manual = None

# # # # # for wvl_k, weight_nm in zip(wavelengths, weights_nm):

# # # # #     theta_k = spec.theta_at_wavelength(zwfs_ns.optics, wvl_k)
# # # # #     mask_diam_k = spec.phasemask_diameter_at_wavelength(
# # # # #         zwfs_ns.optics,
# # # # #         wavelength_m=wvl_k,
# # # # #         default_wvl0=zwfs_ns.optics.wvl0,
# # # # #     )

# # # # #     out_k = bldr.get_frame_fresnel(
# # # # #         opd_input=opd,
# # # # #         amp_input=amp_input,
# # # # #         opd_internal=opd_internal,
# # # # #         zwfs_ns=zwfs_ns,
# # # # #         detector=None,
# # # # #         include_shotnoise=False,
# # # # #         spectral_bandwidth=None,
# # # # #         wavelength=wvl_k,
# # # # #         theta=theta_k,
# # # # #         phasemask_diameter=mask_diam_k,
# # # # #         return_intermediates=False,
# # # # #     )

# # # # #     if I_poly_manual is None:
# # # # #         I_poly_manual = weight_nm * out_k
# # # # #     else:
# # # # #         I_poly_manual += weight_nm * out_k

# # # # # print("\n=== Polychromatic manual integration ===")
# # # # # print("I_poly_manual shape:", I_poly_manual.shape)
# # # # # print("I_poly_manual sum:", np.sum(I_poly_manual))
# # # # # print("I_poly_manual min/max:", np.min(I_poly_manual), np.max(I_poly_manual))

# # # # # assert np.all(np.isfinite(I_poly_manual))
# # # # # assert np.all(I_poly_manual >= 0)
# # # # # print("PASS: manual polychromatic Fresnel integration produced finite positive intensity.")


# # # # # # ============================================================
# # # # # # 7. Detector after polychromatic integration
# # # # # # ============================================================

# # # # # I_poly_det = det.detect(
# # # # #     I_poly_manual,
# # # # #     include_shotnoise=False,
# # # # #     spectral_bandwidth=None,
# # # # # )

# # # # # print("\n=== Polychromatic detector output ===")
# # # # # print("I_poly_det shape:", I_poly_det.shape)
# # # # # print("I_poly_det sum:", np.sum(I_poly_det))
# # # # # print("I_poly_det min/max:", np.min(I_poly_det), np.max(I_poly_det))

# # # # # assert np.all(np.isfinite(I_poly_det))
# # # # # assert np.all(I_poly_det >= 0)
# # # # # print("PASS: detector accepts spectrally integrated polychromatic frame.")


# # # # # # ============================================================
# # # # # # 8. Plot diagnostics
# # # # # # ============================================================

# # # # # fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

# # # # # axes[0, 0].plot(1e6 * wavelengths, weights_norm, "o-")
# # # # # axes[0, 0].set_xlabel("wavelength [um]")
# # # # # axes[0, 0].set_ylabel("normalized weight")
# # # # # axes[0, 0].set_title("Blackbody spectral weights")
# # # # # axes[0, 0].grid(alpha=0.3)

# # # # # axes[0, 1].plot(1e6 * wavelengths, weights_nm, "o-")
# # # # # axes[0, 1].set_xlabel("wavelength [um]")
# # # # # axes[0, 1].set_ylabel("integration weight [nm]")
# # # # # axes[0, 1].set_title("Integration weights")
# # # # # axes[0, 1].grid(alpha=0.3)

# # # # # im = axes[0, 2].imshow(np.abs(psi_C) ** 2, origin="lower")
# # # # # axes[0, 2].set_title("Monochromatic raw ZWFS intensity")
# # # # # plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

# # # # # im = axes[1, 0].imshow(I_pre_detector, origin="lower")
# # # # # axes[1, 0].set_title("Monochromatic Fresnel detector-plane intensity")
# # # # # plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

# # # # # im = axes[1, 1].imshow(I_poly_manual, origin="lower")
# # # # # axes[1, 1].set_title("Polychromatic integrated detector-plane rate")
# # # # # plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

# # # # # im = axes[1, 2].imshow(I_poly_det, origin="lower")
# # # # # axes[1, 2].set_title("After detector binning/QE/DIT")
# # # # # plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

# # # # # for ax in axes.ravel()[2:]:
# # # # #     ax.set_xticks([])
# # # # #     ax.set_yticks([])

# # # # # plt.show()

# # # # # print("\nAll consistency checks passed.")