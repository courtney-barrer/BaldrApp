import numpy as np  ##(version 2.1.1 works but incompatiple with numba)

# from numba import njit
import os

# import json
from pathlib import Path
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy.stats import pearsonr
from scipy.signal import TransferFunction, bode
from scipy.ndimage import binary_erosion
import datetime
from astropy.io import fits
import pickle
import baldrapp.utils.pupils as aperture
from . import utilities as util
from . import DM_basis
from . import phasescreens
from . import DM_registration

from . import config_helper as cfghelp
from . import spectrum as spec
from . import fresnel_old as fresnel
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, List
from numpy.typing import NDArray

from fresnel import InplacePropagator

"""
1/11/25 - updated get_pupil_intensity to use fft and not mft for speed, added coldstop diam and coldstop_offset as input argunments removes and remove precomupted mask 

25/5/26 - updating with fresnel propagation and baldr specific optics, updating polychromatic (gneeralised spectrum) configuration

"""


_AUTO_SPECTRAL_BANDWIDTH = "auto"


def _resolve_detector_spectral_bandwidth(
    zwfs_ns,
    spectral_bandwidth=_AUTO_SPECTRAL_BANDWIDTH,
):
    """
    Resolve the bandwidth passed to detect(...).

    spectral_bandwidth == "auto"
        Backwards-compatible default. Use zwfs_ns.stellar.bandwidth if present.

    spectral_bandwidth is a float
        Use this explicit bandwidth in nm.

    spectral_bandwidth is None
        Treat the input image as already spectrally integrated.
    """
    if spectral_bandwidth != _AUTO_SPECTRAL_BANDWIDTH:
        return spectral_bandwidth

    if hasattr(zwfs_ns, "stellar") and hasattr(zwfs_ns.stellar, "bandwidth"):
        return zwfs_ns.stellar.bandwidth

    return None




class detector:
    def __init__(self, dit, ron, qe, binning):
        """_summary_

        Args:
            binning (tuple): _description_ binning factor (rows to sum, columns to sum)
            qe (scalar): _description_ quantum efficiency of detector
            dit (scalar): _description_ integration time of detector
            ron (int): _description_. readout noise in electrons per pixel
        """
        self.dit = dit
        self.ron = ron
        self.qe = qe
        self.binning = binning

    def detect(self, i, include_shotnoise=True, spectral_bandwidth=None):
        """
        Convert an input photon-rate image into a detected detector frame.

        Unit convention
        ---------------
        This detector method supports two closely related input conventions,
        controlled by the `spectral_bandwidth` argument.

        1. Monochromatic or spectral-density input

            If `spectral_bandwidth` is not None, the input image `i` is assumed
            to have units:

                photons / second / wave-space-pixel / nm

            The detector first spatially bins the image, then multiplies by:

                spectral_bandwidth * dit * qe

            to obtain detected photoelectrons.

            This is the historical BaldrApp convention and corresponds to a flat
            spectrum across the supplied spectral bandwidth.

        2. Already spectrally integrated input

            If `spectral_bandwidth` is None, the input image `i` is assumed to
            already have units:

                photons / second / wave-space-pixel

            In this case no spectral-bandwidth multiplication is applied. The
            detector only applies spatial binning, detector integration time, QE,
            optional shot noise, and read noise.

        Polychromatic simulations
        -------------------------
        For non-flat or explicitly sampled spectra, spectral integration should be
        performed before calling this detector method.

        If each monochromatic image `I_density_k` has units:

            photons / second / wave-space-pixel / nm

        then the spectral integration must include wavelength-bin widths. For a
        discrete wavelength grid this is generally:

            I_rate = sum_k I_density_k * spectral_shape_k * delta_lambda_k_nm

        where `delta_lambda_k_nm` is the wavelength-bin width associated with
        wavelength sample k, and `spectral_shape_k` is the relative spectral shape
        evaluated at that sample.

        Equivalently, define integration weights:

            weights_nm_k = spectral_shape_k * delta_lambda_k_nm

        with units of nm. Then:

            I_rate = sum_k weights_nm_k * I_density_k

        The resulting `I_rate` has units:

            photons / second / wave-space-pixel

        and should be passed to this detector method with:

            spectral_bandwidth=None

        If using fractional weights normalized such that:

            sum_k weight_k = 1

        then the full-band integration can also be written as:

            I_rate = bandwidth_nm * sum_k weight_k * I_density_k

        but this is only equivalent when:

            bandwidth_nm * weight_k

        correctly represents the wavelength-bin integration weight for sample k.
        To avoid ambiguity, higher-level polychromatic frame-generation code should
        prefer explicit `weights_nm` or `delta_lambda_k_nm` integration weights.

        Design note
        -----------
        The detector intentionally does not inspect `zwfs_ns.spectrum` or any
        instrument-level spectral configuration. Spectral weighting, chromatic ZWFS
        propagation, Fresnel propagation, and bandwidth integration should be
        handled by higher-level frame-generation functions before this detector
        method is called.

        In other words:

            - Use `spectral_bandwidth` only for the legacy flat-spectrum density
            convention.
            - Use `spectral_bandwidth=None` when the image has already been
            integrated over wavelength.

        Parameters
        ----------
        i : ndarray
            Input photon image. Units depend on `spectral_bandwidth`, as described
            above.

        include_shotnoise : bool
            If True, apply Poisson shot noise after conversion to expected detected
            photoelectrons.

        spectral_bandwidth :  float, None, or "auto"
            Spectral bandwidth in nm for the legacy flat-spectrum convention.
            If None, `i` is treated as already spectrally integrated.

        Returns
        -------
        detected_image : ndarray
            Detector image after binning, detector integration time, QE, optional
            shot noise, and read noise.
        """

        """_summary_
        
        copy of the detect function generalized for this class
        
        assumes input intensity is in photons per second per pixel per nm, 
        if spectral_bandwidth is None than returns photons per pixel per nm of input light,
        otherwise returns photons per pixel
        
        Args:
            i (2D array like): _description_ input intensity (abs(field)**2!!!) before being detected on a detector (generally higher spatial resolution than detector)
            include_shotnoise (bool, optional): _description_. Defaults to True. Sample poisson distribution for each pixel (input intensity is the expected value)
            spectral_bandwidth (_type_, optional): _description_. Defaults to None. if spectral_bandwidth is None than returns photons per pixel per nm of input light,
        """

        i = sum_subarrays(array=i, block_size=(self.binning, self.binning))

        if spectral_bandwidth is None:
            i *= self.qe * self.dit
        else:
            i *= self.qe * self.dit * spectral_bandwidth

        if include_shotnoise:
            noisy_intensity = np.random.poisson(lam=i)
        else:  # no noise
            noisy_intensity = i

        if self.ron > 0:
            noisy_intensity += np.random.normal(
                0, self.ron, noisy_intensity.shape
            ).astype(int)

        return np.clip(noisy_intensity, 0, np.inf)


### DM actuator influence options
# @njit
def gaussian_displacement(c_i, sigma_i, x, y, x0, y0):
    """Compute Gaussian displacement for a single actuator."""
    return c_i * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / sigma_i**2)
    # return c_i / (1 + ((x - x0)**2 + (y - y0)**2) / (0.7*sigma_i)**2) #lorentzian approximation to speed things up - good blanace accuracy and speed
    # return np.clip(c_i * (1 - ((x - x0)**2 + (y - y0)**2)/ (1.2 * sigma_i**2)), a_min=0, a_max=None) #quadratic
    # return c_i * ( (x - x0)**2 + (y - y0)**2 <= sigma_i**2 ) # box profile'



def dm_to_pixel(dm_coord, transform_matrix):
    """
    Converts DM coordinates to pixel coordinates using the affine transformation.

    Args:
        dm_coord: A tuple (x, y) in DM space.
        transform_matrix: The affine transformation matrix from DM space to pixel space.

    Returns:
        Tuple (x_pixel, y_pixel) in pixel coordinates.
    """
    dm_coord = np.array(dm_coord)
    pixel_coord = np.dot(transform_matrix[:, :2], dm_coord) + transform_matrix[:, 2]
    return tuple(pixel_coord)


def convert_to_serializable(obj):
    """
    Recursively converts NumPy arrays and other non-serializable objects to serializable forms.
    Also converts dictionary keys to standard types (str, int, float).
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    elif isinstance(obj, np.integer):
        return int(obj)  # Convert NumPy integers to Python int
    elif isinstance(obj, np.floating):
        return float(obj)  # Convert NumPy floats to Python float
    elif isinstance(obj, dict):
        return {
            str(key): convert_to_serializable(value) for key, value in obj.items()
        }  # Ensure keys are strings
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj  # Base case: return the object itself if it doesn't need conversion


# @njit(parallel=True)
def get_dm_displacement(command_vector, gain, sigma, X, Y, x0, y0):
    """_summary_

    Args:
        command_vector (_type_): _description_
        gain (float): command to opd for all actuators
        sigma (1D array): interactuator coupling, length= # actuators
        X (2D meshgrid): X coordinates of the space you want the DM to be in (e.g. pixel space)
        Y (2D meshgrid): Y coordinates of the space you want the DM to be in (e.g. pixel space)
        x0 (1D array): DM actuator x centers in X,Y space length= # actuators
        y0 (1D array): DM actuator y centers in X,Y space length= # actuators

    Returns:
        2D array: displancement map of DM in X,Y space
    """
    displacement_map = np.zeros(X.shape)
    for i in range(len(command_vector)):
        # print(i)
        displacement_map += gaussian_displacement(
            c_i=gain * command_vector[i], sigma_i=sigma[i], x=X, y=Y, x0=x0[i], y0=y0[i]
        )

    return displacement_map


def get_zwfs_output_field(
    phi,
    amp,
    theta,
    phasemask_diameter,
    phasemask_mask=None,
    pupil_diameter=None,
    fplane_pixels=300,
    pixels_across_mask=10,
    return_terms=False,
):
    """
    Compute the analytic ZWFS exit-pupil complex field using N'Diaye Eq. 6.

    This returns the field immediately after the ZWFS phase-mask operation,
    before any downstream cold stop, relay propagation, detector binning, etc.

    Model:
        psi_A = amp * exp(i phi)

        b = IFFT{ M * FFT{psi_A} }

        psi_C = psi_A - (1 - exp(i theta)) * b
              = psi_A + (exp(i theta) - 1) * b

    Parameters
    ----------
    phi : 2D array
        Input phase in radians.
    amp : 2D array
        Input field amplitude. Usually includes the pupil support.
    theta : float
        Phase shift of the ZWFS mask in radians.
    phasemask_diameter : float
        Phase-mask diameter in the same lambda/D convention used by
        get_pupil_intensity().
    phasemask_mask : 2D array or None
        Optional precomputed focal-plane mask. If None, build a circular
        mask using phasemask_diameter and the FFT sampling.
    pupil_diameter : float or None
        Kept for interface compatibility. Not required by this implementation.
    fplane_pixels : int
        FFT/focal-plane grid size. Internally promoted to odd size if needed.
    pixels_across_mask : float
        Fallback sampling scale if the mask/pupil support cannot be inferred.
    return_terms : bool
        If True, return a dict with intermediate fields and masks.

    Returns
    -------
    psi_C : 2D complex array
        Analytic ZWFS output field cropped back to the original phi/amp shape.

    Or, if return_terms=True, returns dict with:
        psi_C, psi_A, b, phase_disc_fp, pix_per_wvld
    """

    del pupil_diameter  # currently unused; kept for compatibility

    phi = np.asarray(phi)
    amp = np.asarray(amp)

    if phi.shape != amp.shape or phi.ndim != 2:
        raise ValueError("phi and amp must be 2D arrays with the same shape.")

    if phi.shape[0] != phi.shape[1]:
        raise ValueError("phi and amp must be square arrays.")

    N_orig = phi.shape[0]

    # Match the even-grid safety convention already used in get_pupil_intensity().
    padded_even = False
    if (N_orig % 2) == 0:
        phi_work = np.pad(phi, ((0, 1), (0, 1)), mode="constant")
        amp_work = np.pad(amp, ((0, 1), (0, 1)), mode="constant")
        N = N_orig + 1
        padded_even = True
    else:
        phi_work = phi
        amp_work = amp
        N = N_orig

    # Entrance-pupil field.
    psi_A = amp_work.astype(np.complex128) * np.exp(1j * phi_work.astype(np.float64))

    # Use an odd focal grid to keep centering unambiguous.
    Nf = int(max(fplane_pixels, N))
    if (Nf % 2) == 0:
        Nf += 1

    pad = (Nf - N) // 2

    psi_A_pad = np.zeros((Nf, Nf), dtype=np.complex128)
    psi_A_pad[pad : pad + N, pad : pad + N] = psi_A

    # Focal-plane field before phase mask.
    psi_B = np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(psi_A_pad),
            norm="ortho",
        )
    )

    # Build or resize the focal-plane phase-disc mask.
    if phasemask_mask is not None:
        m = np.asarray(phasemask_mask, dtype=float)

        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("phasemask_mask must be a square 2D array.")

        if m.shape[0] == Nf:
            phase_disc = (m > 0.5).astype(float)
        else:
            phase_disc = np.zeros((Nf, Nf), dtype=float)

            Ns = m.shape[0]
            c_src = (Ns - 1) // 2
            c_dst = (Nf - 1) // 2

            src_y0 = max(0, c_src - c_dst)
            src_x0 = max(0, c_src - c_dst)
            dst_y0 = max(0, c_dst - c_src)
            dst_x0 = max(0, c_dst - c_src)

            h = min(Ns - src_y0, Nf - dst_y0)
            w = min(Ns - src_x0, Nf - dst_x0)

            phase_disc[
                dst_y0 : dst_y0 + h,
                dst_x0 : dst_x0 + w,
            ] = m[
                src_y0 : src_y0 + h,
                src_x0 : src_x0 + w,
            ]

            phase_disc = (phase_disc > 0.5).astype(float)

        # Infer pixels per lambda/D from the supplied mask.
        yy, xx = np.indices((Nf, Nf))
        c = (Nf - 1) / 2.0
        rr = np.hypot(yy - c, xx - c)

        if np.any(phase_disc > 0):
            r_pix = rr[phase_disc > 0].max()
            pix_per_wvld = (2.0 * r_pix) / float(phasemask_diameter)
        else:
            pix_per_wvld = float(pixels_across_mask)

    else:
        # Infer pixels per lambda/D from the padded pupil support.
        amp_pad = np.zeros((Nf, Nf), dtype=float)
        amp_pad[pad : pad + N, pad : pad + N] = np.abs(amp_work)

        support = amp_pad > 0

        yy, xx = np.indices((Nf, Nf))
        c = (Nf - 1) / 2.0
        rr = np.hypot(yy - c, xx - c)

        if np.any(support):
            r_pupil_pix = rr[support].max()
            d_pupil_pix = 2.0 * r_pupil_pix
            pix_per_wvld = float(Nf) / max(d_pupil_pix, 1e-12)
        else:
            pix_per_wvld = float(pixels_across_mask)

        r_mask_pix = 0.5 * float(phasemask_diameter) * pix_per_wvld
        phase_disc = (rr <= r_mask_pix).astype(float)

    # Reference wave b = IFFT{M FFT{psi_A}}.
    b_pad = np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(phase_disc * psi_B),
            norm="ortho",
        )
    )

    # N'Diaye Eq. 6:
    # psi_C = psi_A - (1 - exp(i theta)) b
    #       = psi_A + (exp(i theta) - 1) b
    psi_C_pad = psi_A_pad + (np.exp(1j * float(theta)) - 1.0) * b_pad

    # Crop back to the caller's original pupil/grid size.
    psi_A_crop = psi_A_pad[pad : pad + N, pad : pad + N]
    b_crop = b_pad[pad : pad + N, pad : pad + N]
    psi_C_crop = psi_C_pad[pad : pad + N, pad : pad + N]

    if padded_even:
        psi_A_crop = psi_A_crop[:N_orig, :N_orig]
        b_crop = b_crop[:N_orig, :N_orig]
        psi_C_crop = psi_C_crop[:N_orig, :N_orig]

    if return_terms:
        return {
            "psi_C": psi_C_crop,
            "psi_A": psi_A_crop,
            "b": b_crop,
            "phase_disc_fp": phase_disc,
            "pix_per_wvld": pix_per_wvld,
        }

    return psi_C_crop


def get_pupil_intensity(
    phi,
    amp,
    theta,
    phasemask_diameter,
    phasemask_mask,
    pupil_diameter,
    fplane_pixels=300,
    pixels_across_mask=10,
    *,
    coldstop_diam=None,
    coldstop_offset=(0.0, 0.0),
    coldstop_mask=None,
    include_beta=True,
    return_field=False,
    return_terms=False,
    debug=False,
):
    """
    ZWFS pupil intensity with analytic output field and focal-plane cold stop.
    Tilt-safe for even-sized inputs: internally pads to odd size to avoid half-pixel centering errors,
    then crops back to the caller’s original size.

    #     Args:
    #         phi (_type_): input phase (radians)
    #         amp (_type_): input amplitude of field (sqrt of intensity)
    #         theta (_type_): phaseshift of mask (rad)
    #         phasemask_diameter ( ) : diameter in units of 1.22 * F * lambda/D of phasemask
    #         phasemask_mask ( ) : 2D array of the phaseshifting region in image plane (input to make things quicker)
    #         pupil_diameter () : diameter of pupil in pixels
    #         fplane_pixels (int) : number of pixels in focal plane
    #         pixels_across_mask (int) : number of pixels across the phase shifting region of mask in focal plane
    #     Returns:
    #         _type_: ZWFS pupil intensity

    """
    # t0 = time.time()

    # ---------------- even-grid safe: promote to odd internally ----------------
    phi = np.asarray(phi)
    amp = np.asarray(amp)
    assert phi.shape == amp.shape and phi.ndim == 2, "phi and amp must be same 2D shape"
    N_orig = amp.shape[0]
    assert amp.shape[1] == N_orig, "phi/amp must be square"

    padded_even = False
    if (N_orig % 2) == 0:
        # pad ONE row/col (bottom/right) -> internal odd grid; center aligns to a pixel
        amp_work = np.pad(amp, ((0, 1), (0, 1)), mode="constant")
        phi_work = np.pad(phi, ((0, 1), (0, 1)), mode="constant")
        N = N_orig + 1
        padded_even = True
    else:
        amp_work = amp
        phi_work = phi
        N = N_orig

    # ---------------- build pupil-plane field Ψ_A on the odd (or original odd) grid ----------------
    psi_A = amp_work.astype(np.complex128) * np.exp(1j * phi_work.astype(np.float64))

    # ---------------- choose focal-plane FFT size (prefer odd to keep centering simple) -------------
    Nf = int(max(fplane_pixels, N))
    if (Nf % 2) == 0:  # enforce odd focal grid as well
        Nf += 1

    # ---------------- center embed to focal grid ----------------
    pad = (Nf - N) // 2  # integer because both are odd
    psi_A_pad = np.zeros((Nf, Nf), dtype=np.complex128)
    psi_A_pad[pad : pad + N, pad : pad + N] = psi_A

    # ---------------- forward FFT to focal plane ----------------
    psi_B = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(psi_A_pad), norm="ortho")
    )  # norm="ortho" to maintain flux conservation

    # ---------------- phase disc (derive physical pixels-per-(λ/D) correctly) ----------------
    def _center_resize_mask(mask, tgtN):
        m = np.asarray(mask, float)
        assert m.ndim == 2 and m.shape[0] == m.shape[1], "mask must be square"
        if m.shape[0] == tgtN:
            return (m > 0.5).astype(float)
        out = np.zeros((tgtN, tgtN), float)
        c0 = (m.shape[0] - 1) // 2
        ct = (tgtN - 1) // 2
        y0s = max(0, ct - c0)
        y0d = max(0, c0 - ct)
        h = min(m.shape[0], tgtN)
        out[y0s : y0s + h, y0s : y0s + h] = m[y0d : y0d + h, y0d : y0d + h]
        return (out > 0.5).astype(float)

    def _disc_radius_in_pixels(bin_mask):
        if bin_mask.max() == 0:
            return 0.0
        yy, xx = np.indices(bin_mask.shape)
        c = (bin_mask.shape[0] - 1) / 2.0
        r = np.hypot(yy - c, xx - c)
        return r[bin_mask.astype(bool)].max()

    if phasemask_mask is not None:
        phase_disc = _center_resize_mask(phasemask_mask, Nf)
        r_pix = _disc_radius_in_pixels(phase_disc)
        if r_pix <= 0 or phasemask_diameter <= 0:
            # fallback; will be superseded by cold-stop scaling anyway
            pix_per_wvld = float(pixels_across_mask)
        else:
            pix_per_wvld = (2.0 * r_pix) / float(phasemask_diameter)
    else:
        # derive physical pixels-per-(λ/D) from the pupil diameter on the padded grid
        amp_pad = np.zeros((Nf, Nf), dtype=float)
        amp_pad[pad : pad + N, pad : pad + N] = np.abs(amp_work)
        support = amp_pad > 0
        yy, xx = np.indices((Nf, Nf))
        c = (Nf - 1) / 2.0
        r = np.hypot(yy - c, xx - c)
        if np.any(support):
            Rpix = r[support].max()
            Dp_pix = 2.0 * Rpix
            pix_per_wvld = float(Nf) / max(Dp_pix, 1e-9)
        else:
            pix_per_wvld = float(pixels_across_mask)

        r_mask_pix = 0.5 * float(phasemask_diameter) * pix_per_wvld
        phase_disc = (r <= r_mask_pix).astype(float)

    # ---------------- reference arm & analytic pupil field ----------------
    b = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(phase_disc * psi_B), norm="ortho")
    )
    ejt_minus_1 = np.exp(1j * float(theta)) - 1.0
    psi_theta_pad = psi_A_pad + ejt_minus_1 * b

    # ---------------- forward to focal plane, apply cold stop ----------------
    psi_theta_B = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(psi_theta_pad), norm="ortho")
    )

    if coldstop_mask is not None:
        C = _center_resize_mask(coldstop_mask, Nf)
    else:
        if coldstop_diam is None:
            C = np.ones((Nf, Nf), float)
        else:
            dx_wvld, dy_wvld = coldstop_offset
            dx_pix = float(dx_wvld) * pix_per_wvld
            dy_pix = float(dy_wvld) * pix_per_wvld
            r_cs_pix = 0.5 * float(coldstop_diam) * pix_per_wvld
            yy, xx = np.indices((Nf, Nf))
            c = (Nf - 1) / 2.0
            r_cs = np.hypot(yy - (c + dy_pix), xx - (c + dx_pix))
            C = (r_cs <= r_cs_pix).astype(float)

    psi_out_pad = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(C * psi_theta_B), norm="ortho")
    )

    # ---------------- crop back to caller's original size ----------------
    # First crop to the internal pupil block (N×N), then drop the extra row/col if we padded.
    psi_crop = psi_out_pad[pad : pad + N, pad : pad + N]
    if padded_even:
        psi_out = psi_crop[:N_orig, :N_orig]
    else:
        psi_out = psi_crop
    Ic = np.abs(psi_out) ** 2

    # ---------------- optional diag ----------------
    if debug:
        psf = np.abs(psi_theta_B) ** 2
        if psf.max() > 0:
            psf /= psf.max()
        plt.figure(figsize=(5, 4))
        plt.imshow(psf, origin="lower", cmap="gray")
        plt.contour(phase_disc, levels=[0.5], colors="w", linewidths=1.0)
        if coldstop_diam is not None or coldstop_mask is not None:
            plt.contour(C, levels=[0.5], colors="r", linewidths=1.0)
        plt.title("Focal-plane |Ψ_θ|² (phase disc white, cold stop red)")
        plt.tight_layout()

    # ---------------- returns ----------------
    if return_terms or return_field:
        out = {"Ic": Ic}
        if return_field:
            out["field_pupil"] = psi_out
        if return_terms:
            # also return pupil-plane terms cropped back to caller’s shape for convenience
            b_crop = b[pad : pad + N, pad : pad + N]
            psi_theta_crop = psi_theta_pad[pad : pad + N, pad : pad + N]
            if padded_even:
                b_crop = b_crop[:N_orig, :N_orig]
                psi_theta_crop = psi_theta_crop[:N_orig, :N_orig]
            out.update(
                {
                    "psi_A": amp * np.exp(1j * phi),  # original-resolution pupil field
                    "b_crop": b_crop,
                    "psi_theta_crop": psi_theta_crop,
                    "phase_disc_fp": phase_disc,
                    "cold_stop_fp": C,
                    "pix_per_wvld": pix_per_wvld,
                }
            )
        return out
    # t1 = time.time()
    # print(t1-t0)
    return Ic


def update_dm_registration_wavespace(transform_matrix, zwfs_ns):
    """_summary_
    # STANDARD WAY TO UPDATE THE REGISTRATION OF THE DM IN WAVE SPACE
    # UPDATES --> zwfs_ns <--- name space !!! Only use this method to update registration

    Args:
        transform_matrix (_type_): _description_ affine transform describing the mapping from DM actuators to the wavefront
        zwfs_ns (_type_): _description_ the zwfs name space (holding configuration details)

    zwfs_ns dependancies (must have in namespace for code to work):
        zwfs_ns.dm.Nact_x  (int)
        zwfs_ns.dm.Nact_y  (int)
        zwfs_ns.dm.dm_pitch  (float)
        zwfs_ns.dm.actuator_coupling_factor (float)
        zwfs_ns.dm.current_cmd (1D array, size 140 for BMC multi-3.5 DM)

    """

    dm_coords, dm_actuator_to_coord, dm_coord_to_actuator = (
        DM_registration.generate_dm_coordinates(
            Nx=zwfs_ns.dm.Nact_x, Ny=zwfs_ns.dm.Nact_y, spacing=zwfs_ns.dm.dm_pitch
        )
    )

    # plt.figure(); plt.scatter([xx[0] for xx in dm_coords], [xx[1] for xx in dm_coords] ); plt.show();

    pixel_coord_list = np.array([dm_to_pixel(c, transform_matrix) for c in dm_coords])

    # plt.figure(); plt.scatter([xx[0] for xx in pixel_coord_list], [xx[1] for xx in pixel_coord_list] ); plt.show();

    # projecting the DM actuator space to wavespace. For convinience this is same as pixel space (before binnning)
    sigma = (
        zwfs_ns.dm.actuator_coupling_factor
        * abs(pixel_coord_list[0][0] - pixel_coord_list[1][0])
        * np.ones(len(zwfs_ns.dm.current_cmd))
    )  # coupling of actuators projected to wavespace

    x0_list = [xx[0] for xx in pixel_coord_list]
    y0_list = [yy[1] for yy in pixel_coord_list]

    dm2wavespace_registration_dict = {
        "dm_to_wavesp_transform_matrix": transform_matrix,  # affine transform from DM coordinates to wave coordinates
        "dm_actuator_to_coord": dm_actuator_to_coord,
        "dm_coord_to_actuator": dm_coord_to_actuator,
    }

    dm_coord_dict = {
        "dm_coords": dm_coords,  # DM coordinates DM space
        "dm_coord_wavesp": pixel_coord_list,
        "act_x0_list_wavesp": x0_list,  # actuator x coorindate in pixel space
        "act_y0_list_wavesp": y0_list,  # actuator y coordinate in pixel space
        "act_sigma_wavesp": sigma,
    }

    dm2wavespace_registration_ns = SimpleNamespace(**dm2wavespace_registration_dict)
    # wave_coord_ns = SimpleNamespace(**wave_coord_dict )
    dm_coord_ns = SimpleNamespace(**dm_coord_dict)

    # Add DM and wave coorindates to grid namespace
    zwfs_ns.grid.dm_coord = dm_coord_ns
    zwfs_ns.dm2wavespace_registration = dm2wavespace_registration_ns

    zwfs_ns.compiled_dm = CompiledDM(
        zwfs_ns.grid.wave_coord.X,
        zwfs_ns.grid.wave_coord.Y,
        np.array(zwfs_ns.grid.dm_coord.act_x0_list_wavesp),
        np.array(zwfs_ns.grid.dm_coord.act_y0_list_wavesp),
        np.array(zwfs_ns.grid.dm_coord.act_sigma_wavesp),
    )

    return zwfs_ns


def init_zwfs(grid_ns, optics_ns, dm_ns):
    #############
    #### GRID

    if hasattr(grid_ns, "telescope"):
        if grid_ns.telescope.upper() == "UT":
            pupil = aperture.vlt_pupil(
                diameter=grid_ns.N,
                dim=int(grid_ns.dim),
                spiders_thickness=0.008 * grid_ns.N / int(grid_ns.dim),
            )  # padding_factor = 2 )
        elif grid_ns.telescope.upper() == "AT":
            pupil = aperture.AT_pupil(
                diameter=grid_ns.N,
                dim=int(grid_ns.dim),
                spiders_thickness=0.016 * grid_ns.N / int(grid_ns.dim),
                strict=False,
                cpix=False,
            )  # , padding_factor = 2 )
        elif grid_ns.telescope.upper() == "DISC":
            pupil = aperture.disc(
                dim=int(grid_ns.dim),
                size=grid_ns.N,
                diameter=True,
                strict=False,
                center=(),
                cpix=False,
                invert=False,
                mask=False,
            )
        elif (
            grid_ns.telescope.upper() == "SOLARSTEIN"
        ):  # ASGARD internal source SOLARSTEIN which has UT like secondary with no spiders
            pupil = aperture.disc_obstructed(
                dim=int(grid_ns.dim),
                size=grid_ns.N,
                obs=1100 / 8000,
                diameter=True,
                strict=False,
            )

        else:
            raise TypeError("telescope not implemented. Try DISC, AT or UT")

    else:
        pupil = aperture.disc(
            dim=int(grid_ns.dim),
            size=grid_ns.N,
            diameter=True,
            strict=False,
            center=(),
            cpix=False,
            invert=False,
            mask=False,
        )
        # raise UserWarning("telescope not defined in grid_ns. Defaulting to disk pupil")
        print("telescope not defined in grid_ns. Defaulting to disk pupil")
    grid_ns.pupil_mask = pupil
    # grid_ns.phasemask_mask = phasemask

    # coorindates in the pupil plance
    padding_factor = grid_ns.dim / grid_ns.N
    x = np.linspace(
        -(grid_ns.D * padding_factor) // 2,
        (grid_ns.D * padding_factor) // 2,
        pupil.shape[0],
    )
    y = np.linspace(
        -(grid_ns.D * padding_factor) // 2,
        (grid_ns.D * padding_factor) // 2,
        pupil.shape[0],
    )
    X, Y = np.meshgrid(x, y)

    wave_coord_dict = {
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
    }

    wave_coord_ns = SimpleNamespace(**wave_coord_dict)

    grid_ns.wave_coord = wave_coord_ns

    # focal plane sampling is hardcoded. we use mft to sample when calculating output intensity.
    # hold a fixed phasemask mask here so don't have to re-init each run
    focal_plane_sampling_dict = {"fplane_pixels": 300, "pixels_across_mask": 10}

    phasemask_mask = aperture.disc(
        focal_plane_sampling_dict["fplane_pixels"],
        focal_plane_sampling_dict["fplane_pixels"]
        // focal_plane_sampling_dict["pixels_across_mask"],
        diameter=True,
        cpix=True,
        strict=False,
    )

    # attach to grid_ns for legacy reasons (not have to make lots of edits )
    grid_ns.phasemask_mask = phasemask_mask

    # grid_ns.coldstop_mask =

    # get coordinates in focal plane
    xmin = (
        -focal_plane_sampling_dict["fplane_pixels"]
        / focal_plane_sampling_dict["pixels_across_mask"]
        * optics_ns.mask_diam
        * optics_ns.wvl0
        * optics_ns.F_number
        / 2
    )
    xmax = (
        focal_plane_sampling_dict["fplane_pixels"]
        / focal_plane_sampling_dict["pixels_across_mask"]
        * optics_ns.mask_diam
        * optics_ns.wvl0
        * optics_ns.F_number
        / 2
    )
    x = np.linspace(xmin, xmax, focal_plane_sampling_dict["fplane_pixels"])
    y = np.linspace(xmin, xmax, focal_plane_sampling_dict["fplane_pixels"])
    X, Y = np.meshgrid(x, y)

    focal_plane_sampling_dict["x"] = x
    focal_plane_sampling_dict["X"] = X
    focal_plane_sampling_dict["y"] = y
    focal_plane_sampling_dict["Y"] = Y

    focal_plane_ns = SimpleNamespace(**focal_plane_sampling_dict)

    #############
    #### DM

    if dm_ns.dm_model == "BMC-multi-3.5":
        # 12x12 with missing corners
        dm_ns.Nact_x = 12  # actuators along DM x axis
        dm_ns.Nact_y = 12  # actuators along DM y axis
        dm_ns.Nact = 140  # total number of actuators
        dm_ns.dm_flat = 0.5 + dm_ns.flat_rmse * np.random.rand(
            dm_ns.Nact
        )  # add some noise to the DM flat
        dm_ns.current_cmd = dm_ns.dm_flat  # default set to dm flat
    else:
        raise TypeError("input DM model not implemented. Try BMC-multi-3.5")

    # set so pupil covers about 10 actuators on BMC multi 3.5 DM
    # np.ptp(wave_coord_dict['y'])
    a, b, c, d = (
        12 / 10 * (grid_ns.D / 2) / (dm_ns.Nact_x / 2 - 0.5),
        0,
        0,
        12 / 10 * (grid_ns.D / 2) / (dm_ns.Nact_x / 2 - 0.5),
    )  # Parameters for affine transform (identity for simplicity)
    # set by default to be centered and overlap with pupil (pupil touches edge of DM )

    t_x, t_y = np.mean(wave_coord_dict["x"]), np.mean(
        wave_coord_dict["y"]
    )  # Translation in phase space

    # we can introduce mis-registrations by rolling input pupil

    dm_act_2_wave_space_transform_matrix = np.array([[a, b, t_x], [c, d, t_y]])

    #### Stellar Name Space
    # detect requires stellar.bandwidth input so we set as None as default
    stellar_ns = SimpleNamespace(**{"bandwidth": None})

    # nned to add
    # ZWFS NAME SPACE
    zwfs_dict = {
        "grid": grid_ns,
        "optics": optics_ns,
        "dm": dm_ns,
        "focal_plane": focal_plane_ns,
        "stellar": stellar_ns,
        # "dm2wavespace_registration" : dm2wavespace_registration_ns
    }

    zwfs_ns = SimpleNamespace(**zwfs_dict)

    # user warning: only ever use update_dm_registration_wavespace if you want a consistent update across all variables
    # this updates the zwfs_ns.grid with dm coords in DM and wavespace as well as defining dm2wavespace_registration namespace
    zwfs_ns = update_dm_registration_wavespace(
        dm_act_2_wave_space_transform_matrix, zwfs_ns
    )

    zwfs_ns.propagator = InplacePropagator(zwfs_ns.grid.wave_coord.X.shape)

    return zwfs_ns


def first_stage_ao(
    atm_scrn, Nmodes_removed, basis, phase_scaling_factor=1, return_reconstructor=False
):
    """Remove the first Nmodes_removed modes by a sampled least-squares fit.

    Fit and reconstruct using the supplied basis on the same pupil pixels.
    Sampled Zernikes are not exactly orthogonal, so independent projections
    can leak atmospheric piston into other modes as the phase screen evolves.

    If return_reconstructor is True, also return the fitted phase so AO
    latency can be simulated by applying it after advancing the phase screen.

    Args:
        atm_scrn (phasescreens.PhaseScreenKolmogorov): atmospheric phase screen object initialized from aotools or common/phasescreens.py
        Nmodes_removed (int): Number of Zernike modes removed in first stage ao
        basis (list of 2D arrays): Zernike basis function on the input atm_scrn pupil footprint
            IMPORTANT - basis[0] should be the pupil disk without secondary mirror
        phase_scaling_factor (float, optional): Scale the phase screen before fitting. Defaults to 1.
        return_reconstructor (bool): Also return the fitted phase for latency simulation.
    """
    basis = np.asarray(basis).T.reshape([-1, *atm_scrn.scrn.shape])
    pupil_disk = basis[0]
    pupil_pixels = pupil_disk > 0
    phase = phase_scaling_factor * atm_scrn.scrn
    modes = basis[:Nmodes_removed]

    # Solve all requested modes together on their original sampling, including
    # piston. Do not crop and regenerate a differently sampled Zernike basis.
    mode_coefficients = np.linalg.lstsq(
        modes[:, pupil_pixels].T, phase[pupil_pixels], rcond=None
    )[0]
    reco = np.sum(mode_coefficients[:, np.newaxis, np.newaxis] * modes, axis=0)

    ao_1 = pupil_disk * (phase - reco)

    if return_reconstructor:
        return ao_1, reco
    else:
        return ao_1


def sum_subarrays(array, block_size):
    """
    Averages non-overlapping sub-arrays of a given 2D NumPy array.

    Parameters:
    array (numpy.ndarray): Input 2D array of shape (N, M).
    block_size (tuple): Size of the sub-array blocks (height, width).

    Returns:
    numpy.ndarray: 2D array containing the averaged values of the sub-arrays.
    """
    # Check if the array dimensions are divisible by the block size
    if array.shape[0] % block_size[0] != 0 or array.shape[1] % block_size[1] != 0:
        raise ValueError("Array dimensions must be divisible by the block size.")

    # Reshape the array to isolate the sub-arrays
    reshaped = array.reshape(
        array.shape[0] // block_size[0],
        block_size[0],
        array.shape[1] // block_size[1],
        block_size[1],
    )

    # Compute the mean of the sub-arrays
    summed_subarrays = reshaped.sum(axis=(1, 3))

    return summed_subarrays


def detect(i, binning, qe, dit, ron=0, include_shotnoise=True, spectral_bandwidth=None):
    """
    Convert an input photon-rate image into a detected detector frame.

    Unit convention
    ---------------
    This detector method supports two closely related input conventions,
    controlled by the `spectral_bandwidth` argument.

    1. Monochromatic or spectral-density input

        If `spectral_bandwidth` is not None, the input image `i` is assumed
        to have units:

            photons / second / wave-space-pixel / nm

        The detector first spatially bins the image, then multiplies by:

            spectral_bandwidth * dit * qe

        to obtain detected photoelectrons.

        This is the historical BaldrApp convention and corresponds to a flat
        spectrum across the supplied spectral bandwidth.

    2. Already spectrally integrated input

        If `spectral_bandwidth` is None, the input image `i` is assumed to
        already have units:

            photons / second / wave-space-pixel

        In this case no spectral-bandwidth multiplication is applied. The
        detector only applies spatial binning, detector integration time, QE,
        optional shot noise, and read noise.

    Polychromatic simulations
    -------------------------
    For non-flat or explicitly sampled spectra, spectral integration should be
    performed before calling this detector method.

    If each monochromatic image `I_density_k` has units:

        photons / second / wave-space-pixel / nm

    then the spectral integration must include wavelength-bin widths. For a
    discrete wavelength grid this is generally:

        I_rate = sum_k I_density_k * spectral_shape_k * delta_lambda_k_nm

    where `delta_lambda_k_nm` is the wavelength-bin width associated with
    wavelength sample k, and `spectral_shape_k` is the relative spectral shape
    evaluated at that sample.

    Equivalently, define integration weights:

        weights_nm_k = spectral_shape_k * delta_lambda_k_nm

    with units of nm. Then:

        I_rate = sum_k weights_nm_k * I_density_k

    The resulting `I_rate` has units:

        photons / second / wave-space-pixel

    and should be passed to this detector method with:

        spectral_bandwidth=None

    If using fractional weights normalized such that:

        sum_k weight_k = 1

    then the full-band integration can also be written as:

        I_rate = bandwidth_nm * sum_k weight_k * I_density_k

    but this is only equivalent when:

        bandwidth_nm * weight_k

    correctly represents the wavelength-bin integration weight for sample k.
    To avoid ambiguity, higher-level polychromatic frame-generation code should
    prefer explicit `weights_nm` or `delta_lambda_k_nm` integration weights.

    Design note
    -----------
    The detector intentionally does not inspect `zwfs_ns.spectrum` or any
    instrument-level spectral configuration. Spectral weighting, chromatic ZWFS
    propagation, Fresnel propagation, and bandwidth integration should be
    handled by higher-level frame-generation functions before this detector
    method is called.

    In other words:

        - Use `spectral_bandwidth` only for the legacy flat-spectrum density
        convention.
        - Use `spectral_bandwidth=None` when the image has already been
        integrated over wavelength.

    Parameters
    ----------
    i : ndarray
        Input photon image. Units depend on `spectral_bandwidth`, as described
        above.

    include_shotnoise : bool
        If True, apply Poisson shot noise after conversion to expected detected
        photoelectrons.

    spectral_bandwidth : float, None, or "auto"
        Spectral bandwidth in nm for the legacy flat-spectrum convention.
        If None, `i` is treated as already spectrally integrated.

    Returns
    -------
    detected_image : ndarray
        Detector image after binning, detector integration time, QE, optional
        shot noise, and read noise.
    """
    """_summary_
    assumes input intensity is in photons per second per pixel per nm, 
    if spectral_bandwidth is None than returns photons per pixel per nm of input light,
    otherwise returns photons per pixel
    
    Args:
        i (2D array like): _description_ input intensity before being detected on a detector (generally higher spatial resolution than detector)
        binning (tuple): _description_ binning factor (rows to sum, columns to sum) 
        qe (scalar): _description_ quantum efficiency of detector
        dit (scalar): _description_ integration time of detector
        ron (int, optional): _description_. Defaults to 1. readout noise in electrons per pixel
        include_shotnoise (bool, optional): _description_. Defaults to True. Sample poisson distribution for each pixel (input intensity is the expected value)
        spectral_bandwidth (_type_, optional): _description_. Defaults to None. if spectral_bandwidth is None than returns photons per pixel per nm of input light,
    """

    i = sum_subarrays(array=i, block_size=binning)

    if spectral_bandwidth is None:
        i *= qe * dit
    else:
        i *= qe * dit * spectral_bandwidth

    if include_shotnoise:
        noisy_intensity = np.random.poisson(lam=i)
    else:  # no noise
        noisy_intensity = i

    if ron > 0:
        noisy_intensity += np.random.normal(0, ron, noisy_intensity.shape).astype(int)

    return noisy_intensity


def get_frame(
    opd_input,
    amp_input,
    opd_internal,
    zwfs_ns,
    detector=None,
    include_shotnoise=True,
    spectral_bandwidth=_AUTO_SPECTRAL_BANDWIDTH,
):
    """_summary_
    propagates the input field with phase described by opd_input and internal aberrations described by opd_internal, field flux described by amp_input
    through a Zernike wavefront sensor system described by the zwfs_ns namespace.

    Args:
        opd_input (2D array like): _description_ opd map in units of meters in wavespace
        amp_input (2D array like): _description_ flux of the input field in wavespace in units of sqrt( photons / pixel / second / nm )
        opd_internal (2D array like): _description_ opd map in units of meters in wavespace for internal aberrations (somewhat redudant with opd_input.. i know)
        zwfs_ns (simple name space): _description_ namespace containing all the information about the zwfs system. Use configuration file and init_zwfs to get this
        detector (_type_, optional): _description_. Defaults to None. if None returns intensity in wave space, otherwise returns intensity on detector.
            Detector should be a class or namespace with dit, qe, ron, binning, attributes. If you want to use the spectral_bandwidth attribute of zwfs_ns.stellar should also be present
        include_shotnoise (bool, optional): _description_. Defaults to True.

    Raises:
        ValueError: _description_

    Returns:
        _type_: field intensity
    """

    # I could do this outside to save time but for now just do it here
    opd_current_dm = zwfs_ns.compiled_dm.eval(
        zwfs_ns.dm.current_cmd * zwfs_ns.dm.opd_per_cmd
    )

    opd_map = opd_input + opd_internal + opd_current_dm

    # convert phase in radians
    phi = zwfs_ns.grid.pupil_mask * 2 * np.pi / zwfs_ns.optics.wvl0 * (opd_map)

    if not hasattr(zwfs_ns.optics, "coldstop_offset"):
        print(
            "zwfs_ns.optics has no coldstop_offset attribute, assigning aligned coldstop ( coldstop_offset=(0,0))"
        )
        zwfs_ns.optics.coldstop_offset = (0, 0)

    # update 1/11/25 got rid of phasemask_mask input since with fft method (faster) we just generate this with the  phasemask_diameter depending on grid ,fplane_pixels and pixels_across_mask.
    Intensity = get_pupil_intensity(
        phi=phi,
        amp=amp_input,
        theta=zwfs_ns.optics.theta,
        phasemask_diameter=zwfs_ns.optics.mask_diam,
        coldstop_diam=zwfs_ns.optics.coldstop_diam,  # new
        coldstop_offset=zwfs_ns.optics.coldstop_offset,  # new
        pupil_diameter=zwfs_ns.grid.N,
        fplane_pixels=zwfs_ns.focal_plane.fplane_pixels,
        pixels_across_mask=zwfs_ns.focal_plane.pixels_across_mask,
        phasemask_mask=None,
    )  # phasemask_mask = None since with fft we dont keep fixed fourier plane sampling and just init phasemask each iteration... its still quicker than mft!

    if detector is not None:
        detector_spectral_bandwidth = _resolve_detector_spectral_bandwidth(
            zwfs_ns,
            spectral_bandwidth=spectral_bandwidth,
        )

        Intensity = detect(
            Intensity,
            binning=(detector.binning, detector.binning),
            qe=detector.qe,
            dit=detector.dit,
            ron=detector.ron,
            include_shotnoise=include_shotnoise,
            spectral_bandwidth=detector_spectral_bandwidth,
        )

    return Intensity


##########################################
##########################################
# Adding new configuration input via json and initialization of Fresnel propagation
##########################################
##########################################


def init_zwfs_from_json(
    json_config,
    derive_fresnel=True,
    derive_spectrum=True,
    instantiate_detector=True,
):
    """
    Initialise a Baldr ZWFS namespace directly from a generic JSON
    configuration file.

    This initializer is intended for generic
    BaldrApp simulations where the optical model, Fresnel relay, spectrum, and
    optional simulator metadata are described by a JSON file.

    Required JSON sections
    ----------------------
    The JSON file must contain:

        grid
        optics
        dm

    These are passed to the existing init_zwfs(grid_ns, optics_ns, dm_ns)
    function.

    Optional derived sections
    -------------------------
    fresnel_relay
        If present and derive_fresnel is True, derived physical relay
        quantities are added, including:

            D_phys
            M_nominal
            s_object_nominal
            s_image_nominal
            z_mirror_to_lens
            s_object
            s_image
            z_focus_to_detector_nominal
            z_focus_to_detector
            M_pupil
            D_detector_predicted

    spectrum
        If present and derive_spectrum is True, this section is passed to
        baldrapp.common.spectrum.derive_spectrum(...). The returned namespace
        is attached as zwfs_ns.spectrum.

        If absent, a backwards-compatible monochromatic spectrum is attached:

            wavelength = optics.wvl0
            weight = 1

    detector
        If present, the raw detector config is attached as
        zwfs_ns.detector_config. If instantiate_detector is True, a BaldrApp
        detector object is also created as zwfs_ns.detector.

    Optional metadata/runtime sections
    ----------------------------------
    The following sections are attached as namespaces if present, but are not
    interpreted by this initializer:

        meta
        source
        stellar
        throughput
        internal_aberrations
        atmosphere
        first_stage_ao
        simulator_runtime

    Design note
    -----------
    This function configures the ZWFS/instrument namespace. It should not
    generate atmospheric screens, scintillation screens, DM commands, or
    detector frames. Those should be handled by higher-level simulation code or
    frame-generation wrappers.

    Example
    -------
    zwfs_ns = init_zwfs_from_json("baldr_config.json")

    fr = zwfs_ns.fresnel_relay
    sp = zwfs_ns.spectrum
    """
    json_config = Path(json_config)
    cfg = cfghelp.load_json_config(json_config)

    for section in ["grid", "optics", "dm"]:
        if section not in cfg:
            raise ValueError(f"JSON config must contain a '{section}' section.")

    # Core sections needed by the existing init_zwfs(...)
    grid_ns = cfghelp.dict_to_namespace_recursive(cfg["grid"])
    optics_ns = cfghelp.dict_to_namespace_recursive(cfg["optics"])
    dm_ns = cfghelp.dict_to_namespace_recursive(cfg["dm"])

    optics_ns = cfghelp.ensure_optics_defaults(optics_ns)

    zwfs_ns = init_zwfs(grid_ns, optics_ns, dm_ns)

    # Keep provenance/debug information.
    zwfs_ns.config_json = cfg
    zwfs_ns.config_path = str(json_config)

    # Attach optional metadata/runtime sections.
    zwfs_ns = cfghelp.attach_optional_config_sections(zwfs_ns, cfg)

    # Optional runtime source profiles.
    # This is metadata for higher-level simulators. It is not interpreted here,
    # so old configs remain fully backwards compatible.
    if "source_profiles" in cfg:
        zwfs_ns.source_profiles = cfghelp.dict_to_namespace_recursive(
            cfg["source_profiles"]
        )

    # Detector config/object.
    # Detector instantiation stays here, not in config_helper.py, because the
    # detector class is defined in baldr_core.py.
    if "detector" in cfg:
        det_cfg = cfghelp.dict_to_namespace_recursive(cfg["detector"])
        zwfs_ns.detector_config = det_cfg

        if instantiate_detector and getattr(det_cfg, "enabled", True):
            zwfs_ns.detector = detector(
                binning=int(getattr(det_cfg, "binning", 1)),
                dit=float(getattr(det_cfg, "dit", 1.0)),
                ron=float(getattr(det_cfg, "ron", 0.0)),
                qe=float(getattr(det_cfg, "qe", 1.0)),
            )

    # # Spectrum config.
    # # Always attach zwfs_ns.spectrum when derive_spectrum=True, even if the
    # # JSON has no spectrum section. This keeps downstream wrappers simple and
    # # backwards-compatible.
    # if derive_spectrum:
    #     if "spectrum" in cfg:
    #         spectrum_ns = spec.derive_spectrum(
    #             cfg["spectrum"],
    #             default_wvl0=zwfs_ns.optics.wvl0,
    #         )
    #     else:
    #         spectrum_ns = spec.default_monochromatic_spectrum(
    #             zwfs_ns.optics.wvl0
    #         )

    #     spectrum_ns = cfghelp.complete_spectrum_integration_fields(
    #         spectrum_ns
    #     )
    #     zwfs_ns.spectrum = spectrum_ns

    # Stellar / spectrum config.
    # Preferred user config is now cfg["stellar"]["spectrum"].
    # Backwards-compatible fallback is top-level cfg["spectrum"].
    spectrum_cfg = cfghelp.get_spectrum_config_from_cfg(cfg)

    stellar_ns = getattr(zwfs_ns, "stellar", None)
    stellar_ns, bandwidth_source = cfghelp.ensure_stellar_bandwidth(
        stellar_ns,
        spectrum_cfg=spectrum_cfg,
        default_nm=1.0,
        verbose=True,
    )
    zwfs_ns.stellar = stellar_ns

    cfghelp.check_stellar_bandwidth_consistency(
        zwfs_ns.stellar,
        spectrum_cfg=spectrum_cfg,
        relative_tolerance=1e-3,
        verbose=True,
    )

    if derive_spectrum:
        if spectrum_cfg is not None:
            spectrum_ns = spec.derive_spectrum(
                spectrum_cfg,
                default_wvl0=zwfs_ns.optics.wvl0,
            )
        else:
            spectrum_ns = spec.default_monochromatic_spectrum(zwfs_ns.optics.wvl0)

        spectrum_ns = cfghelp.complete_spectrum_integration_fields(spectrum_ns)

        spectrum_ns = cfghelp.apply_stellar_bandwidth_to_spectrum(
            spectrum_ns,
            stellar_ns=zwfs_ns.stellar,
            warn=True,
        )

        zwfs_ns.spectrum = spectrum_ns

    # Fresnel relay config.
    if "fresnel_relay" in cfg:
        fresnel_ns = cfghelp.dict_to_namespace_recursive(cfg["fresnel_relay"])

        if derive_fresnel:
            fresnel_ns = cfghelp.derive_fresnel_relay_ns(fresnel_ns)

        zwfs_ns.fresnel_relay = fresnel_ns

    return zwfs_ns


def get_zwfs_output_field_from_opd(
    opd,
    amp,
    wavelength,
    zwfs_ns,
    theta=None,
    phasemask_diameter=None,
    phasemask_mask=None,
    return_terms=True,
):
    """
    Convenience wrapper around get_zwfs_output_field(...) that accepts OPD in
    metres instead of phase in radians.

    This is useful for chromatic propagation, where the same OPD map must be
    converted to phase separately at each wavelength.

    Parameters
    ----------
    opd : ndarray
        Optical path difference in metres, on the BaldrApp wave-space grid.

    amp : ndarray
        Input field amplitude. If amp**2 has units photons / s / pixel / nm,
        then the output intensity from this function has the same spectral
        density convention.

    wavelength : float
        Wavelength in metres.

    zwfs_ns : SimpleNamespace
        Baldr ZWFS namespace.

    theta : float or None
        ZWFS phase-mask phase shift in radians. If None, use
        zwfs_ns.optics.theta.

    phasemask_diameter : float or None
        Phase-mask diameter in the convention expected by
        get_zwfs_output_field(...). If None, use zwfs_ns.optics.mask_diam.

    phasemask_mask : ndarray or None
        Optional precomputed phase-mask support mask.

    return_terms : bool
        Passed through to get_zwfs_output_field(...).

    Returns
    -------
    output : dict or ndarray
        Output from get_zwfs_output_field(...).
    """
    pupil = zwfs_ns.grid.pupil_mask

    if theta is None:
        theta = zwfs_ns.optics.theta

    if phasemask_diameter is None:
        phasemask_diameter = zwfs_ns.optics.mask_diam

    phi = pupil * 2.0 * np.pi / wavelength * opd

    return get_zwfs_output_field(
        phi=phi,
        amp=amp,
        theta=theta,
        phasemask_diameter=phasemask_diameter,
        phasemask_mask=phasemask_mask,
        pupil_diameter=zwfs_ns.grid.N,
        fplane_pixels=zwfs_ns.focal_plane.fplane_pixels,
        pixels_across_mask=zwfs_ns.focal_plane.pixels_across_mask,
        return_terms=return_terms,
    )


####################################
# get frame wrappers for polychromatic and fresnel versions
def get_frame_polychromatic(
    opd_input,
    amp_input,
    opd_internal,
    zwfs_ns,
    detector=None,
    include_shotnoise=True,
    return_intermediates=False,
):
    """
    Generate a polychromatic Baldr ZWFS frame using the existing non-Fresnel
    propagation path, i.e. get_frame(...).

    This function loops over zwfs_ns.spectrum, computes one monochromatic
    spectral-density image per wavelength using get_frame(..., detector=None),
    integrates those images over wavelength using weights_nm, and then applies
    the detector once at the end if detector is provided.

    Unit convention
    ---------------
    This function assumes that amp_input**2 has units:

        photons / second / wave-space-pixel / nm

    Therefore each monochromatic image returned by get_frame(..., detector=None)
    has units:

        photons / second / wave-space-pixel / nm

    Spectral integration is performed as:

        I_rate = sum_k weights_nm_k * I_density_k

    where weights_nm_k has units of nm. The integrated image therefore has
    units:

        photons / second / wave-space-pixel

    If detector is provided, detector.detect(...) is called with:

        spectral_bandwidth=None

    because the spectral integration has already been done explicitly.

    Notes
    -----
    This function temporarily updates:

        zwfs_ns.optics.wvl0
        zwfs_ns.optics.theta
        zwfs_ns.optics.mask_diam

    inside the wavelength loop, then restores their original values.

    """
    # Backwards-compatible fallback if no spectrum namespace exists.
    if not hasattr(zwfs_ns, "spectrum"):
        wavelengths = np.array([zwfs_ns.optics.wvl0], dtype=float)
        weights_nm = np.array([1.0], dtype=float)
    else:
        wavelengths = np.asarray(zwfs_ns.spectrum.wavelengths, dtype=float)

        if hasattr(zwfs_ns.spectrum, "weights_nm"):
            weights_nm = np.asarray(zwfs_ns.spectrum.weights_nm, dtype=float)
        else:
            if hasattr(zwfs_ns.spectrum, "weights_normalized"):
                weights_normalized = np.asarray(
                    zwfs_ns.spectrum.weights_normalized,
                    dtype=float,
                )
            else:
                weights_normalized = np.ones_like(wavelengths, dtype=float) / len(
                    wavelengths
                )

            bandwidth_nm = float(getattr(zwfs_ns.spectrum, "bandwidth_nm", 1.0))

            weights_nm = bandwidth_nm * weights_normalized

    if wavelengths.ndim != 1:
        raise ValueError("zwfs_ns.spectrum.wavelengths must be one-dimensional.")

    if weights_nm.ndim != 1:
        raise ValueError("zwfs_ns.spectrum.weights_nm must be one-dimensional.")

    if wavelengths.shape != weights_nm.shape:
        raise ValueError(
            "zwfs_ns.spectrum.wavelengths and weights_nm must have the same shape."
        )

    if len(wavelengths) < 1:
        raise ValueError("Spectrum must contain at least one wavelength sample.")

    if np.any(wavelengths <= 0):
        raise ValueError("All spectrum wavelengths must be positive.")

    if np.any(weights_nm < 0):
        raise ValueError("All spectrum weights_nm must be non-negative.")

    if np.sum(weights_nm) <= 0:
        raise ValueError("Spectrum weights_nm must have positive sum.")

    # Save original optical settings so we can restore them after the loop.
    original_wvl0 = zwfs_ns.optics.wvl0
    original_theta = zwfs_ns.optics.theta
    original_mask_diam = zwfs_ns.optics.mask_diam

    I_rate = None
    per_wavelength_outputs = []

    try:
        for wavelength, weight_nm in zip(wavelengths, weights_nm):

            theta = spec.theta_at_wavelength(
                zwfs_ns.optics,
                wavelength,
            )

            phasemask_diameter = spec.phasemask_diameter_at_wavelength(
                zwfs_ns.optics,
                wavelength_m=wavelength,
                default_wvl0=original_wvl0,
            )

            # Temporarily update the namespace so get_frame(...) uses this
            # wavelength and phase-mask setting internally.
            zwfs_ns.optics.wvl0 = float(wavelength)
            zwfs_ns.optics.theta = float(theta)
            zwfs_ns.optics.mask_diam = float(phasemask_diameter)

            I_density = get_frame(
                opd_input=opd_input,
                amp_input=amp_input,
                opd_internal=opd_internal,
                zwfs_ns=zwfs_ns,
                detector=None,
                include_shotnoise=False,
            )

            if I_rate is None:
                I_rate = float(weight_nm) * I_density
            else:
                I_rate = I_rate + float(weight_nm) * I_density

            if return_intermediates:
                per_wavelength_outputs.append(
                    {
                        "wavelength": float(wavelength),
                        "weight_nm": float(weight_nm),
                        "theta": float(theta),
                        "phasemask_diameter": float(phasemask_diameter),
                        "intensity_density": I_density,
                    }
                )

    finally:
        # Always restore original settings, even if an error occurs.
        zwfs_ns.optics.wvl0 = original_wvl0
        zwfs_ns.optics.theta = original_theta
        zwfs_ns.optics.mask_diam = original_mask_diam

    # if detector is not None:
    #     intensity = detector.detect(
    #         I_rate,
    #         include_shotnoise=include_shotnoise,
    #         spectral_bandwidth=None,
    #     )
    # else:
    #     intensity = I_rate
    if detector is not None:
        intensity = detect(
            I_rate,
            binning=(detector.binning, detector.binning),
            qe=detector.qe,
            dit=detector.dit,
            ron=detector.ron,
            include_shotnoise=include_shotnoise,
            spectral_bandwidth=None,
        )
    else:
        intensity = I_rate

    if not return_intermediates:
        return intensity

    return {
        "intensity": intensity,
        "intensity_pre_detector": I_rate,
        "wavelengths": wavelengths,
        "weights_nm": weights_nm,
        "weights_nm_sum": float(np.sum(weights_nm)),
        "per_wavelength_outputs": per_wavelength_outputs,
    }


def get_frame_fresnel(
    opd_input,
    amp_input,
    opd_internal,
    zwfs_ns,
    detector=None,
    include_shotnoise=True,
    spectral_bandwidth=_AUTO_SPECTRAL_BANDWIDTH,
    wavelength=None,
    theta=None,
    phasemask_diameter=None,
    return_intermediates=False,
):
    """
    Generate a monochromatic detector-plane Baldr frame using the analytic ZWFS
    output field followed by the explicit Fresnel relay.

    This function does not perform spectral integration. For polychromatic
    propagation, call a separate polychromatic wrapper that loops over
    zwfs_ns.spectrum and integrates the monochromatic spectral-density images.

    Unit convention
    ---------------
    The pre-detector intensity has the same photon-rate convention as
    amp_input**2. For example, if amp_input**2 is in:

        photons / second / wave-space-pixel / nm

    then intensity_pre_detector is a monochromatic spectral-density image.
    """
    if not hasattr(zwfs_ns, "fresnel_relay"):
        raise ValueError(
            "zwfs_ns has no fresnel_relay attribute. "
            "Initialise from a config with a fresnel_relay section."
        )

    fr = zwfs_ns.fresnel_relay

    if wavelength is None:
        wavelength = zwfs_ns.optics.wvl0

    # pupil = zwfs_ns.grid.pupil_mask
    # opd_total = pupil * (opd_input + opd_internal)

    pupil = zwfs_ns.grid.pupil_mask

    opd_current_dm = zwfs_ns.compiled_dm.eval(
        zwfs_ns.dm.current_cmd * zwfs_ns.dm.opd_per_cmd
    )

    opd_total = pupil * (opd_input + opd_internal + opd_current_dm)

    if theta is None:
        theta = spec.theta_at_wavelength(zwfs_ns.optics, wavelength)

    if phasemask_diameter is None:
        phasemask_diameter = spec.phasemask_diameter_at_wavelength(
            zwfs_ns.optics,
            wavelength_m=wavelength,
            default_wvl0=zwfs_ns.optics.wvl0,
        )

    zwfs_terms = get_zwfs_output_field_from_opd(
        opd=opd_total,
        amp=amp_input,
        wavelength=wavelength,
        zwfs_ns=zwfs_ns,
        theta=theta,
        phasemask_diameter=phasemask_diameter,
        phasemask_mask=None,
        return_terms=True,
    )

    psi_C = np.ascontiguousarray(zwfs_terms["psi_C"])

    dx_phys = fr.D_phys / zwfs_ns.grid.N

    X_phys, Y_phys = fresnel.make_coordinate_grid(
        psi_C.shape,
        dx_phys,
    )

    # 1. Propagate from post-collimator ZWFS pupil plane to D/knife-edge mirror.
    zwfs_ns.propagator.ang_spec(
        psi_C,
        fr.z_to_mirror,
        wavelength,
        (dx_phys * zwfs_ns.grid.dim, dx_phys * zwfs_ns.grid.dim),
    )
    psi_mirror = psi_C

    mirror_mask_full = fresnel.circular_aperture(
        psi_mirror.shape,
        dx=dx_phys,
        radius=fr.D_mirror / 2.0,
    ).astype(float)

    Xr = X_phys * np.cos(fr.edge_angle) + Y_phys * np.sin(fr.edge_angle)

    mirror_mask_D = (Xr >= fr.edge_offset).astype(float) * mirror_mask_full

    psi_after_mirror = psi_mirror * mirror_mask_D

    # 2. Propagate mirror to re-imaging lens.
    zwfs_ns.propagator.ang_spec(
        psi_after_mirror,
        fr.z_mirror_to_lens,
        wavelength,
        (dx_phys * zwfs_ns.grid.dim, dx_phys * zwfs_ns.grid.dim),
    )
    field_at_lens = psi_after_mirror

    # 3. Apply re-imaging lens.
    field_after_lens = fresnel.apply_thin_lens(
        field_at_lens,
        wavelength=wavelength,
        dx=dx_phys,
        focal_length=fr.f_imaging,
    )

    # 4. Propagate lens to cold-stop / star-image plane.
    field_coldstop_plane, dx_cold, dy_cold = fresnel.fresnel_one_step_propagate(
        field_after_lens,
        wavelength=wavelength,
        dx=dx_phys,
        z=fr.f_imaging,
        include_global_phase=False,
    )

    X_cold, Y_cold = fresnel.make_coordinate_grid(
        field_coldstop_plane.shape,
        dx=dx_cold,
        dy=dy_cold,
    )

    R_cold = np.hypot(
        X_cold - fr.coldstop_x_offset,
        Y_cold - fr.coldstop_y_offset,
    )

    coldstop_mask_phys = (R_cold <= fr.D_coldstop / 2.0).astype(float)

    field_after_coldstop = field_coldstop_plane * coldstop_mask_phys

    # 5. Propagate cold-stop / star-image plane to detector pupil plane.
    field_detector_pupil, dx_detector, dy_detector = fresnel.fresnel_one_step_propagate(
        field_after_coldstop,
        wavelength=wavelength,
        dx=dx_cold,
        dy=dy_cold,
        z=fr.z_focus_to_detector,
        include_global_phase=False,
    )

    intensity_pre_detector = np.abs(field_detector_pupil) ** 2

    # Convert Fresnel detector-plane irradiance samples back to the historical
    # BaldrApp photon-rate-per-wave-space-pixel convention used by detect(...).
    #
    # Fresnel propagation changes the physical pixel scale from dx_phys to
    # dx_detector, dy_detector. Raw array sums therefore do not represent the same
    # physical photon rate unless pixel area is accounted for.
    fresnel_pixel_area_scale = (dx_detector * dy_detector) / (dx_phys**2)
    intensity_pre_detector = intensity_pre_detector * fresnel_pixel_area_scale

    if detector is not None:
        detector_spectral_bandwidth = _resolve_detector_spectral_bandwidth(
            zwfs_ns,
            spectral_bandwidth=spectral_bandwidth,
        )

        intensity = detect(
            intensity_pre_detector,
            binning=(detector.binning, detector.binning),
            qe=detector.qe,
            dit=detector.dit,
            ron=detector.ron,
            include_shotnoise=include_shotnoise,
            spectral_bandwidth=detector_spectral_bandwidth,
        )

    else:
        intensity = intensity_pre_detector

    if not return_intermediates:
        return intensity

    return {
        "intensity": intensity,
        "intensity_pre_detector": intensity_pre_detector,
        "zwfs_terms": zwfs_terms,
        "psi_C": psi_C,
        "psi_mirror": psi_mirror,
        "mirror_mask_D": mirror_mask_D,
        "field_at_lens": field_at_lens,
        "field_after_lens": field_after_lens,
        "field_coldstop_plane": field_coldstop_plane,
        "coldstop_mask_phys": coldstop_mask_phys,
        "field_after_coldstop": field_after_coldstop,
        "field_detector_pupil": field_detector_pupil,
        "dx_phys": dx_phys,
        "dx_cold": dx_cold,
        "dy_cold": dy_cold,
        "dx_detector": dx_detector,
        "dy_detector": dy_detector,
        "wavelength": wavelength,
        "theta": theta,
        "phasemask_diameter": phasemask_diameter,
    }


def get_frame_fresnel_polychromatic(
    opd_input,
    amp_input,
    opd_internal,
    zwfs_ns,
    detector=None,
    include_shotnoise=True,
    return_intermediates=False,
):
    """
    Generate a polychromatic detector-plane Baldr frame using the analytic ZWFS
    output field followed by the explicit Fresnel relay.

    This function loops over zwfs_ns.spectrum, computes one monochromatic
    Fresnel relay frame per wavelength, and integrates the resulting
    spectral-density images over wavelength before detector detection.

    Unit convention
    ---------------
    This function assumes that amp_input**2 has units:

        photons / second / wave-space-pixel / nm

    Therefore each monochromatic image returned by get_frame_fresnel(...) has
    units:

        photons / second / detector-plane-pixel / nm

    Spectral integration is then performed as:

        I_rate = sum_k weights_nm_k * I_density_k

    where weights_nm_k has units of nm.

    The integrated pre-detector image therefore has units:

        photons / second / detector-plane-pixel

    If detector is provided, detector.detect(...) is called with:

        spectral_bandwidth=None

    because the spectral bandwidth has already been integrated explicitly.

    Parameters
    ----------
    opd_input : ndarray
        Input OPD in metres on the BaldrApp wave-space grid.

    amp_input : ndarray
        Input field amplitude. amp_input**2 is assumed to be a photon spectral
        density image in photons / second / wave-space-pixel / nm.

    opd_internal : ndarray
        Static/internal OPD in metres on the BaldrApp wave-space grid.

    zwfs_ns : SimpleNamespace
        Baldr ZWFS namespace. Must contain fresnel_relay. Should contain
        spectrum. If no spectrum is present, a single-wavelength 1 nm
        monochromatic spectrum is used.

    detector : detector or None
        Optional BaldrApp detector object. If provided, the spectrally
        integrated photon-rate image is passed through detector.detect(...).

    include_shotnoise : bool
        Passed to detector.detect(...) if detector is provided.

    return_intermediates : bool
        If False, return only the final image.
        If True, return a dictionary containing the final image, pre-detector
        integrated image, wavelength grid, integration weights, and optionally
        per-wavelength intermediate outputs.

    Returns
    -------
    intensity : ndarray
        If return_intermediates is False, returns the final image. If detector
        is None, this is the spectrally integrated pre-detector photon-rate
        image. If detector is not None, this is the detected image.

    output : dict
        If return_intermediates is True.
    """
    if not hasattr(zwfs_ns, "fresnel_relay"):
        raise ValueError(
            "zwfs_ns has no fresnel_relay attribute. "
            "Initialise from a config with a fresnel_relay section."
        )

    # Backwards-compatible fallback if zwfs_ns was not initialised from the
    # new JSON/config path.
    if not hasattr(zwfs_ns, "spectrum"):
        wavelengths = np.array([zwfs_ns.optics.wvl0], dtype=float)
        weights_nm = np.array([1.0], dtype=float)
    else:
        wavelengths = np.asarray(zwfs_ns.spectrum.wavelengths, dtype=float)

        if hasattr(zwfs_ns.spectrum, "weights_nm"):
            weights_nm = np.asarray(zwfs_ns.spectrum.weights_nm, dtype=float)
        else:
            # Robust fallback. Prefer explicit weights_nm from config_helper,
            # but allow older derived spectrum namespaces to work.
            if hasattr(zwfs_ns.spectrum, "weights_normalized"):
                weights_normalized = np.asarray(
                    zwfs_ns.spectrum.weights_normalized,
                    dtype=float,
                )
            else:
                weights_normalized = np.ones_like(wavelengths, dtype=float) / len(
                    wavelengths
                )

            bandwidth_nm = float(getattr(zwfs_ns.spectrum, "bandwidth_nm", 1.0))

            weights_nm = bandwidth_nm * weights_normalized

    if wavelengths.ndim != 1:
        raise ValueError("zwfs_ns.spectrum.wavelengths must be one-dimensional.")

    if weights_nm.ndim != 1:
        raise ValueError("zwfs_ns.spectrum.weights_nm must be one-dimensional.")

    if wavelengths.shape != weights_nm.shape:
        raise ValueError(
            "zwfs_ns.spectrum.wavelengths and weights_nm must have the same shape."
        )

    if len(wavelengths) < 1:
        raise ValueError("Spectrum must contain at least one wavelength sample.")

    if np.any(wavelengths <= 0):
        raise ValueError("All spectrum wavelengths must be positive.")

    if np.any(weights_nm < 0):
        raise ValueError("All spectrum weights_nm must be non-negative.")

    if np.sum(weights_nm) <= 0:
        raise ValueError("Spectrum weights_nm must have positive sum.")

    I_rate = None
    per_wavelength_outputs = []

    for wavelength, weight_nm in zip(wavelengths, weights_nm):

        theta = spec.theta_at_wavelength(
            zwfs_ns.optics,
            wavelength,
        )

        phasemask_diameter = spec.phasemask_diameter_at_wavelength(
            zwfs_ns.optics,
            wavelength_m=wavelength,
            default_wvl0=zwfs_ns.optics.wvl0,
        )

        out = get_frame_fresnel(
            opd_input=opd_input,
            amp_input=amp_input,
            opd_internal=opd_internal,
            zwfs_ns=zwfs_ns,
            detector=None,
            include_shotnoise=False,
            spectral_bandwidth=None,
            wavelength=float(wavelength),
            theta=theta,
            phasemask_diameter=phasemask_diameter,
            return_intermediates=return_intermediates,
        )

        if return_intermediates:
            I_density = out["intensity_pre_detector"]
            out["wavelength"] = float(wavelength)
            out["weight_nm"] = float(weight_nm)
            out["theta"] = float(theta)
            out["phasemask_diameter"] = float(phasemask_diameter)
            per_wavelength_outputs.append(out)
        else:
            I_density = out

        if I_rate is None:
            I_rate = float(weight_nm) * I_density
        else:
            I_rate = I_rate + float(weight_nm) * I_density

    # if detector is not None:
    #     intensity = detector.detect(
    #         I_rate,
    #         include_shotnoise=include_shotnoise,
    #         spectral_bandwidth=None,
    #     )
    # else:
    #     intensity = I_rate
    if detector is not None:
        intensity = detect(
            I_rate,
            binning=(detector.binning, detector.binning),
            qe=detector.qe,
            dit=detector.dit,
            ron=detector.ron,
            include_shotnoise=include_shotnoise,
            spectral_bandwidth=None,
        )
    else:
        intensity = I_rate

    if not return_intermediates:
        return intensity

    return {
        "intensity": intensity,
        "intensity_pre_detector": I_rate,
        "wavelengths": wavelengths,
        "weights_nm": weights_nm,
        "weights_nm_sum": float(np.sum(weights_nm)),
        "per_wavelength_outputs": per_wavelength_outputs,
    }


def get_frame_configured(
    opd_input,
    amp_input,
    opd_internal,
    zwfs_ns,
    detector=None,
    include_shotnoise=True,
    spectral_bandwidth=_AUTO_SPECTRAL_BANDWIDTH,
    force_fresnel=None,
    force_polychromatic=None,
    return_intermediates=False,
):
    """
    Generate a Baldr frame using the propagation mode configured in zwfs_ns.

    This is a convenience dispatcher around the explicit frame-generation
    functions:

        get_frame(...)
        get_frame_polychromatic(...)
        get_frame_fresnel(...)
        get_frame_fresnel_polychromatic(...)

    It does not implement new propagation physics. It only chooses the
    appropriate already-tested path based on the configuration.

    Routing logic
    -------------
    Fresnel propagation is enabled if:

        zwfs_ns.fresnel_relay.enabled is True

    unless overridden by force_fresnel.

    Polychromatic propagation is enabled if:

        zwfs_ns.spectrum.enabled is True
        and len(zwfs_ns.spectrum.wavelengths) > 1

    unless overridden by force_polychromatic.

    Unit convention
    ---------------
    For monochromatic legacy paths, if spectral_bandwidth is not None, the
    detector interprets the input intensity as:

        photons / second / pixel / nm

    and multiplies by spectral_bandwidth.

    For polychromatic paths, the spectrum is integrated before detector
    detection using weights_nm, and the detector is called with:

        spectral_bandwidth=None

    to avoid double-counting bandwidth.

    Parameters
    ----------
    opd_input : ndarray
        Input OPD in metres.

    amp_input : ndarray
        Input field amplitude.

    opd_internal : ndarray
        Internal/static OPD in metres.

    zwfs_ns : SimpleNamespace
        Configured Baldr ZWFS namespace.

    detector : detector or None
        Optional detector object. If None and zwfs_ns.detector exists, this
        function uses zwfs_ns.detector.

    include_shotnoise : bool
        Whether to include shot noise if a detector is used.

    spectral_bandwidth :  float, None, or "auto"
        Used only for monochromatic detector paths. For polychromatic paths,
        this is ignored and detector.detect(...) is called with
        spectral_bandwidth=None.

    force_fresnel : bool or None
        If None, use zwfs_ns.fresnel_relay.enabled. If True/False, override.

    force_polychromatic : bool or None
        If None, use zwfs_ns.spectrum.enabled and the number of wavelengths.
        If True/False, override.

    return_intermediates : bool
        Only supported by the Fresnel and polychromatic wrapper paths. For the
        plain get_frame(...) path, this must be False.

    Returns
    -------
    frame : ndarray or dict
        Frame output from the selected propagation path.
    """
    if detector is None and hasattr(zwfs_ns, "detector"):
        detector = zwfs_ns.detector

    # Decide Fresnel mode.
    if force_fresnel is None:
        use_fresnel = hasattr(zwfs_ns, "fresnel_relay") and getattr(
            zwfs_ns.fresnel_relay, "enabled", False
        )
    else:
        use_fresnel = bool(force_fresnel)

    # Decide spectral mode.
    if force_polychromatic is None:
        if hasattr(zwfs_ns, "spectrum"):
            spectrum_enabled = getattr(zwfs_ns.spectrum, "enabled", True)
            wavelengths = np.asarray(zwfs_ns.spectrum.wavelengths, dtype=float)
            use_polychromatic = bool(spectrum_enabled) and len(wavelengths) > 1
        else:
            use_polychromatic = False
    else:
        use_polychromatic = bool(force_polychromatic)

    # ------------------------------------------------------------
    # Fresnel + polychromatic
    # ------------------------------------------------------------
    if use_fresnel and use_polychromatic:
        return get_frame_fresnel_polychromatic(
            opd_input=opd_input,
            amp_input=amp_input,
            opd_internal=opd_internal,
            zwfs_ns=zwfs_ns,
            detector=detector,
            include_shotnoise=include_shotnoise,
            return_intermediates=return_intermediates,
        )

    # ------------------------------------------------------------
    # Fresnel + monochromatic
    # ------------------------------------------------------------
    if use_fresnel and not use_polychromatic:
        return get_frame_fresnel(
            opd_input=opd_input,
            amp_input=amp_input,
            opd_internal=opd_internal,
            zwfs_ns=zwfs_ns,
            detector=detector,
            include_shotnoise=include_shotnoise,
            spectral_bandwidth=spectral_bandwidth,
            wavelength=zwfs_ns.optics.wvl0,
            return_intermediates=return_intermediates,
        )

    # ------------------------------------------------------------
    # Non-Fresnel + polychromatic
    # ------------------------------------------------------------
    if (not use_fresnel) and use_polychromatic:
        return get_frame_polychromatic(
            opd_input=opd_input,
            amp_input=amp_input,
            opd_internal=opd_internal,
            zwfs_ns=zwfs_ns,
            detector=detector,
            include_shotnoise=include_shotnoise,
            return_intermediates=return_intermediates,
        )

    # ------------------------------------------------------------
    # Non-Fresnel + monochromatic legacy path
    # ------------------------------------------------------------
    if return_intermediates:
        raise ValueError(
            "return_intermediates=True is not supported for the plain "
            "non-Fresnel monochromatic get_frame(...) path."
        )

    return get_frame(
        opd_input=opd_input,
        amp_input=amp_input,
        opd_internal=opd_internal,
        zwfs_ns=zwfs_ns,
        detector=detector,
        include_shotnoise=include_shotnoise,
        spectral_bandwidth=spectral_bandwidth,
    )


class Actuator(ABC):
    @abstractmethod
    def sample(self, x: float, y: float) -> float:
        pass

    @abstractmethod
    def sample_vec(self, xx: NDArray, yy: NDArray) -> NDArray:
        pass

    def evaluate(self, com: float, xx: NDArray, yy: NDArray) -> NDArray:
        zz = com * self.sample_vec(xx.flatten(), yy.flatten())
        return zz.reshape(xx.shape)


class DM(ABC):
    @abstractmethod
    def actuator(self, idx: int) -> Actuator:
        pass

    @property
    @abstractmethod
    def nact(self) -> int:
        pass

    def sample(self, command: NDArray, x: float, y: float) -> float:
        z: float = 0.0
        for i in range(self.nact):
            z += command[i] * self.actuator(i).sample(x, y)
        return z

    def evaluate(self, command: NDArray, xx: NDArray, yy: NDArray) -> NDArray:
        zz = (xx * 0.0).flatten()
        for i, (x, y) in enumerate(zip(xx.ravel(), yy.ravel())):
            zz[i] += self.sample(command, x, y)
        return zz.reshape(xx.shape)


class GaussianActuator(Actuator):
    sigma: float
    x0: float
    y0: float

    def __init__(self, sigma: float, x0: float, y0: float):
        self.sigma = sigma
        self.x0 = x0
        self.y0 = y0

    def sample(self, x: float, y: float) -> float:
        return np.exp(-((x - self.x0) ** 2 + (y - self.y0) ** 2) / self.sigma**2)

    def sample_vec(
        self,
        xx: NDArray,
        yy: NDArray,
    ) -> NDArray:
        return np.exp(-((xx - self.x0) ** 2 + (yy - self.y0) ** 2) / self.sigma**2)


class GaussianDM(DM):
    _nact: int
    actuators: List[GaussianActuator]

    def __init__(self, x0: NDArray, y0: NDArray, sigma: NDArray):
        self._nact = x0.shape[0]
        actuators: List[GaussianActuator] = []
        for x, y, sig in zip(x0, y0, sigma):
            actuators.append(GaussianActuator(sig, x, y))
        self.actuators = actuators

    @property
    def nact(self) -> int:
        return self._nact

    def actuator(self, idx: int) -> Actuator:
        return self.actuators[idx]


class CompiledDM(GaussianDM):
    """This object constructs the DM based on static parameters, and allows
    efficient evaluation (once initialised)"""

    influ: NDArray  # array of influence functions (nact, ncoordx, ncoordy)

    def __init__(self, xx: NDArray, yy: NDArray, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)
        influ = []
        for c in range(self.nact):
            influ.append(self.actuator(c).evaluate(1.0, xx, yy))
        # truncate to f32 for speed, speeds up this calc by about 40% on JC laptop
        self.influ = np.array(influ).astype(np.float32)

    def eval(self, com: NDArray) -> NDArray:
        return np.einsum("i...,i->...", self.influ, com)
