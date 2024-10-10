
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import importlib # reimport package after edits: importlib.reload(bldr)

# from courtney-barrer's fork of pyzelda
import pyzelda.zelda as zelda
import pyzelda.ztools as ztools
import pyzelda.utils.aperture as aperture

from common import phasescreens as ps
from common import utilities as util
from common import baldr_core as bldr

# create phase screen 

# function to remove N modes from the phase screen

# create vibration spectrum 

# create time series of the vibration spectrum

# for a non-aberrated system 
#   calculate the non-observable parameters 
#       - optical gain (b)
#   simulate the observable parameters
#       - ZWFS signal with and without phasemask
#   find a way to estimate b0. Problem is that we don't know the internal aberrations.
#   The general model is:
#       I0 - N0 = abs(psi_A)**2 + abs(psi_R)**2 + 2 * abs(psi_A) * abs(psi_R) * np.cos(  mu ) - abs(psi_A)**2    
#   Problem statement:
#   We have measurements of two images (intensities in a NxM camera defined by x,y pixel coordinates)  I0(x,y), N0(x,y). 
#   I0(x,y) is the intensity with a phase shifting phase mask inserted at focus and N0(x,y) is the intensity without 
#   the phasemask which is essentailly a uniformly illuminated pupil. The difference  of these two quantities follows the model 
#       delta(x,y)  = I0(x,y) - N0(x,y) 
#                   = abs(psi_R(x,y)) * ( abs(psi_R(x,y)) + 2 * sqrt(N0(x,y)) * np.cos( phi0(x,y) - mu ) ) 
#   We know mu (which has no spatial dependence) by design but need to estimate phi0(x,y) and psiR(x,y). 
#   A reasonable prior for phi0(x,y) is a zero mean Guassian with some variance sigma^2_phi. 
#   We also know N0(x,y) is zero outside the pupil (i.e. x,y not in P) - which allows direct sampling of  
#       delta(x,y not in P) =  abs(psi_R(x,y not in P))^2
#   We also have a theoretical model of psi_R(x,y) across all regions that 
#       psi_R(x,y) = A * iFFT( Pi(x,y) * FFT( N0 * exp(1j phi0 (x,y) ) ) where iFFT and FFT are the Fourier transforms, 
#   and A is a known (by construction) parameter and Pi(x,y) defines the phaseshifting region on the mask (0 if no phase shift applie, 1 otherwise).
 

#   The problem is then to estimate phi0(x,y) and psi_R(x,y) from the measurements I0(x,y) and N0(x,y).
#    open question 
#       - fit psi_R first from samples outside pupil and then use this to estimate phi0. 
#     Then repeat using full function of phi0 est, to get psi_R. Is this convex or non-convex? i.e. will it always converge to global maxima?
#       - Assume phi0 = 0 and then solve abs(psi_R) pixelwise. Selecting smoothest solution. 
#          Then use psi_R as normal with some uncertainty and update solve phi0 (bayesian).

# the general pixelwise model:
#           y = a*X^2 + b * X * cos( phi + a )


#%%
#Define the function f(X,ϕ)=a⋅X2+b⋅X⋅cos⁡(ϕ+c) .
# For a fixed value of y, plot the contour where f(X,ϕ)=y


# Given constants
a = 1.0   # Fixed a value

# Define the function f(|psi_R|, phi)
def f(psi_R, phi, psi_A, mu):
    b = 2 * np.sqrt(psi_A)  # Compute b from |psi_A|
    return a * psi_R**2 + b * psi_R * np.cos(phi + mu)

# Define the function for y = |\psi_C|^2 - |\psi_A|^2
def psi_C_squared(psi_R, phi, psi_A, mu):
    return f(psi_R, phi, psi_A, mu) + psi_A**2  # y is relabeled as |\psi_C|^2 - |\psi_A|^2

# Define the range of |psi_R| (X) and phi values
psi_R_vals = np.linspace(-10, 10, 400)  # range of |psi_R| (X) values
phi_vals = np.linspace(-np.pi, np.pi, 400)  # range of phi values

# Create meshgrid for |psi_R| and phi
psi_R, phi = np.meshgrid(psi_R_vals, phi_vals)

# Initial parameters
psi_A_init = 2.0  # Initial |psi_A|
mu_init = 0.5     # Initial mu value
psi_C_init = 10.0  # Initial value for |psi_C|

# Create the figure and the contour plot
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(left=0.1, bottom=0.35)  # Make room for sliders
f_vals = psi_C_squared(psi_R, phi, psi_A_init, mu_init)
contour = ax.contour(psi_R, phi, f_vals, levels=[psi_C_init], colors='r')
ax.set_title(r'Phase Space Contour for $|\psi_C|^2 - |\psi_A|^2 = {}$'.format(psi_C_init))
ax.set_xlabel(r'$|\psi_R|$')
ax.set_ylabel(r'$\phi$')
ax.grid(True)

# Adjust the position of the sliders
ax_slider_psi_A = plt.axes([0.1, 0.20, 0.8, 0.03], facecolor='lightgoldenrodyellow')
ax_slider_mu = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor='lightgoldenrodyellow')
ax_slider_psi_C = plt.axes([0.1, 0.10, 0.8, 0.03], facecolor='lightgoldenrodyellow')

# Define the sliders
slider_psi_A = Slider(ax_slider_psi_A, r'$|\psi_A|$', 0.1, 5.0, valinit=psi_A_init)
slider_mu = Slider(ax_slider_mu, r'$\mu$', -np.pi, np.pi, valinit=mu_init)
slider_psi_C = Slider(ax_slider_psi_C, r'$|\psi_C|^2 - |\psi_A|^2$', 0.1, 20.0, valinit=psi_C_init)

# Update function to redraw the plot when sliders change
def update(val):
    psi_A = slider_psi_A.val
    mu = slider_mu.val
    psi_C = slider_psi_C.val
    
    ax.clear()
    f_vals = psi_C_squared(psi_R, phi, psi_A, mu)
    ax.contour(psi_R, phi, f_vals, levels=[psi_C], colors='r')
    ax.set_title(r'Phase Space Contour for $|\psi_C|^2 - |\psi_A|^2 = {}$'.format(psi_C))
    ax.set_xlabel(r'$|\psi_R|$')
    ax.set_ylabel(r'$\phi$')
    ax.grid(True)
    fig.canvas.draw_idle()

# Attach the update function to sliders
slider_psi_A.on_changed(update)
slider_mu.on_changed(update)
slider_psi_C.on_changed(update)

plt.show()


# I think the best route (similar to what Zelda does) is to estimate psi_R first from measured (clear) pupil
# then reconstruct phase analytically.


#%%

#### AFTER CALIBRATION OF THE OPTICAL GAIN REFERENCE (ZERO ABERRATIONS)
# simulate simulate the ZWFS signal with no-aberrations, es 
# iterate phase screens with vibrations
#   for each phase screen,
#       remove N modes
#       add any modal vibrations
#       calculate the non-observable parameters 
#       - strehl 
#       - optical gain (b)
#       simulate the observable parameters
#       - ZWFS signal
#   goal is to build a model that predicts non-observable parameters from observable parameters
#   baseline model for Strehl is a linear combination of ZWFS intensity in sub-regions of the image  
#   baseline model for the optical gain is then b = sqrt(S) * b0. Where b0 is the optical gain without aberrations  


# ZWFS initialization 
z = zelda.Sensor('BALDR_UT_J3')

# grid 
D = 8.0 #m
dim = z.pupil.shape[0]
Dpix = z.pupil_diameter
dx = D / Dpix
dt = 0.001 # s

# stellar parameters 
wvl0 = 1.25e-6 # m central wavelength for simulation 
waveband = 'J'
magnitude = 3.0 

# telescope throughput 
vlti_throughput = 0.1

# atmosphere 
r0 = 0.1 * (wvl0/500e-9)**(6/5) #m - applying Fried parameter wavelength scaling 
L0 = 25 #m
scrn_scaling = 0.3 # to make variance of phase screen match measured Strehl for given  number of modes removed (Naomi data) 
scrn = ps.PhaseScreenKolmogorov(nx_size=dim, pixel_scale=dx, r0=r0, L0=L0, random_seed=1)

# detector 
dit = 1e-3 # s
ron = 1 # electrons
qe = 0.7 # quantum efficiency


# first stage AO 
basis_cropped = ztools.zernike.zernike_basis(nterms=150, npix=z.pupil_diameter)
# we have padding around telescope pupil (check z.pupil.shape and z.pupil_diameter) 
# so we need to put basis in the same frame  
basis_template = np.zeros( z.pupil.shape )
basis = np.array( [ util.insert_concentric( np.nan_to_num(b, 0), basis_template) for b in basis_cropped] )

pupil_disk = basis[0] # we define a disk pupil without secondary - useful for removing Zernike modes later

Nmodes_removed = 14 # Default will be to remove Zernike modes 

# vibrations 
mode_indicies = [0, 1]
spectrum_type = ['1/f', '1/f']
opd = [50e-9, 50e-9]
vibration_frequencies = [15, 45] #Hz


# calculate reference (perfect system) optical gain (b0)
b0, expi = ztools.create_reference_wave_beyond_pupil(z.mask_diameter, z.mask_depth, z.mask_substrate, z.mask_Fratio,
                                       z.pupil_diameter, z.pupil, wvl0, clear=np.array([]), 
                                       sign_mask=np.array([]), cpix=False)





it = 0 

# crop the pupil disk and the phasescreen within it (remove padding outside pupil)
pupil_disk_cropped, atm_in_pupil = util.crop_pupil(pupil_disk, scrn.scrn)

# test project onto Zernike modes 
mode_coefficients = np.array( ztools.zernike.opd_expand(atm_in_pupil * pupil_disk_cropped,\
    nterms=len(basis), aperture =pupil_disk_cropped))

# do the reconstruction for N modes
reco = np.sum( mode_coefficients[:Nmodes_removed,np.newaxis, np.newaxis] * basis[:Nmodes_removed,:,:] ,axis = 0) 

# remove N modes 
ao_1 = scrn_scaling * pupil_disk * (scrn.scrn - reco) 


# add vibrations
# TO DO 

# for calibration purposes
print( f'for {Nmodes_removed} Zernike modes removed (scrn_scaling={scrn_scaling}),\n \
    atmospheric conditions r0= {round(r0,2)}m at a central wavelength {round(1e6*wvl0,2)}um\n\
        post 1st stage AO rmse [nm rms] = ',\
    round( 1e9 * (wvl0 / (2*np.pi) * ao_1)[z.pupil>0.5].std() ) )


# apply DM 
#ao1 *= DM_field

# convert to OPD map
opd_map = z.pupil * wvl0 / (2*np.pi) * ao_1 

# caclulate Strehl ratio
strehl = np.exp( - np.var( ao_1[z.pupil>0.5]) )

b, _ = ztools.create_reference_wave_beyond_pupil_with_aberrations(opd_map, z.mask_diameter, z.mask_depth, z.mask_substrate, z.mask_Fratio,
                                       z.pupil_diameter, z.pupil, wvl0, clear=np.array([]), 
                                       sign_mask=np.array([]), cpix=False)

N0 = ztools.propagate_opd_map(0*opd_map, z.mask_diameter, 0*z.mask_depth, z.mask_substrate,
                                        z.mask_Fratio, z.pupil_diameter, z.pupil, wave=wvl0)

I0 = ztools.propagate_opd_map(0*opd_map, z.mask_diameter, z.mask_depth, z.mask_substrate,
                                        z.mask_Fratio, z.pupil_diameter, z.pupil, wave=wvl0)


# normalized such that np.sum( I0 ) / np.sum( N0 ) ~ 1 where N0.max() = 1. 
# do normalization by known area of the pupil and the input stellar magnitude at the given wavelength 
# represent as #photons / s / pixel / nm

# input amplitude of the star 
photon_scaling = vlti_throughput * (np.pi * (D/2)**2) / (np.pi * z.pupil_diameter/2)**2 * util.magnitude_to_photon_flux(magnitude=magnitude, band = waveband, wavelength= 1e9*wvl0)

Ic = photon_scaling * z.propagate_opd_map( opd_map , wave = wvl0 )

det_binning = round( bldr.calculate_detector_binning_factor(grid_pixels_across_pupil = z.pupil_diameter, detector_pixels_across_pupil = 12) )

i = bldr.detect( Ic, binning = (16, 16), qe=qe , dit=dit, ron= ron, include_shotnoise=True, spectral_bandwidth = None )

plt.imshow( i ) ; plt.colorbar(); plt.show()


# IF YOU WANT TO VISUALIZE ANY INTERMEDIATE STEPS
#plt.figure(); plt.imshow( z.pupil * scrn.scrn); plt.show()
#plt.imshow( reco ); plt.colorbar(); plt.show()
#plt.imshow( ao_1 ); plt.colorbar(); plt.show()
#plt.imshow( z.pupil * np.abs(b0) ); plt.colorbar(); plt.show()
#plt.imshow( z.pupil * np.angle(b0) ); plt.colorbar(); plt.show()
#plt.imshow( z.pupil * np.abs(b) ); plt.colorbar(); plt.show()
#plt.imshow( z.pupil * np.angle(b) ); plt.colorbar(); plt.show()
#plt.imshow( Ic ); plt.show()
