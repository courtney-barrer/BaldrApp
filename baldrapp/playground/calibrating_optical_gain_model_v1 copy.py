
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import pearsonr
import pickle
from types import SimpleNamespace
from sklearn.linear_model import LinearRegression
import importlib # reimport package after edits: importlib.reload(bldr)

# from courtney-barrer's fork of pyzelda
import pyzelda.zelda as zelda
import pyzelda.ztools as ztools
import pyzelda.utils.aperture as aperture
import pyzelda.utils.imutils as imutils
from baldrapp.common import phasescreens as ps
from baldrapp.common import utilities as util
from baldrapp.common import baldr_core as bldr

from baldrapp.common.baldr_core import StrehlModel 

"""
Most of the calibration of the Strehl model done in this script has been generalized to a function in the baldr_core module:

bldr.calibrate_strehl_model( zwfs_ns, save_results_path = 'path/to/save/results', train_fraction = 0.6, correlation_threshold = 0.5, \
    number_of_screen_initiations = 50, scrn_scaling_grid = np.logspace(-1,0.2,5) )    
    

The end of this script does some analysis on the residual of the predicted optical gain from the Strehl model (b=b0 * sqrt(S))
in different strehl regions and as a function of the radial profile of the optical gain.

results were saved in sydney_test overleaf project.
"""



def compute_correlation_map(intensity_frames, strehl_ratios):
    # intensity_frames: k x N x M array (k frames of N x M pixels)
    # strehl_ratios: k array (Strehl ratio for each frame)
    
    k, N, M = intensity_frames.shape
    correlation_map = np.zeros((N, M))
    
    for i in range(N):
        for j in range(M):
            pixel_intensity_series = intensity_frames[:, i, j]
            correlation_map[i, j], _ = pearsonr(pixel_intensity_series, strehl_ratios)
    
    return correlation_map





# Function to load the model from a pickle file
def load_model_from_pickle(filename):
    """
    Loads the StrehlModel object from a pickle file.
    
    Args:
        filename (str): The file path from where the model should be loaded.
    
    Returns:
        StrehlModel: The loaded StrehlModel instance.
    """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
        
    return model

# Function to save the model using pickle
# def save_model(filename, model):
#     with open(filename, 'wb') as f:
#         pickle.dump(model, f)

# # Function to load the model using pickle
# def load_model(filename):
#     with open(filename, 'rb') as f:
#         return pickle.load(f)
    
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






# initialize our ZWFS instrument
wvl0=1.25e-6
config_ini = 'BaldrApp/baldrapp/configurations/BALDR_UT_J3.ini'
zwfs_ns = bldr.init_zwfs_from_config_ini( config_ini=config_ini , wvl0=wvl0)


# short hand for pupil dimensions (pixels)
#dim = zwfs_ns.grid.N * zwfs_ns.grid.padding_factor # should match zwfs_ns.pyZelda.pupil_dim
# spatial differential in pupil space 
dx = zwfs_ns.grid.D / zwfs_ns.grid.N
# get required simulation sampling rate to match physical parameters 
dt = dx * zwfs_ns.atmosphere.pixels_per_iteration / zwfs_ns.atmosphere.v # s # simulation sampling rate

print(f'current parameters have effective wind velocity = {round(zwfs_ns.atmosphere.v )}m/s')
scrn = ps.PhaseScreenKolmogorov(nx_size=zwfs_ns.grid.dim, pixel_scale=dx, r0=zwfs_ns.atmosphere.r0, L0=zwfs_ns.atmosphere.l0, random_seed=1)


# first stage AO 
basis_cropped = ztools.zernike.zernike_basis(nterms=150, npix=zwfs_ns.pyZelda.pupil_diameter)
# we have padding around telescope pupil (check zwfs_ns.pyZelda.pupil.shape and zwfs_ns.pyZelda.pupil_diameter) 
# so we need to put basis in the same frame  
basis_template = np.zeros( zwfs_ns.pyZelda.pupil.shape )
basis = np.array( [ util.insert_concentric( np.nan_to_num(b, 0), basis_template) for b in basis_cropped] )

pupil_disk = basis[0] # we define a disk pupil without secondary - useful for removing Zernike modes later

Nmodes_removed = 14 # Default will be to remove Zernike modes 

# vibrations 
mode_indicies = [0, 1]
spectrum_type = ['1/f', '1/f']
opd = [50e-9, 50e-9]
vibration_frequencies = [15, 45] #Hz


# calculate reference (perfect system) optical gain (b0)
b0, expi = ztools.create_reference_wave_beyond_pupil(zwfs_ns.pyZelda.mask_diameter, zwfs_ns.pyZelda.mask_depth, zwfs_ns.pyZelda.mask_substrate, zwfs_ns.pyZelda.mask_Fratio,
                                       zwfs_ns.pyZelda.pupil_diameter, zwfs_ns.pyZelda.pupil, wvl0, clear=np.array([]), 
                                       sign_mask=np.array([]), cpix=False)

# to put in pixel space 
b0_pixelspace = bldr.average_subarrays( abs(b0) , (12,12)) 


# input amplitude of the star 
photon_scaling = zwfs_ns.throughput.vlti_throughput * (np.pi * (zwfs_ns.grid.D/2)**2) / (np.pi * zwfs_ns.pyZelda.pupil_diameter/2)**2 * util.magnitude_to_photon_flux(magnitude=zwfs_ns.stellar.magnitude, band = zwfs_ns.stellar.waveband, wavelength= 1e9*wvl0)

telemetry = {
    'I0':[],
    'N0':[],
    'scrn':[],
    'ao_1':[],
    'Ic':[],
    'i':[],
    'strehl':[],
    'dm_cmd':[],
    'b':[],
    'b_pixelspace':[],
    'ao_2':[]
}

telem_ns = SimpleNamespace(**telemetry)

it_grid = np.arange(0, 100)
scrn_scaling_grid = np.logspace(0,1,10)
dm_phasescreen = True # apply the phasescreen to the DM (this is how we will calibrate the real system)
# DO THIS - BUT PUT ABERRATION ON THE DM!!!! (SAME WAY WE WOULD CALIBRATE THE REAL SYSTEM)

scrn_list = []
for _ in range(100):
    #scrn = ps.PhaseScreenKolmogorov(nx_size=zwfs_ns.grid.N, pixel_scale=dx, r0=zwfs_ns.atmosphere.r0, L0=zwfs_ns.atmosphere.l0, random_seed=1)
    scrn = ps.PhaseScreenKolmogorov(nx_size=24, pixel_scale = zwfs_ns.grid.D / 24, r0=zwfs_ns.atmosphere.r0, L0=zwfs_ns.atmosphere.l0, random_seed=None)
    scrn_list.append( scrn ) 
    #zwfs_ns.grid.pupil_mask * util.insert_concentric( scrn.scrn, zwfs_ns.pyZelda.pupil ) )

for it in range(len(scrn_list)):

    # roll screen
    #scrn.add_row()     
    for ph_scale in scrn_scaling_grid: 
        

        if dm_phasescreen:
            #scaling_factor=0.05, drop_indicies = [0, 11, 11 * 12, -1] , plot_cmd=False
            zwfs_ns.dm.current_cmd =  util.create_phase_screen_cmd_for_DM(scrn_list[it],  scaling_factor=ph_scale , drop_indicies = [0, 11, 11 * 12, -1] , plot_cmd=False) 
        
            opd_current_dm = bldr.get_dm_displacement( command_vector= zwfs_ns.dm.current_cmd   , gain=zwfs_ns.dm.opd_per_cmd, \
                sigma= zwfs_ns.grid.dm_coord.act_sigma_wavesp, X=zwfs_ns.grid.wave_coord.X, Y=zwfs_ns.grid.wave_coord.Y,\
                    x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp )
            
            phi = zwfs_ns.grid.pupil_mask  *  2*np.pi / zwfs_ns.optics.wvl0 * (  opd_current_dm  )
            
            pupil_disk_cropped, atm_in_pupil = util.crop_pupil(pupil_disk, phi)
        else: 
            # crop the pupil disk and the phasescreen within it (remove padding outside pupil)
            pupil_disk_cropped, atm_in_pupil = util.crop_pupil(pupil_disk, ph_scale * scrn.scrn)
            
            
        # test project onto Zernike modes 
        mode_coefficients = np.array( ztools.zernike.opd_expand(atm_in_pupil * pupil_disk_cropped,\
            nterms=len(basis), aperture =pupil_disk_cropped))

        # do the reconstruction for N modes
        reco = np.sum( mode_coefficients[:Nmodes_removed,np.newaxis, np.newaxis] * basis[:Nmodes_removed,:,:] ,axis = 0) 

        # remove N modes 
        
        if dm_phasescreen:
            ao_1 = zwfs_ns.atmosphere.scrn_scaling * pupil_disk * (phi - reco) 
        else:
            ao_1 = zwfs_ns.atmosphere.scrn_scaling * pupil_disk * (scrn.scrn - reco)     
            
        # add vibrations
        # TO DO 

        # for calibration purposes
        print( f'for {Nmodes_removed} Zernike modes removed (scrn_scaling={ph_scale}),\n \
            atmospheric conditions r0= {round(zwfs_ns.atmosphere.r0,2)}m at a central wavelength {round(1e6*wvl0,2)}um\n\
                post 1st stage AO rmse [nm rms] = ',\
            round( 1e9 * (wvl0 / (2*np.pi) * ao_1)[zwfs_ns.pyZelda.pupil>0.5].std() ) )


        # apply DM 
        #ao1 *= DM_field

        # convert to OPD map
        opd_map = zwfs_ns.pyZelda.pupil * wvl0 / (2*np.pi) * ao_1 
        
        if it==0:
            N0 = ztools.propagate_opd_map(0*opd_map, zwfs_ns.pyZelda.mask_diameter, 0*zwfs_ns.pyZelda.mask_depth, zwfs_ns.pyZelda.mask_substrate,
                                            zwfs_ns.pyZelda.mask_Fratio, zwfs_ns.pyZelda.pupil_diameter, zwfs_ns.pyZelda.pupil, wave=wvl0)

            I0 = ztools.propagate_opd_map(0*opd_map, zwfs_ns.pyZelda.mask_diameter, zwfs_ns.pyZelda.mask_depth, zwfs_ns.pyZelda.mask_substrate,
                                            zwfs_ns.pyZelda.mask_Fratio, zwfs_ns.pyZelda.pupil_diameter, zwfs_ns.pyZelda.pupil, wave=wvl0)
        
            telem_ns.I0.append(I0)
            telem_ns.N0.append(N0)  

        # caclulate Strehl ratio
        strehl = np.exp( - np.var( ao_1[zwfs_ns.pyZelda.pupil>0.5]) )

        b, _ = ztools.create_reference_wave_beyond_pupil_with_aberrations(opd_map, zwfs_ns.pyZelda.mask_diameter, zwfs_ns.pyZelda.mask_depth, zwfs_ns.pyZelda.mask_substrate, zwfs_ns.pyZelda.mask_Fratio,
                                            zwfs_ns.pyZelda.pupil_diameter, zwfs_ns.pyZelda.pupil, wvl0, clear=np.array([]), 
                                            sign_mask=np.array([]), cpix=False)


        b_pixelspace = bldr.average_subarrays( abs(b) , (zwfs_ns.detector.binning, zwfs_ns.detector.binning)) 

        # normalized such that np.sum( I0 ) / np.sum( N0 ) ~ 1 where N0.max() = 1. 
        # do normalization by known area of the pupil and the input stellar magnitude at the given wavelength 
        # represent as #photons / s / pixel / nm

        Ic = photon_scaling * zwfs_ns.pyZelda.propagate_opd_map( opd_map , wave = wvl0 )

        det_binning = round( bldr.calculate_detector_binning_factor(grid_pixels_across_pupil = zwfs_ns.pyZelda.pupil_diameter, detector_pixels_across_pupil = 12) )

        i = bldr.detect( Ic, binning = (zwfs_ns.detector.binning, zwfs_ns.detector.binning), qe=zwfs_ns.detector.qe , dit=zwfs_ns.detector.dit, ron= zwfs_ns.detector.ron, include_shotnoise=True, spectral_bandwidth = zwfs_ns.stellar.bandwidth )

        #telem_ns.ao_1.append(zwfs_ns.pyZelda.pupil * ao_1)
        telem_ns.i.append(i)
        telem_ns.Ic.append(Ic)
        telem_ns.strehl.append(strehl)
        telem_ns.b.append(b)
        telem_ns.b_pixelspace.append(b_pixelspace)
        telem_ns.dm_cmd.append(zwfs_ns.dm.current_cmd )
        
    print( f'iteration {it} done')

# IF YOU WANT TO VISUALIZE ANY INTERMEDIATE STEPS
#plt.figure(); plt.imshow( zwfs_ns.pyZelda.pupil * scrn.scrn); plt.show()
#plt.imshow( reco ); plt.colorbar(); plt.show()
#plt.imshow( ao_1 ); plt.colorbar(); plt.show()
#plt.imshow( zwfs_ns.pyZelda.pupil * np.abs(b0) ); plt.colorbar(); plt.show()
#plt.imshow( zwfs_ns.pyZelda.pupil * np.angle(b0) ); plt.colorbar(); plt.show()
#plt.imshow( zwfs_ns.pyZelda.pupil * np.abs(b) ); plt.colorbar(); plt.show()
#plt.imshow( zwfs_ns.pyZelda.pupil * np.angle(b) ); plt.colorbar(); plt.show()
#plt.imshow( Ic ); plt.show()
#plt.imshow( i ) ; plt.colorbar(); plt.show()

# Example usage:
# example_images = [telem_ns.ao_1, telem_ns.strehl, telem_ns.i, [abs(b) for b in telem_ns.b] ]  # 3 lists of 10 images each

# # Call the function to dynamically slide through the images
# util.display_images_with_slider(example_images)
# # Call the function to create and save the movie
# util.display_images_as_movie(example_images, plot_titles=None, cbar_labels=None, save_path="output_movie.mp4", fps=5)

"""
observable is DM command and ZWFS signal

non-observable is Strehl and optical gain that we wish to model 

use known DM commands to calibrate Strehl and optical gain model 

we want to model optical gain in the pixel space. 
Therefore any iteration model can be scaled by it.

proceedure: 

roll various phase screens across DM and get ZWFS intensity response 

dm shape related to strehl via DM influence function

filter pixels that above Pearson correlation threshold
fit a linear model to the data ( S ~ exp(-var(dm_rmse)) ~ sum_i alpha_i * I_i + i0_i )

verify optical gain model 
b = sqrt(S) * b0

metrics 
- rmse inside the pupil vs strehl

for given strehl bin  
- mean residual vs radial profile (look at bias in the model)
- rmse vs radial profile 


store the results in a config file 
- strehl model pixels  
- coefficients
- b0 

"""

fig_path = '/home/benja/Downloads/'
# Example usage
correlation_map = compute_correlation_map(np.array( telem_ns.i ), np.array( telem_ns.strehl) )

# SNR 
SNR = np.mean( telem_ns.i ,axis=0 ) / np.std( telem_ns.i ,axis=0  )

im_list =  [ correlation_map ]
cbar_list = ['pearson R']

util.nice_heatmap_subplots( im_list = [ correlation_map ] , cbar_label_list = ['Pearson R'] , \
    savefig = None) #fig_path + 'strehl_vs_intensity_pearson_R.png' )

util.nice_heatmap_subplots( im_list = [ SNR / np.max( SNR ) ] , cbar_label_list = ['normalized SNR'] ,\
    savefig = None)# fig_path + 'SNR_simulation.png' )

# Select top 5% of pixels with the highest correlation
threshold = 0.9
selected_pixels = correlation_map > threshold

plt.figure()
plt.imshow( selected_pixels)
plt.colorbar(label = "filter")
plt.savefig(fig_path + 'selected_pixels.png', bbox_inches='tight', dpi=300)
plt.show()

## FITTING THE MODEL 
model_description = "Linear regression model fitting intensities to Strehl ratio."

model = StrehlModel(model_description)

#pixel_indices = np.where( selected_pixels )

i_train = int( 0.6 * len( telem_ns.i ) )

y_train = np.array(  telem_ns.strehl )[:i_train]
X_train = np.array( telem_ns.i )[:i_train] 

y_test = np.array(  telem_ns.strehl )[i_train:]
X_test = np.array( telem_ns.i )[i_train:]


#coefficients, intercept = model.fit_linear_model(x, y)
model.fit(X = X_train,\
        y = y_train ,\
        pixel_filter=selected_pixels )

y_fit = model.apply_model(X_test) 

# add the pupil in 
model.name = zwfs_ns.name # so we know what config file was used 

#y_fit = model.predict(x)

# show out of sample test results 
util.plot_data_and_residuals(y_test, y_test, y_fit, xlabel=r'$\text{Strehl Ratio}$', ylabel=r'$\text{Predicted Strehl Ratio}$', \
    residual_ylabel=r'$\Delta$',label_1="1:1", label_2="model", savefig='{}strehl_linear_fit.png'.format(fig_path) )


# save the model
model.save_model_to_pickle(filename=fig_path + 'strehl_model.pkl')

# read it in 
model = load_model_from_pickle(filename=fig_path + 'strehl_model.pkl')



### MODELLING OPTICAL GAIN - VERIFICATION 
"""
# baseline model is b = sqrt(S) * b0
# quality metric = rmse inside the pupil! 
# using meeasured Strehl (which we won't have in the real system)
"""

strehl = np.array( telem_ns.strehl )
y = np.array( [abs(b)for b in  telem_ns.b ] )

# using real strehl values
#y_model = np.array( [ss* np.abs(b0) for ss in np.sqrt( np.array( telem_ns.strehl ) ) ] )

# using the strehl model 
y_model = np.array( [ss * np.abs(b0) for ss in  np.sqrt( model.apply_model( np.array( telem_ns.i  ) ) ) ] )

## get azimuth statistics 
#y_az = np.array([imutils.profile(yy, ptype='mean', step=1, mask=None, center=None, rmax=0, clip=True, exact=False)[0] for yy in y])


# to adjust for small statistics we use Bessel correction for variance
img = y[0].copy()

# array dimensions
dimx = img.shape[1]
dimy = img.shape[0]
# center
center = (dimx // 2, dimy // 2)

# intermediate cartesian arrays
xx, yy = np.meshgrid(np.arange(dimx, dtype=np.int64) - center[0], np.arange(dimy, dtype=np.int64) - center[1])
rr = np.sqrt(xx**2 + yy**2)

# rounds for faster calculation
rr = np.round(rr, decimals=0)

# find unique radial values
uniq = np.unique(rr, return_inverse=True, return_counts=True)

r_uniq_val = uniq[0]
r_uniq_inv = uniq[1]
r_uniq_cnt = uniq[2]


# get radial vector in pixels 
_, r_pix  = imutils.profile(y[0], ptype='mean', step=1, mask=None, center=None, rmax=0, clip=True, exact=False)

mean_b0_az = imutils.profile(abs(b0), ptype='mean', step=1, mask=None, center=None, rmax=0, clip=True, exact=False)[0] 
mean_residual_az = np.array([imutils.profile(meas-model, ptype='mean', step=1, mask=None, center=None, rmax=0, clip=True, exact=False)[0] for meas,model in zip(y, y_model) ])
std_residual_az = np.array([imutils.profile(meas-model, ptype='std', step=1, mask=None, center=None, rmax=0, clip=True, exact=False)[0] for meas,model in zip(y, y_model) ])
# editted function for ptype = var using Bessel correction for small sample size 
var_residual_az = np.array([imutils.profile(meas-model, ptype='var', step=1, mask=None, center=None, rmax=0, clip=True, exact=False)[0] for meas,model in zip(y, y_model) ])

bessel_correction = np.array( [rr / (rr - 1) if rr>1 else np.nan for rr in r_uniq_cnt] )

# Normalize r_pix
r_pix_normalized = r_pix / (zwfs_ns.grid.N / 2)

# Define Strehl bins and labels
strehl_bins = [(0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
colors = ['blue', 'green', 'red']
labels = ['0.4 ≤ Strehl < 0.6', '0.6 ≤ Strehl < 0.8', '0.8 ≤ Strehl < 1.0']

# Create the plot
plt.figure(figsize=(8, 6))

for (low, high), color, label in zip(strehl_bins, colors, labels):
    # Define the Strehl filter for the current bin
    strehl_filter = (strehl >= low) & (strehl < high)
    
    # Calculate mean and standard deviation for the current Strehl bin
    mean_residual = np.mean(mean_residual_az[strehl_filter], axis=0)
    #std_residual = np.mean( std_residual_az[strehl_filter], axis=0)  # Standard deviation
    #std_residual = np.mean( np.sqrt(var_residual_az[strehl_filter] ), axis=0)  # editted function for ptype = var using Bessel correction for small sample size 
    std_residual = np.std( mean_residual_az[strehl_filter], axis=0, ddof=1)  # Standard deviation
    
    # Plot the mean as a solid line for the current bin
    plt.plot(r_pix_normalized, mean_residual/mean_b0_az, color=color, label=label)
    
    # Fill the area around the mean within 1 standard deviation
    plt.fill_between(r_pix_normalized, (mean_residual - std_residual)/mean_b0_az, 
                     (mean_residual + std_residual)/mean_b0_az, color=color, alpha=0.2)

# Formatting the plot
plt.ylabel('Expected Fractional Residual\n'+r'$\left \langle \frac{|b| - |b_{model}|}{b_0} \right \rangle$',fontsize=15)
plt.xlabel(r'Fractional Pupil Radius',fontsize=15)
plt.xlim( [0, 1.05])
plt.gca().tick_params(axis='both', which='major', labelsize=15)
#plt.title('Expected Residual and Standard Deviation for Different Strehl Bins')
plt.grid(True)
plt.legend(loc='best',fontsize=15)  # Add a legend to differentiate Strehl bins


# Show the plot
plt.tight_layout()
plt.savefig(fig_path + 'residuals_optical_gain_radial_profile_WITH_STREHL_MODEL.png', bbox_inches='tight', dpi=300)
plt.show()




