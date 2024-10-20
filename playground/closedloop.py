
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import pearsonr
import pickle
from types import SimpleNamespace
from sklearn.linear_model import LinearRegression
import importlib # reimport package after edits: importlib.reload(bldr)
import os
import datetime
# from courtney-barrer's fork of pyzelda
import pyzelda.zelda as zelda
import pyzelda.ztools as ztools
import pyzelda.utils.aperture as aperture
import pyzelda.utils.imutils as imutils
from common import phasescreens as ps
from common import utilities as util
from common import baldr_core as bldr
from common import DM_registration
from common import DM_basis

from common.baldr_core import StrehlModel 

import numpy as np
from sklearn.model_selection import train_test_split


import numpy as np
from sklearn.model_selection import train_test_split



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



class my_lin_fit:
    # Rows are samples, columns are features
    def __init__(self, model_name='pixelwise_first'):
        """
        Initialize the linear fit model.
        
        Parameters:
        - model_name: str, the name/type of model (currently supports 'pixelwise_first')
        """
        self.model_name = model_name
        self.models = None
        
    def fit(self, X, Y):
        """
        Fit the model based on the input features X and target Y.
        
        Parameters:
        - X: np.ndarray, shape (N, P), input data matrix (N samples, P features)
        - Y: np.ndarray, shape (N, P), target data matrix (same shape as X)
        
        Returns:
        - coe: list of model coefficients for each feature
        """
        if self.model_name == 'pixelwise_first':
            coe = []
            # Fit a first-order polynomial (linear) for each feature (each column)
            for v in range(X.shape[1]):
                coe.append(np.polyfit(X[:, v], Y[:, v], 1))  # Linear fit for each feature
            self.models = coe
            return coe 
        
    def apply(self, X):
        """
        Apply the fitted model to new input data X to make predictions.
        
        Parameters:
        - X: np.ndarray, input data for which to predict Y.
        
        Returns:
        - Y_pred: np.ndarray, predicted values based on the fitted models
        """
        if self.model_name == 'pixelwise_first':
            Y_pred = []
            # Apply the model to each feature
            for v in range(len(self.models)):
                a_i, b_i = self.models[v]
                if len(X.shape) == 1:
                    # X is 1D (single sample)
                    assert len(X) == len(self.models), "Dimension mismatch: X does not match model dimensions."
                    Y_pred.append(a_i * X[v] + b_i)
                elif len(X.shape) == 2:
                    # X is 2D (multiple samples)
                    assert X.shape[1] == len(self.models), "Dimension mismatch: X columns do not match model dimensions."
                    Y_pred.append(a_i * X[:, v] + b_i)
            return np.array(Y_pred).T  # Transpose to match the input shape
        else:
            return None
        
        
tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S")

# initialize our ZWFS instrument
wvl0=1.25e-6
config_ini = '/home/benja/Documents/BALDR/BaldrApp/configurations/BALDR_UT_J3.ini'
zwfs_ns = bldr.init_zwfs_from_config_ini( config_ini=config_ini , wvl0=wvl0)

fig_path = f'/home/benja/Downloads/act_cross_coupling_{zwfs_ns.dm.actuator_coupling_factor}_{tstamp}/'
if os.path.exists(fig_path) == False:
    os.makedirs(fig_path)
    
plot_intermediate_results = False


# set up detector class from zwfs_ns.detector 
# bldr_detector = bldr.detector( binning = (zwfs_ns.detector.binning, zwfs_ns.detector.binning), qe=zwfs_ns.detector.qe ,\
#     dit=zwfs_ns.detector.dit, ron= zwfs_ns.detector.ron)


# short hand for pupil dimensions (pixels)
#dim = zwfs_ns.grid.N * zwfs_ns.grid.padding_factor # should match zwfs_ns.pyZelda.pupil_dim
# spatial differential in pupil space 
dx = zwfs_ns.grid.D / zwfs_ns.grid.N
# get required simulation sampling rate to match physical parameters 
dt = dx * zwfs_ns.atmosphere.pixels_per_iteration / zwfs_ns.atmosphere.v # s # simulation sampling rate

print(f'current parameters have effective wind velocity = {round(zwfs_ns.atmosphere.v )}m/s')
scrn = ps.PhaseScreenKolmogorov(nx_size=zwfs_ns.grid.dim, pixel_scale=dx, r0=zwfs_ns.atmosphere.r0, L0=zwfs_ns.atmosphere.l0, random_seed=1)

phase_scaling_factor = 0.3

# first stage AO 
basis_cropped = ztools.zernike.zernike_basis(nterms=150, npix=zwfs_ns.pyZelda.pupil_diameter)
# we have padding around telescope pupil (check zwfs_ns.pyZelda.pupil.shape and zwfs_ns.pyZelda.pupil_diameter) 
# so we need to put basis in the same frame  
basis_template = np.zeros( zwfs_ns.pyZelda.pupil.shape )
basis = np.array( [ util.insert_concentric( np.nan_to_num(b, 0), basis_template) for b in basis_cropped] )

#pupil_disk = basis[0] # we define a disk pupil without secondary - useful for removing Zernike modes later

Nmodes_removed = 14 # Default will be to remove Zernike modes 

# vibrations 
mode_indicies = [0, 1]
spectrum_type = ['1/f', '1/f']
opd = [50e-9, 50e-9]
vibration_frequencies = [15, 45] #Hz


# input flux scaling (photons / s / wavespace_pixel / nm) 
photon_flux_per_pixel_at_vlti = zwfs_ns.throughput.vlti_throughput * (np.pi * (zwfs_ns.grid.D/2)**2) / (np.pi * zwfs_ns.pyZelda.pupil_diameter/2)**2 * util.magnitude_to_photon_flux(magnitude=zwfs_ns.stellar.magnitude, band = zwfs_ns.stellar.waveband, wavelength= 1e9*wvl0)


# internal aberrations (opd in meters)
opd_internal = util.apply_parabolic_scratches(np.zeros( zwfs_ns.grid.pupil_mask.shape ) , dx=dx, dy=dx, list_a= [ 0.1], list_b = [0], list_c = [-2], width_list = [2*dx], depth_list = [100e-9])

opd_flat_dm = bldr.get_dm_displacement( command_vector= zwfs_ns.dm.dm_flat  , gain=zwfs_ns.dm.opd_per_cmd, \
                sigma= zwfs_ns.grid.dm_coord.act_sigma_wavesp, X=zwfs_ns.grid.wave_coord.X, Y=zwfs_ns.grid.wave_coord.Y,\
                    x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp )


# calculate reference (only with internal aberrations) optical gain (b0)
b0_wsp, _ = ztools.create_reference_wave_beyond_pupil_with_aberrations(opd_internal + opd_flat_dm , zwfs_ns.pyZelda.mask_diameter, zwfs_ns.pyZelda.mask_depth, zwfs_ns.pyZelda.mask_substrate, zwfs_ns.pyZelda.mask_Fratio,
                                       zwfs_ns.pyZelda.pupil_diameter, zwfs_ns.pyZelda.pupil, wvl0, clear=np.array([]), 
                                       sign_mask=np.array([]), cpix=False)

# # 
# b0_perfect, _ = ztools.create_reference_wave_beyond_pupil(zwfs_ns.pyZelda.mask_diameter, zwfs_ns.pyZelda.mask_depth, zwfs_ns.pyZelda.mask_substrate, zwfs_ns.pyZelda.mask_Fratio,
#                                        zwfs_ns.pyZelda.pupil_diameter, zwfs_ns.pyZelda.pupil, wvl0, clear=np.array([]), 
#                                        sign_mask=np.array([]), cpix=False)
# plt.figure(); plt.imshow( abs( b0_wsp ) - abs( b0_perfect) ) );plt.colorbar(); plt.show()
## >>>>>>> Note: the peak difference in b0 when including internal aberrations is 0.001. i.e. 0.008/0.8 = 1% difference in optical gain
    
    
# to put in pixel space (we just average with the same binning as the bldr detector)
b0 = bldr.average_subarrays( abs(b0_wsp) , (zwfs_ns.detector.binning, zwfs_ns.detector.binning) )
#plt.imshow( b0_pixelspace ); plt.colorbar(); plt.show()

# propagate through ZWFS to the detector plane intensities (in wavespace)
# phasemask in  
# I0_wsp =  ztools.propagate_opd_map( zwfs_ns.pyZelda.pupil * (opd_internal + opd_flat_dm ), zwfs_ns.pyZelda.mask_diameter, zwfs_ns.pyZelda.mask_depth, zwfs_ns.pyZelda.mask_substrate,
#                                             zwfs_ns.pyZelda.mask_Fratio, zwfs_ns.pyZelda.pupil_diameter, photon_flux_per_pixel_at_vlti**0.5 *zwfs_ns.pyZelda.pupil, wave=zwfs_ns.optics.wvl0)

# # phasemask out
# N0_wsp =  ztools.propagate_opd_map(zwfs_ns.pyZelda.pupil * (opd_internal + opd_flat_dm ), zwfs_ns.pyZelda.mask_diameter, 0*zwfs_ns.pyZelda.mask_depth, zwfs_ns.pyZelda.mask_substrate,
#                                             zwfs_ns.pyZelda.mask_Fratio, zwfs_ns.pyZelda.pupil_diameter,  photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns.pyZelda.pupil, wave=zwfs_ns.optics.wvl0)

# # bin to detector pixelspace 
# I0 = bldr.detect( I0_wsp, binning = (zwfs_ns.detector.binning, zwfs_ns.detector.binning), qe=zwfs_ns.detector.qe , dit=zwfs_ns.detector.dit, ron= zwfs_ns.detector.ron, include_shotnoise=True, spectral_bandwidth = zwfs_ns.stellar.bandwidth )
# N0 = bldr.detect( N0_wsp, binning = (zwfs_ns.detector.binning, zwfs_ns.detector.binning), qe=zwfs_ns.detector.qe , dit=zwfs_ns.detector.dit, ron= zwfs_ns.detector.ron, include_shotnoise=True, spectral_bandwidth = zwfs_ns.stellar.bandwidth )


# quicker way - make sure get frames returns the same as the above!!!! 
I0 = bldr.get_I0( opd_input = 0 * zwfs_ns.pyZelda.pupil ,  amp_input = photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns.pyZelda.pupil, opd_internal=zwfs_ns.pyZelda.pupil * (opd_internal + opd_flat_dm), \
    zwfs_ns=zwfs_ns, detector=zwfs_ns.detector, include_shotnoise=True , use_pyZelda = True)

N0 = bldr.get_N0( opd_input = 0 * zwfs_ns.pyZelda.pupil ,  amp_input = photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns.pyZelda.pupil, opd_internal=zwfs_ns.pyZelda.pupil * (opd_internal + opd_flat_dm), \
    zwfs_ns=zwfs_ns, detector=zwfs_ns.detector, include_shotnoise=True , use_pyZelda = True)


# Build a basic Interaction matrix (IM) for the ZWFS
basis_name = 'Zonal_pinned_edges'
Nmodes = 100
M2C_0 = DM_basis.construct_command_basis( basis= basis_name, number_of_modes = Nmodes, without_piston=True).T  

### The amplitude input here is sqrt(photon flux)
zwfs_ns = bldr.classify_pupil_regions( opd_input = 0*opd_internal ,  amp_input = photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns.pyZelda.pupil , \
    opd_internal=opd_internal,  zwfs_ns = zwfs_ns , detector=zwfs_ns.detector , pupil_diameter_scaling = 1.0, pupil_offset = (0,0)) 



zwfs_ns = bldr.build_IM( zwfs_ns ,  calibration_opd_input = 0*opd_internal , calibration_amp_input = photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns.pyZelda.pupil  , \
            opd_internal = opd_internal,  basis = basis_name, Nmodes =  Nmodes, poke_amp = 0.05, poke_method = 'double_sided_poke',\
                imgs_to_mean = 1, detector=zwfs_ns.detector)

zwfs_ns = bldr.add_controllers( zwfs_ns, TT = 'PID', HO = 'leaky')

if plot_intermediate_results:
    bldr.plot_eigenmodes( zwfs_ns , descr_label = f'dm_interactuator_coupling-{zwfs_ns.dm.actuator_coupling_factor}', save_path = fig_path )

_ = bldr.construct_ctrl_matricies_from_IM( zwfs_ns,  method = 'Eigen_TT-HO', Smax = 60, TT_vectors = DM_basis.get_tip_tilt_vectors() )

### 
# DM registration 
###

# get inner corners for estiamting DM center in pixel space (have to deal seperately with pinned actuator basis)
if zwfs_ns.reco.IM.shape[0] == 100: # outer actuators are pinned, 
    corner_indicies = DM_registration.get_inner_square_indices(outer_size=10, inner_offset=3, without_outer_corners=False)
    
elif zwfs_ns.reco.IM.shape[0] == 140: # outer acrtuators are free 
    print(140)
    corner_indicies = DM_registration.get_inner_square_indices(outer_size=12, inner_offset=4, without_outer_corners=True)
else:
    print("CASE NOT MATCHED  d['I2M'].data.shape = { d['I2M'].data.shape}")
    
img_4_corners = []
dm_4_corners = []
for i in corner_indicies:
    dm_4_corners.append( np.where( M2C_0[i] )[0][0] )
    #dm2px.get_DM_command_in_2D( d['M2C'].data[:,i]  # if you want to plot it 

    tmp = np.zeros( zwfs_ns.pupil_regions.pupil_filt.shape )
    tmp.reshape(-1)[zwfs_ns.pupil_regions.pupil_filt.reshape(-1)] = zwfs_ns.reco.IM[i] 

    #plt.imshow( tmp ); plt.show()
    img_4_corners.append( abs(tmp ) )

#plt.imshow( np.sum( tosee, axis=0 ) ); plt.show()

# dm_4_corners should be an array of length 4 corresponding to the actuator index in the (flattened) DM command space
# img_4_corners should be an array of length 4xNxM where NxM are the image dimensions.
# !!! It is very important that img_4_corners are registered in the same order as dm_4_corners !!!
transform_dict = DM_registration.calibrate_transform_between_DM_and_image( dm_4_corners, img_4_corners, debug=plot_intermediate_results, fig_path = None )

interpolated_intensities = DM_registration.interpolate_pixel_intensities(image = I0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])


# interpolate these fields onto the registered actuator grid
b0_dm = DM_registration.interpolate_pixel_intensities(image = I0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])
I0_dm = DM_registration.interpolate_pixel_intensities(image = b0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])
N0_dm = DM_registration.interpolate_pixel_intensities(image = N0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])

# calibrate a model to map a subset of pixel intensities to Strehl Ratio 
# should eventually come back and debug for model_type = lin_comb - since it seemed to work better intially
# = bldr.calibrate_strehl_model( zwfs_ns, save_results_path = fig_path, train_fraction = 0.6, correlation_threshold = 0.6, \
#    number_of_screen_initiations = 60, scrn_scaling_grid = np.logspace(-2, -0.5, 5), model_type = 'PixelWiseStrehlModel' ) #lin_comb') 

# or read one in  
strehl_model_file = '/home/benja/Documents/BALDR/BaldrApp/configurations/strehl_model_config-BALDR_UT_J3_2024-10-19T09.28.27.pkl'
strehl_model = load_model_from_pickle(filename=strehl_model_file)

###
### FITTING LINEAR ZONAL MODEL FROM APPLYING KOLMOGOROV PHASE SCREENS TO DM COMMANDS
#### 

telemetry = {
    'I0':[I0],
    'I0_dm':[I0_dm],
    'N0':[N0],
    'N0_dm':[N0_dm],
    'b0':[b0],
    'b0_dm':[b0_dm],
    'dm_cmd':[],
    'b':[],
    'b_est':[],
    'b_dm_est':[],
    'i':[],
    'Ic':[],
    'i_dm':[],
    's':[],
    'strehl_0':[],
    'strehl_1':[],
    'strehl_2':[],
    'strehl_2_est':[],

}

telem_ns = SimpleNamespace(**telemetry)

zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy()  
for it in range(100):
    
    print( it )
    
    # roll screen
    for _ in range(10):
        scrn.add_row()
    
    
    zwfs_ns.dm.current_cmd =  util.create_phase_screen_cmd_for_DM( scrn,  scaling_factor=0.2 , drop_indicies = [0, 11, 11 * 12, -1] , plot_cmd=False) 
    
    # first stage AO
    if np.mod(it, 1) == 0: # only update the AO every few iterations to simulate latency 
        _ , reco_1 = bldr.first_stage_ao( scrn, Nmodes_removed , basis  , phase_scaling_factor = phase_scaling_factor, return_reconstructor = True )   
         
    ao_1 =  basis[0] * (phase_scaling_factor * scrn.scrn - reco_1)
    
    # opd after first stage AO
    opd_ao_1 = zwfs_ns.pyZelda.pupil * zwfs_ns.optics.wvl0 / (2*np.pi) * ao_1
    
    # add vibrations OPD
    opd_vibrations = np.zeros( ao_1.shape )
    
    # add BALDR DM OPD 
    opd_current_dm = bldr.get_dm_displacement( command_vector= zwfs_ns.dm.current_cmd   , gain=zwfs_ns.dm.opd_per_cmd, \
                sigma= zwfs_ns.grid.dm_coord.act_sigma_wavesp, X=zwfs_ns.grid.wave_coord.X, Y=zwfs_ns.grid.wave_coord.Y,\
                    x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp )
    
    # sum all opd contributions in the Baldr input pupil plane 
    # set opd_ao_1 = 0 if rolling phasescreen on DM 
    bldr_opd_map = np.sum( [ 0 * opd_ao_1, opd_vibrations, opd_internal, opd_current_dm ] , axis=0)
    
    
    # propagate to the detector plane
    Ic = photon_flux_per_pixel_at_vlti * zwfs_ns.pyZelda.propagate_opd_map( bldr_opd_map , wave = zwfs_ns.optics.wvl0 )
    
    # detect the intensity
    i = bldr.detect( Ic, binning = (zwfs_ns.detector.binning, zwfs_ns.detector.binning), qe=zwfs_ns.detector.qe , dit=zwfs_ns.detector.dit,\
        ron= zwfs_ns.detector.ron, include_shotnoise=True, spectral_bandwidth = zwfs_ns.stellar.bandwidth )

    # estimate the Strehl ratio after baldr (why we use index = 2 : second stage ao)
    Strehl_2_est = strehl_model.apply_model( np.array( [i / np.mean( N0[strehl_model.detector_pupilmask] ) ] ) )
    
    # get the real strehl ratios at various points (for tracking performance) 
    Strehl_0 = np.exp( - np.var( phase_scaling_factor * scrn.scrn[zwfs_ns.pyZelda.pupil>0.5]) ) # atmospheric strehl 
    Strehl_1 = np.exp( - np.var( ao_1[zwfs_ns.pyZelda.pupil>0.5]) ) # strehl after first stage AO 
    Strehl_2 = np.exp( - np.var( ao_1[zwfs_ns.pyZelda.pupil>0.5]) ) # strehl after baldr 
    
    # interpolate signals onto registered actuator grid
    i_dm = DM_registration.interpolate_pixel_intensities(image = i, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])
    
    # get the real optical gain and also estimate the optical gain 
    b, _ = ztools.create_reference_wave_beyond_pupil_with_aberrations(bldr_opd_map, zwfs_ns.pyZelda.mask_diameter, zwfs_ns.pyZelda.mask_depth, zwfs_ns.pyZelda.mask_substrate, zwfs_ns.pyZelda.mask_Fratio,
                                            zwfs_ns.pyZelda.pupil_diameter, zwfs_ns.pyZelda.pupil, zwfs_ns.optics.wvl0, clear=np.array([]), 
                                            sign_mask=np.array([]), cpix=False)
    b_est = np.sqrt( Strehl_2_est ) * b0
    
    b_dm_est = np.sqrt( Strehl_2_est ) * b0_dm
    
    # interpolate signals onto registered actuator grid
    
    # estimate the ZWFS signal 
    s = i_dm / ( b_dm_est * N0_dm) - I0_dm / (b0_dm * N0_dm) 
    
            
    # apply BALDR phase reconstruction 
    # learn a model 
    
    # update BALDR DM command 
    
    
    # get telemetry 
    
    telem_ns.i.append( i )
    telem_ns.Ic.append( Ic )
    telem_ns.i_dm.append(i_dm )
    telem_ns.strehl_0.append( Strehl_0 )
    telem_ns.strehl_1.append( Strehl_1 )
    telem_ns.strehl_2.append( Strehl_2 )
    telem_ns.strehl_2_est.append(Strehl_2_est )
    telem_ns.b.append( b )
    telem_ns.b_est.append( b_est )
    telem_ns.b_dm_est.append( b_dm_est )
    telem_ns.dm_cmd.append( zwfs_ns.dm.current_cmd )


# save fits 
# plot the  interpolated intensity on DM and the DM command
if plot_intermediate_results:
    bldr.save_telemetry( telem_ns , savename = fig_path + f'telem_with_dm_interactuator_coupling-{zwfs_ns.dm.actuator_coupling_factor}.fits', overwrite=True, return_fits = False)




# let have a dybnamic plot of the telemetry
image_lists = [[ util.get_DM_command_in_2D( a ) for a in telem_ns.i_dm], \
    [ util.get_DM_command_in_2D( a ) for a in telem_ns.dm_cmd], \
     telem_ns.Ic] 
util.display_images_with_slider(image_lists = image_lists,\
    plot_titles=['intensity interp dm', 'dm cmd', 'intensity wavespace'], cbar_labels=None)
       
# make a movie
util.display_images_as_movie( image_lists = image_lists,\
    plot_titles=['intensity interp dm', 'dm cmd', 'intensity wavespace'], cbar_labels=None, save_path = fig_path + 'zonal_model_calibration_dm_interactuator_coupling-{zwfs_ns.dm.actuator_coupling_factor}.mp4', fps=5) 
                             
                            
# plot the  interpolated intensity on DM and the DM command
if 1:
    act=65
    plt.figure()
    plt.plot(  np.array( telem_ns.dm_cmd ).T[act], np.array( telem_ns.i_dm ).T[act],'.')
    plt.xlabel('dm cmd')
    plt.ylabel('intensity interp dm')
    if plot_intermediate_results:
        plt.savefig(fig_path + f'dmcmd_vs_dmIntensity_actuator-{act}_dm_interactuator_coupling-{zwfs_ns.dm.actuator_coupling_factor}.png')
    plt.show()

    # look at the correlation between the DM command and the interpolated intensity (Pearson R) 
    R_list = []
    for act in range(140):
        R_list.append( pearsonr([a[act] for a in telem_ns.i_dm ], [a[act] for a in telem_ns.dm_cmd]).statistic )

    plt.figure() 
    plt.imshow( util.get_DM_command_in_2D( R_list ) )
    plt.colorbar(label='Pearson R') 
    plt.title( 'Pearson R between DM command and \ninterpolated intensity onto DM actuator space')
    #plt.savefig(fig_path + f'pearson_r_dmcmd_dmIntensity_dm_interactuator_coupling-{zwfs_ns.dm.actuator_coupling_factor}.png')
    plt.show()  


act_filt = np.array( R_list ) > 0.65
util.nice_heatmap_subplots( [util.get_DM_command_in_2D(act_filt) ] )
plt.show()

# define X, Y for fitting model to go from interpolated intensity to DM command
X = np.array( telem_ns.i_dm  ) #[act_filt]
#X = ( np.array( telem_ns.i_dm  )/ np.array( telem_ns.N0_dm ) ).T[act_filt]
#X = ( np.array( telem_ns.i_dm  )/ np.array( telem_ns.b_dm_est ) - np.array( telem_ns.I0_dm  )/np.array( telem_ns.b0_dm  ) ).T[act_filt]
Y = np.array( telem_ns.dm_cmd ) #[act_filt]

#B = multivariate_polynomial_fit( X = X.T , Y = Y.T , model="first" , train_split=0.9)



# Assuming telem_ns contains the necessary data
X = np.array(telem_ns.i_dm)  # Input features (samples x features)
Y = np.array(telem_ns.dm_cmd)  # Target values (samples x features)

# Initialize the linear fit model
model_1 = my_lin_fit(model_name='pixelwise_first')

# Fit the model to X and Y
model_1.fit(X=X, Y=Y)

# Apply the model to make predictions
Y_pred = model_1.apply(X)

# Select an actuator/feature to plot
act = 65  # Example actuator/feature index

# Plot the true values vs. the model predictions for the selected feature
plt.plot(X[:, act], Y_pred[:, act], '.', label='Model Prediction')
plt.plot(X[:, act], Y[:, act], '.', label='True Data')
plt.legend()
plt.show()



###
### OPEN LOOP SIMULATION
#### 
# Compare reconstructors between linear zonal model and linear modal model 

# for eigenmodes just add proportional gain to the model at unity
zwfs_ns.ctrl.HO_ctrl.kp = np.ones( zwfs_ns.ctrl.HO_ctrl.kp.shape ) 
zwfs_ns.ctrl.TT_ctrl.kp = np.ones( zwfs_ns.ctrl.TT_ctrl.kp.shape ) 
# try simple static reconstruction 
zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy() 
dm_cmd_est = np.zeros( 140 )
phase_scaling_factor = 0.1

if 1:
    it = 0    
    print( it )
    
    # roll screen
    for _ in range(10):
        scrn.add_row()
    
    # first stage AO
    if np.mod(it, 1) == 0: # only update the AO every few iterations to simulate latency 
        _ , reco_1 = bldr.first_stage_ao( scrn, Nmodes_removed , basis  , phase_scaling_factor = phase_scaling_factor, return_reconstructor = True )   
         
    ao_1 =  basis[0] * (phase_scaling_factor * scrn.scrn - reco_1)
    
    # opd after first stage AO
    opd_ao_1 = zwfs_ns.pyZelda.pupil * zwfs_ns.optics.wvl0 / (2*np.pi) * ao_1
    
    # add vibrations OPD
    opd_vibrations = np.zeros( ao_1.shape )
    
    # add BALDR DM OPD 
    opd_current_dm = bldr.get_dm_displacement( command_vector= zwfs_ns.dm.current_cmd   , gain=zwfs_ns.dm.opd_per_cmd, \
                sigma= zwfs_ns.grid.dm_coord.act_sigma_wavesp, X=zwfs_ns.grid.wave_coord.X, Y=zwfs_ns.grid.wave_coord.Y,\
                    x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp )
    
    # sum all opd contributions in the Baldr input pupil plane 
    # set opd_ao_1 = 0 if rolling phasescreen on DM 
    bldr_opd_map = np.sum( [  opd_ao_1, opd_vibrations, opd_internal, opd_current_dm ] , axis=0 )
    bldr_opd_map-= np.mean( bldr_opd_map[zwfs_ns.pyZelda.pupil>0.5] ) # remove piston  
    
    ao_2 = zwfs_ns.pyZelda.pupil * (2*np.pi) / zwfs_ns.optics.wvl0  *  bldr_opd_map # phase radians 
    
    # get the real strehl ratios at various points (for tracking performance) 
    Strehl_0 = np.exp( - np.var( phase_scaling_factor * scrn.scrn[zwfs_ns.pyZelda.pupil>0.5]) ) # atmospheric strehl 
    Strehl_1 = np.exp( - np.var( ao_1[zwfs_ns.pyZelda.pupil>0.5]) ) # strehl after first stage AO 
    Strehl_2 = np.exp( - np.var( ao_2[zwfs_ns.pyZelda.pupil>0.5]) ) # strehl after baldr     

    
    # propagate to the detector plane
    Ic = photon_flux_per_pixel_at_vlti * zwfs_ns.pyZelda.propagate_opd_map( bldr_opd_map , wave = zwfs_ns.optics.wvl0 )
    
    # detect the intensity
    i = bldr.detect( Ic, binning = (zwfs_ns.detector.binning, zwfs_ns.detector.binning), qe=zwfs_ns.detector.qe , dit=zwfs_ns.detector.dit,\
        ron= zwfs_ns.detector.ron, include_shotnoise=True, spectral_bandwidth = zwfs_ns.stellar.bandwidth )

    
    # interpolate signals onto registered actuator grid
    i_dm = DM_registration.interpolate_pixel_intensities(image = i, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])


    dm_cmd_est[act_filt] =  model_1.apply(i_dm)[act_filt]


    sig = bldr.process_zwfs_signal( i, I0, zwfs_ns.pupil_regions.pupil_filt ) # I0_theory/ np.mean(I0_theory) #

    e_TT = zwfs_ns.reco.I2M_TT @ sig

    u_TT = zwfs_ns.ctrl.TT_ctrl.process( e_TT )

    c_TT = zwfs_ns.reco.M2C_TT @ u_TT 

    e_HO = zwfs_ns.reco.I2M_HO @ sig

    u_HO = zwfs_ns.ctrl.HO_ctrl.process( e_HO )

    c_HO = zwfs_ns.reco.M2C_HO @ u_HO 

    # using zonal model
    opd_current_dm = bldr.get_dm_displacement( command_vector= zwfs_ns.dm.current_cmd +  dm_cmd_est  , gain=zwfs_ns.dm.opd_per_cmd, \
                sigma= zwfs_ns.grid.dm_coord.act_sigma_wavesp, X=zwfs_ns.grid.wave_coord.X, Y=zwfs_ns.grid.wave_coord.Y,\
                    x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp )
    
    # using eigenmode model
    opd_current_dm_1 = bldr.get_dm_displacement( command_vector= c_HO + c_HO  , gain=zwfs_ns.dm.opd_per_cmd, \
                sigma= zwfs_ns.grid.dm_coord.act_sigma_wavesp, X=zwfs_ns.grid.wave_coord.X, Y=zwfs_ns.grid.wave_coord.Y,\
                    x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp )
    
    opd_current_dm -= np.mean( opd_current_dm[zwfs_ns.pyZelda.pupil>0.5] ) # remove piston
    opd_current_dm_1 -= np.mean( opd_current_dm[zwfs_ns.pyZelda.pupil>0.5] ) # remove piston
    
util.nice_heatmap_subplots( [opd_ao_1, Ic, util.get_DM_command_in_2D( i_dm ), util.get_DM_command_in_2D( dm_cmd_est ), zwfs_ns.pyZelda.pupil *( opd_ao_1 - opd_current_dm ) ]  )
plt.show( )

std_before = np.std( ( opd_ao_1 )[zwfs_ns.pyZelda.pupil>0.5] )
std_after = np.std( ( opd_ao_1 - opd_current_dm )[zwfs_ns.pyZelda.pupil>0.5] )
std_after_1 = np.std( ( opd_ao_1 - opd_current_dm_1 )[zwfs_ns.pyZelda.pupil>0.5] )
                     
                    
print( f'rmse before = {round( 1e9 * std_before )}nm,\n rmse after = {round(1e9*std_after)}nm')
print( f'strehl before = {np.exp(- (2*np.pi/ zwfs_ns.optics.wvl0 * std_before)**2)},\n strehl after = {np.exp(-(2*np.pi/ zwfs_ns.optics.wvl0 *std_after)**2)}')
print( f'WITH EIGENMODE strehl before = {np.exp(- (2*np.pi/ zwfs_ns.optics.wvl0 * std_before)**2)},\n strehl after = {np.exp(-(2*np.pi/ zwfs_ns.optics.wvl0 *std_after_1)**2)}')




###
### CLOSED LOOP SIMULATION
#### 
opd_input = 0.1* zwfs_ns.pyZelda.pupil * zwfs_ns.optics.wvl0 / (2*np.pi) *  (basis[5] + basis[10])
amp_input = photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns.pyZelda.pupil
dm_disturbance = np.zeros( 140 )

zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat + dm_disturbance
zwfs_ns = bldr.reset_telemetry( zwfs_ns ) # initialize telemetry to empty list 
zwfs_ns.ctrl.TT_ctrl.reset()
zwfs_ns.ctrl.HO_ctrl.reset()
zwfs_ns.ctrl.TT_ctrl.set_all_gains_to_zero()
zwfs_ns.ctrl.HO_ctrl.set_all_gains_to_zero()

close_after = 10

kpTT = 1
ki_grid = np.linspace(0,0.9,15)
for cnt, kiTT in enumerate( [0.9]) :
    zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat + dm_disturbance
    zwfs_ns = bldr.reset_telemetry( zwfs_ns )
    zwfs_ns.ctrl.TT_ctrl.reset()
    zwfs_ns.ctrl.TT_ctrl.ki = 0 * np.zeros( len(zwfs_ns.ctrl.TT_ctrl.ki) )
    zwfs_ns.ctrl.TT_ctrl.kp = 0 * np.ones( len(zwfs_ns.ctrl.TT_ctrl.kp) )
    for i in range(100):
        print(f'iteration {i}')
        if i > close_after : 
            zwfs_ns.ctrl.HO_ctrl.ki = 0.95 * np.ones( len(zwfs_ns.ctrl.HO_ctrl.ki) )
            zwfs_ns.ctrl.HO_ctrl.kp = 0.2 * np.ones( len(zwfs_ns.ctrl.HO_ctrl.kp) )

            zwfs_ns.ctrl.TT_ctrl.kp = kpTT * np.ones( len(zwfs_ns.ctrl.TT_ctrl.kp) )
            zwfs_ns.ctrl.TT_ctrl.ki = kiTT * np.ones( len(zwfs_ns.ctrl.TT_ctrl.ki) )
            
        bldr.AO_iteration( opd_input, amp_input, opd_internal, zwfs_ns.reco.I0,  zwfs_ns, dm_disturbance, record_telemetry=True ,detector=zwfs_ns.detector)
    
    _ = bldr.save_telemetry( zwfs_ns, savename=fig_path + f'SIM_CL_TT_kiTT-{kiTT}_kpTT-{1}_{tstamp}.fits' )
    # Generate some data


i = -1
#im_dm_dist = np.array( [util.get_DM_command_in_2D( a ) for a in zwfs_ns.telem.dm_disturb_list] )
im_phase = np.array( zwfs_ns.telem.field_phase ) 
im_int = np.array( zwfs_ns.telem.i_list  ) 
im_cmd = np.array( [util.get_DM_command_in_2D( a ) for a in (np.array(zwfs_ns.telem.c_TT_list) + np.array(zwfs_ns.telem.c_HO_list)  ) ] )


#line_x = np.linspace(0, i, i)
# line_eHO = np.array( zwfs_ns.telem.e_HO_list ) [:i]
# line_eTT = np.array( zwfs_ns.telem.e_TT_list )[:i]
# line_S = np.array( zwfs_ns.telem.strehl )[:i]
# line_rmse = np.array( zwfs_ns.telem.rmse_list )[:i]
line_eHO = np.array( zwfs_ns.telem.e_HO_list ) 
line_eTT = np.array( zwfs_ns.telem.e_TT_list )
line_S = np.array( zwfs_ns.telem.strehl )
line_rmse = np.array( zwfs_ns.telem.rmse_list )

# Define plot data
#image_list =  [im_phase[-1], im_phase[-1], im_int[-1], im_cmd[-1]]
image_list =  [[opd_input for _ in im_phase], im_phase, im_int, im_cmd]
image_title_list =  ['DM disturbance', 'input phase', 'intensity', 'reco. command']
image_colorbar_list = ['DM units', 'radians', 'adu', 'DM units']

plot_list = [ line_eHO, line_eTT, line_S, line_rmse ] 
plot_ylabel_list = ['e_HO', 'e_TT', 'Strehl', 'rmse']
plot_xlabel_list = ['iteration' for _ in plot_list]
plot_title_list = ['' for _ in plot_list]

#vlims = [(0, 1), (0, 1), (0, 1)]  # Set vmin and vmax for each image

util.create_telem_mosaic([a[-1] for a in image_list], image_title_list, image_colorbar_list, 
                plot_list, plot_title_list, plot_xlabel_list, plot_ylabel_list)

util.display_images_with_slider(image_lists = image_list)
       








#create_telem_mosaic(image_list=[], image_title_list, image_colorbar_list, \
#    plot_list, plot_title_list, plot_xlabel_list, plot_ylabel_list)




# try reconstruction on sky 


telemetry_2 = {
    
    'I0':[I0],
    'I0_dm':[I0_dm],
    'N0':[N0],
    'N0_dm':[N0_dm],
    'b0':[b0],
    'b0_dm':[b0_dm],
    'dm_cmd':[],
    'ao_0':[],
    'ao_1':[],
    'ao_2':[],
    'b':[],
    'b_est':[],
    'b_dm_est':[],
    'i':[],
    'Ic':[],
    'i_dm':[],
    's':[],
    'strehl_0':[],
    'strehl_1':[],
    'strehl_2':[],
    'strehl_2_est':[],
}

telem_ns_2 = SimpleNamespace(**telemetry_2)
zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy() 
phase_scaling_factor = 0.2
close_after = 5
dm_cmd_est = np.zeros( zwfs_ns.dm.dm_flat.shape )
for it in range(10):

    print( it )
    
    # roll screen
    for _ in range(10):
        scrn.add_row()
    
    # first stage AO
    if np.mod(it, 1) == 0: # only update the AO every few iterations to simulate latency 
        _ , reco_1 = bldr.first_stage_ao( scrn, Nmodes_removed , basis  , phase_scaling_factor = phase_scaling_factor, return_reconstructor = True )   
         
    ao_1 =  basis[0] * (phase_scaling_factor * scrn.scrn - reco_1)
    
    # opd after first stage AO
    opd_ao_1 = zwfs_ns.pyZelda.pupil * zwfs_ns.optics.wvl0 / (2*np.pi) * ao_1
    
    # add vibrations OPD
    opd_vibrations = np.zeros( ao_1.shape )
    
    # add BALDR DM OPD 
    opd_current_dm = bldr.get_dm_displacement( command_vector= zwfs_ns.dm.current_cmd   , gain=zwfs_ns.dm.opd_per_cmd, \
                sigma= zwfs_ns.grid.dm_coord.act_sigma_wavesp, X=zwfs_ns.grid.wave_coord.X, Y=zwfs_ns.grid.wave_coord.Y,\
                    x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp )
    
    # sum all opd contributions in the Baldr input pupil plane 
    # set opd_ao_1 = 0 if rolling phasescreen on DM 
    bldr_opd_map = np.sum( [  opd_ao_1, opd_vibrations, opd_internal, opd_current_dm ] , axis=0 )
    bldr_opd_map-= np.mean( bldr_opd_map[zwfs_ns.pyZelda.pupil>0.5] ) # remove piston  
    
    ao_2 = zwfs_ns.pyZelda.pupil * (2*np.pi) / zwfs_ns.optics.wvl0  *  bldr_opd_map # phase radians 
    
    # get the real strehl ratios at various points (for tracking performance) 
    Strehl_0 = np.exp( - np.var( phase_scaling_factor * scrn.scrn[zwfs_ns.pyZelda.pupil>0.5]) ) # atmospheric strehl 
    Strehl_1 = np.exp( - np.var( ao_1[zwfs_ns.pyZelda.pupil>0.5]) ) # strehl after first stage AO 
    Strehl_2 = np.exp( - np.var( ao_2[zwfs_ns.pyZelda.pupil>0.5]) ) # strehl after baldr     

    
    # propagate to the detector plane
    Ic = photon_flux_per_pixel_at_vlti * zwfs_ns.pyZelda.propagate_opd_map( bldr_opd_map , wave = zwfs_ns.optics.wvl0 )
    
    # detect the intensity
    i = bldr.detect( Ic, binning = (zwfs_ns.detector.binning, zwfs_ns.detector.binning), qe=zwfs_ns.detector.qe , dit=zwfs_ns.detector.dit,\
        ron= zwfs_ns.detector.ron, include_shotnoise=True, spectral_bandwidth = zwfs_ns.stellar.bandwidth )

    
    # interpolate signals onto registered actuator grid
    i_dm = DM_registration.interpolate_pixel_intensities(image = i, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])


    dm_cmd_est[act_filt] = model_1.apply(i_dm)[act_filt]  #B[0] + i_dm[act_filt] @ B[1]   
    print( np.std(zwfs_ns.dm.current_cmd))
    
    ##### UPDATE DM COMMAND ##### OPEN LOOP
    if it > close_after:
        print('here')
        zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat - dm_cmd_est 

    plt.figure() ; plt.imshow( util.get_DM_command_in_2D( dm_cmd_est ) ) ; plt.colorbar() ; plt.show()

    # get telemetry 
    telem_ns_2.ao_0.append( phase_scaling_factor * scrn.scrn )
    telem_ns_2.ao_1.append( ao_1 )
    telem_ns_2.ao_2.append( ao_2 )
    telem_ns_2.i.append( i )
    telem_ns_2.Ic.append( Ic )
    telem_ns_2.i_dm.append(i_dm )
    telem_ns_2.strehl_0.append( Strehl_0 )
    telem_ns_2.strehl_1.append( Strehl_1 )
    telem_ns_2.strehl_2.append( Strehl_2 )
    telem_ns_2.strehl_2_est.append(Strehl_2_est )
    telem_ns_2.b.append( b )
    telem_ns_2.b_est.append( b_est )
    telem_ns_2.b_dm_est.append( b_dm_est )
    telem_ns_2.dm_cmd.append( dm_cmd_est )

# let have a dybnamic plot of the telemetry
image_lists = [telem_ns_2.ao_0,\
        telem_ns_2.ao_1,\
        telem_ns_2.ao_2,\
    [ util.get_DM_command_in_2D( a ) for a in telem_ns_2.i_dm], \
    [ util.get_DM_command_in_2D( a ) for a in telem_ns_2.dm_cmd]] 
util.display_images_with_slider(image_lists = image_lists,\
    plot_titles=['phase atm','phase first stage ao','phase second stage ao','intensity interp dm', 'dm cmd'], cbar_labels=None)
       

plt.figure()
plt.plot(telem_ns_2.strehl_0, label='strehl_0')  
plt.plot(telem_ns_2.strehl_1, label='strehl_1')  
plt.plot(telem_ns_2.strehl_2, label='strehl_2')  
plt.ylabel('Strehl Ratio')
plt.xlabel('Iteration')
plt.legend(loc='best')
plt.show()

import numpy as np
from sklearn.model_selection import train_test_split

def multivariate_polynomial_fit(X, Y, model="first", train_split=0.6, plot_results=True):
    """
    Fits a multivariate first- or second-order polynomial to the data, and returns separate coefficients.
    The data is split into training and testing sets. Optionally plots the results.
    
    Parameters:
    - X: np.ndarray, shape (N, P), input data matrix
    - Y: np.ndarray, shape (N, P), output data matrix
    - model: str, "first" for linear fit, "second" for quadratic fit
    - train_split: float, fraction of data to be used for training (default=0.6)
    - plot_results: bool, if True plots the model vs measured for train and test sets with residuals
    
    Returns:
    - intercept: np.ndarray, shape (P,), intercept terms
    - linear_coeff: np.ndarray, shape (P, P), linear coefficients
    - quadratic_coeff: np.ndarray (only for second-order), shape (P', P), quadratic coefficients (squared and interaction terms)
    """
    # Ensure X and Y have the same number of rows (observations)
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of observations (N)."
    
    N, P = X.shape

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_split, random_state=42)

    # Add a column of ones to X_train and X_test to include the intercept term
    X_train_augmented = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test_augmented = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

    if model == "first":
        # First-order (linear) fit
        B = np.linalg.inv(X_train_augmented.T @ X_train_augmented) @ X_train_augmented.T @ Y_train

        # Separate the intercept and linear coefficients
        intercept = B[0, :]  # First row is the intercept
        linear_coeff = B[1:, :]  # Remaining rows are the linear coefficients

        # Predict on both training and test sets
        Y_train_pred = X_train_augmented @ B
        Y_test_pred = X_test_augmented @ B

    elif model == "second": # over fits - tomany parameters in quadratic coeffients
        
        # Second-order (quadratic) fit
        Z_train = augment_with_quadratic_terms(X_train)
        Z_test = augment_with_quadratic_terms(X_test)

        # Add a column of ones to Z_train and Z_test to include the intercept term
        Z_train_augmented = np.hstack([np.ones((Z_train.shape[0], 1)), Z_train])
        Z_test_augmented = np.hstack([np.ones((Z_test.shape[0], 1)), Z_test])

        # Solve for Theta
        Theta = np.linalg.inv(Z_train_augmented.T @ Z_train_augmented) @ Z_train_augmented.T @ Y_train

        # Separate the intercept, linear, and quadratic coefficients
        intercept = Theta[0, :]  # First row is the intercept
        linear_coeff = Theta[1:P+1, :]  # Next P rows are the linear coefficients
        quadratic_coeff = Theta[P+1:, :]  # Remaining rows are the quadratic coefficients

        # Predict on both training and test sets
        Y_train_pred = Z_train_augmented @ Theta
        Y_test_pred = Z_test_augmented @ Theta

    else:
        raise ValueError("Model type must be 'first' or 'second'.")

    # Plot the results if requested
    if plot_results:
        util.plot_data_and_residuals(
            X_train, Y_train, Y_train_pred, xlabel='X (train)', ylabel='Y (train)', 
            residual_ylabel='Residual (train)', label_1=None, label_2=None
        )
        
        util.plot_data_and_residuals(
            X_test, Y_test, Y_test_pred, xlabel='X (test)', ylabel='Y (test)', 
            residual_ylabel='Residual (test)', label_1=None, label_2=None
        )

    if model == "first":
        return intercept, linear_coeff
    elif model == "second":
        return intercept, linear_coeff, quadratic_coeff











#fir Y=M@X
M = np.linalg.lstsq(np.array( telem_ns.i_dm ).T[filt].T , np.array( telem_ns.dm_cmd ).T[filt].T, rcond=None)[0]

plt.figure(); plt.plot( M @ (np.array( telem_ns.i_dm ).T[filt] ), np.array( telem_ns.dm_cmd ).T[filt], '.'); plt.show()

act = 20
plt.figure(); plt.plot( (M @ (np.array( telem_ns.i_dm ).T[filt]) )[act], np.array( telem_ns.dm_cmd ).T[filt][act], '.'); plt.show()


filt = np.var( telem_ns.i_dm , axis= 0 ) > 180000 
plt.imshow( util.get_DM_command_in_2D( filt ) ) ; plt.colorbar() ; plt.show()



   
ii = [ ]

for i , cmd in zip(telem_ns.i_dm, telem_ns.dm_cmd):
    
pearsonr(pixel_intensity_series, strehl_ratios)

plt.plot( telem_ns.strehl_2, telem_ns.strehl_2_est, 'o')


    
i_fft = np.fft.fftshift( np.fft.fft2( telem_ns.i[0] ) )

# def calibrate strehl model    

# input 
# zwfs_ns 
# train_fraction = 0.6
# correlation threshold = 0.9
# save_results_path = None
