
import numpy as np #(version 2.1.1 works but incompatiple with numba)
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import pearsonr
import pickle
from types import SimpleNamespace
from sklearn.linear_model import LinearRegression
import importlib # reimport package after edits: importlib.reload(bldr)
import os
import datetime
import scipy.interpolate as interpolate
# from courtney-barrer's fork of pyzelda
import pyzelda.zelda as zelda
import pyzelda.ztools as ztools
import pyzelda.utils.aperture as aperture
import pyzelda.utils.imutils as imutils
from baldrapp.common import phasescreens as ps
from baldrapp.common import utilities as util
from baldrapp.common import baldr_core as bldr
from baldrapp.common import DM_registration
from baldrapp.common import DM_basis

from baldrapp.common.baldr_core import StrehlModel 

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

proj_path =  os.getcwd() #'/home/benja/Documents/BALDR/BaldrApp/' #'/home/rtc/Documents/BaldrApp/'

# initialize our ZWFS instrument
wvl0=1.25e-6
config_ini = proj_path  + '/baldrapp/configurations/BALDR_UT_J3.ini'#'/home/benja/Documents/BALDR/BaldrApp/configurations/BALDR_UT_J3.ini'
zwfs_ns = bldr.init_zwfs_from_config_ini( config_ini=config_ini , wvl0=wvl0)

fig_path = '/Users/bencb/Downloads/'#f'/home/benja/Downloads/act_cross_coupling_{zwfs_ns.dm.actuator_coupling_factor}_{tstamp}/' #f'/home/rtc/Documents/act_cross_coupling_{zwfs_ns.dm.actuator_coupling_factor}_{tstamp}/'
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

# from IM register the DM in the detector pixelspace 
zwfs_ns = bldr.register_DM_in_pixelspace_from_IM( zwfs_ns, plot_intermediate_results=True  )


# build control matrices from IM 
zwfs_ns = bldr.construct_ctrl_matricies_from_IM( zwfs_ns,  method = 'Eigen_TT-HO', Smax = 60, TT_vectors = DM_basis.get_tip_tilt_vectors() )

# or we create fit a linear zonal model to the IM
zwfs_ns = bldr.fit_linear_zonal_model( zwfs_ns, opd_internal, iterations = 100, photon_flux_per_pixel_at_vlti = photon_flux_per_pixel_at_vlti , \
    pearson_R_threshold = 0.6, phase_scaling_factor=0.2,   plot_intermediate=True , fig_path = None)


# example to interpolate any detected image onto the DM actuator grid
#DM_registration.interpolate_pixel_intensities(image = I0, pixel_coords = zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space)

# add controllers 
zwfs_ns = bldr.add_controllers_for_MVM_TT_HO( zwfs_ns, TT = 'PID', HO = 'leaky')
#zwfs_ns = bldr.add_controllers_for_MVM_TT_HO( zwfs_ns, TT = 'PID', HO = 'leaky')

if plot_intermediate_results:
    bldr.plot_eigenmodes( zwfs_ns , descr_label = f'dm_interactuator_coupling-{zwfs_ns.dm.actuator_coupling_factor}', save_path = fig_path )

### 
# DM registration 
###

# # get inner corners for estiamting DM center in pixel space (have to deal seperately with pinned actuator basis)
# if zwfs_ns.reco.IM.shape[0] == 100: # outer actuators are pinned, 
#     corner_indicies = DM_registration.get_inner_square_indices(outer_size=10, inner_offset=3, without_outer_corners=False)
    
# elif zwfs_ns.reco.IM.shape[0] == 140: # outer acrtuators are free 
#     print(140)
#     corner_indicies = DM_registration.get_inner_square_indices(outer_size=12, inner_offset=4, without_outer_corners=True)
# else:
#     print("CASE NOT MATCHED  d['I2M'].data.shape = { d['I2M'].data.shape}")
    
# img_4_corners = []
# dm_4_corners = []
# for i in corner_indicies:
#     dm_4_corners.append( np.where( M2C_0[i] )[0][0] )
#     #dm2px.get_DM_command_in_2D( d['M2C'].data[:,i]  # if you want to plot it 

#     tmp = np.zeros( zwfs_ns.pupil_regions.pupil_filt.shape )
#     tmp.reshape(-1)[zwfs_ns.pupil_regions.pupil_filt.reshape(-1)] = zwfs_ns.reco.IM[i] 

#     #plt.imshow( tmp ); plt.show()
#     img_4_corners.append( abs(tmp ) )

# #plt.imshow( np.sum( tosee, axis=0 ) ); plt.show()

# # dm_4_corners should be an array of length 4 corresponding to the actuator index in the (flattened) DM command space
# # img_4_corners should be an array of length 4xNxM where NxM are the image dimensions.
# # !!! It is very important that img_4_corners are registered in the same order as dm_4_corners !!!
# transform_dict = DM_registration.calibrate_transform_between_DM_and_image( dm_4_corners, img_4_corners, debug=plot_intermediate_results, fig_path = None )

# interpolated_intensities = DM_registration.interpolate_pixel_intensities(image = I0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])


# interpolate these fields onto the registered actuator grid
b0_dm = DM_registration.interpolate_pixel_intensities(image = b0, pixel_coords = zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space) #DM_registration.interpolate_pixel_intensities(image = I0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])
I0_dm = DM_registration.interpolate_pixel_intensities(image = I0, pixel_coords = zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space) #DM_registration.interpolate_pixel_intensities(image = b0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])
N0_dm = DM_registration.interpolate_pixel_intensities(image = N0, pixel_coords = zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space) #DM_registration.interpolate_pixel_intensities(image = N0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])

# calibrate a model to map a subset of pixel intensities to Strehl Ratio 
# should eventually come back and debug for model_type = lin_comb - since it seemed to work better intially
#strehl_model = bldr.calibrate_strehl_model( zwfs_ns, save_results_path = fig_path, train_fraction = 0.6, correlation_threshold = 0.6, \
#    number_of_screen_initiations = 60, scrn_scaling_grid = np.logspace(-2, -0.5, 5), model_type = 'PixelWiseStrehlModel' ) #lin_comb') 

# or read one in  
strehl_model_file = proj_path  + '/baldrapp/configurations/strehl_model_config-BALDR_UT_J3_2024-10-19T09.28.27.pkl'
strehl_model = load_model_from_pickle(filename=strehl_model_file)



###
### OPEN LOOP SIMULATION
#### 
# Compare reconstructors between linear zonal model and linear modal model 

# for eigenmodes just add proportional gain to the model at unity
# zwfs_ns.ctrl.HO_ctrl.kp = np.ones( zwfs_ns.ctrl.HO_ctrl.kp.shape ) 
# zwfs_ns.ctrl.TT_ctrl.kp = np.ones( zwfs_ns.ctrl.TT_ctrl.kp.shape ) 
# zwfs_ns.ctrl.zonal_ctrl.kp = np.ones( zwfs_ns.ctrl.TT_ctrl.kp.shape ) 
zonal_ctrl_dict = bldr.add_controllers_for_zonal_interp_no_projection( zwfs_ns ,  HO = 'PID' , return_controller =True)
MVM_TT_HO_ctrl_dict = bldr.add_controllers_for_MVM_TT_HO( zwfs_ns ,  TT='PID', HO = 'PID' , return_controller =True)

for k,v in MVM_TT_HO_ctrl_dict.items():
    v.kp = np.ones( v.kp.shape )
    
for k,v in zonal_ctrl_dict.items():
    v.kp = np.ones( v.kp.shape )
    
# try simple static reconstruction 
zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy() 
dm_cmd_est = np.zeros( 140 )
phase_scaling_factor = 0.1

N0_dm = DM_registration.interpolate_pixel_intensities(image = N0, pixel_coords = zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space) #DM_registration.interpolate_pixel_intensities(image = N0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])

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


    ### ZONAL MODEL

    kwargs = {"N0_dm":N0_dm, "HO_ctrl": zonal_ctrl_dict['HO_ctrl']  } 
    delta_cmd = bldr.process_zwfs_intensity( i, zwfs_ns, method = 'zonal_interp_no_projection', record_telemetry = True , **kwargs )
    # # interpolate signals onto registered actuator grid
    #i_dm = DM_registration.interpolate_pixel_intensities(image = i, pixel_coords = zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space)

    # #dm_cmd_est[act_filt] =  model_1.apply(i_dm/ np.mean(N0[ N0 > np.mean( N0 ) ] ))[act_filt]
    # act_filt = zwfs_ns.reco.linear_zonal_model.act_filt_recommended 
    # sig = zwfs_ns.reco.linear_zonal_model.process_signal( i_dm, N0_dm, act_filt)
    # dm_cmd_est[act_filt] = zwfs_ns.reco.linear_zonal_model.apply(sig)[act_filt]

    ### EIGEN MODEL
    kwargs = {"I0":I0 , "HO_ctrl": MVM_TT_HO_ctrl_dict['HO_ctrl'], "TT_ctrl": MVM_TT_HO_ctrl_dict['TT_ctrl'] }
    delta_cmd_1 = bldr.process_zwfs_intensity( i, zwfs_ns, method = 'MVM-TT-HO', record_telemetry = True , **kwargs )
    # # now doing the same with the eigenmode model
    # sig = bldr.process_zwfs_signal( i, I0, zwfs_ns.pupil_regions.pupil_filt ) # I0_theory/ np.mean(I0_theory) #

    # e_TT = zwfs_ns.reco.I2M_TT @ sig

    # u_TT = zwfs_ns.ctrl.TT_ctrl.process( e_TT )

    # c_TT = zwfs_ns.reco.M2C_TT @ u_TT 

    # e_HO = zwfs_ns.reco.I2M_HO @ sig

    # u_HO = zwfs_ns.ctrl.HO_ctrl.process( e_HO )

    # c_HO = zwfs_ns.reco.M2C_HO @ u_HO 


    # using zonal model
    opd_current_dm = bldr.get_dm_displacement( command_vector= delta_cmd  , gain=zwfs_ns.dm.opd_per_cmd, \
                sigma= zwfs_ns.grid.dm_coord.act_sigma_wavesp, X=zwfs_ns.grid.wave_coord.X, Y=zwfs_ns.grid.wave_coord.Y,\
                    x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp )
    
    # using eigenmode model
    opd_current_dm_1 = bldr.get_dm_displacement( command_vector= delta_cmd_1 , gain=zwfs_ns.dm.opd_per_cmd, \
                sigma= zwfs_ns.grid.dm_coord.act_sigma_wavesp, X=zwfs_ns.grid.wave_coord.X, Y=zwfs_ns.grid.wave_coord.Y,\
                    x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp )
    # # using zonal model
    # opd_current_dm = bldr.get_dm_displacement( command_vector= zwfs_ns.dm.current_cmd +  dm_cmd_est  , gain=zwfs_ns.dm.opd_per_cmd, \
    #             sigma= zwfs_ns.grid.dm_coord.act_sigma_wavesp, X=zwfs_ns.grid.wave_coord.X, Y=zwfs_ns.grid.wave_coord.Y,\
    #                 x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp )
    
    # # using eigenmode model
    # opd_current_dm_1 = bldr.get_dm_displacement( command_vector= c_HO + c_HO  , gain=zwfs_ns.dm.opd_per_cmd, \
    #             sigma= zwfs_ns.grid.dm_coord.act_sigma_wavesp, X=zwfs_ns.grid.wave_coord.X, Y=zwfs_ns.grid.wave_coord.Y,\
    #                 x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp )
    
    opd_current_dm -= np.mean( opd_current_dm[zwfs_ns.pyZelda.pupil>0.5] ) # remove piston
    opd_current_dm_1 -= np.mean( opd_current_dm[zwfs_ns.pyZelda.pupil>0.5] ) # remove piston
    
#util.nice_heatmap_subplots( [opd_ao_1, Ic, util.get_DM_command_in_2D( i_dm ), util.get_DM_command_in_2D( dm_cmd_est ), zwfs_ns.pyZelda.pupil *( opd_ao_1 - opd_current_dm ) ]  )
#plt.show( )

std_before = np.std( ( opd_ao_1 )[zwfs_ns.pyZelda.pupil>0.5] )
std_after = np.std( ( opd_ao_1 - opd_current_dm )[zwfs_ns.pyZelda.pupil>0.5] )
std_after_1 = np.std( ( opd_ao_1 - opd_current_dm_1 )[zwfs_ns.pyZelda.pupil>0.5] )
                     
                    
print( f'rmse before = {round( 1e9 * std_before )}nm,\n rmse after = {round(1e9*std_after)}nm')
print( f'strehl before = {np.exp(- (2*np.pi/ zwfs_ns.optics.wvl0 * std_before)**2)},\n strehl after = {np.exp(-(2*np.pi/ zwfs_ns.optics.wvl0 *std_after)**2)}')
print( f'WITH EIGENMODE strehl before = {np.exp(- (2*np.pi/ zwfs_ns.optics.wvl0 * std_before)**2)},\n strehl after = {np.exp(-(2*np.pi/ zwfs_ns.optics.wvl0 *std_after_1)**2)}')



###
### TRY OPTIMIZE GAINS
#### 
from scipy.signal import welch, TransferFunction, bode, csd
from scipy.optimize import minimize

# get open loop data (get timestamps too)
OL_data = bldr.roll_screen_on_dm( zwfs_ns=zwfs_ns,  Nmodes_removed=14, ph_scale = 0.2, \
    actuators_per_iteration = 0.5, number_of_screen_initiations= 1000, opd_internal=opd_internal)


# project to modes 
zonal_ctrl_dict = bldr.add_controllers_for_zonal_interp_no_projection( zwfs_ns ,  HO = 'PID' , return_controller =True) # HO = 'leaky'
zonal_ctrl_dict['HO_ctrl'].kp = np.ones( len( zonal_ctrl_dict['HO_ctrl'].kp ) ) # set kp to one (ki, kd to zero) so u_HO = e_HO.
N0_dm = DM_registration.interpolate_pixel_intensities(image = OL_data.N0[0], pixel_coords = zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space) #DM_registration.interpolate_pixel_intensities(image = N0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])
kwargs = {"N0_dm":N0_dm, "HO_ctrl": zonal_ctrl_dict['HO_ctrl']  } 
bldr.reset_telemetry( zwfs_ns )
for i in OL_data.i: # from the intensities we can process the control signals which are appended to the telemetry namespace in zwfs_ns
    _ = bldr.process_zwfs_intensity(  i, zwfs_ns=zwfs_ns, method='zonal_interp_no_projection', record_telemetry = True, **kwargs )

# read in telemetry data
dm_cmd_filt = np.array( [c[zwfs_ns.reco.linear_zonal_model.act_filt_recommended ] for c in np.array( OL_data.dm_cmd) ])
e_HO = np.array( zwfs_ns.telem.e_HO_list ) 
t_dm0 = np.array( OL_data.t_dm0 ) # before DM command sent 
t_dm1 = np.array( OL_data.t_dm1 ) # after DM command sent 
t_i0 = np.array( OL_data.t_i0 ) # before readout command sent 
t_i1 = np.array( OL_data.t_i1 ) # after readout command sent

print( f'average time spent putting DM shape on wavespace : {np.mean( t_dm1 - t_dm0  )}')
print( f'average time spent detecting the field : {np.mean( t_i1 - t_i0) }')
print( f'mean processing time to propagate to detector plane through ZWFS : {np.mean(t_i0 - t_dm1)}') # mean processing time to propagate to detector plane through ZWFS 

# timestamps interpoalte to commoon grid

t0 = np.min( [ t_dm0, t_dm1, t_i0, t_i1 ] )
t1 = np.max( [ t_dm0, t_dm1, t_i0, t_i1 ] )
dt = np.min( [ np.diff(t_dm0), np.diff(t_dm1), np.diff( t_i0), np.diff(t_i1) ] )

t = np.arange( t0, t1, dt )
dt = np.mean( np.diff( t ) )
fs = 1 / dt 

e_HO_interp = np.zeros( e_HO.shape )
dm_cmd_interp = np.zeros( dm_cmd_filt.shape )
for i in range( e_HO.shape[1] ):
    fdm = interpolate.interp1d( t_dm0, dm_cmd_filt[:,i], bounds_error=None, fill_value='extrapolate' )
    fi = interpolate.interp1d( t_i0, e_HO[:,i] , bounds_error=None, fill_value='extrapolate')
    e_HO_interp[:,i] = fi( e_HO[:,i]  )
    dm_cmd_interp[:,i] = fdm( dm_cmd_filt[:,i] )
# iterpolate onto same even sized grid 



# zonal_ctrl_dict = bldr.add_controllers_for_zonal_interp_no_projection( zwfs_ns ,  HO = 'PID' , return_controller =True) # HO = 'leaky'
# zonal_ctrl_dict['HO_ctrl'].kp = 1 * np.ones( len( zonal_ctrl_dict['HO_ctrl'].kp ) ) 
# zonal_ctrl_dict['HO_ctrl'].ki = 0.1 * np.ones( len( zonal_ctrl_dict['HO_ctrl'].kp ) ) 
# zonal_ctrl_dict['HO_ctrl'].plot_bode( mode_index=10 )


i = 1
f, cpsd = csd(dm_cmd_filt[:,i], e_HO[:,i], fs=fs)
_, psd_in = welch(dm_cmd_filt[:,i], fs=fs)

plt.loglog ( f, abs(cpsd)**2 ); plt.show()


KP, KI = np.meshgrid( np.linspace( 0. , 1 , 100), np.linspace( 0 , 1 , 100))
kd = 0
fs = 1
delay = 0.1
err = []


for kp, ki in zip( KP.reshape(-1), KI.reshape(-1) ):

    # plant TF  
    G = cpsd / psd_in #response_psd[1] / disturbance_psd[1]
    
    # feedback TF  Interpolate model magnitude to match disturbance frequencies for comparison
    num = [kd, kp, ki]  # PID terms in s-domain
    den = [1, 0]        # Integral term in s-domain
    tf_pid = TransferFunction(num, den)
    w, mag, phase = bode( tf_pid )
    
    
    H_mag = np.interp(f, w/(2*np.pi), 10**(mag/20) )
    H_phase = np.interp(f, w/(2*np.pi), np.pi * phase/180 )

    H = H_mag * np.exp(1j * H_phase)
    
    S_TF = 1 / ( 1 + G * H )  # sensitivity TF 
    N_TF = G / ( 1 + G * H ) # Noise Transfer Function (N)*
    # Mean squared error between model PSD and actual response PSD
    error = np.sum( psd_in * abs( S_TF )**2 ) 
    err.append( error )

kp_opt, ki_opt = np.unravel_index( np.argmin( err ) , KP.shape )
plt.figure(); plt.semilogy( err );plt.show()
plt.figure(); plt.imshow( np.log10( np.array(err).reshape( KP.shape) )) ;plt.colorbar(); plt.show()
plt.loglog( f, abs( CL_TF )**2 ); plt.show()



# calculate the plant transfer function

# add pure delay to the plant transfer function 



###
### CLOSED LOOP SIMULATION
#### 


### with zonal 

# kp = 1 * np.ones( np.sum(act_filt))
# ki = 0.2 * np.ones( np.sum(act_filt))
# kd = 0. * np.ones( np.sum(act_filt) )
# setpoint = np.zeros( np.sum(act_filt) )
# lower_limit_pid = -100 * np.ones( np.sum(act_filt) )
# upper_limit_pid = 100 * np.ones( np.sum(act_filt) )

# HO_ctrl = bldr.PIDController(kp, ki, kd, upper_limit_pid, lower_limit_pid, setpoint)

#HO_ctrl.reset()
zonal_ctrl_dict = bldr.add_controllers_for_zonal_interp_no_projection( zwfs_ns ,  HO = 'PID' , return_controller = True) # HO = 'leaky'
# init all gains to 0

amp_input =  photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns.pyZelda.pupil
dm_disturbance = np.zeros( 140 )


zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy() 
zwfs_ns = bldr.reset_telemetry( zwfs_ns )

N0_dm = DM_registration.interpolate_pixel_intensities(image = N0, pixel_coords = zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space) #DM_registration.interpolate_pixel_intensities(image = N0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])

kwargs = {"N0_dm":N0_dm, "HO_ctrl": zonal_ctrl_dict['HO_ctrl']  } 
phase_scaling_factor  = 0.1

kp=0
ki=0.5

Strehl_0_list = []
Strehl_1_list = []
Strehl_2_list = []
Strehl_est_list = []

close_after = 10
iterations = 100

# open / close with strehl estimate 
# project out piston / tip/tilt
# optimize gains!!! 
for it in range(iterations) :

    print( it )
    #print( kwargs['HO_ctrl'].integrals ) # <- this was the bug previously
    if it == close_after:

        kwargs["HO_ctrl"].kp = kp * np.ones( zonal_ctrl_dict['HO_ctrl'].kp.shape )
        kwargs["HO_ctrl"].ki = ki * np.ones( zonal_ctrl_dict['HO_ctrl'].ki.shape )
        #kwargs["HO_ctrl"].kd = 0 * np.ones( zonal_ctrl_dict['HO_ctrl'].kd.shape )
    
    # roll screen
    #for _ in range(10):
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
    opd_current_dm = bldr.get_dm_displacement( command_vector = zwfs_ns.dm.current_cmd   , gain=zwfs_ns.dm.opd_per_cmd, \
                sigma = zwfs_ns.grid.dm_coord.act_sigma_wavesp, X=zwfs_ns.grid.wave_coord.X, Y=zwfs_ns.grid.wave_coord.Y,\
                    x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp )
    
    # sum all opd contributions in the Baldr input pupil plane 
    # set opd_ao_1 = 0 if rolling phasescreen on DM 
    bldr_opd_map = np.sum( [  opd_ao_1, opd_vibrations, opd_internal, opd_current_dm ] , axis=0 )
    bldr_opd_map -= np.mean( bldr_opd_map[zwfs_ns.pyZelda.pupil>0.5] ) # remove piston  
    
    ao_2 = zwfs_ns.pyZelda.pupil * (2*np.pi) / zwfs_ns.optics.wvl0  *  bldr_opd_map # phase radians 
    
    # get the real strehl ratios at various points (for tracking performance) 
    Strehl_0 = np.exp( - np.var( phase_scaling_factor * scrn.scrn[zwfs_ns.pyZelda.pupil>0.5]) ) # atmospheric strehl 
    Strehl_1 = np.exp( - np.var( ao_1[zwfs_ns.pyZelda.pupil>0.5]) ) # strehl after first stage AO 
    Strehl_2 = np.exp( - np.var( ao_2[zwfs_ns.pyZelda.pupil>0.5]) ) # strehl after baldr     

    
    i = bldr.AO_iteration( opd_input=bldr_opd_map, amp_input=amp_input, opd_internal=0*opd_internal, zwfs_ns=zwfs_ns, dm_disturbance = np.zeros(140),\
        record_telemetry=True, method='zonal_interp_no_projection', detector=zwfs_ns.detector, obs_intermediate_field=True, \
            use_pyZelda = True, include_shotnoise=True, **kwargs)

    S_est = strehl_model.apply_model( np.array( [i / np.mean( N0[ strehl_model.detector_pupilmask ] )] ) ) 
    Strehl_est_list.append( S_est )
    
    if  S_est < 0.1:
        zonal_ctrl_dict['HO_ctrl'].reset()
        zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy()
    
    # propagate to the detector plane
    #Ic = photon_flux_per_pixel_at_vlti * zwfs_ns.pyZelda.propagate_opd_map( bldr_opd_map , wave = zwfs_ns.optics.wvl0 )
    
    # # detect the intensity
    #i = bldr.detect( Ic, binning = (zwfs_ns.detector.binning, zwfs_ns.detector.binning), qe=zwfs_ns.detector.qe , dit=zwfs_ns.detector.dit,\
    #     ron= zwfs_ns.detector.ron, include_shotnoise=True, spectral_bandwidth = zwfs_ns.stellar.bandwidth )


    #delta_cmd = bldr.process_zwfs_intensity( i, zwfs_ns, method = 'zonal_interp_no_projection', record_telemetry = True , **kwargs )
    
    # interpolate signals onto registered actuator grid
    # i_dm = DM_registration.interpolate_pixel_intensities(image = i, pixel_coords = zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space)

    # e_HO  = model_1.apply(i_dm/ np.mean(N0[ N0 > np.mean( N0 ) ] ))[act_filt]
     
    # u_HO = HO_ctrl.process( e_HO )
    
    # dm_cmd_est[act_filt] =  u_HO - np.mean( u_HO ) #model_1.apply(i_dm/ np.mean(N0[ N0 > np.mean( N0 ) ] ))[act_filt]

        
    Strehl_0_list.append( Strehl_0 )
    Strehl_1_list.append( Strehl_1 )
    Strehl_2_list.append( Strehl_2 )

    print( round(Strehl_0,2), round(Strehl_1,2) , round(Strehl_2,2), 'S2 est ', S_est )


i = -1
#im_dm_dist = np.array( [util.get_DM_command_in_2D( a ) for a in zwfs_ns.telem.dm_disturb_list] )
im_phase = np.array( [util.get_DM_command_in_2D(a) for a  in zwfs_ns.telem.i_dm_list ] ) # zwfs_ns.telem.field_phase ) 
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
image_list =  [ zwfs_ns.telem.field_phase, im_phase, im_int, im_cmd]
image_title_list =  [ 'input phase', 'DM interpolated intensity', 'ZWFS intensity', 'reco. command']
image_colorbar_list = ['DM units', 'radians', 'adu', 'DM units']

plot_list = [ line_eHO, line_eTT, line_S, line_rmse ] 
plot_ylabel_list = ['e_HO', 'e_TT', 'Strehl', 'rmse']
plot_xlabel_list = ['iteration' for _ in plot_list]
plot_title_list = ['' for _ in plot_list]

#vlims = [(0, 1), (0, 1), (0, 1)]  # Set vmin and vmax for each image

util.create_telem_mosaic([a[-1] for a in image_list], image_title_list, image_colorbar_list, 
                plot_list, plot_title_list, plot_xlabel_list, plot_ylabel_list)

util.display_images_with_slider(image_lists = image_list,  plot_titles=image_title_list, cbar_labels=image_colorbar_list)
util.display_images_with_slider(image_lists = image_list+[zwfs_ns.telem.strehl] ) #,  plot_titles=image_title_list, cbar_labels=image_colorbar_list)



### good movie 

field_phase_list=[]
for ff in zwfs_ns.telem.field_phase:
    ff[zwfs_ns.pyZelda.pupil==0] = np.nan
    field_phase_list.append( ff )


image_list =  [ field_phase_list, zwfs_ns.telem.i_list , [util.get_DM_command_in_2D( a ) for a in (np.array(zwfs_ns.telem.c_TT_list) + np.array(zwfs_ns.telem.c_HO_list)  ) ], zwfs_ns.telem.strehl]
image_title_list =  [ 'input phase', 'ZWFS intensity', 'reco. command', 'Strehl Ratio']
image_colorbar_list = [ 'radians', 'adu', 'DM units','']

util.display_images_as_movie( image_lists = image_list, plot_titles=image_title_list, cbar_labels=image_colorbar_list  , save_path="output_movie.mp4", fps=25 )

# check with actuators go bad 
tmpcmd = np.zeros(140)
tmpcmd[zwfs_ns.reco.linear_zonal_model.act_filt_recommended] = zwfs_ns.telem.e_HO_list[-1]
plt.imshow( util.get_DM_command_in_2D( tmpcmd ) )
plt.imshow( util.get_DM_command_in_2D( tmpcmd ) ); plt.show()






### with eigenmodes 
dynamic  = True

phase_scaling_factor = 0.2

if dynamic:
    scrn = ps.PhaseScreenKolmogorov(nx_size=zwfs_ns.grid.dim, pixel_scale=dx, r0=zwfs_ns.atmosphere.r0, L0=zwfs_ns.atmosphere.l0, random_seed=1)
    opd_input =  zwfs_ns.pyZelda.pupil * zwfs_ns.optics.wvl0 / (2*np.pi) *  scrn.scrn
else:# static 
    opd_input = 1 * zwfs_ns.pyZelda.pupil * zwfs_ns.optics.wvl0 / (2*np.pi) *  (basis[5] + basis[10])

amp_input = photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns.pyZelda.pupil
dm_disturbance = np.zeros( 140 )

zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat + dm_disturbance
zwfs_ns = bldr.reset_telemetry( zwfs_ns ) # initialize telemetry to empty list 

close_after = 10

Strehl_0_list = []
Strehl_1_list = []

kpTT = 1
ki_grid = np.linspace(0, 0.9, 15)
for cnt, kiTT in enumerate( [0.9]) :
    zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat + dm_disturbance
    zwfs_ns = bldr.reset_telemetry( zwfs_ns )
    zwfs_ns.ctrl.TT_ctrl.reset()
    zwfs_ns.ctrl.TT_ctrl.ki = 0 * np.zeros( len(zwfs_ns.ctrl.TT_ctrl.ki) )
    zwfs_ns.ctrl.TT_ctrl.kp = 0 * np.ones( len(zwfs_ns.ctrl.TT_ctrl.kp) )
    for i in range(100):
        print(f'iteration {i}')
        if i == close_after : 
            zwfs_ns.ctrl.HO_ctrl.ki = 0.99 * np.ones( len(zwfs_ns.ctrl.HO_ctrl.ki) )
            zwfs_ns.ctrl.HO_ctrl.kp = 0.3 * np.ones( len(zwfs_ns.ctrl.HO_ctrl.kp) )

            zwfs_ns.ctrl.TT_ctrl.kp = kpTT * np.ones( len(zwfs_ns.ctrl.TT_ctrl.kp) )
            zwfs_ns.ctrl.TT_ctrl.ki = kiTT * np.ones( len(zwfs_ns.ctrl.TT_ctrl.ki) )
            
        if dynamic:
            # roll screen
            if np.mod(i, 1) == 0:
                scrn.add_row()
            # first stage AO
            if np.mod(it, 1) == 0: # only update the AO every few iterations to simulate latency 
                _ , reco_1 = bldr.first_stage_ao( scrn, Nmodes_removed , basis  , phase_scaling_factor = phase_scaling_factor, return_reconstructor = True )   
                
            ao_1 =  basis[0] * (phase_scaling_factor * scrn.scrn - reco_1)

            # opd after first stage AO
            opd_ao_1 = zwfs_ns.pyZelda.pupil * zwfs_ns.optics.wvl0 / (2*np.pi) * ao_1
        else:
            opd_ao_1 = opd_input
        # add vibrations OPD
        opd_vibrations = np.zeros( ao_1.shape )
                
        # put them all together to get the input to the second stage AO
        bldr_opd_map = np.sum( [  opd_ao_1, opd_vibrations, opd_internal] , axis=0 )
        
        # second stage AO (uses opd_input + current DM command to get signal, then updates current DM command based on control law
        bldr.AO_iteration( opd_input = bldr_opd_map, amp_input=amp_input, opd_internal = opd_internal, I0 = zwfs_ns.reco.I0,  zwfs_ns=zwfs_ns, dm_disturbance = dm_disturbance, record_telemetry=True , detector=zwfs_ns.detector)

        # keep these seperate from telemetry because in real system you would not have access to these    
        Strehl_0 = np.exp( - np.var( phase_scaling_factor * scrn.scrn[zwfs_ns.pyZelda.pupil>0.5]) ) # atmospheric strehl 
        Strehl_1 = np.exp( - np.var( ao_1[zwfs_ns.pyZelda.pupil>0.5]) ) 

        Strehl_0_list.append( Strehl_0 )
        Strehl_1_list.append( Strehl_1 )
        
    _ = bldr.save_telemetry( zwfs_ns, savename=fig_path + f'SIM_CL_TT_kiTT-{kiTT}_kpTT-{1}_{tstamp}.fits' )
    # Generate some data


### with eigenmodes 
dynamic  = True

phase_scaling_factor = 0.2

if dynamic:
    scrn = ps.PhaseScreenKolmogorov(nx_size=zwfs_ns.grid.dim, pixel_scale=dx, r0=zwfs_ns.atmosphere.r0, L0=zwfs_ns.atmosphere.l0, random_seed=1)
    opd_input =  zwfs_ns.pyZelda.pupil * zwfs_ns.optics.wvl0 / (2*np.pi) *  scrn.scrn
else:# static 
    opd_input = 1 * zwfs_ns.pyZelda.pupil * zwfs_ns.optics.wvl0 / (2*np.pi) *  (basis[5] + basis[10])

amp_input = photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns.pyZelda.pupil
dm_disturbance = np.zeros( 140 )

zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat + dm_disturbance
zwfs_ns = bldr.reset_telemetry( zwfs_ns ) # initialize telemetry to empty list 
zwfs_ns.ctrl.TT_ctrl.reset()
zwfs_ns.ctrl.HO_ctrl.reset()
zwfs_ns.ctrl.TT_ctrl.set_all_gains_to_zero()
zwfs_ns.ctrl.HO_ctrl.set_all_gains_to_zero()

close_after = 10

Strehl_0_list = []
Strehl_1_list = []

kpTT = 1
ki_grid = np.linspace(0, 0.9, 15)
for cnt, kiTT in enumerate( [0.9]) :
    zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat + dm_disturbance
    zwfs_ns = bldr.reset_telemetry( zwfs_ns )
    zwfs_ns.ctrl.TT_ctrl.reset()
    zwfs_ns.ctrl.TT_ctrl.ki = 0 * np.zeros( len(zwfs_ns.ctrl.TT_ctrl.ki) )
    zwfs_ns.ctrl.TT_ctrl.kp = 0 * np.ones( len(zwfs_ns.ctrl.TT_ctrl.kp) )
    for i in range(100):
        print(f'iteration {i}')
        if i == close_after : 
            zwfs_ns.ctrl.HO_ctrl.ki = 0.99 * np.ones( len(zwfs_ns.ctrl.HO_ctrl.ki) )
            zwfs_ns.ctrl.HO_ctrl.kp = 0.3 * np.ones( len(zwfs_ns.ctrl.HO_ctrl.kp) )

            zwfs_ns.ctrl.TT_ctrl.kp = kpTT * np.ones( len(zwfs_ns.ctrl.TT_ctrl.kp) )
            zwfs_ns.ctrl.TT_ctrl.ki = kiTT * np.ones( len(zwfs_ns.ctrl.TT_ctrl.ki) )
            
        if dynamic:
            # roll screen
            if np.mod(i, 1) == 0:
                scrn.add_row()
            # first stage AO
            if np.mod(it, 1) == 0: # only update the AO every few iterations to simulate latency 
                _ , reco_1 = bldr.first_stage_ao( scrn, Nmodes_removed , basis  , phase_scaling_factor = phase_scaling_factor, return_reconstructor = True )   
                
            ao_1 =  basis[0] * (phase_scaling_factor * scrn.scrn - reco_1)

            # opd after first stage AO
            opd_ao_1 = zwfs_ns.pyZelda.pupil * zwfs_ns.optics.wvl0 / (2*np.pi) * ao_1
        else:
            opd_ao_1 = opd_input
        # add vibrations OPD
        opd_vibrations = np.zeros( ao_1.shape )
                
        # put them all together to get the input to the second stage AO
        bldr_opd_map = np.sum( [  opd_ao_1, opd_vibrations, opd_internal] , axis=0 )
        
        # second stage AO (uses opd_input + current DM command to get signal, then updates current DM command based on control law
        bldr.AO_iteration( opd_input = bldr_opd_map, amp_input=amp_input, opd_internal = opd_internal, I0 = zwfs_ns.reco.I0,  zwfs_ns=zwfs_ns, dm_disturbance = dm_disturbance, record_telemetry=True , detector=zwfs_ns.detector)

        # keep these seperate from telemetry because in real system you would not have access to these    
        Strehl_0 = np.exp( - np.var( phase_scaling_factor * scrn.scrn[zwfs_ns.pyZelda.pupil>0.5]) ) # atmospheric strehl 
        Strehl_1 = np.exp( - np.var( ao_1[zwfs_ns.pyZelda.pupil>0.5]) ) 

        Strehl_0_list.append( Strehl_0 )
        Strehl_1_list.append( Strehl_1 )
        
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
    i_dm = DM_registration.interpolate_pixel_intensities(image = i, pixel_coords = zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space)


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
