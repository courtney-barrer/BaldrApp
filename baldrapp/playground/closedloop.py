
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


zwfs_ns.dm.actuator_coupling_factor = 0.9


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
scrn = ps.PhaseScreenKolmogorov(nx_size=zwfs_ns.grid.dim, 
                                pixel_scale=dx, 
                                r0=zwfs_ns.atmosphere.r0 * ( wvl0 /550e-9)**(6/5) , 
                                L0=zwfs_ns.atmosphere.l0, 
                                random_seed=1)

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
#zwfs_ns = bldr.construct_ctrl_matricies_from_IM( zwfs_ns,  method = 'Eigen_TT-HO', Smax = 60, TT_vectors = DM_basis.get_tip_tilt_vectors() )

# or we create fit a linear zonal model to the IM
zwfs_ns = bldr.fit_linear_zonal_model( zwfs_ns, opd_internal, iterations = 100, photon_flux_per_pixel_at_vlti = photon_flux_per_pixel_at_vlti , \
    pearson_R_threshold = 0.6, phase_scaling_factor=phase_scaling_factor,   plot_intermediate=True , fig_path = None)


# example to interpolate any detected image onto the DM actuator grid
#DM_registration.interpolate_pixel_intensities(image = I0, pixel_coords = zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space)

# add controllers 
#zwfs_ns = bldr.add_controllers_for_MVM_TT_HO( zwfs_ns, TT = 'PID', HO = 'leaky')
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

#calibrate a model to map a subset of pixel intensities to Strehl Ratio 
#should eventually come back and debug for model_type = lin_comb - since it seemed to work better intially
strehl_model = bldr.calibrate_strehl_model( zwfs_ns, save_results_path = fig_path, train_fraction = 0.6, correlation_threshold = 0.6, \
   number_of_screen_initiations = 60, scrn_scaling_grid = np.logspace(-2, -0.5, 5), model_type = 'PixelWiseStrehlModel' ) #lin_comb') 

# or read one in  
# strehl_model_file = proj_path  + '/baldrapp/configurations/strehl_model_config-BALDR_UT_J3_2024-10-19T09.28.27.pkl'
# strehl_model = load_model_from_pickle(filename=strehl_model_file)




###
### CLOSED LOOP SIMULATION
#### 


### with zonal 

# optimize gains!!!
zonal_ctrl_dict = bldr.add_controllers_for_zonal_interp_no_projection( zwfs_ns ,  HO = 'PID' , return_controller = True) # HO = 'leaky'

amp_input =  photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns.pyZelda.pupil
dm_disturbance = np.zeros( 140 )

zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy() 
zwfs_ns = bldr.reset_telemetry( zwfs_ns )

N0_dm = DM_registration.interpolate_pixel_intensities(image = N0, pixel_coords = zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space) #DM_registration.interpolate_pixel_intensities(image = N0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])

kwargs = {"N0_dm":N0_dm, "HO_ctrl": zonal_ctrl_dict['HO_ctrl']  } 

Strehl_0_list = []
Strehl_1_list = []
Strehl_2_list = []
Strehl_est_list = []

static_aberrations_opd =  1e-7 * np.sum( [ b * a for a,b in zip( [0.5, 0.5, 0.2, 0.1, 0.1], basis[1:5])  ] , axis=0) 
print(  f'using { 1e9 * np.std(  static_aberrations_opd[zwfs_ns.pyZelda.pupil > 0.5] )}nm rms static aberrations' )
close_after = 5
iterations = 20

# open / close with strehl estimate 
# project out piston / tip/tilt
# optimize gains!!! 

improvement_ratio = {}
baldr_strehl_ratio = {}
for kp in [0, 0.1, 0.5, 1]:
    baldr_strehl_ratio[kp] = {}
    improvement_ratio[kp] = {}
    for ki in [0, 0.1 , 0.5, 0.9, 1]:
        print(f'   kp = {kp}, ki={ki}')

        #### SETUP #### 
        # same screen for all simulations
        scrn = ps.PhaseScreenKolmogorov(nx_size=zwfs_ns.grid.dim, 
                                        pixel_scale=dx, 
                                        r0=zwfs_ns.atmosphere.r0 * ( wvl0 /550e-9)**(6/5) , 
                                        L0=zwfs_ns.atmosphere.l0, 
                                        random_seed=1)

        #HO_ctrl.reset()
        zonal_ctrl_dict = bldr.add_controllers_for_zonal_interp_no_projection( zwfs_ns ,  HO = 'PID' , return_controller = True) # HO = 'leaky'
        # init all gains to 0

        amp_input =  photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns.pyZelda.pupil
        dm_disturbance = np.zeros( 140 )


        zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy() 
        zwfs_ns = bldr.reset_telemetry( zwfs_ns )

        N0_dm = DM_registration.interpolate_pixel_intensities(image = N0, pixel_coords = zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space) #DM_registration.interpolate_pixel_intensities(image = N0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])

        kwargs = {"N0_dm":N0_dm, "HO_ctrl": zonal_ctrl_dict['HO_ctrl']  } 

        Strehl_0_list = []
        Strehl_1_list = []
        Strehl_2_list = []
        Strehl_est_list = []

        #
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
            
            # add vibrations OPD and/or static aberrations
            opd_vibrations = static_aberrations_opd #np.zeros( ao_1.shape )
            
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
            
            if S_est < 0.1:
                zonal_ctrl_dict['HO_ctrl'].reset()
                zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy()
           
            Strehl_0_list.append( Strehl_0 )
            Strehl_1_list.append( Strehl_1 )
            Strehl_2_list.append( Strehl_2 )

            print( round(Strehl_0,2), round(Strehl_1,2) , round(Strehl_2,2), 'S2 est ', S_est )

        baldr_strehl_ratio[kp][ki] =  Strehl_2_list
        improvement_ratio[kp][ki] =  np.array( [S2/ S1 for S1 , S2 in zip( Strehl_1_list[close_after:], Strehl_2_list[close_after:] ) ]  )



# Extract the keys for rows (kp) and columns (ki)
kp_values = sorted(improvement_ratio.keys())  # e.g., [1, 2, 3]
ki_values = sorted(improvement_ratio[kp_values[0]].keys())  # e.g., [1, 2, 3]

# Create a 2D numpy array
improvement_matrix_mean = np.array([[np.mean( improvement_ratio[kp][ki]) for ki in ki_values] for kp in kp_values])

plt.imshow( improvement_matrix_mean, cmap='viridis', origin='lower', extent=[ki_values[0], ki_values[-1], kp_values[0], kp_values[-1]], aspect='auto')
plt.colorbar()
plt.show() 

cax = plt.imshow(improvement_matrix_mean, cmap='YlGnBu', interpolation='nearest')

# Add color bar
plt.colorbar(cax)

# Labeling the axes
plt.xticks(np.arange(len(ki_values)), ki_values)  # x-axis ticks
plt.yticks(np.arange(len(kp_values)), kp_values)  # y-axis ticks
plt.xlabel('ki')
plt.ylabel('kp')

# Add a title
plt.title('Improvement Ratio Heatmap')
plt.show() 




# impact of actuator cross coupling on correction performance
# s




#### SETUP #### 
# same screen for all simulations
scrn = ps.PhaseScreenKolmogorov(nx_size=zwfs_ns.grid.dim, 
                                pixel_scale=dx, 
                                r0=zwfs_ns.atmosphere.r0 * ( wvl0 /550e-9)**(6/5) , 
                                L0=zwfs_ns.atmosphere.l0, 
                                random_seed=1)

#HO_ctrl.reset()
zonal_ctrl_dict = bldr.add_controllers_for_zonal_interp_no_projection( zwfs_ns ,  HO = 'PID' , return_controller = True) # HO = 'leaky'
# init all gains to 0

amp_input =  photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns.pyZelda.pupil
dm_disturbance = np.zeros( 140 )


zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy() 
zwfs_ns = bldr.reset_telemetry( zwfs_ns )

N0_dm = DM_registration.interpolate_pixel_intensities(image = N0, pixel_coords = zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space) #DM_registration.interpolate_pixel_intensities(image = N0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])


static_aberrations_opd =  1e-7 * np.sum( [ b * a for a,b in zip( [0.5, 0.5, 0.2, 0.1, 0.1], basis[1:5])  ] , axis=0) 
print(  f'using { 1e9 * np.std(  static_aberrations_opd[zwfs_ns.pyZelda.pupil > 0.5] )}nm rms static aberrations' )
close_after = 5
iterations = 20

kwargs = {"N0_dm":N0_dm, "HO_ctrl": zonal_ctrl_dict['HO_ctrl']  } 

Strehl_0_list = []
Strehl_1_list = []
Strehl_11_list = []
Strehl_2_list = []
Strehl_est_list = []

kp = 0.0
ki = 0.4 

#
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
    
    # add vibrations OPD and/or static aberrations
    opd_vibrations = static_aberrations_opd #np.zeros( ao_1.shape )
    

    rad_before_baldr_dm = (2*np.pi) / zwfs_ns.optics.wvl0  * np.sum( [  opd_ao_1, opd_vibrations, opd_internal ] , axis=0 )

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
    Strehl_11 = np.exp( - np.var( rad_before_baldr_dm[zwfs_ns.pyZelda.pupil>0.5]  ) )
    Strehl_2 = np.exp( - np.var( ao_2[zwfs_ns.pyZelda.pupil>0.5]) ) # strehl after baldr     

    
    i = bldr.AO_iteration( opd_input=bldr_opd_map, amp_input=amp_input, opd_internal=0*opd_internal, zwfs_ns=zwfs_ns, dm_disturbance = np.zeros(140),\
        record_telemetry=True, method='zonal_interp_no_projection', detector=zwfs_ns.detector, obs_intermediate_field=True, \
            use_pyZelda = True, include_shotnoise=True, **kwargs)

    S_est = strehl_model.apply_model( np.array( [i / np.mean( N0[ strehl_model.detector_pupilmask ] )] ) ) 
    Strehl_est_list.append( S_est )
    
    if S_est < 0.1:
        zonal_ctrl_dict['HO_ctrl'].reset()
        zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy()

    Strehl_0_list.append( Strehl_0 )
    Strehl_1_list.append( Strehl_1 )
    Strehl_11_list.append( Strehl_11 )
    Strehl_2_list.append( Strehl_2 )

    print( round(Strehl_0,2), round(Strehl_1,2) , round(Strehl_11,2) , round(Strehl_2,2), 'S2 est ', S_est )





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
       

# check with actuators go bad 
tmpcmd = np.zeros(140)
tmpcmd[zwfs_ns.reco.linear_zonal_model.act_filt_recommended] = zwfs_ns.telem.e_HO_list[-1]
plt.imshow( util.get_DM_command_in_2D( tmpcmd ) )
plt.imshow( util.get_DM_command_in_2D( tmpcmd ) ); plt.show()




