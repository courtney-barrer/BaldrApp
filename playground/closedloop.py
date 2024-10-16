
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
from common import phasescreens as ps
from common import utilities as util
from common import baldr_core as bldr
from common import DM_registration
from common import DM_basis

from common.baldr_core import StrehlModel 


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


# initialize our ZWFS instrument
wvl0=1.25e-6
config_ini = '/home/benja/Documents/BALDR/BaldrApp/configurations/BALDR_UT_J3.ini'
zwfs_ns = bldr.init_zwfs_from_config_ini( config_ini=config_ini , wvl0=wvl0)

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

bldr.plot_eigenmodes( zwfs_ns , save_path = None )

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
transform_dict = DM_registration.calibrate_transform_between_DM_and_image( dm_4_corners, img_4_corners, debug=True, fig_path = None )

interpolated_intensities = DM_registration.interpolate_pixel_intensities(image = I0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])


# interpolate these fields onto the registered actuator grid
b0_dm = DM_registration.interpolate_pixel_intensities(image = I0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])
I0_dm = DM_registration.interpolate_pixel_intensities(image = b0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])
N0_dm = DM_registration.interpolate_pixel_intensities(image = N0, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])

# calibrate a model to map a subset of pixel intensities to Strehl Ratio 
strehl_model = bldr.calibrate_strehl_model( zwfs_ns, save_results_path = '/home/benja/Downloads/', train_fraction = 0.6, correlation_threshold = 0.6, \
    number_of_screen_initiations = 10, scrn_scaling_grid = np.logspace(-2, -0.5, 5), model_type = 'PixelWiseStrehlModel' ) #lin_comb') 

# or read one in  
#strehl_model_file = '/home/benja/Documents/BALDR/BaldrApp/configurations/strehl_model_config-BALDR_UT_J3.pkl'
#strehl_model = load_model_from_pickle(filename=strehl_model_file)


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

zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat 
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

#fir Y=M@X
M = np.linalg.lstsq(np.array( telem_ns.i_dm ).T[filt].T , np.array( telem_ns.dm_cmd ).T[filt].T, rcond=None)[0]

plt.figure(); plt.plot( M @ (np.array( telem_ns.i_dm ).T[filt] ), np.array( telem_ns.dm_cmd ).T[filt], '.'); plt.show()

act = 20
plt.figure(); plt.plot( (M @ (np.array( telem_ns.i_dm ).T[filt]) )[act], np.array( telem_ns.dm_cmd ).T[filt][act], '.'); plt.show()


filt = np.var( telem_ns.i_dm , axis= 0 ) > 180000 
plt.imshow( util.get_DM_command_in_2D( filt ) ) ; plt.colorbar() ; plt.show()

R_list = []
for act in range(140):
    R_list.append( pearsonr([a[act] for a in telem_ns.i_dm ], [a[act] for a in telem_ns.dm_cmd]).statistic )

    
plt.imshow( util.get_DM_command_in_2D( R_list ) )
plt.colorbar() 
plt.show()  

   
ii = [ ]

for i , cmd in zip(telem_ns.i_dm, telem_ns.dm_cmd):
    
pearsonr(pixel_intensity_series, strehl_ratios)

plt.plot( telem_ns.strehl_2, telem_ns.strehl_2_est, 'o')


image_lists = [[ util.get_DM_command_in_2D( a ) for a in telem_ns.i_dm], \
    [ util.get_DM_command_in_2D( a ) for a in telem_ns.dm_cmd], \
     telem_ns.Ic] 
util.display_images_with_slider(image_lists = image_lists,\
    plot_titles=None, cbar_labels=None)
       
    
i_fft = np.fft.fftshift( np.fft.fft2( telem_ns.i[0] ) )

# def calibrate strehl model    

# input 
# zwfs_ns 
# train_fraction = 0.6
# correlation threshold = 0.9
# save_results_path = None
