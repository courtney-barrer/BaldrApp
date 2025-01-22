"""
2) build IM onsky - identify mode that corresponds to optical gain
for a pure scaling term in iinput aberations see how ratio of modes change with ratio of scaling factor. This should be constant besides the mode
most correlated with optical gain 
3) try get zonal closed loop working with proper DM registration and interpolation. Create basis based on registration. 
Then define tip / tilt and optical gain modes and filter them out 


I have a square deformable mirror that will actuator on a circular pupil of light. Only a few actuators in the circular region are registered with the actual pupil (lay within the illuminated region). I want to define a basis on these actuators, but also need the actuators outside the illuminated pupil (outside the registered basis) to vary smoothly with the registered actuators due to inter-actuator coupling. """



# Call the function to add the project root

import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import importlib 
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os 
import sys
#import aotools

def add_project_root_to_sys_path(project_root_name="BaldrApp"):
    """
    Adds the project root directory to sys.path to allow importing from shared modules.
    
    Args:
        project_root_name (str): The name of the project root directory.
    """
    try:
        # Attempt to use __file__ to get the current script's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback to current working directory (useful in interactive environments)
        current_dir = os.getcwd()

    # Traverse up the directory tree to find the project root
    project_root = current_dir
    while True:
        if os.path.basename(project_root) == project_root_name:
            break
        new_project_root = os.path.dirname(project_root)
        if new_project_root == project_root:
            # Reached the filesystem root without finding the project root
            project_root = None
            break
        project_root = new_project_root

    if project_root and project_root not in sys.path:
        sys.path.append(project_root)
        print(f"Added '{project_root}' to sys.path")
    elif project_root is None:
        print(f"Error: '{project_root_name}' directory not found in the directory hierarchy.")
    else:
        print(f"'{project_root}' is already in sys.path")

# Call the function to add the project root
add_project_root_to_sys_path()
from baldrapp.common import baldr_core as bldr
from baldrapp.common import DM_basis as gen_basis
from baldrapp.common import utilities as util
from baldrapp.common import DM_registration as DM_reg
from baldrapp.common import phasescreens

""" 
1) how square vs circular and different radii pupil region classification affects eigenvectors 

"""


################## TEST 0 
# configure our zwfs 
grid_dict = {
    "D":1, # diameter of beam 
    "N" : 64, # number of pixels across pupil diameter
    "padding_factor" : 4, # how many pupil diameters fit into grid x axis
    # TOTAL NUMBER OF PIXELS = padding_factor * N 
    }

optics_dict = {
    "wvl0" :1.65e-6, # central wavelength (m) 
    "F_number": 21.2, # F number on phasemask
    "mask_diam": 1.06, # diameter of phaseshifting region in diffraction limit units (physical unit is mask_diam * 1.22 * F_number * lambda)
    "theta": 1.57079, # phaseshift of phasemask 
}

dm_dict = {
    "dm_model":"BMC-multi-3.5",
    "actuator_coupling_factor":0.7,# std of in actuator spacing of gaussian IF applied to each actuator. (e.g actuator_coupling_factor = 1 implies std of poke is 1 actuator across.)
    "dm_pitch":1,
    "dm_aoi":0, # angle of incidence of light on DM 
    "opd_per_cmd" : 3e-6, # peak opd applied at center of actuator per command unit (normalized between 0-1) 
    "flat_rmse" : 20e-9 # std (m) of flatness across Flat DM  
    }

grid_ns = SimpleNamespace(**grid_dict)
optics_ns = SimpleNamespace(**optics_dict)
dm_ns = SimpleNamespace(**dm_dict)

detector = (4,4) 

zwfs_ns = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)

# CALIBRATION FIELD PARAMETERS
cal_opd_input  = 0 * zwfs_ns.grid.pupil_mask *  np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

cal_opd_internal = 10e-9 * zwfs_ns.grid.pupil_mask * np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

cal_amp_input = 1e4 * zwfs_ns.grid.pupil_mask



basis = 'Zonal_pinned_edges'
Nmodes = 100
M2C_0 = gen_basis.construct_command_basis( basis= basis, number_of_modes = Nmodes, without_piston=True).T  

zwfs_ns = bldr.classify_pupil_regions( cal_opd_input ,  cal_amp_input ,  cal_opd_internal,  zwfs_ns , detector=detector ) 

zwfs_ns = bldr.build_IM( zwfs_ns ,  calibration_opd_input = cal_opd_input  , calibration_amp_input = cal_amp_input , \
            opd_internal = cal_opd_internal,  basis = basis, Nmodes =  Nmodes, poke_amp = 0.05, poke_method = 'double_sided_poke',\
                imgs_to_mean = 1, detector=detector)



### 
# DM registration 
###

# get inner corners for estiamting DM center in pixel space (have to deal seperately with pinned actuator basis)
if zwfs_ns.reco.IM.shape[0] == 100: # outer actuators are pinned, 
    corner_indicies = DM_reg.get_inner_square_indices(outer_size=10, inner_offset=3, without_outer_corners=False)
    
elif zwfs_ns.reco.IM.shape[0] == 140: # outer acrtuators are free 
    print(140)
    corner_indicies = DM_reg.get_inner_square_indices(outer_size=12, inner_offset=4, without_outer_corners=True)
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
transform_dict = DM_reg.calibrate_transform_between_DM_and_image( dm_4_corners, img_4_corners, debug=True, fig_path = None )


fig = plt.figure(3)
N0 = bldr.get_N0( cal_opd_input , cal_amp_input, cal_opd_internal, zwfs_ns  , detector=detector )
im = plt.imshow( N0 )
cbar = fig.colorbar(im, ax=plt.gca(), pad=0.01)
cbar.set_label(r'Intensity', fontsize=15, labelpad=10)

plt.scatter(transform_dict['actuator_coord_list_pixel_space'][:, 0],\
    transform_dict['actuator_coord_list_pixel_space'][:, 1], \
        color='blue', marker='.', label = 'DM actuators')

plt.legend() 
plt.show() 







### 
# Build control model  
#  apply Kolmogorov statistics to DM and measure intensity response, fit linear model to data 
#  then try fitting more advanced models, compare improvement 
###

i = bldr.get_frame( cal_opd_input , cal_amp_input, cal_opd_internal, zwfs_ns )
interpolated_intensities = DM_reg.interpolate_pixel_intensities(image = i, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])

act_per_it = 0.5 # how many actuators does the screen pass per iteration 
V = 10 / act_per_it  / zwfs_ns.grid.D #m/s (10 actuators across pupil on DM)
#scrn = aotools.infinitephasescreen.PhaseScreenVonKarman(nx_size= int( 12 / act_per_it ) , pixel_scale= zwfs_ns.grid.D / zwfs_ns.grid.N , r0=0.1, L0=12)
scrn = phasescreens.PhaseScreenVonKarman(nx_size= int( 12 / act_per_it ) , pixel_scale= zwfs_ns.grid.D / zwfs_ns.grid.N , r0=0.1, L0=12)
corner_indicies = [0, 11, 11 * 12, -1] # DM corner indidices



I0_before = bldr.get_I0( cal_opd_input , cal_amp_input, cal_opd_internal, zwfs_ns , detector=detector )
N0_before = bldr.get_N0( cal_opd_input , cal_amp_input, cal_opd_internal, zwfs_ns  , detector=detector )

dm_I0_before = DM_reg.interpolate_pixel_intensities(image = I0_before, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])
dm_N0_before = DM_reg.interpolate_pixel_intensities(image = N0_before, pixel_coords = transform_dict['actuator_coord_list_pixel_space'])
training_data = {'I0':I0_before, 'dm_I0':dm_I0_before , 'dm_N0':dm_N0_before , 'N0':N0_before, 'dm_flat':zwfs_ns.dm.dm_flat, 'rmse':[], 'dm_cmd':[], 'i':[], 'interp_i':[]} 

no_iters = 100 
for i in range(no_iters):
    print( i / no_iters )
    # roll phase screen
    scrn.add_row()
    # bin phase screen onto DM space 
    dm_scrn = util.create_phase_screen_cmd_for_DM(scrn,  scaling_factor=0.05, drop_indicies = [0, 11, 11 * 12, -1] , plot_cmd=False)
    # update DM command 
    zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat + dm_scrn
    # get ZWFS intensity 
    i = bldr.get_frame( cal_opd_input , cal_amp_input, cal_opd_internal, zwfs_ns , detector=detector )
    # interpolate onto registered DM actuators
    interpolated_intensities = DM_reg.interpolate_pixel_intensities(image = i , pixel_coords = transform_dict['actuator_coord_list_pixel_space'])

    training_data['dm_cmd'].append( dm_scrn )
    training_data['i'].append( i )
    training_data['rmse'].append( np.std( dm_scrn ) )
    training_data['interp_i'].append( interpolated_intensities )
 












i = -1
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

im1 = ax[0].imshow(util.get_DM_command_in_2D(training_data['dm_cmd'][i]))
cbar1 = plt.colorbar(im1, ax=ax[0])
cbar1.ax.set_ylabel('Colorbar 1', rotation=270, labelpad=15)

im2 = ax[1].imshow(util.get_DM_command_in_2D(training_data['interp_i'][i] - dm_I0_before))
#im2 = ax[1].imshow(util.get_DM_command_in_2D(training_data['interp_i'][i] ))
cbar2 = plt.colorbar(im2, ax=ax[1])
cbar2.ax.set_ylabel('Colorbar 2', rotation=270, labelpad=15)

plt.tight_layout()

plt.show()

#a = 65
#plt.plot( training_data['dm_cmd'], training_data['interp_i'] ,'.') ; plt.show()



# Extract the input (dm_cmd) and output (interp_i) from training data
X = training_data['dm_cmd']  # Shape: (N, 140) where N is the number of training samples
y = training_data['interp_i']  # Shape: (N, 140) corresponding outputs


from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import numpy as np

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Ridge regression model
linear_model = Ridge(alpha=1.0)  # Regularization parameter can be tuned
linear_model.fit(X_train, y_train)

# Make predictions
y_pred_linear = linear_model.predict(X_test)

# Evaluate performance (e.g., RMSE)
rmse_linear = np.sqrt(np.mean((y_pred_linear - y_test) ** 2))
print(f'Linear Model RMSE: {rmse_linear}')
