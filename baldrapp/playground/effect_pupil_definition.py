
# Call the function to add the project root

import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import importlib 
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os 
import sys

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

################## TEST 7
# Build IM  and look at Eigenmodes! 
zwfs_ns_c = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)
zwfs_ns_s = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)

opd_input = 0 * zwfs_ns_c.grid.pupil_mask *  np.random.randn( *zwfs_ns_c.grid.pupil_mask.shape)

opd_internal = 10e-9 * zwfs_ns_c.grid.pupil_mask * np.random.randn( *zwfs_ns_c.grid.pupil_mask.shape)

amp_input = 1e4 * zwfs_ns_c.grid.pupil_mask


# we make the pupil filter the entire square region for an extreme case ..
zwfs_ns_s = bldr.classify_pupil_regions( opd_input,  amp_input ,  opd_internal,  zwfs_ns_s , detector=detector)
zwfs_ns_s.pupil_regions.pupil_filt = np.ones( zwfs_ns_s.pupil_regions.pupil_filt.shape).astype(bool)

# extreme case 
# IM with square pupil (entire image)
IM_s = bldr.build_IM( zwfs_ns_s ,  calibration_opd_input= 0 *zwfs_ns_s.grid.pupil_mask , calibration_amp_input = amp_input , \
    opd_internal = opd_internal,  basis = basis, Nmodes =  Nmodes, poke_amp = 0.05, poke_method = 'double_sided_poke',\
        imgs_to_mean = 1, detector=detector)

basis_name_list = ['Hadamard', "Zonal", "Zonal_pinned_edges", "Zernike", "Zernike_pinned_edges", "fourier", "fourier_pinned_edges"]

# perfect field only with internal opd aberrations 
# different poke methods 
Nmodes = 100
basis = 'Zonal_pinned_edges'
M2C_0 = gen_basis.construct_command_basis( basis= basis, number_of_modes = Nmodes, without_piston=True).T  



fig_path = '/home/benja/Documents/BALDR/figures/effect_pupil_definition/'

if os.path.exists(fig_path) == False:
    os.makedirs(fig_path)

plt.ion()

D = zwfs_ns_c.grid.D 
for r in [0.5, 0.8, 1.0, 1.2]:
    print('doing r = ', r)
    for offset in [(0,0), (0, D/10), (0, D/5) ]:
        
        zwfs_ns_c = bldr.classify_pupil_regions( opd_input,  amp_input ,  opd_internal,  zwfs_ns_c , detector=detector, pupil_diameter_scaling = r, pupil_offset = offset)
        # IM with circular pupil 
        zwfs_ns_c = bldr.build_IM( zwfs_ns_c ,  calibration_opd_input= 0 *zwfs_ns_c.grid.pupil_mask , calibration_amp_input = amp_input , \
            opd_internal = opd_internal,  basis = basis, Nmodes =  Nmodes, poke_amp = 0.05, poke_method = 'double_sided_poke',\
                imgs_to_mean = 1, detector=detector)

        fname = 'dilation_factor-{r}_offset_{offset}'.format(r=r, offset=offset)
        bldr.plot_eigenmodes( zwfs_ns_c , save_path = fig_path , descr_label =  fname )
        plt.close( 'all')
        
