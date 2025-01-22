"""
Lets see what the optical gain looks like for a simple system rolling Kolmogorov phase screens at different Strehl ratios across the pupil.

explore the real and imaginary components of the optical gain both inside and outside the pupil.

"""
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
from baldrapp.common import pupils




################## 
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



scrn = phasescreens.PhaseScreenVonKarman(nx_size= zwfs_ns.grid.N * zwfs_ns.grid.padding_factor , pixel_scale= zwfs_ns.grid.D / zwfs_ns.grid.N , r0=0.1, L0=12)

tel = 'UT'
vibs = 'none'
pup = pupils.pick_pupil(tel, dim = zwfs_ns.grid.pupil_mask.shape[0], diameter = zwfs_ns.grid.N)
phase_scaling_factor  = 0.3


fig_path = f'/home/benja/Documents/BALDR/figures/exploring_optical_gain/gif_{tel}_phSclF-{phase_scaling_factor}_vibs-{vibs}/'

if os.path.exists(fig_path) == False:
    os.makedirs(fig_path)
    
for i in range(100):
    print(i)
    for _ in range(zwfs_ns.grid.N//10):
        scrn.add_row()

    input_phase = phase_scaling_factor * pup * scrn.scrn # radians 

    strehl = np.exp(-np.var( input_phase[pup>0.3] ) )
    print( 'S=', strehl )

    psi_A =  pup * np.exp(1j *input_phase)
    Nb = 128
    no_loD = Nb//2
    psi_B = mft( psi_A, Na = input_phase.shape[0], Nb = 128, m = no_loD, cpix=False)
    #plt.imshow( np.abs( psi_B ) ); plt.show()


    x = np.linspace( -no_loD/2 , no_loD/2 , Nb)
    y = x.copy()
    X,Y = np.meshgrid(x,y) 

    phasemask_mask = X**2 + Y**2 <= (no_loD / 32 )**2
    #plt.imshow( phasemask_mask );plt.show()


    b = imft(phasemask_mask * psi_B, Nb = input_phase.shape[0], Na = Nb, m = no_loD, cpix=False)

    r1, r2, c1, c2 = (zwfs_ns.grid.padding_factor-1)//2 *zwfs_ns.grid.N  , (  zwfs_ns.grid.padding_factor-1) *zwfs_ns.grid.N , (zwfs_ns.grid.padding_factor-1)//2 *zwfs_ns.grid.N  , (zwfs_ns.grid.padding_factor-1) *zwfs_ns.grid.N 
    im_list = [ input_phase[ r1:r2, c1:c2  ] , abs(psi_B)**2, np.real(b),  np.imag(b), np.abs(b)]
    xlabel_list = ['' for _ in im_list]
    ylabel_list = ['' for _ in im_list]
    title_list = [f'input phase\n(Strehl={round(100* strehl)}%)', 'PSF', 'Real\noptical gain', 'Imaginary\noptical gain', 'Magnitude\noptical gain']
    cbar_label_list = ['radian', 'ADU', 'Re[b]', 'Im[b]', '|b|']
    util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15,\
        cbar_orientation = 'bottom', axis_off=True, vlims=None, savefig=fig_path + f'frame_{i}.png')

    #plt.show()
    plt.close('all')
    

# psi_A2 = imft(psi_B, Nb = input_phase.shape[0], Na = 100, m = 50, cpix=False)
# plt.imshow( np.abs( psi_B ) ); plt.show()



b = bldr.get_b( phi = input_phase , phasemask = zwfs_ns.grid.phasemask_mask )

b = bldr.get_b_fresnel( phi = input_phase , phasemask = zwfs_ns.grid.phasemask_mask , wavelength=zwfs_ns.optics.wvl0, dx = zwfs_ns.grid.D / zwfs_ns.grid.N , z = zwfs_ns.grid.D**2 / zwfs_ns.optics.wvl0)

im_list = [ input_phase , np.real(b), np.imag(b), np.abs(b)]
xlabel_list = ['' for _ in im_list]
ylabel_list = ['' for _ in im_list]
title_list = ['input phase', 'Real', 'Imaginary', 'Magnitude']
cbar_label_list = ['radian','Re[b]', 'Im[b]', '|b|']
util.nice_heatmap_subplots( im_list , xlabel_list, ylabel_list, title_list, cbar_label_list, fontsize=15,\
    cbar_orientation = 'bottom', axis_off=True, vlims=None, savefig=None)








def _mft(array, Na, Nb, m, inverse=False, cpix=False):
    '''
    Performs the matrix Fourier transform of an array.
    
    Based on the formalism detailed in:
    
        Fast computation of Lyot-style coronagraph propagation
        Soummer, Pueyo, Sivaramakrishnan and Vanderbei
        2007, Optics Express, 15, 15935

    Parameters
    ----------
    array : array
        The input array
    
    Na : int
        Number of pixels in direct space
    
    Nb : int
        Number of pixels in Fourier space
    
    m : float
        Number of lambda/D elements in Fourier space
    
    inverse : bool, optional
        Control if the direct or inverse transform is performed. Default is `False`
    
    cpix : bool, optional
        Center the MFT on one pixel, instead of between 4 pixels. Default is 'False'
        
    Returns
    -------
    return_value : array
        MFT of the input array
    '''
        
    if (inverse is True):
        sign = 1
    else:
        sign = -1
    
    # For pixel centering
    offsetA = 0
    offsetB = 0
    
    # If centering between 4 pixels
    if not(cpix):
        offsetA = .5/Na
        offsetB = .5/Nb
    
    coeff = m / (Na * Nb)
    
    x = np.linspace(-0.5, 0.5, Na, endpoint=False, dtype=np.double) + offsetA
    y = x
    
    u = m * (np.linspace(-0.5, 0.5, Nb, endpoint=False, dtype=np.double) + offsetB)
    v = u
    
    A1 = np.exp(sign*2j*np.pi*np.outer(u, x))
    A3 = np.exp(sign*2j*np.pi*np.outer(v, y))
    
    B = coeff * A1.dot(array).dot(A3.T)

    return B


def mft(array, Na, Nb, m, cpix=False):
    '''
    Performs the matrix Fourier transform of an array.
    
    Based on the formalism detailed in:
    
        Fast computation of Lyot-style coronagraph propagation
        Soummer, Pueyo, Sivaramakrishnan and Vanderbei
        2007, Optics Express, 15, 15935

    Parameters
    ----------
    array : array
        The input array
    
    Na : int
        Number of pixels in direct space
    
    Nb : int
        Number of pixels in Fourier space
    
    m : float
        Number of lambda/D elements in Fourier space
    
    cpix : bool, optional
        Center the MFT on one pixel, instead of between 4 pixels. Default is 'False'
         
    Returns
    -------
    return_value : array
        MFT of the input array
    '''
    
    return _mft(array, Na, Nb, m, inverse=False, cpix=cpix)


def imft(array, Na, Nb, m, cpix=True):
    '''
    Performs the inverse matrix Fourier transform of an array.
    
    Based on the formalism detailed in:
    
        Fast computation of Lyot-style coronagraph propagation
        Soummer, Pueyo, Sivaramakrishnan and Vanderbei
        2007, Optics Express, 15, 15935

    Parameters
    ----------
    array : array
        The input array
    
    Na : int
        Number of pixels in direct space
    
    Nb : int
        Number of pixels in Fourier space
    
    m : float
        Number of lambda/D elements in Fourier space
        
    Returns
    -------
    return_value : array
        MFT of the input array
    '''
    
    return _mft(array, Na, Nb, m, inverse=True, cpix=cpix)
