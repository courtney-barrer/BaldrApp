
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import pyzelda.zelda as zelda
import pyzelda.ztools as ztools
import pyzelda.utils.aperture as aperture
import pyzelda.utils.zernike as zernike
import pyzelda.utils.mft as mft

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


class RecursiveNamespaceToClass:
    def __init__(self, namespace):
        # Iterate through all attributes of the SimpleNamespace
        for key, value in vars(namespace).items():
            # If the attribute is another SimpleNamespace, recursively initialize it as an instance of this class
            if isinstance(value, SimpleNamespace):
                value = RecursiveNamespaceToClass(value)
            # Set the attribute on the instance
            setattr(self, key, value)

    def __repr__(self):
        return f"{self.__class__.__name__}({vars(self)})"
    
    
z = zelda.Sensor('BALDR_UT_J3')



################## 
# configure our zwfs 
wvl0 = 1.25e-6
grid_dict = {
    "D":8, # diameter of beam (m)
    "N" : z.pupil_diameter, # number of pixels across pupil diameter
    "padding_factor" : z.pupil_dim // z.pupil_diameter, # how many pupil diameters fit into grid x axis
    # TOTAL NUMBER OF PIXELS = padding_factor * N 
    }

optics_dict = {
    "wvl0" :wvl0 , # central wavelength (m) 
    "F_number": z.mask_Fratio   , # F number on phasemask
    "mask_diam": z.mask_diameter / (1.22 * z.mask_Fratio * wvl0 ), # diameter of phaseshifting region in diffraction limit units (physical unit is mask_diam * 1.22 * F_number * lambda)
    "theta": z.mask_phase_shift( wvl0 )  # phaseshift of phasemask 
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

zwfs_ns = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)

opd_input = zwfs_ns.grid.pupil_mask *  np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

opd_internal = 0 * zwfs_ns.grid.pupil_mask * np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

amp_input = 1e4 * zwfs_ns.grid.pupil_mask



# ++++++++++++++++++++++++++++++++++
# Zernike mask parameters
# ++++++++++++++++++++++++++++++++++

mask_substrate = 'fused_silica'
n_substrate = ztools.refractive_index(zwfs_ns.optics.wvl0 , mask_substrate)
# physical diameter and depth, in m
d_m = zwfs_ns.optics.mask_diam * ( 1.22 * zwfs_ns.optics.wvl0 * zwfs_ns.optics.F_number )
z_m = zwfs_ns.optics.theta * (zwfs_ns.optics.wvl0 / (2 * np.pi * (n_substrate - 1) ) )


I0 = ztools.propagate_opd_map(opd_map= 0 * opd_input, mask_diameter = d_m, mask_depth = z_m, mask_substrate = mask_substrate, mask_Fratio=zwfs_ns.optics.F_number,
                      pupil_diameter=zwfs_ns.grid.N, pupil = amp_input**0.5 , wave = zwfs_ns.optics.wvl0)


I = ztools.propagate_opd_map(opd_map= opd_input, mask_diameter = d_m, mask_depth = z_m, mask_substrate = mask_substrate, mask_Fratio=zwfs_ns.optics.F_number,
                      pupil_diameter=zwfs_ns.grid.N, pupil = amp_input**0.5 , wave = zwfs_ns.optics.wvl0)





#### TESTING MFT 
phi = zwfs_ns.grid.pupil_mask * 0* np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

pupil = 1e4 * zwfs_ns.grid.pupil_mask

theta = zwfs_ns.optics.theta 
psi_A = pupil * np.exp( 1j * ( phi ) )

# R_mask: mask radius in lam0/D unit
R_mask = 0.5 * zwfs_ns.optics.mask_diam

array_dim = phi.shape[0]
pupil_radius = zwfs_ns.grid.N // 2


# mask sampling in the focal plane
D_mask_pixels = 300

# ++++++++++++++++++++++++++++++++++
# Numerical simulation part
# ++++++++++++++++++++++++++++++++++

# --------------------------------
# plane A (Entrance pupil plane)

# definition of m1 parameter for the Matrix Fourier Transform (MFT)
# here equal to the mask size
mmm = 10

m1 = mmm * 2 * R_mask * (array_dim / (2. * pupil_radius))

ampl_PA = pupil * np.exp(1j * 2. * np.pi * phi / zwfs_ns.optics.wvl0)

ampl_PB = mft.mft(ampl_PA, array_dim, D_mask_pixels, m1)

phasemask_mask = aperture.disc(D_mask_pixels, D_mask_pixels//mmm, diameter=True, cpix=True, strict=False)
             
plt.figure(1)
plt.imshow( abs(ampl_PB) , extent=[-1, 1, -1, 1], origin='lower' )
X, Y = np.meshgrid( np.linspace(-1, 1, ampl_PB.shape[0]), np.linspace(-1, 1, ampl_PB.shape[0]) )
plt.colorbar()
plt.contour(X, Y, phasemask_mask, levels=10, colors='white', label='phasemask')
plt.show()
b = mft.imft(  phasemask_mask * ampl_PB , D_mask_pixels, array_dim, m1)

plt.figure(1)
plt.imshow( abs( b ), extent=[-1, 1, -1, 1], origin='lower' )
X, Y = np.meshgrid( np.linspace(-1, 1, b.shape[0]), np.linspace(-1, 1, b.shape[0]) )
plt.colorbar()
plt.contour(X, Y, abs( ampl_PA), levels=10, colors='white', label='phasemask')
plt.show()




psi_R = abs( b ) * np.sqrt((np.cos(theta)-1)**2 + np.sin(theta)**2)
mu = np.angle((np.exp(1J*theta)-1) ) # np.arctan( np.sin(theta)/(np.cos(theta)-1) ) #
beta = np.angle( b )
# out formula ----------
#if measured_pupil!=None:
#    P = measured_pupil / np.mean( P[P > np.mean(P)] ) # normalize by average value in Pupil

Ic = abs(psi_A)**2 + abs(psi_R)**2 + 2 * abs(psi_A) * abs(psi_R) * np.cos( phi - mu - beta )  #+ beta)

IcpyZ = z.propagate_opd_map(phi*1e9, mask_diameter, mask_depth, mask_substrate, mask_Fratio,
                      pupil_diameter, pupil, wave)


np.sum( abs(ampl_PA)**2 ) / np.sum( Ic )











Ime = bldr.get_frame( opd_input,  amp_input ,  opd_internal,  zwfs_ns )

fig,ax = plt.subplots( 3, 1, figsize=(10, 10) )
ax[0].imshow( I )
ax[1].imshow( Ime )
ax[2].imshow( Ime - I )
plt.show()



reference_wave, expi = create_reference_wave_beyond_pupil(mask_diameter=d_m, mask_depth=z_m, mask_substrate= mask_substrate,
                                                              mask_Fratio=zwfs_ns.optics.F_number, pupil_diameter=zwfs_ns.grid.N, pupil=zwfs_ns.grid.pupil_mask, \
                                                                  wave = zwfs_ns.optics.wvl0,
                                                              clear=clear, sign_mask=sign_mask, cpix=cpix)



plt.imshow( I - I0 ) ; plt.show()



propagate_opd_map(opd_map, mask_diameter, mask_depth, mask_substrate, mask_Fratio,
                      pupil_diameter, pupil, wave):

# substrate refractive index
n_substrate = ztools.refractive_index(zwfs_ns.optics.wvl0 , mask_substrate)

# R_mask: mask radius in lam0/D unit
R_mask = 0.5 * d_m / (zwfs_ns.optics.wvl0 * zwfs_ns.optics.F_number)

# ++++++++++++++++++++++++++++++++++
# Dimensions
# ++++++++++++++++++++++++++++++++++

# array and pupil
array_dim = zwfs_ns.grid.pupil_mask.shape[-1]
pupil_radius = zwfs_ns.grid.D // 2

# mask sampling in the focal plane
D_mask_pixels = 300








I = ztools.propagate_opd_map(opd_map= opd_input, mask_diameter, mask_depth, mask_substrate, mask_Fratio,
                      pupil_diameter=zwfs_ns.grid.D, pupil = zwfs_ns.grid.pupil_mask, wave = zwfs_ns.optics.wvl0)



wave = 1.642e-6

z = zelda.Sensor('SPHERE-IRDIS')

#################################

# Aberration: astigmatism
basis = np.nan_to_num(zernike.zernike_basis(nterms=5, npix=z.pupil_diameter))*1e-9
aberration = 20*basis[4]
gaussian_std = 1

# Gaussian apodized entrance pupil
x_vals = np.arange(z.pupil_diameter)
xx, yy = np.meshgrid(x_vals, x_vals)
cx, cy = x_vals[z.pupil_diameter//2], x_vals[z.pupil_diameter//2]
r = np.sqrt((xx-cx)**2+(yy-cy)**2)
r = r/r[0, z.pupil_diameter//2]

apodizer = np.exp(-(r**2)/2/gaussian_std**2)

#%%#############################
# Simulation of the wavefront. #
################################

# Clear pupil image
clear_pupil = abs(apodizer)**2

zelda_pupil0 = ztools.propagate_opd_map(0*aberration, z.mask_diameter, z.mask_depth, z.mask_substrate,
                                        z.mask_Fratio, z.pupil_diameter, apodizer, wave)

zelda_pupil = ztools.propagate_opd_map(aberration, z.mask_diameter, z.mask_depth, z.mask_substrate,
                                        z.mask_Fratio, z.pupil_diameter, apodizer, wave)



pupil_roi = aperture.disc(z.pupil_diameter, z.pupil_diameter, diameter=True, cpix=False)
z_opd_advanced = z.analyze(clear_pupil, zelda_pupil, wave,
                           use_arbitrary_amplitude=True,
                           refwave_from_clear=True,
                           cpix=False, pupil_roi=pupil_roi)

#%% plot

fig = plt.figure(0, figsize=(24, 4))
plt.clf()

gs = gridspec.GridSpec(ncols=7, nrows=1, figure=fig, width_ratios=[.1,1,1,1,1,1,.1])

ax = fig.add_subplot(gs[0,1])
mappable = ax.imshow(clear_pupil, aspect='equal', vmin=0, vmax=1)
ax.set_title('Clear pupil')

ax = fig.add_subplot(gs[0,0])
cbar1 = fig.colorbar(mappable=mappable, cax=ax)
cbar1.set_label('Normalized intensity')

ax = fig.add_subplot(gs[0,2])
ax.imshow(zelda_pupil, aspect='equal', vmin=0, vmax=1)
ax.set_title('ZELDA pupil')

ax = fig.add_subplot(gs[0,3])
ax.imshow(aberration*1e9, aspect='equal', vmin=-40, vmax=40, cmap='magma')
ax.set_title('Introduced aberration (nm)')

ax = fig.add_subplot(gs[0,4])
cax = ax.imshow(z_opd_standard, aspect='equal', vmin=-40, vmax=40, cmap='magma')
ax.set_title('Reconstructed aberration - standard')

ax = fig.add_subplot(gs[0,5])
cax = ax.imshow(z_opd_advanced, aspect='equal', vmin=-40, vmax=40, cmap='magma')
ax.set_title('Reconstructed aberration - advanced')

ax = fig.add_subplot(gs[0,6])
cbar = fig.colorbar(mappable=cax, cax=ax)
cbar.set_label('OPD [nm]')

plt.tight_layout()
plt.show()