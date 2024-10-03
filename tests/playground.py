
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import importlib 
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy


import Baldr_closeloop as bldr
import DM_basis as gen_basis
import utilities as util

fig_path = f'/Users/bencb/Documents/baldr/data_sydney/analysis_scripts/analysis_results/TT_CL_SIM'

################## TEST 0 
# configure our zwfs 
grid_dict = {
    "D":1, # diameter of beam 
    "N" : 64, # number of pixels across pupil diameter (NOT THE TOTAL GRID!)
    "padding_factor" : 4, # how many pupil diameters fit into grid x-axis
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

zwfs_ns = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)


opd_input = 0 * zwfs_ns.grid.pupil_mask *  np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

opd_internal = 10e-9 * zwfs_ns.grid.pupil_mask * np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

amp_input = 1e4 * zwfs_ns.grid.pupil_mask

Nmodes = 100
basis = 'Zonal_pinned_edges'
poke_amp = 0.05
Smax = 30
detector = (4,4) # for binning , zwfs_ns.grid.N is #pixels across pupil diameter (64) therefore division 4 = 16 pixels (between CRed2 and Cred1 )

# we must first define our pupil regions before building 
zwfs_ns = bldr.classify_pupil_regions( opd_input,  amp_input ,  opd_internal,  zwfs_ns , detector= detector) # For now detector is just tuple of pixels to average. useful to know is zwfs_ns.grid.N is number of pixels across pupil. # from this calculate an appropiate binning for detector 

#basis_name_list = ['Hadamard', "Zonal", "Zonal_pinned_edges", "Zernike", "Zernike_pinned_edges", "fourier", "fourier_pinned_edges"]
zwfs_ns = bldr.build_IM( zwfs_ns,  calibration_opd_input= 0 *zwfs_ns.grid.pupil_mask , calibration_amp_input = amp_input , \
    opd_internal = opd_internal,  basis = basis, Nmodes =  Nmodes, poke_amp = poke_amp, poke_method = 'double_sided_poke',\
        imgs_to_mean = 1, detector=detector )

TT_vectors = gen_basis.get_tip_tilt_vectors()


zwfs_ns_2 = copy.deepcopy( zwfs_ns ) 

# useing R_TT 
zwfs_ns = bldr.construct_ctrl_matricies_from_IM(zwfs_ns,  method = 'Eigen_TT-HO', Smax = 50, TT_vectors = TT_vectors )

zwfs_ns_2 = bldr.construct_ctrl_matricies_from_IM(zwfs_ns_2,  method = 'Eigen_TT-HO_2', Smax = 50, TT_vectors = TT_vectors )

zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat + 0.2 * TT_vectors[:,0] 

i = bldr.get_frame( opd_input,  amp_input ,  opd_internal,  zwfs_ns , detector= detector )

#bldr.plot_current_dm_cmd(zwfs_ns)
#bldr.plot_zwfs_signal( i ,zwfs_ns ) 

s = bldr.process_zwfs_signal( i, zwfs_ns.reco.I0, zwfs_ns.pupil_regions.pupil_filt ) 

## sanity check that Eigen_TT-HO, and Eigen_TT-HO_2 are equivilant (just Eigen_TT-HO projects to a 2 dimensional TT mode rather than the Poke eigenbasis)
zwfs_ns_2.reco.M2C_TT @ zwfs_ns_2.reco.I2M_TT @ s

cmd = util.get_DM_command_in_2D(  zwfs_ns.dm.current_cmd - zwfs_ns.dm.dm_flat ) 
reco_cmd = util.get_DM_command_in_2D(  1/zwfs_ns_2.reco.poke_amp/2  * zwfs_ns.reco.M2C_TT @ zwfs_ns.reco.I2M_TT @ s )
reco_cmd_2 = util.get_DM_command_in_2D( 1/zwfs_ns_2.reco.poke_amp/2 * zwfs_ns_2.reco.M2C_TT @ zwfs_ns_2.reco.I2M_TT @ s )
fig,ax = plt.subplots( 4,1 )
im0=ax[0].imshow( cmd )
im1=ax[1].imshow( reco_cmd - reco_cmd_2 )    
im2=ax[2].imshow( reco_cmd_2 )
im3=ax[3].imshow( cmd - reco_cmd_2 )
for i,axx in zip([im0,im1,im2, im3],ax.reshape(-1)):
    plt.colorbar(i, ax=axx)
plt.show()

plt.show() 
plt.imshow( util.get_DM_command_in_2D( zwfs_ns_2.reco.M2C_TT @ zwfs_ns_2.reco.I2M_TT @ s ) ) ;plt.show()

TT_reco_list = []
HO_reco_list = []
amp_grid = np.linspace( -1, 1, 20)

TT_idx = 0
for amp in amp_grid:
    zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat + amp * TT_vectors[:,TT_idx] 

    i = bldr.get_frame( opd_input,  amp_input ,  opd_internal,  zwfs_ns , detector= detector )

    #bldr.plot_current_dm_cmd(zwfs_ns)
    #bldr.plot_zwfs_signal( i ,zwfs_ns ) 

    s = bldr.process_zwfs_signal( i, zwfs_ns.reco.I0, zwfs_ns.pupil_regions.pupil_filt ) 

    #TT_modes = ( M2C_0 @ zwfs_ns.reco.I2M_TT @ s)[:2]
    #HO_modes = ( zwfs_ns.reco.I2M_HO @ s )[2:]
    #TT_reco_list.append( TT_modes ) 
    #HO_reco_list.append( HO_modes )
    
    TT_reco_list.append( zwfs_ns.reco.I2M_TT @ s )
    HO_reco_list.append( zwfs_ns.reco.I2M_HO @ s )
    
fig,ax = plt.subplots(2,1,sharex=True, figsize=(10,8))  
ax[0].plot( amp_grid, np.array( TT_reco_list)[:,TT_idx] , alpha = 1, color='g', label='Tilt') 
ax[0].plot( amp_grid, np.array( TT_reco_list )[:,1], alpha = 0.3, color='k', label='Tip') 
ax[1].plot( amp_grid, HO_reco_list, alpha=0.3, color='k' ) 
ax[1].plot( amp_grid, np.array(HO_reco_list)[:,1], alpha=0.3, color='k' ,label='HO modes')
ax[0].legend()
ax[1].legend() 
ax[0].axhline(0, ls=':',color='r')
ax[1].axhline(0, ls=':',color='r')
ax[0].axvline(0, ls=':',color='r')
ax[1].axvline(0, ls=':',color='r')
ax[1].set_xlabel( 'Tilt amplitude applied to DM')
ax[0].set_ylabel('reconstructed TT mode amplitude')
ax[1].set_ylabel('reconstructed HO mode amplitude')
plt.savefig( fig_path + 'simulation_mode_cross_coupling_vs_tip_aberration.png' , bbox_inches='tight', dpi=200)
plt.show()






U, S, Vt = np.linalg.svd( zwfs_ns.reco.IM.T, full_matrices=False)

plt.figure(1); plt.imshow( util.get_DM_command_in_2D( (zwfs_ns.reco.M2C_0.T @ Vt[1] ) ) )

plt.figure(2); plt.imshow( util.get_DM_command_in_2D( (zwfs_ns.reco.M2C_0.T @ Vt[0] ) ) )


Tip_new = zwfs_ns.reco.M2C_0.T @ Vt[0] 
Tilt_new = zwfs_ns.reco.M2C_0.T @ Vt[1] 

amp_grid = np.linspace(-1, 1, 15)
I2M_HO = (1/S * U).T  
M2C_HO = zwfs_ns.reco.M2C_0.T @ Vt
TT_idx = 0
TT_reco_list = []
HO_reco_list = []
for amp in amp_grid:
    zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat + amp * Tip_new 

    i = bldr.get_frame( opd_input,  amp_input ,  opd_internal,  zwfs_ns , detector= detector )

    #bldr.plot_current_dm_cmd(zwfs_ns)
    #bldr.plot_zwfs_signal( i ,zwfs_ns ) 

    s = bldr.process_zwfs_signal( i, zwfs_ns.reco.I0, zwfs_ns.pupil_regions.pupil_filt ) 

    #TT_modes = ( M2C_0 @ zwfs_ns.reco.I2M_TT @ s)[:2]
    #HO_modes = ( zwfs_ns.reco.I2M_HO @ s )[2:]
    #TT_reco_list.append( TT_modes ) 
    #HO_reco_list.append( HO_modes )
    
    TT_reco_list.append( (I2M_HO @ s)[:2] )
    HO_reco_list.append( (I2M_HO @ s)[2:] )
    
fig,ax = plt.subplots(2,1,sharex=True, figsize=(10,8))  
ax[0].plot( amp_grid, np.array( TT_reco_list)[:,TT_idx] , alpha = 1, color='g', label='Tilt') 
ax[0].plot( amp_grid, np.array( TT_reco_list )[:,1], alpha = 0.3, color='k', label='Tip') 
ax[1].plot( amp_grid, HO_reco_list, alpha=0.3, color='k' ) 
ax[1].plot( amp_grid, np.array(HO_reco_list)[:,1], alpha=0.3, color='k' ,label='HO modes')
ax[0].legend()
ax[1].legend() 
ax[0].axhline(0, ls=':',color='r')
ax[1].axhline(0, ls=':',color='r')
ax[0].axvline(0, ls=':',color='r')
ax[1].axvline(0, ls=':',color='r')
ax[1].set_xlabel( 'Tilt amplitude applied to DM')
ax[0].set_ylabel('reconstructed TT mode amplitude')
ax[1].set_ylabel('reconstructed HO mode amplitude')
#plt.savefig( fig_path + 'simulation_mode_cross_coupling_vs_tip_aberration.png' , bbox_inches='tight', dpi=200)
plt.show()







