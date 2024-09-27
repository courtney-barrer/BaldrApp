
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import importlib 
import Baldr_closeloop as bldr
import DM_basis as gen_basis
import utilities as util
# importlib.reload( bldr )

################## TEST 0 
# configure our zwfs 
grid_dict = {
    "D":1, # diameter of beam 
    "N" : 64, # number of pixels across pupil
    "padding_factor" : 4, # how many pupil diameters fit into grid x axis
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

################## TEST 1
# check dm registration on pupil (wavespace)
zwfs_ns = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)

phi, phi_internal,  N0, I0, Intensity = bldr.test_propagation( zwfs_ns )

fig = plt.figure() 
im = plt.imshow( N0, extent=[np.min(zwfs_ns.grid.wave_coord.x), np.max(zwfs_ns.grid.wave_coord.x),\
    np.min(zwfs_ns.grid.wave_coord.y), np.max(zwfs_ns.grid.wave_coord.y)] )
cbar = fig.colorbar(im, ax=plt.gca(), pad=0.01)
cbar.set_label(r'Pupil Intensity', fontsize=15, labelpad=10)

plt.scatter(zwfs_ns.grid.dm_coord.act_x0_list_wavesp, zwfs_ns.grid.dm_coord.act_y0_list_wavesp, color='blue', marker='.', label = 'DM actuators')
plt.show() 

################## TEST 2
# check pupil intensities 
fig,ax = plt.subplots( 1,4 )
ax[0].imshow( bldr.get_DM_command_in_2D( zwfs_ns.dm.current_cmd ))
ax[0].set_title('dm cmd')
ax[1].set_title('OPD wavespace')
ax[1].imshow( phi )
ax[2].set_title('ZWFS Intensity')
ax[2].imshow( Intensity )
ax[3].set_title('ZWFS reference Intensity')
ax[3].imshow( I0 )


################## TEST 3 
# test updating the DM registration 
zwfs_ns = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)

# redefining the affine transform between DM coordinates and the wavefront space
a, b, c, d = zwfs_ns.grid.D/np.ptp(zwfs_ns.grid.wave_coord.x)/2, 0, 0, grid_ns.D/np.ptp(zwfs_ns.grid.wave_coord.x)/2  # Parameters for affine transform (identity for simplicity)

# offset 5% of pupil 
t_x, t_y = np.mean(zwfs_ns.grid.wave_coord.x) + 0.05 * zwfs_ns.grid.D, np.mean(zwfs_ns.grid.wave_coord.x)  # Translation in phase space

# we could also introduce mis-registrations by rolling input pupil 
dm_act_2_wave_space_transform_matrix = np.array( [[a,b,t_x],[c,d,t_y]] )

zwfs_ns = bldr.update_dm_registration( dm_act_2_wave_space_transform_matrix , zwfs_ns )

opd_atm, opd_internal, opd_dm, phi, N0, I0,  Intensity  = bldr.test_propagation( zwfs_ns )


fig = plt.figure() 
im = plt.imshow( N0, extent=[np.min(zwfs_ns.grid.wave_coord.x), np.max(zwfs_ns.grid.wave_coord.x),\
    np.min(zwfs_ns.grid.wave_coord.y), np.max(zwfs_ns.grid.wave_coord.y)] )
cbar = fig.colorbar(im, ax=plt.gca(), pad=0.01)
cbar.set_label(r'Pupil Intensity', fontsize=15, labelpad=10)

plt.scatter(zwfs_ns.grid.dm_coord.act_x0_list_wavesp, zwfs_ns.grid.dm_coord.act_y0_list_wavesp, color='blue', marker='.', label = 'DM actuators')
plt.show() 



################## TEST 4
# test DM basis generation 
basis_name_list = ['Hadamard', "Zonal", "Zonal_pinned_edges", "Zernike", "Zernike_pinned_edges", "fourier", "fourier_pinned_edges"]

fig, ax = plt.subplots( len( basis_name_list), len( basis_name_list) ,figsize=(15,15) )
for i,b in enumerate( basis_name_list ) :
    
    print( b )
    basis_test = gen_basis.construct_command_basis( basis= b, number_of_modes = 20, Nx_act_DM = 12, Nx_act_basis = 12, act_offset=(0,0), without_piston=True)
    
    for m in range( ax.shape[1] ):
        ax[ i, m ].imshow( util.get_DM_command_in_2D( basis_test.T[m] ) )
    
    ax[i, 0].set_ylabel( b )
    #print( basis_test )
    
#plt.savefig( "/Users/bencb/Downloads/baldr_bases.png",dpi=300)
plt.show() 



################## TEST 5
# Get reference intensities 
zwfs_ns = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)

opd_input = 0 * zwfs_ns.grid.pupil_mask *  np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

opd_internal = 10e-9 * zwfs_ns.grid.pupil_mask * np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

amp_input = 1e2 * zwfs_ns.grid.pupil_mask

I0 = bldr.get_I0(  opd_input  = opd_input ,   amp_input = amp_input,\
    opd_internal = opd_internal,  zwfs_ns= zwfs_ns , detector=None )

N0 = bldr.get_N0(  opd_input  = opd_input ,   amp_input = amp_input,\
    opd_internal = opd_internal,  zwfs_ns= zwfs_ns , detector=None )


################## TEST 6
# classify pupil regions and plot them
zwfs_ns = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)

opd_input = 0 * zwfs_ns.grid.pupil_mask *  np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

opd_internal = 10e-9 * zwfs_ns.grid.pupil_mask * np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

amp_input = 1e2 * zwfs_ns.grid.pupil_mask

zwfs_ns = bldr.classify_pupil_regions( opd_input,  amp_input ,  opd_internal,  zwfs_ns , detector=None)

fig,ax = plt.subplots( 1, len(zwfs_ns.pupil_regions.__dict__ ) ,figsize=(10,10) )
for k,axx in zip( zwfs_ns.pupil_regions.__dict__,ax.reshape(-1)):
    axx.imshow(zwfs_ns.pupil_regions.__dict__[k] )
    axx.set_title( k ) 
plt.show()


################## TEST 7
# Build IM  and look at Eigenmodes! 
zwfs_ns = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)

opd_input = 0 * zwfs_ns.grid.pupil_mask *  np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

opd_internal = 10e-9 * zwfs_ns.grid.pupil_mask * np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

amp_input = 1e4 * zwfs_ns.grid.pupil_mask

# we must first define our pupil regions before building 
zwfs_ns = bldr.classify_pupil_regions( opd_input,  amp_input ,  opd_internal,  zwfs_ns , detector=None)

basis_name_list = ['Hadamard', "Zonal", "Zonal_pinned_edges", "Zernike", "Zernike_pinned_edges", "fourier", "fourier_pinned_edges"]

# perfect field only with internal opd aberrations 
# different poke methods 
Nmodes = 100
basis = 'Zonal_pinned_edges'
M2C_0 = gen_basis.construct_command_basis( basis= basis, number_of_modes = Nmodes, without_piston=True).T  

IM_00 = bldr.build_IM( zwfs_ns ,  calibration_opd_input= 0 *zwfs_ns.grid.pupil_mask , calibration_amp_input = amp_input , \
    opd_internal = opd_internal,  basis = basis, Nmodes =  Nmodes, poke_amp = 0.05, poke_method = 'double_sided_poke',\
        imgs_to_mean = 1, detector=None)

IM_10 = bldr.build_IM( zwfs_ns ,  calibration_opd_input= 0 *zwfs_ns.grid.pupil_mask , calibration_amp_input = amp_input , \
    opd_internal = opd_internal,  basis = basis, Nmodes =  Nmodes, poke_amp = 0.05, poke_method = 'single_sided_poke',\
        imgs_to_mean = 1, detector=None)

# different basis 
basis = 'Hadamard'
M2C_0 = gen_basis.construct_command_basis( basis= basis, number_of_modes = Nmodes, without_piston=True).T  

IM_11 = bldr.build_IM( zwfs_ns ,  calibration_opd_input= 0 *zwfs_ns.grid.pupil_mask , calibration_amp_input = amp_input , \
    opd_internal = opd_internal,  basis = basis, Nmodes =  Nmodes, poke_amp = 0.05, poke_method = 'double_sided_poke',\
        imgs_to_mean = 1, detector=None)

# compare SVDs 
if 1: 
    U,S,Vt = np.linalg.svd( IM_00 , full_matrices=True)

    #singular values
    plt.figure() 
    plt.semilogy(S) #/np.max(S))
    #plt.axvline( np.pi * (10/2)**2, color='k', ls=':', label='number of actuators in pupil')
    plt.legend() 
    plt.xlabel('mode index')
    plt.ylabel('singular values')

    #plt.savefig(current_path + f'singularvalues_{tstamp}.png', bbox_inches='tight', dpi=200)
    plt.show()
    
    # THE IMAGE MODES 
    n_row = round( np.sqrt( Nmodes ) ) - 1
    fig,ax = plt.subplots(n_row  ,n_row ,figsize=(30,30))
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    for i,axx in enumerate(ax.reshape(-1)):
        # we filtered circle on grid, so need to put back in grid
        tmp =  zwfs_ns.pupil_regions.pupil_filt.copy()
        vtgrid = np.zeros(tmp.shape)
        vtgrid[tmp] = Vt[i]
        r1,r2,c1,c2 = 10,-10,10,-10
        axx.imshow( vtgrid.reshape( zwfs_ns.pupil_regions.pupil_filt.shape )[r1:r2,c1:c2] ) #cp_x2-cp_x1,cp_y2-cp_y1) )
        #axx.set_title(f'\n\n\nmode {i}, S={round(S[i]/np.max(S),3)}',fontsize=5)
        #
        axx.text( 10,10,f'{i}',color='w',fontsize=4)
        axx.text( 10,20,f'S={round(S[i]/np.max(S),3)}',color='w',fontsize=4)
        axx.axis('off')
        #plt.legend(ax=axx)
    plt.tight_layout()

    #plt.savefig(current_path + f'det_eignmodes_{tstamp}.png',bbox_inches='tight',dpi=200)
    plt.show()
    
    # THE DM MODES 

    # NOTE: if not zonal (modal) i might need M2C to get this to dm space 
    # if zonal M2C is just identity matrix. 
    fig,ax = plt.subplots(n_row, n_row, figsize=(30,30))
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    for i,axx in enumerate(ax.reshape(-1)):
        axx.imshow( util.get_DM_command_in_2D( M2C_0.T @ U.T[i] ) )
        #axx.set_title(f'mode {i}, S={round(S[i]/np.max(S),3)}')
        axx.text( 1,2,f'{i}',color='w',fontsize=6)
        axx.text( 1,3,f'S={round(S[i]/np.max(S),3)}',color='w',fontsize=6)
        axx.axis('off')
        #plt.legend(ax=axx)
    plt.tight_layout()
    #plt.savefig(current_path + f'dm_eignmodes_{tstamp}.png',bbox_inches='tight',dpi=200)
    plt.show()




################## TEST 8
# project onto TT and HO

# Build IM  and look at Eigenmodes! 
zwfs_ns = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)

opd_input = 0 * zwfs_ns.grid.pupil_mask *  np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

opd_internal = 10e-9 * zwfs_ns.grid.pupil_mask * np.random.randn( *zwfs_ns.grid.pupil_mask.shape)

amp_input = 1e4 * zwfs_ns.grid.pupil_mask

# we must first define our pupil regions before building 
zwfs_ns = bldr.classify_pupil_regions( opd_input,  amp_input ,  opd_internal,  zwfs_ns , detector=None)

basis_name_list = ['Hadamard', "Zonal", "Zonal_pinned_edges", "Zernike", "Zernike_pinned_edges", "fourier", "fourier_pinned_edges"]

# perfect field only with internal opd aberrations 
# different poke methods 
Nmodes = 100
basis = 'Zonal_pinned_edges'
poke_amp = 0.05
M2C_0 = gen_basis.construct_command_basis( basis= basis, number_of_modes = Nmodes, without_piston=True).T  

IM_00 = bldr.build_IM( zwfs_ns ,  calibration_opd_input= 0 *zwfs_ns.grid.pupil_mask , calibration_amp_input = amp_input , \
    opd_internal = opd_internal,  basis = basis, Nmodes =  Nmodes, poke_amp = poke_amp, poke_method = 'double_sided_poke',\
        imgs_to_mean = 1, detector=None)


# -- Build our matricies 

U, S, Vt = np.linalg.svd( IM, full_matrices=False)

Smax = 30
R  = (Vt.T * [1/ss if i < Smax else 0 for i,ss in enumerate(S)])  @ U.T

TT_vectors = gen_basis.get_tip_tilt_vectors()

TT_space = M2C_0 @ TT_vectors
    
U_TT, S_TT, Vt_TT = np.linalg.svd( TT_space, full_matrices=False)

I2M_TT = U_TT.T @ R.T 

M2C_TT = poke_amp * M2C_0.T @ U_TT # since pinned need M2C to go back to 140 dimension vector  

R_HO = (np.eye(U_TT.shape[0])  - U_TT @ U_TT.T) @ R.T

# go to Eigenmodes for modal control in higher order reconstructor
U_HO, S_HO, Vt_HO = np.linalg.svd( R_HO, full_matrices=False)
I2M_HO = Vt_HO  
M2C_HO = poke_amp *  M2C_0.T @ (U_HO * S_HO) # since pinned need M2C to go back to 140 dimension vector


plt.figure(); plt.imshow( util.get_DM_command_in_2D( M2C_HO @ I2M_HO @ IM[65] ) ); plt.show()
 
 tmp.reshape(-1)[zwfs_ns.pupil_regions.pupil_filt.reshape(-1)] =  IM.T[65]
################## TEST 9 
#  Reconstruction static 
