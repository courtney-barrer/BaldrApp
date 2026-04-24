## RUn ./shm_creator_sim  first to create the shm in the right format 
# and size (do not pass data here - otherwise it will overwrite and recreate - but the python shmlib.py wrapper is different format than the rtc required format)
import numpy as np
import json 
import zmq
import time 
import os
from xaosim.shmlib import shm
import subprocess
from pathlib import Path


###### TESTING THE ACTUAL BALDR SIMULATION FROM BALDRAPP  
## note this should be installed in virtual environment (venv) with pyZelda fork also installed!

from baldrapp.common import baldr_core as bldr
from baldrapp.common import utilities as util
from importlib.resources import files


def get_git_root() -> Path:
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    )



# THIS SHOULD BE PUT IN util..
def convert_12x12_to_140(arr):
    # Convert input to a NumPy array (if it isn't already)
    arr = np.asarray(arr)
    
    if arr.shape != (12, 12):
        raise ValueError("Input must be a 12x12 array.")
    
    # Flatten the array (row-major order)
    flat = arr.flatten()
    
    # The indices for the four corners in a 12x12 flattened array (row-major order):
    # Top-left: index 0
    # Top-right: index 11
    # Bottom-left: index 11*12 = 132
    # Bottom-right: index 143 (11*12 + 11)
    corner_indices = [0, 11, 132, 143]
    
    # Delete the corner elements from the flattened array
    vector = np.delete(flat, corner_indices)
    
    return vector


root_path = get_git_root()

# configuration file (pyZelda style)
config_path = files("baldrapp.configurations") / "BALDR_UT_J3.ini"

# initialize our ZWFS instrument
wvl0=1.25e-6

zwfs_ns = {i:None for i in [1,2,3,4]} #zwfs namespace per telescope 

for i in [1,2,3,4]:  

    zwfs_ns[i] = bldr.init_zwfs_from_config_ini( config_ini=config_path, wvl0=wvl0 )
    zwfs_ns[i].dm.actuator_coupling_factor = 0.9

    ## these things can just be defined equally for each telescope 
    dx = zwfs_ns[i].grid.D / zwfs_ns[i].grid.N

    # input flux scaling (photons / s / wavespace_pixel / nm) 
    photon_flux_per_pixel_at_vlti = zwfs_ns[i].throughput.vlti_throughput * (np.pi * (zwfs_ns[i].grid.D/2)**2) / (np.pi * zwfs_ns[i].pyZelda.pupil_diameter/2)**2 * util.magnitude_to_photon_flux(magnitude=zwfs_ns[i].stellar.magnitude, band = zwfs_ns[i].stellar.waveband, wavelength= 1e9*wvl0)

    # internal aberrations (opd in meters)
    opd_internal = util.apply_parabolic_scratches(np.zeros( zwfs_ns[i].grid.pupil_mask.shape ) , dx=dx, dy=dx, list_a= [ 0.1], list_b = [0], list_c = [-2], width_list = [2*dx], depth_list = [100e-9])

    opd_flat_dm = bldr.get_dm_displacement( command_vector= zwfs_ns[i].dm.dm_flat  , gain=zwfs_ns[i].dm.opd_per_cmd, \
                    sigma= zwfs_ns[i].grid.dm_coord.act_sigma_wavesp, X=zwfs_ns[i].grid.wave_coord.X, Y=zwfs_ns[i].grid.wave_coord.Y,\
                        x0=zwfs_ns[i].grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns[i].grid.dm_coord.act_y0_list_wavesp )

    #telescope.append( zwfs_ns[i] )
    #del zwfs_ns[i]

# # reference pupils 
# I0 = bldr.get_I0( opd_input = 0 * zwfs_ns[i].pyZelda.pupil ,  amp_input = photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns[i].pyZelda.pupil, opd_internal=zwfs_ns[i].pyZelda.pupil * (opd_internal + opd_flat_dm), \
#     zwfs_ns[i]=zwfs_ns[i], detector=zwfs_ns[i].detector, include_shotnoise=True , use_pyZelda = True)

# N0 = bldr.get_N0( opd_input = 0 * zwfs_ns[i].pyZelda.pupil ,  amp_input = photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns[i].pyZelda.pupil, opd_internal=zwfs_ns[i].pyZelda.pupil * (opd_internal + opd_flat_dm), \
#     zwfs_ns[i]=zwfs_ns[i], detector=zwfs_ns[i].detector, include_shotnoise=True , use_pyZelda = True)






# nrs should really be read from this file! 
split_filename = root_path / "baldrapp/apps/paranal_simulator/fake_configs/cred1_split.json"

# subframe size definition from CRED 1 server input json
with open(split_filename, 'r') as file:
    split_dict = json.load( file )

# frame sizes
#global_frame_size = [256,320] # [nx,ny]
nrs = 5#200 # number of reads without reset , make small for simulation mode for sake of time 
global_frame_size = [256, 320, nrs]  # e.g., nrs = 100
baldr_frame_sizes = []
baldr_frame_corners = []
for i in [1,2,3,4]:

    nx = split_dict[f"baldr{i}"]['xsz']
    ny = split_dict[f"baldr{i}"]['ysz']
    baldr_frame_sizes.append( [nx,ny] )
    
    x0 = split_dict[f"baldr{i}"]['x0']
    y0 = split_dict[f"baldr{i}"]['y0']
    baldr_frame_corners.append( [x0,y0] )


# init SHM object/lists
baldr_sub_shms = {i:None for i in [1,2,3,4]} #sorted(glob.glob(f"/dev/shm/baldr*.im.shm"))
dm_shms = {i:None for i in [1,2,3,4]}
global_frame_shm = None 

f_cred1_global = "/dev/shm/cred1.im.shm"
#if os.path.exists(f_cred1_global):
#    os.remove(f_cred1_global) 
#global_frame_shm = shm(f_cred1_global, data = np.zeros(global_frame_size) , nosem=False) 




global_frame_shm = shm(f_cred1_global,  nosem=False)  # do not pass data use ./shm_creator_sim!!! 
global_frame_shm.set_data( np.zeros(global_frame_size).astype(dtype=np.uint16) )  # create empty global frame shm)

# create SHM
for ct, i in enumerate([1,2,3,4]):

    f_baldr = f"/dev/shm/baldr{i}.im.shm"
    f_dm = f"/dev/shm/dm{i}.im.shm"


    #_ = global_frame_shm.shape  # This triggers metadata parsing inside shmlib
    #if os.path.exists(f_baldr):
    #    os.remove(f_baldr)  # force re-creation with correct semaphores
    
    ss = shm(f_baldr,  nosem=False) 
    ss.set_data( np.zeros(baldr_frame_sizes[ct]) )
    #baldr_sub_shms.append( shm(f_baldr, data= np.zeros(baldr_frame_sizes[i]), nosem=False) )
    baldr_sub_shms[i] = ss  # do not pass data use ./shm_creator_sim!!! 

    # should be running sim_mdm_server, if not we need to initialise as follows: shm(f_dm, data=np.zeros([12,12]), nosem=False)
    dm_shms[i] = shm(f_dm,  nosem=False) 



### read in the camera settings 

# --- Test Camera Server ---
ctx = zmq.Context()
cam_socket = ctx.socket(zmq.REQ)
cam_socket.connect("tcp://127.0.0.1:6667")
cam_socket.send_string('cli "gain"')
print("Camera reply:", cam_socket.recv_string())
# egg. should get aduoffset from here 
adu_offset = 1000

# --- Test MDS Server ---
mds_socket = ctx.socket(zmq.REQ)
mds_socket.connect("tcp://127.0.0.1:5555")
mds_socket.send_string("on SBB")
print("MDS reply (on):", mds_socket.recv_string())
mds_socket.send_string("off SBB")
print("MDS reply (off):", mds_socket.recv_string())


# setup parameters 
default_tel = 1 # default telescope to use to set up parameters that are common amgost all telescopes
noise_std = 100
amp_input =  photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns[default_tel].pyZelda.pupil
use_pyZelda = True
assert len(dm_shms) == len(baldr_sub_shms)


#move to the configured phasemask in MDS state 
for beam in [1,2,3,4]:
    mds_socket.send_string(f"fpm_move {beam} J3")
    print( mds_socket.recv_string() )

# configured in phaseshift 
theta_in = zwfs_ns[default_tel].optics.theta # do i need to do this through zwfs_ns[i].pyZelda?
mask_depth_in = zwfs_ns[default_tel].pyZelda.mask_depth

liveindex = global_frame_shm.mtdata['cnt0'] #0 # counter 
while True:

    # the simulator is responsible for checking the MDS and Camera server state and updating 
    # the simulation accordinly 

    # key baldr states are 
    # source 
    # phasemask position (mask name)
    # collimator lens (bright / faint mode)

    for i in [1,2,3,4]:
        mds_socket.send_string(f"fpm_whereami {i}")
        mask = mds_socket.recv_string()  # to respect zmq order 
        # here we only check for mask out, and assume the phasemask doesnt swap to another one during the siumulatuon 
        
        ## this only properly works if we do zwfs_ns per beam!!! 
        if mask == "": 
            mask_out = True 
            zwfs_ns[i].optics.theta = 0 # we have no phaseshift from mask 
            if use_pyZelda:
                zwfs_ns[i].pyZelda._mask_depth = 0


        else:
            mask_out = False 
            zwfs_ns[i].optics.theta = theta_in # we use our configured phaseshift 
            if use_pyZelda:
                zwfs_ns[i].pyZelda._mask_depth = mask_depth_in

    # the simulator is responsible for updating the SHM counters in a consistent way with the camera server
    # live index is defined around line 613 in asgard-cred1-server/asgard_ZMQ_CRED1_server.c
    #.   liveindex = shm_img->md->cnt0 % shm_img->md->size[2];
    #.    shm_img->md->cnt1 = liveindex;       // idem

    #       shm_img->md->cnt0++;                 // increment internal counter


    # we do similar here 

    cnt0 = liveindex 
    cnt1 = liveindex % global_frame_size[2]
    
    for ct, i in enumerate([1,2,3,4]): #range(len(dm_shms)):
        # reads the combined channel of the DMs (no need to update here - this is done in the RTC )
        dmcmd_2d = dm_shms[i].get_data() 
        
        # update dm in simulation (2D function to 1D)
        dmcmd = convert_12x12_to_140( dmcmd_2d )
        
        # rtc only returns ch2 , we can put the simulated flat (zwfs_ns[i].dm.dm_flat.copy()) on ch0 somewhere else
        zwfs_ns[i].dm.current_cmd = dmcmd.copy()
        
        # update opd space with current dm shape
        opd_current_dm = bldr.get_dm_displacement( command_vector = zwfs_ns[i].dm.current_cmd   , gain=zwfs_ns[i].dm.opd_per_cmd, \
                        sigma = zwfs_ns[i].grid.dm_coord.act_sigma_wavesp, X=zwfs_ns[i].grid.wave_coord.X, Y=zwfs_ns[i].grid.wave_coord.Y,\
                            x0=zwfs_ns[i].grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns[i].grid.dm_coord.act_y0_list_wavesp )
            
            
        intensity = bldr.get_frame(  opd_input  = opd_current_dm,   amp_input = amp_input,\
            opd_internal = opd_internal,  zwfs_ns= zwfs_ns[i] , detector= zwfs_ns[i].detector, use_pyZelda = use_pyZelda )
        
        #print( np.mean( dmcmd ) )
        # ---------------------------------------
        # tmp simulation (fill this section with my baldrApp)
        subim_tmp =  adu_offset + noise_std * np.random.randn( *baldr_frame_sizes[ct] )

        # insert image
        simImg = util.insert_concentric(intensity , np.zeros_like( subim_tmp ) ) # insert 
        subim_tmp += simImg # add ontop
        
        # ---------------------------------------

        # update the shm 
        baldr_sub_shms[i].set_data( subim_tmp.astype(dtype=np.int32) )
        baldr_sub_shms[i].mtdata['cnt0'] = cnt0 
        baldr_sub_shms[i].mtdata['cnt1'] = cnt1 
        # post semaphore 
        baldr_sub_shms[i].post_sems(1)

    ### CHECK 
    #print( baldr_sub_shms[i].get_data()[:5] )
    # working 

    # now do global frame (this is not necessary for the RTC but nice if we want to look at the shmview to get a sense of how things are working)

    # some random noise across the image 
    
    
    #### Baldr shm are 2D in og camera server, but global cred1 is 3D (nrs)

    global_frame_shm.mtdata['cnt0'] = cnt0
    global_frame_shm.mtdata['cnt1'] = cnt1 
    global_im_tmp = adu_offset + noise_std * np.random.randn( *global_frame_size[:-1] )
    ###frame_index = global_frame_shm.mtdata['cnt1'] % global_frame_size[2]
    for ct, i in enumerate([1,2,3,4]):#range(len(dm_shms)):
        x0, y0    = baldr_frame_corners[ct]
        xsz, ysz  = baldr_frame_sizes[ct]

        # add in the baldr subframe to the global frame 
        global_im_tmp[y0:y0+ysz,x0:x0+xsz] = baldr_sub_shms[i].get_data().copy()


    gframe_now = global_frame_shm.get_data().copy() 
    print( f"frame {cnt1}" ) 
    gframe_now[cnt1,:,:] = global_im_tmp # set slice to the current one 
    
    global_frame_shm.set_data( gframe_now.astype(np.uint16) )
    time.sleep(0.01) # just in case of write delays (bug shooting here, delete later)
    global_frame_shm.post_sems(1)

    #sub_im = baldr_sub_shms[i].get_data()

    liveindex += 1 # increment counter 

    #time.sleep(1)
    
    # global_im_tmp = adu_offset + noise_std * np.random.randn( *global_frame_size )
    # for i in range(len(dm_shms)):
    #     x0, y0    = baldr_frame_corners[i]
    #     xsz, ysz  = baldr_frame_sizes[i]

    #     # add in the baldr subframe to the global frame 
    #     global_im_tmp[y0:y0+ysz,x0:x0+xsz] = baldr_sub_shms[i].get_data().copy() 


        
    # global_frame_shm.set_data( global_im_tmp.astype(np.int32) )

    # global_frame_shm.post_sems(1)

    # #sub_im = baldr_sub_shms[i].get_data()

    # time.sleep(1)


# python3 -m venv venv 
# git clone BaldrApp
# cd /to/cloned/directory

# git clone pyZelda (fork)
# cd /to/cloned/directory
# pip install -e .

# pip install anything else you want (zmq, etc)

# cd dcs/simulation 
# run bash script to start servers (from dcs/simulation)

# ./shm_creator_sim 
# ./sim_mdm_server
# source venv/bin/activate
# python3 -i simulation/baldr_sim.py 


# see the shm:  shmview /dev/shm/cred1.im.shm
# dm lab gui : lab-MDM-control & (needs to be in venv.. install in base!)


####### OLD TESTING THINGS 


# #### eg. DMs 
# class SimDM:
#     def __init__(self, dmid=1):
#         self.dmid = dmid
#         self.shms = []
#         self.shm0 = None
#         self.setup_shm()

#     def setup_shm(self):
#         shmfs = sorted(glob.glob(f"/dev/shm/dm{self.dmid}disp*.im.shm"))
#         shmf0 = f"/dev/shm/dm{self.dmid}.im.shm"
#         self.nch = len(shmfs)

#         self.shms = [shm(f, nosem=False) for f in shmfs]

#         if self.nch > 0 and os.path.exists(shmf0):
#             self.shm0 = shm(shmf0, nosem=False)
#         else:
#             print(f"[WARNING] SHM for dm{self.dmid} not found. Is the sim server running?")

# dm = SimDM(dmid=1)
# # check data ch 1 
# dm.shms[1].get_data()

# # check data on combined channel
# dm.shm0.get_data()

# # set data on ch 1
# dm.shms[1].set_data( np.eye(12, dtype=np.int32) )

# # post semaphore to update it on combined ch 
# dm.shm0.post_sems(1)

# # check combined channel is updated
# dm.shm0.get_data()
