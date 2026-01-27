"""
lucky imaging and building new on-sky reference intensity I0 after first developing an internal performance metric model 
(continued from # #/Users/bencb/Documents/ASGARD/paranal_onsky_comissioning/simulation_experiments/lucky_imaging_experiement.py)

test strehl mode (piece wise) fitting on sky with different pupils 

also partially in /Users/bencb/Documents/ASGARD/paranal_onsky_comissioning/simulation_experiments/pupil_definition_experiment_2.py

.. here we try look in more detail, care about building a new I0 on sky with 
then how to optimally deal with IM and projecting AT pupil out of control is worth while 

we aim to analyse and IDEAL SYSTEM: no RON, no DM noise, no internal aberrations

- configure ZWFS and build IM 
- check reconstructor modal space on internal source are not biased (before moving to on-sky)
- build a strehl model on internal source and select optimal pixels (check filtering space)
- go on-sky, try lucky imaging to build a new I0 


"""




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from astropy.io import fits 
from types import SimpleNamespace
import importlib 
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
from scipy.signal import welch
from scipy.ndimage import binary_erosion 
from scipy.stats import pearsonr
from scipy.optimize import least_squares
import copy 

import os 
import time
from pathlib import Path
if sys.version_info < (3, 0):
    import ConfigParser
else:
    import configparser as ConfigParser
    
import pyzelda.zelda as zelda
import pyzelda.ztools as ztools
import pyzelda.utils.aperture as aperture

# path to the folder that contains your module/package
module_dir = Path('/Users/bencb/Documents/ASGARD/BaldrApp/')  # e.g. Path.home() / "projects/my_pkg/src"
sys.path.insert(0, str(module_dir))  

from baldrapp.common import baldr_core as bldr
from baldrapp.common import DM_basis
from baldrapp.common import utilities as util
from baldrapp.common import phasescreens as ps
from baldrapp.common import DM_registration
import aotools



def update_scintillation( high_alt_phasescreen , pxl_scale, wavelength, final_size = None,jumps = 1,propagation_distance=10000):
    for _ in range(jumps):
        high_alt_phasescreen.add_row()
    wavefront = np.exp(1J *  high_alt_phasescreen.scrn ) # amplitude mean ~ 1 
    propagated_screen = aotools.opticalpropagation.angularSpectrum(inputComplexAmp=wavefront,
                                                               z=propagation_distance, 
                                                               wvl=wavelength, 
                                                               inputSpacing = pxl_scale, 
                                                               outputSpacing = pxl_scale
                                                               )
    #print("upsample it scintillation screen")
    if final_size is not None:
        amp = util.upsample(propagated_screen, final_size ) # This oversamples to nearest multiple size, and then pads the rest with repeated rows, not the most accurate but fastest. Negligible if +1 from even number
    else:
        amp = propagated_screen

    return( abs(amp) ) # amplitude of field, not intensity (amp^2)! rotate 90 degrees so not correlated with phase 







### Set up manually 
### HERE WE KEEP SYSTEMS PERFECT (NO RMSE ON DM FLAT )
grid_dict = {
    "telescope":"solarstein", #"DISC", #'AT',
    "D":1.8, # diameter of beam 
    "N" : 72, #64, # number of pixels across pupil diameter
    "dim": 72 * 4, #64 * 4 #4 
    #"padding_factor" : 4, # how many pupil diameters fit into grid x axis
    # TOTAL NUMBER OF PIXELS = padding_factor * N 
    }

# I should include coldstop here!! 
optics_dict = {
    "wvl0" :1.65e-6, # central wavelength (m) 
    "F_number": 21.2, # F number on phasemask
    "mask_diam": 1.06, # diameter of phaseshifting region in diffraction limit units (physical unit is mask_diam * 1.22 * F_number * lambda)
    "theta": 1.57079, # phaseshift of phasemask 
    ### NEw have not consistenty propagate this in functions in baldr_core
    "coldstop_diam": 8.4, #8, #1.22 lambda/D units
    "coldstop_offset": (0,0) #(0,cs_offset) #(-cs_offset,cs_offset) #(cs_offset, 0.0)
}

dm_dict = {
    "dm_model":"BMC-multi-3.5",
    "actuator_coupling_factor": 0.75, #0.7,# std of in actuator spacing of gaussian IF applied to each actuator. (e.g actuator_coupling_factor = 1 implies std of poke is 1 actuator across.)
    "dm_pitch":1,
    "dm_aoi":0, # angle of incidence of light on DM 
    "opd_per_cmd" : 3e-6, # peak opd applied at center of actuator per command unit (normalized between 0-1) 
    "flat_rmse" : 0.0 # std (m) of flatness across Flat DM  
    }

grid_ns = SimpleNamespace(**grid_dict)
optics_ns = SimpleNamespace(**optics_dict)
dm_ns = SimpleNamespace(**dm_dict)

zwfs_ns = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)
zwfs_ns.stellar.bandwidth = 300 # spectral bandwidth in nm (Critical to include if we do mangitude studies)


# atmosphere 
#wvl0 =  zwfs_ns.optics.wvl0
dx = zwfs_ns.grid.D / zwfs_ns.grid.N
r0=0.1 #m
L0 = 0.1 #m

include_scintillation = True # to include scintillation?
r0_scint = 0.164
L0_scint = 10
r0_500 = 0.10 #m
seeing = 0.98 * 500e-9 / r0_500 * 3600 * 180/np.pi # 
r0 = (r0_500) * (zwfs_ns.optics.wvl0 / 0.5e-6) ** (6 / 5)
L0 = 25
propagation_distance = 10000 # scintillation

# input phase and scintillation screens 
scrn = ps.PhaseScreenKolmogorov(
    nx_size=zwfs_ns.grid.dim, pixel_scale=dx, r0=r0, L0=L0, random_seed=2
)
scint_phasescreen = aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman(
    nx_size=zwfs_ns.grid.dim, pixel_scale=dx, r0=r0_scint, L0=L0_scint, random_seed=2
)


# stellar
throughput = 1 #0.1
waveband = "H"
# magnitude of calibration source 
solarstein_mag = -5

# Baldr detector 
fps = 1730 # baldr camera fps 

detector = bldr.detector(binning=6 ,
                            dit=1/fps,
                            ron=0, #12,#15.0, #15.0, # 10 # 1
                            qe=0.7)
zwfs_ns.detector = detector


####### LETS BUILD IT MANUALLY 
calibration_opd_input=0 * np.zeros_like(zwfs_ns.grid.pupil_mask)

calibration_amp_input=(throughput *
            (np.pi * (zwfs_ns.grid.D/2)**2) / 
            (np.pi * (zwfs_ns.grid.N/2)**2) *
            util.magnitude_to_photon_flux(magnitude=solarstein_mag,
                                            band=waveband,
                                            wavelength=1e9*zwfs_ns.optics.wvl0))**0.5 * zwfs_ns.grid.pupil_mask

calibration_opd_internal = np.zeros_like(zwfs_ns.grid.pupil_mask)


# 0 - get clear pupil estimate 
zwfs_ns = bldr.classify_pupil_regions( opd_input = 0 * calibration_opd_internal ,  amp_input = calibration_amp_input, \
    opd_internal=calibration_opd_internal,  zwfs_ns = zwfs_ns , 
    detector=zwfs_ns.detector , pupil_diameter_scaling = 1.0, 
    pupil_offset = (0,0), use_pyZelda= False) 

# defined from pupil classification algorithm 
interior_pup_filt = binary_erosion(zwfs_ns.pupil_regions.pupil_filt * (~zwfs_ns.pupil_regions.secondary_strehl_filt), structure=np.ones((3, 3), dtype=bool))

N_N0_samples = 100
zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy()
N0_list = []
for _ in range(N_N0_samples):
    N0_sample = bldr.get_N0( calibration_opd_input,   
                    calibration_amp_input ,  
                    calibration_opd_internal,  
                    zwfs_ns , 
                    detector=detector, 
                    use_pyZelda = False)

    N0_list.append( N0_sample ) 

N0 = np.mean( N0_list , axis=0)
N0_norm = np.mean( N0[interior_pup_filt] ) 

util.nice_heatmap_subplots( im_list = [ N0 ] , cbar_label_list = ['N0'] )


# 1 init DM phase screens 
number_of_screen_initiations = 50
scrn_list = []
for _ in range(number_of_screen_initiations):
    #scrn = ps.PhaseScreenKolmogorov(nx_size=zwfs_ns.grid.N, pixel_scale=dx, r0=zwfs_ns.atmosphere.r0, L0=zwfs_ns.atmosphere.l0, random_seed=1)
    dm_scrn = ps.PhaseScreenKolmogorov(nx_size=24, pixel_scale = zwfs_ns.grid.D / 24, r0=r0, L0=10, random_seed=None)
    scrn_list.append( dm_scrn ) 

# 2. init telemetry to build model ()
telem = {
    "N0":N0,
    "N0_norm":N0_norm,
    "i":[],
    "s":[],
    "opd_rms_est":[], # opd
    "opd_rms_true":[] # opd 
}

# 3. measure telemetry 
scrn_scaling_grid = np.logspace(-1,0.2,5)
for it in range(len(scrn_list)):
    print( f"input aberation {it}/{len(scrn_list)}" )
    # roll screen
    #scrn.add_row()     
    for ph_scale in scrn_scaling_grid: 
        #scaling_factor=0.05, drop_indicies = [0, 11, 11 * 12, -1] , plot_cmd=False
        zwfs_ns.dm.current_cmd =  util.create_phase_screen_cmd_for_DM(scrn_list[it],  scaling_factor=ph_scale , drop_indicies = [0, 11, 11 * 12, -1] , plot_cmd=False) 
    
        opd_current_dm =  bldr.get_dm_displacement(
                            command_vector=zwfs_ns.dm.current_cmd,
                            gain=zwfs_ns.dm.opd_per_cmd,
                            sigma=zwfs_ns.grid.dm_coord.act_sigma_wavesp,
                            X=zwfs_ns.grid.wave_coord.X,
                            Y=zwfs_ns.grid.wave_coord.Y,
                            x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp,
                            y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp
                        )
        

        i = bldr.get_frame(
            np.zeros_like( calibration_opd_internal),
            calibration_amp_input,
            np.zeros_like( calibration_opd_internal),
            zwfs_ns, 
            detector=detector,
            use_pyZelda=False
        ).astype(float)

        s = i / N0_norm  # we do like this because its strehl model! 

        opd_true = np.std( opd_current_dm[zwfs_ns.grid.pupil_mask.astype(bool)] ) # *  2*np.pi / zwfs_ns.optics.wvl0 * (  opd_current_dm  )
        opd_est =  np.std( zwfs_ns.dm.opd_per_cmd * np.array( zwfs_ns.dm.current_cmd) )
        #plt.figure(); plt.imshow( util.get_DM_command_in_2D( zwfs_ns.dm.opd_per_cmd * np.array( zwfs_ns.dm.current_cmd)));plt.colorbar();plt.show()

        telem['i'].append( i )
        telem['s'].append( s )
        telem['opd_rms_true'].append( opd_true )
        telem['opd_rms_est'].append( opd_est )




correlation_map = util.compute_correlation_map(np.array( telem['s'] ), np.array( telem['opd_rms_est'] ) )


yy, xx = np.ogrid[:telem['s'][0].shape[0], :telem['s'][0].shape[0]]
snr = (np.mean( np.array( telem['s'] ) , axis =0 ) / np.std(  np.array( telem['s'] ) , axis =0  )) 
radial_constraint = ((xx - telem['s'][0].shape[0]//2)**2 + (yy - telem['s'][0].shape[0]//2)**2 <= 20**2) * ( (xx - telem['s'][0].shape[0]//2)**2 + (yy - telem['s'][0].shape[0]//2)**2 >= 6**2 )
# some criteria to filter (this could be way more advanced if we wanted)
strehl_filt = (correlation_map < -0.7) & (snr > 1.) & radial_constraint
strehl_pixels = np.where( strehl_filt )


util.nice_heatmap_subplots( im_list = [ correlation_map, strehl_filt ] , cbar_label_list = ['Pearson R','filt'] )
#savefig = save_results_path + 'strehl_vs_intensity_pearson_R.png' ) #fig_path + 'strehl_vs_intensity_pearson_R.png' )

plt.figure()
plt.plot( [np.mean( ss[strehl_filt] ) for ss in telem['s']] , 1e9 * np.array( telem['opd_rms_est'] )  ,'.', label='est')
plt.plot( [np.mean( ss[strehl_filt] ) for ss in telem['s']] , 1e9 * np.array( telem['opd_rms_true'] )  ,'.', label='true')
plt.xlabel('<s>')
plt.ylabel('OPD RMS [nm RMS]')
plt.legend()
plt.show()



filtered_sigs = np.array( [np.mean( ss[strehl_filt] ) for ss in telem['s']] )
opd_nm_est =  1e9 * np.array( telem['opd_rms_est'] ) 

opd_model_params = util.fit_piecewise_continuous(x=filtered_sigs, y=opd_nm_est, n_grid=80, trim=0.15)


#%% # verify it on-sky 

#####################################
# Init on-sky parameters 
# first stage AO 
ao_sys = "NAOMI faint (AT)"
Nmodes_removed = 14            # pick the AO1 regime you want here
N_iter = 1000                   # total closed-loop iterations
N_burn = 0                   # throw away transient
jumps_per_iter = 1             # scintillation decorrelation per iter

phase_scaling_factor = 1.0
it_lag = 3 # how many exposures does fist stage AO lag 

# CONVERT TO AT PUPIL 
zwfs_ns.grid.pupil_mask = aperture.baldr_AT_pupil( diameter=grid_ns.N, dim=int(grid_ns.dim), spiders_thickness=0.016, strict=False, cpix=False) 

pm = zwfs_ns.grid.pupil_mask.astype(bool) # pupil mask in wavespace

# phase space basis 
basis_cropped = ztools.zernike.zernike_basis(
    nterms=np.max([Nmodes_removed, 5]),
    npix=zwfs_ns.grid.N
)
basis_template = np.zeros(zwfs_ns.grid.pupil_mask.shape)
basis = np.array([util.insert_concentric(np.nan_to_num(b, 0), basis_template) for b in basis_cropped])



N_iter_onsky = 1000 

magnitude_onsky = 1 # on sky magnitude at zwfs_ns.optics.wvl0 given zwfs_ns.stellar.bandwidth

# amplitude on-sky 
amp_input_0 = (throughput *
        (np.pi * (zwfs_ns.grid.D / 2) ** 2) /
        (np.pi * (zwfs_ns.grid.N / 2) ** 2) *
        util.magnitude_to_photon_flux(
            magnitude=magnitude_onsky, band=waveband, wavelength=1e9 * zwfs_ns.optics.wvl0
        )) ** 0.5 * zwfs_ns.grid.pupil_mask

# for now no additional intenral OPD when going on-sky 
opd_internal_onsky = np.zeros_like( calibration_opd_internal )

phase_jumps_per_it = 1


################################
# Measure new N0 on-sky to calculate new N0_normalization parameter for opd model 


#general function now in baldr_core module that provides and estimate of the clear pupil on-sky 
N0_onsky_list = bldr.estimate_clear_pupil_onsky(
    zwfs_ns=zwfs_ns,
    detector=detector,
    # screen modules + params (match your sim settings, but independent seeds)
    ps_module=ps,
    aotools_module=aotools,
    dx=dx,
    r0=r0,
    L0=L0,
    phase_seed=43,                 # independent from your main scrn
    include_scintillation=include_scintillation,
    r0_scint=r0_scint,
    L0_scint=L0_scint,
    scint_seed=42,                 # independent from your main scint_phasescreen
    propagation_distance=propagation_distance,
    jumps_per_iter=jumps_per_iter,
    update_scintillation_fn=update_scintillation,
    amp_input_0=amp_input_0,
    # AO1 residual model
    it_lag=it_lag,
    Nmodes_removed=Nmodes_removed,
    basis=basis,
    phase_scaling_factor=phase_scaling_factor,
    ao1_add_rows_per_iter=phase_jumps_per_it ,       # matches your earlier logic
    # Monte Carlo samples
    N_samples=N_N0_samples,        # e.g. 100
    opd_internal_onsky=opd_internal_onsky,
    use_pyZelda=False,
    return_intermediates=False,
)

# our new N0 
N0_onsky = np.mean( N0_onsky_list, axis = 0 )


telem_onsky = {
    "N0":N0_onsky
    "i":[],
    "s":[],
    "opd_est_nm":[], # opd
    "opd_true_nm":[] # opd 
}
reco_list = [] # first stage AO rolling buffer to simulate latency (defined via 'it_lag' parameter)
for _ in range(int(it_lag)):  # populate the initial values 
    scrn.add_row()
    _, reco_1 = bldr.first_stage_ao(
        atm_scrn = scrn,
        Nmodes_removed=Nmodes_removed,
        basis=basis,
        phase_scaling_factor=phase_scaling_factor,
        return_reconstructor=True,
    )
    reco_list.append(reco_1)

# begin 
zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy()
for it in range(N_iter_onsky):
    if np.mod( it , 100 ) == 0:
        print( f"complete {it}/{N_iter}" )
    # --- evolve turbulence + AO1 residual (update every iter) ---
    for _ in range(phase_jumps_per_it):
        scrn.add_row()

    _, reco_1 = bldr.first_stage_ao(
        scrn,
        Nmodes_removed=Nmodes_removed,
        basis=basis,
        phase_scaling_factor=phase_scaling_factor,
        return_reconstructor=True
    )
    reco_list.append( reco_1 )

    ao_1 = basis[0] * (phase_scaling_factor * scrn.scrn - reco_list.pop(0)) # pop out the last one

    opd_input = phase_scaling_factor * zwfs_ns.optics.wvl0 / (2 * np.pi) * ao_1  # [m]

    # --- evolve scintillation + amplitude ---
    for _ in range(jumps_per_iter):
        scint_phasescreen.add_row()

    amp_scint = update_scintillation(
        high_alt_phasescreen=scint_phasescreen,
        pxl_scale=dx,
        wavelength=zwfs_ns.optics.wvl0,
        final_size=None,
        jumps=0,
        propagation_distance=propagation_distance
    )
    if include_scintillation:
        amp_input = amp_input_0 * amp_scint
    else:
        amp_input = amp_input_0 

    # --- apply current DM command to compute DM OPD contribution ---
    opd_dm = bldr.get_dm_displacement(
        command_vector=zwfs_ns.dm.current_cmd,
        gain=zwfs_ns.dm.opd_per_cmd,
        sigma=zwfs_ns.grid.dm_coord.act_sigma_wavesp,
        X=zwfs_ns.grid.wave_coord.X,
        Y=zwfs_ns.grid.wave_coord.Y,
        x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp,
        y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp
    )

    # residual OPD presented to ZWFS optics
    opd_total = opd_input + opd_dm # we dont need opd_internal explicity here since its accounted for internally in get_frame method 


    # --- ZWFS measurement ---
    i = bldr.get_frame(
        opd_total,
        amp_input,
        opd_internal_onsky,
        zwfs_ns, 
        detector=detector,
        use_pyZelda=False
    ).astype(float)

    perf_sig = np.mean( i[strehl_filt] / np.mean( N0_onsky[interior_pup_filt] )  )


    opd_rms_nm = util.piecewise_continuous(x=perf_sig, 
                          interc=opd_model_params['interc'], 
                          slope_1=opd_model_params['slope_1'], 
                          slope_2=opd_model_params['slope_2'], 
                          x_knee=opd_model_params['x_knee'])

    telem_onsky['i'].append( i )
    telem_onsky['s'].append( perf_sig )
    telem_onsky['opd_true_nm'].append( 1e9 * np.std( opd_total[zwfs_ns.grid.pupil_mask.astype(bool)] ) )
    telem_onsky['opd_est_nm'].append( opd_rms_nm )


# final verification plot: true vs predicted OPD (nm) + RMSE
y_true = np.asarray(telem_onsky["opd_true_nm"], dtype=float)
y_pred = np.asarray(telem_onsky["opd_est_nm"], dtype=float)

# if somehow y_pred is (N,1) or list of arrays, flatten safely
y_pred = np.array([float(np.ravel(v)[0]) for v in y_pred], dtype=float)

rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

plt.figure()
plt.plot(y_true, y_pred, ".", label=f"pred vs true (RMSE={rmse:.2f} nm)")
lo = float(np.min([y_true.min(), y_pred.min()]))
hi = float(np.max([y_true.max(), y_pred.max()]))
plt.plot([lo, hi], [lo, hi], "--", label="y = x")

plt.xlabel("True OPD RMS [nm]")
plt.ylabel("Predicted OPD RMS [nm]")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#%% Lucky imaging to update I0 on-sky

# prep for lucky_img function
def image_processing_fn(i, filt=strehl_filt):
    return np.mean( i[filt] ) 
keys = ["interc", "slope_1", "slope_2", "x_knee"]
model_params = {k: opd_model_params[k] for k in keys}

lucky_I0s = util.lucky_img( I0_meas=telem_onsky['i'], 
                       performance_model=util.piecewise_continuous, 
                       image_processing_fn=image_processing_fn, 
                       model_params=model_params, 
                       quantile_threshold = 0.05 , 
                       keep = "<threshold" )



unlucky_I0s = util.lucky_img( I0_meas=telem_onsky['i'], 
                       performance_model=util.piecewise_continuous, 
                       image_processing_fn=image_processing_fn, 
                       model_params=model_params, 
                       quantile_threshold = 0.95 , 
                       keep = ">threshold" )



N0_meas = telem_onsky["N0"]
I0_lucky = np.mean( lucky_I0s,axis=0)
I0_unlucky = np.mean( unlucky_I0s,axis=0)
util.nice_heatmap_subplots( im_list = [ I0_lucky, I0_unlucky ] , 
                           title_list = ['lucky I0', 'unlucky I0'],
                           cbar_label_list = ['Intensity [ADU]', 'Intensity [ADU]'] )


#Nice! 

#Now an Bayesian update 
# to do:    move machinary in /Users/bencb/Documents/ASGARD/paranal_onsky_comissioning/simulation_experiments/pupil_definition_experiment_2.0.py
#           to util or baldr_core, implment it here. And use theoretical model of pupil as prior! 

# theoretical model 
zwfs_ns_theory = copy.deepcopy(zwfs_ns)

I0_theory = bldr.get_I0( calibration_opd_input,   
                        calibration_amp_input ,  
                        calibration_opd_internal,  
                        zwfs_ns_theory , 
                        detector=detector, 
                        use_pyZelda = False)

N0_theory = bldr.get_N0( calibration_opd_input,   
                        calibration_amp_input ,  
                        calibration_opd_internal,  
                        zwfs_ns_theory , 
                        detector=detector, 
                        use_pyZelda = False)

I_prior = I0_theory / np.mean( N0_theory[interior_pup_filt]) 

I_meas = I0_lucky / np.mean( N0_meas[interior_pup_filt])


# center and derotate (returns the prior rotated to the measured frame )



aligned_intensities = util.align_prior_to_meas_using_spiders_matched_filter(
                                                                            I_meas=I_meas,
                                                                            I_prior=I_prior,
                                                                            N0_meas=N0_meas, 
                                                                            N0_prior=N0_theory, 
                                                                            pupil_mask=interior_pup_filt,
                                                                            detect_pupil_fn=util.detect_pupil,
                                                                            angle_search_deg=(-60, 60),      # keep it tight if you expect small rotations
                                                                            angle_step_deg=0.5,
                                                                            refine=True,
                                                                            refine_half_width_deg=2.0,
                                                                            refine_step_deg=0.05,
                                                                            debug_plot=False,
                                                                        )


# check (we generally want to use *_prior_aligned_in_meas_frame to always put in measurement frame)

util.nice_heatmap_subplots( im_list = [ I0_theory, I_meas, aligned_intensities['I_prior_aligned_in_meas_frame']], #aligned_intensities['I_prior_aligned'] ] , 
                           title_list = ['theory','meas','theory aligned'],
                           )


sigma_meas = np.std(np.array(unlucky_I0s) / np.mean( N0_meas[interior_pup_filt]) , axis=0) / np.sqrt(len( unlucky_I0s ))
alpha = 0.1 # 0.05 = strong prior, 0.2 = weakish prior , guided more by data 
sigma_prior = alpha * np.abs(I_prior)  # alpha ~ 0.05–0.2 typical

# Bayesian update
w_meas  = 1.0 / sigma_meas**2
w_prior = 1.0 / sigma_prior**2

I0_post = (w_meas * I_meas + w_prior * aligned_intensities['I_prior_aligned_in_meas_frame']) / (w_meas + w_prior)
# return also an updated version of N0 which we can use in the normalized signal calculation s = I/<N0>_p - I0_post / N0_post
N0_post = N0_meas / np.mean( N0_meas[interior_pup_filt] )


util.nice_heatmap_subplots( im_list = [ I0_theory, N0_theory, I_prior , I_meas, I0_post ] , 
                           title_list = ['I0 theory','N0 theory', r'$I0_{theory}/<N0_{theory}>_p$', r'$I0_{lucky}/<N0_{meas}>_p$',r'$I0_{post}/<N0_{post}>_p$'],
                           cbar_label_list = ['Intensity [ADU]', 'Intensity [ADU]', 'Signal [unitless]', 'Signal [unitless]', 'Signal [unitless]'] )



## Verification that rotate and align function works! 

from scipy.ndimage import rotate as nd_rotate
from scipy.ndimage import shift as nd_shift

def rot_shift(img, rot_deg=0.0, shift_yx=(0.0, 0.0), order=3):
    
    #Apply rotation about image center, then a (dy, dx) shift.
    
    out = nd_rotate(img, angle=rot_deg, reshape=False, order=order, mode="constant", cval=0.0)
    out = nd_shift(out, shift=shift_yx, order=order, mode="constant", cval=0.0)
    return out

inj_rot = +23.0
inj_shift_meas = (+2.7, -4.1)  # (dy, dx) in pixels
inj_shift_theo = (-4, 0) 
for misalign in ['theory','measure','both']:

    if misalign == 'theory':
        I0_prior_in = rot_shift(I0_theory, rot_deg=inj_rot, shift_yx=inj_shift_theo, order=3)
        N0_prior_in = rot_shift(N0_theory, rot_deg=inj_rot, shift_yx=inj_shift_theo, order=3)
        I0_meas_in = I0_lucky
        N0_meas_in = N0_meas
    elif misalign == 'measure' :
        I0_prior_in = I0_theory
        N0_prior_in = N0_theory
        I0_meas_in = rot_shift(I0_lucky, rot_deg=inj_rot, shift_yx=inj_shift_meas, order=3)
        N0_meas_in = rot_shift(N0_meas, rot_deg=inj_rot, shift_yx=inj_shift_meas, order=3)
    elif misalign == 'both' :
        I0_prior_in = rot_shift(I0_theory, rot_deg=inj_rot, shift_yx=inj_shift_theo, order=3)
        N0_prior_in = rot_shift(N0_theory, rot_deg=inj_rot, shift_yx=inj_shift_theo, order=3)
        I0_meas_in = rot_shift(I0_lucky, rot_deg=inj_rot, shift_yx=inj_shift_meas, order=3)
        N0_meas_in = rot_shift(N0_meas, rot_deg=inj_rot, shift_yx=inj_shift_meas, order=3)

    aligned = util.align_prior_to_meas_using_spiders_matched_filter(
        I_meas=I0_meas_in,
        I_prior=I0_prior_in,
        N0_meas=N0_meas_in,
        N0_prior=N0_prior_in,
        pupil_mask=interior_pup_filt,
        detect_pupil_fn=util.detect_pupil,
        angle_search_deg=(-60, 60),      # keep it tight if you expect small rotations
        angle_step_deg=0.5,
        refine=True,
        refine_half_width_deg=2.0,
        refine_step_deg=0.05,
        debug_plot=False,
    )


    # print(f"Injected PRIOR rot: {inj_rot:+.3f} deg, shift(dy,dx): {inj_shift}")
    # print(f"Recovered dtheta (applied to prior): {aligned['dtheta_deg']:+.3f} deg  (expect ~{-inj_rot:+.3f})")
    # print(f"Recovered shifts: meas {aligned['shift_meas']}, prior {aligned['shift_prior']}")

    util.nice_heatmap_subplots(
        im_list=[I0_meas_in, I0_prior_in, aligned["I_prior_centered"], aligned["I_prior_aligned"], aligned['I_prior_aligned_in_meas_frame']],
        title_list=["original measured","original prior","prior centered", f"prior aligned\n(dθ={aligned['dtheta_deg']:+.2f}°)","prior aligned \nto meas frame"]
    )


