"""
An example
testing reconstructors in different signal space (pix vs DM) but here using the baldr_experiments.py module 
to run the experiements 
BaldrApp/baldrapp/common/baldr_experiments.py

key point is we use seperate zwfs_ns objects, one that was used for calibration  (pupil definition, matrix calibration, etc)
and one used for measuring (onsky) zwfs intensity (which could have different optical alignment, pupil definition etc)

"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from types import SimpleNamespace
import importlib 
import sys
import copy 

import os 
import time
from pathlib import Path
if sys.version_info < (3, 0):
    import ConfigParser
else:
    import configparser as ConfigParser

import aotools

#import pyzelda.zelda as zelda
import pyzelda.ztools as ztools
import pyzelda.utils.aperture as aperture



# path to the folder that contains your module/package
module_dir = Path('/Users/bencb/Documents/ASGARD/BaldrApp/')  # e.g. Path.home() / "projects/my_pkg/src"
sys.path.insert(0, str(module_dir))  

from baldrapp.common import baldr_core as bldr
#from baldrapp.common import DM_basis
from baldrapp.common import utilities as util
from baldrapp.common import phasescreens as ps
from baldrapp.common import DM_registration
from baldrapp.common import baldr_experiments as bld_experiment



def update_scintillation( high_alt_phasescreen , pxl_scale, wavelength, final_size = None,jumps = 1, propagation_distance=10000):
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
        amp = bldr.upsample(propagated_screen, final_size ) # This oversamples to nearest multiple size, and then pads the rest with repeated rows, not the most accurate but fastest. Negligible if +1 from even number
    else:
        amp = propagated_screen

    return( abs(amp) ) # amplitude of field, not intensity (amp^2)! rotate 90 degrees so not correlated with phase 


## HERE WE KEEP SYSTEMS PERFECT (NO RMSE ON DM FLAT)
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

# first stage AO 
ao_sys = "NAOMI faint (AT)"
Nmodes_removed = 14 #7            # pick the AO1 regime you want here
N_iter = 1000                   # total closed-loop iterations
N_burn = 0                   # throw away transient
jumps_per_iter = 1             # scintillation decorrelation per iter

phase_scaling_factor = 1.0
it_lag = 3 #10 #3 # how many exposures does fist stage AO lag 


pm = zwfs_ns.grid.pupil_mask.astype(bool)

# phase space basis 
basis_cropped = ztools.zernike.zernike_basis(
    nterms=np.max([Nmodes_removed, 5]),
    npix=zwfs_ns.grid.N
)
basis_template = np.zeros(zwfs_ns.grid.pupil_mask.shape)
basis = np.array([util.insert_concentric(np.nan_to_num(b, 0), basis_template) for b in basis_cropped])


# stellar
throughput = 1 #0.1
waveband = "H"
magnitude = 1 #-5
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

poke_amp= 0.05 # 0.02 
poke_method='double_sided_poke'
basis_name =  "Zonal"
Nmodes = 140
imgs_to_mean=10
use_pyZelda = False 

zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy()

#clear pupil intensity in pixelspace on internal source 
N0 = bldr.get_N0( calibration_opd_input,   calibration_amp_input ,  calibration_opd_internal,  zwfs_ns , detector=detector, use_pyZelda = False)

#ZWFS pupil intensity in pixelspace on internal source 
I0 = bldr.get_I0( calibration_opd_input,   calibration_amp_input ,  calibration_opd_internal,  zwfs_ns , detector=detector, use_pyZelda = False)

# Dark in pixel space
DARK = bldr.get_I0( calibration_opd_input,   0*calibration_amp_input ,  calibration_opd_internal,  zwfs_ns , detector=detector, use_pyZelda = False)


# classify the pupil regions 
zwfs_ns = bldr.classify_pupil_regions( opd_input = 0 * calibration_opd_internal ,  amp_input = calibration_amp_input, \
    opd_internal=calibration_opd_internal,  zwfs_ns = zwfs_ns , 
    detector=zwfs_ns.detector , pupil_diameter_scaling = 1.0, 
    pupil_offset = (0,0), use_pyZelda= False) 


#zwfs_ns.grid.pupil_mask = aperture.disc_obstructed(dim=int(grid_ns.dim), size= grid_ns.N, obs = 1100/8000, diameter=True, strict=False )
zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy()
# build interaction matrix now in pixel space with zonal basis (so we can register DM with inbuilt functions)
zwfs_ns = bldr.build_IM( zwfs_ns ,  calibration_opd_input = 0 * calibration_opd_internal , calibration_amp_input = calibration_amp_input , \
            opd_internal = calibration_opd_internal,  basis = basis_name , Nmodes =  Nmodes, poke_amp = poke_amp, poke_method = 'double_sided_poke',\
                imgs_to_mean = imgs_to_mean, detector=zwfs_ns.detector,use_pyZelda= False, normalization_method='clear pupil mean')


#from IM register the DM in the detector pixelspace 
zwfs_ns = bldr.register_DM_in_pixelspace_from_IM( zwfs_ns, plot_intermediate_results=True  )

# just look at the pupil 
util.nice_heatmap_subplots( im_list = [zwfs_ns.grid.pupil_mask], title_list=['Solarstein pupil']) 

##########
# Experiment 1 
##########




zwfs_ns_dict = {}

calibration_dict = {"sol_dm_zonal": {"what_space":"dm","HO_inv_method":'zonal'},
                    "sol_pix_eigen": {"what_space":"pix","HO_inv_method":'eigen'},
                    "AT_dm_zonal": {"what_space":"dm","HO_inv_method":'zonal'},
                    "AT_pix_eigen": {"what_space":"pix","HO_inv_method":'eigen'}
                    }

for tag, cal_sub_dict in calibration_dict.items():

    # fresh copy
    zwfs_ns_cal = copy.deepcopy(zwfs_ns)

    # flatten DM
    zwfs_ns_cal.dm.current_cmd = zwfs_ns_cal.dm.dm_flat.copy()

    # 
    if 'AT_' in tag :
        zwfs_ns_cal.grid.pupil_mask = aperture.baldr_AT_pupil( diameter=zwfs_ns_cal.grid.N, dim=int(zwfs_ns_cal.grid.dim), spiders_thickness=0.016, strict=False, cpix=False) #, padding_factor = 2 )
    
    # build interaction matrix
    zwfs_ns_cal = bldr.build_IM(
        zwfs_ns_cal,
        calibration_opd_input=0 * calibration_opd_internal,
        calibration_amp_input=calibration_amp_input,
        opd_internal=calibration_opd_internal,
        basis="TT_w_zonal",
        Nmodes=Nmodes,
        poke_amp=poke_amp,
        poke_method="double_sided_poke",
        imgs_to_mean=imgs_to_mean,
        detector=zwfs_ns.detector,
        use_pyZelda=False,
        normalization_method="clear pupil mean",
    )

    # build reconstructor
    _ = bldr.reco_method(
        zwfs_ns_cal,
        LO=2,
        LO_inv_method="eigen",
        HO_inv_method=cal_sub_dict['HO_inv_method'],
        project_out_of_LO=None,
        project_out_of_HO="lo_command",
        truncation_idx=30,
        filter_dm_pupil=None,
        eps=1e-12,
        what_space=cal_sub_dict['what_space'],
    )

    zwfs_ns_dict[tag] = zwfs_ns_cal

# # unpack if you want the original names
# zwfs_ns_CPM = zwfs_ns_dict["CPM"]
# zwfs_ns_SFM = zwfs_ns_dict["SFM"]


# ---

def make_ctrl(ki, leak, key):
    """
    Build a controller with dimensions that match the reconstructor.

    BUGFIX:
      - Use I2M_TT (not I2M_LO) because eval_onsky uses zwfs_ns_calibration.reco.I2M_TT.
      - Tie controller dimensions to a known calibration object (CPM here) ONLY for
        construction; we enforce per-config consistency below.
    """
    return bld_experiment.LeakyIntegratorController(
        n_lo=zwfs_ns_dict[key].reco.I2M_TT.shape[0],
        n_ho=zwfs_ns_dict[key].reco.I2M_HO.shape[0],
        ki_LO=ki,
        ki_HO=ki,
        leak=leak,
    )


def make_scrn_factory(*, nx_size, dx, r0, L0, seed=None):
    def scrn_factory():
        return ps.PhaseScreenKolmogorov(
            nx_size=nx_size,
            pixel_scale=dx,
            r0=r0,
            L0=L0,
            random_seed=seed,
        )
    return scrn_factory



# N0_list_AT = bld_experiment.update_N0(  zwfs_ns = zwfs_ns_dict['CPM_AT'],  
#             phasescreen = scrn, 
#             scintillation_screen = scint_phasescreen, 
#             update_scintillation_fn = update_scintillation,   # pass your update_scintillation
#             basis = basis,
#             detector=detector,
#             dx=dx,
#             amp_input_0=calibration_amp_input,
#             propagation_distance=propagation_distance,
#             static_input_field=None,
#             opd_internal=calibration_opd_internal, 
#             N_iter_estimation =100, 
#             it_lag=it_lag, 
#             Nmodes_removed=Nmodes_removed, 
#             phase_scaling_factor=1.0, 
#             include_scintillation=include_scintillation, 
#             jumps_per_iter=jumps_per_iter,  
#             verbose_every = 100 
#             )

# experiment grid
configs = [
    dict(
        name=k,
        loop_schedule=[(0, "open"), (100, "fast")],
        user_ref_intensities=None, #(zwfs_ns_dict[k].reco.I0, zwfs_ns_dict[k].reco.N0), #None,
        ctrl_slow=None,
        ctrl_fast=make_ctrl(ki=0.25, leak=0.95, key = k),
        disable_ho=[],
        cal_tag=k,  # used to deepcopy the relevant zwfs_ns object
        signal_space=calibration_dict[k]['what_space']
    ) for k in zwfs_ns_dict] + [
         dict(
        name='sol_AT_dm_zonal',
        loop_schedule=[(0, "open"), (100, "fast")],
        user_ref_intensities=None, #(zwfs_ns_dict[k].reco.I0, zwfs_ns_dict[k].reco.N0), #None,
        ctrl_slow=None,
        ctrl_fast=make_ctrl(ki=0.25, leak=0.95, key = "AT_dm_zonal"), # key is to ensure correct size 
        disable_ho=[],
        cal_tag='sol_AT_dm_zonal',  # used to deepcopy the relevant zwfs_ns object
        signal_space='dm')
    ]



def zwfs_cal_factory_from_cfg(cfg):
    tag = (cfg.get("cal_tag") or "")
    if 'sol_AT_dm_zonal' not in tag:
        return copy.deepcopy(zwfs_ns_dict[tag])
    else:
        return copy.deepcopy(zwfs_ns_dict['sol_dm_zonal']) # calibrated on solarstein pupil 


# callable function to copy the zwfs_ns for experiment grid
def zwfs_current_factory_from_cfg(cfg):
    tag = (cfg.get("cal_tag") or "")
    if 'sol_AT_dm_zonal' not in tag:
        return copy.deepcopy(zwfs_ns_dict[tag])
    else:
        return copy.deepcopy(zwfs_ns_dict['AT_dm_zonal']) # measuring on AT pupil

# run
results = {}
for cfg in configs:
    # BUGFIX: ensure controller dimensions match THIS config's calibration reconstructor
    # (otherwise SFM vs CPM could differ and the loop will error or behave incorrectly)
    zwfs_tmp = zwfs_cal_factory_from_cfg(cfg)
    if cfg.get("ctrl_fast") is not None:

        if (cfg["ctrl_fast"].n_lo != zwfs_tmp.reco.I2M_TT.shape[0]) or (cfg["ctrl_fast"].n_ho != zwfs_tmp.reco.I2M_HO.shape[0]):
            cfg["ctrl_fast"] = bld_experiment.LeakyIntegratorController(
                n_lo=zwfs_tmp.reco.I2M_TT.shape[0],
                n_ho=zwfs_tmp.reco.I2M_HO.shape[0],
                ki_LO=0.25,
                ki_HO=0.25,
                leak=0.95,
            )

    if cfg.get("ctrl_slow") is not None:
        if (cfg["ctrl_slow"].n_lo != zwfs_tmp.reco.I2M_TT.shape[0]) or (cfg["ctrl_slow"].n_ho != zwfs_tmp.reco.I2M_HO.shape[0]):
            cfg["ctrl_slow"] = bld_experiment.LeakyIntegratorController(
                n_lo=zwfs_tmp.reco.I2M_TT.shape[0],
                n_ho=zwfs_tmp.reco.I2M_HO.shape[0],
                ki_LO=cfg["ctrl_slow"].ki_LO[0] if np.ndim(cfg["ctrl_slow"].ki_LO) else cfg["ctrl_slow"].ki_LO,
                ki_HO=cfg["ctrl_slow"].ki_HO[0] if np.ndim(cfg["ctrl_slow"].ki_HO) else cfg["ctrl_slow"].ki_HO,
                leak=cfg["ctrl_slow"].leak_LO[0] if np.ndim(cfg["ctrl_slow"].leak_LO) else cfg["ctrl_slow"].leak_LO,
            )


    # ----------------------------------------
    # ALWAYS apply per-config gain masking
    # ----------------------------------------
    disable_ho = cfg.get("disable_ho", [])
    disable_lo = cfg.get("disable_lo", [])

    # ensure vector form
    cfg["ctrl_fast"].ki_LO = np.atleast_1d(cfg["ctrl_fast"].ki_LO).astype(float)
    cfg["ctrl_fast"].ki_HO = np.atleast_1d(cfg["ctrl_fast"].ki_HO).astype(float)

    if len(disable_lo):
        cfg["ctrl_fast"].ki_LO[np.asarray(disable_lo, int)] = 0.0
        cfg["ctrl_fast"].u_LO[np.asarray(disable_lo, int)] = 0.0  # flush state

    if len(disable_ho):
        cfg["ctrl_fast"].ki_HO[np.asarray(disable_ho, int)] = 0.0
        cfg["ctrl_fast"].u_HO[np.asarray(disable_ho, int)] = 0.0  # flush state

        
    results.update(
        bld_experiment.run_experiment_grid(
            zwfs_current_factory=lambda cfg=cfg: zwfs_current_factory_from_cfg(cfg),
            zwfs_cal_factory=lambda cfg=cfg: zwfs_cal_factory_from_cfg(cfg), #zwfs_cal_factory_from_cfg, 
            scrn_factory=make_scrn_factory(
                nx_size=grid_dict["dim"],
                dx=dx,
                r0=r0,
                L0=L0,
                seed=3,
            ),
            scint_factory=make_scrn_factory(
                nx_size=grid_dict["dim"],
                dx=dx,
                r0=r0_scint,
                L0=L0_scint,
                seed=3,
            ),
            basis=basis,
            detector=detector,
            amp_input_0=calibration_amp_input,
            dx=dx,
            propagation_distance=propagation_distance,
            update_scintillation_fn=update_scintillation,
            DM_interpolate_fn=DM_registration.interpolate_pixel_intensities,
            configs=[cfg],
            common_kwargs=dict(
                N_iter=600,
                N_burn=0,
                it_lag=it_lag,
                Nmodes_removed=Nmodes_removed,
                phase_scaling_factor=phase_scaling_factor,
                include_scintillation=include_scintillation,
                jumps_per_iter=jumps_per_iter,
                #signal_space="pix",
                opd_threshold=np.inf,
                verbose_every=100,
            ),
            keys_to_not_evaluate={"name","ctrl_fast","ctrl_slow","loop_schedule","user_ref_intensities","disable_ho"},
        )
    )
    # keys_to_not_evaluate are keys to not pass to underlying eval_onsky(..) method (these are more pre-config or descriptive keys in the configuration file)


widget_handles = bld_experiment.quicklook_experiment_results(
    results,
    labels=['AT_dm_zonal', 'AT_pix_eigen','sol_AT_dm_zonal'], #list(results.keys()),  # optional, default = all
    wvl0=zwfs_ns.optics.wvl0,                  # important for Strehl proxy
    init_ho=(0, 20),                           # initial HO mode range
    init_lo=(0, 2),                            # initial LO mode range (TT)
    plot_u=True,                               # show controller states
    plot_errors=True,                          # show modal errors
    interactive=True,                          # enable sliders + toggles
    show=True,
)

# HO modes 3,4 on 'CPM_AT_AT' seem probablamatic, lets be quantitative a look at bar plot RMSE and bias of HO and LO in each case 

# results is dict[name -> telem] returned by run_experiment_grid
bld_experiment.plot_offenders(
    results,
    metric="mean", # mean | rms
    include=("e_LO", "e_HO", "u_LO", "u_HO"),
    u_source="fast",
    topk=5,
    burn=np.min( np.where(np.sum( results['AT_dm_zonal']['u_HO_fast'] , axis=1) > 0)[0]),
    figsize=(14, 8),
)
plt.show()


# get results of the worst offenders for bias and RMSE e_HO in closed loop conditions
bias_offenders = bld_experiment.get_worst_offenders(
    results,
    measurement="e_HO",
    metric="mean",#"rms",
    top_n=5,
    absolute=True,
    burn_in=np.min( np.where(np.sum( results['CPM_Sol_Sol']['u_HO_fast'] , axis=1) > 0)[0]), # when HO is in closed loop
)

rmse_offenders = bld_experiment.get_worst_offenders(
    results,
    measurement="e_HO",
    metric="rms",#"rms",
    top_n=5,
    absolute=True,
    burn_in=np.min( np.where(np.sum( results['CPM_Sol_Sol']['u_HO_fast'] , axis=1) > 0)[0]),  # when HO is in closed loop
)

# ----------------------------
# Example
# ----------------------------
# Different mode indices per config (keyed by cfg["name"])
mode_indices_by_cfg = {k:bias_offenders[k]['indices'] for k in bias_offenders} #rmse_offenders}

fig, ax = bld_experiment.plot_modes_in_intensity_space_per_config(
    configs=configs,
    zwfs_cal_factory_from_cfg=zwfs_cal_factory_from_cfg,
    which="HO",
    mode_indices_by_cfg=mode_indices_by_cfg,
    pix_shape=(48, 48),
    cbar_location="right",
    clip_percentile=99.5,
    suptitle="Per-config HO I2M templates (custom indices)",
)
plt.show()