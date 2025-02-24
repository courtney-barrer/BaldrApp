
import numpy as np #(version 2.1.1 works but incompatiple with numba)
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import pearsonr
import pickle
from types import SimpleNamespace
from sklearn.linear_model import LinearRegression
import importlib # reimport package after edits: importlib.reload(bldr)
import os
import datetime
import scipy.interpolate as interpolate
# from courtney-barrer's fork of pyzelda
import pyzelda.zelda as zelda
import pyzelda.ztools as ztools
import pyzelda.utils.aperture as aperture
import pyzelda.utils.imutils as imutils
from baldrapp.common import phasescreens as ps
from baldrapp.common import utilities as util
from baldrapp.common import baldr_core as bldr
from baldrapp.common import DM_registration
from baldrapp.common import DM_basis
from statsmodels.tsa.ar_model import AutoReg
import numpy as np
from scipy.linalg import block_diag



###############################################################################
# Build overall measurement matrix H_total
###############################################################################
def build_measurement_matrix(n_act, lags=20):
    """
    Build a measurement matrix H_total of shape (n_act, n_act*lags)
    that extracts the first state element (the current phase) for each actuator.
    """
    H_blocks = [np.hstack([np.eye(1), np.zeros((1, lags-1))]) for _ in range(n_act)]
    H_total = block_diag(*H_blocks)
    return H_total

###############################################################################
# Fit AR(20) model using statsmodels for each actuator
###############################################################################
def fit_AR_models(phi_history, lags=20):
    """
    Fits an AR(lags) model to the calibration time series for each actuator.
    
    Parameters:
        phi_history : numpy.ndarray of shape (N, n_act)
                      where N is the number of calibration iterations and
                      n_act is the number of DM actuators (e.g., 140).
        lags : int, the AR order (default 20).
        
    Returns:
        ar_params : numpy.ndarray of shape (n_act, lags+1)
                    Each row contains [intercept, a1, ..., a_lags] for that actuator.
        noise_var : numpy.ndarray of shape (n_act,)
                    Estimated residual variance for each actuator.
    """
    N, n_act = phi_history.shape
    ar_params = np.zeros((n_act, lags+1))
    noise_var = np.zeros(n_act)
    for i in range(n_act):
        ts = phi_history[:, i]
        # Fit AR model of specified order
        model = AutoReg(ts, lags=lags, old_names=False)
        res = model.fit()
        ar_params[i, :] = res.params  # first element is intercept, then coefficients
        noise_var[i] = res.sigma2
    return ar_params, noise_var


###############################################################################
# Build block-diagonal state-space matrices for all actuators
###############################################################################
def build_block_state_space(ar_params_all, noise_var_all, lags=20):
    """
    Given AR model parameters for each actuator, build the overall block-diagonal
    state-transition matrix A_total and process noise covariance Q_total.
    
    Parameters:
        ar_params_all : numpy.ndarray of shape (n_act, lags+1)
        noise_var_all : numpy.ndarray of shape (n_act,)
        lags : int, AR order.
    
    Returns:
        A_total : numpy.ndarray of shape (n_act*lags, n_act*lags)
        Q_total : numpy.ndarray of shape (n_act*lags, n_act*lags)
    """
    n_act = ar_params_all.shape[0]
    A_blocks = []
    Q_blocks = []
    for i in range(n_act):
        A_comp, Q_comp = build_state_space_from_AR(ar_params_all[i, :], noise_var_all[i], lags=lags)
        A_blocks.append(A_comp)
        Q_blocks.append(Q_comp)
    A_total = block_diag(*A_blocks)
    Q_total = block_diag(*Q_blocks)
    return A_total, Q_total



###############################################################################
# Kalman filter class (state-space version with block matrices)
###############################################################################
class BlockKalmanFilter:
    def __init__(self, A, Q, R, x0, P0):
        """
        A, Q, P0: state-space matrices of shape (n, n) where n = n_act * lags.
        R: measurement noise covariance of shape (n_act, n_act)
        x0: initial state (n x 1)
        """
        self.A = A
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        self.K = 0

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x

    def update(self, z, H):
        y = z - H @ self.x  # z is opd measurement (subtracting DM cmd estimate of opd. H just filters x for the mpst recent opd estimate (state) 
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.K = K
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P
        return self.x





###############################################################################
# Build companion state-space matrices for an AR(p) model (p=lags)
###############################################################################
def build_state_space_from_AR(ar_params, noise_var, lags=20):
    """
    Constructs the companion matrix A and process noise covariance Q for one actuator
    given the AR model parameters.
    
    For an AR(p) process:
        φ[t] = μ + a1 φ[t-1] + a2 φ[t-2] + ... + ap φ[t-p] + w[t]
    The state vector is:
        x[t] = [ φ[t], φ[t-1], ..., φ[t-p+1] ]^T,
    and the companion matrix is:
        A = [ a1, a2, ..., ap ]
            [ 1,  0, ..., 0  ]
            [ 0,  1, ..., 0  ]
            ...
            [ 0,  0, ..., 1  ]
    The process noise is assumed to affect only the first state element.
    
    Parameters:
        ar_params : array-like, shape (lags+1,)
                    [intercept, a1, ..., a_p]
        noise_var : float, estimated variance of the residual (w[t])
        lags : int, AR order.
    
    Returns:
        A_comp : numpy.ndarray of shape (lags, lags)
        Q_comp : numpy.ndarray of shape (lags, lags)
    """
    A_comp = np.zeros((lags, lags))
    # Use the AR coefficients (skip intercept) in the first row.
    A_comp[0, :] = ar_params[1:]
    # Place ones on the first subdiagonal.
    if lags > 1:
        A_comp[1:, :-1] = np.eye(lags-1)
    # Process noise only enters the first state element.
    Q_comp = np.zeros((lags, lags))
    Q_comp[0, 0] = noise_var
    return A_comp, Q_comp

proj_path = os.getcwd()
# initialize our ZWFS instrument
wvl0=1.25e-6
config_ini = proj_path  + '/baldrapp/configurations/BALDR_UT_J3.ini'#'/home/benja/Documents/BALDR/BaldrApp/configurations/BALDR_UT_J3.ini'
zwfs_ns = bldr.init_zwfs_from_config_ini( config_ini=config_ini , wvl0=wvl0)

fig_path = '/Users/bencb/Downloads/'#f'/home/benja/Downloads/act_cross_coupling_{zwfs_ns.dm.actuator_coupling_factor}_{tstamp}/' #f'/home/rtc/Documents/act_cross_coupling_{zwfs_ns.dm.actuator_coupling_factor}_{tstamp}/'
if os.path.exists(fig_path) == False:
    os.makedirs(fig_path) 
    

# Sampling parameters
dx = zwfs_ns.grid.D / zwfs_ns.grid.N
dt = dx * zwfs_ns.atmosphere.pixels_per_iteration / zwfs_ns.atmosphere.v
print(f'Effective wind velocity = {round(zwfs_ns.atmosphere.v)} m/s')

# Create a Kolmogorov phase screen to simulate turbulence:
scrn = ps.PhaseScreenKolmogorov(
    nx_size=zwfs_ns.grid.dim, 
    pixel_scale=dx, 
    r0=zwfs_ns.atmosphere.r0 * (wvl0 / 550e-9)**(6/5), 
    L0=zwfs_ns.atmosphere.l0, 
    random_seed=1
)
phase_scaling_factor = 0.3

# Build the interaction matrix and register the DM in pixel space:
basis_name = 'Zonal_pinned_edges'
Nmodes = 100
M2C_0 = DM_basis.construct_command_basis(basis=basis_name, number_of_modes=Nmodes, without_piston=True).T  

zwfs_ns = bldr.classify_pupil_regions(
    opd_input=0 * zwfs_ns.pyZelda.pupil,  
    amp_input=(zwfs_ns.throughput.vlti_throughput *
               (np.pi * (zwfs_ns.grid.D/2)**2) / 
               (np.pi * (zwfs_ns.pyZelda.pupil_diameter/2)**2) *
               util.magnitude_to_photon_flux(magnitude=zwfs_ns.stellar.magnitude,
                                              band=zwfs_ns.stellar.waveband,
                                              wavelength=1e9*wvl0))**0.5 * zwfs_ns.pyZelda.pupil,
    opd_internal=np.zeros_like(zwfs_ns.pyZelda.pupil),
    zwfs_ns=zwfs_ns,
    detector=zwfs_ns.detector,
    pupil_diameter_scaling=1.0,
    pupil_offset=(0,0)
)

zwfs_ns = bldr.build_IM(
    zwfs_ns,
    calibration_opd_input=0 * np.zeros_like(zwfs_ns.pyZelda.pupil),
    calibration_amp_input=(zwfs_ns.throughput.vlti_throughput *
                           (np.pi * (zwfs_ns.grid.D/2)**2) /
                           (np.pi * (zwfs_ns.pyZelda.pupil_diameter/2)**2) *
                           util.magnitude_to_photon_flux(magnitude=zwfs_ns.stellar.magnitude,
                                                          band=zwfs_ns.stellar.waveband,
                                                          wavelength=1e9*wvl0))**0.5 * zwfs_ns.pyZelda.pupil,
    opd_internal=np.zeros_like(zwfs_ns.pyZelda.pupil),
    basis=basis_name,
    Nmodes=Nmodes,
    poke_amp=0.05,
    poke_method='double_sided_poke',
    imgs_to_mean=1,
    detector=zwfs_ns.detector
)

zwfs_ns = bldr.register_DM_in_pixelspace_from_IM(zwfs_ns, plot_intermediate_results=True)


# zwfs_ns.dm.current_cmd=np.zeros(140)
# imgtmp = bldr.get_I0( opd_input = 0 * zwfs_ns.pyZelda.pupil ,  amp_input = photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns.pyZelda.pupil, opd_internal=zwfs_ns.pyZelda.pupil * (opd_internal + opd_flat_dm), \
#     zwfs_ns=zwfs_ns, detector=zwfs_ns.detector, include_shotnoise=True , use_pyZelda = True)

# x_target = np.array([ x for x,_ in zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space])
# y_target = np.array([ y for _,y in zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space])
# x_grid = np.arange( imgtmp.shape[0])
# y_grid = np.arange( imgtmp.shape[1])
# M = DM_registration.construct_bilinear_interpolation_matrix( imgtmp.shape, x_grid, y_grid, x_target, y_target)
# imgint_1 = M @ imgtmp.flatten()
# imgint_2 = DM_registration.interpolate_pixel_intensities(
#         image=imgtmp,
#         pixel_coords=zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space
#     )

# fig,ax = plt.subplots(1,4)
# ax[3].imshow(imgtmp)
# iii=ax[0].imshow(util.get_DM_command_in_2D(imgint_1))
# plt.colorbar(iii,ax=ax[0])
# ax[1].imshow(util.get_DM_command_in_2D(imgint_2))
# im=ax[2].imshow(util.get_DM_command_in_2D(imgint_2-imgint_1))
# plt.colorbar(im,ax=ax[2])
# plt.show()

# Precompute calibration quantities:
photon_flux_per_pixel_at_vlti = (
    zwfs_ns.throughput.vlti_throughput *
    (np.pi * (zwfs_ns.grid.D/2)**2) /
    (np.pi * (zwfs_ns.pyZelda.pupil_diameter/2)**2) *
    util.magnitude_to_photon_flux(magnitude=zwfs_ns.stellar.magnitude,
                                   band=zwfs_ns.stellar.waveband,
                                   wavelength=1e9*wvl0)
)
opd_internal = np.zeros_like(zwfs_ns.pyZelda.pupil)
opd_flat_dm = bldr.get_dm_displacement(
    command_vector=zwfs_ns.dm.dm_flat,
    gain=zwfs_ns.dm.opd_per_cmd,
    sigma=zwfs_ns.grid.dm_coord.act_sigma_wavesp,
    X=zwfs_ns.grid.wave_coord.X,
    Y=zwfs_ns.grid.wave_coord.Y,
    x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp,
    y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp
)

# Acquire calibration telemetry:
telemetry = {
    'dm_cmd': [],
    'i': [],
    'Ic': [],
    'i_dm': [],
    'strehl_0': []
}
telem_ns = SimpleNamespace(**telemetry)

iterations = 100
zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat.copy()
for it in range(iterations):
    print(f"Calibration iteration {it}")
    # Roll phase screen to simulate evolving turbulence
    for _ in range(10):
        scrn.add_row()
    
    # Generate a DM command from the phase screen
    zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat + util.create_phase_screen_cmd_for_DM(
        scrn,
        scaling_factor=phase_scaling_factor,
        drop_indicies=[0, 11, 11*12, -1],
        plot_cmd=False
    )
    
    # Compute current DM OPD:
    opd_current_dm = bldr.get_dm_displacement(
        command_vector=zwfs_ns.dm.current_cmd,
        gain=zwfs_ns.dm.opd_per_cmd,
        sigma=zwfs_ns.grid.dm_coord.act_sigma_wavesp,
        X=zwfs_ns.grid.wave_coord.X,
        Y=zwfs_ns.grid.wave_coord.Y,
        x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp,
        y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp
    )
    
    bldr_opd_map = np.sum([opd_internal, opd_current_dm], axis=0)
    Strehl_0 = np.exp(-np.var(2 * np.pi / zwfs_ns.optics.wvl0 * bldr_opd_map[zwfs_ns.pyZelda.pupil > 0.5]))
    Ic = photon_flux_per_pixel_at_vlti * zwfs_ns.pyZelda.propagate_opd_map(bldr_opd_map, wave=zwfs_ns.optics.wvl0)
    
    i = bldr.detect(
        Ic,
        binning=(zwfs_ns.detector.binning, zwfs_ns.detector.binning),
        qe=zwfs_ns.detector.qe,
        dit=zwfs_ns.detector.dit,
        ron=zwfs_ns.detector.ron,
        include_shotnoise=True,
        spectral_bandwidth=zwfs_ns.stellar.bandwidth
    )
    
    # Interpolate intensity onto the DM actuator grid.
    i_dm = DM_registration.interpolate_pixel_intensities(
        image=i,
        pixel_coords=zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space
    )
    
    telem_ns.i.append(i)
    telem_ns.Ic.append(Ic)
    telem_ns.i_dm.append(i_dm)
    telem_ns.strehl_0.append(Strehl_0)
    # make it relative to the flat 
    telem_ns.dm_cmd.append(zwfs_ns.dm.current_cmd )

# ---------------------------
# (a) CALIBRATE LINEAR MODEL FOR H (with intercept)
# ---------------------------
# We assume during calibration (E=0):
#      (dm_cmd-dm_flat) = H * i_dm + c
# For each actuator, we fit a linear regression with intercept.


#train
nn=len(telem_ns.i_dm)
X_cal = np.array(telem_ns.i_dm)[:nn//2]  # shape: (iterations, 140)
Y_cal = np.array(telem_ns.dm_cmd)[:nn//2] - zwfs_ns.dm.dm_flat  # shape: (iterations, 140)

#test set to estimate covariance and R matric
X_test= np.array(telem_ns.i_dm)[nn//2:]
Y_test= np.array(telem_ns.dm_cmd)[nn//2:] - zwfs_ns.dm.dm_flat 

lin_models = []
s,c=[],[]
Y_pred = np.zeros_like(Y_cal)
for act in range(Y_cal.shape[1]):
    
    stmp, ctmp = np.polyfit(X_cal[:, act], Y_cal[:, act],deg=1)
    s.append(stmp)
    c.append(ctmp)
    # lr = LinearRegression(fit_intercept=True)
    # lr.fit(X_cal[:, act], Y_cal[:, act])
    # Y_pred[:, act] = lr.predict(X_cal)
    # lin_models.append(lr)
s=np.array(s)
c=np.array(c)


Y_predtest = s * X_test + c
Y_pred = s * X_cal + c
# Y_predtest = np.zeros_like(Y_test)
# for act in range(Y_test.shape[1]):
#     Y_predtest[:, act] = lin_models[act].predict(X_test[:, act])


# --- Estimate the measurement noise covariance R ---
# Compute residuals between the actual DM command and the model prediction.
residuals = Y_test - Y_predtest   # shape: (calibration_iterations, 140)
# For each actuator, compute the variance of the residuals.
residual_variances = np.var(residuals, axis=0)
print("Residual variances (first 5 actuators):", residual_variances[:5])

#############################
# Construct the measurement noise covariance matrix as a diagonal matrix.
R_est = zwfs_ns.dm.opd_per_cmd * np.diag(residual_variances) #np.cov(residuals.T) # np.diag(residual_variances) #

print("Estimated R matrix shape:", R_est.shape)

# plt.plot(X_cal[:, act_plot], Y_pred[:, act_plot], '.', label='Model Prediction')
# plt.show()
# # Extract the intercept vector c and slope vector s (assume one coefficient per actuator)
# c = np.array([lr.intercept_ for lr in lin_models])    # shape: (140,)
# s = np.array([lr.coef_[0] for lr in lin_models])        # shape: (140,)
# # (Optionally, you may print or plot these for diagnostics.)

act_plot = 65

#trains set 
plt.figure()
plt.plot(X_cal[:, act_plot], Y_cal[:, act_plot], '.', label='Model Prediction')
plt.plot(X_cal[:, act_plot], Y_pred[:, act_plot], '.', label='Model Prediction')
plt.xlabel('Interpolated intensity at actuator')
plt.ylabel('DM Command')
plt.legend()
plt.title(f'Calibrated Linear Model for actuator {act_plot}')
plt.show()

#Test set 
plt.figure()
plt.plot(X_test[:, act_plot], Y_test[:, act_plot], '.', label='Model Prediction')
plt.plot(X_test[:, act_plot], Y_predtest[:, act_plot], '.', label='Model Prediction')
plt.xlabel('Interpolated intensity at actuator')
plt.ylabel('DM Command')
plt.legend()
plt.title(f'Calibrated Linear Model for actuator {act_plot}')
plt.show()




######### NOW CALIBRATE STATE TRANSITION MODEL BY FITTING AR MODEL


### now need to apply real atmosphere to simulate and estimate state model of phase 
use_pyZelda=True
# first stage AO 
basis_cropped = ztools.zernike.zernike_basis(nterms=150, npix=zwfs_ns.pyZelda.pupil_diameter)
# we have padding around telescope pupil (check zwfs_ns.pyZelda.pupil.shape and zwfs_ns.pyZelda.pupil_diameter) 
# so we need to put basis in the same frame  
basis_template = np.zeros( zwfs_ns.pyZelda.pupil.shape )
basis = np.array( [ util.insert_concentric( np.nan_to_num(b, 0), basis_template) for b in basis_cropped] )
#1=np.sum( basis[3]**2) / np.sum(basis[0])
#pupil_disk = basis[0] # we define a disk pupil without secondary - useful for removing Zernike modes later

amp_input =  photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns.pyZelda.pupil

scrn = ps.PhaseScreenKolmogorov(nx_size=zwfs_ns.grid.dim, 
                                pixel_scale=dx, 
                                r0=zwfs_ns.atmosphere.r0 * ( wvl0 /550e-9)**(6/5) , 
                                L0=zwfs_ns.atmosphere.l0, 
                                random_seed=1)
    
iterations = 500
Nmodes_removed = 14
phi=[]

#flatten DM
zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat

for it in range(iterations):
    print(it)
    # roll screen
    #for _ in range(10):
    scrn.add_row()
    
    # first stage AO
    if np.mod(it, 1) == 0: # only update the AO every few iterations to simulate latency 
        _ , reco_1 = bldr.first_stage_ao( scrn, Nmodes_removed , basis  , phase_scaling_factor = phase_scaling_factor, return_reconstructor = True )   
        
    ao_1 =  basis[0] * (phase_scaling_factor * scrn.scrn - reco_1)
    
    # opd after first stage AO
    opd_ao_1 = zwfs_ns.pyZelda.pupil * zwfs_ns.optics.wvl0 / (2*np.pi) * ao_1
    
    # dm set to zero so measure atm in open loop 
    opd_current_dm = bldr.get_dm_displacement( command_vector = 0*zwfs_ns.dm.current_cmd   , gain=zwfs_ns.dm.opd_per_cmd, \
                sigma = zwfs_ns.grid.dm_coord.act_sigma_wavesp, X=zwfs_ns.grid.wave_coord.X, Y=zwfs_ns.grid.wave_coord.Y,\
                    x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp )
        

    i = bldr.get_frame(  opd_input  = opd_ao_1 + opd_current_dm,   amp_input = amp_input,\
        opd_internal = opd_internal,  zwfs_ns= zwfs_ns , detector= zwfs_ns.detector, use_pyZelda = use_pyZelda )

    i_dm = DM_registration.interpolate_pixel_intensities(
        image=i,
        pixel_coords=zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space
    )

    # phase opd reconstruction from known DM response
    phi.append( zwfs_ns.dm.opd_per_cmd * (s * i_dm + c) )



# what units ?
# units of phi are meters, s = dm units per intensity (140 length), c is dm units
phi= np.array(phi)

### build AR model of atmosphere !! ??

lags=20

ar_params, noise_var = fit_AR_models(np.array(phi), lags=lags)
# Build block–diagonal state-space matrices for all actuators
A_total, Q_total = build_block_state_space(ar_params, noise_var, lags=lags)

# test the state transitiom model 

iterations, n_act = phi.shape
print(f"phi shape: {phi.shape}, expected iterations: {iterations}, n_act: {n_act}")

# Prepare arrays to store predictions and residuals for t from lags to iterations-1.
phi_pred = np.zeros((iterations - lags, n_act))
residuals = np.zeros((iterations - lags, n_act))

for i in range(n_act):
    for t in range(lags, iterations):
        # Predict phi[t, i] using the AR(20) model for actuator i:
        # phi_pred = intercept + a1*phi[t-1] + a2*phi[t-2] + ... + a_lags*phi[t-lags]
        # Note: We use phi[t-lags:t, i] in reverse order so that the most recent (phi[t-1]) multiplies a1, etc.
        phi_pred[t - lags, i] = ar_params[i, 0] + np.dot(ar_params[i, 1:], phi[t-lags:t, i][::-1])
        # Residual: actual minus predicted
        residuals[t - lags, i] = phi[t, i] - phi_pred[t - lags, i]

# For diagnosis, let’s plot for a single actuator (e.g. actuator 0):

actuator_index = 0

plt.figure()
plt.hist(residuals[:, actuator_index], bins=30)
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title(f'Histogram of Residuals for Actuator {actuator_index}')
plt.show()

plt.figure()
plt.plot(residuals[:, actuator_index])
plt.xlabel('Time Index (t - lags)')
plt.ylabel('Residual')
plt.title(f'Time Series of Residuals for Actuator {actuator_index}')
plt.show()

# Also, compute the RMS (root-mean-square) of the residuals for each actuator:
rms_residuals = np.sqrt(np.mean(residuals**2, axis=0))
plt.figure()
plt.plot(rms_residuals, 'o-')
plt.xlabel('Actuator Index')
plt.ylabel('RMS Residual')
plt.title('RMS of AR Model Residuals for Each Actuator')
plt.grid(True)
plt.show()

# Finally, you may compare these RMS values to the noise_var obtained from the AR fitting:
plt.figure()
plt.plot(noise_var, 'o-', label='Estimated Noise Variance')
plt.plot(rms_residuals**2, 'x-', label='RMS Residual^2')
plt.xlabel('Actuator Index')
plt.ylabel('Variance')
plt.title('Comparison: AR Model Noise Variance vs. RMS Residual^2')
plt.legend()
plt.grid(True)
plt.show()










# =============================================================
# Now generate out-of-sample test data (phi_test) using new iterations.
# =============================================================
iterations_test = 500
phi_test = []

#flatten DM
zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat
# You may reset or reinitialize the phase screen if desired.
# Here, we continue from the current state of scrn.
for it in range(iterations_test):
    print(f"Test iteration {it}")
    scrn.add_row()
    
    # First-stage AO: update the reconstructor
    _ , reco_test = bldr.first_stage_ao(scrn, Nmodes_removed, basis,
                                         phase_scaling_factor=phase_scaling_factor,
                                         return_reconstructor=True)
        
    ao_test = basis[0] * (phase_scaling_factor * scrn.scrn - reco_test)
    
    # Compute OPD after first-stage AO
    opd_ao_test = zwfs_ns.pyZelda.pupil * zwfs_ns.optics.wvl0 / (2*np.pi) * ao_test
    
    # DM set to zero so that only atmospheric aberrations are measured
    opd_current_dm_test = bldr.get_dm_displacement(
        command_vector = 0 * zwfs_ns.dm.current_cmd,
        gain = zwfs_ns.dm.opd_per_cmd,
        sigma = zwfs_ns.grid.dm_coord.act_sigma_wavesp,
        X = zwfs_ns.grid.wave_coord.X,
        Y = zwfs_ns.grid.wave_coord.Y,
        x0 = zwfs_ns.grid.dm_coord.act_x0_list_wavesp,
        y0 = zwfs_ns.grid.dm_coord.act_y0_list_wavesp
    )
        
    i_test = bldr.get_frame(
        opd_input  = opd_ao_test + opd_current_dm_test,
        amp_input  = amp_input,
        opd_internal = opd_internal,
        zwfs_ns = zwfs_ns,
        detector = zwfs_ns.detector,
        use_pyZelda = use_pyZelda
    )
    
    i_dm_test = DM_registration.interpolate_pixel_intensities(
        image = i_test,
        pixel_coords = zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space
    )
    
    # Reconstruct phase from the test measurement using the calibrated linear model.
    phi_recon_test = zwfs_ns.dm.opd_per_cmd * (s * i_dm_test + c)
    phi_test.append(phi_recon_test)

phi_test = np.array(phi_test)  # Shape: (iterations_test, 140)

# =============================================================
# Analyze the AR(20) model using the calibration data and then test on phi_test.
# =============================================================

# (Optionally, plot histograms or RMS of residuals for calibration data, as shown earlier.)

# Now, using the fitted AR model, predict the phase for the test data phi_test.
# For each actuator, for t >= lags, predict using the AR model:
N_test, n_act = phi_test.shape
phi_pred_test = np.zeros((N_test - lags, n_act))
residuals_test = np.zeros((N_test - lags, n_act))

for i in range(n_act):
    for t in range(lags, N_test):
        # Prediction: intercept + a1 * phi[t-1] + ... + a_lags * phi[t-lags]
        phi_pred_test[t - lags, i] = ar_params[i, 0] + np.dot(ar_params[i, 1:], phi_test[t-lags:t, i][::-1])
        residuals_test[t - lags, i] = phi_test[t, i] - phi_pred_test[t - lags, i]

# Plot the residuals for one actuator (e.g., actuator 0)
actuator_index = 65


# lets just look at the ts :
plt.figure()
plt.xlabel ('iteration')
plt.ylabel( 'opd (m)')
plt.plot(phi_pred_test[:,actuator_index],label='pred')
plt.plot(phi_test[:,actuator_index], label='meas' )
plt.legend()
plt.show()

plt.figure()
plt.hist(residuals_test[:, actuator_index], bins=30)
plt.xlabel('Residual (radians)')
plt.ylabel('Frequency')
plt.title(f'Histogram of Test Residuals for Actuator {actuator_index}')
plt.show()

plt.figure()
plt.plot(residuals_test[:, actuator_index])
plt.xlabel('Time Index (t - lags)')
plt.ylabel('Residual (radians)')
plt.title(f'Time Series of Test Residuals for Actuator {actuator_index}')
plt.show()

# Compute RMS residuals for the test data across actuators.
rms_residuals_test = np.sqrt(np.mean(residuals_test**2, axis=0))
plt.figure()
plt.plot(rms_residuals_test, 'o-')
plt.xlabel('Actuator Index')
plt.ylabel('RMS Test Residual (radians)')
plt.title('RMS of AR Model Residuals on Test Data')
plt.grid(True)
plt.show()

# Optionally, compare the RMS residuals to the noise variances obtained during calibration.
plt.figure()
plt.plot(noise_var, 'o-', label='Estimated Noise Variance (Calibration)')
plt.plot(rms_residuals_test**2, 'x-', label='RMS Test Residual^2')
plt.xlabel('Actuator Index')
plt.ylabel('Variance (radians^2)')
plt.title('Comparison of Calibration Noise Variance vs. Test Residual Variance')
plt.legend()
plt.grid(True)
plt.show()












# z, H = z_meas, H_total

# print("z shape:", z.shape)                     # Expected: (n_act, 1) i.e., (140, 1)
# print("H shape:", H.shape)                     # Expected: (n_act, n_act*lags) i.e., (140, 2800)
# print("bkf.x shape before update:", bkf.x.shape)  # Expected: (n_act*lags, 1) i.e., (2800, 1)

# y = z - H @ bkf.x
# print("H @ bkf.x shape:", (H @ bkf.x).shape)     # Expected: (140, 1)
# print("y shape:", y.shape)                     # Expected: (140, 1)

# S = H @ bkf.P @ H.T + bkf.R
# print("H @ bkf.P shape:", (H @ bkf.P).shape)     # Expected: (140, 2800)
# print("H.T shape:", H.T.shape)                 # Expected: (2800, 140)
# print("H @ bkf.P @ H.T shape:", (H @ bkf.P @ H.T).shape)  # Expected: (140, 140)
# print("bkf.R shape:", bkf.R.shape)             # Expected: (140, 140)
# print("S shape:", S.shape)                     # Expected: (140, 140)

# K = bkf.P @ H.T @ np.linalg.inv(S)
# print("bkf.P shape:", bkf.P.shape)             # Expected: (2800, 2800)
# print("H.T shape (again):", H.T.shape)         # Expected: (2800, 140)
# print("K shape:", K.shape)                     # Expected: (2800, 140)

# xnew = bkf.x + K @ y
# print("K @ y shape:", (K @ y).shape)           # Expected: (2800, 1)
# print("xnew shape:", xnew.shape)               # Expected: (2800, 1)

# Pnew = (np.eye(len(bkf.x)) - K @ H) @ bkf.P
# print("np.eye(len(bkf.x)) shape:", np.eye(len(bkf.x)).shape)  # Expected: (2800, 2800)
# print("K @ H shape:", (K @ H).shape)           # Expected: (2800, 2800)
# print("Pnew shape:", Pnew.shape)               # Expected: (2800, 2800)


# -------------------------------------------------------------------
# CLOSED-LOOP SIMULATION: Close the loop on the input phase screen.
# -------------------------------------------------------------------

n_state = A_total.shape[0]  # = no_actuators * lags

# Build measurement matrix H: each actuator measurement corresponds to the first element of its state block.
# 0----> z_est = H @ x + noise , where x is  sstate, z is measurement
#  y = z_meas - z_est ## inovation !!
# in our case x is opd, z is opd , so H just takes (filters) most recent x_est 
H_total = build_measurement_matrix(n_act=phi.shape[1], lags=lags)


# Initialize overall state: assume zero initial state for each actuator (20 lags per actuator)
x0_total = np.zeros((n_state, 1))
P0_total = 1e-7 * np.eye(n_state)

# Create block Kalman filter instance
bkf = BlockKalmanFilter(A=A_total, Q=Q_total, R=R_est, x0=x0_total, P0=P0_total)


simulation_iterations = 30
close_after = 5
x_est_history = []
dm_cmd_history = []
strehl_history = []
strehl_before_history = []
P_history=[]
K_history=[]
# n_act: number of DM actuators (should be 140)
n_act = phi.shape[1]  # from your previous calibration, phi is (iterations, 140)

# Initialize "true" atmospheric state for each actuator in its AR(20) representation.
# For each actuator, the state is a vector of length 'lags' (here 20).
# We assume the initial state is zero.
x_true_total = np.zeros((n_act * lags, 1))


#flatten DM
zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat
# opd of flat map
opd_flat_dm = bldr.get_dm_displacement( command_vector = zwfs_ns.dm.dm_flat   , gain=zwfs_ns.dm.opd_per_cmd, \
            sigma = zwfs_ns.grid.dm_coord.act_sigma_wavesp, X=zwfs_ns.grid.wave_coord.X, Y=zwfs_ns.grid.wave_coord.Y,\
                x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp )
    
for it in range(simulation_iterations):
    
    print(it)

    # roll screen
    #for _ in range(10):
    scrn.add_row()
    
    # first stage AO
    if np.mod(it, 1) == 0: # only update the AO every few iterations to simulate latency 
        _ , reco_1 = bldr.first_stage_ao( scrn, Nmodes_removed , basis  , phase_scaling_factor = phase_scaling_factor, return_reconstructor = True )   
        
    ao_1 =  basis[0] * (phase_scaling_factor * scrn.scrn - reco_1)
    
    # opd after first stage AO
    opd_ao_1 = zwfs_ns.pyZelda.pupil * zwfs_ns.optics.wvl0 / (2*np.pi) * ao_1
    
    strehl_before = np.exp(-np.var( (opd_flat_dm+ao_1)[zwfs_ns.pyZelda.pupil>0]))
    strehl_before_history.append(strehl_before)

    # dm set to zero so measure atm in open loop 
    opd_current_dm = bldr.get_dm_displacement( command_vector = zwfs_ns.dm.current_cmd   , gain=zwfs_ns.dm.opd_per_cmd, \
                sigma = zwfs_ns.grid.dm_coord.act_sigma_wavesp, X=zwfs_ns.grid.wave_coord.X, Y=zwfs_ns.grid.wave_coord.Y,\
                    x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp, y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp )
        
    phi_post_dm =  zwfs_ns.pyZelda.pupil * 2 * np.pi / zwfs_ns.optics.wvl0 * (opd_current_dm + opd_ao_1) 
    strehl =  np.exp(-np.var( phi_post_dm[zwfs_ns.pyZelda.pupil>0]))
    strehl_history.append(strehl)


    # # Propagate the true atmospheric state using the AR(20) model:
    # # Sample process noise (w_k) for all state elements using Q_total.
    # w_k = np.random.multivariate_normal(np.zeros(n_act * lags), Q_total).reshape((n_act * lags, 1))
    # x_true_total = A_total @ x_true_total + w_k  # True state update
    
    # # Extract the "true" atmospheric phase at each actuator:
    # # For each actuator, the first element in its AR block represents the current phase.
    # phi_true = x_true_total.reshape(n_act, lags)[:, 0].reshape((n_act, 1))
    

    # Simulate sensor measurement:
    # Measurement noise is added based on R_est (the diagonal measurement noise covariance).
    # (1) Simulate raw ZWFS intensity measurement 'i'
    i = bldr.get_frame(
        opd_input=opd_ao_1 + opd_current_dm,
        amp_input=amp_input,
        opd_internal=opd_internal,
        zwfs_ns=zwfs_ns,
        detector=zwfs_ns.detector,
        use_pyZelda=use_pyZelda
    )

    # (2) Interpolate intensity onto DM actuator positions:
    i_dm = DM_registration.interpolate_pixel_intensities(
        image=i,
        pixel_coords=zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space
    )

    # (3) Reconstruct the total phase using the calibrated linear model:
    # Here, s (slope vector dm command unit per adu) and c (intercept vector) were determined during calibration.
    # units are opd (m)
    phi_total = zwfs_ns.dm.opd_per_cmd * (s * i_dm + c) # this is the DM cmd relative to flat?

    # (4) Subtract the known DM contribution:
    # The DM-induced phase is: phi_dm = f * cmd (with f = zwfs_ns.dm.opd_per_cmd in our simple case)
    # So, the effective measurement for the atmosphere is:
    # units = OPD
    z_meas = phi_total - zwfs_ns.dm.opd_per_cmd * (zwfs_ns.dm.current_cmd - zwfs_ns.dm.dm_flat)

    z_meas = z_meas.reshape((n_act, 1))
    #x_est_total = bkf.update(z_meas, H_total)
    
    # Run the Kalman filter update:
    bkf.predict() # predict hte next state using state transition

    # to understand how H works 
    # aaa=x_true_total.reshape( n_act , lags )
    #aaa[:,0]=1
    #H_total @ aaa.reshape(n_act*lags)
    #filters only for the most recent lag in x state
    x_est_total = bkf.update(z_meas, H_total) # update this with the measurement 
    
    
    x_est_history.append(x_est_total.copy())
    P_history.append(bkf.P)
    K_history.append(bkf.K)

    # Extract the estimated atmospheric phase for each actuator (first element of each state block):
    phi_est = x_est_total.reshape(n_act, lags)[:, 0].reshape((n_act, 1))
    
    # Compute the DM command to cancel the estimated atmospheric phase.
    # The DM-induced phase is given by: φ_dm = opd_per_cmd * dm_cmd.
    # To cancel φ_atmosphere, set: dm_cmd = - (φ_est / opd_per_cmd)
    dm_cmd = - (1.0 / zwfs_ns.dm.opd_per_cmd) * phi_est

    
    # Update the DM command in the system namespace (add to the DM flat):
    if it > close_after:
        zwfs_ns.dm.current_cmd = zwfs_ns.dm.dm_flat + dm_cmd.flatten()

    dm_cmd_history.append(dm_cmd.flatten())    
    

    # Compute the residual phase after DM correction:
    # Ideally, if DM perfectly cancels the disturbance, the residual should be zero.
    # Here: φ_residual = φ_atmosphere + φ_dm.
    # phi_residual = phi_true + zwfs_ns.dm.opd_per_cmd * dm_cmd
    
    #print(f"Iteration {it:03d}: Strehl = {strehl:.3f}")

# Plot the evolution of the Strehl ratio over iterations.
plt.figure(figsize=(8, 4))
plt.plot(strehl_before_history, 'bo-',label='before')
plt.plot(strehl_history, 'ro-',label='after')
plt.axvline(close_after, label='close loop')
plt.xlabel('Iteration')
plt.ylabel('Strehl Ratio')
plt.legend()
plt.title('Closed-Loop AO Simulation (AR(20) Kalman Filter)')
plt.grid(True)
plt.show()

# look what the DM does 

util.create_telem_mosaic(  )

util.display_images_with_slider(image_lists = [[util.get_DM_command_in_2D( np.array(d).ravel() ) for d in dm_cmd_history]])
       


# (Optionally, save the telemetry data for further analysis.)
telemetry_data = {
    'x_est_history': x_est_history,
    'dm_cmd_history': dm_cmd_history,
    'strehl_history': strehl_history,
    'AR_params': ar_params,
    'AR_noise_variance': noise_var
}

save_file = os.path.join(os.getcwd(), f"kalman_AR20_telemetry_{datetime.datetime.now().strftime('%d-%m-%YT%H.%M.%S')}.pkl")
with open(save_file, 'wb') as f:
    pickle.dump(telemetry_data, f)
print(f"Telemetry saved to {save_file}")












########## other information

#the DM command has the following linear relation
zwfs_ns.dm.opd_per_cmd
# .e.g opd = zwfs_ns.dm.opd_per_cmd * dm_cmd

# also the DM flat has rmse noisee 
zwfs_ns.dm.flat_rmse

# example to get natrix to do interpolation
zwfs_ns.dm.current_cmd=np.zeros(140)
imgtmp = bldr.get_I0( opd_input = 0 * zwfs_ns.pyZelda.pupil ,  amp_input = photon_flux_per_pixel_at_vlti**0.5 * zwfs_ns.pyZelda.pupil, opd_internal=zwfs_ns.pyZelda.pupil * (opd_internal + opd_flat_dm), \
    zwfs_ns=zwfs_ns, detector=zwfs_ns.detector, include_shotnoise=True , use_pyZelda = True)

x_target = np.array([ x for x,_ in zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space])
y_target = np.array([ y for _,y in zwfs_ns.dm2pix_registration.actuator_coord_list_pixel_space])
x_grid = np.arange( imgtmp.shape[0])
y_grid = np.arange( imgtmp.shape[1])
M = DM_registration.construct_bilinear_interpolation_matrix( imgtmp.shape, x_grid, y_grid, x_target, y_target)
# now use it to get i_dm from image
i_dm = M @ imgtmp.flatten()
