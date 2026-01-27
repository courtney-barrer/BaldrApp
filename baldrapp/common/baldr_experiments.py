
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from baldrapp.common import baldr_core as bldr


"""
Goal is to have a standard way to run experiments onsky. We develop a eval_onsky function
which has the following logic 

eval_onsky(...)
│
├─[0] Controller init
│   ├─ if ctrl_slow != None -> ctrl_slow.reset()
│   └─ if ctrl_fast != None -> ctrl_fast.reset()
│
├─[1] Defaults / validation
│   ├─ if opd_internal is None -> opd_internal = zeros_like(pupil_mask)
│   ├─ if loop_schedule is None -> loop_schedule = [(0,"open")]
│   ├─ sort loop_schedule by it0
│   └─ validate each schedule mode ∈ {"open","slow","fast","fast+slow"}
│
├─[2] Telemetry dict init
│   ├─ telem = {...}
│   ├─ telem["I0_cal"].append(zwfs_ns_calibration.reco.I0)
│   └─ telem["N0_cal"].append(zwfs_ns_calibration.reco.N0)
│
├─[3] Reference intensities decision (N0 handling)
│   ├─ user_ref_intensities is None ?
│   │    ├─ YES:
│   │    │   ├─ call update_N0(...) using zwfs_ns_current + (AO1/scint options)
│   │    │   ├─ N0_onsky = mean(N0_list)
│   │    │   └─ zwfs_ns_current.reco.N0 = N0_onsky
│   │    └─ NO:
│   │        ├─ unpack (user_I0, user_N0)
│   │        ├─ shape checks vs zwfs_ns_current.reco.I0 and reco.N0
│   │        └─ store to telem["I0_used"], telem["N0_used"]
│   │
│   └─ if user_ref_intensities is None:
│       └─ telem["I0_used"].append(zwfs_ns_current.reco.I0)
│          telem["N0_used"].append(zwfs_ns_current.reco.N0)
│
├─[4] AO1 latency buffer pre-fill
│   └─ reco_list = []
│      repeat it_lag times:
│        phasescreen.add_row()
│        (_, reco_1) = first_stage_ao(...)
│        reco_list.append(reco_1)
│
├─[5] DM init
│   └─ zwfs_ns_current.dm.current_cmd = dm_flat
│
├─[6] Helper: loop mode lookup
│   └─ _mode_at(it):
│        start with loop_schedule[0].mode
│        walk schedule entries in order:
│          if it >= it0 -> update mode
│          else break
│        return mode
│
└─[7] Main loop: for it in range(N_iter)
    │
    ├─[7.1] Progress print (optional)
    │   └─ if verbose_every>0 and it%verbose_every==0 -> print status
    │
    ├─[7.2] Determine loop_mode
    │   └─ loop_mode = _mode_at(it)
    │
    ├─[7.3] Build input OPD + amplitude (turbulence vs static)
    │   ├─ static_input_field is None ?
    │   │    ├─ YES (Kolmogorov + AO1 residual with lag):
    │   │    │   ├─ phasescreen.add_row()
    │   │    │   ├─ (_, reco_1)=first_stage_ao(...)
    │   │    │   ├─ reco_list.append(reco_1)
    │   │    │   ├─ ao_1 = basis[0]*(phase_scaling*scrn - reco_list.pop(0))
    │   │    │   └─ opd_input = phase_scaling*wvl/(2π)*ao_1
    │   │    └─ NO (static field):
    │   │        ├─ check shape(static_input_field)==shape(pupil_mask) else raise
    │   │        └─ opd_input = static_input_field
    │   │
    │   └─ amplitude/scintillation:
    │       ├─ include_scintillation AND scintillation_screen != None ?
    │       │    ├─ YES:
    │       │    │   ├─ advance scint screen jumps_per_iter times
    │       │    │   ├─ amp_scint = update_scintillation_fn(...)
    │       │    │   └─ amp_input = amp_input_0 * amp_scint
    │       │    └─ NO:
    │       │        └─ amp_input = amp_input_0
    │
    ├─[7.4] Apply current DM command to get DM OPD contribution
    │   ├─ opd_dm = get_dm_displacement(command=dm.current_cmd, ...)
    │   └─ opd_total = opd_input + opd_dm
    │      (opd_internal passed separately into get_N0/get_frame)
    │
    ├─[7.5] Measure intensities
    │   ├─ n00 = get_N0(opd_total, amp_input, opd_internal, ...)
    │   └─ i   = get_frame(opd_total, amp_input, opd_internal, ...)
    │
    ├─[7.6] Build signal s (normalization branch)
    │   ├─ user_ref_intensities is None ?
    │   │    ├─ YES: use zwfs_ns_calibration.reco.I0 and zwfs_ns_current.reco.N0
    │   │    │   ├─ normalization_method == "subframe mean" ?
    │   │    │   │    └─ s = i/mean(i) - I0/mean(I0)
    │   │    │   ├─ normalization_method == "clear pupil mean" ?
    │   │    │   │    └─ s = i/mean(N0_current[interior]) - I0/mean(N0_cal[interior])
    │   │    │   └─ else -> raise
    │   │    └─ NO: use user_I0, user_N0
    │   │         ├─ "subframe mean" -> s = i/mean(i) - user_I0/mean(user_I0)
    │   │         ├─ "clear pupil mean" -> s = i/mean(N0_current[interior]) - user_I0/mean(user_N0[interior])
    │   │         └─ else -> raise
    │   └─ s flattened to (P,) where P = num pixels
    │
    ├─[7.7] Reconstruct errors (signal space branch)
    │   ├─ signal_space == "dm" ?
    │   │    ├─ YES:
    │   │    │   ├─ s_dm = DM_interpolate_fn(image=s.reshape(I0.shape), pixel_coords=calib.actuator_coords)
    │   │    │   ├─ e_LO = I2M_TT @ s_dm
    │   │    │   └─ e_HO = I2M_HO @ s_dm
    │   │    └─ NO (signal_space == "pix"):
    │   │         ├─ e_LO = I2M_TT @ s
    │   │         └─ e_HO = I2M_HO @ s
    │   └─ else -> raise
    │
    ├─[7.8] Controller gating by schedule
    │   ├─ do_fast = loop_mode in {"fast","fast+slow"}
    │   ├─ do_slow = loop_mode in {"slow","fast+slow"}
    │   │
    │   ├─ initialize u_*:
    │   │    ├─ if ctrl_fast: u_LO_fast=ctrl_fast.u_LO, u_HO_fast=ctrl_fast.u_HO else zeros
    │   │    └─ if ctrl_slow: u_LO_slow=ctrl_slow.u_LO, u_HO_slow=ctrl_slow.u_HO else zeros
    │   │
    │   ├─ if do_fast:
    │   │    ├─ if ctrl_fast is None -> raise
    │   │    └─ (u_LO_fast,u_HO_fast)=ctrl_fast.process(e_LO,e_HO)
    │   └─ if do_slow:
    │        ├─ if ctrl_slow is None -> raise
    │        └─ (u_LO_slow,u_HO_slow)=ctrl_slow.process(e_LO,e_HO)
    │
    ├─[7.9] Map controller outputs to DM increments (calibration M2C)
    │   ├─ c_LO_fast = M2C_LO @ u_LO_fast
    │   ├─ c_HO_fast = M2C_HO @ u_HO_fast
    │   ├─ c_LO_slow = M2C_LO @ u_LO_slow
    │   ├─ c_HO_slow = M2C_HO @ u_HO_slow
    │   └─ d_cmd = sum of all c_*
    │
    ├─[7.10] Apply command (your sign convention)
    │   ├─ cmd = dm_flat - d_cmd
    │   └─ dm.current_cmd = cmd
    │
    ├─[7.11] Safety + diagnostic OPD
    │   ├─ opd_dm_after = get_dm_displacement(dm.current_cmd, ...)
    │   ├─ opd_input_w_NCPA = opd_input + opd_internal
    │   ├─ opd_res_wo_NCPA  = opd_input + opd_dm_after
    │   ├─ opd_res_w_NCPA   = opd_input + opd_dm_after + opd_internal
    │   │
    │   ├─ sigma_cmd = std(sum of c_*) * opd_per_cmd
    │   └─ sigma_cmd > opd_threshold ?
    │        ├─ YES:
    │        │   ├─ dm.current_cmd = dm_flat
    │        │   ├─ if reset_resets_fast and ctrl_fast: ctrl_fast.reset()
    │        │   ├─ if reset_resets_slow and ctrl_slow: ctrl_slow.reset()
    │        │   ├─ telem["reset_events"].append(it)
    │        │   └─ recompute opd_dm_after and residual maps after reset
    │        └─ NO: continue
    │
    └─[7.12] Telemetry write (burn-in gate)
        ├─ it >= N_burn ?
        │    ├─ YES: append fields: it, loop_mode, n00, i, s, e_LO/e_HO,
        │    │       u_*, c_*, d_cmd, dm_cmd, OPD maps, RMSE metrics
        │    └─ NO: store nothing
        └─ end iter
│
└─[8] Exit
    ├─ dm.current_cmd = dm_flat
    └─ return telem

"""


# example leaky integrator class with process method compatiple with eval_onsky
class LeakyIntegratorController:
    """
    Modal leaky integrator:

        u_LO <- leak_LO * u_LO + ki_LO * e_LO
        u_HO <- leak_HO * u_HO + ki_HO * e_HO

    Gains may be:
      - scalars (broadcast to all modes)
      - vectors (must match modal dimension)
    """

    def __init__(self, n_lo, n_ho, ki_LO=0.0, ki_HO=0.0, leak=1.0):
        self.n_lo = int(n_lo)
        self.n_ho = int(n_ho)

        # --- convert gains to vectors ---
        self.ki_LO = self._as_vector(ki_LO, self.n_lo, name="ki_LO")
        self.ki_HO = self._as_vector(ki_HO, self.n_ho, name="ki_HO")

        # leak can be scalar or vector (apply per mode)
        self.leak_LO = self._as_vector(leak, self.n_lo, name="leak_LO")
        self.leak_HO = self._as_vector(leak, self.n_ho, name="leak_HO")

        # integrator states
        self.u_LO = np.zeros(self.n_lo, dtype=float)
        self.u_HO = np.zeros(self.n_ho, dtype=float)

    @staticmethod
    def _as_vector(val, n, name="param"):
        """
        Convert scalar or vector input to a length-n numpy array.
        """
        if np.isscalar(val):
            return np.full(n, float(val), dtype=float)

        arr = np.asarray(val, dtype=float)
        if arr.ndim != 1 or arr.size != n:
            raise ValueError(
                f"{name} must be scalar or length {n}, got shape {arr.shape}"
            )
        return arr.copy()

    def reset(self):
        self.u_LO[:] = 0.0
        self.u_HO[:] = 0.0

    def process(self, e_LO, e_HO):
        e_LO = np.asarray(e_LO, dtype=float)
        e_HO = np.asarray(e_HO, dtype=float)

        if e_LO.size != self.n_lo:
            raise ValueError(f"e_LO size mismatch: got {e_LO.size}, expected {self.n_lo}")
        if e_HO.size != self.n_ho:
            raise ValueError(f"e_HO size mismatch: got {e_HO.size}, expected {self.n_ho}")

        # element-wise leaky integration
        self.u_LO = self.leak_LO * self.u_LO + self.ki_LO * e_LO
        self.u_HO = self.leak_HO * self.u_HO + self.ki_HO * e_HO

        return self.u_LO.copy(), self.u_HO.copy()


def update_N0(  zwfs_ns,  
                phasescreen, 
                scintillation_screen, 
                update_scintillation_fn,   # pass your update_scintillation
                basis,
                *,
                detector,
                dx,
                amp_input_0,
                propagation_distance,
                static_input_field=None,
                opd_internal=None, 
                N_iter_estimation =100, 
                it_lag=0, 
                Nmodes_removed=0, 
                phase_scaling_factor=1.0, 
                include_scintillation=True, 
                jumps_per_iter=1,  
                verbose_every = 100 
                ):
    
    """
    Estimate the clear-pupil reference intensity N0 under *operating* conditions.

    This routine measures the ZWFS clear-pupil intensity N0 by propagating the
    current optical state through the system over multiple iterations and
    averaging the result. The estimate intentionally reflects *on-sky* operating
    conditions rather than an idealized laboratory reference.

    Key design features
    -------------------
    - Uses the *current deformable-mirror (DM) command*:
        The DM shape stored in `zwfs_ns.dm.current_cmd` is applied during the
        measurement. This allows N0 to include the effects of first-stage AO
        correction, DM print-through, and residual static aberrations.
        The DM is **not flattened inside this function**.

    - Includes first-stage AO residuals:
        If `static_input_field` is None, the input phase is generated from a
        Kolmogorov phase screen with an optional latency buffer (`it_lag`) and
        removal of low-order modes (`Nmodes_removed`), matching the operational
        AO1 configuration.

    - Optional scintillation:
        If `include_scintillation=True`, a high-altitude phase screen is evolved
        and converted to amplitude fluctuations using `update_scintillation_fn`,
        allowing N0 to capture scintillation-induced pupil intensity variations.

    - Realistic normalization reference:
        The resulting N0 represents the *mean clear-pupil intensity seen by the
        ZWFS while the system is running*, not an ideal diffraction-limited pupil.
        This is critical for avoiding normalization biases in on-sky signal
        estimation.

    Parameters
    ----------
    zwfs_ns : object
        ZWFS namespace describing the current optical system, including detector,
        DM, pupil geometry, and wavelength. The current DM command is read from
        `zwfs_ns.dm.current_cmd`.

    phasescreen : PhaseScreen
        Turbulent phase screen used to generate input OPD when
        `static_input_field` is None.

    scintillation_screen : PhaseScreen or None
        High-altitude phase screen used to model scintillation. Ignored if
        `include_scintillation=False`.

    update_scintillation_fn : callable
        Function converting a high-altitude phase screen into an amplitude
        modulation map.

    basis : array_like
        Modal basis used by the first-stage AO model. `basis[0]` is assumed to be
        the piston / pupil mode used for filtering AO1 residuals.

    dx : float
        Pixel scale in meters per pixel.

    amp_input_0 : ndarray
        Nominal (unscintillated) pupil amplitude.

    propagation_distance : float
        Propagation distance used in scintillation modelling.

    static_input_field : ndarray or None, optional
        If provided, bypasses the turbulence model and injects a fixed OPD map.
        Must match the pupil grid shape.

    opd_internal : ndarray or None, optional
        Static internal OPD (e.g. NCPA) added to the optical path.

    N_iter_estimation : int
        Number of iterations used to estimate N0. The returned N0 should be
        averaged externally for noise reduction.

    it_lag : int
        Latency (in iterations) applied to the first-stage AO reconstructor.

    Nmodes_removed : int
        Number of low-order modes removed by the first-stage AO model.

    phase_scaling_factor : float
        Scalar applied to the turbulent phase before AO correction.

    include_scintillation : bool
        Whether to include scintillation effects.

    jumps_per_iter : int
        Number of rows the scintillation phase screen advances per iteration.

    verbose_every : int
        Print progress every `verbose_every` iterations.

    Returns
    -------
    N0_list : list of ndarray
        List of clear-pupil intensity frames. The caller is expected to average
        these (e.g. `np.mean(N0_list, axis=0)`) before use.

    Notes
    -----
    • This function does *not* reset or modify the DM state.
    • The DM should be flattened *once* at the start of an experiment, not here.
    • Measuring N0 under operating conditions is intentional and required for
      unbiased on-sky normalization.
    """
    # populate rolling buffer of first stage AO to account for latency
    reco_list = []
    for _ in range(int(it_lag)):
        phasescreen.add_row()
        _, reco_1 = bldr.first_stage_ao(
            phasescreen,
            Nmodes_removed=Nmodes_removed,
            basis=basis,
            phase_scaling_factor=phase_scaling_factor,
            return_reconstructor=True,
        )
        reco_list.append(reco_1)



    # --- main loop ---
    N0_list = []
    for it in range(int(N_iter_estimation)):

        if it % int(verbose_every) == 0:
            print(f"clear pup estimation iteration : {it}/{N_iter_estimation}  ({100.0*it/max(1,N_iter_estimation):.1f}%)")

        if static_input_field is None: # then we go ahead with Kolmogorov 
            # --- evolve turbulence + AO1 residual (w/ lag) ---
            phasescreen.add_row()

            _, reco_1 = bldr.first_stage_ao(
                phasescreen,
                Nmodes_removed=Nmodes_removed,
                basis=basis,
                phase_scaling_factor=phase_scaling_factor,
                return_reconstructor=True,
            )
            reco_list.append(reco_1)

            ao_1 = basis[0] * (phase_scaling_factor * phasescreen.scrn - reco_list.pop(0))
            opd_input = phase_scaling_factor * zwfs_ns.optics.wvl0 / (2 * np.pi) * ao_1  # [m]

            # --- evolve scintillation + amplitude ---
            if include_scintillation and (scintillation_screen is not None):
                for _ in range(int(jumps_per_iter)):
                    scintillation_screen.add_row()

                amp_scint = update_scintillation_fn(
                    high_alt_phasescreen=scintillation_screen,
                    pxl_scale=dx,
                    wavelength=zwfs_ns.optics.wvl0,
                    final_size=None,
                    jumps=0,
                    propagation_distance=propagation_distance,
                )
                amp_input = amp_input_0 * amp_scint
            else:
                amp_input = amp_input_0

        
        elif static_input_field is not None and (np.shape(static_input_field) == np.shape(zwfs_ns.grid.pupil_mask)):
            opd_input = static_input_field # user defined static input field 

            # --- evolve scintillation + amplitude ---
            if include_scintillation and (scintillation_screen is not None):
                for _ in range(int(jumps_per_iter)):
                    scintillation_screen.add_row()

                amp_scint = update_scintillation_fn(
                    high_alt_phasescreen=scintillation_screen,
                    pxl_scale=dx,
                    wavelength=zwfs_ns.optics.wvl0,
                    final_size=None,
                    jumps=0,
                    propagation_distance=propagation_distance,
                )
                amp_input = amp_input_0 * amp_scint
            else:
                amp_input = amp_input_0
            
        else:
            raise UserWarning(f"input static_input_field shape seems wrong\nstatic_input_field.shape={static_input_field.shape}\n amp_input_0.shape={amp_input_0.shape}")


        # --- apply current DM command to compute DM OPD contribution ---
        opd_dm = bldr.get_dm_displacement(
            command_vector=zwfs_ns.dm.current_cmd,
            gain=zwfs_ns.dm.opd_per_cmd,
            sigma=zwfs_ns.grid.dm_coord.act_sigma_wavesp,
            X=zwfs_ns.grid.wave_coord.X,
            Y=zwfs_ns.grid.wave_coord.Y,
            x0=zwfs_ns.grid.dm_coord.act_x0_list_wavesp,
            y0=zwfs_ns.grid.dm_coord.act_y0_list_wavesp,
        )

        opd_total = opd_input + opd_dm  # opd_internal handled separately in get_frame/get_N0 below


        n00 = bldr.get_N0(
            opd_total,
            amp_input,
            opd_internal,
            zwfs_ns,
            detector=detector,
            use_pyZelda=False,
        ).astype(float)

        N0_list.append( n00 )

    return N0_list 







# we focus on a consistent loop function with different calibration and current zwfs_ns objects 
def eval_onsky(
    zwfs_ns_current,
    zwfs_ns_calibration,
    phasescreen,
    scintillation_screen,
    basis, # basis[0] should be pupil (piston), normalized 0-1, used to filter first stage AO pupil
    *,
    detector,
    amp_input_0,
    dx,
    propagation_distance,
    update_scintillation_fn,   # pass your update_scintillation
    DM_interpolate_fn,         # pass DM_registration.interpolate_pixel_intensities
    static_input_field=None, # ignore kolmogorov turbuelnce and only inject this static field with no AO
    user_ref_intensities=None, # pair of zwfs and clear pupil reference intensities (I0, N0) used in signal calculation s ~ I-I0
    N_iter=1000, # number of iterations
    N_burn=0, # number iterations to burn before recording telemetry
    it_lag=0, # lag for first stage AO to simulate temporal errors, if 0 then perfect first stage removal of Nmodes_removed
    Nmodes_removed=0, # number of modes removed in first stage AO
    phase_scaling_factor=1.0, # scalar phase scaling factor to apply to first stage AO residuals 
    include_scintillation=True, # do we apply scintillation
    jumps_per_iter=1, # how many rows scintillation phase screen jumps per iteration 
    signal_space="pix",        # "pix" | "dm", what space are signals calculated in
    opd_internal=None,         # internal OPD from internal aberrations, lab turbluence etc
    opd_threshold=np.inf,      # threshold on std(c_LO+c_HO)*opd_per_cmd (your convention)
    loop_schedule=None,        # list of tuples [(0,"open"), (100,"slow"), (200,"fast")]
    ctrl_fast=None,            # controller instance with .process(e_LO,e_HO) and .reset() for fast control loop
    ctrl_slow=None,            # controller instance with .process(e_LO,e_HO) and .reset() for slow control loop
    reset_resets_fast=True,    # reset fast controller on safety reset
    reset_resets_slow=True,   # whether to reset slow controller on safety reset
    verbose_every=100, # print something every < verbose_every> iterations
    cal_tag = None, # used to map and copy the correct calibration zwfs_ns object (copy new one to keep experiement clean)
):
    """
    Run a physically faithful on-sky ZWFS control-loop experiment.

    Motivation
    ----------
    This function implements a *realistic on-sky experiment loop* for a Zernike
    Wavefront Sensor (ZWFS), explicitly separating **calibration** and **measurement**
    optical models. The goal is to study control behaviour, biases, and performance
    under operating conditions that closely resemble on-sky operation, including:

    - Use of calibration-derived reconstructors (I0, N0, I2M, M2C, DM registration)
    - A potentially different on-sky pupil and optical train
    - First-stage AO residuals with optional latency
    - Optional scintillation-induced amplitude fluctuations
    - Simultaneous fast and slow control loops
    - On-sky re-estimation of the clear-pupil reference intensity N0

    The function is designed to be:
    - Deterministic and reproducible
    - Explicit about signal normalization conventions
    - Faithful to the way ZWFS loops are operated on-sky
    - Suitable for detailed telemetry analysis and performance diagnostics

    High-level algorithm
    --------------------
    At each iteration, the loop performs the following steps:

    1. Evolve the turbulent phase screen (unless a static input field is supplied).
    2. Apply a first-stage AO model to remove low-order modes with optional latency.
    3. Optionally evolve a scintillation phase screen and generate amplitude
    fluctuations.
    4. Apply the *current* DM command to form the total OPD.
    5. Generate:
    - The ZWFS intensity image
    - The (simulated) clear-pupil intensity image
    6. Compute the ZWFS signal using calibration-defined normalization conventions.
    7. Reconstruct low- and high-order modal errors using calibration operators.
    8. Conditionally run fast and/or slow controllers according to the loop schedule.
    9. Map modal commands to DM space and update the DM.
    10. Apply safety checks and reset logic if command thresholds are exceeded.
    11. Record telemetry after an optional burn-in period.

    Clear-pupil reference handling
    ------------------------------
    If `user_ref_intensities` is not provided, the function *re-estimates N0 on-sky*
    using the current DM shape and optical state before entering the main loop.
    This ensures that signal normalization reflects operating conditions rather than
    an idealized laboratory reference.

    Parameters
    ----------
    zwfs_ns_current : object
        ZWFS namespace representing the *on-sky / measurement* system. This object
        defines the pupil geometry, DM model, optics, and wavelength used to generate
        measurements. Its `reco.N0` field may be updated internally.

    zwfs_ns_calibration : object
        ZWFS namespace representing the *calibration* system. All reconstruction
        operators (I0, N0, I2M, M2C, DM registration) and normalization conventions
        are taken from this object.

    phasescreen : PhaseScreen
        Turbulent phase screen used to generate input OPD when
        `static_input_field` is None.

    scintillation_screen : PhaseScreen or None
        High-altitude phase screen used to model scintillation. Ignored if
        `include_scintillation=False`.

    basis : array_like
        Modal basis used by the first-stage AO model. `basis[0]` is assumed to
        represent the pupil (piston-like) mode and is used to filter AO1 residuals.

    detector : object
        Detector model passed to the ZWFS propagation routines.

    amp_input_0 : ndarray
        Nominal (unscintillated) pupil amplitude.

    dx : float
        Pixel scale in meters per pixel.

    propagation_distance : float
        Propagation distance used in scintillation modelling.

    update_scintillation_fn : callable
        Function converting a high-altitude phase screen into an amplitude modulation
        map.

    DM_interpolate_fn : callable
        Function mapping pixel-space signals to DM actuator sampling
        (e.g. `DM_registration.interpolate_pixel_intensities`).

    static_input_field : ndarray or None, optional
        If provided, bypasses the turbulence model and injects a fixed OPD map.
        Must match the pupil grid shape.

    user_ref_intensities : tuple (I0, N0) or None, optional
        User-supplied ZWFS and clear-pupil reference intensities. If None, N0 is
        estimated on-sky and I0 is taken from calibration.

    N_iter : int
        Total number of loop iterations.

    N_burn : int
        Number of initial iterations to discard before recording telemetry.

    it_lag : int
        Latency (in iterations) applied to the first-stage AO reconstructor.

    Nmodes_removed : int
        Number of low-order modes removed by the first-stage AO model.

    phase_scaling_factor : float
        Scalar applied to the turbulent phase before AO correction.

    include_scintillation : bool
        Whether to include scintillation effects.

    jumps_per_iter : int
        Number of rows the scintillation phase screen advances per iteration.

    signal_space : {"pix", "dm"}
        Space in which the ZWFS signal is reconstructed:
        - "pix": pixel-space reconstruction
        - "dm": DM-sampled signal reconstruction

    opd_internal : ndarray or None
        Static internal OPD (e.g. NCPA) added to the optical path.

    opd_threshold : float
        Safety threshold on the standard deviation of commanded OPD. Exceeding this
        triggers a DM and controller reset.

    loop_schedule : list of (iteration, mode) tuples
        Schedule defining which control loops are active as a function of iteration.
        Valid modes are {"open", "slow", "fast", "fast+slow"}.

    ctrl_fast : object or None
        Fast controller instance with methods `.process(e_LO, e_HO)` and `.reset()`.

    ctrl_slow : object or None
        Slow controller instance with methods `.process(e_LO, e_HO)` and `.reset()`.

    reset_resets_fast : bool
        Whether to reset the fast controller on a safety reset.

    reset_resets_slow : bool
        Whether to reset the slow controller on a safety reset.

    verbose_every : int or None
        Print progress every `verbose_every` iterations. Set to None to disable.

    cal_tag : str or None
        Optional tag identifying the calibration context. Included for bookkeeping
        and experiment hygiene; not used internally by the loop logic.

    Returns
    -------
    telem : dict
        Dictionary containing time-series telemetry of:
        - Intensities, signals, modal errors
        - Controller states and DM commands
        - OPD maps and RMS metrics
        - Loop mode and reset events

    Notes
    -----
    - The DM is flattened once at loop entry and again on exit.
    - Controllers are gated by the schedule but retain internal state.
    - Fast and slow controllers may run simultaneously.
    - No calibration operators are modified inside this function.
    - All signal normalization strictly follows calibration conventions.
    """

    if ctrl_slow is not None :
        ctrl_slow.reset()
    if ctrl_fast is not None :
        ctrl_fast.reset()

    # --- defaults / checks ---
    if opd_internal is None:
        opd_internal = np.zeros_like(zwfs_ns_current.grid.pupil_mask, dtype=float)

    if loop_schedule is None:
        loop_schedule = [(0, "open")]

    # Ensure schedule sorted
    loop_schedule = sorted(loop_schedule, key=lambda x: int(x[0]))

    valid_modes = {"open", "slow", "fast", "fast+slow"}
    for it0, mode in loop_schedule:
        if mode not in valid_modes:
            raise ValueError(f"Invalid loop mode '{mode}'. Must be one of {sorted(valid_modes)}")

    # convenience
    pm = zwfs_ns_current.grid.pupil_mask.astype(bool)

    # --- telemetry init (faithful to your fields; some names cleaned) ---
    telem = {
        # references recorded once (but keep as lists so you can run multiple configs and append)
        "I0_cal": [],
        "N0_cal": [],
        "I0_used": [],
        "N0_used": [],

        # per-iter telemetry
        "it": [],
        "loop_mode": [],           # string mode per iter
        "reset_events": [],

        "clear_pup": [],           # n00
        "i": [],                   # i
        "s": [],                   # s (flattened)

        "e_LO": [],
        "e_HO": [],

        "u_LO_fast": [],
        "u_HO_fast": [],
        "c_LO_fast": [],
        "c_HO_fast": [],

        "u_LO_slow": [],
        "u_HO_slow": [],
        "c_LO_slow": [],
        "c_HO_slow": [],

        "d_cmd": [],               # total DM increment command (140,)
        "dm_cmd": [],              # applied absolute dm cmd (140,)

        # OPD maps + RMS scalars
        "scrn_pre_bld_w_NCPA": [],   # opd_input + opd_internal
        "scrn_post_bld_w_NCPA": [],  # opd_input + opd_dm + opd_internal

        "rmse_before_wo_NCPA": [],
        "rmse_before_w_NCPA": [],
        "rmse_after_wo_NCPA": [],
        "rmse_after_w_NCPA": [],
    }

    # record reference intensities used during calibration
    telem["I0_cal"].append(zwfs_ns_calibration.reco.I0)
    telem["N0_cal"].append(zwfs_ns_calibration.reco.N0)

    # --- init DM to flat at start ---
    zwfs_ns_current.dm.current_cmd = zwfs_ns_current.dm.dm_flat.copy()


    # ------- Update N0 onsky ! this is first step unless user gives the reference intensities
    if user_ref_intensities is None:
        N0_list = update_N0(  
                    zwfs_ns=zwfs_ns_current,  
                    phasescreen=phasescreen, 
                    scintillation_screen=scintillation_screen, 
                    update_scintillation_fn=update_scintillation_fn,   # pass your update_scintillation
                    static_input_field=static_input_field,
                    detector=detector,
                    basis=basis,
                    dx=dx,
                    amp_input_0=amp_input_0,
                    propagation_distance=propagation_distance,
                    opd_internal=opd_internal,         # internal OPD from internal aberrations, lab turbluence etc
                    N_iter_estimation =100, #  
                    it_lag=it_lag, # lag for first stage AO to simulate temporal errors, if 0 then perfect first stage removal of Nmodes_removed
                    Nmodes_removed=Nmodes_removed, # number of modes removed in first stage AO
                    phase_scaling_factor=phase_scaling_factor, # scalar phase scaling factor to apply to first stage AO residuals 
                    include_scintillation=include_scintillation, # do we apply scintillation
                    jumps_per_iter=jumps_per_iter,  # how many rows scintillation phase screen jumps per iteration  ):
                    verbose_every = 20 # print something verbose_every iterations 
                    )

        N0_onsky = np.mean( N0_list, axis = 0)
        # update the current ZWFS appropiately 
        zwfs_ns_current.reco.N0 = N0_onsky

    # those used for processing the signal in the experiement 
    if user_ref_intensities is not None:
        try:
            user_I0, user_N0 = user_ref_intensities
        except:
            raise UserWarning("user_ref_intensities cannot decompose to user_I0, user_N0  = user_ref_intensities. Ensure it is a tuple")
        # make sure they're the right shape 
        if np.shape(user_I0) != np.shape(zwfs_ns_current.reco.I0):
            raise UserWarning("user zwfs reference intensities: np.array(user_I0) != zwfs_ns_current.reco.I0.shape")
        if np.shape(user_N0) != np.shape(zwfs_ns_current.reco.N0):
            raise UserWarning("user clear reference intensities: np.array(user_N0) != zwfs_ns_current.reco.N0.shape")

        telem["I0_used"].append(user_I0)
        telem["N0_used"].append(user_N0)
    else:    
        telem["I0_used"].append(zwfs_ns_calibration.reco.I0) # we use the calibration intentionally here 
        telem["N0_used"].append(zwfs_ns_current.reco.N0) # we use the current intentionally here since we update N0 on sky before starting 

    # --- latency buffer for first-stage AO reconstructor ---
    reco_list = []
    for _ in range(int(it_lag)):
        phasescreen.add_row()
        _, reco_1 = bldr.first_stage_ao(
            phasescreen,
            Nmodes_removed=Nmodes_removed,
            basis=basis,
            phase_scaling_factor=phase_scaling_factor,
            return_reconstructor=True,
        )
        reco_list.append(reco_1)


    # --- helper: mode from schedule at iteration it ---
    def _mode_at(it):
        mode = loop_schedule[0][1]
        for it0, m in loop_schedule:
            if it >= int(it0):
                mode = m
            else:
                break
        return mode



    # --- main loop ---
    for it in range(int(N_iter)):

        if (verbose_every is not None) and (verbose_every > 0) and (it % int(verbose_every) == 0):
            print(f"{it}/{N_iter}  ({100.0*it/max(1,N_iter):.1f}%)")


        
        loop_mode = _mode_at(it)

        if static_input_field is None: # then we go ahead with Kolmogorov 
            # --- evolve turbulence + AO1 residual (w/ lag) ---
            phasescreen.add_row()

            _, reco_1 = bldr.first_stage_ao(
                phasescreen,
                Nmodes_removed=Nmodes_removed,
                basis=basis,
                phase_scaling_factor=phase_scaling_factor,
                return_reconstructor=True,
            )
            reco_list.append(reco_1)

            ao_1 = basis[0] * (phase_scaling_factor * phasescreen.scrn - reco_list.pop(0))
            opd_input = phase_scaling_factor * zwfs_ns_current.optics.wvl0 / (2 * np.pi) * ao_1  # [m]

            # --- evolve scintillation + amplitude ---
            if include_scintillation and (scintillation_screen is not None):
                for _ in range(int(jumps_per_iter)):
                    scintillation_screen.add_row()

                amp_scint = update_scintillation_fn(
                    high_alt_phasescreen=scintillation_screen,
                    pxl_scale=dx,
                    wavelength=zwfs_ns_current.optics.wvl0,
                    final_size=None,
                    jumps=0,
                    propagation_distance=propagation_distance,
                )
                amp_input = amp_input_0 * amp_scint
            else:
                amp_input = amp_input_0

        elif (static_input_field is not None) and (np.shape(static_input_field) == np.shape(zwfs_ns_current.grid.pupil_mask)):
            opd_input = static_input_field # user defined static input field 

            # --- evolve scintillation + amplitude ---
            if include_scintillation and (scintillation_screen is not None):
                for _ in range(int(jumps_per_iter)):
                    scintillation_screen.add_row()

                amp_scint = update_scintillation_fn(
                    high_alt_phasescreen=scintillation_screen,
                    pxl_scale=dx,
                    wavelength=zwfs_ns_current.optics.wvl0,
                    final_size=None,
                    jumps=0,
                    propagation_distance=propagation_distance,
                )
                amp_input = amp_input_0 * amp_scint
            else:
                amp_input = amp_input_0
            
        else:
            raise UserWarning(f"input static_input_field shape seems wrong\nstatic_input_field.shape={static_input_field.shape}\n amp_input_0.shape={amp_input_0.shape}")


        # --- apply current DM command to compute DM OPD contribution ---
        opd_dm = bldr.get_dm_displacement(
            command_vector=zwfs_ns_current.dm.current_cmd,
            gain=zwfs_ns_current.dm.opd_per_cmd,
            sigma=zwfs_ns_current.grid.dm_coord.act_sigma_wavesp,
            X=zwfs_ns_current.grid.wave_coord.X,
            Y=zwfs_ns_current.grid.wave_coord.Y,
            x0=zwfs_ns_current.grid.dm_coord.act_x0_list_wavesp,
            y0=zwfs_ns_current.grid.dm_coord.act_y0_list_wavesp,
        )

        opd_total = opd_input + opd_dm  # opd_internal handled separately in get_frame/get_N0 below

        # clear pupil (in real life we cant measure this simultaneously, but we do here for analytics)
        n00 = bldr.get_N0(
            opd_total,
            amp_input,
            opd_internal,
            zwfs_ns_current,
            detector=detector,
            use_pyZelda=False,
        ).astype(float)

        # --- ZWFS measurement ---
        i = bldr.get_frame(
            opd_total,
            amp_input,
            opd_internal,
            zwfs_ns_current,
            detector=detector,
            use_pyZelda=False,
        ).astype(float)


        # --- signal (keep exactly convention used in interaction matrix) ---
        if user_ref_intensities is None: # no user reference intensities 
            if zwfs_ns_calibration.reco.normalization_method == 'subframe mean':
                s = (
                    i / (np.mean(i) + 1e-18)
                    - zwfs_ns_calibration.reco.I0 / (np.mean(zwfs_ns_calibration.reco.I0) + 1e-18)
                ).reshape(-1)
            elif zwfs_ns_calibration.reco.normalization_method == 'clear pupil mean':
                # we need the calibrated N0 and N0 from sky  (why we normalize i buy current N0, and I0 by calibration N0)
                s = (
                    i / (np.mean(zwfs_ns_current.reco.N0[zwfs_ns_calibration.reco.interior_pup_filt]) + 1e-18)
                    - zwfs_ns_calibration.reco.I0 / (np.mean(zwfs_ns_calibration.reco.N0[zwfs_ns_calibration.reco.interior_pup_filt]) + 1e-18)
                ).reshape(-1)
            else:
                raise UserWarning("no valid normalization method in zwfs_ns_calibration.reco.normalization_method ")
        else: 
            
            if zwfs_ns_calibration.reco.normalization_method == 'subframe mean':
                s = (
                    i / (np.mean(i) + 1e-18)
                    - user_I0 / (np.mean( user_I0 ) + 1e-18)
                ).reshape(-1)
            elif zwfs_ns_calibration.reco.normalization_method == 'clear pupil mean':
                # we need the calibrated N0 and N0 from sky  
                s = (
                    i / (np.mean(zwfs_ns_current.reco.N0[zwfs_ns_calibration.reco.interior_pup_filt]) + 1e-18)
                    - user_I0 / (np.mean(user_N0[zwfs_ns_calibration.reco.interior_pup_filt]) + 1e-18)
                ).reshape(-1)
            else:
                raise UserWarning("no valid normalization method in zwfs_ns_calibration.reco.normalization_method ")
        # --- reconstruction in desired space ---
        if signal_space.strip().lower() == "dm":
            # project pixel signal to DM actuator sampling using CALIBRATION registration
            s_dm = DM_interpolate_fn(
                image=s.reshape(zwfs_ns_calibration.reco.I0.shape),
                pixel_coords=zwfs_ns_calibration.dm2pix_registration.actuator_coord_list_pixel_space,
            )
            e_LO = zwfs_ns_calibration.reco.I2M_TT @ s_dm
            e_HO = zwfs_ns_calibration.reco.I2M_HO @ s_dm

        elif signal_space.strip().lower() == "pix":
            e_LO = zwfs_ns_calibration.reco.I2M_TT @ s
            e_HO = zwfs_ns_calibration.reco.I2M_HO @ s

        else:
            raise ValueError("signal_space must be 'pix' or 'dm'")

        # --- run controllers conditionally (schedule gates process calls) ---
        # Defaults if a controller is not provided: treat as open loop for that branch.

        do_fast = loop_mode in ("fast", "fast+slow")
        do_slow = loop_mode in ("slow", "fast+slow")



        # HOLD previous states by default
        if ctrl_fast is not None:
            u_LO_fast = ctrl_fast.u_LO
            u_HO_fast = ctrl_fast.u_HO
        else:
            u_LO_fast = np.zeros_like(e_LO)
            u_HO_fast = np.zeros_like(e_HO)

        if ctrl_slow is not None:
            u_LO_slow = ctrl_slow.u_LO
            u_HO_slow = ctrl_slow.u_HO
        else:
            u_LO_slow = np.zeros_like(e_LO)
            u_HO_slow = np.zeros_like(e_HO)

        do_fast = loop_mode in ("fast", "fast+slow")
        do_slow = loop_mode in ("slow", "fast+slow")

        if do_fast:
            if ctrl_fast is None:
                raise ValueError("loop_mode requests fast control but ctrl_fast is None")
            u_LO_fast, u_HO_fast = ctrl_fast.process(e_LO, e_HO)

        if do_slow:
            if ctrl_slow is None:
                raise ValueError("loop_mode requests slow control but ctrl_slow is None")
            u_LO_slow, u_HO_slow = ctrl_slow.process(e_LO, e_HO)

    
        # --- map to DM increments (calibration M2C) ---
        c_LO_fast = zwfs_ns_calibration.reco.M2C_LO @ u_LO_fast
        c_HO_fast = zwfs_ns_calibration.reco.M2C_HO @ u_HO_fast

        c_LO_slow = zwfs_ns_calibration.reco.M2C_LO @ u_LO_slow
        c_HO_slow = zwfs_ns_calibration.reco.M2C_HO @ u_HO_slow

        d_cmd = c_LO_fast + c_HO_fast + c_LO_slow + c_HO_slow

        # --- apply to DM: flat - increment (your sign convention) ---
        cmd = zwfs_ns_current.dm.dm_flat - d_cmd
        zwfs_ns_current.dm.current_cmd = cmd

        # --- performance / safety metrics ---
        opd_dm_after = bldr.get_dm_displacement(
            command_vector=zwfs_ns_current.dm.current_cmd,
            gain=zwfs_ns_current.dm.opd_per_cmd,
            sigma=zwfs_ns_current.grid.dm_coord.act_sigma_wavesp,
            X=zwfs_ns_current.grid.wave_coord.X,
            Y=zwfs_ns_current.grid.wave_coord.Y,
            x0=zwfs_ns_current.grid.dm_coord.act_x0_list_wavesp,
            y0=zwfs_ns_current.grid.dm_coord.act_y0_list_wavesp,
        )

        opd_input_w_NCPA = opd_input + opd_internal
        opd_res_wo_NCPA = opd_input + opd_dm_after
        opd_res_w_NCPA = opd_input + opd_dm_after + opd_internal

        # your safety trigger: std of commanded increment converted to OPD scale
        sigma_cmd = np.std(c_HO_fast + c_LO_fast + c_HO_slow + c_LO_slow) * zwfs_ns_current.dm.opd_per_cmd

        if sigma_cmd > opd_threshold:
            # reset DM and (optionally) controller states
            zwfs_ns_current.dm.current_cmd = zwfs_ns_current.dm.dm_flat.copy()

            if ctrl_fast is not None and reset_resets_fast:
                ctrl_fast.reset()
            if ctrl_slow is not None and reset_resets_slow:
                ctrl_slow.reset()

            telem["reset_events"].append(it)

            # recompute OPD after reset for stored diagnostics (optional but sane)
            opd_dm_after = bldr.get_dm_displacement(
                command_vector=zwfs_ns_current.dm.current_cmd,
                gain=zwfs_ns_current.dm.opd_per_cmd,
                sigma=zwfs_ns_current.grid.dm_coord.act_sigma_wavesp,
                X=zwfs_ns_current.grid.wave_coord.X,
                Y=zwfs_ns_current.grid.wave_coord.Y,
                x0=zwfs_ns_current.grid.dm_coord.act_x0_list_wavesp,
                y0=zwfs_ns_current.grid.dm_coord.act_y0_list_wavesp,
            )
            opd_res_wo_NCPA = opd_input + opd_dm_after
            opd_res_w_NCPA = opd_input + opd_dm_after + opd_internal

        # --- store telemetry ---
        if it >= int(N_burn):
            telem["it"].append(it)
            telem["loop_mode"].append(loop_mode)

            telem["clear_pup"].append( n00 )
            telem["i"].append(i)
            telem["s"].append(s)

            telem["e_LO"].append(e_LO)
            telem["e_HO"].append(e_HO)

            telem["u_LO_fast"].append(u_LO_fast)
            telem["u_HO_fast"].append(u_HO_fast)
            telem["c_LO_fast"].append(c_LO_fast)
            telem["c_HO_fast"].append(c_HO_fast)

            telem["u_LO_slow"].append(u_LO_slow)
            telem["u_HO_slow"].append(u_HO_slow)
            telem["c_LO_slow"].append(c_LO_slow)
            telem["c_HO_slow"].append(c_HO_slow)

            telem["d_cmd"].append(d_cmd)
            telem["dm_cmd"].append(zwfs_ns_current.dm.current_cmd.copy())

            telem["scrn_pre_bld_w_NCPA"].append(opd_input_w_NCPA)
            telem["scrn_post_bld_w_NCPA"].append(opd_res_w_NCPA)

            telem["rmse_before_wo_NCPA"].append(np.std(opd_input[pm]))
            telem["rmse_before_w_NCPA"].append(np.std(opd_input_w_NCPA[pm]))
            telem["rmse_after_wo_NCPA"].append(np.std(opd_res_wo_NCPA[pm]))
            telem["rmse_after_w_NCPA"].append(np.std(opd_res_w_NCPA[pm]))

    # --- re-flatten DM on exit ---
    zwfs_ns_current.dm.current_cmd = zwfs_ns_current.dm.dm_flat.copy()
    return telem



def run_experiment_grid(
    *,
    zwfs_current_factory,      # callable -> returns fresh zwfs_ns_current (deepcopy done inside)
    zwfs_cal_factory,          # callable -> returns fresh zwfs_ns_calibration (deepcopy done inside)
    scrn_factory,              # callable -> returns fresh phase screen
    scint_factory,             # callable -> returns fresh scint screen
    basis,
    detector,
    amp_input_0,
    dx,
    propagation_distance,
    update_scintillation_fn,
    DM_interpolate_fn,
    configs,                   # list of dict configs
    common_kwargs=None,        # forwarded to eval_onsky
    keys_to_not_evaluate={"name","ctrl_fast","ctrl_slow","loop_schedule","user_ref_intensities","disable_ho"}
):
    """
    Returns:
      results: dict[name -> telemetry]
    """
    results = {}
    common_kwargs = {} if common_kwargs is None else dict(common_kwargs)

    for cfg in configs:
        name = cfg["name"]

        # fresh objects each run (avoid cross-contamination)
        zwfs_ns_current = zwfs_current_factory()
        zwfs_ns_cal     = zwfs_cal_factory()
        scrn            = scrn_factory()
        scint_scrn      = scint_factory()

        # controllers (new instance per run unless explicitly shared)
        ctrl_fast = cfg.get("ctrl_fast", None)
        ctrl_slow = cfg.get("ctrl_slow", None)

        telem = eval_onsky(
            zwfs_ns_current=zwfs_ns_current,
            zwfs_ns_calibration=zwfs_ns_cal,
            phasescreen=scrn,
            scintillation_screen=scint_scrn,
            basis=basis,
            detector=detector,
            amp_input_0=amp_input_0,
            dx=dx,
            propagation_distance=propagation_distance,
            update_scintillation_fn=update_scintillation_fn,
            DM_interpolate_fn=DM_interpolate_fn,
            loop_schedule=cfg.get("loop_schedule", [(0, "open")]),
            ctrl_fast=ctrl_fast,
            ctrl_slow=ctrl_slow,
            user_ref_intensities = cfg.get("user_ref_intensities", None),
            # allow per-config overrides
            **common_kwargs,
            **{k: v for k, v in cfg.items() if k not in keys_to_not_evaluate},
        )

        results[name] = telem
        print(f"[grid] finished: {name}")

    return results




####### PLOTTING UTILS 
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import RangeSlider, CheckButtons

# results 
def quicklook_experiment_results(
    results,
    *,
    labels=None,
    wvl0=None,
    max_cols=3,
    init_ho=(0, 20),
    init_lo=(0, 2),
    plot_u=True,
    plot_errors=True,
    interactive=True,
    show=True,
):
    """
    Single-figure quicklook for Baldr experiment results returned from "run_experiment_grid(..)" method

    Plots per label (columns):
      row 0: Strehl proxy (pre/post)
      row 1: HO errors (selected mode range)
      row 2: LO errors (selected mode range)
    Optional toggles:
      - plot errors
      - plot u (controller state u_LO/u_HO if present)
    Widgets are embedded in a dedicated bottom row (no overlap).
    """

    if labels is None:
        labels = list(results.keys())
    labels = list(labels)
    ncols = min(max_cols, len(labels))
    nrows_panels = 3 + (1 if plot_u else 0)  # strehl + HO + LO (+ u row)
    # Add one extra row for widgets if interactive
    nrows_total = nrows_panels + (1 if interactive else 0)

    # pick wavelength
    if wvl0 is None:
        # try infer from first result if you stored it; else default H-band
        wvl0 = 1.65e-6

    # ---------- layout ----------
    fig = plt.figure(figsize=(5.2 * ncols, 2.5 * nrows_total))
    gs = GridSpec(
        nrows=nrows_total,
        ncols=ncols,
        figure=fig,
        height_ratios=([1, 1, 1] + ([1] if plot_u else []) + ([0.42] if interactive else [])),
        hspace=0.28,
        wspace=0.22,
    )

    # Axes per panel row/col
    ax_strehl = []
    ax_ho = []
    ax_lo = []
    ax_u = []  # optional

    for j in range(ncols):
        ax_strehl.append(fig.add_subplot(gs[0, j]))
        ax_ho.append(fig.add_subplot(gs[1, j], sharex=ax_strehl[j]))
        ax_lo.append(fig.add_subplot(gs[2, j], sharex=ax_strehl[j]))
        if plot_u:
            ax_u.append(fig.add_subplot(gs[3, j], sharex=ax_strehl[j]))

    # One bottom row spanning all columns for widgets (embedded)
    if interactive:
        ax_widget_row = fig.add_subplot(gs[-1, :])
        ax_widget_row.axis("off")
        # create sub-axes inside this row using figure coordinates
        # (relative placement within the bottom row area)
        bbox = ax_widget_row.get_position()
        left, bottom, width, height = bbox.x0, bbox.y0, bbox.width, bbox.height

        ax_ho_slider = fig.add_axes([left + 0.02 * width, bottom + 0.55 * height, 0.60 * width, 0.35 * height])
        ax_lo_slider = fig.add_axes([left + 0.02 * width, bottom + 0.10 * height, 0.60 * width, 0.35 * height])
        ax_checks    = fig.add_axes([left + 0.66 * width, bottom + 0.08 * height, 0.32 * width, 0.82 * height])
    else:
        ax_ho_slider = ax_lo_slider = ax_checks = None

    # ---------- helpers ----------
    def _get_arr(t, key):
        return np.asarray(t.get(key, []), float)

    def _strehl_from_rmse(rmse):
        rmse = np.asarray(rmse, float)
        return np.exp(-(2 * np.pi / wvl0 * rmse) ** 2)

    # Cache arrays for speed (no recomputation in slider callback)
    cache = {}
    for lab in labels:
        t = results[lab]
        cache[lab] = dict(
            rmse_pre=_get_arr(t, "rmse_before_w_NCPA"),
            rmse_post=_get_arr(t, "rmse_after_w_NCPA"),
            e_HO=_get_arr(t, "e_HO"),
            e_LO=_get_arr(t, "e_LO"),
            u_HO_fast=_get_arr(t, "u_HO_fast") if "u_HO_fast" in t else None,
            u_LO_fast=_get_arr(t, "u_LO_fast") if "u_LO_fast" in t else None,
            loop_mode=list(t.get("loop_mode", [])),
            reset_events=list(t.get("reset_events", [])),
        )

    # determine global mode counts from first label that has data
    nHO0 = None
    nLO0 = None
    for lab in labels:
        e_HO = cache[lab]["e_HO"]
        e_LO = cache[lab]["e_LO"]
        if e_HO.ndim == 2 and e_HO.size:
            nHO0 = e_HO.shape[1]
        if e_LO.ndim == 2 and e_LO.size:
            nLO0 = e_LO.shape[1]
        if nHO0 is not None and nLO0 is not None:
            break
    if nHO0 is None: nHO0 = 1
    if nLO0 is None: nLO0 = 1

    # clamp initial ranges
    def _clamp_rng(rng, n):
        a, b = int(rng[0]), int(rng[1])
        a = max(0, min(a, n))
        b = max(a + 1, min(b, n))
        return (a, b)

    init_ho = _clamp_rng(init_ho, nHO0)
    init_lo = _clamp_rng(init_lo, nLO0)

    # ---------- draw routine ----------
    line_handles = {}  # store per-axis handles so we can clear minimally if desired

    def _clear_axis(ax):
        ax.cla()

    def _draw(ho_rng, lo_rng, do_errors=True, do_u=True):
        ho_a, ho_b = _clamp_rng(ho_rng, nHO0)
        lo_a, lo_b = _clamp_rng(lo_rng, nLO0)

        for j, lab in enumerate(labels[:ncols]):
            d = cache[lab]

            # --- clear axes ---
            _clear_axis(ax_strehl[j])
            _clear_axis(ax_ho[j])
            _clear_axis(ax_lo[j])
            if plot_u:
                _clear_axis(ax_u[j])

            # --- strehl ---
            strehl_pre = _strehl_from_rmse(d["rmse_pre"]) if d["rmse_pre"].size else np.array([])
            strehl_post = _strehl_from_rmse(d["rmse_post"]) if d["rmse_post"].size else np.array([])

            if strehl_pre.size:
                ax_strehl[j].plot(strehl_pre, color="k", ls="--", label="pre")
            if strehl_post.size:
                ax_strehl[j].plot(strehl_post, label="post")
            ax_strehl[j].set_ylim(0, 1.05)
            ax_strehl[j].set_title(lab)

            # --- errors ---
            if do_errors:
                e_HO = d["e_HO"]
                e_LO = d["e_LO"]

                if e_HO.ndim == 2 and e_HO.size:
                    ax_ho[j].plot(e_HO[:, ho_a:ho_b])
                ax_ho[j].axhline(0, color="k", lw=1.5)
                ax_ho[j].set_ylabel("HO err (arb.)")

                if e_LO.ndim == 2 and e_LO.size:
                    ax_lo[j].plot(e_LO[:, lo_a:lo_b])
                ax_lo[j].axhline(0, color="k", lw=1.5)
                ax_lo[j].set_ylabel("LO err (arb.)")
            else:
                ax_ho[j].text(0.5, 0.5, "errors disabled", ha="center", va="center", transform=ax_ho[j].transAxes)
                ax_lo[j].text(0.5, 0.5, "errors disabled", ha="center", va="center", transform=ax_lo[j].transAxes)

            # --- u states (optional) ---
            if plot_u:
                if do_u and (d["u_HO_fast"] is not None) and (d["u_LO_fast"] is not None):
                    u_HO = d["u_HO_fast"]
                    u_LO = d["u_LO_fast"]
                    if u_HO.ndim == 2 and u_HO.size:
                        ax_u[j].plot(u_HO[:, ho_a:ho_b])
                    if u_LO.ndim == 2 and u_LO.size:
                        ax_u[j].plot(u_LO[:, lo_a:lo_b])
                    ax_u[j].axhline(0, color="k", lw=1.0)
                    ax_u[j].set_ylabel("u (fast)")
                else:
                    ax_u[j].text(0.5, 0.5, "u disabled / missing", ha="center", va="center", transform=ax_u[j].transAxes)

            # --- reset events ---
            for reset_event in d["reset_events"]:
                for rr in range(3 + (1 if plot_u else 0)):
                    ax = [ax_strehl, ax_ho, ax_lo] + ([ax_u] if plot_u else [])
                    ax[rr][j].axvline(reset_event, color="r", alpha=0.5, ls=":")

            # --- loop_mode transitions ---
            lm = d["loop_mode"]
            if lm:
                for k in range(1, len(lm)):
                    if lm[k] != lm[k - 1]:
                        for rr in range(3 + (1 if plot_u else 0)):
                            ax = [ax_strehl, ax_ho, ax_lo] + ([ax_u] if plot_u else [])
                            ax[rr][j].axvline(k, color="k", alpha=0.15, ls="-.")

            # cosmetics
            ax_lo[j].set_xlabel("sample")
            if j != 0:
                # keep y labels only on first column to reduce clutter
                ax_ho[j].set_ylabel("")
                ax_lo[j].set_ylabel("")
                if plot_u:
                    ax_u[j].set_ylabel("")

        # legend on first strehl axis
        ax_strehl[0].legend(loc="best", fontsize=10)

        fig.canvas.draw_idle()

    # initial draw
    _draw(init_ho, init_lo, do_errors=plot_errors, do_u=plot_u)

    widget_handles = {}
    if interactive:
        ho_slider = RangeSlider(
            ax=ax_ho_slider, label="HO mode range",
            valmin=0, valmax=nHO0,
            valinit=init_ho, valstep=1
        )
        lo_slider = RangeSlider(
            ax=ax_lo_slider, label="LO mode range",
            valmin=0, valmax=nLO0,
            valinit=init_lo, valstep=1
        )

        check_labels = []
        check_states = []
        if plot_errors:
            check_labels.append("plot errors")
            check_states.append(True)
        else:
            check_labels.append("plot errors")
            check_states.append(False)

        if plot_u:
            check_labels.append("plot u")
            check_states.append(True)

        checks = CheckButtons(ax_checks, check_labels, check_states)

        _in_update = {"flag": False}

        def _update(_=None):
            if _in_update["flag"]:
                return
            _in_update["flag"] = True
            try:
                ho_rng = ho_slider.val
                lo_rng = lo_slider.val

                states = {lab.get_text(): st for lab, st in zip(checks.labels, checks.get_status())}
                do_err = bool(states.get("plot errors", True))
                do_u_ = bool(states.get("plot u", True))

                _draw(ho_rng, lo_rng, do_errors=do_err, do_u=do_u_)
            finally:
                _in_update["flag"] = False

        ho_slider.on_changed(_update)
        lo_slider.on_changed(_update)
        checks.on_clicked(_update)

        widget_handles = dict(fig=fig, ho_slider=ho_slider, lo_slider=lo_slider, checks=checks)

    if show:
        plt.show()

    return widget_handles



def get_worst_offenders(
    results,
    measurement,
    *,
    metric="rms",      # "rms" | "mean" | "max"
    top_n=5,
    absolute=True,
    burn_in=0,
):
    """
    Identify worst offending modes per experiment.

    Parameters
    ----------
    results : dict
        {name -> telemetry dict}
    measurement : str
        Telemetry key, e.g. "e_HO", "e_LO", "u_HO", "u_LO", "s"
    metric : str
        Reduction over time: "rms", "mean", "max"
    top_n : int
        Number of worst offenders to return
    absolute : bool
        Rank by absolute value
    burn_in : int
        Ignore first burn_in samples

    Returns
    -------
    offenders : dict
        name -> dict with keys:
            "indices" : array of indices
            "values"  : array of measurement values
    """
    offenders = {}

    for name, telem in results.items():
        if measurement not in telem:
            continue

        x = np.asarray(telem[measurement], float)

        if x.ndim == 1:
            x = x[burn_in:]
            vals = np.abs(x) if absolute else x
            idx = np.argsort(vals)[::-1][:top_n]
            offenders[name] = dict(indices=idx, values=x[idx])
            continue

        # x shape: (time, modes)
        x = x[burn_in:, :]

        if metric == "rms":
            vals = np.sqrt(np.mean(x**2, axis=0))
        elif metric == "mean":
            vals = np.mean(x, axis=0)
        elif metric == "max":
            vals = np.max(np.abs(x), axis=0)
        else:
            raise ValueError(f"Unknown metric '{metric}'")

        rank = np.argsort(np.abs(vals) if absolute else vals)[::-1]
        idx = rank[:top_n]

        offenders[name] = dict(
            indices=idx,
            values=vals[idx],
        )

    return offenders



def plot_offenders(
    results_or_telem,
    *,
    labels=None,              # if dict provided, optionally choose subset order
    metric="rms",             # "mean" | "absmean" | "rms"
    topk=5,
    include=("e_LO", "e_HO"), # any of: "e_LO","e_HO","s","u_LO","u_HO"
    u_source="fast",          # "fast" | "slow" | "sum" (fast+slow)
    burn=0,                   # drop first burn samples
    max_signal_features=200,  # cap plotted pixel-features if include "s"
    sharey=False,
    figsize=(12, 7),
    title_prefix="",
):
    """
    Plot per-feature metrics and highlight top absolute offenders.

    results_or_telem:
      - dict[name -> telem]  OR
      - single telem dict
    """



    def _as_2d(x, name):
        """Convert list/array to (T, K) float array."""
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 2D after conversion, got shape {arr.shape}")
        return arr


    def _metric_per_feature(X, metric="rms", *, abs_before=False):
        """
        X: (T, K)
        metric:
        - "mean": mean over time (signed)
        - "absmean": mean(|x|) over time
        - "rms": sqrt(mean(x^2)) over time
        abs_before: if True, apply abs before metric (useful for "mean" but usually use "absmean")
        """
        X = np.asarray(X, dtype=float)
        if abs_before:
            X = np.abs(X)

        m = metric.strip().lower()
        if m == "mean":
            return np.mean(X, axis=0)
        if m == "absmean":
            return np.mean(np.abs(X), axis=0)
        if m == "rms":
            return np.sqrt(np.mean(X**2, axis=0))
        raise ValueError("metric must be one of {'mean','absmean','rms'}")


    def _topk_abs(v, k=5):
        """Return indices of top-k by absolute value, in descending order."""
        v = np.asarray(v, dtype=float)
        k = int(min(k, v.size))
        if k <= 0:
            return np.array([], dtype=int)
        idx = np.argpartition(np.abs(v), -k)[-k:]
        idx = idx[np.argsort(np.abs(v[idx]))[::-1]]
        return idx


    # ---- normalize input to dict[name->telem] ----
    if isinstance(results_or_telem, dict) and ("e_LO" in results_or_telem or "e_HO" in results_or_telem):
        runs = {"run": results_or_telem}
    else:
        runs = dict(results_or_telem)

    if labels is None:
        labels = list(runs.keys())
    else:
        labels = list(labels)

    include = tuple(include)
    nrows = len(include)
    ncols = len(labels)

    fig, ax = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        squeeze=False,
        sharex=False,
        sharey=sharey
    )

    # ---- helper to fetch U arrays consistently ----
    def _get_u(telem, which):
        # which: "LO" or "HO"
        if u_source == "fast":
            key = f"u_{which}_fast"
            return telem.get(key, None)
        if u_source == "slow":
            key = f"u_{which}_slow"
            return telem.get(key, None)
        if u_source == "sum":
            a = np.asarray(telem.get(f"u_{which}_fast", 0.0), dtype=float)
            b = np.asarray(telem.get(f"u_{which}_slow", 0.0), dtype=float)
            return a + b
        raise ValueError("u_source must be 'fast', 'slow', or 'sum'")

    # ---- main plotting ----
    for cc, name in enumerate(labels):
        telem = runs[name]

        # slice time
        def _slice_time(arr):
            arr = _as_2d(arr, "telem array")
            if burn > 0:
                return arr[int(burn):, :]
            return arr

        for rr, what in enumerate(include):
            a = ax[rr, cc]

            if what in ("e_LO", "e_HO"):
                X = _slice_time(telem[what])
                vals = _metric_per_feature(X, metric=metric)
                feat_name = what

            elif what == "s":
                X = _slice_time(telem["s"])
                # cap pixels for sanity
                if X.shape[1] > int(max_signal_features):
                    X = X[:, :int(max_signal_features)]
                vals = _metric_per_feature(X, metric=metric)
                feat_name = f"s[:{X.shape[1]}]"

            elif what == "u_LO":
                U = _get_u(telem, "LO")
                if U is None:
                    a.text(0.5, 0.5, "u_LO not found", ha="center", va="center")
                    a.set_axis_off()
                    continue
                X = _slice_time(U)
                vals = _metric_per_feature(X, metric=metric)
                feat_name = f"u_LO ({u_source})"

            elif what == "u_HO":
                U = _get_u(telem, "HO")
                if U is None:
                    a.text(0.5, 0.5, "u_HO not found", ha="center", va="center")
                    a.set_axis_off()
                    continue
                X = _slice_time(U)
                vals = _metric_per_feature(X, metric=metric)
                feat_name = f"u_HO ({u_source})"

            else:
                raise ValueError(f"Unknown include entry: {what}")

            # base curve
            x = np.arange(vals.size)
            a.plot(x, vals, lw=1.5)

            # highlight offenders
            idx = _topk_abs(vals, k=topk)
            for j, ii in enumerate(idx):
                # use a distinct marker and label for legend
                # label includes rounded metric value
                v = vals[ii]
                label = f"#{j+1} i={ii}: {v:+.3g}"
                a.plot(ii, v, marker="x", ms=8, mew=2, linestyle="None", label=label)

            a.axhline(0.0, lw=1.0, alpha=0.4)

            # titles/labels
            if rr == 0:
                a.set_title(f"{title_prefix}{name}")
            if cc == 0:
                a.set_ylabel(f"{feat_name}\n({metric})")
            if rr == nrows - 1:
                a.set_xlabel("index")

            # keep legend readable: only show if there are offenders
            if idx.size > 0:
                a.legend(fontsize=9, loc="best", frameon=True)

    plt.tight_layout()
    return fig, ax




def plot_modes_in_intensity_space_per_config(
    configs,
    zwfs_cal_factory_from_cfg,
    *,
    which="HO",                       # "HO" | "LO"
    mode_indices_by_cfg=None,         # dict[name->list] OR list-of-lists aligned with configs
    max_modes_per_cfg=5,              # used only if mode_indices_by_cfg is None
    pix_shape=None,                   # (ny,nx) or None -> infer from zwfs_tmp.reco.I0.shape
    cmap="RdBu_r",
    clip_percentile=99.5,             # per-subplot robust symmetric scaling
    cbar_location="right",            # "right" | "bottom" | "top"
    cbar_size="4.5%",
    cbar_pad=0.06,
    axis_off=True,
    suptitle=None,
    figsize_per_col=3.2,
    figsize_per_row=2.6,
):
    """
    Plot I2M rows (modes) reshaped into pixel/intensity space.

    Supports different mode indices per config:
      - mode_indices_by_cfg can be:
          (a) dict keyed by cfg["name"]
          (b) list of lists aligned with configs
          (c) None -> uses first max_modes_per_cfg indices [0..max_modes_per_cfg-1]

    Layout:
      - columns = configs
      - rows    = max number of modes across configs (ragged supported)
      - each subplot gets its own colorbar + unique title describing mode index
    """
    # ---- resolve mode indices per config ----
    cfg_names = [cfg.get("name", f"cfg{ii}") for ii, cfg in enumerate(configs)]

    if mode_indices_by_cfg is None:
        mode_lists = [list(range(int(max_modes_per_cfg))) for _ in configs]
    elif isinstance(mode_indices_by_cfg, dict):
        mode_lists = []
        for name in cfg_names:
            if name not in mode_indices_by_cfg:
                mode_lists.append([])
            else:
                mode_lists.append(list(mode_indices_by_cfg[name]))
    else:
        # assume list-of-lists aligned with configs
        if len(mode_indices_by_cfg) != len(configs):
            raise ValueError("mode_indices_by_cfg list must have same length as configs.")
        mode_lists = [list(v) for v in mode_indices_by_cfg]

    ncols = len(configs)
    nrows = max((len(v) for v in mode_lists), default=0)
    if nrows == 0 or ncols == 0:
        raise ValueError("No modes to plot (nrows==0) or no configs (ncols==0).")

    fig_w = max(6.0, figsize_per_col * ncols)
    fig_h = max(4.0, figsize_per_row * nrows)
    fig, ax = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    # ---- helper to fetch I2M ----
    def _get_I2M(zwfs_tmp):
        w = which.strip().upper()
        if w == "HO":
            return np.asarray(zwfs_tmp.reco.I2M_HO, float), "I2M_HO"
        elif w == "LO":
            if hasattr(zwfs_tmp.reco, "I2M_LO"):
                return np.asarray(zwfs_tmp.reco.I2M_LO, float), "I2M_LO"
            if hasattr(zwfs_tmp.reco, "I2M_TT"):
                return np.asarray(zwfs_tmp.reco.I2M_TT, float), "I2M_TT"
            raise AttributeError("Could not find LO reconstructor (I2M_LO or I2M_TT).")
        else:
            raise ValueError("which must be 'HO' or 'LO'")

    # ---- plot ----
    for cc, cfg in enumerate(configs):
        name = cfg_names[cc]
        zwfs_tmp = zwfs_cal_factory_from_cfg(cfg)
        I2M, I2M_name = _get_I2M(zwfs_tmp)

        shp = pix_shape if pix_shape is not None else zwfs_tmp.reco.I0.shape
        ny, nx = shp

        for rr in range(nrows):
            a = ax[rr, cc]

            # this config may not have this many modes -> blank cell
            if rr >= len(mode_lists[cc]):
                a.set_axis_off()
                continue

            mi = int(mode_lists[cc][rr])

            if mi < 0 or mi >= I2M.shape[0]:
                a.set_axis_off()
                a.set_title(f"{name}\n{I2M_name}[{mi}] (OOR)", fontsize=10)
                continue

            row = np.asarray(I2M[mi], float)
            if row.size != ny * nx:
                raise ValueError(
                    f"{name}: {I2M_name}[{mi}] length {row.size} != ny*nx={ny*nx} for pix_shape={shp}. "
                    f"Pass pix_shape explicitly if needed."
                )

            img = row.reshape(shp)

            # per-subplot robust symmetric scaling
            s = np.nanpercentile(np.abs(img), clip_percentile)
            if (not np.isfinite(s)) or s == 0:
                s = 1.0
            vmin, vmax = -s, +s

            im = a.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)

            # unique title per subplot with mode index
            a.set_title(f"{name}\n{I2M_name}[{mi}]", fontsize=10)

            if axis_off:
                a.set_xticks([])
                a.set_yticks([])

            # per-subplot colorbar using make_axes_locatable (stable)
            divider = make_axes_locatable(a)
            cax = divider.append_axes(cbar_location, size=cbar_size, pad=cbar_pad)
            cb = fig.colorbar(im, cax=cax, orientation=("vertical" if cbar_location == "right" else "horizontal"))
            cb.ax.tick_params(labelsize=8)

    if suptitle is None:
        suptitle = f"Mode templates in pixel/intensity space ({which.upper()})"
    fig.suptitle(suptitle, y=0.995)

    # no tight_layout (fights locatable caxes)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.06, wspace=0.25, hspace=0.35)

    return fig, ax




# # Different mode indices per config (keyed by cfg["name"])
# mode_indices_by_cfg = { # keys from config 
#     'CPM_Sol_Sol': [0, 1, 2, 3, 4],
#     'CPM_Sol_AT': [10, 11, 12],   # fewer rows -> blanks below
#     'CPM_AT_AT':[10, 11, 12],
# }

# fig, ax = plot_modes_in_intensity_space_per_config(
#     configs=configs,
#     zwfs_cal_factory_from_cfg=zwfs_cal_factory_from_cfg,
#     which="HO",
#     mode_indices_by_cfg=mode_indices_by_cfg,
#     pix_shape=(48, 48),
#     cbar_location="right",
#     clip_percentile=99.5,
#     suptitle="Per-config HO I2M templates (custom indices)",
# )
# plt.show()


"""
# A complete simple example of end-to-end comparison of different signal processing sematics 
# under (un-optimized) closed loop control 


import numpy as np
import matplotlib.pyplot as plt

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
Nmodes_removed = 7            # pick the AO1 regime you want here
N_iter = 1000                   # total closed-loop iterations
N_burn = 0                   # throw away transient
jumps_per_iter = 1             # scintillation decorrelation per iter

phase_scaling_factor = 1.0
it_lag = 10 #3 # how many exposures does fist stage AO lag 


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
# Build two calibration ZWFS objects with different normalization conventions
#   - CPM : clear pupil mean
#   - SFM : subframe mean

HO_inv_method = "eigen"
what_space = "pix"   # or "dm"

zwfs_ns_dict = {}

for tag, norm_method in {
    "CPM": "clear pupil mean",
    "SFM": "subframe mean",
    "CPM_AT": "clear pupil mean",
}.items():

    # fresh copy
    zwfs_ns_cal = copy.deepcopy(zwfs_ns)

    # flatten DM
    zwfs_ns_cal.dm.current_cmd = zwfs_ns_cal.dm.dm_flat.copy()

    # 
    if tag =='CPM_AT':
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
        normalization_method=norm_method,
    )

    # build reconstructor
    _ = bldr.reco_method(
        zwfs_ns_cal,
        LO=2,
        LO_inv_method="eigen",
        HO_inv_method=HO_inv_method,
        project_out_of_LO=None,
        project_out_of_HO="lo_command",
        truncation_idx=30,
        filter_dm_pupil=None,
        eps=1e-12,
        what_space=what_space,
    )

    zwfs_ns_dict[tag] = zwfs_ns_cal

# # unpack if you want the original names
# zwfs_ns_CPM = zwfs_ns_dict["CPM"]
# zwfs_ns_SFM = zwfs_ns_dict["SFM"]


# ---

def make_ctrl(ki, leak):

    return bld_experiment.LeakyIntegratorController(
        n_lo=zwfs_ns_dict["CPM"].reco.I2M_TT.shape[0],
        n_ho=zwfs_ns_dict["CPM"].reco.I2M_HO.shape[0],
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


# experiment grid
configs = [
    dict(
        name="internal_CPM",
        loop_schedule=[(0, "open"), (100, "fast")],
        user_ref_intensities=None,
        ctrl_slow=None,
        ctrl_fast=make_ctrl(ki=0.25, leak=0.95),
        cal_tag="CPM",  # used to deepcopy the relevant zwfs_ns object
    ),
    dict(
        name="internal_SFM",
        loop_schedule=[(0, "open"), (100, "fast")],
        user_ref_intensities=None,
        ctrl_slow=None,
        ctrl_fast=make_ctrl(ki=0.25, leak=0.95),
        cal_tag="SFM",  # used to deepcopy the relevant zwfs_ns object
    ),
]


# callable function to copy the zwfs_ns for experiment grid
def zwfs_cal_factory_from_cfg(cfg):
    # BUGFIX: avoid None.strip() if cal_tag missing
    tag = (cfg.get("cal_tag") or "").strip().lower()
    if tag == "sfm":
        return copy.deepcopy(zwfs_ns_dict["SFM"])
    elif tag == "cpm":
        return copy.deepcopy(zwfs_ns_dict["CPM"])
    else:
        raise UserWarning('cal tag is not valid (expected "CPM" or "SFM")')


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

    results.update(
        bld_experiment.run_experiment_grid(
            zwfs_current_factory=lambda cfg=cfg: zwfs_cal_factory_from_cfg(cfg),
            zwfs_cal_factory=lambda cfg=cfg: zwfs_cal_factory_from_cfg(cfg),
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
                N_iter=300,
                N_burn=0,
                it_lag=it_lag,
                Nmodes_removed=Nmodes_removed,
                phase_scaling_factor=phase_scaling_factor,
                include_scintillation=include_scintillation,
                jumps_per_iter=jumps_per_iter,
                signal_space="pix",
                opd_threshold=np.inf,
                verbose_every=100,
            ),
        )
    )



widget_handles = bld_experiment.quicklook_experiment_results(
    results,
    labels=["internal_CPM", "internal_SFM"],  # optional, default = all
    wvl0=zwfs_ns.optics.wvl0,                  # important for Strehl proxy
    init_ho=(0, 20),                           # initial HO mode range
    init_lo=(0, 2),                            # initial LO mode range (TT)
    plot_u=True,                               # show controller states
    plot_errors=True,                          # show modal errors
    interactive=True,                          # enable sliders + toggles
    show=True,
)


"""