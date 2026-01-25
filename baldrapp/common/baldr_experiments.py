
import numpy as np
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
            **{k: v for k, v in cfg.items() if k not in {"name","ctrl_fast","ctrl_slow","loop_schedule","user_ref_intensities"}},
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

