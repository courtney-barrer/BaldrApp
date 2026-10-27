"""Headless integration harness for instrument validation.

Uses the *unchanged* production atmosphere, spectrum, Fresnel, ZWFS and
runtime helper function bodies. The instrument is a small synthetic namespace,
with zero DM and a three-mode analytic AO basis, because the live runtime has
SHM/ZMQ side effects and some hardware libraries are not available on CI hosts.

This is NOT a replacement for a live simulator/instrument initialization test.
"""
from __future__ import annotations

import ast
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from baldrapp.utils import atmosphere, pupils, source
from baldrapp.utils.configuration import resolve_config
from test_real_optics import EXAMPLE, CORE, make_instrument, load_real_optics

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'src/baldrapp/paranal_simulator/scripts/baldr_simulator.py'


def three_mode_basis(zwfs, n_modes_removed):
    """Piston, tip, tilt for controlled AO test (not the live HCIPy basis)."""
    n = int(zwfs.grid.dim)
    y, x = np.indices((n, n), dtype=float)
    x = (x - (n-1)/2)/n
    y = (y - (n-1)/2)/n
    basis = np.column_stack([np.ones(n*n), x.ravel(), y.ravel()])
    if n_modes_removed > basis.shape[1]:
        raise ValueError('Integrated headless AO supports 3 modes maximum')
    return basis


def _extract(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise RuntimeError(f'Missing production helpers: {set(names) - {n.name for n in nodes}}')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def production_helpers():
    """Load runtime functions without executing its unguarded SHM main loop."""
    core = _extract(CORE, ['select_source_profile', 'source_amplitude'],
                    {'np': np, 'SimpleNamespace': SimpleNamespace, 'source': source})
    bldr = SimpleNamespace(**core)
    names = ['get_cfg_value', 'make_source_pupil_profiles', 'apply_source_pupil',
             'apply_source_profile_fixed_grid', 'get_source_amplitude_from_config',
             'init_dynamic_atmosphere_for_beam', 'generate_atmosphere_and_source_for_beam',
             'generate_scintillation_fields_for_beam', 'runtime_dynamic_state_from_control']
    ns = {'np': np, 'deque': deque, 'atmosphere': atmosphere,
          'SimpleNamespace': SimpleNamespace, 'aperture': pupils,
          'bldr': bldr, 'make_first_stage_ao_basis': three_mode_basis}
    return _extract(RUNTIME, names, ns), load_real_optics()


def scenario_configuration(name, *, size=64, pupil_diameter=40, n_wavelengths=3):
    """Return a small, strictly configuration-resolved variant of the supplied example."""
    known = {'internal', 'onsky_clear', 'phase_only', 'phase_ao',
             'scintillation', 'scintillation_ao'}
    if name not in known:
        raise ValueError(f'Unknown scenario {name!r}')
    import json
    cfg = json.loads(EXAMPLE.read_text(encoding='utf-8'))
    cfg['grid'].update(N=pupil_diameter, dim=size)
    cfg['bandpass']['n_wavelengths'] = n_wavelengths
    cfg['atmosphere']['enabled'] = name not in ('internal', 'onsky_clear')
    cfg['atmosphere']['scintillation']['enabled'] = name in ('scintillation', 'scintillation_ao')

    cfg['first_stage_ao'].update({
        'enabled': name in ('phase_ao', 'scintillation_ao'),
        'Nmodes_removed': 3,
        'correction_gain': 1.0,
        'phase_scaling_factor': 1.0,
        'lag_frames': 0,
        'remove_piston': True,
    })


    # cfg['first_stage_ao']['enabled'] = name in ('phase_ao', 'scintillation_ao')
    # cfg['first_stage_ao']['Nmodes_removed'] = 3
    cfg['simulator_runtime']['time_step_s'] = 0.01
    return cfg


def start_scenario(name, *, fresnel_enabled=True, size=64, pupil_diameter=40,
                   n_wavelengths=3, helpers=None):
    cfg = scenario_configuration(name, size=size, pupil_diameter=pupil_diameter,
                                 n_wavelengths=n_wavelengths)
    resolved = resolve_config(cfg)
    runtime, optics = helpers if helpers is not None else production_helpers()
    inst, _ = make_instrument(cfg, n_wavelengths, size=size,
                             pupil_diameter=pupil_diameter,
                             fresnel_enabled=fresnel_enabled)
    inst.grid.D = float(cfg['grid']['D'])
    inst.grid.dim = size
    inst.grid.telescope = cfg['grid']['telescope']
    inst.resolved_config = resolved
    inst.source_profiles = cfg['source_profiles']
    inst.atmosphere = SimpleNamespace(**cfg['atmosphere'])
    inst.first_stage_ao = SimpleNamespace(**cfg['first_stage_ao'])
    inst.detector = SimpleNamespace(qe=cfg['detector']['qe'],
                                    dit=cfg['detector']['dit'],
                                    binning=cfg['detector']['binning'], ron=0.0)
    pupils_for_source = runtime['make_source_pupil_profiles'](inst)
    source_name = 'internal' if name == 'internal' else 'onsky'
    runtime['apply_source_profile_fixed_grid'](inst, source_name, cfg['source_profiles'])
    runtime['apply_source_pupil'](inst, source_name, pupils_for_source)
    amp = runtime['get_source_amplitude_from_config'](inst)
    dynamic = runtime['init_dynamic_atmosphere_for_beam'](inst)
    return SimpleNamespace(name=name, config=cfg, instrument=inst, runtime=runtime,
                           optics=optics, dynamic=dynamic, amp=amp,
                           source_profiles=pupils_for_source)


def advance_scenario(state):
    """One runtime-equivalent physics frame before SHM/cropping/background."""
    inst = state.instrument
    mode = 'internal' if state.name == 'internal' else 'onsky'
    control = {'mode': mode, 'phase_enabled': True, 'ao_enabled': True,
               'scint_enabled': True}
    dynamic = state.runtime['runtime_dynamic_state_from_control'](state.dynamic, control)
    incident_fields = None
    scintillation_index = None
    if dynamic['scint_enabled']:
        incident_fields, residual = state.runtime['generate_scintillation_fields_for_beam'](
            inst, state.amp, dynamic)
        opd = np.zeros_like(state.amp)
        weighted_irradiance = np.sum(np.abs(incident_fields)**2 *
                        inst.spectrum.photon_fractions[:, None, None], axis=0)
        entrance_rate = float(np.sum(weighted_irradiance))
        support_illuminated = state.amp > 0
        atmospheric_intensity = (weighted_irradiance[support_illuminated] /
                                  state.amp[support_illuminated]**2)
        scintillation_index = float(np.var(atmospheric_intensity) /
                                     np.mean(atmospheric_intensity)**2)
    else:
        opd, amp = state.runtime['generate_atmosphere_and_source_for_beam'](
            inst, state.amp, dynamic)
        residual = opd
        entrance_rate = float(np.sum(amp**2))
    zero = np.zeros_like(state.amp)
    kwargs = dict(opd_input=opd, amp_input=state.amp,
                  opd_internal=zero, zwfs_ns=inst, include_shotnoise=False,
                  incident_fields=incident_fields)
    # The production dispatcher falls back to zwfs_ns.detector when detector
    # is None. Temporarily remove it to obtain the true pre-detector rate.
    det = inst.detector
    del inst.detector
    try:
        pre = state.optics['get_frame_configured'](**kwargs, detector=None)
    finally:
        inst.detector = det
    detected = state.optics['get_frame_configured'](**kwargs, detector=det)
    assert np.all(np.isfinite(pre)) and np.all(np.isfinite(detected))
    support = np.asarray(inst.grid.pupil_mask) > 0
    values = residual[support]
    residual_rms_nm = float(np.std(values)*1e9) if values.size else 0.0
    return {
        'pre': pre, 'detected': detected, 'entrance_photons_per_s': entrance_rate,
        'output_photons_per_s': float(pre.sum()),
        'detected_electrons': float(detected.sum()),
        'opd_rms_nm': residual_rms_nm,
        'pupil': inst.grid.pupil_mask.copy(),
        'source': inst.source.active_profile,
        'simulation_time_s': (state.dynamic['atmosphere_model'].time_s
            if state.dynamic['atmosphere_model'] is not None else 0.0),
        'complex_fields': incident_fields,
        'scintillation_index': scintillation_index,
    }


def run_scenarios(*, n_frames=3, fresnel_enabled=True, size=64,
                  pupil_diameter=40, n_wavelengths=3):
    if n_frames <= 0:
        raise ValueError('n_frames must be positive')
    helpers = production_helpers()
    reports, snapshots = {}, {}
    for name in ('internal', 'onsky_clear', 'phase_only', 'phase_ao',
                 'scintillation', 'scintillation_ao'):
        state = start_scenario(name, helpers=helpers, fresnel_enabled=fresnel_enabled,
                               size=size, pupil_diameter=pupil_diameter,
                               n_wavelengths=n_wavelengths)
        frames = [advance_scenario(state) for _ in range(n_frames)]
        photon_rate = float(state.instrument.source.photons_per_second)
        detectors = np.stack([f['detected'] for f in frames])
        pre = np.stack([f['pre'] for f in frames])
        entrance = np.asarray([f['entrance_photons_per_s'] for f in frames])
        output = np.asarray([f['output_photons_per_s'] for f in frames])
        detected = np.asarray([f['detected_electrons'] for f in frames])
        expected = output * state.instrument.detector.dit * state.instrument.detector.qe
        if not np.allclose(detected, expected, rtol=2e-12, atol=1e-5):
            raise AssertionError(f'{name}: detector photon accounting mismatch')
        if not np.all(np.isfinite(detectors)) or np.any(detectors < -1e-9):
            raise AssertionError(f'{name}: invalid detector frame')
        reports[name] = {
            'source_profile': frames[0]['source'],
            'source_photons_per_s': photon_rate,
            'entrance_photons_per_s': entrance.tolist(),
            'optical_output_photons_per_s': output.tolist(),
            'optical_throughput': (output/entrance).tolist(),
            'detected_electrons': detected.tolist(),
            'residual_opd_rms_nm': [f['opd_rms_nm'] for f in frames],
            'detector_image_temporal_rms_electrons': float(
                np.sqrt(np.mean(np.var(detectors, axis=0)))),
            'time_last_s': frames[-1]['simulation_time_s'],
            'pupil_illuminated_pixels': int(np.count_nonzero(frames[0]['pupil'])),
            'pupil_scintillation_index': [f['scintillation_index'] for f in frames],
        }
        snapshots[name] = {'first': detectors[0], 'last': detectors[-1],
                           'pupil': frames[0]['pupil'],
                           'pre': pre[0]}
    return {'method': 'headless integration using production numerical functions and synthetic instrument',
            'fresnel_enabled': fresnel_enabled, 'n_frames': n_frames,
            'wavelength_samples': n_wavelengths, 'grid_dim': size,
            'pupil_diameter_pixels': pupil_diameter,
            'scenarios': reports}, snapshots
