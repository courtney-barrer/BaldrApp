"""Stage 4: actual Fresnel atmospheric propagation and real ZWFS dispatch.

No SHM or GUI required. Phase-screen generation is the unchanged original
PhaseScreenKolmogorov implementation. Optical propagation and frame generation
call production functions, rather than numerical mocks.
"""
import ast
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from baldrapp.utils import atmosphere, source
from baldrapp.utils.configuration import resolve_config
from test_real_optics import load_real_optics, make_instrument

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'src/baldrapp/paranal_simulator/scripts/baldr_simulator.py'
REFERENCE = ROOT / 'tests/fixtures/baldr_config_reference.json'


def load_runtime_scintillation():
    names = {'init_dynamic_atmosphere_for_beam',
             'generate_scintillation_fields_for_beam'}
    tree = ast.parse(RUNTIME.read_text(encoding='utf-8'))
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in defs} == names
    ns = {'np': np, 'deque': deque, 'atmosphere': atmosphere,
          'get_cfg_value': lambda obj, key, default=None: (
              obj.get(key, default) if isinstance(obj, dict)
              else getattr(obj, key, default))}
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(RUNTIME), 'exec'), ns)
    return ns


class ScintillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads(REFERENCE.read_text(encoding='utf-8'))
        cls.core = load_real_optics()

    def test_zero_turbulence_and_energy_conservation(self):
        p = atmosphere.propagate_atmospheric_layers
        z = np.zeros((32, 32))
        for heights in ([10000.0], [10000., 5000., 0]):
            fields = p([z for _ in heights], heights, [1.5e-6, 1.8e-6], .02)
            np.testing.assert_allclose(fields, 1, atol=2e-12)
            self.assertAlmostEqual(np.sum(np.abs(fields[0])**2), 32**2)

    def test_zero_height_exactly_reproduces_phase_sum(self):
        p = atmosphere.propagate_atmospheric_layers
        x = np.arange(32)[None, :]
        z0 = np.broadcast_to(50e-9*np.sin(x/6), (32, 32))
        z1 = np.broadcast_to(40e-9*np.cos(x/8), (32, 32))
        for wavelength in (1.5e-6, 1.8e-6):
            observed = p([z0, z1], [0., 0.], [wavelength], .02)[0]
            np.testing.assert_allclose(observed, np.exp(2j*np.pi*(z0+z1)/wavelength), atol=1e-13)

    def test_high_altitude_scintillation_is_chromatic_and_correlated(self):
        p = atmosphere.propagate_atmospheric_layers
        n = 64
        x = np.arange(n)[None, :]
        z = np.broadcast_to(150e-9*np.cos(2*np.pi*x/8), (n, n))
        out = p([z], [12000], [1.5e-6, 1.8e-6], .025)
        intensity = np.abs(out)**2
        self.assertGreater(np.std(intensity[0]), 1e-3)
        self.assertGreater(np.linalg.norm(intensity[0]-intensity[1]), 1e-3)
        for field in out:
            self.assertAlmostEqual(np.sum(np.abs(field)**2)/n**2, 1.0, places=12)
        # Ground-layer-only turbulence is pure phase, no scintillation.
        ground = p([z], [0], [1.5e-6], .025)[0]
        np.testing.assert_allclose(np.abs(ground), 1, atol=1e-12)

    def test_original_infinite_screens_sample_entire_propagation_plane(self):
        from baldrapp.utils.phasescreens import PhaseScreenKolmogorov
        cfg = deepcopy(self.cfg)
        cfg['grid']['dim'] = 24
        cfg['grid']['N'] = 10
        cfg['atmosphere']['scintillation']['enabled'] = True
        cfg['atmosphere']['layers'][0]['wind_direction_deg'] = 45.0
        model = atmosphere.MultilayerAtmosphere.from_resolved_config(
            resolve_config(cfg), grid_dim=24, pupil_pixel_scale_m=.225)
        for layer in model.layers:
            self.assertIsInstance(layer.phase_screen, PhaseScreenKolmogorov)
            self.assertTrue(layer.full_grid)
            self.assertGreater(layer.phase_screen.scrn.shape[0], 24)
        opds = model.next_layer_opds()
        self.assertTrue(all(np.any(opd[:2, :2] != 0) for opd in opds))
        self.assertAlmostEqual(model.time_s, cfg['simulator_runtime']['time_step_s'])

    def test_real_zwfs_fresnel_and_nonfresnel_accept_complex_fields(self):
        for fresnel_enabled in (False, True):
            inst, amp = make_instrument(self.cfg, 3, fresnel_enabled=fresnel_enabled)
            wavelength = inst.spectrum.wavelengths
            n = amp.shape[0]
            y, x = np.indices(amp.shape)
            opd = 60e-9*np.cos(2*np.pi*x/13)*np.sin(2*np.pi*y/17)
            fields = np.asarray([amp * np.exp(2j*np.pi*opd/w) for w in wavelength])
            zeros = np.zeros_like(amp)
            args = dict(opd_input=zeros, amp_input=amp, opd_internal=zeros,
                        zwfs_ns=inst, detector=None, include_shotnoise=False)
            reference = self.core['get_frame_configured'](**{**args, 'opd_input':opd})
            actual = self.core['get_frame_configured'](**args, incident_fields=fields)
            np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-5)
            det = SimpleNamespace(binning=4, qe=.7, dit=.001, ron=0)
            counted = self.core['get_frame_configured'](
                **{**args,'detector':det}, incident_fields=fields)
            self.assertAlmostEqual(counted.sum(), actual.sum()*.7*.001, delta=.005)
            with self.assertRaisesRegex(ValueError, 'shape'):
                self.core['get_frame_configured'](**args,incident_fields=fields[:1])

    def test_modal_ao_removes_opd_at_all_wavelengths(self):
        n = 16
        xx = np.broadcast_to((np.arange(n)-(n-1)/2)[None, :], (n,n))
        opd = 30e-9 + 50e-9 * xx/n
        wavelengths = np.array([1.5e-6, 1.8e-6])
        atmospheric_fields = np.asarray([
            np.exp(2j*np.pi*opd/w) for w in wavelengths
        ])
        class StaticAtmosphere:
            def next_fields(self, requested):
                np.testing.assert_array_equal(requested, wavelengths)
                return atmospheric_fields.copy()
        pupil = np.ones((n,n))
        amp = source.source_input_amplitude(pupil, 1e5)
        basis = np.column_stack([np.ones(n*n), xx.ravel()])
        cfg = dict(scint_enabled=True, atmosphere_model=StaticAtmosphere(),
                   phase_enabled=True, ao_enabled=True, ao_basis=basis,
                   n_modes_removed=2, ao_phase_scaling_factor=1.,
                   ao_correction_gain=1., ao_remove_piston=True,
                   ao_lag_frames=0, ao_history=deque())
        zwfs = SimpleNamespace(grid=SimpleNamespace(pupil_mask=pupil),
             optics=SimpleNamespace(wvl0=1.5e-6),
             spectrum=SimpleNamespace(wavelengths=wavelengths))
        runtime = load_runtime_scintillation()
        corrected, diag = runtime['generate_scintillation_fields_for_beam'](zwfs,amp,cfg)
        np.testing.assert_allclose(corrected, np.broadcast_to(amp,corrected.shape), atol=1e-9, rtol=1e-12)
        self.assertLess(np.max(np.abs(diag)),1e-19)
        cfg['ao_enabled'] = False
        raw, diag = runtime['generate_scintillation_fields_for_beam'](zwfs,amp,cfg)
        self.assertGreater(np.max(np.abs(np.angle(raw[0]))), 0.05)

    def test_dynamic_scintillation_and_source_pupil_normalization(self):
        cfg = deepcopy(self.cfg)
        cfg['grid']['N'] = 8
        cfg['grid']['dim'] = 16
        cfg['atmosphere']['scintillation']['enabled'] = True
        cfg['first_stage_ao']['enabled'] = False
        resolved = resolve_config(cfg)
        pupil = np.zeros((16,16))
        pupil[4:12, 4:12] = 1.0
        amp = source.source_input_amplitude(pupil, 2e6)
        ns = load_runtime_scintillation()
        zwfs = SimpleNamespace(
            resolved_config=resolved,
            grid=SimpleNamespace(D=1.8, N=8, dim=16, pupil_mask=pupil),
            optics=SimpleNamespace(wvl0=1.65e-6),
            spectrum=source.spectrum_for_profile(resolved, 'onsky'),
            first_stage_ao=SimpleNamespace(**cfg['first_stage_ao']),
            atmosphere=SimpleNamespace(**cfg['atmosphere']),
        )
        state = ns['init_dynamic_atmosphere_for_beam'](zwfs)
        self.assertTrue(state['scint_enabled'])
        fields, diag = ns['generate_scintillation_fields_for_beam'](zwfs,amp,state)
        self.assertEqual(fields.shape, (7,16,16))
        self.assertTrue(np.all(fields[:,pupil==0] == 0))
        self.assertEqual(diag.shape, (16,16))
        self.assertAlmostEqual(state['atmosphere_model'].time_s, .001)
        self.assertTrue(np.any(np.abs(fields[0,pupil>0])-amp[pupil>0] != 0))
        # Scintillation changes aperture throughput, which must NOT be
        # renormalized to the constant source rate each frame.
        self.assertFalse(np.allclose(np.sum(np.abs(fields)**2,axis=(1,2)), 2e6, rtol=1e-7))

if __name__ == '__main__':
    unittest.main()
