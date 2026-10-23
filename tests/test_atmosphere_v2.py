"""Stage 3: physical multilayer OPD and v2 runtime integration tests.

No hardware, SHM or HCIPy dependency: the runtime functions are extracted from
production source via AST, as in the Stage-2 wrapper tests.
"""
import ast
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from baldrapp.utils.atmosphere_v2 import (
    FrozenFlowLayer, MultilayerAtmosphere, LAMBDA_REF_M, modal_ao_residual_opd,
)
from baldrapp.utils.configuration import resolve_config
from baldrapp.utils import atmosphere_v2

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'src/baldrapp/paranal_simulator/fake_configs/baldr_config_v2.json'
RUNTIME = ROOT / 'src/baldrapp/paranal_simulator/scripts/baldr_sim_updated_runtime_control.py'


def real_runtime_functions():
    funcs = {'init_dynamic_atmosphere_for_beam', 'generate_atmosphere_and_source_for_beam'}
    tree = ast.parse(RUNTIME.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in funcs]
    assert len(nodes) == 2
    namespace = {
        'np': np, 'deque': deque, 'atmosphere_v2': atmosphere_v2,
        'get_cfg_value': lambda obj, key, default=None: (obj.get(key, default)
                                if isinstance(obj, dict) else getattr(obj, key, default)),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(RUNTIME), 'exec'), namespace)
    return namespace


class MultilayerV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(EXAMPLE.read_text(encoding='utf-8'))

    @staticmethod
    def fast_report(config):
        cfg = deepcopy(config)
        cfg['grid']['dim'] = 16
        cfg['grid']['N'] = 8
        return resolve_config(cfg)

    def test_validate_modal_ao_inputs(self):
        for key, value in [('lag_frames', -1), ('lag_frames', 0.2),
                           ('correction_gain', 1.5), ('remove_piston', 1),
                           ('basis', 'fourier')]:
            cfg = deepcopy(self.config)
            cfg['first_stage_ao'][key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                resolve_config(cfg)

    def test_opd_variance_scales_with_cn2(self):
        # With identical random draws, PSD variance scales exactly as
        # r0**(-5/3). The short periodic grid affects absolute variance,
        # but should not change this amplitude law.
        from baldrapp.utils.phasescreens import ft_phase_screen
        a = ft_phase_screen(r0=0.1, N=64, delta=0.02, L0=25., l0=1e-10, seed=17)
        b = ft_phase_screen(r0=0.2, N=64, delta=0.02, L0=25., l0=1e-10, seed=17)
        self.assertAlmostEqual(np.var(a)/np.var(b), 2.**(5./3.), places=12)

    def test_seed_reproducibility_and_independent_layers(self):
        report = self.fast_report(self.config)
        a = MultilayerAtmosphere.from_resolved_config(report, grid_dim=16,
                                                      pupil_pixel_scale_m=1.8/8)
        b = MultilayerAtmosphere.from_resolved_config(report, grid_dim=16,
                                                      pupil_pixel_scale_m=1.8/8)
        for l1, l2 in zip(a.layers, b.layers):
            np.testing.assert_array_equal(l1.phase_tile_rad, l2.phase_tile_rad)
        self.assertFalse(np.array_equal(a.layers[0].phase_tile_rad, a.layers[1].phase_tile_rad))
        for _ in range(3):
            np.testing.assert_array_equal(a.next_opd(), b.next_opd())
        self.assertAlmostEqual(a.time_s, 3 * report['configuration']['simulator_runtime']['time_step_s'])

    def test_screen_r0_cn2_fractions_and_altitudes(self):
        r = self.fast_report(self.config)
        atmo = MultilayerAtmosphere.from_resolved_config(r, grid_dim=16,
                            pupil_pixel_scale_m=1.8/8)
        self.assertEqual([l.altitude_m for l in atmo.layers], [0.0, 10000.0])
        total = sum((0.126/l.r0_500nm_m)**(5/3) for l in atmo.layers)
        self.assertAlmostEqual(total, 1.0)
        self.assertAlmostEqual(atmo.layers[0].cn2_fraction, 0.7)

    def test_zero_strength_layer_contributes_exactly_zero(self):
        cfg = deepcopy(self.config)
        cfg['atmosphere']['layers'][0]['cn2_fraction'] = 1.0
        cfg['atmosphere']['layers'][1]['cn2_fraction'] = 0.0
        r = self.fast_report(cfg)
        atm = MultilayerAtmosphere.from_resolved_config(r, grid_dim=16,
                          pupil_pixel_scale_m=1.8/8)
        np.testing.assert_array_equal(atm.layers[1].sample_opd(0.1), np.zeros((16, 16)))
        np.testing.assert_allclose(atm.opd_at(0.21), atm.layers[0].sample_opd(0.21))

    def test_positive_wind_moves_pattern_in_increasing_x_y(self):
        n = 16
        y, x = np.mgrid[:n, :n]
        tile = np.sin(2*np.pi*x/n) + 0.2*np.cos(2*np.pi*y/n)
        base = dict(altitude_m=10000, cn2_fraction=1, r0_500nm_m=0.126,
                    pixel_scale_m=0.25, size=8, phase_tile_rad=tile)
        layer_x = FrozenFlowLayer(**base, wind_x_m_per_s=0.25, wind_y_m_per_s=0)
        layer_y = FrozenFlowLayer(**base, wind_x_m_per_s=0, wind_y_m_per_s=0.25)
        origin = (n-8)//2
        factor = LAMBDA_REF_M/(2*np.pi)
        expected_x = tile[origin:origin+8, origin-1:origin+7] * factor
        expected_y = tile[origin-1:origin+7, origin:origin+8] * factor
        np.testing.assert_allclose(layer_x.sample_opd(1), expected_x, atol=1e-19)
        np.testing.assert_allclose(layer_y.sample_opd(1), expected_y, atol=1e-19)
        np.testing.assert_allclose(layer_x.sample_opd(0), layer_y.sample_opd(0))

    def test_subpixel_motion_and_zero_wind(self):
        n = 16
        x = np.broadcast_to(np.arange(n)[None, :], (n, n))
        base = dict(altitude_m=0, cn2_fraction=1, r0_500nm_m=0.126,
                    pixel_scale_m=1.0, size=8, phase_tile_rad=x.astype(float))
        half = FrozenFlowLayer(**base, wind_x_m_per_s=0.5, wind_y_m_per_s=0)
        zero = FrozenFlowLayer(**base, wind_x_m_per_s=0, wind_y_m_per_s=0)
        np.testing.assert_allclose(half.sample_opd(1),
                                   0.5*(half.sample_opd(0)+half.sample_opd(2)))
        np.testing.assert_array_equal(zero.sample_opd(0), zero.sample_opd(1e4))

    def test_additivity_opd_units_and_wavelength_invariance(self):
        r = self.fast_report(self.config)
        atm = MultilayerAtmosphere.from_resolved_config(r, grid_dim=16,
                          pupil_pixel_scale_m=1.8/8)
        summed = sum((layer.sample_opd(0.2) for layer in atm.layers))
        np.testing.assert_allclose(atm.opd_at(0.2), summed)
        phi_500 = 2*np.pi*atm.opd_at(0.2)/(500e-9)
        phi_1000 = 2*np.pi*atm.opd_at(0.2)/(1000e-9)
        np.testing.assert_allclose(phi_500, 2*phi_1000, atol=1e-12)

    def test_modal_ao_gain_piston_and_lag(self):
        n = 8
        x = np.broadcast_to(np.linspace(-1, 1, n)[None, :], (n, n))
        pupil = np.ones((n,n))
        modes = np.stack([np.ones((n,n)), x], axis=-1).reshape((n*n,2))
        opd = 60e-9 + 20e-9*x
        np.testing.assert_allclose(modal_ao_residual_opd(opd,modes,pupil,2), 0, atol=1e-20)
        np.testing.assert_allclose(modal_ao_residual_opd(opd,modes,pupil,2,correction_gain=0.5),
                                   opd*0.5, atol=1e-20)
        np.testing.assert_allclose(modal_ao_residual_opd(opd,modes,pupil,2,remove_piston=False),
                                   np.full_like(opd,60e-9), atol=1e-20)
        history=deque()
        initial=modal_ao_residual_opd(opd,modes,pupil,2,lag_frames=1,history=history)
        np.testing.assert_allclose(initial,opd)
        delayed=modal_ao_residual_opd(2*opd,modes,pupil,2,lag_frames=1,history=history)
        np.testing.assert_allclose(delayed,opd,atol=1e-20)

    def test_v2_runtime_initialization_and_frame_phase(self):
        ns = real_runtime_functions()
        cfg=deepcopy(self.config)
        cfg['grid']['dim']=16
        cfg['grid']['N']=8
        cfg['first_stage_ao']['enabled']=False
        r=resolve_config(cfg)
        zwfs=SimpleNamespace(
            config_schema_version=2, resolved_config=r,
            optics=SimpleNamespace(wvl0=cfg['optics']['wvl0']),
            grid=SimpleNamespace(D=1.8,N=8,dim=16,pupil_mask=np.ones((16,16))),
            atmosphere=SimpleNamespace(**cfg['atmosphere']),
            first_stage_ao=SimpleNamespace(**cfg['first_stage_ao']),
        )
        state=ns['init_dynamic_atmosphere_for_beam'](zwfs)
        self.assertTrue(state['phase_enabled'])
        self.assertFalse(state['scint_enabled'])
        amp=np.ones((16,16))
        first, outamp=ns['generate_atmosphere_and_source_for_beam'](zwfs,amp,state)
        np.testing.assert_array_equal(outamp,amp)
        self.assertEqual(first.shape,(16,16))
        self.assertTrue(np.any(first != 0))
        self.assertAlmostEqual(state['v2_atmosphere'].time_s,0.001)
        second,_=ns['generate_atmosphere_and_source_for_beam'](zwfs,amp,state)
        self.assertFalse(np.array_equal(first,second))
        state['phase_enabled']=False
        zero,_=ns['generate_atmosphere_and_source_for_beam'](zwfs,amp,state)
        np.testing.assert_array_equal(zero,np.zeros((16,16)))
        self.assertAlmostEqual(state['v2_atmosphere'].time_s,0.003)

    def test_runtime_modal_ao_with_one_frame_lag(self):
        ns = real_runtime_functions()
        cfg = deepcopy(self.config)
        cfg['grid']['dim'] = 16
        cfg['grid']['N'] = 8
        cfg['first_stage_ao']['Nmodes_removed'] = 2
        cfg['first_stage_ao']['lag_frames'] = 1
        r = resolve_config(cfg)
        yy, xx = np.mgrid[:16, :16]
        modes = np.stack((np.ones((16, 16)), xx.astype(float)), axis=-1).reshape(16*16, 2)
        ns['make_first_stage_ao_basis'] = lambda zwfs, count: modes
        zwfs = SimpleNamespace(
            config_schema_version=2, resolved_config=r,
            optics=SimpleNamespace(wvl0=cfg['optics']['wvl0']),
            grid=SimpleNamespace(D=1.8,N=8,dim=16,pupil_mask=np.ones((16,16))),
            atmosphere=SimpleNamespace(**cfg['atmosphere']),
            first_stage_ao=SimpleNamespace(**cfg['first_stage_ao']),
        )
        state = ns['init_dynamic_atmosphere_for_beam'](zwfs)
        self.assertTrue(state['ao_enabled'])
        raw0 = state['v2_atmosphere'].opd_at(0)
        raw1 = state['v2_atmosphere'].opd_at(0.001)
        amp = np.ones((16,16))
        first, _ = ns['generate_atmosphere_and_source_for_beam'](zwfs, amp, state)
        np.testing.assert_allclose(first, raw0)
        self.assertEqual(len(state['v2_ao_history']), 1)
        second, _ = ns['generate_atmosphere_and_source_for_beam'](zwfs, amp, state)
        projection = raw1 - second
        # Fitted delayed correction should be exactly representable by the
        # configured piston and x-tilt modes.
        np.testing.assert_allclose(
            projection,
            raw0 - modal_ao_residual_opd(raw0, modes, zwfs.grid.pupil_mask, 2),
            atol=1e-19,
        )
        state['phase_enabled'] = False
        ns['generate_atmosphere_and_source_for_beam'](zwfs, amp, state)
        self.assertEqual(len(state['v2_ao_history']), 0)

    def test_scintillation_explicitly_rejected_until_stage4(self):
        cfg = deepcopy(self.config)
        cfg['atmosphere']['scintillation']['enabled'] = True
        report = self.fast_report(cfg)
        with self.assertRaisesRegex(NotImplementedError,'Stage 4'):
            MultilayerAtmosphere.from_resolved_config(report, grid_dim=16,
                          pupil_pixel_scale_m=1.8/8)

    def test_disabled_atmosphere_keeps_amplitude_and_zero_opd(self):
        ns=real_runtime_functions()
        cfg=deepcopy(self.config)
        cfg['atmosphere']['enabled']=False
        report=self.fast_report(cfg)
        zwfs=SimpleNamespace(
            config_schema_version=2, resolved_config=report,
            optics=SimpleNamespace(wvl0=1.65e-6),
            grid=SimpleNamespace(D=1.8,N=8,dim=16,pupil_mask=np.ones((16,16))),
            atmosphere=SimpleNamespace(**cfg['atmosphere']),
            first_stage_ao=SimpleNamespace(**cfg['first_stage_ao']),
        )
        state=ns['init_dynamic_atmosphere_for_beam'](zwfs)
        amp=np.ones((16,16))*23
        opd, outamp=ns['generate_atmosphere_and_source_for_beam'](zwfs,amp,state)
        np.testing.assert_array_equal(opd,np.zeros((16,16)))
        self.assertIs(outamp,amp)


if __name__=='__main__':
    unittest.main()
