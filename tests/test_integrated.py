"""Stage 5 end-to-end headless physics integration and scenario validation.

Not a live SHM/camera-server test and not a test of HCIPy DM initialization.
"""
import unittest
import numpy as np
from integrated_support import (start_scenario, advance_scenario,
                                   run_scenarios, production_helpers)


class IntegratedV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = production_helpers()

    def test_scenarios_finite_and_detector_accounting(self):
        report, _ = run_scenarios(n_frames=2, fresnel_enabled=False,
                                  n_wavelengths=3)
        for name, scenario in report['scenarios'].items():
            with self.subTest(name=name):
                self.assertEqual(len(scenario['detected_electrons']), 2)
                self.assertTrue(np.all(np.isfinite(scenario['detected_electrons'])))
                self.assertTrue(np.all(np.asarray(scenario['optical_throughput']) > 0))

    def test_source_switch_preserves_wavelengths_and_photon_rate(self):
        state = start_scenario('internal', helpers=self.helpers, fresnel_enabled=False)
        inst = state.instrument
        runtime = state.runtime
        grid_before = inst.spectrum.wavelengths.copy()
        rate_before = float(np.sum(state.amp**2))
        self.assertAlmostEqual(rate_before, inst.source.photons_per_second, delta=.001)
        internal_mask = inst.grid.pupil_mask.copy()
        runtime['apply_source_profile_fixed_grid'](inst, 'onsky', inst.source_profiles)
        runtime['apply_source_pupil'](inst, 'onsky', state.source_profiles)
        amp = runtime['get_source_amplitude_from_config'](inst)
        self.assertFalse(np.array_equal(inst.grid.pupil_mask, internal_mask))
        self.assertAlmostEqual(float(np.sum(amp**2)), inst.source.photons_per_second, delta=.001)
        np.testing.assert_array_equal(inst.spectrum.wavelengths, grid_before)
        runtime['apply_source_profile_fixed_grid'](inst, 'internal', inst.source_profiles)
        runtime['apply_source_pupil'](inst, 'internal', state.source_profiles)
        np.testing.assert_array_equal(inst.grid.pupil_mask, internal_mask)
        np.testing.assert_array_equal(inst.spectrum.wavelengths, grid_before)

    def test_phase_only_determinism_and_evolution(self):
        a = start_scenario('phase_only', helpers=self.helpers, fresnel_enabled=False)
        b = start_scenario('phase_only', helpers=self.helpers, fresnel_enabled=False)
        observed = []
        for _ in range(4):
            x, y = advance_scenario(a), advance_scenario(b)
            np.testing.assert_allclose(x['pre'], y['pre'], rtol=0, atol=1e-8)
            observed.append(x['pre'])
        self.assertGreater(np.linalg.norm(observed[-1]-observed[0]), 1)
        self.assertAlmostEqual(a.dynamic['v2_atmosphere'].time_s, .04)

    def test_phase_only_ao_reduces_modal_residual(self):
        raw = start_scenario('phase_only', helpers=self.helpers, fresnel_enabled=False)
        corrected = start_scenario('phase_ao', helpers=self.helpers, fresnel_enabled=False)
        x, y = advance_scenario(raw), advance_scenario(corrected)
        self.assertLess(y['opd_rms_nm'], x['opd_rms_nm'])
        self.assertEqual(x['source'], 'onsky')
        self.assertEqual(y['source'], 'onsky')

    def test_scintillation_is_chromatic_and_changes_entrance_rate(self):
        state = start_scenario('scintillation', helpers=self.helpers,
                               fresnel_enabled=False)
        out = advance_scenario(state)
        self.assertIsNotNone(out['complex_fields'])
        self.assertEqual(out['complex_fields'].shape[0], 3)
        mask = out['pupil'] > 0
        amps = np.abs(out['complex_fields'])[:, mask]
        self.assertGreater(float(np.std(amps[0])), 0)
        self.assertGreater(float(np.linalg.norm(amps[0]-amps[-1])), 1e-4)
        # No normalization to a fixed pupil-intercepted flux after scintillation.
        self.assertGreater(abs(out['entrance_photons_per_s'] - state.instrument.source.photons_per_second), 1)

    def test_fresnel_ao_and_scintillation_integration(self):
        state = start_scenario('scintillation_ao', helpers=self.helpers,
                               fresnel_enabled=True)
        x = advance_scenario(state)
        self.assertTrue(np.all(np.isfinite(x['detected'])))
        self.assertGreater(x['detected'].sum(), 0)
        expected = x['pre'].sum()*state.instrument.detector.qe*state.instrument.detector.dit
        self.assertAlmostEqual(x['detected'].sum(), expected, delta=.01)

    def test_scintillation_ao_reduces_residual_without_renormalizing_flux(self):
        raw = start_scenario('scintillation', helpers=self.helpers,
                             fresnel_enabled=False)
        corrected = start_scenario('scintillation_ao', helpers=self.helpers,
                                   fresnel_enabled=False)
        x, y = advance_scenario(raw), advance_scenario(corrected)
        self.assertLess(y['opd_rms_nm'], x['opd_rms_nm'])
        # AO modifies only pupil phase, not atmospheric amplitude or source flux.
        self.assertAlmostEqual(x['entrance_photons_per_s'],
                               y['entrance_photons_per_s'], delta=.001)
        np.testing.assert_allclose(np.abs(x['complex_fields']),
                                   np.abs(y['complex_fields']), rtol=1e-12, atol=1e-10)

    def test_internal_and_onsky_clear_have_distinct_pupils_and_frames(self):
        internal = start_scenario('internal', helpers=self.helpers,
                                  fresnel_enabled=False)
        onsky = start_scenario('onsky_clear', helpers=self.helpers,
                               fresnel_enabled=False)
        x, y = advance_scenario(internal), advance_scenario(onsky)
        self.assertFalse(np.array_equal(x['pupil'], y['pupil']))
        self.assertGreater(float(np.linalg.norm(x['detected'] - y['detected'])), 1)
        self.assertAlmostEqual(x['simulation_time_s'], 0.0)
        self.assertAlmostEqual(y['simulation_time_s'], 0.0)


if __name__ == '__main__':
    unittest.main()
