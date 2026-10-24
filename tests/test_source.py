"""Source photon budget / routing tests without hardware-only dependencies.

The optical core currently imports astropy and other instrument packages not
available in every CI environment. For integration tests we extract the actual
BaldrApp wrapper function bodies from source and exercise them against minimal,
deterministic optical-propagation/detector fakes. This tests dispatch, weights,
normalisation and single detection, not physical Fresnel correctness.
"""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from baldrapp.utils.configuration import resolve_config
from baldrapp.utils import source

CORE = Path(__file__).resolve().parents[1] / 'src/baldrapp/utils/baldr_core.py'
EXAMPLE = Path(__file__).resolve().parents[1] / 'src/baldrapp/paranal_simulator/fake_configs/baldr_config.json'


def load_core_functions(names, namespace):
    """Compile specific *actual* core function definitions with mocked optics."""
    tree = ast.parse(CORE.read_text(encoding='utf-8'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(functions) == len(names), (names, [n.name for n in functions])
    code = compile(ast.Module(body=functions, type_ignores=[]), str(CORE), 'exec')
    exec(code, namespace)
    return namespace


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(EXAMPLE.read_text())
        cls.resolved = resolve_config(cls.config)

    def test_photon_rate_independent_of_pupil_and_grid(self):
        rate = 4.5e7
        for n in (16, 37, 128):
            pupil = np.ones((n, n))
            pupil[:n//3, :n//2] = 0
            pupil[2*n//3:, :n//4] = 0.37  # apodization
            amplitude = source.source_input_amplitude(pupil, rate)
            self.assertAlmostEqual(float(np.sum(amplitude**2)) / rate, 1, places=13)
            self.assertTrue(np.all(amplitude[pupil == 0] == 0))

    def test_zero_flux_and_invalid_pupil(self):
        self.assertEqual(np.sum(source.source_input_amplitude(np.ones((4, 4)), 0)), 0)
        for pupil in (np.zeros((4, 4)), np.full((2, 2), np.nan), np.ones((2, 2))*-1):
            with self.assertRaises(ValueError):
                source.source_input_amplitude(pupil, 100)
        with self.assertRaises(ValueError):
            source.source_input_amplitude(np.ones((4, 4)), -1)

    def test_shared_wavelengths_independent_weights(self):
        internal = source.spectrum_for_profile(self.resolved, 'internal')
        onsky = source.spectrum_for_profile(self.resolved, 'onsky')
        np.testing.assert_array_equal(internal.wavelengths, onsky.wavelengths)
        self.assertFalse(np.allclose(internal.photon_fractions, onsky.photon_fractions))
        self.assertAlmostEqual(internal.photon_fractions.sum(), 1)
        self.assertAlmostEqual(onsky.photon_fractions.sum(), 1)
        self.assertEqual(source.integration_weights(internal)[1], 'photon_fraction')

    def test_sample_count_changes_no_band_integrated_flux(self):
        for n in (1, 2, 3, 7, 17, 101):
            cfg = deepcopy(self.config)
            cfg['bandpass']['n_wavelengths'] = n
            report = resolve_config(cfg)
            for name in ('internal', 'onsky'):
                source = report['derived']['source_profiles'][name]
                self.assertAlmostEqual(sum(source['photons_per_second_by_wavelength']), 1e9, delta=1e-5)

    def test_unnormalized_weights_are_rejected(self):
        invalid = SimpleNamespace(wavelengths=np.array([1., 2.]),
                                  photon_fractions=np.array([0.2, 0.2]))
        with self.assertRaisesRegex(ValueError, "sum to one"):
            source.integration_weights(invalid)

    def test_frame_dispatch_both_optical_paths_and_detect_once(self):
        calls = []
        # Each monochromatic optical fake is linear in photon rate. This
        # isolates the spectral quadrature independently of ZWFS optics.
        def fake_get_frame(opd_input, amp_input, opd_internal, zwfs_ns, detector, include_shotnoise):
            self.assertIsNone(detector)
            return amp_input**2 * (zwfs_ns.optics.wvl0 / 1.65e-6)

        def fake_fresnel(opd_input, amp_input, opd_internal, zwfs_ns, detector,
                         include_shotnoise, spectral_bandwidth, wavelength, theta,
                         phasemask_diameter, return_intermediates):
            self.assertIsNone(detector)
            self.assertIsNone(spectral_bandwidth)
            data = amp_input**2 * (wavelength / 1.65e-6)
            return {'intensity_pre_detector': data} if return_intermediates else data

        def fake_detect(image, *, binning, qe, dit, ron, include_shotnoise, spectral_bandwidth):
            self.assertIsNone(spectral_bandwidth)
            calls.append('detect')
            return image * qe * dit

        class FakeSpec:
            @staticmethod
            def theta_at_wavelength(optics, wavelength):
                return optics.theta
            @staticmethod
            def phasemask_diameter_at_wavelength(optics, wavelength_m, default_wvl0):
                return optics.mask_diam

        ns = load_core_functions(
            {'get_frame_polychromatic', 'get_frame_fresnel_polychromatic', 'get_frame_configured'},
            {'np': np, 'source': source, 'spec': FakeSpec,
             'get_frame': fake_get_frame, 'get_frame_fresnel': fake_fresnel,
             'detect': fake_detect, '_AUTO_SPECTRAL_BANDWIDTH': 'auto'},
        )
        detector = SimpleNamespace(binning=1, qe=0.7, dit=0.002, ron=0.0)
        amplitude = source.source_input_amplitude(np.ones((5, 5)), 2e6)
        for n in (1, 7):
            cfg = deepcopy(self.config)
            cfg['bandpass']['n_wavelengths'] = n
            report = resolve_config(cfg)
            spectrum = source.spectrum_for_profile(report, 'internal')
            expected = float(np.dot(spectrum.photon_fractions, spectrum.wavelengths / 1.65e-6))
            for use_fresnel in (False, True):
                instrument = SimpleNamespace(
                                        spectrum=spectrum,
                    optics=SimpleNamespace(wvl0=1.65e-6, theta=1.57, mask_diam=1.06),
                    fresnel_relay=SimpleNamespace(enabled=use_fresnel),
                )
                # Force the driver to route single-node through the spectral wrapper.
                args = dict(opd_input=np.zeros_like(amplitude), amp_input=amplitude,
                            opd_internal=np.zeros_like(amplitude), zwfs_ns=instrument,
                            detector=detector, include_shotnoise=False)
                before = len(calls)
                result = ns['get_frame_configured'](**args)
                self.assertEqual(len(calls) - before, 1)
                self.assertAlmostEqual(float(result.sum()) / (2e6 * 0.7 * 0.002),
                                       expected, places=12)
                self.assertAlmostEqual(instrument.optics.wvl0, 1.65e-6)

    def test_runtime_profile_selection_retains_instrument_grid(self):
        def fake_pupil():
            return np.ones((4, 4))
        ns = load_core_functions({'select_source_profile', 'source_amplitude'},
                                 {'SimpleNamespace': SimpleNamespace, 'source': source})
        inst = SimpleNamespace(resolved_config=self.resolved,
                               grid=SimpleNamespace(pupil_mask=fake_pupil()))
        inst = ns['select_source_profile'](inst, 'internal')
        waves = inst.spectrum.wavelengths.copy()
        first = ns['source_amplitude'](inst)
        inst.grid.pupil_mask[:2, :] = 0
        inst = ns['select_source_profile'](inst, 'onsky')
        second = ns['source_amplitude'](inst)
        np.testing.assert_array_equal(inst.spectrum.wavelengths, waves)
        self.assertNotEqual(float(first[0, 0]), float(second[0, 0]))
        self.assertAlmostEqual(float(np.sum(second**2)), 1e9, delta=1e-5)
        self.assertEqual(inst.source.active_profile, 'onsky')
        self.assertEqual(inst.source.temperature_K, 4500)


if __name__ == '__main__':
    unittest.main()
