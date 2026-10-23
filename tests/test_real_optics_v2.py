"""Stage 2.5: genuine Fourier/Fresnel propagation and detector photon budget.

Unlike test_source_v2, NO optical or detector functions are mocked. We compile
specific function definitions directly from baldr_core.py and spectrum.py so
this numerical suite also runs on hosts without unrelated GUI/SHM/astropy
packages. The extracted bodies are unmodified production implementations.
The only synthetic component is a zero-stroke DM evaluator, which isolates the
optics from hardware registration (not an optical propagator).

All tests are deterministic (shot/read noise disabled). Energies are tracked
at physical planes; aperture clipping and numerical cropping are losses.
"""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from baldrapp.utils import configuration, config_helper, fresnel, source_v2

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / 'src/baldrapp/utils/baldr_core.py'
SPECTRUM = ROOT / 'src/baldrapp/utils/spectrum.py'
EXAMPLE = ROOT / 'src/baldrapp/paranal_simulator/fake_configs/baldr_config_v2.json'

CORE_FUNCTIONS = {
    '_resolve_detector_spectral_bandwidth', 'sum_subarrays', 'detect',
    'get_pupil_intensity', 'get_zwfs_output_field',
    'get_zwfs_output_field_from_opd', 'get_frame',
    'get_frame_polychromatic', 'get_frame_fresnel',
    'get_frame_fresnel_polychromatic', 'get_frame_configured',
}
SPEC_FUNCTIONS = {'theta_at_wavelength', 'phasemask_diameter_at_wavelength'}


def _load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    defs = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in names]
    if {node.name for node in defs} != names:
        raise AssertionError(f'Missing function definitions in {path}: {names - {n.name for n in defs}}')
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def load_real_optics():
    spec_dict = _load_definitions(SPECTRUM, SPEC_FUNCTIONS, {'np': np})
    spec = SimpleNamespace(**{n: spec_dict[n] for n in SPEC_FUNCTIONS})
    optics = {'np': np, 'spec': spec, 'source_v2': source_v2, 'fresnel': fresnel,
              'plt': None, '_AUTO_SPECTRAL_BANDWIDTH': 'auto'}
    return _load_definitions(CORE, CORE_FUNCTIONS, optics)


class ZeroDM:
    def __init__(self, shape):
        self.shape = shape
    def eval(self, command):
        return np.zeros(self.shape, dtype=float)


def make_instrument(cfg, n_wavelengths, *, profile='internal', size=64,
                    pupil_diameter=40, theta=np.pi/2, coldstop_diam=8.4,
                    fresnel_enabled=False):
    """Small deterministic, physically parameterized instrument, no hardware."""
    cfg = deepcopy(cfg)
    cfg['bandpass']['n_wavelengths'] = n_wavelengths
    resolved = configuration.resolve_config(cfg)
    yy, xx = np.indices((size, size), dtype=float)
    rr = np.hypot(xx - size//2, yy - size//2)
    pupil = (rr <= pupil_diameter / 2).astype(float)
    # Use the same geometric pupil for both profiles to isolate the
    # effect of spectral weighting; Stage 2 tests geometrical switching.
    optics = SimpleNamespace(**cfg['optics'])
    optics.theta = float(theta)
    optics.coldstop_diam = coldstop_diam
    fr = config_helper.derive_fresnel_relay_ns(
        config_helper.dict_to_namespace_recursive(cfg['fresnel_relay']))
    fr.enabled = bool(fresnel_enabled)
    inst = SimpleNamespace(
        config_schema_version=2,
        grid=SimpleNamespace(N=pupil_diameter, pupil_mask=pupil),
        focal_plane=SimpleNamespace(fplane_pixels=129, pixels_across_mask=10),
        optics=optics,
        spectrum=source_v2.spectrum_for_profile(resolved, profile),
        fresnel_relay=fr,
        dm=SimpleNamespace(current_cmd=np.zeros(1), opd_per_cmd=1.0),
        compiled_dm=ZeroDM(pupil.shape),
    )
    rate = float(cfg['source_profiles'][profile]['flux']['photons_per_second'])
    amp = source_v2.source_input_amplitude(pupil, rate)
    return inst, amp


class RealOpticsV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads(EXAMPLE.read_text(encoding='utf-8'))
        cls.core = load_real_optics()

    def simulate(self, instrument, amplitude, *, detector=None, opd=None, return_intermediates=False):
        zeros = np.zeros_like(amplitude)
        return self.core['get_frame_configured'](
            opd_input=zeros if opd is None else opd,
            amp_input=amplitude, opd_internal=zeros, zwfs_ns=instrument,
            detector=detector, include_shotnoise=False,
            return_intermediates=return_intermediates,
        )

    def test_unmasked_nofresnel_conserves_photons_all_sample_counts(self):
        # With theta=0 and no cold stop, pupil response is identity; FFTs
        # and wavelength integration must not create or destroy photons.
        for n in (1, 3, 7, 15):
            inst, amp = make_instrument(self.cfg, n, theta=0, coldstop_diam=None)
            out = self.simulate(inst, amp)
            np.testing.assert_allclose(out, amp**2, rtol=3e-12, atol=1e-7)
            self.assertAlmostEqual(out.sum() / np.sum(amp**2), 1.0, places=11)

    def test_real_nofresnel_mask_reduces_photons_only_by_cropping_or_stop(self):
        inst, amp = make_instrument(self.cfg, 7)
        incident = float((amp**2).sum())
        image = self.simulate(inst, amp)
        self.assertTrue(np.all(np.isfinite(image)))
        self.assertTrue(np.all(image >= 0))
        self.assertLessEqual(image.sum(), incident * (1 + 1e-10))
        self.assertGreater(image.sum(), 0)

    def test_real_detector_scaling_and_single_application(self):
        inst, amp = make_instrument(self.cfg, 7, theta=0, coldstop_diam=None)
        base = self.simulate(inst, amp)
        for qe, dit, rate_scale in ((0.7, 0.001, 1), (0.4, 0.002, 1), (0.7, 0.001, 3)):
            det = SimpleNamespace(binning=4, qe=qe, dit=dit, ron=0.0)
            frame = self.simulate(inst, amp*np.sqrt(rate_scale), detector=det)
            self.assertEqual(frame.shape, (16, 16))
            self.assertAlmostEqual(frame.sum() / base.sum(), qe*dit*rate_scale, places=12)
        # The v2 photon fractions must not introduce a 300 nm multiplier.

    def test_real_chromatic_shape_and_convergence_nofresnel(self):
        inst_ref, amp = make_instrument(self.cfg, 51)
        yy, xx = np.indices(amp.shape)
        opd = 1.1e-6 * np.sin(2*np.pi*xx/21) * np.cos(2*np.pi*yy/23)
        reference = self.simulate(inst_ref, amp, opd=opd)
        errors = []
        for n in (3, 7, 15):
            inst, an = make_instrument(self.cfg, n)
            result = self.simulate(inst, an, opd=opd)
            rel = np.linalg.norm(result-reference)/np.linalg.norm(reference)
            errors.append(rel)
        self.assertLess(errors[2], errors[1])
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], 1e-2)

    def test_internal_onsky_spectra_change_real_zwfs_fields(self):
        yy, xx = np.indices((64, 64))
        opd = 1.1e-6 * np.sin(2*np.pi*xx/21) * np.cos(2*np.pi*yy/23)
        for fresnel_enabled in (False, True):
            inst_i, amp_i = make_instrument(self.cfg, 7, profile='internal',
                                            fresnel_enabled=fresnel_enabled)
            inst_s, amp_s = make_instrument(self.cfg, 7, profile='onsky',
                                            fresnel_enabled=fresnel_enabled)
            # Same 1e9 photons/s and entrance geometry; distinguish only
            # the independent source blackbody weights (3000 vs 4500 K).
            np.testing.assert_array_equal(amp_i, amp_s)
            np.testing.assert_array_equal(inst_i.spectrum.wavelengths,
                                          inst_s.spectrum.wavelengths)
            self.assertGreater(np.max(np.abs(inst_i.spectrum.photon_fractions -
                                             inst_s.spectrum.photon_fractions)), 1e-4)
            internal = self.simulate(inst_i, amp_i, opd=opd)
            onsky = self.simulate(inst_s, amp_s, opd=opd)
            rel = np.linalg.norm(internal - onsky)/np.linalg.norm(internal)
            self.assertGreater(rel, 1e-5)
            self.assertLess(rel, .02)
            self.assertTrue(np.all(np.isfinite(onsky)))

    def test_fresnel_photon_rate_linearity_real_zwfs(self):
        inst, amp = make_instrument(self.cfg, 3, fresnel_enabled=True)
        baseline = self.simulate(inst, amp)
        scaled = self.simulate(inst, amp*np.sqrt(2.5))
        np.testing.assert_allclose(scaled, 2.5*baseline, rtol=5e-13, atol=1e-7)

    def test_real_fresnel_planes_physical_energy_accounting(self):
        inst, amp = make_instrument(self.cfg, 1, fresnel_enabled=True)
        results = self.core['get_frame_fresnel'](
            opd_input=np.zeros_like(amp), amp_input=amp,
            opd_internal=np.zeros_like(amp), zwfs_ns=inst,
            detector=None, include_shotnoise=False,
            wavelength=inst.optics.wvl0, return_intermediates=True,
        )
        # The field amplitudes are irradiance samples; multiply by area ratios
        # when comparing grids with different pixel sampling.
        E = lambda a: float(np.vdot(a, a).real)
        postzwfs = E(results['psi_C'])
        mirror = E(results['psi_mirror'])
        mirror_after = E(results['psi_mirror']*results['mirror_mask_D'])
        at_lens = E(results['field_at_lens'])
        after_lens = E(results['field_after_lens'])
        scale_cold = results['dx_cold']*results['dy_cold']/results['dx_phys']**2
        at_cold = E(results['field_coldstop_plane'])*scale_cold
        after_cold = E(results['field_after_coldstop'])*scale_cold
        detector_rate = float(results['intensity_pre_detector'].sum())
        self.assertAlmostEqual(mirror/postzwfs, 1.0, delta=1e-8)
        self.assertLessEqual(mirror_after, mirror*(1+1e-9))
        self.assertAlmostEqual(at_lens/mirror_after, 1.0, delta=1e-8)
        self.assertAlmostEqual(after_lens/at_lens, 1.0, delta=1e-12)
        self.assertAlmostEqual(at_cold/after_lens, 1.0, delta=1e-8)
        self.assertLessEqual(after_cold, at_cold*(1+1e-9))
        self.assertAlmostEqual(detector_rate/after_cold, 1.0, delta=1e-8)
        self.assertLessEqual(detector_rate, (amp**2).sum()*(1+1e-9))

    def test_fresnel_detector_scaling_and_spectral_convergence(self):
        inst, amp = make_instrument(self.cfg, 7, fresnel_enabled=True)
        image = self.simulate(inst, amp)
        det = SimpleNamespace(binning=4, qe=0.62, dit=0.002, ron=0.)
        detected = self.simulate(inst, amp, detector=det)
        self.assertAlmostEqual(detected.sum()/image.sum(), .62*.002, places=12)
        # For a stable chromatic scene, more quadrature nodes should converge
        # towards a high-resolution reference in the actual Fresnel relay.
        inst_ref, amp_ref = make_instrument(self.cfg, 31, fresnel_enabled=True)
        reference = self.simulate(inst_ref, amp_ref)
        errors = []
        for n in (3, 7, 15):
            ns, a = make_instrument(self.cfg, n, fresnel_enabled=True)
            result = self.simulate(ns, a)
            errors.append(np.linalg.norm(result-reference)/np.linalg.norm(reference))
        self.assertLess(errors[-1], errors[0])
        self.assertLess(errors[-1], .05)


if __name__ == '__main__':
    unittest.main()
