"""Regression tests for fast Fresnel frame generation (real optical FFTs)."""
import json
import unittest

import numpy as np
from baldrapp.utils import fresnel
from test_real_optics import EXAMPLE, load_real_optics, make_instrument


class CountingDM:
    def __init__(self, shape):
        self.shape = shape
        self.calls = 0

    def eval(self, command):
        self.calls += 1
        return np.zeros(self.shape)


class FresnelOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads(EXAMPLE.read_text())
        cls.optics = load_real_optics()

    def test_scipy_fft_matches_numpy_normalization(self):
        rng = np.random.default_rng(7)
        field = rng.standard_normal((21, 32)) + 1j * rng.standard_normal((21, 32))
        for norm in (None, 'ortho'):
            np.testing.assert_allclose(fresnel.fft2(field, norm=norm),
                                       np.fft.fft2(field, norm=norm), rtol=1e-13, atol=1e-12)
            np.testing.assert_allclose(fresnel.ifft2(field, norm=norm),
                                       np.fft.ifft2(field, norm=norm), rtol=1e-13, atol=1e-12)

    def test_cached_mirror_geometry_and_invalidation(self):
        shape, dx = (64, 64), 1.0e-5
        X, Y = fresnel.make_coordinate_grid(shape, dx)
        for angle, offset in ((0.0, -1.e-5), (0.5, 2.e-5), (-0.4, -3.e-5)):
            full = fresnel.circular_aperture(shape, dx, radius=2.5e-4).astype(float)
            expected = full * (X*np.cos(angle)+Y*np.sin(angle) >= offset)
            result = fresnel.cached_d_mirror_mask(shape, dx, 5.e-4, offset, angle)
            np.testing.assert_array_equal(result, expected)
            self.assertIs(result, fresnel.cached_d_mirror_mask(shape, dx, 5.e-4, offset, angle))
            self.assertFalse(result.flags.writeable)

    def test_cached_coldstop_geometry_and_invalidation(self):
        shape = (64, 64)
        for scale, x0, y0 in ((8.e-6, 0., 0.), (8.e-6, 2.e-5, 0.),
                              (9.e-6, 0., 3.e-5)):
            X, Y = fresnel.make_coordinate_grid(shape, scale, scale)
            expected = (np.hypot(X-x0, Y-y0) <= 2.e-4).astype(float)
            result = fresnel.cached_coldstop_mask(shape, scale, scale, 4.e-4, x0, y0)
            np.testing.assert_array_equal(result, expected)
            self.assertFalse(result.flags.writeable)

    def test_focal_disc_uses_same_radius_rule(self):
        Nf = 129
        rr = fresnel.centered_pixel_radius((Nf, Nf))
        for radius in (2.1, 6.0, 8.25):
            mask = fresnel.circular_focal_mask(Nf, radius)
            np.testing.assert_array_equal(mask, (rr <= radius).astype(float))
            self.assertIs(mask, fresnel.circular_focal_mask(Nf, radius))

    def test_polychromatic_real_optics_matches_unmodified_reference(self):
        # Calculated using the unmodified Integrated NumPy Fresnel implementation,
        # 64x64, 7 wavelengths, and the deterministic OPD below.
        inst, amp = make_instrument(self.cfg, 7, size=64, pupil_diameter=40,
                                    fresnel_enabled=True)
        yy, xx = np.indices(amp.shape)
        opd = 8.e-8 * np.sin(2*np.pi*xx/33) * np.cos(2*np.pi*yy/39)
        dm = CountingDM(amp.shape)
        inst.compiled_dm = dm
        frame = self.optics['get_frame_fresnel_polychromatic'](
            opd, amp, np.zeros_like(amp), inst, detector=None, include_shotnoise=False)
        self.assertEqual(dm.calls, 1)
        self.assertEqual(frame.shape, (64, 64))
        self.assertAlmostEqual(float(frame.sum()) / 852463099.9513786, 1.0, places=12)
        self.assertAlmostEqual(float(frame[32, 32]) / 380766.36094984354, 1.0, places=12)
        self.assertAlmostEqual(float(frame[32, 37]) / 311101.5710074418, 1.0, places=12)
        self.assertAlmostEqual(float(frame[21, 16]) / 249774.3391428417, 1.0, places=12)

    def test_direct_monochromatic_still_evaluates_dm(self):
        inst, amp = make_instrument(self.cfg, 3, size=64, pupil_diameter=40,
                                    fresnel_enabled=True)
        dm = CountingDM(amp.shape)
        inst.compiled_dm = dm
        z = np.zeros_like(amp)
        result = self.optics['get_frame_fresnel'](
            z, amp, z, inst, detector=None, include_shotnoise=False,
            wavelength=1.65e-6)
        self.assertEqual(dm.calls, 1)
        self.assertTrue(np.isfinite(result).all())

    def test_chromatic_incident_fields_evaluate_dm_once(self):
        inst, amp = make_instrument(self.cfg, 3, size=64, pupil_diameter=40,
                                    fresnel_enabled=True)
        dm = CountingDM(amp.shape)
        inst.compiled_dm = dm
        z = np.zeros_like(amp)
        fields = np.stack([amp.astype(complex)*np.exp(1j*k*.02) for k in (1, 2, 3)])
        frame = self.optics['get_frame_fresnel_polychromatic'](
            z, amp, z, inst, detector=None, include_shotnoise=False,
            incident_fields=fields)
        self.assertEqual(dm.calls, 1)
        self.assertTrue(np.isfinite(frame).all())


if __name__ == '__main__':
    unittest.main()
