"""Regression tests for reusing one physical DM OPD per beam per frame.

Real production optical functions are executed with synthetic, nonzero DM
influence maps. The live runtime is also inspected structurally because
importing its top-level code would start SHM/ZMQ hardware simulation.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from test_real_optics import EXAMPLE, load_real_optics, make_instrument

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "src/baldrapp/paranal_simulator/scripts/baldr_simulator.py"


class CountingDM:
    """Nonzero, spatially varying physical DM surface with eval call count."""

    def __init__(self, shape):
        y, x = np.indices(shape)
        self.surface_per_metre = (
            np.sin(2 * np.pi * x / 27.0) * np.cos(2 * np.pi * y / 34.0)
        )
        self.calls = 0

    def eval(self, commands_m):
        self.calls += 1
        return np.asarray(commands_m)[0] * self.surface_per_metre


class DMReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        cls.core = load_real_optics()

    def make_case(self, *, fresnel, wavelengths):
        inst, amp = make_instrument(
            self.cfg, wavelengths, size=64, pupil_diameter=40,
            fresnel_enabled=fresnel,
        )
        dm = CountingDM(amp.shape)
        inst.compiled_dm = dm
        # OPD = physical command * opd_per_cmd, just as in the live runtime.
        inst.dm.current_cmd = np.array([0.3])
        inst.dm.opd_per_cmd = 3e-7
        y, x = np.indices(amp.shape)
        opd = 8e-8 * np.sin(2 * np.pi * x / 35) * np.cos(2 * np.pi * y / 32)
        return inst, amp, opd, dm

    def frame(self, inst, amp, opd, **kwargs):
        return self.core["get_frame_configured"](
            opd_input=opd,
            amp_input=amp,
            opd_internal=np.zeros_like(opd),
            zwfs_ns=inst,
            detector=None,
            include_shotnoise=False,
            **kwargs,
        )

    def test_all_four_dispatch_paths_preserve_nonzero_dm_frames(self):
        for fresnel in (False, True):
            for wavelengths in (1, 3):
                with self.subTest(fresnel=fresnel, wavelengths=wavelengths):
                    inst, amp, opd, dm = self.make_case(
                        fresnel=fresnel, wavelengths=wavelengths
                    )
                    # Exercise the actual monochromatic dispatcher on the
                    # one-wavelength case, not the v2 1-node spectral wrapper.
                    options = ({"force_polychromatic": False}
                               if wavelengths == 1 else {})
                    reference = self.frame(inst, amp, opd, **options)
                    self.assertGreaterEqual(dm.calls, 1)
                    snapshot = dm.eval(inst.dm.current_cmd * inst.dm.opd_per_cmd)
                    before = dm.calls
                    reused = self.frame(inst, amp, opd, dm_opd=snapshot, **options)
                    self.assertEqual(dm.calls, before)
                    np.testing.assert_allclose(
                        reused, reference, rtol=2e-12, atol=1e-7
                    )

    def test_next_frame_uses_new_dm_command_not_stale_surface(self):
        inst, amp, opd, dm = self.make_case(fresnel=True, wavelengths=3)
        frames = []
        for command in (0.15, -0.48):
            inst.dm.current_cmd = np.array([command])
            surface = dm.eval(inst.dm.current_cmd * inst.dm.opd_per_cmd)
            before = dm.calls
            actual = self.frame(inst, amp, opd, dm_opd=surface)
            self.assertEqual(dm.calls, before)
            np.testing.assert_allclose(
                actual, self.frame(inst, amp, opd), rtol=2e-12, atol=1e-7
            )
            frames.append(actual)
        self.assertGreater(np.linalg.norm(frames[0] - frames[1]), 1e4)

    def test_chromatic_incident_fields_preserve_dm_phase(self):
        for fresnel in (False, True):
            with self.subTest(fresnel=fresnel):
                inst, amp, opd, dm = self.make_case(
                    fresnel=fresnel, wavelengths=3
                )
                y, x = np.indices(amp.shape)
                wavefront = np.sin(2 * np.pi * x / 29) * np.cos(2 * np.pi * y / 41)
                fields = np.stack([
                    amp.astype(complex) * np.exp(1j * wavefront * f)
                    for f in (0.06, -0.10, 0.13)
                ])
                reference = self.frame(inst, amp, opd, incident_fields=fields)
                surface = dm.eval(inst.dm.current_cmd * inst.dm.opd_per_cmd)
                before = dm.calls
                actual = self.frame(
                    inst, amp, opd, incident_fields=fields, dm_opd=surface
                )
                self.assertEqual(dm.calls, before)
                np.testing.assert_allclose(actual, reference, rtol=2e-12, atol=1e-7)

    def test_deterministic_detector_counts_preserved(self):
        for fresnel in (False, True):
            with self.subTest(fresnel=fresnel):
                inst, amp, opd, dm = self.make_case(
                    fresnel=fresnel, wavelengths=3
                )
                detector = SimpleNamespace(binning=2, qe=0.71, dit=0.003, ron=0.)
                inst.detector = detector
                reference = self.frame(inst, amp, opd)
                frame_dm = dm.eval(inst.dm.current_cmd * inst.dm.opd_per_cmd)
                calls_before = dm.calls
                received = self.frame(inst, amp, opd, dm_opd=frame_dm)
                self.assertEqual(dm.calls, calls_before)
                np.testing.assert_allclose(received, reference, rtol=2e-12, atol=1e-8)

    def test_bad_dm_shape_is_rejected(self):
        inst, amp, opd, dm = self.make_case(fresnel=True, wavelengths=3)
        with self.assertRaisesRegex(ValueError, "dm_opd must match"):
            self.frame(inst, amp, opd, dm_opd=np.zeros((4, 4)))
        self.assertEqual(dm.calls, 0)

    def test_diagnostics_use_supplied_map_without_evaluation(self):
        # This helper is extracted unchanged; importing the full runtime would
        # also start camera/SHM servers.
        tree = ast.parse(RUNTIME.read_text(encoding="utf-8"))
        node = next(
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "update_opd_diagnostics"
        )
        namespace = {
            "np": np,
            "SimpleNamespace": SimpleNamespace,
            "measure_opd_components": lambda arr, pupil: {
                "opd_sum_m": float(np.sum(arr * pupil)),
            },
        }
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(RUNTIME), "exec"), namespace)
        inst, amp, opd, dm = self.make_case(fresnel=True, wavelengths=3)
        map_this_frame = dm.eval(inst.dm.current_cmd * inst.dm.opd_per_cmd)
        before = dm.calls
        result = namespace["update_opd_diagnostics"](
            inst, opd, np.zeros_like(opd), 123, "onsky", opd_dm=map_this_frame
        )
        self.assertEqual(dm.calls, before)
        self.assertAlmostEqual(result["opd_sum_m"],
                               float(np.sum((opd + map_this_frame) * inst.grid.pupil_mask)))
        self.assertEqual(result["frame"], 123)

    def test_runtime_snapshots_dm_once_after_shm_read_and_reuses_it(self):
        """Static guard against reintroducing duplicate runtime DM evaluations."""
        tree = ast.parse(RUNTIME.read_text(encoding="utf-8"))
        beam_loops = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.For)
            and isinstance(n.target, ast.Tuple)
            and [getattr(v, "id", None) for v in n.target.elts] == ["ct", "beam"]
        ]
        beam_loops = [
            loop for loop in beam_loops
            if any(isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute)
                   and n.func.attr == "get_frame_configured"
                   for n in ast.walk(loop))
        ]
        self.assertEqual(len(beam_loops), 1)
        beam_loop = beam_loops[0]
        calls = [n for n in ast.walk(beam_loop) if isinstance(n, ast.Call)]
        dm_evals = [
            n for n in calls if isinstance(n.func, ast.Attribute)
            and n.func.attr == "eval"
        ]
        self.assertEqual(len(dm_evals), 1)
        def by_name(name):
            return [n for n in calls
                    if (isinstance(n.func, ast.Attribute) and n.func.attr == name)
                    or (isinstance(n.func, ast.Name) and n.func.id == name)]
        def kw_is(call, name, variable):
            return any(k.arg == name and isinstance(k.value, ast.Name)
                       and k.value.id == variable for k in call.keywords)
        self.assertEqual(len(by_name("get_frame_configured")), 1)
        self.assertTrue(kw_is(by_name("get_frame_configured")[0], "dm_opd", "frame_dm_opd"))
        self.assertEqual(len(by_name("update_opd_diagnostics")), 1)
        self.assertTrue(kw_is(by_name("update_opd_diagnostics")[0], "opd_dm", "frame_dm_opd"))
        publications = by_name("publish_frame")
        self.assertEqual(len(publications), 1)
        self.assertIsInstance(publications[0].args[3], ast.Name)
        self.assertEqual(publications[0].args[3].id, "frame_dm_opd")
        self.assertLess(by_name("read_dm_command")[0].lineno, dm_evals[0].lineno)
        self.assertLess(dm_evals[0].lineno, by_name("get_frame_configured")[0].lineno)


if __name__ == "__main__":
    unittest.main()
