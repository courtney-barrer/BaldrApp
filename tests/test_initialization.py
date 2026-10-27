"""Check the real JSON initializer against the canonical source configuration.

The initializer function body is loaded directly, with only the hardware DM/
optical-object constructor replaced by a small namespace. The full optical
initialization is exercised in the running simulator, not in this test.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from baldrapp.utils import config_helper, source
from baldrapp.utils import configuration
from baldrapp.utils.configuration import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src/baldrapp/paranal_simulator/fake_configs/baldr_config.json"
CORE = ROOT / "src/baldrapp/utils/baldr_core.py"


def load_initializer():
    parsed = ast.parse(CORE.read_text(encoding="utf-8"), filename=str(CORE))
    names = {"init_zwfs_from_json", "select_source_profile", "source_amplitude"}
    nodes = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names

    def fake_core(grid_ns, optics_ns, dm_ns):
        # Avoid importing DM, HCIPy and SHM dependencies in this contract test.
        grid_ns.pupil_mask = np.ones((8, 8), dtype=float)
        return SimpleNamespace(grid=grid_ns, optics=optics_ns, dm=dm_ns)

    ns = {"Path": Path, "SimpleNamespace": SimpleNamespace,
          "cfghelp": config_helper, "source": source,
          "configuration": configuration, "init_zwfs": fake_core}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CORE), "exec"), ns)
    return ns


class CanonicalInitializationTests(unittest.TestCase):
    def test_configuration_validates_and_has_no_old_spectral_blocks(self):
        config = load_config(CONFIG)
        self.assertIn("bandpass", config)
        self.assertEqual(set(config["source_profiles"]), {"internal", "onsky"})
        self.assertNotIn("stellar", config)
        self.assertNotIn("source", config)
        self.assertNotIn("schema_version", config)

    def test_initializer_selects_two_sources_without_rebuilding_bandpass(self):
        ns = load_initializer()
        zwfs = ns["init_zwfs_from_json"](CONFIG, derive_fresnel=False,
                                           instantiate_detector=False)
        self.assertTrue(hasattr(zwfs, "resolved_config"))
        internal_wavelengths = zwfs.spectrum.wavelengths.copy()
        first = ns["source_amplitude"](zwfs)
        self.assertAlmostEqual(float(np.sum(first**2)), zwfs.source.photons_per_second, delta=1e-3)
        ns["select_source_profile"](zwfs, "onsky")
        np.testing.assert_array_equal(zwfs.spectrum.wavelengths, internal_wavelengths)
        self.assertAlmostEqual(float(np.sum(zwfs.spectrum.photon_fractions)), 1.0)
        self.assertAlmostEqual(float(np.sum(ns["source_amplitude"](zwfs)**2)),
                               zwfs.source.photons_per_second, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
