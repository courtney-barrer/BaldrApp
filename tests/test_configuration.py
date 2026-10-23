"""Hardware-independent tests for the BaldrApp v2 configuration contract."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from baldrapp.utils.configuration import (
    ConfigurationError, load_and_resolve_config, load_config,
    resolve_config, validate_config,
)


EXAMPLE = (Path(__file__).resolve().parents[1] / "src" / "baldrapp" /
           "paranal_simulator" / "fake_configs" / "baldr_config_v2.json")
LEGACY = EXAMPLE.with_name("baldr_config.json")
REFERENCE = Path(__file__).resolve().parent / "fixtures" / "baldr_config_reference.json"


class V2ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(REFERENCE.read_text(encoding="utf-8"))

    def assert_invalid(self, path, configure):
        cfg = copy.deepcopy(self.cfg)
        configure(cfg)
        with self.assertRaisesRegex(ConfigurationError, path):
            validate_config(cfg)

    def test_example_is_valid(self):
        # The live example is user-editable (e.g. 3 or 7 spectral samples),
        # whereas the reference fixture below stays fixed for contract tests.
        live_cfg = load_config(EXAMPLE)
        validate_config(live_cfg)
        report = load_and_resolve_config(EXAMPLE)
        self.assertEqual(report["configuration"]["schema_version"], 2)
        self.assertEqual(
            len(report["derived"]["bandpass"]["wavelengths_m"]),
            live_cfg["bandpass"]["n_wavelengths"],
        )
        validate_config(self.cfg)

    def test_original_config_remains_available_for_v1(self):
        legacy = json.loads(LEGACY.read_text(encoding="utf-8"))
        self.assertIn("stellar", legacy)
        self.assertIn("source", legacy)
        self.assertNotIn("schema_version", legacy)
        with self.assertRaisesRegex(ConfigurationError, "legacy v1 field"):
            validate_config(legacy)

    def test_legacy_input_is_rejected_with_location(self):
        self.assert_invalid(r"\$\.stellar", lambda c: c.update(stellar={"bandwidth":300}))

    def test_unknown_top_level_rejected(self):
        self.assert_invalid(r"\$\.extraneous", lambda c: c.update(extraneous=3))

    def test_unknown_nested_field_rejected(self):
        self.assert_invalid(r"\$\.source_profiles\.onsky\.flux\.flux_mode",
                            lambda c: c["source_profiles"]["onsky"]["flux"].update(flux_mode="a"))

    def test_boolean_not_accepted_as_int(self):
        self.assert_invalid(r"\$\.bandpass\.n_wavelengths",
                            lambda c: c["bandpass"].update(n_wavelengths=True))

    def test_bad_wavelength_range(self):
        self.assert_invalid(r"\$\.bandpass\.wavelength_max_m",
                            lambda c: c["bandpass"].update(wavelength_max_m=1.0e-6))

    def test_missing_source(self):
        self.assert_invalid(r"\$\.source_profiles\.internal",
                            lambda c: c["source_profiles"].pop("internal"))

    def test_bad_spectrum_temperature(self):
        self.assert_invalid(r"\$\.source_profiles\.onsky\.spectrum\.temperature_K",
                            lambda c: c["source_profiles"]["onsky"]["spectrum"].update(temperature_K=0))

    def test_flat_spectrum_disallows_temperature(self):
        self.assert_invalid(r"\$\.source_profiles\.internal\.spectrum\.temperature_K",
                            lambda c: c["source_profiles"]["internal"]["spectrum"].update(model="flat"))

    def test_negative_flux(self):
        self.assert_invalid(r"\$\.source_profiles\.internal\.flux\.photons_per_second",
                            lambda c: c["source_profiles"]["internal"]["flux"].update(photons_per_second=-1))

    def test_zero_flux_for_dark_is_valid(self):
        self.cfg["source_profiles"]["internal"]["flux"]["photons_per_second"] = 0
        validate_config(self.cfg)

    def test_missing_atmospheric_layer(self):
        self.assert_invalid(r"\$\.atmosphere\.layers",
                            lambda c: c["atmosphere"].update(layers=[]))

    def test_invalid_cn2_fractions(self):
        self.assert_invalid(r"\$\.atmosphere\.layers",
                            lambda c: c["atmosphere"]["layers"][0].update(cn2_fraction=0.8))

    def test_bad_layer_wind(self):
        self.assert_invalid(r"\$\.atmosphere\.layers\[1\]\.wind_speed_m_per_s",
                            lambda c: c["atmosphere"]["layers"][1].update(wind_speed_m_per_s=-3))

    def test_bad_scintillation_old_field(self):
        self.assert_invalid(r"\$\.atmosphere\.scintillation\.r0_500_m",
                            lambda c: c["atmosphere"]["scintillation"].update(r0_500_m=0.2))

    def test_bad_sim_time(self):
        self.assert_invalid(r"\$\.simulator_runtime\.time_step_s",
                            lambda c: c["simulator_runtime"].update(time_step_s=0))

    def test_at_ut_diameter_mismatch_rejected(self):
        self.assert_invalid(r"\$\.source_profiles\.onsky\.pupil\.model",
                            lambda c: c["source_profiles"]["onsky"]["pupil"].update(model="UT"))

    def test_photon_fractions_positive_and_normalized(self):
        report = resolve_config(self.cfg)["derived"]
        self.assertAlmostEqual(report["bandpass"]["bandwidth_nm"], 300.0)
        for source in report["source_profiles"].values():
            weights = source["photon_fractions"]
            self.assertEqual(len(weights), 7)
            self.assertTrue(all(v >= 0 for v in weights))
            self.assertAlmostEqual(sum(weights), 1)
            self.assertAlmostEqual(sum(source["photons_per_second_by_wavelength"]), 1e9, places=4)
        self.assertNotEqual(report["source_profiles"]["internal"]["photon_fractions"],
                            report["source_profiles"]["onsky"]["photon_fractions"])

    def test_flat_photon_density_integral(self):
        self.cfg["source_profiles"]["internal"]["spectrum"] = {"model": "flat"}
        f = resolve_config(self.cfg)["derived"]["source_profiles"]["internal"]["photon_fractions"]
        self.assertAlmostEqual(f[0], 1/12)
        self.assertAlmostEqual(f[1], 1/6)
        self.assertAlmostEqual(f[-1], 1/12)

    def test_one_wavelength_uses_wvl0_and_full_flux(self):
        self.cfg["bandpass"]["n_wavelengths"] = 1
        report = resolve_config(self.cfg)["derived"]
        self.assertEqual(report["bandpass"]["wavelengths_m"], [self.cfg["optics"]["wvl0"]])
        self.assertEqual(report["source_profiles"]["internal"]["photon_fractions"], [1.0])

    def test_tau0_and_wind_components(self):
        derived = resolve_config(self.cfg)["derived"]["atmosphere"]
        f0, f1 = 0.7, 0.3
        v_eff = (f0*6**(5/3) + f1*20**(5/3))**(3/5)
        self.assertAlmostEqual(derived["effective_wind_speed_m_per_s"], v_eff)
        self.assertAlmostEqual(derived["tau0_500nm_s"], 0.314*0.126/v_eff)
        self.assertAlmostEqual(derived["layers"][0]["r0_500nm_m"], 0.126*f0**(-3/5))
        self.assertAlmostEqual(derived["layers"][1]["wind_x_m_per_s"], 0, places=12)
        self.assertAlmostEqual(derived["layers"][1]["wind_y_m_per_s"], 20)

    def test_zero_wind_reports_infinite_tau0_as_none(self):
        for layer in self.cfg["atmosphere"]["layers"]:
            layer["wind_speed_m_per_s"] = 0
        tau = resolve_config(self.cfg)["derived"]["atmosphere"]["tau0_500nm_s"]
        self.assertIsNone(tau)
        json.dumps(resolve_config(self.cfg), allow_nan=False)

    def test_resolution_does_not_mutate_input(self):
        before = copy.deepcopy(self.cfg)
        resolve_config(self.cfg)
        self.assertEqual(self.cfg, before)

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "duplicate.json"
            p.write_text('{"a": 1, "a": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "duplicate key"):
                load_config(p)

    def test_nan_json_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "nan.json"
            p.write_text('{"x": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "non-finite"):
                load_config(p)

    def test_json_with_comments_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "commented.json"
            p.write_text('{"x": 1 // comment\n}', encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "invalid JSON"):
                load_config(p)


if __name__ == "__main__":
    unittest.main()
