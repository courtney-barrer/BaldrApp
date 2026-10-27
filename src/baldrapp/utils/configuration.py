"""Validate BaldrApp JSON inputs and derive optical and atmospheric parameters.

The user configuration describes physical instrument and source parameters.
Numerical wavelength weights, turbulence strengths and timescales are derived
without changing the supplied input dictionary.

CLI: python -m baldrapp.utils.configuration path/to/baldr_config.json
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any


LAMBDA_REF_M = 500e-9
PLANCK_C2_M_K = 1.438776877e-2  # h*c/k_B in m K

_TOP_LEVEL_KEYS = frozenset({
    "grid", "optics", "dm", "fresnel_relay", "detector",
    "internal_aberrations", "first_stage_ao", "simulator_runtime",
    "bandpass", "source_profiles", "atmosphere", "meta", "throughput",
})
_OLD_TOP_LEVEL_KEYS = {"source", "stellar", "spectrum", "schema_version"}
_PUPIL_MODELS = frozenset({"SOLARSTEIN", "AT", "UT", "DISC"})


class ConfigurationError(ValueError):
    """Invalid or unsupported BaldrApp configuration, with a JSON path."""


def _fail(path: str, message: str) -> None:
    raise ConfigurationError(f"{path}: {message}")


def _object(value: Any, path: str, *, required: set[str] = frozenset(),
            optional: set[str] = frozenset()) -> dict:
    if not isinstance(value, dict):
        _fail(path, "must be a JSON object")
    unexpected = set(value) - set(required) - set(optional)
    if unexpected:
        _fail(path + "." + sorted(unexpected)[0], "unknown configuration field")
    missing = set(required) - set(value)
    if missing:
        _fail(path + "." + sorted(missing)[0], "required field is missing")
    return value


def _number(value: Any, path: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a number (not a Boolean)")
    result = float(value)
    if not math.isfinite(result):
        _fail(path, "must be finite")
    if positive and result <= 0:
        _fail(path, "must be > 0")
    if nonnegative and result < 0:
        _fail(path, "must be >= 0")
    return result


def _integer(value: Any, path: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(path, f"must be an integer >= {minimum}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        _fail(path, "must be a Boolean")
    return value


def _enum(value: Any, path: str, allowed: set[str] | frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        _fail(path, f"must be one of {', '.join(sorted(allowed))}")
    return value


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigurationError(f"JSON contains duplicate key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise ConfigurationError(f"JSON contains invalid non-finite number {token}")


def load_config(path: str | Path) -> dict:
    """Read strict JSON and validate the instrument and source inputs."""
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            cfg = json.load(stream, object_pairs_hook=_pairs_no_duplicates,
                            parse_constant=_reject_nonfinite)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"{config_path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    """Validate physical inputs, including nested source and atmosphere blocks."""
    if not isinstance(cfg, dict):
        _fail("$", "configuration must be a JSON object")
    for old in sorted(_OLD_TOP_LEVEL_KEYS & set(cfg)):
        _fail(f"$.{old}", "obsolete input; use bandpass and source_profiles instead")
    _object(cfg, "$", required={
        "grid", "optics", "dm", "bandpass",
        "source_profiles", "atmosphere", "simulator_runtime",
    }, optional=_TOP_LEVEL_KEYS)

    # Preserve established hardware block contents for the optical initializer.
    for section in ("grid", "optics", "dm", "fresnel_relay", "detector",
                    "internal_aberrations", "first_stage_ao", "simulator_runtime",
                    "throughput", "meta"):
        if section in cfg and not isinstance(cfg[section], dict):
            _fail(f"$.{section}", "must be a JSON object")

    grid = cfg["grid"]
    _number(grid.get("D"), "$.grid.D", positive=True)
    n = _integer(grid.get("N"), "$.grid.N")
    dim = _integer(grid.get("dim"), "$.grid.dim")
    if dim < n:
        _fail("$.grid.dim", "must be >= grid.N")
    if "telescope" in grid:
        _enum(str(grid["telescope"]).upper(), "$.grid.telescope", _PUPIL_MODELS)

    optics = cfg["optics"]
    wvl0 = _number(optics.get("wvl0"), "$.optics.wvl0", positive=True)

    band = _object(cfg["bandpass"], "$.bandpass", required={
        "wavelength_min_m", "wavelength_max_m", "n_wavelengths"
    })
    min_wvl = _number(band["wavelength_min_m"], "$.bandpass.wavelength_min_m", positive=True)
    max_wvl = _number(band["wavelength_max_m"], "$.bandpass.wavelength_max_m", positive=True)
    if max_wvl <= min_wvl:
        _fail("$.bandpass.wavelength_max_m", "must be > wavelength_min_m")
    _integer(band["n_wavelengths"], "$.bandpass.n_wavelengths")
    if not min_wvl <= wvl0 <= max_wvl:
        _fail("$.optics.wvl0", "must lie within the configured bandpass")

    profiles = _object(cfg["source_profiles"], "$.source_profiles",
                       required={"internal", "onsky"})
    for name in ("internal", "onsky"):
        path = f"$.source_profiles.{name}"
        profile = _object(profiles[name], path, required={"pupil", "spectrum", "flux"})
        pupil = _object(profile["pupil"], path + ".pupil", required={"model"})
        model = _enum(pupil["model"], path + ".pupil.model", _PUPIL_MODELS)
        spectrum = _object(profile["spectrum"], path + ".spectrum",
                           required={"model"}, optional={"temperature_K"})
        spectrum_model = _enum(spectrum["model"], path + ".spectrum.model",
                               {"blackbody", "flat"})
        if spectrum_model == "blackbody":
            _number(spectrum.get("temperature_K"), path + ".spectrum.temperature_K",
                    positive=True)
        elif "temperature_K" in spectrum:
            _fail(path + ".spectrum.temperature_K", "only meaningful for blackbody spectra")
        flux = _object(profile["flux"], path + ".flux",
                       required={"photons_per_second"})
        _number(flux["photons_per_second"], path + ".flux.photons_per_second",
                nonnegative=True)
        # Profile switching currently changes the numerical pupil mask but not
        # the physical pupil diameter. Catch an egregious AT<->UT mismatch.
        diameter_m = float(grid["D"])
        if name == "onsky" and model == "UT" and diameter_m < 5.0:
            _fail(path + ".pupil.model", "UT requires a UT-scale grid.D; pupil switching does not rescale diameter")
        if name == "onsky" and model == "AT" and diameter_m > 5.0:
            _fail(path + ".pupil.model", "AT requires an AT-scale grid.D; pupil switching does not rescale diameter")

    atmo = _object(cfg["atmosphere"], "$.atmosphere", required={
        "enabled", "model", "r0_500nm_m", "L0_m", "layers", "scintillation"
    }, optional={"random_seed"})
    _boolean(atmo["enabled"], "$.atmosphere.enabled")
    _enum(atmo["model"], "$.atmosphere.model", {"von_karman"})
    _number(atmo["r0_500nm_m"], "$.atmosphere.r0_500nm_m", positive=True)
    _number(atmo["L0_m"], "$.atmosphere.L0_m", positive=True)
    if "random_seed" in atmo:
        _integer(atmo["random_seed"], "$.atmosphere.random_seed", minimum=0)
    layers = atmo["layers"]
    if not isinstance(layers, list) or not layers:
        _fail("$.atmosphere.layers", "must be a nonempty list")
    total_fraction = 0.0
    for i, layer in enumerate(layers):
        path = f"$.atmosphere.layers[{i}]"
        _object(layer, path, required={
            "altitude_m", "cn2_fraction", "wind_speed_m_per_s", "wind_direction_deg"
        })
        _number(layer["altitude_m"], path + ".altitude_m", nonnegative=True)
        total_fraction += _number(layer["cn2_fraction"], path + ".cn2_fraction",
                                  nonnegative=True)
        _number(layer["wind_speed_m_per_s"], path + ".wind_speed_m_per_s",
                nonnegative=True)
        direction = _number(layer["wind_direction_deg"], path + ".wind_direction_deg")
        if not 0 <= direction < 360:
            _fail(path + ".wind_direction_deg", "must lie in [0, 360) degrees")
    if not math.isclose(total_fraction, 1.0, rel_tol=0.0, abs_tol=1e-6):
        _fail("$.atmosphere.layers", f"Cn2 fractions must sum to 1 (got {total_fraction:.10g})")
    scint = _object(atmo["scintillation"], "$.atmosphere.scintillation",
                    required={"enabled"})
    _boolean(scint["enabled"], "$.atmosphere.scintillation.enabled")

    # Validate modal-AO controls before any atmospheric state is created.
    if "first_stage_ao" in cfg:
        ao = cfg["first_stage_ao"]
        if "enabled" in ao:
            _boolean(ao["enabled"], "$.first_stage_ao.enabled")
        if "model" in ao:
            _enum(ao["model"], "$.first_stage_ao.model", {"modal_removal"})
        if "basis" in ao:
            _enum(ao["basis"], "$.first_stage_ao.basis", {"zernike"})
        if "apply_to" in ao:
            _enum(ao["apply_to"], "$.first_stage_ao.apply_to", {"phase"})
        if "Nmodes_removed" in ao:
            _integer(ao["Nmodes_removed"], "$.first_stage_ao.Nmodes_removed", minimum=0)
        if "lag_frames" in ao:
            _integer(ao["lag_frames"], "$.first_stage_ao.lag_frames", minimum=0)
        if "correction_gain" in ao:
            gain = _number(ao["correction_gain"], "$.first_stage_ao.correction_gain", nonnegative=True)
            if gain > 1:
                _fail("$.first_stage_ao.correction_gain", "must be <= 1")
        if "remove_piston" in ao:
            _boolean(ao["remove_piston"], "$.first_stage_ao.remove_piston")
        if "phase_scaling_factor" in ao:
            _number(ao["phase_scaling_factor"], "$.first_stage_ao.phase_scaling_factor", nonnegative=True)

    runtime = cfg["simulator_runtime"]
    _number(runtime.get("time_step_s"), "$.simulator_runtime.time_step_s", positive=True)
    _number(runtime.get("sleep_time_s"), "$.simulator_runtime.sleep_time_s",
            nonnegative=True)


def _spectral_shape(wvl_m: float, spectrum: dict) -> float:
    if spectrum["model"] == "flat":
        # Flat *photon spectral density* across the bandpass.
        return 1.0
    t = float(spectrum["temperature_K"])
    x = PLANCK_C2_M_K / (wvl_m * t)
    if x > 700.0:
        return 0.0
    # B_lambda * lambda/(h*c) ~ lambda^-4 / [exp(c2/(lambda*T))-1]
    return wvl_m ** -4 / math.expm1(x)


def _spectral_fraction(nodes: list[float], model: dict) -> list[float]:
    if len(nodes) == 1:
        return [1.0]
    shape = [_spectral_shape(wvl, model) for wvl in nodes]
    # Trapezoidal integration on a nonuniform grid; these coefficients
    # approximate the photon integral, not a raw (unweighted) sample sum.
    bin_widths = [
        0.5 * (nodes[1] - nodes[0]) if i == 0 else
        0.5 * (nodes[-1] - nodes[-2]) if i == len(nodes) - 1 else
        0.5 * (nodes[i + 1] - nodes[i - 1])
        for i in range(len(nodes))
    ]
    weights = [p * d for p, d in zip(shape, bin_widths)]
    total = sum(weights)
    if total <= 0 or not math.isfinite(total):
        _fail("$.source_profiles.*.spectrum", "spectrum has zero or invalid integrated photon flux")
    return [weight / total for weight in weights]


def resolve_config(cfg: dict) -> dict:
    """Return validated input plus *separate*, JSON-serialisable derived fields.

    Resolution does not instantiate an instrument or apply optical propagation.
    """
    validate_config(cfg)
    band = cfg["bandpass"]
    n = band["n_wavelengths"]
    lo, hi = band["wavelength_min_m"], band["wavelength_max_m"]
    if n == 1:
        wavelengths = [float(cfg["optics"]["wvl0"])]
    else:
        wavelengths = [float(lo + (hi - lo) * i / (n - 1)) for i in range(n)]

    sources = {}
    for name, source in cfg["source_profiles"].items():
        fractions = _spectral_fraction(wavelengths, source["spectrum"])
        rate = float(source["flux"]["photons_per_second"])
        sources[name] = {
            "pupil_model": source["pupil"]["model"],
            "photon_fractions": fractions,
            "photons_per_second_by_wavelength": [rate * f for f in fractions],
            "total_photons_per_second": rate,
            "atmosphere_applies": name == "onsky" and bool(cfg["atmosphere"]["enabled"]),
        }

    atmo = cfg["atmosphere"]
    r0 = float(atmo["r0_500nm_m"])
    turbulence = []
    v53 = 0.0
    for index, layer in enumerate(atmo["layers"]):
        f = float(layer["cn2_fraction"])
        v = float(layer["wind_speed_m_per_s"])
        direction = math.radians(float(layer["wind_direction_deg"]))
        v53 += f * v**(5.0 / 3.0)
        turbulence.append({
            "index": index,
            "altitude_m": float(layer["altitude_m"]),
            "cn2_fraction": f,
            "r0_500nm_m": r0 * f**(-3.0 / 5.0) if f > 0 else None,
            "wind_x_m_per_s": v * math.cos(direction),
            "wind_y_m_per_s": v * math.sin(direction),
        })
    effective_wind = v53**(3.0 / 5.0)
    # None is JSON-compatible and represents tau0 = infinity (all winds zero).
    tau0 = (0.314 * r0 / effective_wind) if effective_wind > 0 else None
    return {
        "configuration": deepcopy(cfg),
        "derived": {
            "bandpass": {
                "wavelengths_m": wavelengths,
                "bandwidth_nm": (hi - lo) * 1e9,
            },
            "source_profiles": sources,
            "atmosphere": {
                "reference_wavelength_m": LAMBDA_REF_M,
                "random_seed": atmo.get("random_seed", 0),
                "layers": turbulence,
                "effective_wind_speed_m_per_s": effective_wind,
                "tau0_500nm_s": tau0,
            },
            "simulator_runtime": {
                "time_step_s": float(cfg["simulator_runtime"]["time_step_s"]),
                "sleep_time_s": float(cfg["simulator_runtime"]["sleep_time_s"]),
            },
        },
    }


def load_and_resolve_config(path: str | Path) -> dict:
    return resolve_config(load_config(path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate BaldrApp configuration and show derived parameters."
    )
    parser.add_argument("config", type=Path, help="Input BaldrApp JSON configuration")
    parser.add_argument("--dump-resolved", type=Path, metavar="PATH",
                        help="Write a JSON snapshot containing config + derived values")
    args = parser.parse_args(argv)
    try:
        report = load_and_resolve_config(args.config)
    except (ConfigurationError, OSError) as exc:
        parser.exit(2, f"Configuration error: {exc}\n")
    derived = report["derived"]
    print(f"Valid BaldrApp configuration: {args.config}")
    wavelengths = derived["bandpass"]["wavelengths_m"]
    limits = report["configuration"]["bandpass"]
    print(f"  Bandpass: {limits['wavelength_min_m']*1e9:.1f}–{limits['wavelength_max_m']*1e9:.1f} nm"
          f" ({len(wavelengths)} samples; configured width "
          f"{derived['bandpass']['bandwidth_nm']:.1f} nm)")
    for name, source in derived["source_profiles"].items():
        print(f"  {name}: {source['pupil_model']}, "
              f"{source['total_photons_per_second']:.6g} photons/s")
    atm = derived["atmosphere"]
    print(f"  Atmosphere: {len(atm['layers'])} layers, "
          f"v_eff={atm['effective_wind_speed_m_per_s']:.3f} m/s, "
          f"tau0={atm['tau0_500nm_s'] if atm['tau0_500nm_s'] is not None else 'infinite'} s")
    if args.dump_resolved:
        args.dump_resolved.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                      encoding="utf-8")
        print(f"  Saved resolved report: {args.dump_resolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
