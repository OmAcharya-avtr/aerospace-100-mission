"""YAML scenario loading and full twin-report generation.

A scenario file describes one FSO link (see examples/link_10km.yaml):

    name: my-link
    link:
      wavelength_nm: 1550
      tx_power_dbm: 20.0
      tx_efficiency: 0.8
      rx_efficiency: 0.8
      beam_waist_radius_m: 0.02
      rx_aperture_radius_m: 0.05
      range_km: 10.0
      pointing_bias_urad: 0.0
      attenuation_db_per_km: 0.43   # or: visibility_km: 10.0
      rx_sensitivity_dbm: -38.0
    channel:
      cn2: 1.0e-15
      pointing_jitter_urad: 5.0
    monte_carlo:
      n_samples: 100000
      seed: 1

Error-handling policy (documented requirement R11): all scenario problems
raise ScenarioError with the offending key named; the CLI converts this to
a clean non-zero exit, never a traceback.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .budget import LinkParams, kim_attenuation_db_per_km
from .channel import ChannelParams, MonteCarloResult, sample_received_power_dbm
from .stats import (
    analytic_fade_probability_lognormal,
    fade_probability,
    margin_moments,
    margin_percentiles,
)

_ALLOWED_TOP = {"name", "link", "channel", "monte_carlo"}
_ALLOWED_LINK = {
    "wavelength_nm",
    "tx_power_dbm",
    "tx_efficiency",
    "rx_efficiency",
    "beam_waist_radius_m",
    "rx_aperture_radius_m",
    "range_km",
    "pointing_bias_urad",
    "attenuation_db_per_km",
    "visibility_km",
    "rx_sensitivity_dbm",
}
_ALLOWED_CHANNEL = {"cn2", "pointing_jitter_urad"}
_ALLOWED_MC = {"n_samples", "seed"}


class ScenarioError(ValueError):
    """Raised for any invalid, missing, or unparsable scenario content."""


@dataclass(frozen=True)
class Scenario:
    """Validated scenario: link + channel + Monte Carlo settings."""

    name: str
    link: LinkParams
    channel: ChannelParams
    n_samples: int
    seed: int


def _num(section: str, mapping: dict[str, Any], key: str, default: float) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScenarioError(f"{section}.{key} must be a number, got {value!r}")
    if not math.isfinite(float(value)):
        raise ScenarioError(f"{section}.{key} must be finite, got {value!r}")
    return float(value)


def scenario_from_dict(data: Any, name_hint: str = "scenario") -> Scenario:
    """Build a validated Scenario from a parsed YAML mapping.

    Raises ScenarioError naming the offending key for unknown keys, wrong
    types, out-of-range physics values, or conflicting attenuation inputs.
    """
    if not isinstance(data, dict):
        raise ScenarioError(f"scenario root must be a mapping, got {type(data).__name__}")
    unknown = set(data) - _ALLOWED_TOP
    if unknown:
        raise ScenarioError(f"unknown top-level key(s): {sorted(unknown)}")
    link_raw = data.get("link")
    if not isinstance(link_raw, dict):
        raise ScenarioError("scenario must contain a 'link' mapping")
    chan_raw = data.get("channel", {})
    if not isinstance(chan_raw, dict):
        raise ScenarioError("'channel' must be a mapping when present")
    mc_raw = data.get("monte_carlo", {})
    if not isinstance(mc_raw, dict):
        raise ScenarioError("'monte_carlo' must be a mapping when present")
    for section, raw, allowed in (
        ("link", link_raw, _ALLOWED_LINK),
        ("channel", chan_raw, _ALLOWED_CHANNEL),
        ("monte_carlo", mc_raw, _ALLOWED_MC),
    ):
        unknown = set(raw) - allowed
        if unknown:
            raise ScenarioError(f"unknown key(s) in '{section}': {sorted(unknown)}")

    if "attenuation_db_per_km" in link_raw and "visibility_km" in link_raw:
        raise ScenarioError(
            "link: specify either 'attenuation_db_per_km' or 'visibility_km', not both"
        )
    wavelength_m = _num("link", link_raw, "wavelength_nm", 1550.0) * 1e-9
    if "visibility_km" in link_raw:
        try:
            att = kim_attenuation_db_per_km(
                _num("link", link_raw, "visibility_km", 10.0), wavelength_m
            )
        except (ValueError, TypeError) as exc:
            raise ScenarioError(f"link.visibility_km: {exc}") from exc
    else:
        att = _num("link", link_raw, "attenuation_db_per_km", 0.43)

    try:
        link = LinkParams(
            wavelength_m=wavelength_m,
            tx_power_dbm=_num("link", link_raw, "tx_power_dbm", 20.0),
            tx_efficiency=_num("link", link_raw, "tx_efficiency", 0.8),
            rx_efficiency=_num("link", link_raw, "rx_efficiency", 0.8),
            beam_waist_radius_m=_num("link", link_raw, "beam_waist_radius_m", 0.02),
            rx_aperture_radius_m=_num("link", link_raw, "rx_aperture_radius_m", 0.05),
            range_m=_num("link", link_raw, "range_km", 1.0) * 1000.0,
            pointing_bias_rad=_num("link", link_raw, "pointing_bias_urad", 0.0) * 1e-6,
            attenuation_db_per_km=att,
            rx_sensitivity_dbm=_num("link", link_raw, "rx_sensitivity_dbm", -30.0),
        )
        channel = ChannelParams(
            cn2=_num("channel", chan_raw, "cn2", 1e-15),
            pointing_jitter_rad=_num("channel", chan_raw, "pointing_jitter_urad", 0.0) * 1e-6,
        )
    except (ValueError, TypeError) as exc:
        raise ScenarioError(str(exc)) from exc

    n_samples = mc_raw.get("n_samples", 100_000)
    seed = mc_raw.get("seed", 0)
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ScenarioError(f"monte_carlo.n_samples must be a positive integer, got {n_samples!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ScenarioError(f"monte_carlo.seed must be a non-negative integer, got {seed!r}")

    name = data.get("name", name_hint)
    if not isinstance(name, str) or not name.strip():
        raise ScenarioError(f"'name' must be a non-empty string, got {name!r}")
    return Scenario(name=name.strip(), link=link, channel=channel, n_samples=n_samples, seed=seed)


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate a YAML scenario file (raises ScenarioError on any problem)."""
    path = Path(path)
    if not path.exists():
        raise ScenarioError(f"scenario file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ScenarioError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        raise ScenarioError(f"scenario file is empty: {path}")
    return scenario_from_dict(data, name_hint=path.stem)


def run_twin(scenario: Scenario, surrogate: Any | None = None) -> dict[str, Any]:
    """Run the full digital twin for a scenario and return a report dict.

    Contents: deterministic budget, channel statistics with weak-regime
    validity flag, Monte Carlo fade statistics with Wilson 95 % CI, the
    analytic scintillation-only baseline, and (if a surrogate is supplied
    and fitted) the ML fade prediction with ensemble-spread uncertainty.
    """
    mc: MonteCarloResult = sample_received_power_dbm(
        scenario.link, scenario.channel, n_samples=scenario.n_samples, seed=scenario.seed
    )
    est = fade_probability(mc.samples_dbm, scenario.link.rx_sensitivity_dbm)
    report: dict[str, Any] = {
        "name": scenario.name,
        "beamtwin_version": _version(),
        "budget": mc.budget.as_dict(),
        "channel": {
            "cn2": scenario.channel.cn2,
            "pointing_jitter_rad": scenario.channel.pointing_jitter_rad,
            "rytov_variance": mc.model.rytov_variance,
            "scintillation_index": mc.model.scintillation_index,
            "sigma_ln": mc.model.sigma_ln,
            "sigma_disp_m": mc.model.sigma_disp_m,
            "weak_regime_valid": mc.model.weak_regime_valid,
        },
        "monte_carlo": {
            "n_samples": scenario.n_samples,
            "seed": scenario.seed,
            "fade_probability": est.probability,
            "fade_ci95_low": est.ci_low,
            "fade_ci95_high": est.ci_high,
            "n_fades": est.n_fades,
            "margin_percentiles_db": margin_percentiles(
                mc.samples_dbm, scenario.link.rx_sensitivity_dbm
            ),
            "margin_moments": margin_moments(mc.samples_dbm, scenario.link.rx_sensitivity_dbm),
        },
        "analytic_baseline": {
            "fade_probability_scintillation_only": analytic_fade_probability_lognormal(
                mc.budget.margin_db, mc.model.sigma_ln
            ),
            "note": "closed-form lognormal, scintillation only (no jitter)",
        },
    }
    if surrogate is not None and getattr(surrogate, "is_fitted", False):
        pred = surrogate.predict(scenario.link, scenario.channel)
        report["surrogate"] = {
            "fade_probability": pred.probability,
            "p_low": pred.p_low,
            "p_high": pred.p_high,
            "log10_std": pred.log10_std,
            "extrapolating": pred.extrapolating,
            "note": "GradientBoosting ensemble; interval = +/-2 ensemble std in log10 space",
        }
    else:
        report["surrogate"] = None
    return report


def format_report_text(report: dict[str, Any]) -> str:
    """Render a report dict as a human-readable text table."""
    b = report["budget"]
    c = report["channel"]
    m = report["monte_carlo"]
    lines = [
        f"BeamTwin report — {report['name']} (beamtwin {report['beamtwin_version']})",
        "=" * 64,
        "LINK BUDGET (deterministic)",
        f"  Tx power              {b['tx_power_dbm']:+9.2f} dBm",
        f"  Tx optics loss        -{b['tx_optics_loss_db']:8.2f} dB",
        f"  Geometric loss        -{b['geometric_loss_db']:8.2f} dB",
        f"  Pointing loss (bias)  -{b['pointing_loss_db']:8.2f} dB",
        f"  Atmospheric loss      -{b['atmospheric_loss_db']:8.2f} dB",
        f"  Rx optics loss        -{b['rx_optics_loss_db']:8.2f} dB",
        f"  Received power        {b['received_power_dbm']:+9.2f} dBm",
        f"  Rx sensitivity        {b['rx_sensitivity_dbm']:+9.2f} dBm",
        f"  Margin                {b['margin_db']:+9.2f} dB"
        + ("   ** NEGATIVE MARGIN — LINK FAILS DETERMINISTICALLY **" if b["margin_negative"] else ""),
        f"  Beam radius at Rx     {b['beam_radius_at_rx_m']:9.3f} m"
        f"   (divergence {b['divergence_half_angle_rad'] * 1e6:.1f} urad half-angle)",
        "",
        "CHANNEL (stochastic model)",
        f"  Cn2                   {c['cn2']:.3e} m^-2/3",
        f"  Rytov variance        {c['rytov_variance']:.4f}"
        + ("" if c["weak_regime_valid"] else "   ** sigma_R^2 >= 1: weak-fluctuation model INVALID **"),
        f"  Scintillation index   {c['scintillation_index']:.4f}",
        f"  Pointing jitter       {c['pointing_jitter_rad'] * 1e6:.2f} urad/axis"
        f"   (displacement sigma {c['sigma_disp_m']:.3f} m)",
        "",
        f"MONTE CARLO (n={m['n_samples']}, seed={m['seed']})",
        f"  Fade probability      {m['fade_probability']:.4e}"
        f"   [95% CI {m['fade_ci95_low']:.3e}, {m['fade_ci95_high']:.3e}]",
        f"  Analytic baseline     {report['analytic_baseline']['fade_probability_scintillation_only']:.4e}"
        "   (lognormal, scintillation-only)",
        "  Margin percentiles [dB]: "
        + ", ".join(f"{k}={v:+.2f}" for k, v in m["margin_percentiles_db"].items()),
        f"  Margin mean/std       {m['margin_moments']['mean_db']:+.2f} / "
        f"{m['margin_moments']['std_db']:.2f} dB",
    ]
    s = report.get("surrogate")
    if s is not None:
        lines += [
            "",
            "SURROGATE (ML, GradientBoosting ensemble)",
            f"  Fade probability      {s['fade_probability']:.4e}"
            f"   [spread {s['p_low']:.3e}, {s['p_high']:.3e}]"
            + ("   ** EXTRAPOLATING outside training domain **" if s["extrapolating"] else ""),
        ]
    else:
        lines += ["", "SURROGATE: not available (train with scripts/train_surrogate.py)"]
    lines += [
        "",
        "Research-grade MVP — not certified for operational use.",
    ]
    return "\n".join(lines)


def report_to_json(report: dict[str, Any]) -> str:
    """Serialise a report dict to pretty-printed JSON."""
    return json.dumps(report, indent=2, sort_keys=True)


def _version() -> str:
    from . import __version__

    return __version__


__all__ = [
    "Scenario",
    "ScenarioError",
    "format_report_text",
    "load_scenario",
    "report_to_json",
    "run_twin",
    "scenario_from_dict",
]
