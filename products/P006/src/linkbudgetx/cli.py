"""Command-line interface: ``python -m linkbudgetx --config example.yaml``.

The YAML config maps 1:1 onto :class:`linkbudgetx.LinkBudget` fields, plus an
optional ``uncertainties`` block of 1-sigma values:

.. code-block:: yaml

    tx_power_dbm: 20.0
    wavelength_nm: 1550.0
    beam_divergence_rad: 1.0e-3   # FULL angle
    range_km: 10.0
    rx_aperture_diameter_m: 0.1
    rx_sensitivity_dbm: -40.0
    tx_optics_efficiency: 0.8
    rx_optics_efficiency: 0.8
    pointing_error_rad: 0.25e-3
    atmos_attenuation_db_per_km: 0.5
    beam_profile: gaussian
    uncertainties:
      tx_power_dbm: 0.5
      atmos_attenuation_db_per_km: 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import yaml

from .core import LinkBudget
from .uncertainty import propagate_margin_sigma


def _load_config(path: Path) -> tuple[LinkBudget, dict[str, float]]:
    """Parse YAML into a validated LinkBudget and an uncertainties dict."""
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise ValueError(f"Config file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise ValueError(f"Config file {path} is not valid YAML: {exc}") from None
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping.")
    sigmas = raw.pop("uncertainties", {}) or {}
    if not isinstance(sigmas, dict):
        raise ValueError("'uncertainties' must be a mapping of field -> 1-sigma value.")
    known = {f.name for f in fields(LinkBudget)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys {sorted(unknown)}; valid keys: {sorted(known)}.")
    return LinkBudget(**raw), {str(k): float(v) for k, v in sigmas.items()}


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser (exposed for testing)."""
    parser = argparse.ArgumentParser(
        prog="linkbudgetx",
        description="Deterministic free-space optical link-budget calculator "
        "(educational, validation level 1).",
    )
    parser.add_argument("--config", required=True, type=Path, help="YAML config file.")
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON instead of a table."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns process exit code (0 ok, 2 bad input)."""
    args = build_parser().parse_args(argv)
    try:
        budget, sigmas = _load_config(args.config)
        result = budget.compute()
        unc = propagate_margin_sigma(budget, sigmas) if sigmas else None
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        payload: dict[str, object] = {"budget": result.as_dict()}
        if unc is not None:
            payload["uncertainty"] = {
                "sigma_margin_db": unc.sigma_margin_db,
                "partials": unc.partials,
                "contributions_db": unc.contributions_db,
            }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(result.format_table())
        if unc is not None:
            print(f"\nMargin 1-sigma (first-order propagation): {unc.sigma_margin_db:.3f} dB")
            for name, contrib in sorted(
                unc.contributions_db.items(), key=lambda kv: -kv[1]
            ):
                print(f"  {name:<30} +/- {contrib:.3f} dB")
    return 0
