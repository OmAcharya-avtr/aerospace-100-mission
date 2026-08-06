"""BeamTwin CLI.

Usage:
    python -m beamtwin run scenario.yaml [--json out.json] [--no-surrogate]
    python -m beamtwin sweep scenario.yaml --param range_km
        --start 1 --stop 20 --steps 12 [--output sweep.png] [--json out.json]

Exit codes: 0 success, 2 scenario/usage error (clean message, no traceback).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .scenario import (  # noqa: E402
    Scenario,
    ScenarioError,
    format_report_text,
    load_scenario,
    report_to_json,
    run_twin,
)
from .surrogate import FadeSurrogate, default_model_path  # noqa: E402

_SWEEPABLE = {
    "range_km",
    "tx_power_dbm",
    "rx_sensitivity_dbm",
    "attenuation_db_per_km",
    "cn2",
    "pointing_jitter_urad",
}


def _load_surrogate_or_none() -> FadeSurrogate | None:
    try:
        return FadeSurrogate.load(default_model_path())
    except (FileNotFoundError, ValueError):
        return None


def _apply_param(scenario: Scenario, param: str, value: float) -> Scenario:
    link, channel = scenario.link, scenario.channel
    if param == "range_km":
        link = dataclasses.replace(link, range_m=value * 1000.0)
    elif param == "tx_power_dbm":
        link = dataclasses.replace(link, tx_power_dbm=value)
    elif param == "rx_sensitivity_dbm":
        link = dataclasses.replace(link, rx_sensitivity_dbm=value)
    elif param == "attenuation_db_per_km":
        link = dataclasses.replace(link, attenuation_db_per_km=value)
    elif param == "cn2":
        channel = dataclasses.replace(channel, cn2=value)
    elif param == "pointing_jitter_urad":
        channel = dataclasses.replace(channel, pointing_jitter_rad=value * 1e-6)
    else:  # pragma: no cover - guarded by argparse choices
        raise ScenarioError(f"unsupported sweep parameter: {param}")
    return dataclasses.replace(scenario, link=link, channel=channel)


def _cmd_run(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    surrogate = None if args.no_surrogate else _load_surrogate_or_none()
    report = run_twin(scenario, surrogate=surrogate)
    sys.stdout.write(format_report_text(report) + "\n")
    if args.json:
        Path(args.json).write_text(report_to_json(report), encoding="utf-8")
        sys.stdout.write(f"\nJSON report written to {args.json}\n")
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    if args.steps < 2:
        raise ScenarioError("--steps must be >= 2")
    if args.stop <= args.start and not args.log:
        raise ScenarioError("--stop must be > --start")
    base = load_scenario(args.scenario)
    if args.log:
        import numpy as np

        if args.start <= 0 or args.stop <= 0:
            raise ScenarioError("--log sweeps require positive --start/--stop")
        values = list(np.geomspace(args.start, args.stop, args.steps))
    else:
        step = (args.stop - args.start) / (args.steps - 1)
        values = [args.start + i * step for i in range(args.steps)]

    surrogate = None if args.no_surrogate else _load_surrogate_or_none()
    rows = []
    for v in values:
        try:
            scenario = _apply_param(base, args.param, float(v))
        except (ValueError, TypeError) as exc:
            raise ScenarioError(f"sweep value {v} for {args.param}: {exc}") from exc
        report = run_twin(scenario, surrogate=surrogate)
        rows.append(
            {
                "value": float(v),
                "margin_db": report["budget"]["margin_db"],
                "fade_probability": report["monte_carlo"]["fade_probability"],
                "fade_ci95_low": report["monte_carlo"]["fade_ci95_low"],
                "fade_ci95_high": report["monte_carlo"]["fade_ci95_high"],
                "analytic_baseline": report["analytic_baseline"][
                    "fade_probability_scintillation_only"
                ],
                "surrogate": (
                    report["surrogate"]["fade_probability"] if report["surrogate"] else None
                ),
            }
        )
        sys.stdout.write(
            f"{args.param}={v:.6g}: margin={rows[-1]['margin_db']:+.2f} dB, "
            f"P_fade={rows[-1]['fade_probability']:.3e}\n"
        )

    if args.json:
        Path(args.json).write_text(
            json.dumps({"param": args.param, "rows": rows}, indent=2), encoding="utf-8"
        )
    out = Path(args.output)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [r["value"] for r in rows]
    ax.semilogy(xs, [max(r["fade_probability"], 1e-6) for r in rows], "o-", label="Monte Carlo")
    ax.semilogy(
        xs,
        [max(r["analytic_baseline"], 1e-6) for r in rows],
        "s--",
        label="analytic (scint-only)",
    )
    if any(r["surrogate"] is not None for r in rows):
        ax.semilogy(
            xs,
            [max(r["surrogate"], 1e-6) if r["surrogate"] else float("nan") for r in rows],
            "^:",
            label="ML surrogate",
        )
    if args.log:
        ax.set_xscale("log")
    ax.set_xlabel(args.param)
    ax.set_ylabel("fade probability")
    ax.set_title(f"BeamTwin sweep: {base.name}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    sys.stdout.write(f"Sweep plot written to {out}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m beamtwin",
        description="BeamTwin: FSO link digital twin (research-grade MVP, not certified).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the full twin for a YAML scenario")
    p_run.add_argument("scenario", help="path to scenario YAML file")
    p_run.add_argument("--json", help="also write the report as JSON to this path")
    p_run.add_argument(
        "--no-surrogate", action="store_true", help="skip the ML surrogate prediction"
    )
    p_run.set_defaults(func=_cmd_run)

    p_sweep = sub.add_parser("sweep", help="sweep one parameter and plot fade probability")
    p_sweep.add_argument("scenario", help="path to base scenario YAML file")
    p_sweep.add_argument("--param", required=True, choices=sorted(_SWEEPABLE))
    p_sweep.add_argument("--start", type=float, required=True)
    p_sweep.add_argument("--stop", type=float, required=True)
    p_sweep.add_argument("--steps", type=int, default=10)
    p_sweep.add_argument("--log", action="store_true", help="logarithmic sweep spacing")
    p_sweep.add_argument("--output", default="sweep.png", help="PNG output path")
    p_sweep.add_argument("--json", help="also write sweep rows as JSON")
    p_sweep.add_argument("--no-surrogate", action="store_true")
    p_sweep.set_defaults(func=_cmd_sweep)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ScenarioError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2
    except (ValueError, TypeError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
