"""Command line: ``python -m relfiledelivery <command>``.

Commands
--------
``info``          Package identity, the three modes and the modelling choices.
``analytic``      Closed-form prediction for a declared scenario.
``simulate``      One seeded run, with the completion invariant verified.
``distribution``  A Monte Carlo ensemble, as quantiles with the tail apart.
``compare``       Every mode on the same scenario, side by side.
``benchmark``     Fit and score the predictors. Minutes, not seconds.

Every command takes the scenario on the command line, because a measurement
harness whose inputs live in a configuration file is a harness whose inputs
nobody checks.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from .analytic import analytic_delivery, require_closed_form_applicable
from .link import LinkConditions, OutageModel, uniform_windows
from .modes import ALL_MODES, TransferMode, parse_mode
from .montecarlo import run_ensemble, standard_error_of_mean
from .payload import FileTransferSpec
from .report import (
    analytic_report,
    closed_form_agreement,
    distribution_report,
    mode_comparison,
    rule,
    scenario_report,
    simulation_report,
)
from .simulate import simulate_transfer


def _add_scenario_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--file-bytes", type=int, default=262_144, help="source size, bytes")
    parser.add_argument(
        "--pdu-payload-bytes", type=int, default=1024, help="payload per PDU, bytes"
    )
    parser.add_argument(
        "--pdu-header-bytes", type=int, default=12, help="per-PDU wire overhead, bytes"
    )
    parser.add_argument(
        "--forward-rate-bps", type=float, default=1.0e6, help="forward bit rate, bit/s"
    )
    parser.add_argument(
        "--return-rate-bps", type=float, default=1.0e4, help="return bit rate, bit/s"
    )
    parser.add_argument(
        "--light-time-s", type=float, default=0.25, help="one-way light time, seconds"
    )
    parser.add_argument(
        "--loss", type=float, default=0.05, help="independent PDU loss probability in [0, 1)"
    )
    parser.add_argument(
        "--mean-up-s", type=float, default=None, help="mean outage-free interval, seconds"
    )
    parser.add_argument(
        "--mean-down-s", type=float, default=None, help="mean outage duration, seconds"
    )
    parser.add_argument(
        "--windows", type=int, default=0, help="number of contact windows; 0 means continuous"
    )
    parser.add_argument(
        "--window-duration-s", type=float, default=600.0, help="contact window length, seconds"
    )
    parser.add_argument(
        "--window-period-s", type=float, default=5400.0, help="window repeat period, seconds"
    )


def _scenario(args: argparse.Namespace) -> tuple[FileTransferSpec, LinkConditions]:
    spec = FileTransferSpec(args.file_bytes, args.pdu_payload_bytes, args.pdu_header_bytes)
    if (args.mean_up_s is None) != (args.mean_down_s is None):
        raise ValueError("--mean-up-s and --mean-down-s must be given together")
    outage = (
        OutageModel.none()
        if args.mean_up_s is None
        else OutageModel(mean_up_s=args.mean_up_s, mean_down_s=args.mean_down_s)
    )
    windows = (
        ()
        if args.windows <= 0
        else uniform_windows(args.windows, args.window_duration_s, args.window_period_s)
    )
    link = LinkConditions(
        forward_rate_bps=args.forward_rate_bps,
        return_rate_bps=args.return_rate_bps,
        one_way_light_time_s=args.light_time_s,
        pdu_loss_probability=args.loss,
        outage=outage,
        windows=windows,
    )
    return spec, link


def build_parser() -> argparse.ArgumentParser:
    """The argument parser. Separated so the tests can exercise it directly."""
    parser = argparse.ArgumentParser(
        prog="relfiledelivery",
        description=(
            "Measure what a file transfer costs over a declared intermittent link. "
            "Not a CFDP implementation: see cfdp-py and spacepackets for the protocol."
        ),
    )
    parser.add_argument("--version", action="version", version=f"relfiledelivery {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="package identity and declared modelling choices")

    analytic = sub.add_parser("analytic", help="closed-form prediction")
    _add_scenario_arguments(analytic)
    analytic.add_argument("--mode", default="deferred_nak")

    simulate = sub.add_parser("simulate", help="one seeded run")
    _add_scenario_arguments(simulate)
    simulate.add_argument("--mode", default="deferred_nak")
    simulate.add_argument("--seed", type=int, default=20261024)

    distribution = sub.add_parser("distribution", help="Monte Carlo quantiles")
    _add_scenario_arguments(distribution)
    distribution.add_argument("--mode", default="deferred_nak")
    distribution.add_argument("--seed", type=int, default=20261024)
    distribution.add_argument("--runs", type=int, default=500)

    compare = sub.add_parser("compare", help="every mode on one scenario")
    _add_scenario_arguments(compare)
    compare.add_argument("--seed", type=int, default=20261024)
    compare.add_argument("--runs", type=int, default=300)

    benchmark = sub.add_parser("benchmark", help="fit and score the predictors")
    benchmark.add_argument("--train-rows", type=int, default=1200)
    benchmark.add_argument("--test-rows", type=int, default=500)
    benchmark.add_argument("--seed", type=int, default=100_000)
    return parser


def _info_lines() -> list[str]:
    return [
        f"relfiledelivery {__version__}",
        "",
        "What a file transfer costs over an intermittent link, measured against a",
        "declared link model. A measurement harness, not a protocol.",
        "",
        "This is NOT a CFDP implementation. For the CCSDS File Delivery Protocol and",
        "its packet formats use cfdp-py and spacepackets.",
        "",
        "modes:",
        *(f"  {mode.value}" for mode in ALL_MODES),
        "",
        "declared modelling choices (see relfiledelivery.simulate):",
        "  M1 a PDU is never split across a contact boundary",
        "  M2 a PDU whose transmission intersects an outage is lost in full",
        "  M3 the sender transmits during an outage",
        "  M4 control PDUs are never lost; no NAK timer is modelled",
        "  M5 immediate-NAK reveals a gap at the next successful arrival",
        "  M6 delivery time is when the receiver holds the file",
        "  M7 the return link is gated by the same contact windows",
        "",
        "Research grade. Not flight-qualified, not certified, not approved for",
        "operational aerospace use.",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    lines: list[str] = []

    if args.command == "info":
        lines = _info_lines()
    elif args.command == "analytic":
        spec, link = _scenario(args)
        mode = parse_mode(args.mode)
        require_closed_form_applicable(link)
        prediction = analytic_delivery(spec, link, mode)
        lines = [*scenario_report(spec, link), *analytic_report(prediction)]
    elif args.command == "simulate":
        spec, link = _scenario(args)
        mode = parse_mode(args.mode)
        result = simulate_transfer(spec, link, mode, args.seed)
        lines = [*scenario_report(spec, link), *simulation_report(result)]
    elif args.command == "distribution":
        spec, link = _scenario(args)
        mode = parse_mode(args.mode)
        distribution = run_ensemble(spec, link, mode, args.runs, args.seed)
        lines = [*scenario_report(spec, link), *distribution_report(distribution)]
        if (
            link.outage.is_outage_free
            and link.is_continuous
            and mode is TransferMode.DEFERRED_NAK
            and distribution.completed > 1
        ):
            lines.extend(
                closed_form_agreement(
                    analytic_delivery(spec, link, mode),
                    distribution,
                    standard_error_of_mean(distribution.delivery_times_s),
                )
            )
    elif args.command == "compare":
        spec, link = _scenario(args)
        distributions = {
            mode: run_ensemble(spec, link, mode, args.runs, args.seed) for mode in ALL_MODES
        }
        lines = [
            *scenario_report(spec, link),
            *rule("modes compared, quantiles and tail; no mean"),
            *mode_comparison(distributions),
        ]
    elif args.command == "benchmark":
        from .benchmark import render_table, run_benchmark
        from .dataset import fit_split, generate_table
        from .predictors import build_predictors

        train = fit_split(generate_table(args.train_rows, args.seed))
        test = fit_split(generate_table(args.test_rows, args.seed + 500_000))
        rows = run_benchmark(build_predictors(), train, test)
        lines = [
            *rule("predictor benchmark"),
            f"train rows {len(train)}, test rows {len(test)}",
            "",
            *render_table(rows),
        ]
    else:  # pragma: no cover - argparse enforces the choices
        parser.error(f"unknown command {args.command!r}")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    sys.exit(main())
