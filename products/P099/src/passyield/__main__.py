"""Command-line interface: ``python -m passyield``.

Three subcommands, each printing through :mod:`passyield.report` so that a
number has one format everywhere:

- ``decompose`` -- one pass's shortfall and its four terms, from the
  committed record.
- ``distribution`` -- the headline output: the analytic shortfall law of one
  pass with its quantiles, and the Monte Carlo cross-check beside it.
- ``accounted`` -- baseline 3 over the whole committed record, with the
  pass-level bootstrap intervals.

Every path is resolved against the installed package, never against the
caller's working directory, so the interface works from anywhere.
"""

from __future__ import annotations

import argparse
import sys

from .baselines import (
    accounted_yield,
    analytic_shortfall_distribution,
    backlog_dry_time_s,
    booked_shortfall_mbit,
    fit_event_model,
    monte_carlo_shortfall_distribution,
)
from .dataset import (
    DEFAULT_RECORD_PATH,
    events_of,
    load_passes,
    parse_intervals,
    station_by_name,
    window_of,
)
from .decompose import decompose_pass
from .geometry import satellite
from .report import format_accounted, format_decomposition, format_distribution

#: Bootstrap seed for the command-line interface. Declared, so two runs of the
#: same command print the same interval.
CLI_SEED = 99099


def _fit(rows: list) -> object:
    """Fit the event model over a whole record."""
    return fit_event_model(
        acquisition_s=[row.acquisition_s for row in rows],
        outage_counts=[row.outage_count for row in rows],
        window_durations_s=[row.duration_s for row in rows],
        outage_durations_s=[
            duration for row in rows for _start, duration in parse_intervals(row.resync_intervals)
        ],
        idle_totals_s=[row.idle_total_s for row in rows],
    )


def build_parser() -> argparse.ArgumentParser:
    """The argument parser. Separated so a test can exercise it directly."""
    parser = argparse.ArgumentParser(
        prog="python -m passyield",
        description=(
            "Realised versus booked data yield of a ground-station contact. "
            "Research-grade software: not flight-qualified, not certified and not approved "
            "for operational aerospace use."
        ),
    )
    parser.add_argument(
        "--record", default=None, help="per-pass record (default: the committed one)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    one = subparsers.add_parser("decompose", help="decompose one pass from the record")
    one.add_argument("--pass-id", type=int, default=0, help="pass_id to decompose (default 0)")

    law = subparsers.add_parser("distribution", help="the analytic shortfall law of one pass")
    law.add_argument("--pass-id", type=int, default=0, help="pass_id (default 0)")
    law.add_argument(
        "--monte-carlo-draws", type=int, default=50000,
        help="draws for the cross-check (default 50000)",
    )
    law.add_argument(
        "--backlog-aware", action="store_true",
        help="use the strengthened baseline 2 variant with the deterministic dry time",
    )

    whole = subparsers.add_parser("accounted", help="baseline 3 over the whole record")
    whole.add_argument("--resamples", type=int, default=2000, help="bootstrap resamples")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit status."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        rows = load_passes(arguments.record)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    index = {row.pass_id: row for row in rows}
    satrec = satellite()

    if arguments.command == "accounted":
        decompositions = [
            decompose_pass(satrec, station_by_name(row.station), window_of(row), events_of(row))
            for row in rows
        ]
        print(format_accounted(accounted_yield(decompositions, seed=CLI_SEED,
                                               resamples=arguments.resamples)))
        return 0

    row = index.get(arguments.pass_id)
    if row is None:
        print(
            f"error: no pass with pass_id {arguments.pass_id} in the record; "
            f"it holds pass_id 0 to {max(index)}",
            file=sys.stderr,
        )
        return 2
    station = station_by_name(row.station)
    window = window_of(row)
    decomposition = decompose_pass(satrec, station, window, events_of(row))

    if arguments.command == "decompose":
        print(format_decomposition(decomposition))
        print()
        print(f"baseline 1 predicts a shortfall of {booked_shortfall_mbit():.1f} Mbit")
        return 0

    model = _fit(rows)
    dry = (
        backlog_dry_time_s(satrec, station, window, row.backlog_mbit)
        if arguments.backlog_aware
        else None
    )
    analytic = analytic_shortfall_distribution(
        satrec, station, window, model,
        decomposition.nominal_bits, decomposition.capacity_bits,
        deterministic_idle_s=dry,
    )
    cross = monte_carlo_shortfall_distribution(
        satrec, station, window, model,
        decomposition.nominal_bits, decomposition.capacity_bits,
        seed=CLI_SEED, draws=arguments.monte_carlo_draws,
    )
    print(f"pass_id {row.pass_id}, station {row.station}")
    if dry is not None:
        print(f"strengthened variant: deterministic dry time {dry:.3f} s")
    print()
    print(format_distribution(analytic))
    print()
    print(format_distribution(cross))
    print()
    print(f"accounted shortfall of this pass, Mbit: {decomposition.shortfall_bits:.3f}")
    source = DEFAULT_RECORD_PATH.name if arguments.record is None else arguments.record
    print(f"record read from: {source}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
