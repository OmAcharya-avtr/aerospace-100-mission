"""Command-line interface: ``python -m powerenvelope``.

Subcommands, each taking ``--traces`` pointing at a campaign directory:

``manifest``  campaign identity, the INA3221 configuration and the cell table.
``headline``  the understatement factor and its decomposition, both variants.
``report``    floor, all three baselines, the rail-accounting fit, the
              between-run interval, the headline and the learned dataset.
``learned``   the learned predictor against both baselines on both splits.

``--simulate`` replaces ``--traces`` with a freshly written SYNTHETIC campaign
in a temporary directory. Every line of output from that path is prefixed with
a synthetic banner, because a synthetic figure printed where a measured one
belongs would look like a measurement and not be one.

Exit status is 0 on success, 2 on a usage or input error, and 3 when the
requested traces are absent or incomplete. The absent case has its own status
because it is not a defect: the campaign may simply not have written them yet.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from .baselines import (
    VARIANTS,
    between_run_spread,
    duty_cycle_model,
    energy_per_inference,
    floor_power,
    ground_truth,
    rail_accounting_curve,
)
from .decomposition import decompose
from .predictor import (
    ActivePowerPredictor,
    AnalyticActivePowerModel,
    ConstantPowerPredictor,
    score,
)
from .provenance import Origin
from .report import (
    MEASUREMENT_MODEL,
    baseline1_lines,
    baseline2_lines,
    baseline3_lines,
    campaign_lines,
    cell_table,
    floor_lines,
    headline_lines,
    rail_lines,
    score_table,
    spread_lines,
    window_lines,
)
from .schema import SchemaError
from .trace import (
    PERMISSIVE_CELL,
    REFERENCE_CELL,
    Campaign,
    TraceDataAbsentError,
    campaign_is_complete,
    load_campaign,
)
from .windows import extract_windows, grouped_splits

#: Exit status when the requested traces are absent or incomplete.
EXIT_ABSENT = 3

#: Exit status on a usage or input error.
EXIT_USAGE = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m powerenvelope",
        description=(
            "Energy per useful decision from a measured duty-cycled autonomy "
            "pipeline. The headline is the factor by which energy per inference "
            "understates it, decomposed into the rejection factor, the duty-cycle "
            "factor and the nominal-versus-measured active-power error."
        ),
    )
    parser.add_argument(
        "--traces",
        default="data/traces",
        help="campaign directory holding manifest.json and the per-cell CSVs "
        "(default: data/traces)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="write a SYNTHETIC campaign to a temporary directory and use it "
        "instead of --traces. Every figure produced this way is synthetic and is "
        "labelled so.",
    )
    parser.add_argument(
        "--simulate-cycles",
        type=int,
        default=10,
        help="cycles per run for --simulate (default: 10)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("manifest", help="campaign metadata and the cell table")

    headline = subparsers.add_parser("headline", help="the decomposition for one cell")
    headline.add_argument(
        "--cell", default=REFERENCE_CELL, help=f"cell to analyse (default: {REFERENCE_CELL})"
    )
    report = subparsers.add_parser("report", help="every baseline and the headline")
    report.add_argument(
        "--cell", default=REFERENCE_CELL, help=f"cell to analyse (default: {REFERENCE_CELL})"
    )
    learned = subparsers.add_parser("learned", help="learned predictor against both baselines")
    learned.add_argument(
        "--window-s", type=float, default=None, help="window length in seconds"
    )
    return parser


def _load(arguments: argparse.Namespace, emit: Callable[[str], None]) -> Campaign:
    if arguments.simulate:
        from .synthetic import load_synthetic_campaign

        directory = Path(tempfile.mkdtemp(prefix="powerenvelope-simulate-"))
        emit("=" * 78)
        emit("SYNTHETIC CAMPAIGN. Written by src/powerenvelope/synthetic.py in a")
        emit("temporary directory. Nothing below was measured on any board. This path")
        emit("exists to exercise the code, and it is never a fallback for missing")
        emit("measured data: without --simulate, absent traces are a refusal.")
        emit("=" * 78)
        return load_synthetic_campaign(directory, cycles=arguments.simulate_cycles)
    root = Path(arguments.traces)
    if not campaign_is_complete(root):
        raise TraceDataAbsentError(
            f"{arguments.traces}/CAMPAIGN_COMPLETE is absent, and the campaign writes "
            "that marker last of all. No figure is printed from a partial campaign. "
            "Pass --simulate for a SYNTHETIC campaign, which is labelled as one."
        )
    return load_campaign(root, origin=Origin.MEASURED, directory_label=arguments.traces)


def _manifest(campaign: Campaign, emit: Callable[[str], None]) -> int:
    for line in campaign_lines(campaign):
        emit(line)
    emit("")
    for line in cell_table(campaign):
        emit(line)
    return 0


def _headline(campaign: Campaign, cell_label: str, emit: Callable[[str], None]) -> int:
    cell = campaign.require(cell_label)
    emit(MEASUREMENT_MODEL)
    for variant in VARIANTS:
        emit("")
        for line in headline_lines(decompose(cell, variant=variant)):
            emit(line)
    return 0


def _report(campaign: Campaign, cell_label: str, emit: Callable[[str], None]) -> int:
    cell = campaign.require(cell_label)
    emit(MEASUREMENT_MODEL)
    emit("")
    for line in campaign_lines(campaign):
        emit(line)
    emit("")
    for line in cell_table(campaign):
        emit(line)
    emit("")
    emit("-- measured background floor " + "-" * 48)
    floor = floor_power(campaign)
    for line in floor_lines(floor):
        emit(line)
    emit("")
    emit("-- baseline 1, energy per inference (the thing being refuted) " + "-" * 15)
    for variant in VARIANTS:
        for line in baseline1_lines(energy_per_inference(cell, variant=variant)):
            emit(line)
        emit("")
    emit("-- baseline 2, the analytic duty-cycle model " + "-" * 32)
    for line in baseline2_lines(duty_cycle_model(campaign, cell_label, floor=floor)):
        emit(line)
    emit("")
    emit("-- measured rail-accounting ratio against load " + "-" * 30)
    for line in rail_lines(rail_accounting_curve(campaign)):
        emit(line)
    emit("")
    emit("-- baseline 3, ground truth " + "-" * 49)
    for line in baseline3_lines(ground_truth(cell)):
        emit(line)
    emit("")
    emit("-- between-run spread over the reference-cell repeats " + "-" * 23)
    try:
        for line in spread_lines(between_run_spread(campaign)):
            emit(line)
    except (TraceDataAbsentError, ValueError) as error:
        emit(f"NOT AVAILABLE: {error}")
    emit("")
    emit("-- the headline " + "-" * 61)
    for variant in VARIANTS:
        for line in headline_lines(decompose(cell, variant=variant)):
            emit(line)
        emit("")
    if PERMISSIVE_CELL in campaign.cells:
        emit("-- permissive-gate control, where the rejection factor should nearly vanish --")
        for line in headline_lines(decompose(campaign.require(PERMISSIVE_CELL))):
            emit(line)
        emit("")
    emit("-- the learned predictor's dataset " + "-" * 42)
    try:
        for line in window_lines(extract_windows(campaign)):
            emit(line)
    except ValueError as error:
        # A campaign whose runs are shorter than one window has no dataset.
        # That is a property of the campaign, not an input error, so the rest
        # of the report stands and the reason is printed where the table would
        # have been.
        emit(f"NO WINDOW DATASET: {error}")
    return 0


def _learned(
    campaign: Campaign, window_s: float | None, emit: Callable[[str], None]
) -> int:
    windows = extract_windows(campaign) if window_s is None else extract_windows(campaign, window_s)
    for line in window_lines(windows):
        emit(line)
    floor = floor_power(campaign)
    rows: list = []
    for grouping in ("duty", "frame_dim"):
        emit("")
        emit(f"held-out {grouping}, leave-one-group-out")
        grouped: list = []
        for label, train, test in grouped_splits(windows, grouping):
            learned = ActivePowerPredictor().fit(
                windows.features[train], windows.targets[train], windows.feature_names
            )
            analytic = AnalyticActivePowerModel().fit(
                campaign, windows.cells[train], floor_mw=floor.mean_mw
            )
            constant = ConstantPowerPredictor().fit(
                windows.features[train], windows.targets[train]
            )
            truth = windows.targets[test]
            features = windows.features[test]
            grouped.append(score(f"learned forest / {label}", truth, learned.predict(features)))
            grouped.append(
                score(f"analytic duty-cycle / {label}", truth, analytic.predict(features))
            )
            grouped.append(score(f"constant (null) / {label}", truth, constant.predict(features)))
        for line in score_table(grouped):
            emit(line)
        rows.extend(grouped)
    emit("")
    emit("pooled means over both groupings")
    for name in ("learned forest", "analytic duty-cycle", "constant (null)"):
        subset = [row for row in rows if row.label.startswith(name)]
        emit(
            f"  {name:<22} mean MAE {np.mean([row.mae_mw for row in subset]):>10.4f} mW "
            f"over {len(subset)} split(s)"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit status."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)

    try:
        campaign = _load(arguments, emit)
        if arguments.command == "manifest":
            status = _manifest(campaign, emit)
        elif arguments.command == "headline":
            status = _headline(campaign, arguments.cell, emit)
        elif arguments.command == "report":
            status = _report(campaign, arguments.cell, emit)
        elif arguments.command == "learned":
            status = _learned(campaign, arguments.window_s, emit)
        else:  # pragma: no cover -- argparse rejects anything else
            parser.error(f"unknown command {arguments.command!r}")
            status = EXIT_USAGE
    except TraceDataAbsentError as error:
        sys.stdout.write("\n".join(lines) + ("\n" if lines else ""))
        sys.stderr.write(f"TRACES ABSENT: {error}\n")
        return EXIT_ABSENT
    except (SchemaError, ValueError, TypeError) as error:
        sys.stdout.write("\n".join(lines) + ("\n" if lines else ""))
        sys.stderr.write(f"INPUT ERROR: {error}\n")
        return EXIT_USAGE
    sys.stdout.write("\n".join(lines) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
