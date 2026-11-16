"""Command-line interface: ``python -m hiltimingdiff <command>``.

Five commands, each printing figures that trace to a module in this package:

``headline``
    The five SPEC §5 outputs with their pass-unit intervals and both
    pre-registration verdicts. Measures simulation mode first, so it takes as
    long as the pass count asks for.
``baselines``
    The three SPEC §4 baselines, in specification order.
``drift``
    The fitted board drift, the raw and drift-adjusted divergence side by side,
    and the confounding check on randomised cell order.
``clock``
    The measured frequency co-variation SPEC §6 requires.
``gate``
    The gate branch census and the reachability verdict of SPEC §9.

``--traces`` points at a different campaign directory; ``--passes`` sets the
simulation-mode pass count; ``--resamples`` sets the bootstrap resamples.
"""

from __future__ import annotations

import argparse
import sys

from .bootstrap import RESAMPLES
from .divergence import pass_units
from .drift import confounding, drift_comparison
from .frequency import frequency_covariation
from .gate import gate_census
from .report import (
    RESEARCH_GRADE,
    SCOPE_LINE,
    build_headline,
    headline_lines,
    spread_verdict,
    transfer_verdict,
)
from .simulate import DEFAULT_PASSES, calibrate_ns_per_op, measure_simulation
from .sources import verify_source_identity
from .traces import TraceDataAbsentError, load_campaign


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hiltimingdiff",
        description=(
            "Per-stage timing divergence between the simulation-mode and "
            "hardware-mode execution of one PAT pipeline. " + RESEARCH_GRADE
        ),
    )
    parser.add_argument(
        "command",
        choices=("headline", "baselines", "drift", "clock", "gate"),
        help="which report to print",
    )
    parser.add_argument(
        "--traces",
        default=None,
        help="measured campaign directory (default: data/traces under the product root)",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=DEFAULT_PASSES,
        help=f"simulation-mode passes per cell (default: {DEFAULT_PASSES})",
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=RESAMPLES,
        help=f"bootstrap resamples (default: {RESAMPLES})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not write simulation-mode progress to stderr",
    )
    return parser


def _headline(args: argparse.Namespace) -> list[str]:
    headline = build_headline(
        passes=args.passes,
        resamples=args.resamples,
        traces_directory=args.traces,
        progress=not args.quiet,
    )
    lines = headline_lines(headline)
    lines += [
        "",
        f"simulation mode measured here: {headline.simulation.machine}, "
        f"python {headline.simulation.python}, numpy {headline.simulation.numpy}, "
        f"{headline.simulation.passes} passes x {headline.simulation.frames} frames "
        f"x {len(headline.campaign.runs)} cells = "
        f"{headline.simulation.n_timed_frames} timed frames in "
        f"{headline.simulation.elapsed_s:.1f} s",
        f"hardware mode read from {headline.campaign.label}/: "
        f"{headline.n_hardware_frames} timed frames",
    ]
    return lines


def _baselines(args: argparse.Namespace) -> list[str]:
    campaign = load_campaign(args.traces)
    units = pass_units(campaign)
    simulation = measure_simulation(
        campaign, passes=args.passes, progress=not args.quiet
    )
    ns_per_op = calibrate_ns_per_op(simulation)
    from .baselines import baseline_1, baseline_2, baseline_3

    first = baseline_1(units, ns_per_op)
    second = baseline_2(units, simulation, resamples=args.resamples)
    third = baseline_3(units, resamples=args.resamples)
    lines = [
        "BASELINES -- in SPEC §4 order; the classical work is complete before any fitting",
        SCOPE_LINE,
        "",
        f"BASELINE 1 -- the declared timing model (ns_per_op = {ns_per_op:.6f} ns/op)",
        first.frame().to_string(index=False),
        f"  mean relative error over stages : {first.mean_relative_error:.6f}",
        f"  worst stage                     : {first.worst_stage} "
        f"({first.max_relative_error:.6f})",
        f"  whole-pipeline total error      : {first.total_relative_error:.6f}",
        "",
        "BASELINE 2 -- one scalar fitted on training stages, scored on held-out stages",
        f"  training stages : {', '.join(second.declared.train)}",
        f"  held-out stages : {', '.join(second.declared.holdout)}",
        f"  k               : {second.declared.k:.6f}",
        f"  held-out mean relative error : {second.declared_interval.format(6)}",
        f"  leave-one-stage-out mean     : {second.leave_one_out_mean:.6f}",
        f"  all 20 three-and-three splits: min {second.split_error_min:.6f}, "
        f"median {second.split_error_median:.6f}, max {second.split_error_max:.6f}; "
        f"{second.splits_above_preregistration} of {len(second.split_errors)} above 0.5",
        "",
        f"BASELINE 3 -- ground truth, {third.unit}-unit {third.confidence:.0%} interval, "
        f"{third.n_units} units",
        third.frame().to_string(index=False),
        "",
        spread_verdict(
            drift_comparison(campaign, units, ns_per_op, resamples=args.resamples).raw
        ).line(),
        transfer_verdict(second).line(),
        "",
        RESEARCH_GRADE,
    ]
    return lines


def _drift(args: argparse.Namespace) -> list[str]:
    campaign = load_campaign(args.traces)
    units = pass_units(campaign)
    simulation = measure_simulation(
        campaign, passes=args.passes, progress=not args.quiet
    )
    ns_per_op = calibrate_ns_per_op(simulation)
    comparison = drift_comparison(campaign, units, ns_per_op, resamples=args.resamples)
    confound = confounding(campaign, units)
    drift = comparison.drift
    return [
        "BOARD DRIFT -- fitted over eleven calibration blocks, not bracketed",
        SCOPE_LINE,
        "",
        drift.blocks.to_string(index=False),
        "",
        f"  bracket ratio end/start        : {drift.bracket_ratio:.16f}",
        f"  meta.json drift ratio          : {drift.meta_ratio:.16f}",
        f"  fitted ratio over {drift.campaign_elapsed_s:.1f} s      : {drift.fitted_ratio:.16f}",
        f"  straight-line fit r_squared    : {drift.fit.r_squared:.6f}",
        f"  slope                          : {drift.fit.slope:.6f} ns per second",
        f"  adjacent blocks increasing     : {drift.monotonic_blocks} of 10",
        f"  largest single block step      : {drift.max_block_step_ratio:.6f}",
        "",
        "RAW MEASURED (headline) beside the DRIFT-ADJUSTED variant",
        comparison.frame().to_string(index=False),
        f"  raw measured spread   : {comparison.raw.spread:.6f}",
        f"  drift-adjusted spread : {comparison.adjusted.spread:.6f}",
        f"  relative shift        : {comparison.spread_relative_shift:+.6f}",
        f"  largest per-stage shift in R : {comparison.max_abs_ratio_shift:.6f}",
        "",
        "CONFOUNDING CHECK -- cell median total frame time regressed separately",
        f"  on order_index : slope {confound.on_order.slope:.1f} ns/position, "
        f"r_squared {confound.on_order.r_squared:.6f}",
        f"  on W           : slope {confound.on_width.slope:.1f} ns/pixel, "
        f"r_squared {confound.on_width.r_squared:.6f}",
        f"  corr(order_index, W) : Pearson {confound.order_width_correlation:.6f}, "
        f"Spearman {confound.order_width_spearman:.6f}",
        "",
        RESEARCH_GRADE,
    ]


def _clock(args: argparse.Namespace) -> list[str]:
    campaign = load_campaign(args.traces)
    units = pass_units(campaign)
    covariation = frequency_covariation(campaign, units)
    return [
        "UNPINNED CLOCK -- measured co-variation, SPEC §6",
        SCOPE_LINE,
        "",
        f"  governor            : {covariation.governor}",
        f"  power mode          : {covariation.power_mode}",
        f"  passwordless sudo   : {covariation.sudo_passwordless}",
        f"  per-core frequency  : {covariation.frequency_min_khz:.0f} to "
        f"{covariation.frequency_max_khz:.0f} kHz, mean "
        f"{covariation.frequency_mean_khz:.1f} kHz, relative spread "
        f"{covariation.frequency_relative_spread:.6f}",
        f"  paired points       : {len(covariation.paired)} "
        "(6 within-cell env samples x 9 cells)",
        "",
        covariation.frame().to_string(index=False),
        "",
        f"  largest within-cell variance share : {covariation.max_within_cell_share:.6f} "
        f"({covariation.max_within_cell_stage})",
        f"  largest pooled variance share      : {covariation.max_pooled_share:.6f}",
        "",
        RESEARCH_GRADE,
    ]


def _gate(args: argparse.Namespace) -> list[str]:
    campaign = load_campaign(args.traces)
    census = gate_census(campaign)
    silent = census.silent_branches
    return [
        "GATE BRANCH REACHABILITY -- SPEC §9",
        SCOPE_LINE,
        "",
        census.frame().to_string(index=False),
        "",
        f"  every branch fired at least once : {census.all_branches_fired}",
        f"  silent branches                  : "
        f"{', '.join(silent) if silent else 'none'}",
        f"  branches firing in EVERY cell    : "
        f"{', '.join(census.branches_firing_in_every_cell)}",
        f"  per-frame acceptance rate        : {census.acceptance_rate:.6f} "
        f"({census.totals['accept']} of {census.n_frames_total} frames)",
        "",
        "Candidate-level and frame-level counts are not a partition of the frames: "
        "low_score, area_out and border are counted per candidate, accept and "
        "no_candidate per frame.",
        "",
        RESEARCH_GRADE,
    ]


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit status."""
    args = _parser().parse_args(argv)
    try:
        verify_source_identity()
        handler = {
            "headline": _headline,
            "baselines": _baselines,
            "drift": _drift,
            "clock": _clock,
            "gate": _gate,
        }[args.command]
        for line in handler(args):
            print(line)
    except TraceDataAbsentError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    except (ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
