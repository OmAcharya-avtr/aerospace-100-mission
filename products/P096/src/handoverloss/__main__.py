"""Command-line entry point: ``python -m handoverloss``.

Four subcommands, each printing the thing the matching part of the library
returns and nothing else:

``attribute``
    Baseline 3 on the committed corpus: the three mechanisms, an interval on
    each, and which dominates.
``baselines``
    Baseline 1 and baseline 2 scored on one of the two held-out splits.
``learned``
    The learned component scored on the same split, printed beside the
    baselines so the comparison cannot be read out of context.
``sequence``
    The closed-form decomposition of one handover supplied on the command line,
    with the independently derived bracket on its total.

No library module prints. All output is produced here.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import __version__
from .accounting import attribute
from .baselines import acquisition_only_loss_bits, fluid_decomposition
from .bounds import loss_bracket
from .corpus import (
    CorpusAbsentError,
    load_corpus,
    station_pair_split,
    time_split,
)
from .predictor import score_baselines, score_learned
from .sequence import MECHANISMS, HandoverSequence
from .trace import EventParameters, simulate


def _split_for(name: str, corpus):
    return station_pair_split(corpus) if name == "station_pairs" else time_split(corpus)


def _attribute(arguments: argparse.Namespace) -> int:
    corpus = load_corpus()
    table = corpus.table
    limit = min(int(arguments.events), len(corpus))
    index = np.arange(limit)
    parameters = EventParameters(
        source_rate_bps=table["source_rate_bps"][index],
        downlink_rate_bps=table["downlink_rate_bps"][index],
        buffer_capacity_bits=table["buffer_capacity_bits"][index],
        initial_occupancy_bits=table["initial_occupancy_bits"][index],
        acquisition_s=table["realised_acquisition_s"][index],
        restart_s=table["realised_restart_s"][index],
        ramp_s=table["declared_ramp_s"][index],
        horizon_s=float(table["horizon_s"][0]),
    )
    trace = simulate(parameters, np.random.default_rng(arguments.seed))
    report = attribute(trace, resamples=int(arguments.resamples))
    print(f"corpus sha256 {corpus.sha256}")
    print("\n".join(report.lines()))
    return 0


def _baselines(arguments: argparse.Namespace) -> int:
    corpus = load_corpus()
    split = _split_for(arguments.split, corpus)
    print(f"split: {split.name} -- {split.description}")
    print(f"held out: {split.held_out}")
    for score in score_baselines(corpus, split, resamples=int(arguments.resamples)):
        print(
            f"{score.predictor:<34} n={score.n_test:<5} "
            f"dominance accuracy={score.dominance_accuracy:.4f} "
            f"[{score.dominance_accuracy_low:.4f}, {score.dominance_accuracy_high:.4f}] "
            f"log10 MAE(total)={score.log_mae_total:.4f}"
        )
    return 0


def _learned(arguments: argparse.Namespace) -> int:
    corpus = load_corpus()
    split = _split_for(arguments.split, corpus)
    print(f"split: {split.name} -- {split.description}")
    print("baselines first, which is the order this package benchmarks in:")
    scores = list(score_baselines(corpus, split, resamples=int(arguments.resamples)))
    learned, prediction, _ = score_learned(
        corpus, split, resamples=int(arguments.resamples)
    )
    scores.append(learned)
    for score in scores:
        coverage = (
            "n/a" if score.band_coverage != score.band_coverage
            else f"{score.band_coverage:.4f}"
        )
        print(
            f"{score.predictor:<34} n={score.n_test:<5} "
            f"dominance accuracy={score.dominance_accuracy:.4f} "
            f"[{score.dominance_accuracy_low:.4f}, {score.dominance_accuracy_high:.4f}] "
            f"log10 MAE(total)={score.log_mae_total:.4f} band coverage={coverage}"
        )
    best = max(scores, key=lambda s: s.dominance_accuracy)
    print(f"highest dominance accuracy on this split: {best.predictor}")
    print(
        "mean dominance confidence of the learned component: "
        f"{float(np.mean(prediction.dominance_confidence)):.4f} "
        "(ensemble vote share, not a calibrated probability)"
    )
    return 0


def _sequence(arguments: argparse.Namespace) -> int:
    handover = HandoverSequence(
        source_rate_bps=arguments.source_rate,
        downlink_rate_bps=arguments.downlink_rate,
        buffer_capacity_bits=arguments.buffer,
        initial_occupancy_bits=arguments.occupancy,
        acquisition_s=arguments.acquisition,
        restart_s=arguments.restart,
        ramp_s=arguments.ramp,
        horizon_s=arguments.horizon,
    )
    decomposition = fluid_decomposition(handover)
    bracket = loss_bracket(
        offered_bits=handover.source_rate_bps * handover.horizon_s,
        downlink_rate_bps=handover.downlink_rate_bps,
        outage_s=handover.outage_s,
        ramp_s=handover.ramp_s,
        horizon_s=handover.horizon_s,
        buffer_capacity_bits=handover.buffer_capacity_bits,
        initial_occupancy_bits=handover.initial_occupancy_bits,
    )
    print("baseline 2, the analytic fluid decomposition (bits):")
    for name in MECHANISMS:
        print(f"  {name:<14} {decomposition.mechanisms[name]:.6e}")
    print(f"  {'total':<14} {decomposition.total_bits:.6e}")
    print(f"  dominant       {decomposition.dominant}")
    print(f"  sum residual   {decomposition.sum_residual_bits:.6e} bits")
    print(f"  backlog drains {decomposition.drains}")
    print("")
    print("baseline 1, the acquisition-only assumption (bits):")
    print(f"  {acquisition_only_loss_bits(handover):.6e}")
    print("")
    print("independently derived bracket on the total (bits), from conservation and")
    print("capacity counting alone -- none of the models above is used:")
    print(f"  lower          {bracket.lower_bits:.6e}")
    print(f"  upper (tight)  {bracket.upper_tight_bits:.6e}")
    print(f"  slack          {bracket.slack_bits:.6e}")
    print(f"  brackets the fluid total: {bracket.brackets(decomposition.total_bits)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, built separately so the tests can exercise it."""
    parser = argparse.ArgumentParser(
        prog="handoverloss",
        description=(
            "Attribute end-to-end data loss across an optical ground-station "
            "handover to acquisition time, protocol restart and buffer drain, with "
            "an interval on each. Research-grade; not flight-qualified, not "
            "certified, and not approved for operational aerospace use."
        ),
    )
    parser.add_argument("--version", action="version", version=f"handoverloss {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    attribute_parser = subparsers.add_parser(
        "attribute", help="baseline 3 on the committed corpus, with intervals"
    )
    attribute_parser.add_argument("--events", type=int, default=256)
    attribute_parser.add_argument("--resamples", type=int, default=2000)
    attribute_parser.add_argument("--seed", type=int, default=96_096)
    attribute_parser.set_defaults(handler=_attribute)

    for name, handler, help_text in (
        ("baselines", _baselines, "score baseline 1 and baseline 2 on a held-out split"),
        ("learned", _learned, "score the learned component beside both baselines"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument(
            "--split", choices=("station_pairs", "time"), default="station_pairs"
        )
        sub.add_argument("--resamples", type=int, default=1000)
        sub.set_defaults(handler=handler)

    sequence_parser = subparsers.add_parser(
        "sequence", help="closed-form decomposition of one handover, with its bracket"
    )
    sequence_parser.add_argument("--source-rate", type=float, required=True,
                                 help="source rate, bit/s")
    sequence_parser.add_argument("--downlink-rate", type=float, required=True,
                                 help="steady egress rate of the incoming station, bit/s")
    sequence_parser.add_argument("--buffer", type=float, required=True,
                                 help="buffer capacity, bits")
    sequence_parser.add_argument("--occupancy", type=float, required=True,
                                 help="buffer occupancy at the break, bits")
    sequence_parser.add_argument("--acquisition", type=float, required=True,
                                 help="acquisition duration, s")
    sequence_parser.add_argument("--restart", type=float, required=True,
                                 help="protocol restart duration, s")
    sequence_parser.add_argument("--ramp", type=float, required=True,
                                 help="egress ramp duration, s")
    sequence_parser.add_argument("--horizon", type=float, required=True,
                                 help="accounting window, s")
    sequence_parser.set_defaults(handler=_sequence)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and run the chosen subcommand. Returns the exit status."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except CorpusAbsentError as error:
        print(f"handoverloss: {error}", file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"handoverloss: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
