"""Command-line interface: ``python -m evidencestale``.

Three subcommands. ``report`` prints the headline stale fraction and both
baselines' error directions over a committed corpus; ``bracket`` prints the
independently derived bracket on the stale fraction; ``benchmark`` scores the
learned model against the baselines on held-out artifacts and held-out time.

No library module prints. Everything written to standard output is written
here.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .benchmark import run_benchmark
from .bounds import stale_fraction_bracket
from .corpus import MAIN_CORPUS, load_corpus
from .replay import changed_sets_by_item
from .report import build_report, render_report

SAFETY = (
    "This software is research-grade. It is not flight-qualified, not certified "
    "and not approved for operational aerospace use."
)


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, built separately so a test can exercise it."""
    parser = argparse.ArgumentParser(
        prog="python -m evidencestale",
        description=(
            "Measure how much of an assurance case rests on evidence whose subject "
            "has changed since the evidence was produced. " + SAFETY
        ),
    )
    parser.add_argument("--version", action="version", version=f"evidencestale {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("report", "headline stale fraction and both baselines' error directions"),
        ("bracket", "independently derived bracket on the stale fraction"),
        ("benchmark", "learned model against the baselines on held-out splits"),
    ):
        sub = subparsers.add_parser(name, help=help_text, description=help_text)
        sub.add_argument(
            "--corpus",
            default=MAIN_CORPUS,
            help=f"corpus JSON path, relative to the repository root (default: {MAIN_CORPUS})",
        )
        sub.add_argument(
            "--declarations",
            choices=("complete", "partial"),
            default="complete",
            help="dependency-declaration view the methods see (default: complete)",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit status."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        corpus = load_corpus(arguments.corpus)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if arguments.command == "report":
        print(render_report(build_report(corpus, arguments.declarations)))
        return 0

    if arguments.command == "bracket":
        view = corpus.graph(arguments.declarations)
        report = build_report(corpus, arguments.declarations)
        changed = changed_sets_by_item(
            corpus.history,
            [(item.identifier, item.produced_at) for item in corpus.evidence],
            corpus.evaluation_time,
        )
        bracket = stale_fraction_bracket(
            corpus.evidence,
            changed,
            view.dependencies(),
            measured=report.truth.stale_fraction,
            max_out_degree=corpus.declared_max_dependents,
            depth_limit=corpus.declared_max_impact_depth,
            artifact_count=len(corpus.true_graph.nodes),
        )
        print(f"corpus                   : {corpus.name}")
        print(f"declarations             : {arguments.declarations}")
        print(f"changed artifacts (k)    : {bracket.changed_count}")
        print(f"max out-degree (d)       : {bracket.max_out_degree}")
        print(f"depth limit (L)          : {bracket.depth_limit}")
        print(f"artifacts (N)            : {bracket.artifact_count}")
        print(f"max items per subject (m): {bracket.max_items_per_subject}")
        print(f"evidence items (E)       : {bracket.evidence_count}")
        print(f"lower bound, depth 0     : {bracket.lower_depth0:.6f}")
        print(f"lower bound, depth 1     : {bracket.lower_depth1:.6f}")
        print(f"measured stale fraction  : {bracket.measured:.6f}")
        print(f"upper bound (Moore+union): {bracket.upper:.6f}")
        print(f"slack per endpoint       : {bracket.slack:.3e}")
        print(f"bracket holds            : {bracket.holds}")
        print(f"upper bound informative  : {bracket.informative}")
        print()
        print(SAFETY)
        return 0 if bracket.holds else 1

    result = run_benchmark(corpus, arguments.declarations)
    print(f"corpus                   : {corpus.name}")
    print(f"declarations             : {result.declarations}")
    print(f"evidence items           : {result.n_items}")
    print(f"truly stale items        : {result.n_positive}")
    print(f"total fit seconds        : {result.fit_seconds:.3f}")
    print()
    print(
        f"{'method':<30} {'split':<9} {'auc':>8} {'brier':>8} "
        f"{'agree':>8} {'missed':>8} {'alarm':>8}"
    )
    for method in result.methods():
        for kind in ("artifact", "time"):
            summary = result.summary(method, kind)
            print(
                f"{method:<30} {kind:<9} {summary['auc']:8.4f} {summary['brier']:8.4f} "
                f"{summary['agreement']:8.4f} {summary['missed_staleness_rate']:8.4f} "
                f"{summary['false_alarm_rate']:8.4f}"
            )
    print()
    print(SAFETY)
    return 0


if __name__ == "__main__":
    sys.exit(main())
