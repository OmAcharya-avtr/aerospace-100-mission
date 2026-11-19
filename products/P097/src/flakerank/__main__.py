"""Command-line entry point: ``python -m flakerank``.

Three subcommands:

``rank``
    Rank a recorded run history three ways and print the disagreement report.
``geometric``
    Evaluate the closed-form re-run cost for one ``(p, runtime)`` pair, so the
    mathematical core can be checked by hand without a history.
``disagree``
    Print only the headline rank statistics, for a script that wants one line.

The default history path is ``data/run_history.json`` resolved relative to the
**package installation**, not to the caller's working directory, so the CLI
works from anywhere. A validation script in an earlier batch of this mission
failed the release gate because it passed a bare relative path to a loader; the
gate runs scripts with the working directory set elsewhere.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .history import load_history
from .ranking import rank_tests
from .report import render_disagreement, render_report

__all__ = ["main"]

#: The installed package directory, used to resolve the default history path.
_PACKAGE = Path(__file__).resolve().parent


def _default_history() -> Path:
    """Locate ``data/run_history.json`` without consulting the caller's cwd.

    Walks up from the package directory looking for ``data/run_history.json``,
    which finds it for an editable install from a source tree and returns a
    path that simply does not exist for a wheel install -- in which case
    :func:`flakerank.history.load_history` raises with a message naming the
    generator script. That is the honest behaviour: the history is evidence
    committed in the source repository, not package data.
    """
    for parent in [_PACKAGE, *_PACKAGE.parents]:
        candidate = parent / "data" / "run_history.json"
        if candidate.is_file():
            return candidate
    return _PACKAGE.parent.parent / "data" / "run_history.json"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="flakerank",
        description=(
            "Rank intermittently failing tests by the cost of their intermittency, "
            "and report how far that ranking disagrees with ranking by flake rate."
        ),
        epilog=(
            "Research-grade software. Not flight-qualified, not certified and not "
            "approved for operational aerospace use."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rank = subparsers.add_parser(
        "rank", help="rank a recorded history three ways and report the disagreement"
    )
    rank.add_argument(
        "--history",
        type=Path,
        default=None,
        help="path to a run history JSON file (default: the committed data/run_history.json)",
    )
    rank.add_argument("--top", type=int, default=10, help="rows in the top table (default: 10)")
    rank.add_argument(
        "--top-k", type=int, default=10, help="prefix length for the capture figures (default: 10)"
    )
    rank.add_argument(
        "--movers", type=int, default=10, help="movers listed per comparison (default: 10)"
    )
    rank.add_argument(
        "--resamples", type=int, default=2000, help="bootstrap resamples (default: 2000)"
    )
    rank.add_argument(
        "--confidence", type=float, default=0.95, help="interval coverage (default: 0.95)"
    )
    rank.add_argument("--seed", type=int, default=97044, help="resampling seed (default: 97044)")
    rank.add_argument(
        "--minimum-attempts",
        type=int,
        default=1,
        help="omit tests with fewer attempts in the window (default: 1)",
    )

    disagree = subparsers.add_parser(
        "disagree", help="print only the headline rank statistics"
    )
    disagree.add_argument("--history", type=Path, default=None, help="path to a run history JSON")
    disagree.add_argument("--resamples", type=int, default=2000, help="bootstrap resamples")
    disagree.add_argument("--seed", type=int, default=97044, help="resampling seed")

    geometric = subparsers.add_parser(
        "geometric", help="evaluate the closed-form re-run cost for one (p, runtime) pair"
    )
    geometric.add_argument(
        "--failure-probability",
        type=float,
        required=True,
        help="per-attempt failure probability p in [0, 1]",
    )
    geometric.add_argument(
        "--failing-attempt-seconds",
        type=float,
        required=True,
        help="mean runtime of a FAILING attempt, seconds",
    )
    geometric.add_argument(
        "--retry-budget",
        type=int,
        default=None,
        help="maximum attempts per invocation; omit for an unbounded retry policy",
    )
    geometric.add_argument(
        "--invocations", type=int, default=1, help="invocations over the period (default: 1)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: argument vector without the program name. ``None`` uses
            ``sys.argv[1:]``.

    Returns:
        0 on success, 2 on a user error with the message on stderr. A user
        error is reported as a message and a status, never as a traceback.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "geometric":
            from .geometric import expected_failed_attempts, expected_wasted_seconds

            expected = expected_failed_attempts(args.failure_probability, args.retry_budget)
            wasted = expected_wasted_seconds(
                args.failure_probability,
                args.failing_attempt_seconds,
                args.retry_budget,
                args.invocations,
            )
            budget = "unbounded" if args.retry_budget is None else str(args.retry_budget)
            sys.stdout.write(
                f"failure probability p          : {args.failure_probability!r}\n"
                f"failing-attempt runtime, s     : {args.failing_attempt_seconds!r}\n"
                f"retry budget (max attempts)    : {budget}\n"
                f"invocations                    : {args.invocations}\n"
                f"expected failed attempts E[F]  : {expected!r}\n"
                f"expected wasted seconds        : {wasted!r}\n"
            )
            return 0

        path = args.history if args.history is not None else _default_history()
        history = load_history(path)
        if args.command == "disagree":
            report = rank_tests(history, resamples=args.resamples, seed=args.seed)
            sys.stdout.write(render_disagreement(report.truth_vs_rate, movers=5) + "\n")
            return 0
        report = rank_tests(
            history,
            minimum_attempts=args.minimum_attempts,
            top_k=args.top_k,
            top_movers=args.movers,
            resamples=args.resamples,
            confidence=args.confidence,
            seed=args.seed,
        )
        sys.stdout.write(render_report(report, count=args.top, movers=args.movers) + "\n")
        return 0
    except (FileNotFoundError, ValueError, TypeError) as error:
        sys.stderr.write(f"flakerank: {error}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
