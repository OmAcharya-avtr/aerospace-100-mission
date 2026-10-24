"""``python -m cmdguard``: forward check, lint, compare, gate, rank.

Exit codes are part of the contract, because the point of a checker is to fail
a pipeline:

0
    Nothing found against the declared model. Not a statement about safety.
1
    At least one violation found, none of them gate-blocking.
2
    The irreversible-command gate refused. Reported separately from 1 so that a
    pipeline can treat an unverified irreversible command differently from
    anything else, which is the whole reason the gate is a separate verdict.
3
    The input could not be loaded or the arguments were wrong. A declaration
    error is never reported as a clean check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .benchmark import assert_disjoint, lint_miss_rate, ranker_comparison
from .dataset import build_corpus
from .defects import FAMILIES
from .features import FEATURE_NAMES
from .gate import irreversible_gate
from .lint import LintStrength, static_lint
from .loader import load_case
from .model import EffectPolicy
from .ranker import RiskRanker, involved_mask
from .report import (
    render_check,
    render_comparison,
    render_dictionary_summary,
    render_state_at,
)
from .trace import ViolationKind, forward_execute

EXIT_CLEAN = 0
EXIT_VIOLATIONS = 1
EXIT_GATE_REFUSED = 2
EXIT_INPUT_ERROR = 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cmdguard",
        description=(
            "Check a command sequence against the spacecraft state it will meet. "
            "Research-grade; a clean report is not a statement that a sequence is safe "
            "to uplink."
        ),
    )
    parser.add_argument("--version", action="version", version=f"cmdguard {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="print what this package checks and declares")
    info.add_argument("case", nargs="?", help="optional case file to summarise")

    check = subparsers.add_parser("check", help="forward execution with state traces")
    check.add_argument("case", help="path to a case YAML file")
    check.add_argument(
        "--policy",
        choices=[policy.value for policy in EffectPolicy],
        default=EffectPolicy.APPLY.value,
        help="what to do with the effects of a violating command (default: apply)",
    )
    check.add_argument(
        "--state-at",
        type=int,
        default=None,
        metavar="STEP",
        help="also print the state reached after STEP",
    )

    lint = subparsers.add_parser("lint", help="per-command static lint (baseline 2)")
    lint.add_argument("case", help="path to a case YAML file")
    lint.add_argument(
        "--strength",
        choices=[strength.value for strength in LintStrength],
        default=LintStrength.PER_COMMAND.value,
        help="lint strength (default: per_command)",
    )

    compare = subparsers.add_parser("compare", help="forward execution against the lint")
    compare.add_argument("case", help="path to a case YAML file")
    compare.add_argument(
        "--strength",
        choices=[strength.value for strength in LintStrength],
        default=LintStrength.WITH_TIMING.value,
        help="lint strength to compare against (default: with_timing)",
    )

    gate = subparsers.add_parser("gate", help="irreversible-command gate; exits 2 on refusal")
    gate.add_argument("case", help="path to a case YAML file")

    rank = subparsers.add_parser(
        "rank", help="learned risk ranking for review triage (trains a model first)"
    )
    rank.add_argument("case", help="path to a case YAML file")
    rank.add_argument("--k", type=int, default=10, help="how many steps to list (default: 10)")
    rank.add_argument(
        "--train-seeds",
        type=int,
        default=24,
        metavar="N",
        help="clean-case seeds to train on, 0..N-1 (default: 24)",
    )

    bench = subparsers.add_parser(
        "benchmark", help="lint miss rate and ranker precision at k on generated cases"
    )
    bench.add_argument("--train-seeds", type=int, default=40, help="training seeds (default: 40)")
    bench.add_argument("--test-seeds", type=int, default=20, help="held-out seeds (default: 20)")

    return parser


def _load(path: str) -> tuple[object | None, int]:
    try:
        return load_case(Path(path)), EXIT_CLEAN
    except (FileNotFoundError, ValueError) as error:
        sys.stderr.write(f"cmdguard: cannot load case: {error}\n")
        return None, EXIT_INPUT_ERROR


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit code; does not call ``sys.exit``."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "info":
        if args.case:
            case, code = _load(args.case)
            if case is None:
                return code
            print(render_dictionary_summary(case))  # type: ignore[arg-type]
            return EXIT_CLEAN
        print(f"cmdguard {__version__}")
        print("")
        print("baseline 1 (ground truth) : forward symbolic execution, cmdguard.trace")
        print("baseline 2               : per-command static lint, cmdguard.lint")
        print("gate                     : irreversible-command gate, cmdguard.gate")
        print("learned component        : risk ranker for review triage, cmdguard.ranker")
        print("")
        print(f"violation kinds ({len(ViolationKind)}):")
        for kind in ViolationKind:
            print(f"  {kind.value}")
        print(f"defect families ({len(FAMILIES)}):")
        for family in FAMILIES:
            print(f"  {family}")
        print(f"ranker features ({len(FEATURE_NAMES)}):")
        for name in FEATURE_NAMES:
            print(f"  {name}")
        print("")
        print("exit codes: 0 nothing found, 1 violations, 2 gate refused, 3 input error")
        print(
            "A clean check is a statement about the declarations supplied, not a statement "
            "that a sequence is safe to uplink."
        )
        return EXIT_CLEAN

    case, code = _load(args.case) if hasattr(args, "case") else (None, EXIT_CLEAN)
    if hasattr(args, "case") and case is None:
        return code

    if args.command == "check":
        report = forward_execute(case, EffectPolicy(args.policy))  # type: ignore[arg-type]
        print(render_check(report, case))  # type: ignore[arg-type]
        if args.state_at is not None:
            try:
                print("")
                print(render_state_at(report, args.state_at))
            except IndexError as error:
                sys.stderr.write(f"cmdguard: {error}\n")
                return EXIT_INPUT_ERROR
        gate = irreversible_gate(report, case)  # type: ignore[arg-type]
        if gate.refused:
            print("")
            print(gate.render())
            return EXIT_GATE_REFUSED
        return EXIT_VIOLATIONS if report.violations else EXIT_CLEAN

    if args.command == "lint":
        report = static_lint(case, LintStrength(args.strength))  # type: ignore[arg-type]
        print(f"cmdguard static lint ({report.strength.value}): {report.case_name}")
        print(f"findings: {report.n_findings}")
        for kind, count in report.kind_counts().items():
            print(f"  {kind:<26s} {count}")
        for finding in report.findings:
            print(finding.render())
        if report.clean:
            print(
                "The lint found nothing. The lint has no sequence state, so this says nothing "
                "about ordering defects; run 'check'."
            )
        return EXIT_VIOLATIONS if report.findings else EXIT_CLEAN

    if args.command == "compare":
        forward = forward_execute(case)  # type: ignore[arg-type]
        lint = static_lint(case, LintStrength(args.strength))  # type: ignore[arg-type]
        print(render_comparison(forward, lint))
        return EXIT_VIOLATIONS if forward.violations else EXIT_CLEAN

    if args.command == "gate":
        gate = irreversible_gate(forward_execute(case), case)  # type: ignore[arg-type]
        print(gate.render())
        return gate.exit_code

    if args.command == "rank":
        if args.k < 1:
            sys.stderr.write(f"cmdguard: --k must be >= 1, got {args.k}\n")
            return EXIT_INPUT_ERROR
        if args.train_seeds < 1:
            sys.stderr.write(
                f"cmdguard: --train-seeds must be >= 1, got {args.train_seeds}\n"
            )
            return EXIT_INPUT_ERROR
        corpus = build_corpus(range(args.train_seeds), include_clean=False)
        ranker = RiskRanker().fit([(entry.case, entry.report) for entry in corpus])
        assert ranker.metrics is not None
        scores = ranker.score(case)  # type: ignore[arg-type]
        report = forward_execute(case)  # type: ignore[arg-type]
        labels = involved_mask(report, len(case.sequence))  # type: ignore[arg-type]
        print(f"cmdguard risk ranking: {case.name}")  # type: ignore[attr-defined]
        print(f"trained on {ranker.metrics.n_cases} generated cases, "
              f"{ranker.metrics.n_rows} command occurrences, "
              f"{ranker.metrics.positive_fraction:.6f} positive")
        print(f"fit time {ranker.metrics.fit_seconds:.3f} s (wall clock, unstable)")
        print("")
        print("rank step        t_s command                risk  spread  involved")
        for rank, index in enumerate(scores.top_k(min(args.k, len(case.sequence))), start=1):  # type: ignore[arg-type]
            step = case.sequence.steps[index]  # type: ignore[attr-defined]
            print(
                f"{rank:>4d} {index:>4d} {step.t_s:>10.1f} {step.command:<22s} "
                f"{scores.risk[index]:.4f}  {scores.spread[index]:.4f}  "
                f"{'yes' if labels[index] else 'no'}"
            )
        print("")
        print(
            "The spread is the dispersion of the forest's per-tree probabilities. It is not "
            "a calibrated confidence interval; see MODEL_CARD.md."
        )
        print(
            "The ranking is review triage. Detection is done by 'check', which is exact for "
            "a linear sequence against the declared model."
        )
        return EXIT_VIOLATIONS if report.violations else EXIT_CLEAN

    if args.command == "benchmark":
        train_seeds = range(args.train_seeds)
        test_seeds = range(100000, 100000 + args.test_seeds)
        n_train, n_test = assert_disjoint(train_seeds, test_seeds)
        train = build_corpus(train_seeds, include_clean=False)
        test = build_corpus(test_seeds, include_clean=False)
        print(f"train seeds {n_train}, test seeds {n_test}")
        for strength in LintStrength:
            result = lint_miss_rate(tuple(train) + tuple(test), strength)
            print("")
            print(f"static lint miss rate ({strength.value}):")
            print(
                f"  violations {result.total.n_truth}, missed {result.total.n_missed}, "
                f"miss rate {result.total.miss_rate:.6f}"
            )
            print(
                f"  ordering-only violations {result.total.n_truth_ordering}, missed "
                f"{result.total.n_missed_ordering}, miss rate "
                f"{result.total.ordering_miss_rate:.6f}"
            )
            print(
                f"  lint findings {result.total.n_lint_findings}, unconfirmed "
                f"{result.total.n_lint_false_positive}, rate "
                f"{result.total.false_positive_rate:.6f}"
            )
        comparison = ranker_comparison(train, test)
        print("")
        print("precision at k on held-out cases:")
        for line in comparison.table():
            print(f"  {line}" if line else "")
        return EXIT_CLEAN

    parser.error(f"unhandled command {args.command!r}")  # pragma: no cover
    return EXIT_INPUT_ERROR  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
