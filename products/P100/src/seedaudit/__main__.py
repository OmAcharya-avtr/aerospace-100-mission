"""Command-line interface: ``python -m seedaudit``.

Five subcommands, one per question the product answers.

``power``
    Print the closed-form detection-power table and the escaped-rate bound.
    Pure arithmetic, no execution, no filesystem access.
``static``
    Run the self-pollution screen over a tree and print the findings.
``arrangements``
    Run one script in the pristine pair and the reused tree and print the
    classification.
``corpus``
    Generate the injected corpus and measure detection power per rate.
``audit``
    Audit a repository's committed numeric claims. Prints the headline, the
    coverage and the findings, in that order.

No command writes to the tree it is given. Every execution happens in a
pristine copy in a temporary directory.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

from .audit import DEFAULT_EXCLUSIONS, DEFAULT_ROOTS, audit_repository, static_scan
from .corpus import (
    INJECTED_RATES,
    block_detection,
    generate_corpus,
    manifestation_indicators,
)
from .dynamic import DEFAULT_TIMEOUT, two_arrangements
from .power import DEFAULT_ALPHA, MISSION_RATE, bootstrap_interval
from .power import detection_probability_pairwise as pairwise
from .power import detection_probability_reference as reference
from .report import (
    AUDIT_MODEL,
    BOOTSTRAP_RESAMPLES,
    CONFIDENCE,
    REPORTED_REPEATS,
    SEED,
    format_audit,
    format_power_table,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m seedaudit",
        description=(
            "Audit a repository's own reproducibility claims, with the detection "
            "power of the audit stated in closed form."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    power = sub.add_parser("power", help="print the closed-form detection-power table")
    power.add_argument(
        "--rate", type=float, default=MISSION_RATE, help="per-run manifestation rate"
    )
    power.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="significance level")

    static = sub.add_parser("static", help="self-pollution screen over a tree")
    static.add_argument("root", type=Path, help="directory to screen")
    static.add_argument("--subdir", action="append", default=None, help="subdirectory to walk")

    arrangements = sub.add_parser("arrangements", help="pristine pair against reused tree")
    arrangements.add_argument("root", type=Path, help="tree to copy")
    arrangements.add_argument("script", help="script path relative to the tree")
    arrangements.add_argument("--artifact", default=None, help="output file relative to the tree")
    arrangements.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)

    corpus = sub.add_parser("corpus", help="measure detection power on the injected corpus")
    corpus.add_argument("--runs", type=int, default=440, help="independent runs per rate")
    corpus.add_argument("--repeats", type=int, default=2, help="runs per simulated audit")

    audit = sub.add_parser("audit", help="audit a repository's committed numeric claims")
    audit.add_argument("root", type=Path, help="repository root")
    audit.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    audit.add_argument("--workers", type=int, default=2)
    audit.add_argument("--limit", type=int, default=None)
    audit.add_argument("--deadline", type=float, default=None, help="global budget, seconds")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit status. 0 when the requested report was produced. A
        finding is **not** an error: an audit that exited non-zero on a finding
        would push a user towards not running it.
    """
    args = _build_parser().parse_args(argv)

    if args.command == "power":
        print(AUDIT_MODEL)
        print()
        print(format_power_table(args.rate, REPORTED_REPEATS, args.alpha))
        return 0

    if args.command == "static":
        roots = tuple(args.subdir) if args.subdir else (".",)
        findings, failures, scanned = static_scan(args.root, roots, ())
        print(f"files parsed: {scanned}")
        print(f"files with findings: {len(findings)}")
        print(f"findings: {sum(len(v) for v in findings.values())}")
        print(f"files that would not parse: {len(failures)}")
        for path in sorted(findings):
            for finding in findings[path]:
                print(f"  {path}: {finding.describe()}")
        return 0

    if args.command == "arrangements":
        result = two_arrangements(args.root, args.script, args.artifact, args.timeout)
        print(f"script: {result.script}")
        print(f"pristine runs agree: {result.pristine_agree}")
        print(f"reused-tree runs agree: {result.reused_agree}")
        print(f"classification: {result.classification}")
        print(f"escaped-rate upper bound at n={len(result.pristine)}: {result.escaped_bound!r}")
        if result.note:
            print(f"note: {result.note}")
        return 0

    if args.command == "corpus":
        rng = np.random.default_rng(SEED)
        with tempfile.TemporaryDirectory(prefix="seedaudit-corpus-") as temporary:
            manifest = generate_corpus(Path(temporary) / "corpus")
            print(f"{'rate':>12}  {'p_hat':>10}  {'power':>10}  {'lo':>10}  {'hi':>10}"
                  f"  {'derived':>10}  {'blocks':>7}")
            for rate in INJECTED_RATES:
                name = manifest.rate_scripts[rate]
                indicators = manifestation_indicators(
                    manifest.root, name, manifest.references[name], args.runs
                )
                _, pair = block_detection(indicators, args.repeats)
                point, low, high = bootstrap_interval(
                    pair, rng, BOOTSTRAP_RESAMPLES, CONFIDENCE
                )
                derived = pairwise(rate, args.repeats)
                print(
                    f"{rate:>12.6f}  {indicators.mean():>10.6f}  {point:>10.6f}"
                    f"  {low:>10.6f}  {high:>10.6f}  {derived:>10.6f}  {pair.size:>7d}"
                )
            print()
            print(f"reference-design derived power at p=1/44, n=2: {reference(MISSION_RATE, 2)!r}")
        return 0

    audit = audit_repository(
        args.root,
        DEFAULT_ROOTS,
        DEFAULT_EXCLUSIONS,
        timeout=args.timeout,
        workers=args.workers,
        deadline_seconds=args.deadline,
        limit=args.limit,
    )
    print(AUDIT_MODEL)
    print()
    print(format_audit(audit))
    for result in audit.results:
        if result.seed_attributable or result.declared_seed_moved:
            print()
            print(f"{result.candidate.label}  [{result.status}]")
            for moved in result.seed_attributable:
                print(f"  SEED-ATTRIBUTABLE line {moved.line}: {moved.before} -> {moved.after}")
            for moved in result.declared_seed_moved:
                print(f"  DECLARED-SEED      line {moved.line}: {moved.before} -> {moved.after}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
