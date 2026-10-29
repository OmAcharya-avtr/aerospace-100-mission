"""Command-line interface: ``python -m tmtcdiff``.

Subcommands
-----------

``info``
    Versions, the change-class taxonomy and the impact classes.
``taxonomy``
    The taxonomy table alone.
``diff OLD NEW``
    Semantic diff of two dictionary JSON files, silent tier first.
``impact OLD NEW --corpus DIR``
    Diff plus the impact report and the coverage verdict.
``demo``
    A generated pair and corpus, diffed and assessed, so the tool can be run
    without supplying files.
``sweep``
    A small rename threshold sweep. The full swept grid lives in
    ``validation/validate_rename_sweep.py``.
``negative-control``
    The reordering control: a textually reordered copy of one dictionary,
    verified to produce an empty semantic diff.

``--fail-on-silent`` makes ``diff``, ``impact`` and ``demo`` exit 2 when a
silent change is present, for use in a pipeline. It is off by default, so a
plain run exits 0 whatever it finds.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from .changes import ChangeClass
from .corpus import load_corpus
from .diff import DEFAULT_RENAME_THRESHOLD, DiffConfig, diff_dictionaries
from .impact import assess_impact
from .model import Dictionary
from .parse import DictionaryFormatError, load_dictionary, loads
from .rename import STRATEGIES, match_renames
from .report import (
    format_table,
    impact_class_table,
    render_diff,
    render_impact,
    taxonomy_table,
)
from .synth import (
    baseline_dictionary,
    change_pair,
    rename_pair,
    reordering_control,
    synthetic_corpus,
)

EXIT_SILENT_FOUND = 2


def _config(arguments: argparse.Namespace) -> DiffConfig:
    return DiffConfig(
        rename_threshold=arguments.threshold,
        strategy=arguments.strategy,
        detect_renames=not arguments.no_renames,
    )


def _environment_lines() -> list[str]:
    from importlib.metadata import PackageNotFoundError, version

    def installed(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "not installed"

    return [
        f"tmtcdiff     : {__version__}",
        f"python       : {sys.version.split()[0]}",
        f"numpy        : {installed('numpy')}",
        f"scipy        : {installed('scipy')}",
        f"deepdiff     : {installed('deepdiff')}  (named in the alternatives table)",
        f"jsonschema   : {installed('jsonschema')}  (named in the alternatives table)",
    ]


def _cmd_info(arguments: argparse.Namespace) -> int:
    del arguments
    print("\n".join(_environment_lines()))
    print()
    print("change-class taxonomy, silent tier first:")
    print(taxonomy_table())
    print()
    print("impact classes, in report order:")
    print(impact_class_table())
    print()
    print(f"default rename threshold: {DEFAULT_RENAME_THRESHOLD} (chosen from the measured")
    print("sweep in validation/validate_rename_sweep.py, not picked)")
    return 0


def _cmd_taxonomy(arguments: argparse.Namespace) -> int:
    del arguments
    print(taxonomy_table())
    return 0


def _load(path: str) -> Dictionary:
    try:
        return load_dictionary(path)
    except DictionaryFormatError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


def _cmd_diff(arguments: argparse.Namespace) -> int:
    old = _load(arguments.old)
    new = _load(arguments.new)
    result = diff_dictionaries(old, new, _config(arguments))
    print(render_diff(result, max_rows=arguments.max_rows))
    return EXIT_SILENT_FOUND if (arguments.fail_on_silent and result.silent) else 0


def _cmd_impact(arguments: argparse.Namespace) -> int:
    old = _load(arguments.old)
    new = _load(arguments.new)
    try:
        corpus = load_corpus(arguments.corpus)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    result = diff_dictionaries(old, new, _config(arguments))
    report = assess_impact(result, corpus, old, new)
    print(render_diff(result, max_rows=arguments.max_rows))
    print()
    print(render_impact(report, max_rows=arguments.max_rows))
    return EXIT_SILENT_FOUND if (arguments.fail_on_silent and result.silent) else 0


def _cmd_demo(arguments: argparse.Namespace) -> int:
    pair = change_pair(arguments.size, arguments.seed)
    result = diff_dictionaries(pair.old, pair.new, _config(arguments))
    corpus = synthetic_corpus(
        pair.old,
        arguments.seed + 1,
        names=[change.name for change in pair.truth],
        include_unreadable=True,
    )
    report = assess_impact(result, corpus, pair.old, pair.new)
    print(f"generated pair: size {arguments.size}, seed {arguments.seed}")
    print(f"injected changes: {len(pair.truth)}")
    print()
    print(render_diff(result, max_rows=arguments.max_rows))
    print()
    print(render_impact(report, max_rows=arguments.max_rows))
    return EXIT_SILENT_FOUND if (arguments.fail_on_silent and result.silent) else 0


def _cmd_sweep(arguments: argparse.Namespace) -> int:
    rows: list[list[object]] = []
    for size in arguments.sizes:
        pair = rename_pair(
            size, arguments.seed, renames=max(1, size // 10), removals=2, additions=2
        )
        removed = [
            pair.old.parameter_map[name]
            for name in sorted(set(pair.old.parameter_map) - set(pair.new.parameter_map))
        ]
        added = [
            pair.new.parameter_map[name]
            for name in sorted(set(pair.new.parameter_map) - set(pair.old.parameter_map))
        ]
        truth = {change.name: change.new_name for change in pair.truth_for(ChangeClass.RENAMED)}
        for threshold in arguments.thresholds:
            for strategy in STRATEGIES:
                candidates = match_renames(
                    removed, added, threshold=threshold, strategy=strategy
                )
                accepted = [c for c in candidates if c.accepted]
                correct = sum(1 for c in accepted if truth.get(c.old_name) == c.new_name)
                false_matches = len(accepted) - correct
                rows.append(
                    [
                        size,
                        f"{threshold:.2f}",
                        strategy,
                        len(truth),
                        len(accepted),
                        correct,
                        false_matches,
                        f"{correct / len(truth):.4f}" if truth else "nan",
                        f"{false_matches / len(accepted):.4f}" if accepted else "nan",
                    ]
                )
    print(
        format_table(
            [
                "size",
                "threshold",
                "strategy",
                "true renames",
                "accepted",
                "correct",
                "false",
                "recall",
                "false-match rate",
            ],
            rows,
        )
    )
    print()
    print("The full swept grid, the selection criterion and the chosen threshold are in")
    print("validation/validate_rename_sweep_output.txt.")
    return 0


def _cmd_negative_control(arguments: argparse.Namespace) -> int:
    import difflib

    dictionary = baseline_dictionary(arguments.size, arguments.seed)
    canonical, reordered = reordering_control(dictionary, arguments.seed + 1)
    rebuilt = loads(reordered)
    result = diff_dictionaries(dictionary, rebuilt, _config(arguments))
    differing = sum(
        1
        for line in difflib.unified_diff(
            canonical.splitlines(), reordered.splitlines(), n=0, lineterm=""
        )
        if line[:1] in "+-" and not line.startswith(("---", "+++"))
    )
    print(f"dictionary size            : {len(dictionary)} items")
    print(f"canonical text lines       : {len(canonical.splitlines())}")
    print(f"reordered text lines       : {len(reordered.splitlines())}")
    print(f"text lines differing       : {differing}")
    print(f"semantic changes reported  : {len(result.changes)}")
    print(f"silent changes reported    : {len(result.silent)}")
    print(f"loud changes reported      : {len(result.loud)}")
    if result.changes:
        print("UNEXPECTED: the control reported changes")
        for change in result.changes:
            print(f"  {change.one_line()}")
        return 1
    print("control held: substantial textual reordering, empty semantic diff")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Exposed so the test suite can exercise it directly."""
    parser = argparse.ArgumentParser(
        prog="python -m tmtcdiff",
        description=(
            "Diff two telemetry and command dictionaries and report which existing "
            "references break or silently change meaning."
        ),
    )
    parser.add_argument("--version", action="version", version=f"tmtcdiff {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--threshold",
            type=float,
            default=DEFAULT_RENAME_THRESHOLD,
            help="rename acceptance threshold in [0, 1]",
        )
        target.add_argument(
            "--strategy", choices=STRATEGIES, default="optimal", help="rename matching strategy"
        )
        target.add_argument(
            "--no-renames", action="store_true", help="disable rename detection entirely"
        )
        target.add_argument(
            "--max-rows", type=int, default=0, help="limit rows listed per section; 0 lists all"
        )
        target.add_argument(
            "--fail-on-silent",
            action="store_true",
            help="exit 2 when a silent change is present",
        )

    info = subparsers.add_parser("info", help="versions, taxonomy and impact classes")
    info.set_defaults(handler=_cmd_info)

    taxonomy = subparsers.add_parser("taxonomy", help="the change-class taxonomy")
    taxonomy.set_defaults(handler=_cmd_taxonomy)

    diff_parser = subparsers.add_parser("diff", help="semantic diff of two dictionary files")
    diff_parser.add_argument("old", help="path to the old dictionary JSON")
    diff_parser.add_argument("new", help="path to the new dictionary JSON")
    add_common(diff_parser)
    diff_parser.set_defaults(handler=_cmd_diff)

    impact = subparsers.add_parser("impact", help="diff plus impact on a corpus directory")
    impact.add_argument("old", help="path to the old dictionary JSON")
    impact.add_argument("new", help="path to the new dictionary JSON")
    impact.add_argument("--corpus", required=True, help="directory of procedures, scripts, configs")
    add_common(impact)
    impact.set_defaults(handler=_cmd_impact)

    demo = subparsers.add_parser("demo", help="generated pair, diffed and assessed")
    demo.add_argument("--size", type=int, default=60, help="number of parameters")
    demo.add_argument("--seed", type=int, default=20261029, help="generator seed")
    add_common(demo)
    demo.set_defaults(handler=_cmd_demo)

    sweep = subparsers.add_parser("sweep", help="small rename threshold sweep")
    sweep.add_argument("--sizes", type=int, nargs="+", default=[50, 200])
    sweep.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.50, 0.60, 0.70, 0.80, 0.90]
    )
    sweep.add_argument("--seed", type=int, default=20261029)
    sweep.set_defaults(handler=_cmd_sweep)

    control = subparsers.add_parser(
        "negative-control", help="textually reordered copy, empty semantic diff expected"
    )
    control.add_argument("--size", type=int, default=60)
    control.add_argument("--seed", type=int, default=20261029)
    add_common(control)
    control.set_defaults(handler=_cmd_negative_control)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit status."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
