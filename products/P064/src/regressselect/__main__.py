"""Command line interface: ``python -m regressselect``.

Subcommands:

``graph``      print the import graph and the dependency-based selection
``select``     rank the suite for one declared change, with probabilities
``benchmark``  score all three strategies on a committed history
``mutate``     run the seeded-fault evaluation against a sample project
``apfd``       APFD of a test order against a declared detection matrix

Exit status is 0 on success and 2 on a usage or input error. Nothing in the
library prints; all output is written here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmark import (
    dependency_order,
    evaluate_on_fault_matrix,
    evaluate_strategies,
    fault_matrix_apfd,
)
from .history import Revision, load_history
from .importgraph import build_import_graph
from .metrics import apfd
from .mutation import load_faults, run_mutation_evaluation
from .prioritise import CoFailurePrioritiser
from .selector import LearnedSelector


def _add_graph(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("graph", help="print the import graph and a dependency selection")
    p.add_argument("root", help="project root whose .py files are parsed")
    p.add_argument(
        "--changed", nargs="*", default=[], help="dotted module names treated as changed"
    )


def _add_select(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("select", help="rank the suite for one declared change")
    p.add_argument("root", help="project root of the suite under selection")
    p.add_argument("--history", required=True, help="committed history JSON")
    p.add_argument("--changed", nargs="+", required=True, help="dotted changed module names")
    p.add_argument(
        "--churn", nargs="*", default=[], help="per-module churn as module=lines", metavar="M=N"
    )
    p.add_argument("--budget", type=float, default=0.25, help="fraction of the suite to run")
    p.add_argument(
        "--strategy",
        choices=("dependency", "history", "learned"),
        default="learned",
        help="which strategy produces the order",
    )
    p.add_argument("--train-fraction", type=float, default=0.70)


def _add_benchmark(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("benchmark", help="score all three strategies on a history")
    p.add_argument("root", help="project root of the suite under selection")
    p.add_argument("--history", required=True, help="committed history JSON")
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument(
        "--budgets", nargs="*", type=float, default=[0.10, 0.25, 0.50, 0.75, 1.00]
    )
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")


def _add_mutate(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("mutate", help="seeded-fault evaluation against a sample project")
    p.add_argument("root", help="sample project root")
    p.add_argument("--faults", required=True, help="seeded fault JSON")
    p.add_argument("--history", required=True, help="committed history JSON")
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--churn-lines", type=int, default=1)
    p.add_argument("--out", default=None, help="write the measured fault matrix JSON here")


def _add_apfd(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("apfd", help="APFD of an order against a detection matrix")
    p.add_argument("--order", required=True, help="JSON file: list of test node ids, in order")
    p.add_argument(
        "--detecting", required=True, help="JSON file: {fault_id: [detecting node ids]}"
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``python -m regressselect``."""
    parser = argparse.ArgumentParser(
        prog="python -m regressselect",
        description=(
            "Regression test selection with measured safety loss. Research-grade; "
            "not flight-qualified, not certified, not approved for operational "
            "aerospace use."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_graph(sub)
    _add_select(sub)
    _add_benchmark(sub)
    _add_mutate(sub)
    _add_apfd(sub)
    return parser


def _parse_churn(pairs: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"churn must be given as module=lines, got {item!r}")
        key, value = item.split("=", 1)
        out[key] = int(value)
    return out


def _cmd_graph(args: argparse.Namespace, out) -> int:
    graph = build_import_graph(args.root)
    out.write(f"modules: {len(graph.modules)}  test modules: {len(graph.test_modules)}\n")
    for module in sorted(graph.modules):
        imports = sorted(graph.imports_of(module))
        out.write(f"  {module} -> {imports}\n")
    if args.changed:
        selected = sorted(graph.select_tests(args.changed))
        frac = len(selected) / len(graph.test_modules)
        out.write(
            f"\ndependency selection for {list(args.changed)}: "
            f"{len(selected)}/{len(graph.test_modules)} test modules ({frac:.4f})\n"
        )
        for module in selected:
            out.write(f"  {module}\n")
    return 0


def _rank_for(args: argparse.Namespace, out) -> int:
    graph = build_import_graph(args.root)
    history = load_history(args.history)
    train, _ = history.split_chronological(args.train_fraction)
    churn = _parse_churn(args.churn) or dict.fromkeys(args.changed, 1)
    revision = Revision(
        revision="working-tree", changed_modules=tuple(args.changed), churn=churn
    )
    nodes = list(history.test_ids)
    if args.strategy == "dependency":
        _, selected = dependency_order(graph, revision, nodes)
        out.write(
            f"dependency selection: {len(selected)}/{len(nodes)} nodes "
            f"({len(selected) / len(nodes):.4f}); the budget does not apply\n"
        )
        for node in sorted(selected):
            out.write(f"  {node}\n")
        return 0
    if args.strategy == "history":
        ranked = CoFailurePrioritiser().fit(train).rank(revision.changed_modules, nodes)
    else:
        ranked = LearnedSelector().fit(train, graph).rank(revision, nodes)
    k = max(1, int(round(args.budget * len(nodes))))
    out.write(
        f"strategy={args.strategy} budget={args.budget:.4f} "
        f"running {k}/{len(nodes)} nodes\n"
    )
    for i, item in enumerate(ranked, start=1):
        mark = "run " if i <= k else "skip"
        out.write(f"  {mark} {item.score:.6f}  {item.node_id}\n")
    return 0


def _cmd_benchmark(args: argparse.Namespace, out) -> int:
    graph = build_import_graph(args.root)
    history = load_history(args.history)
    train, test = history.split_chronological(args.train_fraction)
    scores = evaluate_strategies(train, test, graph, budgets=tuple(args.budgets))
    if args.json:
        payload = {
            name: {
                "apfd_full_order": s.apfd_full_order,
                "revisions_scored": s.revisions_scored,
                "points": [
                    {
                        "budget": p.budget,
                        "tests_run_fraction": p.tests_run_fraction,
                        "safety_loss": p.safety_loss,
                        "regressions_caught": p.regressions_caught,
                    }
                    for p in s.points
                ],
            }
            for name, s in scores.items()
        }
        out.write(json.dumps(payload, indent=1) + "\n")
        return 0
    out.write(
        f"history: {len(history)} revisions, train {len(train)}, test {len(test)}, "
        f"{scores['learned'].revisions_scored} evaluation revisions with a failure\n\n"
    )
    out.write(f"{'strategy':<12}{'APFD':>8}{'budget':>9}{'cost':>9}{'loss':>9}{'caught':>9}\n")
    for name, score in scores.items():
        for point in score.points:
            out.write(
                f"{name:<12}{score.apfd_full_order:>8.4f}{point.budget:>9.2f}"
                f"{point.tests_run_fraction:>9.4f}{point.safety_loss:>9.4f}"
                f"{point.regressions_caught:>9.4f}\n"
            )
    return 0


def _cmd_mutate(args: argparse.Namespace, out) -> int:
    graph = build_import_graph(args.root)
    history = load_history(args.history)
    train, _ = history.split_chronological(args.train_fraction)
    prioritiser = CoFailurePrioritiser().fit(train)
    selector = LearnedSelector().fit(train, graph)
    faults = load_faults(args.faults)
    matrix = run_mutation_evaluation(args.root, faults)
    if args.out:
        Path(args.out).write_text(matrix.to_json(), encoding="utf-8")
    out.write(
        f"seeded faults: {len(matrix.fault_ids)}  detected by the full suite: "
        f"{len(matrix.detected_fault_ids)} ({matrix.full_suite_detection():.4f})\n"
        f"undetected by any test: "
        f"{[f for f in matrix.fault_ids if not matrix.detecting.get(f)]}\n\n"
    )
    points = evaluate_on_fault_matrix(
        matrix, graph, prioritiser, selector, churn_lines=args.churn_lines
    )
    out.write(
        f"{'strategy':<12}{'budget':>8}{'cost':>9}{'det/all':>9}{'det/able':>10}{'bound':>8}\n"
    )
    for name, series in points.items():
        for point in series:
            out.write(
                f"{name:<12}{point.budget:>8.2f}{point.tests_run_fraction:>9.4f}"
                f"{point.faults_detected_all:>9.4f}{point.faults_detected_detectable:>10.4f}"
                f"{point.full_suite_upper_bound:>8.4f}\n"
            )
    apfds = fault_matrix_apfd(
        matrix, graph, prioritiser, selector, churn_lines=args.churn_lines
    )
    out.write("\nmulti-fault APFD over the union change:\n")
    for name, value in apfds.items():
        out.write(f"  {name:<12}{value:.4f}\n")
    return 0


def _cmd_apfd(args: argparse.Namespace, out) -> int:
    order = json.loads(Path(args.order).read_text(encoding="utf-8"))
    detecting = json.loads(Path(args.detecting).read_text(encoding="utf-8"))
    out.write(f"APFD = {apfd(order, detecting):.6f}  (n={len(order)}, m={len(detecting)})\n")
    return 0


def main(argv: list[str] | None = None, out=None) -> int:
    """Run the CLI. Returns the process exit status."""
    stream = out if out is not None else sys.stdout
    args = build_parser().parse_args(argv)
    handlers = {
        "graph": _cmd_graph,
        "select": _rank_for,
        "benchmark": _cmd_benchmark,
        "mutate": _cmd_mutate,
        "apfd": _cmd_apfd,
    }
    try:
        return handlers[args.command](args, stream)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
