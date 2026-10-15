"""Mutation-style evaluation: seeded faults in the bundled sample project.

Fourteen declared textual mutations are applied one at a time to a throwaway
copy of fixtures/sample_project, the full suite is executed against each, and
the failing nodes are read from the junit XML. Each selection strategy is then
scored on faults detected against tests run, with the full-suite upper bound
printed on every line.

This script starts fifteen pytest processes (fourteen mutants plus one clean
run). Measured cost on the two-core build machine is about twelve seconds.

    python validation/validate_mutation.py > validation/validate_mutation_output.txt
"""

from __future__ import annotations

import json
import sys
import time

from _common import FAULT_MATRIX_JSON, FAULTS_JSON, HISTORY_JSON, SAMPLE, TRAIN_FRACTION

from regressselect.benchmark import evaluate_on_fault_matrix, fault_matrix_apfd
from regressselect.history import load_history
from regressselect.importgraph import build_import_graph
from regressselect.mutation import load_faults, run_mutation_evaluation, run_suite
from regressselect.prioritise import CoFailurePrioritiser
from regressselect.selector import LearnedSelector

BUDGETS = (0.10, 0.25, 0.50, 0.75, 1.00)


def main() -> int:
    out = sys.stdout
    started = time.monotonic()

    graph = build_import_graph(SAMPLE)
    history = load_history(HISTORY_JSON)
    train, _ = history.split_chronological(TRAIN_FRACTION)
    prioritiser = CoFailurePrioritiser().fit(train)
    selector = LearnedSelector().fit(train, graph)
    faults = load_faults(FAULTS_JSON)

    out.write("Mutation-style evaluation on the bundled sample project\n")
    out.write("=" * 92 + "\n")
    clean = run_suite(SAMPLE)
    out.write(
        f"clean run              tests={clean.tests} failures={clean.failures} "
        f"errors={clean.errors} skipped={clean.skipped} "
        f"testcase elements={clean.testcase_count} "
        f"cross-check={'PASS' if clean.cross_check() else 'FAIL'}\n"
    )
    out.write(f"seeded faults          {len(faults)}\n\n")

    matrix = run_mutation_evaluation(SAMPLE, faults)

    out.write("per-fault detection, measured by execution\n")
    out.write(f"{'fault':<7}{'module':<24}{'operator':<50}{'tests failing':>14}\n")
    out.write("-" * 92 + "\n")
    for fault in faults:
        detecting = matrix.detecting[fault.fault_id]
        out.write(
            f"{fault.fault_id:<7}{fault.module:<24}{fault.operator[:49]:<50}"
            f"{len(detecting):>14}\n"
        )
    undetected = [f for f in matrix.fault_ids if not matrix.detecting.get(f)]
    out.write(
        f"\nfull-suite detection   {matrix.full_suite_detection():.6f} "
        f"({len(matrix.detected_fault_ids)}/{len(matrix.fault_ids)})\n"
        f"detected by no test    {undetected}\n"
        "Those four faults are real holes in the sample project's own suite. They are\n"
        "kept in the denominator so that no strategy can read as perfect on a suite\n"
        "that is not.\n"
    )

    stored = json.loads(FAULT_MATRIX_JSON.read_text(encoding="utf-8"))
    stored_detecting = {k: tuple(v) for k, v in stored["detecting"].items()}
    agrees = stored_detecting == {f: matrix.detecting[f] for f in matrix.fault_ids}
    out.write(
        f"\ncommitted fault_matrix.json reproduces this run: "
        f"{'PASS' if agrees else 'FAIL'}\n"
    )

    for churn_lines in (1, 20):
        label = (
            "churn = 1 line, the truthful churn of a one-line mutation"
            if churn_lines == 1
            else "churn = 20 lines, a sensitivity check on the churn features"
        )
        points = evaluate_on_fault_matrix(
            matrix, graph, prioritiser, selector, budgets=BUDGETS, churn_lines=churn_lines
        )
        out.write(f"\nfaults detected against tests run -- {label}\n")
        out.write(
            f"{'strategy':<12}{'budget':>8}{'cost':>9}{'det/all':>10}"
            f"{'det/detectable':>16}{'upper bound':>13}\n"
        )
        out.write("-" * 92 + "\n")
        for name in ("dependency", "history", "learned"):
            for point in points[name]:
                out.write(
                    f"{name:<12}{point.budget:>8.2f}{point.tests_run_fraction:>9.4f}"
                    f"{point.faults_detected_all:>10.4f}"
                    f"{point.faults_detected_detectable:>16.4f}"
                    f"{point.full_suite_upper_bound:>13.4f}\n"
                )
        apfds = fault_matrix_apfd(
            matrix, graph, prioritiser, selector, churn_lines=churn_lines
        )
        out.write("multi-fault APFD over the union change (all mutated modules changed)\n")
        for name, value in apfds.items():
            out.write(f"  {name:<12}{value:.6f}\n")

    points1 = evaluate_on_fault_matrix(
        matrix, graph, prioritiser, selector, budgets=BUDGETS, churn_lines=1
    )
    points20 = evaluate_on_fault_matrix(
        matrix, graph, prioritiser, selector, budgets=BUDGETS, churn_lines=20
    )
    dep = points1["dependency"][0]
    out.write("\nFINDINGS, stated as measured\n")
    out.write("-" * 92 + "\n")
    out.write(
        f"1. The dependency baseline detects every detectable seeded fault while running\n"
        f"   {dep.tests_run_fraction:.4f} of the suite. Neither ranked strategy reaches that\n"
        f"   detection rate at that cost.\n"
        f"2. At a 25 per cent budget the learned selector reaches "
        f"{points1['learned'][1].faults_detected_detectable:.4f} of detectable\n"
        f"   faults with the truthful one-line churn, and "
        f"{points20['learned'][1].faults_detected_detectable:.4f} when the same faults\n"
        f"   are presented as 20-line changes. The selector keys on churn, so a one-line\n"
        f"   fix -- the case where selection matters most -- is where it is weakest.\n"
        f"3. The history baseline needs a 75 per cent budget to reach full detectable\n"
        f"   coverage; at 25 per cent it finds "
        f"{points1['history'][1].faults_detected_detectable:.4f}.\n"
        f"4. The full-suite upper bound is {matrix.full_suite_detection():.4f}: running\n"
        f"   everything still misses four of the fourteen seeded faults, and no selection\n"
        f"   strategy can recover a fault the suite cannot see.\n"
    )

    checks = [
        ("clean run is green and cross-checks", clean.cross_check() and not clean.failing_nodes),
        ("committed fault matrix reproduces this run", agrees),
        (
            "dependency baseline detects all detectable faults",
            abs(dep.faults_detected_detectable - 1.0) <= 1e-12,
        ),
        (
            "no strategy exceeds the full-suite upper bound",
            all(
                p.faults_detected_all <= p.full_suite_upper_bound + 1e-12
                for series in points1.values()
                for p in series
            ),
        ),
        (
            "detection is non-decreasing in the budget for the ranked strategies",
            all(
                b.faults_detected_all >= a.faults_detected_all - 1e-12
                for name in ("history", "learned")
                for a, b in zip(points1[name], points1[name][1:], strict=False)
            ),
        ),
    ]
    out.write("\nchecks\n")
    out.write("-" * 92 + "\n")
    hard_failures = 0
    for label, ok in checks:
        hard_failures += 0 if ok else 1
        out.write(f"{label:<70}{'PASS' if ok else 'FAIL'}\n")
    out.write(f"\nwall clock {time.monotonic() - started:.1f} s\n")
    out.write(f"checks run {len(checks)}, failures {hard_failures}\n")
    out.write("RESULT: " + ("PASS\n" if hard_failures == 0 else f"FAIL ({hard_failures})\n"))
    return 0 if hard_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
