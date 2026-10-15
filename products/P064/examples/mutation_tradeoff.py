"""Seeded-fault detection against tests run, with the full-suite upper bound.

Runs the fourteen seeded faults against the bundled sample project, so it
starts fourteen pytest processes. Measured cost on the two-core build machine
is roughly twelve seconds. Writes ../screenshots/mutation_tradeoff.png.

    python examples/mutation_tradeoff.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regressselect.benchmark import evaluate_on_fault_matrix, fault_matrix_apfd  # noqa: E402
from regressselect.history import load_history  # noqa: E402
from regressselect.importgraph import build_import_graph  # noqa: E402
from regressselect.mutation import load_faults, run_mutation_evaluation  # noqa: E402
from regressselect.plotting import plot_fault_detection  # noqa: E402
from regressselect.prioritise import CoFailurePrioritiser  # noqa: E402
from regressselect.selector import LearnedSelector  # noqa: E402

BUDGETS = (0.10, 0.25, 0.50, 0.75, 1.00)


def main() -> int:
    sample = ROOT / "fixtures" / "sample_project"
    graph = build_import_graph(sample)
    history = load_history(ROOT / "fixtures" / "history" / "history.json")
    train, _ = history.split_chronological(0.70)
    prioritiser = CoFailurePrioritiser().fit(train)
    selector = LearnedSelector().fit(train, graph)

    started = time.monotonic()
    matrix = run_mutation_evaluation(sample, load_faults(ROOT / "fixtures/faults/faults.json"))
    elapsed = time.monotonic() - started

    undetected = [f for f in matrix.fault_ids if not matrix.detecting.get(f)]
    sys.stdout.write(
        f"{len(matrix.fault_ids)} seeded faults evaluated in {elapsed:.1f} s\n"
        f"full-suite detection = {matrix.full_suite_detection():.4f} "
        f"({len(matrix.detected_fault_ids)}/{len(matrix.fault_ids)})\n"
        f"detected by no test: {undetected}\n\n"
    )

    points = evaluate_on_fault_matrix(
        matrix, graph, prioritiser, selector, budgets=BUDGETS, churn_lines=1
    )
    sys.stdout.write(f"{'strategy':<12}{'budget':>8}{'cost':>9}{'det/all':>9}{'det/able':>10}\n")
    for name, series in points.items():
        for point in series:
            sys.stdout.write(
                f"{name:<12}{point.budget:>8.2f}{point.tests_run_fraction:>9.4f}"
                f"{point.faults_detected_all:>9.4f}{point.faults_detected_detectable:>10.4f}\n"
            )
    apfds = fault_matrix_apfd(matrix, graph, prioritiser, selector, churn_lines=1)
    sys.stdout.write("\nmulti-fault APFD over the union change:\n")
    for name, value in apfds.items():
        sys.stdout.write(f"  {name:<12}{value:.4f}\n")

    out = ROOT / "screenshots" / "mutation_tradeoff.png"
    plot_fault_detection(
        points,
        out,
        title="Seeded-fault detection, bundled sample project, churn = 1 line",
    )
    sys.stdout.write(f"wrote {out.relative_to(ROOT)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
