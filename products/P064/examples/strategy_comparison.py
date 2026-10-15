"""Compare the three strategies on the committed fixture history.

Writes ../screenshots/strategy_comparison.png. Run from the product root:

    python examples/strategy_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regressselect.benchmark import evaluate_strategies  # noqa: E402
from regressselect.history import load_history  # noqa: E402
from regressselect.importgraph import build_import_graph  # noqa: E402
from regressselect.plotting import plot_safety_loss_vs_cost  # noqa: E402

BUDGETS = (0.10, 0.25, 0.4167, 0.50, 0.75, 1.00)


def main() -> int:
    history = load_history(ROOT / "fixtures" / "history" / "history.json")
    graph = build_import_graph(ROOT / "fixtures" / "sample_project")
    train, test = history.split_chronological(0.70)
    scores = evaluate_strategies(train, test, graph, budgets=BUDGETS)

    out = ROOT / "screenshots" / "strategy_comparison.png"
    plot_safety_loss_vs_cost(
        scores,
        out,
        title=(
            f"{len(train)} training revisions, {len(test)} held out, "
            f"{scores['learned'].revisions_scored} with a real failure"
        ),
    )

    sys.stdout.write(
        f"train={len(train)} test={len(test)} "
        f"evaluation revisions with a failure={scores['learned'].revisions_scored}\n"
    )
    sys.stdout.write(f"{'strategy':<12}{'APFD':>9}{'budget':>9}{'cost':>9}{'loss':>9}\n")
    for name, score in scores.items():
        for point in score.points:
            sys.stdout.write(
                f"{name:<12}{score.apfd_full_order:>9.4f}{point.budget:>9.4f}"
                f"{point.tests_run_fraction:>9.4f}{point.safety_loss:>9.4f}\n"
            )
    sys.stdout.write(f"wrote {out.relative_to(ROOT)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
