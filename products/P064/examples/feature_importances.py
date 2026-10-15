"""What the learned selector actually keys on.

Writes ../screenshots/feature_importances.png.

    python examples/feature_importances.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regressselect.history import load_history  # noqa: E402
from regressselect.importgraph import build_import_graph  # noqa: E402
from regressselect.plotting import plot_feature_importances  # noqa: E402
from regressselect.selector import LearnedSelector  # noqa: E402


def main() -> int:
    history = load_history(ROOT / "fixtures" / "history" / "history.json")
    graph = build_import_graph(ROOT / "fixtures" / "sample_project")
    train, test = history.split_chronological(0.70)
    selector = LearnedSelector().fit(train, graph)
    importances = selector.feature_importances()

    sys.stdout.write(
        f"training rows={selector.n_train_rows} positives={selector.n_train_positives} "
        f"({selector.n_train_positives / selector.n_train_rows:.4f} of rows)\n"
        f"Brier score: train={selector.brier_score(train):.6f} "
        f"test={selector.brier_score(test):.6f}\n\n"
    )
    for name, value in sorted(importances.items(), key=lambda kv: -kv[1]):
        sys.stdout.write(f"  {name:<28}{value:.4f}\n")

    out = ROOT / "screenshots" / "feature_importances.png"
    plot_feature_importances(importances, out)
    sys.stdout.write(f"\nwrote {out.relative_to(ROOT)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
