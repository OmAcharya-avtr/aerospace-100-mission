"""Interval width is the price of coverage, measured.

Writes ``screenshots/interval_width.png`` and prints the width table,
including the in-distribution comparison between the parametric baseline and
split conformal that this product's specification expected to go the other
way.
"""

from __future__ import annotations

from pathlib import Path

from conformalband.audit import coverage_audit
from conformalband.plotting import plot_interval_width

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "screenshots" / "interval_width.png"

REPLICATES = 30
N_TEST = 600
SEED = 57011


def main() -> None:
    result = coverage_audit(replicates=REPLICATES, n_test=N_TEST, seed=SEED)
    header = f"{'model':8s} {'method':18s} {'sev':>4s} {'width_Wh':>9s} {'coverage':>9s}"
    print(header)
    print("-" * len(header))
    for row in result.rows:
        print(
            f"{row.model:8s} {row.method:18s} {row.severity:4.1f} "
            f"{row.mean_width:9.5f} {row.coverage:9.5f}"
        )
    print()
    for model in sorted({row.model for row in result.rows}):
        parametric = result.select(model=model, method="parametric")[0]
        split = result.select(model=model, method="split")[0]
        ratio = parametric.mean_width / split.mean_width
        print(
            f"{model}: in distribution the parametric baseline is {ratio:.6f} times the "
            f"width of split conformal ({parametric.mean_width:.6f} against "
            f"{split.mean_width:.6f} Wh)"
        )
    path = plot_interval_width(result, OUTPUT)
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
