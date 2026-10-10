"""Where weighted conformal stops holding when its weights are wrong.

Writes ``screenshots/weight_breaking_point.png`` and prints the sweep with the
breaking point named.
"""

from __future__ import annotations

from pathlib import Path

from conformalband.audit import breaking_point_sweep
from conformalband.plotting import plot_breaking_point

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "screenshots" / "weight_breaking_point.png"

REPLICATES = 40
N_TEST = 600
SEED = 57021
TRUE_SEVERITY = 2.0


def main() -> None:
    result = breaking_point_sweep(
        true_severity=TRUE_SEVERITY,
        replicates=REPLICATES,
        n_test=N_TEST,
        seed=SEED,
        model="learned",
    )
    print(result.table())
    print()
    print(f"true severity                      : {result.true_severity}")
    print(f"smallest fraction still holding    : {result.last_holding_fraction}")
    print(f"breaking fraction                  : {result.breaking_fraction}")
    unweighted = result.rows[0]
    correct = [row for row in result.rows if row.fraction == 1.0][0]
    print(
        f"no weighting at all (f=0.00)       : coverage {unweighted.coverage:.5f}, "
        f"deficit {correct.coverage - unweighted.coverage:.5f} against the correct weights"
    )
    print(
        f"correct weights (f=1.00)           : coverage {correct.coverage:.5f}, "
        f"width {correct.mean_width:.5f} Wh, ESS {correct.ess:.1f}"
    )
    path = plot_breaking_point(result, OUTPUT)
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
