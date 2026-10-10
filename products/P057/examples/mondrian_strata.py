"""Conditional coverage: what the marginal band hides and Mondrian exposes.

Writes ``screenshots/mondrian_strata.png``.
"""

from __future__ import annotations

from pathlib import Path

from conformalband.audit import stratified_coverage
from conformalband.plotting import plot_stratified_coverage

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "screenshots" / "mondrian_strata.png"

REPLICATES = 20
SEVERITY = 2.0
SEED = 57031


def main() -> None:
    tally = stratified_coverage(
        replicates=REPLICATES, severity=SEVERITY, seed=SEED, model="learned"
    )
    header = f"{'method':10s} {'tercile':>8s} {'coverage':>9s} {'width_Wh':>9s} {'points':>8s}"
    print(header)
    print("-" * len(header))
    for method in ("split", "mondrian"):
        for index in sorted(tally[method]):
            coverage, width, nominal, count = tally[method][index]
            print(
                f"{method:10s} {index + 1:8d} {coverage:9.5f} {width:9.5f} {count:8d}"
            )
    print()
    for method in ("split", "mondrian"):
        values = [tally[method][b][0] for b in sorted(tally[method])]
        print(
            f"{method:10s} coverage spread across terciles: "
            f"{max(values) - min(values):.5f} (min {min(values):.5f}, max {max(values):.5f})"
        )
    nominal = tally["split"][0][2]
    print(f"nominal coverage: {nominal}")
    path = plot_stratified_coverage(
        tally,
        OUTPUT,
        title=(
            f"Conditional coverage by tercile of predicted energy, severity {SEVERITY}, "
            f"{REPLICATES} replicates"
        ),
    )
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
