"""The finite-sample coverage bound as a staircase in the calibration size.

Writes ``screenshots/coverage_bound.png``. No data and no model: this figure is
exact arithmetic.
"""

from __future__ import annotations

from pathlib import Path

from conformalband.bounds import split_conformal_coverage_bound
from conformalband.plotting import plot_coverage_bound

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "screenshots" / "coverage_bound.png"

ALPHA = 0.1
SIZES = (9, 19, 20, 49, 99, 100, 200, 500, 1000)


def main() -> None:
    header = f"{'n':>6s} {'rank k':>7s} {'exact':>14s} {'1-alpha':>9s} {'upper':>14s} {'gap':>13s}"
    print(header)
    print("-" * len(header))
    for n in SIZES:
        bound = split_conformal_coverage_bound(n, ALPHA)
        print(
            f"{bound.n_calibration:6d} {bound.rank:7d} {bound.exact:14.12f} "
            f"{bound.lower:9.4f} {bound.upper:14.12f} {bound.conservatism:13.12f}"
        )
    print()
    print(
        "The exact value is k/(n+1) with k = ceil((n+1)(1-alpha)); it equals 1-alpha only "
        "when (n+1)(1-alpha) is an integer, and otherwise over-covers by less than 1/(n+1)."
    )
    path = plot_coverage_bound(OUTPUT, alpha=ALPHA, max_n=200)
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
