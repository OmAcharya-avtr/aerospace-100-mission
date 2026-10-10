"""The coverage audit: empirical coverage against the declared shift severity.

Writes ``screenshots/coverage_audit.png`` and prints the table it was drawn
from. Runtime is about one minute on two contended cores.
"""

from __future__ import annotations

from pathlib import Path

from conformalband.audit import coverage_audit
from conformalband.plotting import plot_coverage_audit

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "screenshots" / "coverage_audit.png"

REPLICATES = 40
N_TEST = 600
SEED = 57001


def main() -> None:
    result = coverage_audit(replicates=REPLICATES, n_test=N_TEST, seed=SEED)
    print(result.table())
    print()
    nominal = 1.0 - result.alpha
    failures = [row for row in result.rows if row.below_nominal]
    print(f"rows demonstrably below nominal coverage {nominal}: {len(failures)}/{len(result.rows)}")
    for row in failures:
        print(
            f"  {row.model}/{row.method} severity {row.severity:.1f}: "
            f"coverage {row.coverage:.5f} CI [{row.ci_low:.5f}, {row.ci_high:.5f}]"
        )
    path = plot_coverage_audit(result, OUTPUT)
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
