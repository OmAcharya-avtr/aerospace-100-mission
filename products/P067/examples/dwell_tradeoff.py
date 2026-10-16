"""The dwell-time trade: expected acquisition time against dwell time.

    python examples/dwell_tradeoff.py

Writes ../screenshots/dwell_tradeoff.png relative to this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from acqsearch import (  # noqa: E402
    PointingUncertainty,
    PoissonDwellDetector,
    TerminalGeometry,
    build_cell_grid,
    cell_visit_order,
    optimal_dwell_time,
    standard_paths,
)
from acqsearch.plotting import plot_dwell_tradeoff  # noqa: E402


def main() -> int:
    """Sweep the dwell time and draw the trade."""
    uncertainty = PointingUncertainty(150e-6)
    geometry = TerminalGeometry(100e-6, 200e-6, 10e-3, 2e-3)
    detector = PoissonDwellDetector(2.0e4, 2.0e3, 8, 0.2)
    grid = build_cell_grid(uncertainty, geometry, extent_sigma=2.5)
    order = cell_visit_order(standard_paths(grid)["spiral"], grid)

    optimum = optimal_dwell_time(
        grid, order, geometry, detector, uncertainty, np.geomspace(2.0e-4, 2.0e-2, 41), 0.5
    )
    print(f"grid minimum {optimum.best_expected_time_s:.6f} s at "
          f"{optimum.best_dwell_s * 1e3:.4f} ms")
    print(f"E[T] at 0.2000 ms {optimum.expected_time_s[0]:.6f} s")
    print(f"E[T] at 20.0000 ms {optimum.expected_time_s[-1]:.6f} s")
    index = int(np.argmin(np.abs(optimum.dwell_grid_s - optimum.best_dwell_s)))
    print(f"at the optimum: mean q {optimum.detection_probability[index]:.6f}, "
          f"P_fa {optimum.false_alarm_probability[index]:.6f}")

    output = plot_dwell_tradeoff(
        optimum,
        ROOT / "screenshots" / "dwell_tradeoff.png",
        title="Dwell-time trade, spiral order over the 89-cell reference grid",
    )
    print(f"wrote {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
