"""Covered prior mass against dwell index for the four scan patterns.

    python examples/coverage_curves.py

Writes ../screenshots/coverage_curves.png relative to this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from acqsearch import (  # noqa: E402
    PointingUncertainty,
    TerminalGeometry,
    build_cell_grid,
    coverage_curve,
    spiral_closed_form_coverage,
    spiral_closed_form_valid,
    spiral_path,
    standard_paths,
)
from acqsearch.plotting import plot_coverage_curves  # noqa: E402

SAMPLES = 60000
SEED = 20261016


def main() -> int:
    """Draw the coverage curves and print the final coverage of each pattern."""
    uncertainty = PointingUncertainty(150e-6)
    geometry = TerminalGeometry(100e-6, 200e-6, 10e-3, 2e-3)
    grid = build_cell_grid(uncertainty, geometry, extent_sigma=2.5)
    footprint = geometry.footprint_radius_rad

    curves = {}
    for name, path in standard_paths(grid).items():
        curves[name] = coverage_curve(path, uncertainty, footprint, SAMPLES, seed=SEED)
        print(f"{name:<16} dwells {path.n_points:>4}  final covered mass "
              f"{curves[name].final_coverage():.6f}")

    pitch = grid.side_rad
    spacing = 0.5 * grid.side_rad
    turns = 6
    closed = spiral_closed_form_coverage(pitch, turns, footprint, uncertainty)
    marker_path = spiral_path(pitch, float(turns), spacing)
    print(f"spiral closed form after {turns} turns ({marker_path.n_points} dwells): "
          f"{closed:.6f}, applies: {spiral_closed_form_valid(pitch, spacing, footprint)}")

    output = plot_coverage_curves(
        curves,
        ROOT / "screenshots" / "coverage_curves.png",
        closed_form=(marker_path.n_points, closed),
        title="Covered prior mass against dwell index, 89-cell reference configuration",
    )
    print(f"wrote {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
