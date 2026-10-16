"""The worked example quoted in README.md, run end to end.

    python validation/worked_example.py > validation/worked_example_output.txt
"""

from __future__ import annotations

import sys

import numpy as np
from _common import rule

from acqsearch import (
    PointingUncertainty,
    PoissonDwellDetector,
    TerminalGeometry,
    build_cell_grid,
    build_schedule,
    cell_detection_probabilities,
    cell_visit_order,
    expected_acquisition_time,
    monte_carlo_acquisition,
    probability_acquired_by,
    spiral_closed_form_coverage,
    standard_paths,
)


def main() -> int:
    """Run the README snippet and print exactly what it prints."""
    out = sys.stdout
    rule(out, "acqsearch worked example")

    uncertainty = PointingUncertainty(sigma_rad=150e-6)
    geometry = TerminalGeometry(
        beam_divergence_rad=100e-6,
        detector_fov_rad=200e-6,
        slew_rate_rad_s=10e-3,
        settle_time_s=2e-3,
    )
    detector = PoissonDwellDetector(
        signal_rate_hz=2.0e4,
        background_rate_hz=2.0e3,
        threshold_counts=8,
        rytov_variance=0.2,
    )

    grid = build_cell_grid(uncertainty, geometry, extent_sigma=2.5)
    dwell_s = 1.4e-3
    q = cell_detection_probabilities(grid, geometry, detector, dwell_s, uncertainty)
    f = detector.false_alarm_probability(dwell_s)

    out.write(f"cells {grid.n_cells}, side {grid.side_rad * 1e6:.3f} urad, ")
    out.write(f"prior mass inside {grid.prior_mass.sum():.6f}\n")
    mean_q = float(np.average(q, weights=grid.prior_mass))
    out.write(f"mean cell detection probability {mean_q:.6f}")
    out.write(f", per-dwell false alarm probability {f:.6f}\n")

    for name, path in standard_paths(grid).items():
        order = cell_visit_order(path, grid)
        schedule = build_schedule(grid, order, geometry, dwell_s, f, 0.5)
        estimate = expected_acquisition_time(grid, schedule, q)
        by_deadline = probability_acquired_by(grid, schedule, q, 2.0)
        out.write(
            f"{name:<16} period {schedule.period_s:7.4f} s   "
            f"E[T] {estimate.expected_time_s:7.4f} s   P(acquire by 2 s) {by_deadline:.4f}\n"
        )

    order = cell_visit_order(standard_paths(grid)["spiral"], grid)
    schedule = build_schedule(grid, order, geometry, dwell_s, f, 0.5)
    mc = monte_carlo_acquisition(grid, schedule, q, n_trials=200000, seed=20261016)
    out.write(
        f"spiral Monte Carlo cross-check {mc.mean_time_s:.6f} s "
        f"+- {mc.standard_error_s:.6f} s over {mc.n_inside} in-grid trials\n"
    )

    closed = spiral_closed_form_coverage(grid.side_rad, 6, geometry.footprint_radius_rad,
                                         uncertainty)
    out.write(f"spiral coverage closed form after 6 turns {closed:.6f}\n")

    rule(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
