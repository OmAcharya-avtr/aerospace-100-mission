"""The dwell-time trade and its grid optimum.

    python validation/validate_dwell_optimum.py > validation/validate_dwell_optimum_output.txt

A longer dwell raises the detection probability (more signal counts), raises
the false-alarm probability (more background counts past the same fixed
threshold, Gagliardi & Karp 1995 Ch. 5) and raises the per-dwell cost
directly. The expected acquisition time therefore has an interior minimum.
The optimum is a GRID optimum: no convexity is assumed and none is claimed.
"""

from __future__ import annotations

import sys

import numpy as np
from _common import VERIFICATION_TIME_S, reference, rule

from acqsearch import (
    PoissonDwellDetector,
    cell_visit_order,
    optimal_dwell_time,
    standard_paths,
)


def main() -> int:
    """Write the dwell-trade evidence."""
    out = sys.stdout
    uncertainty, geometry, detector, grid = reference()
    order = cell_visit_order(standard_paths(grid)["spiral"], grid)
    rule(out, "acqsearch dwell-time trade, spiral order over the 89-cell grid")

    dwells = np.geomspace(2.0e-4, 2.0e-2, 41)
    optimum = optimal_dwell_time(
        grid, order, geometry, detector, uncertainty, dwells, VERIFICATION_TIME_S
    )
    out.write(f"{'dwell_ms':>10}{'E[T]_s':>14}{'mean q':>10}{'P_fa':>14}\n")
    for tau, expected, q, f in zip(
        optimum.dwell_grid_s,
        optimum.expected_time_s,
        optimum.detection_probability,
        optimum.false_alarm_probability,
        strict=True,
    ):
        out.write(f"{tau * 1e3:>10.4f}{expected:>14.6f}{q:>10.6f}{f:>14.6e}\n")
    out.write(
        f"\ngrid minimum {optimum.best_expected_time_s:.6f} s at "
        f"{optimum.best_dwell_s * 1e3:.4f} ms\n"
    )
    out.write(f"E[T] at the shortest dwell  {optimum.expected_time_s[0]:.6f} s\n")
    out.write(f"E[T] at the longest dwell   {optimum.expected_time_s[-1]:.6f} s\n")
    ratio = optimum.expected_time_s.max() / optimum.best_expected_time_s
    out.write(f"worst / best over the sweep {ratio:.2f}\n")
    out.write(
        "Monotonicity over the sweep: "
        f"q non-decreasing {bool(np.all(np.diff(optimum.detection_probability) >= -1e-12))}, "
        f"P_fa non-decreasing {bool(np.all(np.diff(optimum.false_alarm_probability) >= -1e-12))}\n"
    )

    out.write("\nSensitivity of the optimum to the count threshold\n")
    out.write(f"{'threshold':>10}{'best dwell_ms':>16}{'best E[T]_s':>14}{'P_fa at best':>15}\n")
    for threshold in (4, 6, 8, 10, 12):
        trial = PoissonDwellDetector(
            detector.signal_rate_hz,
            detector.background_rate_hz,
            threshold,
            detector.rytov_variance,
        )
        result = optimal_dwell_time(
            grid, order, geometry, trial, uncertainty, dwells, VERIFICATION_TIME_S
        )
        out.write(
            f"{threshold:>10d}{result.best_dwell_s * 1e3:>16.4f}"
            f"{result.best_expected_time_s:>14.6f}"
            f"{trial.false_alarm_probability(result.best_dwell_s):>15.6e}\n"
        )

    out.write("\nSensitivity of the optimum to the verification time\n")
    out.write(f"{'t_v (s)':>10}{'best dwell_ms':>16}{'best E[T]_s':>14}\n")
    for verification in (0.0, 0.1, 0.5, 2.0, 10.0):
        result = optimal_dwell_time(
            grid, order, geometry, detector, uncertainty, dwells, verification
        )
        out.write(
            f"{verification:>10.2f}{result.best_dwell_s * 1e3:>16.4f}"
            f"{result.best_expected_time_s:>14.6f}\n"
        )

    out.write("\nSensitivity of the optimum to the Rytov variance\n")
    out.write(f"{'rytov':>10}{'best dwell_ms':>16}{'best E[T]_s':>14}\n")
    for rytov in (0.0, 0.1, 0.2, 0.4, 0.8):
        trial = PoissonDwellDetector(
            detector.signal_rate_hz, detector.background_rate_hz, 8, rytov
        )
        result = optimal_dwell_time(
            grid, order, geometry, trial, uncertainty, dwells, VERIFICATION_TIME_S
        )
        out.write(
            f"{rytov:>10.2f}{result.best_dwell_s * 1e3:>16.4f}"
            f"{result.best_expected_time_s:>14.6f}\n"
        )

    rule(out)
    out.write("checks failed: 0\n")
    out.write(
        "Nothing in this file is a pass/fail check: it is the shape of the trade.\n"
        "The assertions on it are in tests/test_acquisition.py.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
