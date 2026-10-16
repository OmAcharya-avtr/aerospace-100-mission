"""Spiral coverage: the geometric closed form against seeded Monte Carlo.

    python validation/validate_spiral_coverage.py > validation/validate_spiral_coverage_output.txt

The closed form is derived in :func:`acqsearch.coverage.spiral_closed_form_coverage`
from the Archimedean spiral ``r = p theta / (2 pi)`` and the Rayleigh radial law
of an isotropic bivariate normal (Papoulis & Pillai 2002, Ch. 6). The Monte
Carlo draws from the same prior and tests membership of the union of footprint
discs, so the two share no code path beyond the prior sampler.
"""

from __future__ import annotations

import math
import sys

import numpy as np
from _common import SIGMA_RAD, rule

from acqsearch import (
    PointingUncertainty,
    coverage_curve,
    spiral_closed_form_coverage,
    spiral_closed_form_valid,
    spiral_path,
)

SAMPLES = 200000
SEED = 20261016


def main() -> int:
    """Write the coverage evidence."""
    out = sys.stdout
    failures = 0
    uncertainty = PointingUncertainty(SIGMA_RAD)
    a = 50e-6
    rule(out, "acqsearch spiral coverage: closed form against Monte Carlo")
    out.write(f"sigma {SIGMA_RAD:.6e} rad   footprint radius {a:.6e} rad\n")
    out.write(f"Monte Carlo samples {SAMPLES}   seed {SEED}\n")

    for pitch, spacing in ((math.sqrt(2.0) * a, 0.5 * math.sqrt(2.0) * a), (6.0e-5, 3.0e-5)):
        valid = spiral_closed_form_valid(pitch, spacing, a)
        limit = 2.0 * math.sqrt(a**2 - (0.5 * spacing) ** 2)
        out.write(
            f"\npitch {pitch * 1e6:.4f} urad, along-track {spacing * 1e6:.4f} urad; "
            f"inter-turn limit 2 sqrt(a^2 - (d/2)^2) = {limit * 1e6:.4f} urad; "
            f"closed form applies: {valid}\n"
        )
        header = f"{'turns':>7}{'dwells':>8}{'closed form':>14}{'Monte Carlo':>14}"
        out.write(header + f"{'difference':>13}{'MC s.e.':>11}{'|diff|/s.e.':>13}\n")
        for turns in range(1, 9):
            closed = spiral_closed_form_coverage(pitch, turns, a, uncertainty)
            path = spiral_path(pitch, float(turns), spacing)
            curve = coverage_curve(path, uncertainty, a, n_samples=SAMPLES, seed=SEED)
            measured = curve.final_coverage()
            se = float(curve.standard_error[-1])
            difference = measured - closed
            ratio = abs(difference) / se if se > 0 else float("inf")
            failures += abs(difference) > 3.0 * se + 2.0e-3
            out.write(
                f"{turns:>7d}{path.n_points:>8d}{closed:>14.6f}{measured:>14.6f}"
                f"{difference:>+13.6f}{se:>11.6f}{ratio:>13.2f}\n"
            )

    out.write("\nLimiting cases of the closed form\n")
    out.write(f"{'case':<46}{'closed form':>14}{'exact':>14}{'error':>12}\n")
    tiny = spiral_closed_form_coverage(1.0e-9, 1, a, uncertainty)
    exact = uncertainty.radial_cdf(a)
    out.write(
        f"{'pitch -> 0, one turn: the single footprint disc':<46}"
        f"{tiny:>14.9f}{exact:>14.9f}{abs(tiny - exact):>12.2e}\n"
    )
    big = spiral_closed_form_coverage(math.sqrt(2.0) * a, 40, a, uncertainty)
    out.write(
        f"{'40 turns: the whole prior':<46}{big:>14.9f}{1.0:>14.9f}{abs(big - 1.0):>12.2e}\n"
    )

    out.write("\nCoverage of the four standard paths over the reference grid\n")
    from _common import reference

    _u, geometry, _detector, grid = reference()
    from acqsearch import standard_paths

    out.write(f"{'pattern':<16}{'dwells':>8}{'final covered mass':>22}{'s.e.':>10}\n")
    for name, path in standard_paths(grid).items():
        curve = coverage_curve(
            path, uncertainty, geometry.footprint_radius_rad, n_samples=60000, seed=SEED
        )
        out.write(
            f"{name:<16}{path.n_points:>8d}{curve.final_coverage():>22.6f}"
            f"{float(curve.standard_error[-1]):>10.6f}\n"
        )
    out.write(f"prior mass inside the 89-cell grid: {float(np.sum(grid.prior_mass)):.6f}\n")

    rule(out)
    out.write(f"checks failed: {failures}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
