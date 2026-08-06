"""V1 - spiral coverage vs the geometric track-spacing argument.

Claim under test
----------------
An Archimedean spiral with radial pitch s = 2 R_beam (1 - overlap) and a beam
footprint of radius R_beam leaves no radial gap when s <= 2 R_beam, i.e. for
any overlap in [0, 1). Consequently the fraction of the Gaussian uncertainty
mass covered by a spiral designed for containment C should be >= C, up to
(a) the finite along-track dwell step and (b) Monte Carlo error.

Method
------
For a grid of overlap factors, generate the spiral, then estimate the covered
probability mass by sampling 200 000 targets from the 2-D Gaussian and asking
whether each lies within R_beam of some dwell point (k-d tree query). Compare
against the design containment and against the "no-gap" geometric criterion
s <= 2 R_beam.

Run: python validation/v1_spiral_coverage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trackforge.scan import (  # noqa: E402
    GaussianUncertainty,
    coverage_fraction,
    raster_scan,
    spiral_scan,
    track_spacing,
)

SIGMA = 3.0e-4
BEAM = 2.0e-5
CONTAINMENT = 0.995
N_SAMPLES = 200_000


def main() -> int:
    """Run the coverage validation and print a table."""
    u = GaussianUncertainty(SIGMA)
    print("V1 - spiral coverage vs geometric track-spacing argument")
    print(f"sigma = {SIGMA:.3e} rad, R_beam = {BEAM:.3e} rad, "
          f"design containment = {CONTAINMENT}")
    print(f"design radius r_max = {u.containment_radius(CONTAINMENT):.6e} rad "
          f"= {u.containment_radius(CONTAINMENT) / SIGMA:.4f} sigma")
    print(f"Monte Carlo samples per case: {N_SAMPLES}")
    print()
    header = (f"{'overlap':>8} {'s [urad]':>10} {'s/2R':>7} {'no-gap':>7} "
              f"{'points':>8} {'covered':>9} {'covered-C':>10}")
    print(header)
    print("-" * len(header))
    worst_with_margin = 1.0
    cov_tangent = None
    for overlap in (0.0, 0.1, 0.25, 0.4, 0.5):
        p = spiral_scan(u, BEAM, overlap=overlap, containment=CONTAINMENT,
                        step_fraction=0.5)
        s = track_spacing(BEAM, overlap)
        cov = coverage_fraction(p, u, n_samples=N_SAMPLES, rng=np.random.default_rng(1))
        if overlap == 0.0:
            cov_tangent = cov
        else:
            worst_with_margin = min(worst_with_margin, cov)
        print(f"{overlap:8.2f} {s * 1e6:10.3f} {s / (2 * BEAM):7.3f} "
              f"{'yes' if s <= 2 * BEAM else 'NO':>7} {p.n_points:8d} "
              f"{cov:9.5f} {cov - CONTAINMENT:+10.5f}")
    print()
    print("REPORTED FINDING (not a tolerance adjustment): at overlap = 0 the tracks are")
    print("exactly tangent (s = 2 R_beam). The 1-D 'no radial gap' argument holds along")
    print("the radial direction only; because the spiral turns, tangent circles on")
    print("adjacent turns leave curvature gaps, and the measured coverage falls")
    print(f"{CONTAINMENT - cov_tangent:.5f} below the design containment "
          f"({cov_tangent:.5f} vs {CONTAINMENT}).")
    print("This is the reason an overlap margin is specified in PAT scan design; with")
    print("overlap >= 0.10 the shortfall disappears into Monte Carlo noise (see table).")
    print()

    # deliberate gap case: fly the pattern with a beam 40 % of the design size
    p = spiral_scan(u, BEAM, overlap=0.25, containment=CONTAINMENT)
    p.beam_radius = 0.4 * BEAM
    gap_cov = coverage_fraction(p, u, n_samples=N_SAMPLES, rng=np.random.default_rng(2))
    print("negative control (pattern designed for R_beam but flown with 0.4 R_beam,")
    print(f"  so s / 2R = {track_spacing(BEAM, 0.25) / (2 * 0.4 * BEAM):.3f} > 1): "
          f"covered = {gap_cov:.5f}")
    print()

    ra = raster_scan(u, BEAM, overlap=0.25, containment=CONTAINMENT)
    ra_cov = coverage_fraction(ra, u, n_samples=N_SAMPLES, rng=np.random.default_rng(3))
    sp = spiral_scan(u, BEAM, overlap=0.25, containment=CONTAINMENT)
    print(f"raster: {ra.n_points} points, covered = {ra_cov:.5f}")
    print(f"spiral: {sp.n_points} points, covered = "
          f"{coverage_fraction(sp, u, n_samples=N_SAMPLES, rng=np.random.default_rng(4)):.5f}")
    print(f"raster/spiral dwell-count ratio = {ra.n_points / sp.n_points:.3f} "
          "(raster sweeps the bounding square)")
    print()

    # MC standard error at C = 0.995, N = 2e5 is sqrt(C(1-C)/N) = 1.6e-4;
    # 1e-3 allows 6 sigma plus the along-track discretisation.
    tol = 1.0e-3
    ok = worst_with_margin >= CONTAINMENT - tol and gap_cov < 0.90
    print(f"PASS criterion (overlap >= 0.10 only): min covered >= "
          f"{CONTAINMENT - tol:.4f} and negative control < 0.90")
    print(f"min covered (overlap >= 0.10) = {worst_with_margin:.5f}; "
          f"negative control = {gap_cov:.5f}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
