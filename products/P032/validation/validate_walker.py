"""Walker constellation geometry: what the generator actually produces.

Why this check exists
---------------------
:func:`constellink.constellation.walker_delta` sets the SGP4 mean motion from
the two-body relation ``n = sqrt(mu / a^3)``.  SGP4 consumes the
Brouwer-Lyddane "Kozai" mean motion, which differs from the Kepler mean motion
by the J2 secular correction, so the realised orbit radius is NOT exactly the
requested one.  The offset is small but it is real, and a tool that silently
hands back a different altitude from the one asked for is a tool that will be
blamed for someone else's error later.

This script measures:

1. the realised geocentric radius against the requested radius, over a full
   orbit, for a range of altitudes;
2. the realised inclination, RAAN spacing and in-plane mean-anomaly spacing
   against the Walker definition (Walker 1984, "Satellite constellations",
   J. British Interplanetary Society 37, 559-572; also Wertz & Larson (eds.)
   1999, "Space Mission Analysis and Design", 3rd ed., Ch. 7);
3. the radius variation over one orbit, which bounds the eccentricity the
   propagator introduces.

No tolerance here is a physical accuracy claim.  They are declared bounds on
the generator's own self-consistency, and the measured numbers are reported so
a user can decide whether the offset matters for their case.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from constellink.constellation import walker_delta  # noqa: E402
from constellink.frames import WGS84_A_KM  # noqa: E402

EPOCH = datetime(2026, 4, 1, tzinfo=UTC)
RADIUS_OFFSET_TOL_KM = 5.0
RADIUS_VARIATION_TOL_KM = 15.0
ANGLE_TOL_DEG = 1e-6


def main() -> int:
    print("Walker constellation generator: realised vs requested geometry")
    print("=" * 86)
    print("")

    print("Part 1 -- realised radius vs requested (Kozai vs Kepler mean motion)")
    print(f"  declared bounds: mean radius offset <= {RADIUS_OFFSET_TOL_KM:g} km, "
          f"peak-to-peak radius variation over one orbit "
          f"<= {RADIUS_VARIATION_TOL_KM:g} km")
    head = (f"  {'alt [km]':>10}{'requested r [km]':>19}{'mean r [km]':>15}"
            f"{'offset [km]':>14}{'p-p var [km]':>14}{'pass':>6}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    ok1 = True
    for alt in (400.0, 550.0, 800.0, 1200.0):
        const = walker_delta(4, 2, 1, 53.0, alt, EPOCH, max_epoch_age_days=2.0)
        r_req = WGS84_A_KM + alt
        sat = const.satellites[0]
        period_s = 2.0 * np.pi / (sat.satrec.no_kozai / 60.0)
        times = [EPOCH + timedelta(seconds=period_s * k / 120.0)
                 for k in range(121)]
        r, _ = sat.propagate(times)
        radii = np.linalg.norm(r, axis=1)
        mean_r = float(radii.mean())
        offset = mean_r - r_req
        var = float(radii.max() - radii.min())
        good = abs(offset) <= RADIUS_OFFSET_TOL_KM and var <= RADIUS_VARIATION_TOL_KM
        ok1 &= good
        print(f"  {alt:>10.0f}{r_req:>19.3f}{mean_r:>15.3f}{offset:>14.3f}"
              f"{var:>14.3f}{'PASS' if good else 'FAIL':>6}")
    print(f"  result: {'PASS' if ok1 else 'FAIL'}")
    print("")

    print("Part 2 -- Walker pattern angles against the i:T/P/F definition")
    t_total, p_planes, f_phase, inc = 24, 4, 1, 53.0
    const = walker_delta(t_total, p_planes, f_phase, inc, 550.0, EPOCH,
                         max_epoch_age_days=2.0)
    per_plane = t_total // p_planes
    print(f"  pattern: {inc:g} deg : {t_total}/{p_planes}/{f_phase} "
          f"({per_plane} satellites per plane)")
    print(f"  tolerance: {ANGLE_TOL_DEG:g} deg on every angle")
    ok2 = True
    worst = 0.0
    head = (f"  {'satellite':<12}{'inc [deg]':>12}{'RAAN [deg]':>12}"
            f"{'exp RAAN':>11}{'M [deg]':>12}{'exp M':>11}{'pass':>6}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for idx, sat in enumerate(const.satellites):
        plane, slot = divmod(idx, per_plane)
        exp_raan = 360.0 * plane / p_planes
        exp_m = (360.0 * slot / per_plane
                 + 360.0 * f_phase * plane / t_total) % 360.0
        got_inc = float(np.rad2deg(sat.satrec.inclo))
        got_raan = float(np.rad2deg(sat.satrec.nodeo)) % 360.0
        got_m = float(np.rad2deg(sat.satrec.mo)) % 360.0
        errs = [abs(got_inc - inc), abs(got_raan - exp_raan), abs(got_m - exp_m)]
        worst = max(worst, max(errs))
        good = max(errs) <= ANGLE_TOL_DEG
        ok2 &= good
        if idx < 8 or not good:
            print(f"  {sat.name:<12}{got_inc:>12.7f}{got_raan:>12.7f}"
                  f"{exp_raan:>11.4f}{got_m:>12.7f}{exp_m:>11.4f}"
                  f"{'PASS' if good else 'FAIL':>6}")
    print(f"  ... {max(0, len(const.satellites) - 8)} further satellites checked")
    print(f"  worst angle error over all {len(const.satellites)} satellites: "
          f"{worst:.3e} deg")
    print(f"  result: {'PASS' if ok2 else 'FAIL'}")
    print("")

    print("Part 3 -- intra-plane ISL separation is constant (a design invariant)")
    eph = const.ephemeris(EPOCH, EPOCH + timedelta(hours=2), 120.0)
    i0, i1 = eph.index("W00-00"), eph.index("W00-01")
    ra, rb = eph.r_teme_km[i0], eph.r_teme_km[i1]
    sep = np.linalg.norm(rb - ra, axis=1)
    print(f"  W00-00 to W00-01 range: mean {sep.mean():.3f} km, "
          f"p-p variation {sep.max() - sep.min():.3f} km")
    expected = 2.0 * (WGS84_A_KM + 550.0) * np.sin(np.pi / per_plane)
    print(f"  chord for a perfect circle at the requested radius: "
          f"{expected:.3f} km")
    print(f"  difference from that chord: {sep.mean() - expected:+.3f} km")
    print("  (reported, not pass/fail: the difference is the same Kozai/Kepler")
    print("   radius offset measured in Part 1, scaled by the chord geometry)")
    print("")

    overall = ok1 and ok2
    print("=" * 86)
    print(f"part 1 (realised radius) : {'PASS' if ok1 else 'FAIL'}")
    print(f"part 2 (pattern angles)  : {'PASS' if ok2 else 'FAIL'}")
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
