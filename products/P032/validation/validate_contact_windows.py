"""Contact windows against the closed-form central-angle expressions.

Three independent checks, all on a two-body circular orbit so that the only
approximation left in the comparison is the window finder itself.

Check 1 -- ground access, central angle at the window edges
-----------------------------------------------------------
For a spherical Earth of radius ``R_e``, a station at zero altitude and a
satellite at radius ``r`` seen at minimum elevation ``eps``, the Earth central
angle between the sub-satellite point and the station at the access boundary is

    lambda_max = arccos( (R_e / r) cos eps ) - eps

(Wertz & Larson (eds.) 1999, "Space Mission Analysis and Design", 3rd ed.,
Ch. 5).  The script finds the windows numerically with
:func:`constellink.contacts.contact_windows_ground`, then evaluates the
central angle between the station's Earth-fixed position and the satellite's
Earth-fixed position at the reported rise and set times, and compares it with
``lambda_max``.

The station is placed on the equator at zero altitude, where the WGS-84
ellipsoid radius equals the equatorial radius exactly, so the spherical
closed form and the ellipsoidal site-position algorithm the library uses agree
by construction rather than approximately.  The orbit is equatorial, which
keeps the geometry in one plane.

Check 2 -- ground access, pass duration
---------------------------------------
For the same equatorial geometry the satellite's longitude relative to the
station advances at ``n - omega_E``, so the pass duration is

    T = 2 lambda_max / (n - omega_E)

with ``n = sqrt(mu / r^3)`` the two-body mean motion.  Two values of
``omega_E`` are reported: the WGS-84 defining sidereal rate, and the rate
implied by the IAU 1982 GMST polynomial the library actually rotates with.
The difference between them is the dominant term in the residual, which is why
both are shown instead of one being quietly preferred.

Check 3 -- inter-satellite Earth-limb clearance
-----------------------------------------------
For two satellites at the same radius ``r`` the chord's closest approach to
the Earth's centre is ``r cos(gamma / 2)``, so the link is clear iff

    gamma <= gamma_max = 2 arccos(r_block / r),   r_block = R_e + h_grazing

The script sweeps the central-angle separation of two co-planar circular
satellites, bisects the separation at which
:func:`constellink.geometry.isl_clear` flips, and compares that with
:func:`constellink.geometry.max_isl_central_angle`.  It then takes a genuine
time-varying ISL (two satellites at slightly different radii), finds its
windows, and checks that the segment's minimum geocentric radius at the
reported window edges equals ``r_block``.
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from constellink.constellation import (  # noqa: E402
    CircularOrbit,
    Ephemeris,
    GroundStation,
    time_grid,
)
from constellink.contacts import contact_windows_ground, contact_windows_isl  # noqa: E402
from constellink.frames import (  # noqa: E402
    EARTH_ROTATION_RAD_S,
    MU_EARTH_KM3_S2,
    WGS84_A_KM,
    datetime_to_jd,
    geodetic_to_ecef,
    teme_to_ecef,
)
from constellink.geometry import (  # noqa: E402
    DEFAULT_GRAZING_ALTITUDE_KM,
    central_angle,
    ground_max_central_angle,
    isl_clear,
    max_isl_central_angle,
    segment_min_radius,
)

EPOCH = datetime(2026, 4, 1, tzinfo=UTC)
ALT_KM = 550.0
R_ORBIT = WGS84_A_KM + ALT_KM
MASK_DEG = 10.0

# Tolerances, each justified where it is used.
EDGE_ANGLE_TOL_DEG = 1e-4
DURATION_TOL_S = 0.5
GAMMA_TOL_DEG = 1e-5
LIMB_TOL_KM = 1e-3

# GMST advances at 360.98564736629 deg per Julian day (IAU 1982 polynomial,
# linear term; Aoki et al. 1982 / Meeus 1998 Eq. 12.4).
GMST_RATE_RAD_S = np.deg2rad(360.98564736629) / 86400.0


def check_ground() -> tuple[bool, bool]:
    """Checks 1 and 2."""
    orbit = CircularOrbit(radius_km=R_ORBIT, inclination_deg=0.0, raan_deg=0.0,
                          arg_lat0_deg=0.0, epoch=EPOCH, name="TWOBODY")
    station = GroundStation("EQ", 0.0, 0.0, 0.0, MASK_DEG)
    eph = orbit.ephemeris(EPOCH, EPOCH + timedelta(hours=6), 30.0)
    windows = contact_windows_ground(eph, orbit, station, max_range_km=1.0e5,
                                     refine_tol_s=1e-4)
    lam_max = ground_max_central_angle(R_ORBIT, MASK_DEG)

    print("Check 1 -- central angle at the window edges")
    print(f"  orbit radius          : {R_ORBIT:.3f} km ({ALT_KM:g} km altitude)")
    print(f"  elevation mask        : {MASK_DEG:g} deg")
    print(f"  lambda_max (closed)   : {np.rad2deg(lam_max):.9f} deg")
    print(f"  windows found         : {len(windows)} "
          f"(6 h horizon, 30 s scan, 1e-4 s bisection)")
    print(f"  tolerance             : {EDGE_ANGLE_TOL_DEG:g} deg "
          f"(the 1e-4 s bisection tolerance moves the sub-satellite point by "
          f"~7e-4 km, i.e. ~6e-6 deg of central angle)")
    head = f"  {'window':>7}{'edge':>7}{'gamma [deg]':>16}{'residual [deg]':>18}{'pass':>6}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    r_site = geodetic_to_ecef(station.lat_deg, station.lon_deg, station.alt_km)
    ok1 = True
    for k, w in enumerate(windows):
        if w.clipped_start or w.clipped_end:
            continue
        for label, t in (("rise", w.t_open), ("set", w.t_close)):
            r_teme, _ = orbit.propagate([t])
            jd, fr = datetime_to_jd(t)
            r_ecef = teme_to_ecef(r_teme[0], jd, fr)
            gamma = float(central_angle(r_site, r_ecef)[0])
            resid = np.rad2deg(gamma - lam_max)
            good = abs(resid) <= EDGE_ANGLE_TOL_DEG
            ok1 &= good
            print(f"  {k:>7}{label:>7}{np.rad2deg(gamma):>16.9f}{resid:>18.2e}"
                  f"{'PASS' if good else 'FAIL':>6}")
    print(f"  result                : {'PASS' if ok1 else 'FAIL'}")
    print("")

    print("Check 2 -- pass duration")
    n_rad_s = float(np.sqrt(MU_EARTH_KM3_S2 / R_ORBIT ** 3))
    t_wgs = 2.0 * lam_max / (n_rad_s - EARTH_ROTATION_RAD_S)
    t_gmst = 2.0 * lam_max / (n_rad_s - GMST_RATE_RAD_S)
    measured = [w.duration_s for w in windows
                if not (w.clipped_start or w.clipped_end)]
    print(f"  two-body mean motion n       : {n_rad_s:.12e} rad/s")
    print(f"  omega_E (WGS-84 sidereal)    : {EARTH_ROTATION_RAD_S:.12e} rad/s")
    print(f"  omega_E (IAU 1982 GMST rate) : {GMST_RATE_RAD_S:.12e} rad/s")
    print(f"  predicted duration (WGS-84)  : {t_wgs:.6f} s")
    print(f"  predicted duration (GMST)    : {t_gmst:.6f} s")
    print(f"  measured durations           : "
          f"{', '.join(f'{d:.6f}' for d in measured)} s")
    resid = [d - t_gmst for d in measured]
    print(f"  residual vs GMST prediction  : "
          f"{', '.join(f'{d:+.6f}' for d in resid)} s")
    print(f"  tolerance                    : {DURATION_TOL_S:g} s "
          f"(the equatorial pass is only ~511 s long and the closed form "
          f"ignores the GMST quadratic term and the Earth's finite "
          f"figure off-equator)")
    ok2 = bool(measured) and all(abs(d) <= DURATION_TOL_S for d in resid)
    print(f"  result                       : {'PASS' if ok2 else 'FAIL'}")
    print("")
    return ok1, ok2


def check_isl() -> tuple[bool, bool]:
    """Check 3, both halves."""
    r_block = WGS84_A_KM + DEFAULT_GRAZING_ALTITUDE_KM
    gamma_max = max_isl_central_angle(R_ORBIT)
    print("Check 3a -- ISL clearance flip against the closed-form gamma_max")
    print(f"  blocking radius r_block : {r_block:.3f} km "
          f"(WGS-84 equatorial + {DEFAULT_GRAZING_ALTITUDE_KM:g} km grazing)")
    print(f"  gamma_max (closed form) : {np.rad2deg(gamma_max):.9f} deg")
    lo, hi = 0.0, np.pi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        r1 = np.array([R_ORBIT, 0.0, 0.0])
        r2 = R_ORBIT * np.array([np.cos(mid), np.sin(mid), 0.0])
        if bool(isl_clear(r1, r2)[0]):
            lo = mid
        else:
            hi = mid
    gamma_flip = 0.5 * (lo + hi)
    resid_deg = np.rad2deg(gamma_flip - gamma_max)
    ok3a = abs(resid_deg) <= GAMMA_TOL_DEG
    print(f"  bisected flip angle     : {np.rad2deg(gamma_flip):.9f} deg")
    print(f"  residual                : {resid_deg:.3e} deg")
    print(f"  tolerance               : {GAMMA_TOL_DEG:g} deg "
          f"(80 bisection steps on [0, pi] resolve to ~1e-22 rad; the "
          f"tolerance is set by double-precision arccos conditioning)")
    print(f"  result                  : {'PASS' if ok3a else 'FAIL'}")
    print("")

    print("Check 3b -- segment minimum radius at a real ISL window edge")
    a = CircularOrbit(radius_km=R_ORBIT, inclination_deg=0.0, raan_deg=0.0,
                      arg_lat0_deg=0.0, epoch=EPOCH, name="SAT-A")
    b = CircularOrbit(radius_km=R_ORBIT + 120.0, inclination_deg=0.0,
                      raan_deg=0.0, arg_lat0_deg=70.0, epoch=EPOCH, name="SAT-B")
    # The two radii differ, so the central-angle separation drifts; a 20 h
    # horizon lets it fall below gamma_max and rise back above it, which gives
    # one window with both edges inside the horizon.
    times = time_grid(EPOCH, EPOCH + timedelta(hours=20), 30.0)
    ra, va = a.propagate(times)
    rb, vb = b.propagate(times)
    eph = Ephemeris(times=times, sat_names=["SAT-A", "SAT-B"],
                    r_teme_km=np.stack([ra, rb]),
                    v_teme_km_s=np.stack([va, vb]))
    windows = contact_windows_isl(eph, a, b, max_range_km=1.0e6, refine_tol_s=1e-4)
    print(f"  radii                   : {R_ORBIT:.3f} / {R_ORBIT + 120.0:.3f} km")
    print(f"  windows found           : {len(windows)}")
    print(f"  tolerance               : {LIMB_TOL_KM:g} km "
          f"(a 1e-4 s bisection tolerance with a ~2 km/s closing rate on the "
          f"limb distance bounds the edge error well below this)")
    head = f"  {'window':>7}{'edge':>7}{'d_min [km]':>16}{'residual [km]':>16}{'pass':>6}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    ok3b = True
    any_edge = False
    for k, w in enumerate(windows):
        for label, t, clipped in (("open", w.t_open, w.clipped_start),
                                  ("close", w.t_close, w.clipped_end)):
            if clipped:
                continue
            pa, _ = a.propagate([t])
            pb, _ = b.propagate([t])
            d_min = float(segment_min_radius(pa[0], pb[0])[0])
            resid = d_min - r_block
            good = abs(resid) <= LIMB_TOL_KM
            ok3b &= good
            any_edge = True
            print(f"  {k:>7}{label:>7}{d_min:>16.6f}{resid:>16.2e}"
                  f"{'PASS' if good else 'FAIL':>6}")
    ok3b = ok3b and any_edge
    print(f"  result                  : {'PASS' if ok3b else 'FAIL'}")
    print("")
    return ok3a, ok3b


def main() -> int:
    print("Contact windows vs closed-form central-angle expressions")
    print("=" * 78)
    print("")
    ok1, ok2 = check_ground()
    ok3a, ok3b = check_isl()
    overall = ok1 and ok2 and ok3a and ok3b
    print("=" * 78)
    print(f"check 1 (edge central angle) : {'PASS' if ok1 else 'FAIL'}")
    print(f"check 2 (pass duration)      : {'PASS' if ok2 else 'FAIL'}")
    print(f"check 3a (gamma_max flip)    : {'PASS' if ok3a else 'FAIL'}")
    print(f"check 3b (limb at ISL edge)  : {'PASS' if ok3b else 'FAIL'}")
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
