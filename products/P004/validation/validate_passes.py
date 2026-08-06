"""Validation 1: rise/set times vs an independent fine-grid recomputation.

WHAT THIS CHECKS (and what it does NOT)
---------------------------------------
This is a NUMERICAL-METHOD verification, not an external cross-validation.
The reference is an independent implementation path inside this repository:
a brute-force dense-grid sampling of the same elevation function, used to
(a) count passes and (b) locate the mask crossings by direct grid search plus
a local ultra-fine sweep.  It shares the SGP4 propagator and the frame code
with the code under test, so it validates the coarse-scan + bisection root
finder and the pass-assembly logic, NOT the underlying SGP4/TEME->ECEF model.
No comparison against any external service (STK, GMAT, Heavens-Above,
Celestrak SatVis, ...) was performed; none is claimed.

Independently of that, the analytic circular-orbit case in
tests/test_passes.py checks the geometry against a closed-form solution.

Run: python validation/validate_passes.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from passplanner import Station, find_passes  # noqa: E402
from passplanner.fixtures import ISS_2008, NOAA14_1997  # noqa: E402
from passplanner.frames import datetime_to_jd, ecef_to_azel, teme_to_ecef  # noqa: E402

COARSE_GRID_S = 1.0     # brute-force detection grid
FINE_GRID_S = 0.002     # local sweep resolution around each crossing
LOCAL_HALF_WIDTH_S = 3.0

CASES = [
    (ISS_2008, Station(name="Alpengipfel OGS", lat_deg=47.10, lon_deg=10.90, alt_km=2.00,
                       min_elevation_deg=20.0),
     datetime(2008, 9, 20, 12, 0, tzinfo=timezone.utc), timedelta(days=1)),
    (ISS_2008, Station(name="Cerro Ficticio OGS", lat_deg=-24.50, lon_deg=-70.20, alt_km=2.60,
                       min_elevation_deg=10.0),
     datetime(2008, 9, 20, 12, 0, tzinfo=timezone.utc), timedelta(days=1)),
    (NOAA14_1997, Station(name="Karoo Vlakte OGS", lat_deg=-31.50, lon_deg=21.00, alt_km=1.60,
                          min_elevation_deg=5.0),
     datetime(1997, 11, 17, 0, 0, tzinfo=timezone.utc), timedelta(days=1)),
]


def elevation_fn(tle, station):
    sat = tle.to_satrec()

    def elev(t: datetime) -> float:
        jd, fr = datetime_to_jd(t)
        err, r_teme, _v = sat.sgp4(jd, fr)
        if err != 0:
            raise RuntimeError(f"sgp4 error {err}")
        r_ecef = teme_to_ecef(np.array(r_teme), jd, fr)
        _az, el, _rng = ecef_to_azel(r_ecef, station.lat_deg, station.lon_deg, station.alt_km)
        return el

    return elev


def brute_force_passes(elev, t0: datetime, span: timedelta, mask_deg: float):
    """Dense-grid reference: returns [(rise_s, set_s, max_el)] offsets from t0."""
    total_s = span.total_seconds()
    n = int(total_s / COARSE_GRID_S) + 1
    grid = np.arange(n) * COARSE_GRID_S
    vals = np.array([elev(t0 + timedelta(seconds=float(s))) for s in grid]) - mask_deg
    above = vals >= 0.0
    events = []
    start_idx = None
    for i in range(n):
        if above[i] and start_idx is None:
            start_idx = i
        elif not above[i] and start_idx is not None:
            events.append((start_idx, i - 1))
            start_idx = None
    if start_idx is not None:
        events.append((start_idx, n - 1))

    out = []
    for i0, i1 in events:
        rise = _fine_crossing(elev, t0, grid[max(i0 - 1, 0)], grid[i0], mask_deg)
        set_ = _fine_crossing(elev, t0, grid[i1], grid[min(i1 + 1, n - 1)], mask_deg)
        max_el = float(np.max(vals[i0:i1 + 1]) + mask_deg)
        out.append((rise, set_, max_el))
    return out


def _fine_crossing(elev, t0, lo_s, hi_s, mask_deg):
    """Locate the mask crossing in [lo_s, hi_s] by an ultra-fine linear sweep."""
    lo = max(lo_s - LOCAL_HALF_WIDTH_S, 0.0)
    hi = hi_s + LOCAL_HALF_WIDTH_S
    samples = np.arange(lo, hi + FINE_GRID_S, FINE_GRID_S)
    vals = np.array([elev(t0 + timedelta(seconds=float(s))) for s in samples]) - mask_deg
    sign_changes = np.where(np.diff(np.signbit(vals)))[0]
    if len(sign_changes) == 0:
        return float("nan")
    k = sign_changes[0]
    # Linear interpolation between the two bracketing fine samples.
    frac = vals[k] / (vals[k] - vals[k + 1])
    return float(samples[k] + frac * FINE_GRID_S)


def main() -> int:
    lines = []
    w = lines.append
    w("PassPlanner validation 1 -- rise/set vs independent dense-grid recomputation")
    w(f"run: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    w(f"reference method: brute-force elevation sampling at {COARSE_GRID_S} s for detection, "
      f"local sweep at {FINE_GRID_S} s (+- {LOCAL_HALF_WIDTH_S} s) for the crossings")
    w("code under test: coarse scan (30 s) + bisection to 0.05 s")
    w("NOTE: this is an internal numerical-method check; SGP4 and the frame code are")
    w("      shared between the two paths. No external service was consulted.")
    w("")

    worst = 0.0
    all_ok = True
    for tle, station, t0, span in CASES:
        elev = elevation_fn(tle, station)
        t_a = time.perf_counter()
        found = find_passes(tle, station, t0, t0 + span)
        t_test = time.perf_counter() - t_a
        t_b = time.perf_counter()
        ref = brute_force_passes(elev, t0, span, station.min_elevation_deg)
        t_ref = time.perf_counter() - t_b

        w(f"--- {tle.name} @ {station.name} (mask {station.min_elevation_deg:.0f} deg, "
          f"{t0.date()} +{span.days} d)")
        w(f"    passes: code-under-test {len(found)}, dense-grid reference {len(ref)}  "
          f"[runtime {t_test:.2f} s vs {t_ref:.1f} s]")
        if len(found) != len(ref):
            all_ok = False
            w("    FAIL: pass counts differ")
            continue
        w(f"    {'#':>2} {'rise ref [s]':>14} {'d_rise [s]':>11} {'d_set [s]':>10} "
          f"{'d_dur [s]':>10} {'maxEl test':>11} {'maxEl ref':>10}")
        for i, (p, (rise_s, set_s, max_el)) in enumerate(zip(found, ref)):
            d_rise = (p.t_rise - t0).total_seconds() - rise_s
            d_set = (p.t_set - t0).total_seconds() - set_s
            d_dur = p.duration_s - (set_s - rise_s)
            worst = max(worst, abs(d_rise), abs(d_set))
            w(f"    {i:>2} {rise_s:>14.3f} {d_rise:>11.4f} {d_set:>10.4f} "
              f"{d_dur:>10.4f} {p.max_elevation_deg:>11.4f} {max_el:>10.4f}")
        w("")

    tol = 0.05  # the bisection tolerance requested from find_passes
    w(f"worst |delta| on any rise/set time: {worst:.4f} s")
    w(f"tolerance (bisection refine_tol_s):  {tol:.4f} s")
    verdict = "PASS" if (all_ok and worst <= tol) else "FAIL"
    w(f"VERDICT: {verdict}")
    text = "\n".join(lines)
    print(text)
    (Path(__file__).parent / "validate_passes_output.txt").write_text(text + "\n")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
