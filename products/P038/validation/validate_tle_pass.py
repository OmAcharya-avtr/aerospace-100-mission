"""Supplementary check: the sgp4 + real-station path, and what station motion costs.

This script is not one of the five Level 1 checks. It exists for two reasons.

1. Regression pinning. The closed-form checks exercise a non-rotating-Earth
   model. The numbers below pin the frame reduction (GMST, geodetic site
   position, omega_E x r station velocity) against a fixed TLE and a fixed
   epoch, so that a change in any of them is visible in a diff. They are
   self-consistency regression values produced by this repository, NOT an
   external reference, and they are labelled as such in VALIDATION.md.

2. Quantifying the station-velocity term. A ground station held at zero
   inertial velocity -- the mistake this package's frames module exists to
   prevent -- produces the range-rate error measured here, in m/s and in Hz.

TLE: the ISS two-line element set used as the worked example in the sgp4
package documentation, epoch 2019-12-09. Station: Chilbolton, UK
(51.1450 N, 1.4365 W, 100 m), a real radio site, used purely as a geometry
input.

Frame reduction and its accuracy class: see the module docstring of
dopplerkit/frames.py. GMST-only (IAU 1982; Aoki et al. 1982), UT1 taken as
UTC, no polar motion. Dominant frame-related range-rate error, computed there
rather than measured: about 0.03 m/s, i.e. about 0.22 Hz at 2.2 GHz.

Run:  python validation/validate_tle_pass.py
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sgp4.api import Satrec

from dopplerkit.doppler import one_way_doppler_hz
from dopplerkit.frames import (
    elevation_deg,
    julian_date,
    satellite_state_teme,
    station_state_teme,
)
from dopplerkit.geometry import State, range_rate_mps, slant_range_m

ISS_LINE1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
ISS_LINE2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"
LAT_DEG, LON_DEG, ALT_M = 51.1450, -1.4365, 100.0
CARRIER_HZ = 2.2e9
T0 = datetime(2019, 12, 9, 16, 0, 0, tzinfo=UTC)


def main() -> int:
    satrec = Satrec.twoline2rv(ISS_LINE1, ISS_LINE2)
    print("validate_tle_pass.py")
    print("=" * 78)
    print("TLE                ISS, sgp4 documentation example, epoch 2019-12-09")
    print(f"station            {LAT_DEG:+.4f} deg lat, {LON_DEG:+.4f} deg lon, {ALT_M:.1f} m")
    print(f"carrier            {CARRIER_HZ / 1e9:.6f} GHz, downlink, one-way")
    print("frame              TEME; station rotated by GMST with omega_E x r velocity")
    print()

    print("A -- culmination search, 5 s elevation scan over one hour from 16:00 UTC")
    best = None
    for k in range(0, 3600, 5):
        t = T0 + timedelta(seconds=k)
        sat = satellite_state_teme(satrec, t)
        sta = station_state_teme(LAT_DEG, LON_DEG, ALT_M, t)
        jd, fr = julian_date(t)
        el = elevation_deg(sta, sat, LAT_DEG, LON_DEG, jd, fr)
        if best is None or el > best[1]:
            best = (k, el, slant_range_m(sta, sat), range_rate_mps(sta, sat))
    assert best is not None
    k_best, el_best, rng_best, rate_best = best
    t_best = T0 + timedelta(seconds=k_best)
    print(f"  culmination epoch            {t_best.isoformat()}")
    print(f"  elevation                    {el_best:.6f} deg")
    print(f"  slant range                  {rng_best / 1e3:.6f} km")
    print(f"  range-rate                   {rate_best:+.6f} m/s")
    print(f"  one-way Doppler              {one_way_doppler_hz(rate_best, CARRIER_HZ):+.4f} Hz")
    print("  scan step 5 s, so the culmination epoch is located to +/- 5 s;")
    print("  the range-rate is not zero at culmination because maximum elevation")
    print("  and closest approach are not the same instant for an off-zenith pass.")
    print()

    print("B -- profile across the pass, 60 s steps")
    print()
    print(f"  {'dt [s]':>8} {'el [deg]':>10} {'range [km]':>12} {'rho_dot [m/s]':>14} "
          f"{'Doppler [Hz]':>14} {'precomp [Hz]':>14}")
    for dt_s in range(-360, 361, 60):
        t = t_best + timedelta(seconds=dt_s)
        sat = satellite_state_teme(satrec, t)
        sta = station_state_teme(LAT_DEG, LON_DEG, ALT_M, t)
        jd, fr = julian_date(t)
        el = elevation_deg(sta, sat, LAT_DEG, LON_DEG, jd, fr)
        rate = range_rate_mps(sta, sat)
        shift = one_way_doppler_hz(rate, CARRIER_HZ)
        print(f"  {dt_s:8d} {el:10.4f} {slant_range_m(sta, sat) / 1e3:12.4f} {rate:14.4f} "
              f"{shift:14.2f} {-shift:14.2f}")
    print()
    print("  Sign check across the table: Doppler is positive (up-shift) while the")
    print("  range-rate is negative (approaching) and negative afterwards, and the")
    print("  pre-compensation column is the negative of the Doppler column.")
    print()

    print("C -- what dropping the station velocity would cost")
    print("comparing the correct station state against one with velocity forced to zero")
    print()
    print(f"  {'dt [s]':>8} {'|v_sta| [m/s]':>14} {'rho_dot ok [m/s]':>18} "
          f"{'rho_dot zero-v':>16} {'error [m/s]':>13} {'error [Hz]':>12}")
    worst_err_mps = 0.0
    worst_err_hz = 0.0
    for dt_s in range(-300, 301, 100):
        t = t_best + timedelta(seconds=dt_s)
        sat = satellite_state_teme(satrec, t)
        sta = station_state_teme(LAT_DEG, LON_DEG, ALT_M, t)
        sta_still = State(
            position_m=sta.position_m, velocity_mps=np.zeros(3), label="station, v forced to 0"
        )
        good = range_rate_mps(sta, sat)
        bad = range_rate_mps(sta_still, sat)
        err = bad - good
        err_hz = one_way_doppler_hz(bad, CARRIER_HZ) - one_way_doppler_hz(good, CARRIER_HZ)
        worst_err_mps = max(worst_err_mps, abs(err))
        worst_err_hz = max(worst_err_hz, abs(err_hz))
        print(f"  {dt_s:8d} {sta.speed_mps:14.4f} {good:18.4f} {bad:16.4f} "
              f"{err:13.4f} {err_hz:12.2f}")
    print()
    print(f"  worst range-rate error from a zero-velocity station  {worst_err_mps:.4f} m/s")
    print(f"  worst Doppler error at {CARRIER_HZ / 1e9:.1f} GHz                   "
          f"{worst_err_hz:.2f} Hz")
    print("  For comparison, every O(beta^2) relativistic term in this package is")
    print("  under 1.3 Hz at this carrier (validate_two_way_relativistic.py).")
    print("  Station velocity is therefore the larger error by roughly three orders")
    print("  of magnitude, which is why frames.station_state_teme never omits it.")
    print()

    print("D -- regression values pinned by tests/test_frames.py")
    t_pin = datetime(2019, 12, 9, 16, 39, 5, tzinfo=UTC)
    sat = satellite_state_teme(satrec, t_pin)
    sta = station_state_teme(LAT_DEG, LON_DEG, ALT_M, t_pin)
    jd, fr = julian_date(t_pin)
    print(f"  epoch                        {t_pin.isoformat()}")
    print(f"  elevation                    {elevation_deg(sta, sat, LAT_DEG, LON_DEG, jd, fr):.6f} deg")
    print(f"  slant range                  {slant_range_m(sta, sat) / 1e3:.6f} km")
    print(f"  range-rate                   {range_rate_mps(sta, sat):+.6f} m/s")
    print(f"  station inertial speed       {sta.speed_mps:.6f} m/s")
    print("  tolerances in the test: 0.01 deg, 0.01 km, 0.01 m/s")
    print()
    print("ALL CHECKS IN THIS SCRIPT: reported (regression values, no external reference)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
