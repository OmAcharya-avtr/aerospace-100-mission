"""L1 check 4: two-way is exactly twice one-way; the relativistic terms, with magnitudes.

Check 4a (exact doubling). In the non-relativistic limit, composing the two
one-way legs of a coherent transponder at unit turnaround ratio and keeping
only first order in v/c gives Delta_f_2way = 2 Delta_f_1way. This is asserted
bit-exactly, not to a tolerance, and over a Hypothesis sweep in
tests/test_properties.py. Reference for the two-way formulation: Moyer,
"Formulation for Observed and Computed Values of Deep Space Network Data
Types for Navigation", JPL Deep Space Communications and Navigation Series
Monograph 2, 2000.

Check 4b (what is dropped at O(beta^2)). Three separate terms are reported
with their magnitudes in Hz for a representative 500 km LEO pass, so a reader
can decide whether any of them matters at their carrier:

  1. The special-relativistic term (rho_dot/c)^2 - v^2/(2c^2), from expanding
     the exact relativistic one-way ratio sqrt(1-v^2/c^2)/(1+rho_dot/c)
     (standard special-relativity result).
  2. The two-way classical cascade term +f rho_dot_up rho_dot_down / c^2,
     which the first-order two-way expression drops.
  3. An order-of-magnitude static gravitational term (mu/c^2)(1/r_rx - 1/r_tx),
     point-mass Newtonian potential only.

Order kept by the package: O(beta^1). None of the three terms above is
applied. Dropped entirely and not estimated anywhere: Shapiro delay,
tropospheric and ionospheric delay and their time derivatives, transponder
group delay, oscillator instability, higher relativistic orders.

Check 4c (light-time separation of the two legs). With finite light time the
up-leg and down-leg range-rates are evaluated at different epochs, so the
two-way shift is not exactly twice the instantaneous one-way shift. The size
of that departure is measured here.

Run:  python validation/validate_two_way_relativistic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dopplerkit.analytic import CircularOverheadPass
from dopplerkit.constants import C_M_S, WGS84_A_M
from dopplerkit.doppler import (
    gravitational_shift_fraction,
    one_way_doppler_hz,
    relativistic_correction_hz,
    relativistic_fraction_second_order,
    two_way_doppler_hz,
    two_way_doppler_two_leg_hz,
)
from dopplerkit.lighttime import two_way_light_time

ALT_M = 500.0e3
CARRIERS_HZ = (401.0e6, 2.2e9, 8.4e9, 26.0e9)


def main() -> int:
    orbit = CircularOverheadPass(orbit_radius_m=WGS84_A_M + ALT_M)
    horizon = orbit.horizon_time_s()
    speed = orbit.orbital_speed_mps
    rate_peak = orbit.range_rate_mps(horizon)

    print("validate_two_way_relativistic.py")
    print("=" * 78)
    print(f"model           circular overhead pass, non-rotating Earth, {ALT_M / 1e3:.0f} km")
    print(f"orbital speed   {speed:.6f} m/s")
    print(f"beta_orbit      {speed / C_M_S:.6e}  (= v/c)")
    print(f"peak rho_dot    {rate_peak:.6f} m/s at the geometric horizon")
    print(f"beta_rho_dot    {rate_peak / C_M_S:.6e}")
    print()

    print("CHECK 4a -- two-way is EXACTLY twice one-way (G = 1), bit-exact equality")
    print()
    print(f"  {'rho_dot [m/s]':>14} {'carrier [GHz]':>14} {'1-way [Hz]':>18} "
          f"{'2-way [Hz]':>18} {'2*1way == 2way':>16}")
    exact_ok = True
    for rate in (-7059.216450, -3000.0, 0.0, 1234.5, 7059.216450):
        for carrier in CARRIERS_HZ:
            one = one_way_doppler_hz(rate, carrier)
            two = two_way_doppler_hz(rate, carrier, 1.0)
            ok = two == 2.0 * one
            exact_ok = exact_ok and ok
            print(f"  {rate:14.6f} {carrier / 1e9:14.4f} {one:18.6f} {two:18.6f} "
                  f"{'exact' if ok else 'DIFFERS':>16}")
    print()
    print(f"  tolerance: ZERO (bit-exact) -> {'PASS' if exact_ok else 'FAIL'}")
    print()

    print("CHECK 4b -- O(beta^2) terms, reported with their magnitudes in Hz")
    print()
    print("  at the geometric horizon (peak |rho_dot|):")
    print(f"  {'carrier [GHz]':>14} {'classical [Hz]':>18} {'SR O(b^2) [Hz]':>16} "
          f"{'cascade [Hz]':>14} {'grav [Hz]':>12} {'|SR/cls|':>10}")
    for carrier in CARRIERS_HZ:
        classical = one_way_doppler_hz(rate_peak, carrier)
        sr = relativistic_correction_hz(rate_peak, speed, carrier)
        _, cascade = two_way_doppler_two_leg_hz(rate_peak, rate_peak, carrier, 1.0)
        grav = gravitational_shift_fraction(orbit.orbit_radius_m, orbit.observer_radius_m) * carrier
        print(f"  {carrier / 1e9:14.4f} {classical:18.6f} {sr:16.6f} {cascade:14.6f} "
              f"{grav:12.6f} {abs(sr / classical):10.2e}")
    print()
    print("  at closest approach (rho_dot = 0, so the classical shift vanishes):")
    print(f"  {'carrier [GHz]':>14} {'classical [Hz]':>18} {'SR O(b^2) [Hz]':>16} "
          f"{'grav [Hz]':>12}")
    for carrier in CARRIERS_HZ:
        classical = one_way_doppler_hz(0.0, carrier)
        sr = relativistic_correction_hz(0.0, speed, carrier)
        grav = gravitational_shift_fraction(orbit.orbit_radius_m, orbit.observer_radius_m) * carrier
        print(f"  {carrier / 1e9:14.4f} {classical:18.6f} {sr:16.6f} {grav:12.6f}")
    print()
    frac_tca = relativistic_fraction_second_order(0.0, speed)
    frac_grav = gravitational_shift_fraction(orbit.orbit_radius_m, orbit.observer_radius_m)
    print(f"  SR fractional term at closest approach   {frac_tca:+.6e}  (= -v^2/(2c^2))")
    print(f"  gravitational fractional term            {frac_grav:+.6e}")
    print(f"  ratio |SR| / |grav|                      {abs(frac_tca / frac_grav):.3f}")
    print()
    print("  Reading: at S-band (2.2 GHz) the whole O(beta^2) budget is under 1.3 Hz")
    print("  against a 51.8 kHz classical excursion -- a relative 2.4e-05. At Ka-band")
    print("  (26 GHz) it reaches about 15 Hz against 612 kHz. It matters only for")
    print("  sub-Hz carrier knowledge, and at that level the neglected tropospheric")
    print("  delay rate is larger still, so this package does not apply any of it.")
    print()
    print("  The SR term does NOT vanish at closest approach: there it is pure")
    print("  transmitter time dilation -v^2/(2c^2), independent of range-rate.")
    print()

    print("CHECK 4c -- light-time separation of the two legs")
    print("two-way Doppler from light-time-corrected per-leg range-rates vs the")
    print("instantaneous exactly-2x form, at 2.2 GHz uplink, G = 1")
    print()
    carrier = 2.2e9

    def sat_pos(t: float) -> np.ndarray:
        return orbit.satellite_state(t).position_m

    def sta_pos(t: float) -> np.ndarray:
        return orbit.observer_state(t).position_m

    peak_two_way = abs(two_way_doppler_hz(rate_peak, carrier, 1.0))
    print("  normalising denominator for the 'rel' column: the peak two-way")
    print(f"  excursion over this pass, {peak_two_way:.4f} Hz")
    print()
    print("  Note: this model holds the station inertially fixed (non-rotating")
    print("  Earth), so the up-leg and down-leg light times of one round trip are")
    print("  equal to double precision; the tau_up - tau_down column is therefore")
    print("  exactly zero here by construction, and would not be for a real station.")
    print()
    print(f"  {'t_tx [s]':>10} {'tau_up [ms]':>14} {'tau_up-tau_dn [ns]':>20} "
          f"{'2-way inst [Hz]':>17} {'2-way 2-leg [Hz]':>18} {'diff [Hz]':>11} "
          f"{'diff/peak':>10}")
    worst_abs = 0.0
    worst_rel = 0.0
    for t_tx in np.linspace(-horizon, horizon, 9):
        tw = two_way_light_time(sat_pos, sta_pos, float(t_tx))
        # Range-rate on each leg, evaluated at that leg's own mid-epoch.
        t_up_mid = 0.5 * (tw.up.emission_time_s + tw.up.reception_time_s)
        t_down_mid = 0.5 * (tw.down.emission_time_s + tw.down.reception_time_s)
        rate_up = orbit.range_rate_mps(t_up_mid)
        rate_down = orbit.range_rate_mps(t_down_mid)
        inst = two_way_doppler_hz(orbit.range_rate_mps(float(t_tx)), carrier, 1.0)
        two_leg, _ = two_way_doppler_two_leg_hz(rate_up, rate_down, carrier, 1.0)
        diff = two_leg - inst
        tau_diff_ns = (tw.up.light_time_s - tw.down.light_time_s) * 1e9
        worst_abs = max(worst_abs, abs(diff))
        worst_rel = max(worst_rel, abs(diff) / peak_two_way)
        print(f"  {t_tx:10.3f} {tw.up.light_time_s * 1e3:14.9f} {tau_diff_ns:20.3f} "
              f"{inst:17.4f} {two_leg:18.4f} {diff:11.4f} "
              f"{abs(diff) / peak_two_way:10.2e}")
    print()
    print(f"  worst absolute departure from exactly-2x: {worst_abs:.4f} Hz")
    print(f"  worst departure as a fraction of the peak two-way excursion: {worst_rel:.3e}")
    print("  This is NOT a failure of check 4a: 4a is the identity at a single epoch,")
    print("  and this is the physical fact that the two legs are not at one epoch.")
    print("  Both are reported because a reader needs to know which one their")
    print("  application is using.")
    print()
    rate_tca_two_way = 2.0 * orbit.max_doppler_rate_hz_per_s(carrier)
    tau_tca = two_way_light_time(sat_pos, sta_pos, 0.0).up.light_time_s
    print("  Quantitative explanation of the worst case, at closest approach: the")
    print("  two legs' mid-epochs sit at t_tx + tau/2 and t_tx + 3tau/2, a mean")
    print("  offset of one light time tau from t_tx, so the departure is the")
    print("  two-way Doppler rate times tau:")
    print(f"    two-way Doppler rate at TCA  {rate_tca_two_way:+.4f} Hz/s")
    print(f"    light time at TCA            {tau_tca * 1e3:.6f} ms")
    print(f"    product                      {rate_tca_two_way * tau_tca:+.4f} Hz")
    print(f"    measured departure           {-worst_abs:+.4f} Hz")
    print(f"    |product - measured|         {abs(rate_tca_two_way * tau_tca + worst_abs):.4f} Hz")
    print()

    overall = exact_ok
    print("ALL CHECKS IN THIS SCRIPT:", "PASS" if overall else "FAIL")
    print("(checks 4b and 4c report magnitudes; they have no pass/fail criterion)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
