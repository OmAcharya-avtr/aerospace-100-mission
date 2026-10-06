"""Validation: dead-time relations, their inverses, and the non-monotonicity trap.

Checks
------
1. **Known answers at hand-computable rates.** tau = 1 us, n = 1e5 /s gives
   n tau = 0.1, so the non-paralyzable observed rate is 1e5/1.1 = 90909.0909...
   and the paralyzable one is 1e5 exp(-0.1) = 90483.7418... Both to full double
   precision.
2. **The paralyzable maximum.** dm/dn = 0 at n = 1/tau with m = 1/(e tau).
   Verified against a dense sweep of the forward map, and the derivative is
   checked by finite difference.
3. **Non-monotonicity and the two roots.** Every observed rate below the maximum
   has two true rates. Both are recovered by (D5) on the two Lambert branches
   and pushed back through (D3).
4. **The inverse is ill-conditioned near the maximum.** dn/dm is reported across
   the range: it diverges at the maximum, which is the quantitative reason a
   correction near saturation cannot be trusted.
5. **Loss and live-time fractions** at the reference loading n tau = 1.
6. **Model ordering.** exp(-x) <= 1/(1+x) for all x >= 0, so a paralyzable
   detector always loses at least as much as a non-paralyzable one with the same
   dead time. Checked over a sweep.

Reference: G. F. Knoll, *Radiation Detection and Measurement*, 4th ed., Wiley
2010, chapter 4 (dead time: the paralyzable and non-paralyzable models). The
relations are standard; the numbers below are computed here.

Runtime: under 2 s on one core.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    _SRC = Path(__file__).resolve().parents[1] / "src"
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from photoncount.deadtime import (  # noqa: E402
    dead_time_loss_fraction,
    live_time_fraction,
    nonparalyzable_observed,
    nonparalyzable_true,
    paralyzable_maximum,
    paralyzable_observed,
    paralyzable_true,
)

TAU = 1e-6


def check_known_answers() -> bool:
    print("1. Known answers at tau = 1e-6 s, n = 1e5 counts/s (n tau = 0.1)")
    np_obs = float(nonparalyzable_observed(1e5, TAU)[0])
    par_obs = float(paralyzable_observed(1e5, TAU)[0])
    np_hand = 1e5 / 1.1
    par_hand = 1e5 * np.exp(-0.1)
    print(f"   non-paralyzable  m = {np_obs:.10f}  hand = {np_hand:.10f}  "
          f"rel = {abs(np_obs - np_hand) / np_hand:.2e}")
    print(f"   paralyzable      m = {par_obs:.10f}  hand = {par_hand:.10f}  "
          f"rel = {abs(par_obs - par_hand) / par_hand:.2e}")
    ok = abs(np_obs - np_hand) / np_hand < 1e-14 and abs(par_obs - par_hand) / par_hand < 1e-14
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def check_maximum() -> bool:
    print("\n2. Paralyzable maximum: dm/dn = 0 at n = 1/tau, m = 1/(e tau)")
    n_max, m_max = paralyzable_maximum(TAU)
    sweep = np.linspace(1e4, 5e6, 2_000_001)
    obs = paralyzable_observed(sweep, TAU)
    arg = int(np.argmax(obs))
    h = 1.0
    deriv = float(
        (paralyzable_observed(n_max + h, TAU)[0] - paralyzable_observed(n_max - h, TAU)[0])
        / (2 * h)
    )
    print(f"   closed form     n_max = {n_max:.6e}  m_max = {m_max:.10f}")
    print(f"   dense sweep     n_max = {sweep[arg]:.6e}  m_max = {float(obs[arg]):.10f}")
    print(f"   central-difference dm/dn at n_max = {deriv:.3e} (expected 0)")
    within = float(obs.max()) <= m_max * (1 + 1e-12)
    print(f"   sweep maximum does not exceed closed form: {within}")
    ok = (
        abs(sweep[arg] - n_max) / n_max < 2e-3
        and abs(deriv) < 1e-6
        and float(obs.max()) <= m_max * (1 + 1e-12)
    )
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def check_two_roots() -> bool:
    print("\n3. The paralyzable inverse is two-valued below the maximum")
    _, m_max = paralyzable_maximum(TAU)
    print(f"   {'m (counts/s)':>14} {'m / m_max':>10} {'n lower':>14} {'n upper':>14} "
          f"{'round-trip rel':>15}")
    ok = True
    for frac in (0.1, 0.5, 0.9, 0.99, 0.9999):
        m = frac * m_max
        lo = float(paralyzable_true(m, TAU, "lower")[0])
        hi = float(paralyzable_true(m, TAU, "upper")[0])
        back_lo = float(paralyzable_observed(lo, TAU)[0])
        back_hi = float(paralyzable_observed(hi, TAU)[0])
        worst = max(abs(back_lo - m), abs(back_hi - m)) / m
        print(f"   {m:14.4f} {frac:10.4f} {lo:14.4f} {hi:14.4f} {worst:15.3e}")
        ok &= lo <= 1.0 / TAU <= hi and worst < 1e-9
    print("   both roots reproduce the same observed rate; only knowledge of the")
    print("   illumination distinguishes them")
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def check_conditioning() -> bool:
    print("\n4. Conditioning of the inverse: dn/dm on the lower branch")
    _, m_max = paralyzable_maximum(TAU)
    print(f"   {'m / m_max':>10} {'n tau':>10} {'dn/dm':>14} "
          f"{'1 % in m becomes':>18}")
    rows = []
    for frac in (0.1, 0.5, 0.8, 0.95, 0.99, 0.999):
        m = frac * m_max
        dm = m * 1e-6
        n0 = float(paralyzable_true(m, TAU, "lower")[0])
        n1 = float(paralyzable_true(m + dm, TAU, "lower")[0])
        slope = (n1 - n0) / dm
        amplification = slope * m / n0
        rows.append(amplification)
        print(f"   {frac:10.4f} {n0 * TAU:10.4f} {slope:14.6f} "
              f"{amplification:17.2f} %")
    monotone = all(b > a for a, b in zip(rows, rows[1:], strict=False))
    print("   the amplification of a relative error in m diverges at the maximum;")
    print("   this is why preflight refuses to run at or above 1/tau")
    print(f"   monotone increasing: {monotone}")
    print(f"   verdict: {'PASS' if monotone else 'FAIL'}")
    return monotone


def check_loss_and_live_time() -> bool:
    print("\n5. Loss and live-time fractions at n tau = 1")
    n = 1.0 / TAU
    loss_np = float(dead_time_loss_fraction(n, TAU, "nonparalyzable")[0])
    loss_par = float(dead_time_loss_fraction(n, TAU, "paralyzable")[0])
    live_np = float(live_time_fraction(n, TAU, "nonparalyzable")[0])
    live_par = float(live_time_fraction(n, TAU, "paralyzable")[0])
    print(f"   non-paralyzable  loss = {loss_np:.10f} (expected 0.5)")
    print(f"   paralyzable      loss = {loss_par:.10f} (expected 1 - 1/e = "
          f"{1 - np.exp(-1):.10f})")
    print(f"   non-paralyzable  live = {live_np:.10f} (expected 1 - m tau = 0.5)")
    print(f"   paralyzable      live = {live_par:.10f} (expected exp(-1) = "
          f"{np.exp(-1):.10f})")
    ok = (
        abs(loss_np - 0.5) < 1e-14
        and abs(loss_par - (1 - np.exp(-1))) < 1e-14
        and abs(live_np - 0.5) < 1e-14
        and abs(live_par - np.exp(-1)) < 1e-14
    )
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def check_model_ordering() -> bool:
    print("\n6. A paralyzable detector always loses at least as much")
    x = np.logspace(-4, 1.5, 20001)
    n = x / TAU
    par = paralyzable_observed(n, TAU)
    nonpar = nonparalyzable_observed(n, TAU)
    worst = float(np.max(par - nonpar))
    print(f"   max(m_paralyzable - m_nonparalyzable) over n tau in [1e-4, 31.6]: "
          f"{worst:.3e} counts/s")
    rel = float(np.max((par - nonpar) / nonpar))
    print(f"   worst relative excess: {rel:.3e}")
    ok = worst <= 1e-6
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def check_nonparalyzable_round_trip() -> bool:
    print("\n7. Non-paralyzable round trip over five decades of n tau")
    x = np.logspace(-5, 1, 60001)
    n = x / TAU
    back = nonparalyzable_true(nonparalyzable_observed(n, TAU), TAU)
    worst = float(np.max(np.abs(back - n) / n))
    print(f"   worst relative round-trip error: {worst:.3e}")
    ok = worst < 1e-12
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("validate_deadtime.py")
    print("Knoll, Radiation Detection and Measurement, 4th ed., Wiley 2010, ch. 4")
    print("=" * 78)
    results = [
        check_known_answers(),
        check_maximum(),
        check_two_roots(),
        check_conditioning(),
        check_loss_and_live_time(),
        check_model_ordering(),
        check_nonparalyzable_round_trip(),
    ]
    print("\n" + "=" * 78)
    print(f"checks passed: {sum(results)} of {len(results)}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
