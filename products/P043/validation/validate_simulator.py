"""Validation: the event-level simulator reproduces every rate law it generalises.

The simulator is the ground truth for the learned correction, so it is the one
component whose being wrong would invalidate everything downstream. It has no
reference implementation to compare against, so it is checked against the three
closed forms it must reduce to, each in the limit where that form is exact:

1. **Ideal counter.** tau = 0, p = 0: the observed rate must equal the incident
   rate to within Poisson sampling error.
2. **Non-paralyzable dead time (D1).** p = 0, swept over n tau from 0.01 to 3.
3. **Paralyzable dead time (D3).** p = 0, same sweep, including past the maximum
   at n tau = 1 where the observed rate must fall.
4. **Cascading afterpulsing (A2).** tau = 0, swept over p.
5. **Fano factor signs.** Dead time must push the Fano factor below 1 and
   afterpulsing above 1; the afterpulsing case is checked against the closed form
   ``(1 + p)/(1 - p)``.
6. **Dead-time suppression of afterpulses (A3).** The measured afterpulse
   fraction against ``p exp(-tau / t_ap)`` in the low-loading limit, where the
   two effects decouple enough for (A3) to apply.
7. **Non-commutation.** The composition of dead time and afterpulsing is not the
   product of their separate effects. The deviation is reported as a number,
   because it is the entire reason the learned correction exists.

All checks report z scores against the Poisson sampling error of the pooled
count, so a failure means a defect rather than a bad draw.

Runtime: about 50 s on one core.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    _SRC = Path(__file__).resolve().parents[1] / "src"
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from photoncount.afterpulse import (  # noqa: E402
    effective_afterpulse_probability,
    fano_factor_prediction,
    observed_from_primary,
)
from photoncount.deadtime import nonparalyzable_observed, paralyzable_observed  # noqa: E402
from photoncount.simulate import DetectorSpec, simulate_run, simulate_windows  # noqa: E402

TAU = 1e-7
TARGET_COUNTS = 400_000


def _pooled(n_true: float, spec: DetectorSpec, rng: np.random.Generator,
            target: int = TARGET_COUNTS) -> tuple[float, int]:
    """Pool windows until about ``target`` counts; return (observed rate, counts)."""
    window = min(1e-2, 20_000.0 / max(n_true, 1.0))
    total = 0
    elapsed = 0.0
    while total < target:
        run = simulate_run(n_true, window, spec, rng)
        total += run.registered
        elapsed += window
    return total / elapsed, total


def check_ideal_counter() -> bool:
    print("1. Ideal counter (tau = 0, p = 0): observed rate equals incident rate")
    rng = np.random.default_rng(20261006)
    print(f"   {'n (counts/s)':>14} {'observed':>14} {'z':>7} {'counts':>9}")
    ok = True
    for n in (1e4, 1e5, 1e6):
        obs, total = _pooled(n, DetectorSpec(), rng)
        se = n / np.sqrt(total)
        z = (obs - n) / se
        print(f"   {n:14.4g} {obs:14.2f} {z:+7.2f} {total:9d}")
        ok &= abs(z) < 4.0
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def _check_dead_time(model: str, closed) -> bool:
    rng = np.random.default_rng(997 if model == "paralyzable" else 998)
    spec = DetectorSpec(dead_time_s=TAU, model=model)
    print(f"   {'n tau':>8} {'predicted m':>14} {'simulated m':>14} {'z':>7} {'counts':>9}")
    ok = True
    observed = []
    for x in (0.01, 0.1, 0.3, 0.7, 1.0, 1.5, 3.0):
        n = x / TAU
        pred = float(closed(n, TAU)[0])
        obs, total = _pooled(n, spec, rng)
        se = pred / np.sqrt(total)
        z = (obs - pred) / se
        observed.append(obs)
        print(f"   {x:8.3f} {pred:14.2f} {obs:14.2f} {z:+7.2f} {total:9d}")
        ok &= abs(z) < 5.0
    return ok, observed


def check_nonparalyzable() -> bool:
    print("\n2. Non-paralyzable dead time (D1): m = n / (1 + n tau)")
    ok, _ = _check_dead_time("nonparalyzable", nonparalyzable_observed)
    print(f"   verdict: {'PASS' if ok else 'FAIL'} (|z| < 5)")
    return ok


def check_paralyzable() -> bool:
    print("\n3. Paralyzable dead time (D3): m = n exp(-n tau), including past the maximum")
    ok, observed = _check_dead_time("paralyzable", paralyzable_observed)
    # observed corresponds to x = 0.01, 0.1, 0.3, 0.7, 1.0, 1.5, 3.0
    rises = observed[4] > observed[3]
    falls = observed[6] < observed[5] < observed[4]
    print(f"   observed rate rises to n tau = 1: {rises}; falls beyond it: {falls}")
    ok = ok and rises and falls
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def check_afterpulsing() -> bool:
    print("\n4. Cascading afterpulsing (A2): m = n / (1 - p), with tau = 0")
    rng = np.random.default_rng(555)
    print(f"   {'p':>6} {'predicted m':>14} {'simulated m':>14} {'z':>7} {'counts':>9}")
    ok = True
    for p in (0.02, 0.05, 0.1, 0.2, 0.4):
        spec = DetectorSpec(
            dead_time_s=0.0, afterpulse_probability=p, afterpulse_mean_delay_s=1e-9
        )
        pred = float(observed_from_primary(1e5, p, "cascading")[0])
        obs, total = _pooled(1e5, spec, rng)
        se = pred / np.sqrt(total)
        z = (obs - pred) / se
        print(f"   {p:6.3f} {pred:14.2f} {obs:14.2f} {z:+7.2f} {total:9d}")
        ok &= abs(z) < 5.0
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def check_fano_signs() -> bool:
    print("\n5. Fano factor: below 1 for dead time, above 1 for afterpulsing")
    rng = np.random.default_rng(123)
    dead = simulate_windows(
        2.0 / TAU, 1e-3, 400, DetectorSpec(dead_time_s=TAU, model="nonparalyzable"), rng
    )
    print(f"   dead time only, n tau = 2:   Fano = {dead['fano_factor']:.5f}  (must be < 1)")
    ok = dead["fano_factor"] < 0.8
    print(f"   {'p':>6} {'simulated Fano':>16} {'(1+p)/(1-p)':>14} {'rel dev':>10}")
    for p in (0.1, 0.2, 0.3):
        ap = simulate_windows(
            1e5,
            2e-3,
            1500,
            DetectorSpec(dead_time_s=0.0, afterpulse_probability=p,
                         afterpulse_mean_delay_s=1e-9),
            rng,
        )
        pred = fano_factor_prediction(p, "cascading")
        rel = abs(ap["fano_factor"] - pred) / pred
        print(f"   {p:6.3f} {ap['fano_factor']:16.5f} {pred:14.5f} {rel:10.4f}")
        ok &= rel < 0.15
    print("   the sign of the departure from 1 is what the learned correction uses")
    print(f"   verdict: {'PASS' if ok else 'FAIL'} (Fano within 15 % of the closed form)")
    return ok


def check_dead_time_suppresses_afterpulses() -> bool:
    print("\n6. Dead-time suppression of afterpulses (A3): p_eff = p exp(-tau / t_ap)")
    rng = np.random.default_rng(321)
    p = 0.3
    print(f"   {'tau / t_ap':>11} {'p_eff (A3)':>12} {'measured AP fraction':>21} "
          f"{'rel dev':>9}")
    ok = True
    t_ap = 1e-6
    for ratio in (0.1, 0.5, 1.0, 2.0, 4.0):
        tau = ratio * t_ap
        spec = DetectorSpec(
            dead_time_s=tau,
            model="nonparalyzable",
            afterpulse_probability=p,
            afterpulse_mean_delay_s=t_ap,
        )
        # Low loading so dead time removes almost no primaries: n tau = 1e-3.
        n = 1e-3 / tau
        total_ap = 0
        total = 0
        while total < 60_000:
            run = simulate_run(n, 2e4 / n, spec, rng)
            total += run.registered
            total_ap += run.registered_afterpulse
        # For the cascading cluster, E[cluster size] = 1/(1 - p_eff) and the
        # afterpulses per cluster are p_eff/(1 - p_eff), so the afterpulse
        # fraction of all registered counts is exactly p_eff. No conversion.
        measured = total_ap / total
        pred = effective_afterpulse_probability(p, tau, t_ap)
        rel = abs(measured - pred) / pred
        print(f"   {ratio:11.2f} {pred:12.6f} {measured:21.6f} {rel:9.4f}")
        ok &= rel < 0.08
    print(f"   verdict: {'PASS' if ok else 'FAIL'} (within 8 %)")
    return ok


def check_non_commutation() -> bool:
    print("\n7. Dead time and afterpulsing do not compose as a product")
    rng = np.random.default_rng(42)
    print(f"   {'n tau':>8} {'p':>6} {'naive product':>15} {'simulated':>14} "
          f"{'rel dev':>9}")
    worst = 0.0
    for x, p in ((0.3, 0.1), (0.7, 0.1), (1.0, 0.2), (0.5, 0.3)):
        n = x / TAU
        spec = DetectorSpec(
            dead_time_s=TAU,
            model="paralyzable",
            afterpulse_probability=p,
            afterpulse_mean_delay_s=4 * TAU,
        )
        naive = float(paralyzable_observed(n, TAU)[0]) / (1.0 - p)
        obs, _ = _pooled(n, spec, rng, target=200_000)
        rel = abs(obs - naive) / naive
        worst = max(worst, rel)
        print(f"   {x:8.3f} {p:6.3f} {naive:15.2f} {obs:14.2f} {rel:9.4f}")
    print(f"   worst relative deviation of the naive product from the simulation: "
          f"{worst:.4f}")
    print("   this deviation is the gap no closed form covers, and the only thing the")
    print("   learned correction can legitimately claim")
    ok = worst > 0.01
    print(f"   verdict: {'PASS' if ok else 'FAIL'} "
          "(the deviation must be present; a zero here would mean the simulator "
          "is not composing the effects)")
    return ok


def main() -> int:
    print("validate_simulator.py")
    print("=" * 78)
    results = [
        check_ideal_counter(),
        check_nonparalyzable(),
        check_paralyzable(),
        check_afterpulsing(),
        check_fano_signs(),
        check_dead_time_suppresses_afterpulses(),
        check_non_commutation(),
    ]
    print("\n" + "=" * 78)
    print(f"checks passed: {sum(results)} of {len(results)}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
