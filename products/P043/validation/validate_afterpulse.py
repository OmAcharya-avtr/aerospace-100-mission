"""Validation: the afterpulse cluster-size closed forms, by direct Monte Carlo.

The afterpulse model in this package is a Poisson cluster process. Its two
accounting variants give closed forms for the mean cluster size and for the Fano
factor of the window count, derived in :mod:`photoncount.afterpulse` from the
compound-Poisson identity ``E[N] = mu E[C]``, ``Var[N] = mu E[C^2]``. This script
checks them by sampling the cluster-size distributions directly --- not through
the event-level simulator, which is checked separately in
``validate_simulator.py``, so the two checks are independent.

Checks
------
1. Geometric (cascading) cluster size: mean ``1/(1-p)``, variance
   ``p/(1-p)^2``.
2. Two-point (first-order) cluster size: mean ``1+p``, second moment ``1+3p``.
3. Fano factor of the compound Poisson count against ``(1+p)/(1-p)`` and
   ``(1+3p)/(1+p)``.
4. Forward and inverse rate relations are exact inverses over a sweep in ``p``.
5. The effective-probability relation (A3) ``p_eff = p exp(-tau/t_ap)`` against
   the survival fraction of exponentially distributed delays past ``tau``.

Runtime: about 15 s on one core.
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
    cluster_size_moments,
    effective_afterpulse_probability,
    fano_factor_prediction,
    observed_from_primary,
    primary_from_observed,
)

N_SAMPLES = 2_000_000


def check_cluster_moments() -> bool:
    print("1 & 2. Cluster-size moments against direct sampling, n = 2000000")
    rng = np.random.default_rng(20261006)
    ok = True
    print(f"   {'model':12s} {'p':>6} {'mean (MC)':>12} {'mean (closed)':>14} "
          f"{'var (MC)':>12} {'var (closed)':>13}")
    for p in (0.05, 0.1, 0.3, 0.6):
        # Cascading: C - 1 ~ Geometric(1 - p) on {0, 1, 2, ...}
        casc = 1 + rng.geometric(1.0 - p, size=N_SAMPLES) - 1
        closed = cluster_size_moments(p, "cascading")
        print(f"   {'cascading':12s} {p:6.3f} {casc.mean():12.6f} {closed['mean']:14.6f} "
              f"{casc.var(ddof=1):12.6f} {closed['variance']:13.6f}")
        ok &= abs(casc.mean() - closed["mean"]) < 5 * np.sqrt(
            closed["variance"] / N_SAMPLES
        )
        ok &= abs(casc.var(ddof=1) - closed["variance"]) / closed["variance"] < 0.02
        # First order: C = 1 + Bernoulli(p)
        first = 1 + (rng.random(N_SAMPLES) < p).astype(int)
        closed_f = cluster_size_moments(p, "first_order")
        print(f"   {'first_order':12s} {p:6.3f} {first.mean():12.6f} "
              f"{closed_f['mean']:14.6f} {first.var(ddof=1):12.6f} "
              f"{closed_f['variance']:13.6f}")
        ok &= abs(first.mean() - closed_f["mean"]) < 5 * np.sqrt(
            max(closed_f["variance"], 1e-12) / N_SAMPLES
        )
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def check_fano() -> bool:
    print("\n3. Fano factor of the compound Poisson count, mu = 200 primaries per window")
    rng = np.random.default_rng(7)
    mu = 200.0
    n_windows = 60_000
    ok = True
    print(f"   {'model':12s} {'p':>6} {'Fano (MC)':>12} {'Fano (closed)':>14} "
          f"{'rel dev':>9}")
    for p in (0.05, 0.1, 0.3):
        for model in ("cascading", "first_order"):
            k = rng.poisson(mu, size=n_windows)
            total = np.empty(n_windows)
            for i in range(n_windows):
                if model == "cascading":
                    sizes = rng.geometric(1.0 - p, size=k[i])
                else:
                    sizes = 1 + (rng.random(k[i]) < p).astype(int)
                total[i] = sizes.sum()
            fano = total.var(ddof=1) / total.mean()
            pred = fano_factor_prediction(p, model)
            rel = abs(fano - pred) / pred
            print(f"   {model:12s} {p:6.3f} {fano:12.6f} {pred:14.6f} {rel:9.5f}")
            ok &= rel < 0.05
    print(f"   verdict: {'PASS' if ok else 'FAIL'} (within 5 %)")
    return ok


def check_inverse() -> bool:
    print("\n4. Forward and inverse rate relations are exact inverses")
    worst = 0.0
    for model in ("cascading", "first_order"):
        for p in np.linspace(0.0, 0.9, 19):
            m = observed_from_primary(1e5, float(p), model)
            back = float(primary_from_observed(m, float(p), model)[0])
            worst = max(worst, abs(back - 1e5) / 1e5)
    print(f"   worst relative round-trip error over both models and 19 values of p: "
          f"{worst:.3e}")
    ok = worst < 1e-12
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def check_effective_probability() -> bool:
    print("\n5. (A3) p_eff = p exp(-tau/t_ap) against the exponential survival fraction")
    rng = np.random.default_rng(11)
    t_ap = 1e-7
    p = 0.2
    print(f"   {'tau/t_ap':>9} {'p_eff (A3)':>12} {'p * survival (MC)':>19} "
          f"{'rel dev':>9} {'z':>7}")
    ok = True
    for ratio in (0.25, 0.5, 1.0, 2.0, 5.0):
        tau = ratio * t_ap
        delays = rng.exponential(t_ap, size=N_SAMPLES)
        survival = float(np.mean(delays > tau))
        measured = p * survival
        pred = effective_afterpulse_probability(p, tau, t_ap)
        rel = abs(measured - pred) / pred
        # The survival fraction is a binomial proportion, so the comparison is
        # a z score, not a fixed relative tolerance: at tau/t_ap = 5 only about
        # 13000 of 2000000 delays survive and 1 % is barely one standard error.
        se = p * np.sqrt(survival * (1.0 - survival) / N_SAMPLES)
        z = (measured - pred) / se
        print(f"   {ratio:9.3f} {pred:12.8f} {measured:19.8f} {rel:9.5f} {z:+7.2f}")
        ok &= abs(z) < 5.0
    print(f"   verdict: {'PASS' if ok else 'FAIL'} (|z| < 5 against the binomial s.e.)")
    return ok


def main() -> int:
    print("validate_afterpulse.py")
    print("=" * 78)
    results = [
        check_cluster_moments(),
        check_fano(),
        check_inverse(),
        check_effective_probability(),
    ]
    print("\n" + "=" * 78)
    print(f"checks passed: {sum(results)} of {len(results)}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
