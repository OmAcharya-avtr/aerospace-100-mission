"""Validation: the Webb distribution against its own moments and its two limits.

Without a lab there is no reference APD histogram to compare against, so this
script checks the three things that can be checked:

1. **Closed-form moments against the density.** (W2) claims mean ``m``,
   variance ``m F`` and third central moment ``3 m F (F - 1)``. Those are
   verified by numerical quadrature of the density itself, over four parameter
   pairs, with the quadrature normalisation reported so the residuals can be
   read against it.
2. **Gaussian limit.** At ``F = 1`` the density must equal ``N(m, m)`` exactly
   (to floating point), and as ``F -> 1`` the sup-norm distance to ``N(m, m F)``
   must fall monotonically.
3. **Poisson limit.** The integer-binned Webb probabilities at ``F = 1`` must
   approach the Poisson pmf with total variation falling as ``m^(-1/2)``, that
   being the rate of the Gaussian approximation to a Poisson. Reported as the
   ratio of successive total variations against ``sqrt(10)``; anything much
   below that ratio would mean the binning is wrong, and anything much above it
   would mean the agreement is better than the approximation allows, which is
   impossible and would indicate an error.

Reference: P. P. Webb, R. J. McIntyre and J. Conradi, "Properties of avalanche
photodiodes", RCA Review 35(2):234-278, 1974. The functional form is taken from
that work as reproduced in the optical-receiver literature; the moment
expressions are derived and verified here, not quoted.

Runtime: about 20 s on one core.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    _SRC = Path(__file__).resolve().parents[1] / "src"
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from photoncount.webb import (  # noqa: E402
    WebbParameters,
    apd_excess_noise_factor,
    binned_pmf,
    gaussian_limit_pdf,
    moments,
    pdf,
    sample,
    support_lower_bound,
)


def check_moments() -> bool:
    print("1. Closed-form moments (W2) against quadrature of the density")
    print(f"   {'m':>8} {'F':>6} {'norm':>14} {'mean resid':>13} {'var resid':>13} "
          f"{'3rd resid':>13}")
    ok = True
    for m, f in ((10.0, 2.0), (100.0, 2.0), (50.0, 3.0), (200.0, 1.5)):
        params = WebbParameters(m, f)
        closed = moments(params)
        sd = np.sqrt(m * f)
        lo = max(support_lower_bound(params), m - 60 * sd)
        grid = np.linspace(lo, m + 250 * sd, 4_000_001)
        dens = pdf(grid, params)
        norm = float(np.trapezoid(dens, grid))
        mean = float(np.trapezoid(grid * dens, grid)) / norm
        var = float(np.trapezoid((grid - mean) ** 2 * dens, grid)) / norm
        third = float(np.trapezoid((grid - mean) ** 3 * dens, grid)) / norm
        r_mean = abs(mean - closed["mean"]) / closed["mean"]
        r_var = abs(var - closed["variance"]) / closed["variance"]
        r_third = abs(third - closed["third_central"]) / closed["third_central"]
        print(f"   {m:8.1f} {f:6.2f} {norm:14.10f} {r_mean:13.3e} {r_var:13.3e} "
              f"{r_third:13.3e}")
        ok &= (abs(norm - 1.0) < 1e-6) and r_mean < 1e-5 and r_var < 1e-4 and r_third < 5e-3
    print(f"   verdict: {'PASS' if ok else 'FAIL'} "
          "(tolerances: norm 1e-6, mean 1e-5, var 1e-4, third 5e-3)")
    return ok


def check_gaussian_limit() -> bool:
    print("\n2. Gaussian limit: F = 1 exactly, then F -> 1")
    m = 200.0
    exact = WebbParameters(m, 1.0)
    grid = np.linspace(m - 6 * np.sqrt(m), m + 6 * np.sqrt(m), 4001)
    exact_dev = float(np.max(np.abs(pdf(grid, exact) - stats.norm.pdf(grid, m, np.sqrt(m)))))
    print(f"   F = 1 sup-norm against scipy N(m, m): {exact_dev:.3e}")
    print(f"   {'F':>6} {'sup-norm to N(m, mF)':>22}")
    devs = []
    for f in (2.0, 1.5, 1.2, 1.05, 1.01, 1.001):
        params = WebbParameters(m, f)
        dev = float(np.max(np.abs(pdf(grid, params) - gaussian_limit_pdf(grid, params))))
        devs.append(dev)
        print(f"   {f:6.3f} {dev:22.6e}")
    monotone = all(b < a for a, b in zip(devs, devs[1:], strict=False))
    ok = exact_dev < 1e-15 and monotone and devs[-1] < 1e-5
    print(f"   monotone decreasing: {monotone}")
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def check_poisson_limit() -> bool:
    print("\n3. Poisson limit: total variation of binned Webb(F=1) to Poisson(m)")
    print(f"   {'m':>8} {'total variation':>17} {'TV * sqrt(m)':>14} {'ratio to previous':>18}")
    means = (100.0, 1000.0, 10000.0)
    tvs = []
    for m in means:
        k_max = int(m + 12 * np.sqrt(m))
        probs = binned_pmf(WebbParameters(m, 1.0), k_max)
        poisson = stats.poisson.pmf(np.arange(k_max + 1), m)
        tv = float(0.5 * np.abs(probs - poisson).sum())
        ratio = tvs[-1] / tv if tvs else float("nan")
        tvs.append(tv)
        print(f"   {m:8.0f} {tv:17.8e} {tv * np.sqrt(m):14.6f} {ratio:18.4f}")
    ratios = [tvs[i] / tvs[i + 1] for i in range(len(tvs) - 1)]
    expected = np.sqrt(10.0)
    ok = all(abs(r - expected) / expected < 0.08 for r in ratios)
    print(f"   expected ratio sqrt(10) = {expected:.4f}; measured {[round(r, 4) for r in ratios]}")
    print(f"   verdict: {'PASS' if ok else 'FAIL'} (each ratio within 8 % of sqrt(10))")
    print("   note: this is the honest limit. Webb at F = 1 is a continuous Gaussian,")
    print("   so it cannot match the Poisson third central moment m (it is 0), and the")
    print("   agreement improves only as m^-1/2.")
    return ok


def check_sampling() -> bool:
    print("\n4. Inverse-CDF sampling against the closed-form moments")
    params = WebbParameters(100.0, 2.0)
    closed = moments(params)
    draws = sample(params, 400_000, np.random.default_rng(20261006))
    se_mean = np.sqrt(closed["variance"] / draws.size)
    z_mean = (draws.mean() - closed["mean"]) / se_mean
    rel_var = abs(draws.var(ddof=1) - closed["variance"]) / closed["variance"]
    skew = float(((draws - draws.mean()) ** 3).mean() / draws.var() ** 1.5)
    print(f"   n = {draws.size}")
    print(f"   mean      {draws.mean():12.5f} vs {closed['mean']:.5f}  z = {z_mean:+.3f}")
    print(f"   variance  {draws.var(ddof=1):12.5f} vs {closed['variance']:.5f}  "
          f"rel = {rel_var:.5f}")
    print(f"   skewness  {skew:12.5f} vs {closed['skewness']:.5f}")
    ok = abs(z_mean) < 4.0 and rel_var < 0.02 and abs(skew - closed["skewness"]) < 0.02
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def check_excess_noise_relation() -> bool:
    print("\n5. Excess noise factor F = k G + (2 - 1/G)(1 - k)")
    print(f"   {'G':>8} {'k':>6} {'F':>12}")
    rows = ((1.0, 0.5), (50.0, 0.0), (50.0, 0.02), (50.0, 1.0), (1000.0, 0.0))
    for g, k in rows:
        print(f"   {g:8.1f} {k:6.3f} {apd_excess_noise_factor(g, k):12.5f}")
    ok = (
        abs(apd_excess_noise_factor(1.0, 0.5) - 1.0) < 1e-12
        and abs(apd_excess_noise_factor(50.0, 1.0) - 50.0) < 1e-12
        and abs(apd_excess_noise_factor(1000.0, 0.0) - 1.999) < 1e-12
    )
    print("   limits: F(G, k=1) = G, F(G -> inf, k=0) -> 2, F(1, k) = 1")
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("validate_webb_limits.py")
    print("Webb, McIntyre & Conradi, RCA Review 35(2):234-278 (1974)")
    print("=" * 78)
    results = [
        check_moments(),
        check_gaussian_limit(),
        check_poisson_limit(),
        check_sampling(),
        check_excess_noise_relation(),
    ]
    print("\n" + "=" * 78)
    print(f"checks passed: {sum(results)} of {len(results)}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
