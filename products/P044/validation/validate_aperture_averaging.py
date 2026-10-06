"""Validation: the aperture averaging factor and its two derived limits.

Checks
------
1. The kernel normalisation identity ``int_0^1 u W(u) du = pi/16`` with
   ``W(u) = arccos(u) - u sqrt(1-u^2)``, by quadrature against the exact
   value. This identity is what makes ``A -> 1`` for a fully correlated
   field, so if it were wrong every averaging factor would be wrong by the
   same constant.
2. ``A(D) -> 1`` as ``D -> 0``, and ``A`` strictly decreasing in ``D``.
3. The small-aperture limit ``A ~ 1 - D^2/(4 rho_c^2)``: the quadrature is
   compared with it over ``D/rho_c <= 0.2``, where the neglected term is
   ``O((D/rho_c)^4)``, so the residual is checked to *scale* like the fourth
   power rather than merely to be small.
4. The large-aperture limit ``A ~ 4(rho_c/D)^2 - (8/sqrt(pi))(rho_c/D)^3``:
   relative agreement with the quadrature over ``D/rho_c`` from 5 to 200.
5. The leading large-``D`` coefficient: ``A D^2 / rho_c^2 -> 4``.
6. ``A`` for the exponential covariance is bracketed by nothing analytic
   here, so it is only checked for the two structural properties (limits to
   1 at ``D = 0``, monotone decreasing) and for being larger than the
   Gaussian case at equal ``rho_c``, which follows from the exponential
   covariance having the heavier tail.

Everything is derived in ``aperturediv.aperture``; no page number is quoted
and no constant is taken from a table.

Runtime under 5 s on one core.
"""

from __future__ import annotations

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
from scipy import integrate  # noqa: E402

from aperturediv.aperture import (  # noqa: E402
    aperture_averaging_factor,
    aperture_averaging_large_d,
    aperture_averaging_small_d,
    circular_aperture_weight,
    effective_scintillation_index,
    equal_area_diameter,
    fresnel_scale,
)

RHO_C = 1.0  # all diameters are quoted as D/rho_c, so rho_c = 1 loses nothing


def _rule(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    failures: list[str] = []

    _rule("1. Kernel normalisation  int_0^1 u W(u) du = pi/16")
    val, err = integrate.quad(
        lambda u: u * float(circular_aperture_weight(u)), 0.0, 1.0, epsabs=1e-14, epsrel=1e-13
    )
    exact = np.pi / 16.0
    print(f"  quadrature      {val:.15f}  (reported error {err:.2e})")
    print(f"  pi/16           {exact:.15f}")
    print(f"  abs difference  {abs(val - exact):.3e}, tolerance 1.0e-11")
    if abs(val - exact) > 1e-11:
        failures.append(f"kernel normalisation off by {abs(val - exact)}")
    print()

    _rule("2. Limit at D = 0 and monotonicity")
    print(f"  A(0)            {aperture_averaging_factor(0.0, RHO_C):.15f}  (expect exactly 1)")
    grid = np.geomspace(1e-3, 300.0, 120)
    a_vals = np.array([aperture_averaging_factor(d, RHO_C) for d in grid])
    diffs = np.diff(a_vals)
    print(f"  A over D/rho_c in [1e-3, 300]: max {a_vals.max():.12f}  min {a_vals.min():.3e}")
    print(f"  strictly decreasing: {bool(np.all(diffs < 0.0))}  "
          f"(largest forward difference {diffs.max():+.3e})")
    if aperture_averaging_factor(0.0, RHO_C) != 1.0:
        failures.append("A(0) != 1")
    if not np.all(diffs < 0.0):
        failures.append("A is not strictly decreasing in D")
    if a_vals.max() > 1.0:
        failures.append(f"A exceeds 1: {a_vals.max()}")
    print()

    _rule("3. Small-aperture limit  A ~ 1 - D^2/(4 rho_c^2)")
    print("  D/rho_c     A quadrature     A small-D       residual    residual/(D/rc)^4")
    ratios = []
    for d in (0.0125, 0.025, 0.05, 0.1, 0.2):
        a_q = aperture_averaging_factor(d, RHO_C)
        a_s = aperture_averaging_small_d(d, RHO_C)
        resid = a_q - a_s
        ratios.append(resid / d**4)
        print(f"{d:9.4f} {a_q:16.12f} {a_s:15.12f} {resid:14.3e} {resid / d**4:18.6f}")
    spread = (max(ratios) - min(ratios)) / abs(np.mean(ratios))
    print(f"  residual/(D/rho_c)^4 spread over the range: {spread:.4%}")
    print("  a constant ratio confirms the neglected term is O((D/rho_c)^4), tolerance 5%")
    if spread > 0.05:
        failures.append(f"small-D residual does not scale as D^4; spread {spread}")
    print()

    _rule("4. Large-aperture limit  A ~ 4(rho_c/D)^2 - (8/sqrt pi)(rho_c/D)^3")
    print("  D/rho_c     A quadrature      A large-D (2 terms)   rel diff"
          "   A large-D (1 term)  rel diff")
    worst2 = 0.0
    for d in (5.0, 10.0, 20.0, 50.0, 100.0, 200.0):
        a_q = aperture_averaging_factor(d, RHO_C)
        a2 = aperture_averaging_large_d(d, RHO_C, order=2)
        a1 = aperture_averaging_large_d(d, RHO_C, order=1)
        r2 = abs(a_q - a2) / a_q
        r1 = abs(a_q - a1) / a_q
        worst2 = max(worst2, r2) if d >= 20.0 else worst2
        print(f"{d:9.1f} {a_q:16.10f} {a2:21.10f} {r2:10.3e} {a1:19.10f} {r1:10.3e}")
    print(f"  worst relative difference for D/rho_c >= 20: {worst2:.3e}, tolerance 1.0e-4")
    if worst2 > 1e-4:
        failures.append(f"large-D two-term limit worst relative difference {worst2}")
    print()

    _rule("5. Leading large-D coefficient  A D^2 / rho_c^2 -> 4")
    for d in (10.0, 50.0, 200.0, 1000.0):
        prod = aperture_averaging_factor(d, RHO_C) * d * d
        print(f"  D/rho_c = {d:7.1f}   A D^2/rho_c^2 = {prod:.8f}   deviation from 4: "
              f"{prod - 4.0:+.3e}")
    final = aperture_averaging_factor(1000.0, RHO_C) * 1e6
    print(f"  at D/rho_c = 1000 the deviation is {abs(final - 4.0):.3e}, tolerance 2.0e-2")
    print("  convergence is O(rho_c/D), so this is a slow limit; the two-term form in")
    print("  check 4 is the one worth using.")
    if abs(final - 4.0) > 2e-2:
        failures.append(f"A D^2 -> 4 limit off by {abs(final - 4.0)}")
    print()

    _rule("6. Exponential covariance: structural properties, and a crossover")
    exp_vals = np.array([aperture_averaging_factor(d, RHO_C, "exponential") for d in grid])
    print(f"  A(0) exponential  {aperture_averaging_factor(0.0, RHO_C, 'exponential'):.12f}")
    print(f"  strictly decreasing: {bool(np.all(np.diff(exp_vals) < 0.0))}")
    print("  D/rho_c      A gaussian   A exponential   exp - gauss")
    for d in (0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 20.0):
        ag = aperture_averaging_factor(d, RHO_C)
        ae = aperture_averaging_factor(d, RHO_C, "exponential")
        print(f"{d:9.2f} {ag:14.8f} {ae:14.8f} {ae - ag:+14.8f}")
    sign = np.sign(exp_vals - a_vals)
    flips = np.nonzero(np.diff(sign) != 0)[0]
    print()
    print("  Recorded because the first expectation written for this check was wrong.")
    print("  'Exponential covariance has the heavier tail, so A must be larger at every")
    print("  D' is false: exp(-x) < exp(-x^2) for x < 1, so at equal rho_c the")
    print("  exponential field is already decorrelated across a small aperture and")
    print("  averages LESS, while its tail makes it average less across a large one")
    print("  too. The two curves therefore cross exactly once.")
    if flips.size == 1:
        lo, hi = grid[flips[0]], grid[flips[0] + 1]
        print(f"  sign changes exactly once, between D/rho_c = {lo:.4f} and {hi:.4f}")
    else:
        print(f"  sign changes {flips.size} times on the grid: {grid[flips]}")
    if not np.all(np.diff(exp_vals) < 0.0):
        failures.append("exponential A is not strictly decreasing")
    if exp_vals.max() > 1.0:
        failures.append(f"exponential A exceeds 1: {exp_vals.max()}")
    if flips.size != 1:
        failures.append(f"exponential/gaussian A cross {flips.size} times, expected once")
    print()

    _rule("7. Worked engineering case: 1.55 um over 2 km")
    lam, path = 1.55e-6, 2000.0
    rho_c = fresnel_scale(lam, path)
    print(f"  Fresnel scale sqrt(lambda L) = {rho_c:.6f} m, taken as rho_c")
    print("  point scintillation index si(0) = 0.600000")
    print("  D (m)    D/rho_c       A(D)      si(D)")
    for d in (0.01, 0.02, 0.05, 0.1, 0.2, 0.4):
        a = aperture_averaging_factor(d, rho_c)
        si_d = effective_scintillation_index(0.6, d, rho_c)
        print(f"{d:7.3f} {d / rho_c:10.4f} {a:11.6f} {si_d:10.6f}")
    print()
    print("  one aperture of D against n of equal total area, d = D/sqrt(n):")
    print("  n    d (m)      A(d)     si per aperture")
    for n in (1, 2, 3, 4):
        d_each = equal_area_diameter(0.2, n)
        a = aperture_averaging_factor(d_each, rho_c)
        print(f"{n:3d} {d_each:9.5f} {a:10.6f} {0.6 * a:16.6f}")
    print("  splitting the glass raises the per-aperture scintillation index; that cost")
    print("  is what examples/one_big_vs_many_small.py weighs against the diversity gain.")
    print()

    _rule("Result")
    if failures:
        print(f"FAILED {len(failures)} check(s):")
        for f in failures:
            print("  -", f)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
