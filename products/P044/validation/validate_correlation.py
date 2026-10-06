"""Validation: inter-aperture correlation, log domain against irradiance.

Checks
------
1. The exact lognormal relation
   ``corr(I_j, I_k) = (exp(R_jk s^2) - 1)/(exp(s^2) - 1)``
   against Monte Carlo, over a grid of log-domain correlations and
   scintillation indices. This is the relation that makes every quoted
   "correlation between apertures" ambiguous unless the domain is stated,
   and the gap between the two numbers is reported in the table.
2. The copula sampler reproduces its inputs: unit-mean marginals, the
   requested marginal scintillation index, and the requested *log-domain*
   correlation matrix.
3. Monotonicity: the irradiance correlation is at or below the log-domain
   correlation for every ``R in [0, 1]`` and every ``si``, with equality
   only at ``R in {0, 1}``.
4. The correlated gamma-gamma sampler reproduces its marginals (unit mean,
   closed-form scintillation index) and produces the expected ordering of
   inter-aperture irradiance correlations when only the large-scale factor
   is correlated.
5. ``nearest_psd`` is the identity, to rounding, on a matrix that is
   already a valid correlation matrix, and repairs one that is not.

Sizing: 400000 realisations per point; the standard error on a sample
correlation near 0.4 is about 1/sqrt(n) = 0.0016, so a tolerance of 4
standard errors is 0.0063.

Runtime about 30 s on one core.
"""

from __future__ import annotations

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from aperturediv.channel import (  # noqa: E402
    gamma_gamma_params_from_rytov,
    gamma_gamma_scintillation_index,
    lognormal_sigma_log,
)
from aperturediv.correlation import (  # noqa: E402
    correlation_matrix,
    equispaced_positions,
    irradiance_correlation_matrix,
    log_to_irradiance_correlation,
    nearest_psd,
    sample_correlated_gamma_gamma,
    sample_correlated_lognormal,
)

N = 400_000
SEED = 44044
R_GRID = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0)
SI_GRID = (0.1, 0.3, 0.6, 0.9)


def _rule(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    failures: list[str] = []
    rng = np.random.default_rng(SEED)

    _rule("1. Irradiance correlation from log-domain correlation, closed form vs MC")
    print(f"  n = {N} realisations per point, two apertures")
    print("   si   R_log   corr(I) closed   corr(I) sampled      diff   4*se     gap R-corrI")
    worst = 0.0
    for si in SI_GRID:
        for r in R_GRID:
            mat = np.array([[1.0, r], [r, 1.0]])
            irr = sample_correlated_lognormal(N, si, mat, rng)
            closed = float(log_to_irradiance_correlation(r, si))
            samp = float(np.corrcoef(irr.T)[0, 1])
            se4 = 4.0 / np.sqrt(N)
            worst = max(worst, abs(samp - closed))
            print(
                f"{si:5.2f} {r:7.3f} {closed:16.9f} {samp:17.9f} {samp - closed:+10.2e} "
                f"{se4:7.4f} {r - closed:+13.6f}"
            )
            if abs(samp - closed) > se4:
                failures.append(
                    f"lognormal irradiance correlation si={si} R={r}: "
                    f"closed {closed:.6f} sampled {samp:.6f}"
                )
    print(f"  worst absolute deviation {worst:.3e}, tolerance 4/sqrt(n) = {4 / np.sqrt(N):.4e}")
    print()
    print("  The last column is the amount by which quoting the log-domain correlation")
    print("  overstates the correlation the combiner actually sees. It reaches")
    print(f"  {max(r - float(log_to_irradiance_correlation(r, 0.9)) for r in R_GRID):.4f}"
          " at si = 0.9 on this grid.")
    print()

    _rule("2. Copula sampler reproduces its inputs (4 apertures on a line)")
    si = 0.6
    pos = equispaced_positions(4, 0.05)
    r_log = correlation_matrix(pos, 0.10, "gaussian")
    irr = sample_correlated_lognormal(N, si, r_log, rng)
    s = lognormal_sigma_log(si)
    z = (np.log(irr) + 0.5 * s * s) / s
    print(f"  spacing 0.05 m, rho_c 0.10 m, si {si}")
    print(f"  column means              {np.array2string(irr.mean(0), precision=6)}")
    print(f"  column scintillation idx  "
          f"{np.array2string(irr.var(0) / irr.mean(0) ** 2, precision=6)}")
    print("  requested log correlation")
    print("   ", np.array2string(r_log, precision=6).replace("\n", "\n    "))
    print("  sampled log correlation")
    print("   ", np.array2string(np.corrcoef(z.T), precision=6).replace("\n", "\n    "))
    print("  requested irradiance correlation")
    print("   ", np.array2string(
        irradiance_correlation_matrix(pos, 0.10, si), precision=6
    ).replace("\n", "\n    "))
    print("  sampled irradiance correlation")
    print("   ", np.array2string(np.corrcoef(irr.T), precision=6).replace("\n", "\n    "))
    worst_mean = float(np.max(np.abs(irr.mean(0) - 1.0)))
    worst_si = float(np.max(np.abs(irr.var(0) / irr.mean(0) ** 2 - si)))
    worst_log = float(np.max(np.abs(np.corrcoef(z.T) - r_log)))
    worst_irr = float(np.max(np.abs(np.corrcoef(irr.T) - irradiance_correlation_matrix(
        pos, 0.10, si))))
    print(f"  worst |mean - 1|                   {worst_mean:.5f}, tolerance 0.01")
    print(f"  worst |sampled si - si|            {worst_si:.5f}, tolerance 0.02")
    print(f"  worst |sampled - requested| log    {worst_log:.5f}, tolerance 0.01")
    print(f"  worst |sampled - requested| irrad  {worst_irr:.5f}, tolerance 0.01")
    for name, val, tol in (
        ("mean", worst_mean, 0.01),
        ("si", worst_si, 0.02),
        ("log correlation", worst_log, 0.01),
        ("irradiance correlation", worst_irr, 0.01),
    ):
        if val > tol:
            failures.append(f"copula sampler {name} off by {val}")
    print()

    _rule("3. Irradiance correlation never exceeds log correlation")
    fine_r = np.linspace(0.0, 1.0, 201)
    ok = True
    for si in (0.05, 0.2, 0.5, 1.0, 2.0):
        c = log_to_irradiance_correlation(fine_r, si)
        gap = fine_r - c
        strict_interior = bool(np.all(gap[1:-1] > 0.0))
        ends = float(abs(c[0])) + float(abs(c[-1] - 1.0))
        print(f"  si={si:5.2f}: max gap {gap.max():.6f} at R={fine_r[int(np.argmax(gap))]:.3f}; "
              f"strictly positive on 0<R<1: {strict_interior}; endpoint error {ends:.2e}")
        ok = ok and strict_interior and ends < 1e-12
    if not ok:
        failures.append("irradiance correlation ordering or endpoints violated")
    print()

    _rule("4. Correlated gamma-gamma sampler")
    alpha, beta = gamma_gamma_params_from_rytov(1.0)
    si_gg = gamma_gamma_scintillation_index(alpha, beta)
    r_big = correlation_matrix(pos, 0.10, "gaussian")
    gg = sample_correlated_gamma_gamma(N, alpha, beta, r_big, rng)
    print(f"  alpha {alpha:.6f}  beta {beta:.6f}  closed-form si {si_gg:.6f}")
    print(f"  column means              {np.array2string(gg.mean(0), precision=6)}")
    print(f"  column scintillation idx  "
          f"{np.array2string(gg.var(0) / gg.mean(0) ** 2, precision=6)}")
    print("  sampled irradiance correlation (only the large-scale factor is correlated)")
    print("   ", np.array2string(np.corrcoef(gg.T), precision=6).replace("\n", "\n    "))
    print(f"  large-scale copula correlation of adjacent apertures: {r_big[0, 1]:.6f}")
    print(f"  resulting irradiance correlation of adjacent apertures: "
          f"{np.corrcoef(gg.T)[0, 1]:.6f}")
    print("  the second is far below the first because the independent small-scale")
    print("  factor contributes variance but no covariance.")
    gg_mean_err = float(np.max(np.abs(gg.mean(0) - 1.0)))
    gg_si_err = float(np.max(np.abs(gg.var(0) / gg.mean(0) ** 2 - si_gg)))
    print(f"  worst |mean - 1| {gg_mean_err:.5f}, tolerance 0.02")
    print(f"  worst |sampled si - closed si| {gg_si_err:.5f}, tolerance 0.03")
    if gg_mean_err > 0.02:
        failures.append(f"gamma-gamma copula mean off by {gg_mean_err}")
    if gg_si_err > 0.03:
        failures.append(f"gamma-gamma copula si off by {gg_si_err}")
    off_diag = np.corrcoef(gg.T)[0, 1]
    if not 0.0 < off_diag < r_big[0, 1]:
        failures.append(f"gamma-gamma adjacent irradiance correlation {off_diag} out of range")
    print()

    _rule("5. nearest_psd")
    repaired = nearest_psd(r_log)
    print(f"  max |nearest_psd(valid R) - R|  {np.max(np.abs(repaired - r_log)):.3e}, "
          "tolerance 1.0e-10")
    if np.max(np.abs(repaired - r_log)) > 1e-10:
        failures.append("nearest_psd is not the identity on a valid correlation matrix")
    bad = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
    eig_bad = np.linalg.eigvalsh(bad)
    fixed = nearest_psd(bad, floor=1e-12)
    eig_fixed = np.linalg.eigvalsh(fixed)
    print(f"  an invalid input with eigenvalues {np.array2string(eig_bad, precision=5)}")
    print(f"  repaired to eigenvalues          {np.array2string(eig_fixed, precision=5)}")
    print(f"  unit diagonal preserved: {bool(np.allclose(np.diag(fixed), 1.0))}")
    if eig_fixed.min() < -1e-12 or not np.allclose(np.diag(fixed), 1.0):
        failures.append("nearest_psd did not repair an invalid matrix")
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
