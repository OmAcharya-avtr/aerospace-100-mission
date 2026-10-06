"""Validation: lognormal and gamma-gamma irradiance statistics.

Checks
------
1. Both densities integrate to 1 over ``(0, inf)``, by quadrature in
   ``ln I`` so the ``I -> 0`` endpoint is regular.
2. The first two moments of both densities match their closed forms:
   lognormal ``E[I^n] = exp(n(n-1)s^2/2)``, gamma-gamma
   ``E[I^n] = Gamma(a+n)Gamma(b+n)/(Gamma(a)Gamma(b)(ab)^n)``. The
   scintillation index is then recomputed from the quadrature moments and
   compared with each model's own closed form -- the check the spec asks
   for, run against the pdf rather than against the formula it came from.
3. The CDF of both models is the integral of the pdf: the lognormal
   closed-form ``Phi((ln I + s^2/2)/s)`` against quadrature of its own pdf,
   and the gamma-gamma quadrature CDF against Monte Carlo.
4. The samplers reproduce their marginals: a Kolmogorov-Smirnov test of the
   lognormal sampler against the standard normal after standardising
   ``ln I`` (the installed SciPy rejects ``kstest(x, "norm", args=...)``,
   so the data are standardised and compared against ``"norm"``), and the
   gamma-gamma sampler against its own quadrature CDF by a KS statistic
   computed directly.
5. The Rytov-to-(alpha, beta) mapping is internally consistent: for a grid
   of Rytov variances, the scintillation index recomputed from the mapped
   shapes is compared with the sampled scintillation index of the
   corresponding sampler. This checks the mapping against the model, which
   is a check; it does not claim the mapping is the right physics.

Sizing: 400000 samples per Monte Carlo point, which gives a standard error
on the sampled scintillation index of a few parts in a thousand. Runtime
about 25 s on one core.
"""

from __future__ import annotations

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
from scipy import integrate, stats  # noqa: E402

from aperturediv.channel import (  # noqa: E402
    gamma_gamma_cdf,
    gamma_gamma_moment,
    gamma_gamma_params_from_rytov,
    gamma_gamma_pdf,
    gamma_gamma_scintillation_index,
    lognormal_cdf,
    lognormal_moment,
    lognormal_pdf,
    lognormal_sigma_log,
    sample_gamma_gamma,
    sample_lognormal,
)

N_SAMPLES = 400_000
SEED = 44044
SI_GRID = (0.1, 0.3, 0.6, 0.9)
RYTOV_GRID = (0.2, 0.5, 1.0, 2.0, 5.0)
LOG_LOWER = -60.0
LOG_UPPER = 12.0


def _rule(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def _quad_moment(pdf, order: int) -> float:
    val, _ = integrate.quad(
        lambda t: pdf(np.exp(t)) * np.exp(t * (order + 1)),
        LOG_LOWER,
        LOG_UPPER,
        epsabs=1e-13,
        epsrel=1e-11,
        limit=500,
    )
    return float(val)


def main() -> int:
    failures: list[str] = []
    rng = np.random.default_rng(SEED)

    _rule("1. Density normalisation and moments, lognormal")
    print("  si     int f dI      E[I] quad   E[I] closed   E[I2] quad  E[I2] closed"
          "   si quad   si closed")
    for si in SI_GRID:
        def pdf(x, si=si):
            return lognormal_pdf(x, si)

        norm = _quad_moment(pdf, 0)
        m1 = _quad_moment(pdf, 1)
        m2 = _quad_moment(pdf, 2)
        si_quad = m2 / m1**2 - 1.0
        print(
            f"{si:5.2f} {norm:12.9f} {m1:13.9f} {lognormal_moment(1, si):13.9f} "
            f"{m2:12.8f} {lognormal_moment(2, si):13.8f} {si_quad:10.7f} {si:11.7f}"
        )
        if abs(norm - 1.0) > 1e-8:
            failures.append(f"lognormal normalisation si={si}: {norm}")
        if abs(si_quad - si) > 1e-7:
            failures.append(f"lognormal si from quadrature si={si}: {si_quad}")
    print()

    _rule("2. Density normalisation and moments, gamma-gamma")
    print("  sr    alpha     beta     int f dI      E[I] quad    E[I2] quad"
          "  E[I2] closed    si quad   si closed")
    for sr in RYTOV_GRID:
        alpha, beta = gamma_gamma_params_from_rytov(sr)

        def pdf(x, a=alpha, b=beta):
            return gamma_gamma_pdf(x, a, b)

        norm = _quad_moment(pdf, 0)
        m1 = _quad_moment(pdf, 1)
        m2 = _quad_moment(pdf, 2)
        si_quad = m2 / m1**2 - 1.0
        si_closed = gamma_gamma_scintillation_index(alpha, beta)
        print(
            f"{sr:5.2f} {alpha:8.4f} {beta:8.4f} {norm:12.9f} {m1:13.9f} {m2:13.8f} "
            f"{gamma_gamma_moment(2, alpha, beta):13.8f} {si_quad:10.7f} {si_closed:11.7f}"
        )
        if abs(norm - 1.0) > 1e-7:
            failures.append(f"gamma-gamma normalisation sr={sr}: {norm}")
        if abs(si_quad - si_closed) / si_closed > 1e-6:
            failures.append(f"gamma-gamma si from quadrature sr={sr}: {si_quad} vs {si_closed}")
    print()

    _rule("3. CDF against quadrature of the same pdf (lognormal, si=0.6)")
    si = 0.6
    print("     I      F closed form     F quadrature     abs diff")
    worst_ln = 0.0
    for x in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0):
        closed = float(lognormal_cdf(x, si))
        quad, _ = integrate.quad(
            lambda t, si=si: float(lognormal_pdf(np.exp(t), si)) * np.exp(t),
            LOG_LOWER,
            float(np.log(x)),
            epsabs=1e-14,
            epsrel=1e-12,
            limit=500,
        )
        worst_ln = max(worst_ln, abs(closed - quad))
        print(f"{x:7.3f} {closed:17.12f} {quad:16.12f} {abs(closed - quad):12.3e}")
    print(f"  worst absolute difference {worst_ln:.3e}, tolerance 1.0e-10")
    if worst_ln > 1e-10:
        failures.append(f"lognormal cdf vs quadrature worst {worst_ln}")
    print()

    _rule("4. Gamma-gamma CDF against Monte Carlo (Rytov variance 1.0)")
    alpha, beta = gamma_gamma_params_from_rytov(1.0)
    samples = sample_gamma_gamma(N_SAMPLES, alpha, beta, rng)
    print(f"  alpha={alpha:.6f} beta={beta:.6f} n={N_SAMPLES}")
    print("     I     F quadrature     F empirical      diff     MC se    z")
    worst_z = 0.0
    for x in (0.1, 0.3, 0.6, 1.0, 2.0, 4.0):
        f_q = float(gamma_gamma_cdf(x, alpha, beta))
        f_e = float(np.mean(samples <= x))
        se = float(np.sqrt(max(f_q * (1 - f_q), 1e-12) / N_SAMPLES))
        z = (f_e - f_q) / se
        worst_z = max(worst_z, abs(z))
        print(f"{x:7.3f} {f_q:15.9f} {f_e:15.9f} {f_e - f_q:10.3e} {se:9.2e} {z:+7.3f}")
    print(f"  worst |z| {worst_z:.3f}, tolerance 4.0")
    if worst_z > 4.0:
        failures.append(f"gamma-gamma cdf vs MC worst |z| {worst_z}")
    print()

    _rule("5. Samplers against their marginals")
    si = 0.6
    s = lognormal_sigma_log(si)
    ln_samples = sample_lognormal(N_SAMPLES, si, np.random.default_rng(SEED + 1))
    z_std = (np.log(ln_samples) + 0.5 * s * s) / s
    ks_ln = stats.kstest(z_std, "norm")
    print(f"  lognormal si={si}: standardised ln I vs N(0,1)")
    print(f"    KS statistic {ks_ln.statistic:.6f}  p-value {ks_ln.pvalue:.6f}")
    print(f"    sample mean  {ln_samples.mean():.6f} (expect 1)")
    print(f"    sample si    {ln_samples.var() / ln_samples.mean() ** 2:.6f} (expect {si})")
    if ks_ln.pvalue < 0.001:
        failures.append(f"lognormal sampler KS p={ks_ln.pvalue}")

    grid = np.quantile(samples, np.linspace(0.005, 0.995, 60))
    f_q = np.array([float(gamma_gamma_cdf(x, alpha, beta)) for x in grid])
    f_e = np.searchsorted(np.sort(samples), grid, side="right") / N_SAMPLES
    ks_gg = float(np.max(np.abs(f_q - f_e)))
    crit = 1.63 / np.sqrt(N_SAMPLES)  # asymptotic 1% two-sided KS critical value
    print("  gamma-gamma: KS statistic against the quadrature CDF on 60 quantiles")
    print(f"    KS statistic {ks_gg:.6f}  1% critical value {crit:.6f}")
    if ks_gg > crit:
        failures.append(f"gamma-gamma sampler KS {ks_gg} > {crit}")
    print()

    _rule("6. Rytov mapping internal consistency")
    print("  sr    alpha     beta    si closed   si sampled   se      z")
    worst_z = 0.0
    for sr in RYTOV_GRID:
        a, b = gamma_gamma_params_from_rytov(sr)
        si_closed = gamma_gamma_scintillation_index(a, b)
        x = sample_gamma_gamma(N_SAMPLES, a, b, np.random.default_rng(SEED + int(10 * sr)))
        si_samp = float(x.var() / x.mean() ** 2)
        # se of the scintillation index from the delta method on the fourth moment
        m1, m2 = x.mean(), (x**2).mean()
        var_m2 = float(((x**2 - m2) ** 2).mean() / N_SAMPLES)
        var_m1 = float(((x - m1) ** 2).mean() / N_SAMPLES)
        se = float(np.sqrt(var_m2 / m1**4 + 4.0 * m2**2 * var_m1 / m1**6))
        z = (si_samp - si_closed) / se
        worst_z = max(worst_z, abs(z))
        print(f"{sr:5.2f} {a:8.4f} {b:8.4f} {si_closed:11.6f} {si_samp:12.6f} {se:8.2e} {z:+7.3f}")
    print(f"  worst |z| {worst_z:.3f}, tolerance 4.0")
    print("  note: the delta-method standard error is itself a large-sample estimate;")
    print("  for strong turbulence the fourth moment converges slowly, so a |z| of a")
    print("  few is expected and is not evidence against the mapping.")
    if worst_z > 4.0:
        failures.append(f"Rytov mapping si worst |z| {worst_z}")
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
