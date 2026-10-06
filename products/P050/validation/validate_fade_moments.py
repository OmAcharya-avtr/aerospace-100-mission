"""Validation: the fade models reproduce the identities they are defined by.

Checks
------
1. **Lognormal dB parameters.** ``sigma_lnI = sqrt(ln(1 + sigma_I^2))`` and
   ``sigma_dB = (10/ln 10) sigma_lnI``, against values computed independently
   in this script from ``math.log`` -- i.e. the package formula is re-derived,
   not re-used.
2. **Lognormal normalisation.** Under ``E[I] = 1`` the sampled mean irradiance
   must be 1 to within the Monte Carlo standard error, and the mean of
   ``10 log10 I`` must be the *negative* number the model reports. This is the
   assumption most easily got wrong, so it is measured.
3. **Lognormal availability.** The closed-form availability against a Monte
   Carlo of the same model, reported as a z score against the binomial
   standard error.
4. **Gamma-gamma moments.** Total mass, mean and scintillation index by
   adaptive quadrature of the pdf, against 1, 1 and the requested
   ``sigma_I^2`` from ``1/alpha + 1/beta + 1/(alpha beta)``.
5. **Gamma-gamma survival.** The quadrature survival function against a Monte
   Carlo of a gamma-gamma variate built as a product of two gamma variates,
   which is the model's own construction and an independent route to the same
   distribution.
6. **Gamma-gamma against lognormal in the weak-turbulence limit.** They are
   different models, so this reports the size of the disagreement rather than
   asserting agreement.
7. **Empirical survival function.** Exact values on a hand-written sample, and
   the ``1/n`` resolution floor.

References
----------
* Andrews, L. C. and Phillips, R. L., *Laser Beam Propagation through Random
  Media*, 2nd ed., SPIE Press, 2005 -- lognormal irradiance model and the
  scintillation index.
* Al-Habash, M. A., Andrews, L. C. and Phillips, R. L., *Optical Engineering*
  40(8), 2001 -- the gamma-gamma irradiance model and its moments.

Runtime: about 20 s on one contended core.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
from scipy import integrate  # noqa: E402

from coderateopt import (  # noqa: E402
    EmpiricalFade,
    GammaGammaFade,
    LognormalFade,
    db_to_linear,
    gamma_gamma_pdf,
    sigma_ln_i_from_scintillation,
)

SEED = 20261006
failures: list[str] = []


def check(label: str, ok: bool) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        failures.append(label)


print("1. Lognormal dB parameters, package value against an independent re-derivation")
print(f"  {'sigma_I^2':>10s} {'sigma_lnI':>14s} {'independent':>14s} {'sigma_dB':>12s}")
for si in (0.01, 0.05, 0.2, 0.5, 1.0):
    model = LognormalFade(si)
    independent = math.sqrt(math.log(1.0 + si))
    print(
        f"  {si:>10.3f} {model.sigma_ln_i:>14.10f} {independent:>14.10f} {model.sigma_db:>12.8f}"
    )
    check(
        f"sigma_lnI at sigma_I^2 = {si}",
        abs(model.sigma_ln_i - independent) < 1e-14,
    )
    check(
        f"sigma_dB at sigma_I^2 = {si}",
        abs(model.sigma_db - (10.0 / math.log(10.0)) * independent) < 1e-13,
    )
print(f"  sigma_lnI_from_scintillation(0.2) = {sigma_ln_i_from_scintillation(0.2):.12f}")

print()
print("2. Lognormal normalisation: E[I] = 1 and E[10 log10 I] < 0")
rng = np.random.default_rng(SEED)
n = 400_000
for si in (0.05, 0.2, 0.6):
    model = LognormalFade(si)
    samples_db = model.sample_db(n, rng)
    irradiance = np.asarray(db_to_linear(samples_db))
    mean_i = float(irradiance.mean())
    se_i = float(irradiance.std(ddof=1) / math.sqrt(n))
    z = (mean_i - 1.0) / se_i
    print(
        f"  sigma_I^2 = {si:.2f}  E[I] = {mean_i:.6f} +/- {se_i:.6f}  z = {z:+.3f}   "
        f"E[dB] sample {samples_db.mean():+.6f}  model {model.mean_db:+.6f}"
    )
    check(f"E[I] = 1 at sigma_I^2 = {si} (|z| < 4)", abs(z) < 4.0)
    check(f"E[dB] < 0 at sigma_I^2 = {si}", model.mean_db < 0.0)
    check(
        f"sample E[dB] matches model at sigma_I^2 = {si}",
        abs(samples_db.mean() - model.mean_db) < 5.0 * model.sigma_db / math.sqrt(n),
    )

print()
print("3. Lognormal availability, closed form against Monte Carlo (n = 400000)")
print(f"  {'sigma_I^2':>10s} {'level dB':>9s} {'closed form':>13s} {'monte carlo':>13s} {'z':>8s}")
for si in (0.05, 0.2, 0.6):
    model = LognormalFade(si)
    samples_db = model.sample_db(n, rng)
    for level in (-1.0, -3.0, -6.0):
        closed = float(model.exceedance_db(level))
        empirical = float(np.mean(samples_db >= level))
        se = math.sqrt(max(closed * (1.0 - closed), 1e-15) / n)
        z = (empirical - closed) / se
        print(f"  {si:>10.2f} {level:>9.1f} {closed:>13.8f} {empirical:>13.8f} {z:>+8.3f}")
        check(f"availability at sigma_I^2={si}, level={level} (|z| < 4)", abs(z) < 4.0)

print()
print("4. Gamma-gamma moments by quadrature against the defining identities")
print(
    f"  {'sigma_I^2':>10s} {'alpha':>10s} {'beta':>10s} {'mass':>12s} {'E[I]':>12s} "
    f"{'SI from moments':>16s}"
)
for si in (0.05, 0.2, 0.6, 1.5):
    for ratio in (1.0, 0.4):
        gg = GammaGammaFade.from_scintillation(si, ratio=ratio)

        def density(t: float, model: GammaGammaFade = gg) -> float:
            return float(gamma_gamma_pdf(np.array([t]), model.alpha, model.beta)[0])

        mass, _ = integrate.quad(density, 0.0, np.inf, limit=400)
        mean, _ = integrate.quad(lambda t: t * density(t), 0.0, np.inf, limit=400)
        second, _ = integrate.quad(lambda t: t * t * density(t), 0.0, np.inf, limit=400)
        si_moments = second / mean**2 - 1.0
        print(
            f"  {si:>10.2f} {gg.alpha:>10.5f} {gg.beta:>10.5f} {mass:>12.9f} "
            f"{mean:>12.9f} {si_moments:>16.9f}"
        )
        check(f"mass = 1 at sigma_I^2={si}, ratio={ratio}", abs(mass - 1.0) < 1e-7)
        check(f"E[I] = 1 at sigma_I^2={si}, ratio={ratio}", abs(mean - 1.0) < 1e-7)
        check(
            f"SI from moments = {si} at ratio={ratio}",
            abs(si_moments - si) < 1e-5 * max(si, 1.0),
        )
        check(
            f"closed-form SI identity at sigma_I^2={si}, ratio={ratio}",
            abs(gg.scintillation_index - si) < 1e-12,
        )

print()
print("5. Gamma-gamma survival: quadrature against a product-of-gammas Monte Carlo")
print(f"  {'sigma_I^2':>10s} {'level dB':>9s} {'quadrature':>12s} {'monte carlo':>12s} {'z':>8s}")
m = 300_000
for si in (0.2, 0.6, 1.5):
    gg = GammaGammaFade.from_scintillation(si)
    # I = X Y with X ~ Gamma(alpha, 1/alpha), Y ~ Gamma(beta, 1/beta): the
    # model's own construction, giving E[I] = 1 and the stated moments.
    x = rng.gamma(shape=gg.alpha, scale=1.0 / gg.alpha, size=m)
    y = rng.gamma(shape=gg.beta, scale=1.0 / gg.beta, size=m)
    irradiance = x * y
    for level in (-1.0, -3.0, -6.0):
        quad_value = float(gg.exceedance_db(level))
        empirical = float(np.mean(irradiance >= float(db_to_linear(level))))
        se = math.sqrt(max(quad_value * (1.0 - quad_value), 1e-15) / m)
        z = (empirical - quad_value) / se
        print(f"  {si:>10.2f} {level:>9.1f} {quad_value:>12.8f} {empirical:>12.8f} {z:>+8.3f}")
        check(f"gamma-gamma survival at sigma_I^2={si}, level={level} (|z| < 4)", abs(z) < 4.0)
    sample_si = float(np.var(irradiance) / np.mean(irradiance) ** 2)
    print(f"    sample scintillation index {sample_si:.6f} against requested {si:.6f}")

print()
print("6. Gamma-gamma against lognormal in weak turbulence (different models)")
print(
    f"  {'sigma_I^2':>10s} {'level dB':>9s} {'gamma-gamma':>13s} "
    f"{'lognormal':>13s} {'diff':>11s}"
)
worst = 0.0
for si in (0.01, 0.02, 0.05, 0.2):
    gg = GammaGammaFade.from_scintillation(si)
    ln = LognormalFade(si)
    for level in (-1.0, -2.0, -3.0):
        a, b = float(gg.exceedance_db(level)), float(ln.exceedance_db(level))
        worst = max(worst, abs(a - b))
        print(f"  {si:>10.2f} {level:>9.1f} {a:>13.8f} {b:>13.8f} {a - b:>+11.8f}")
print(f"  worst absolute availability difference over the block: {worst:.6f}")
print("  reported, not asserted: these are two distinct models, not one identity")

print()
print("7. Empirical survival function on a hand-written sample")
emp = EmpiricalFade(np.array([-6.0, -4.0, -2.0, 0.0]))
expected = {-7.0: 1.0, -6.0: 1.0, -5.9: 0.75, -2.0: 0.5, 0.0: 0.25, 0.1: 0.0}
for level, want in expected.items():
    got = float(emp.exceedance_db(level))
    print(f"  P[fade >= {level:>5.1f}] = {got:.4f}, expected {want:.4f}")
    check(f"empirical survival at {level}", abs(got - want) < 1e-15)
print(f"  resolution floor 1/n = {emp.resolution:.6f} for n = {emp.samples_db.size}")
check("empirical resolution is 1/n", abs(emp.resolution - 0.25) < 1e-15)

print()
print("8. Quantile inversion: exceedance(quantile(p)) == p")
for model_name, model in (
    ("lognormal(0.2)", LognormalFade(0.2)),
    ("gamma-gamma(0.5)", GammaGammaFade.from_scintillation(0.5)),
):
    for prob in (0.5, 0.9, 0.99, 0.999):
        level = model.quantile_db(prob)
        back = float(model.exceedance_db(level))
        print(f"  {model_name:<18s} p = {prob:<7.4f} level = {level:>9.4f} dB -> {back:.9f}")
        check(f"quantile inversion {model_name} at p={prob}", abs(back - prob) < 1e-7)

print()
print(f"checks run: {len(failures)} failed")
if failures:
    for item in failures:
        print(f"  FAILED: {item}")
    sys.exit(1)
print("all fade-model identity checks passed")
