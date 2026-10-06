"""Validation 1: the fading quadrature against Monte Carlo, and the samplers.

The exact log-likelihood ratios in ``softdecode.llr`` replace an integral over
the fading density by a quadrature sum. Nothing in the package proves that
substitution is accurate, so it is measured here against a Monte Carlo
estimate of the same integral, with the Monte Carlo standard error reported so
that agreement means something.

Checks
------
1. ``sum(w) = 1``, ``sum(w h) = E[h] = 1`` and ``sum(w h**2) = E[h**2]``
   against the closed-form moments of each model.
2. ``p(y | b = 1) = integral N(y; a h, sigma**2) f(h) dh`` by quadrature
   against a Monte Carlo average over ``MC_SAMPLES`` draws of ``h``, for a
   grid of ``y``. Reported as ``|quad - mc| / mc`` and as a z score against
   the Monte Carlo standard error.
3. The same comparison in LLR units, which is what the package actually
   returns.
4. Node-count convergence of the quadrature.
5. The posterior quadrature of ``softdecode.csi`` against a Monte Carlo over
   the same posterior.
6. Sample moments and a Kolmogorov-Smirnov test of each sampler. The
   lognormal sample is standardised before the test because
   ``scipy.stats.kstest(x, "norm", args=(loc, scale))`` raises ``TypeError``
   on the installed SciPy; the standardised form is equivalent.

Runtime: about 20 s on one shared core.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from softdecode.channel import GammaGammaFading, LognormalFading, amplitude_quadrature
from softdecode.csi import MultiplicativeCsiError, csi_aware_llr_ook
from softdecode.detection import DetectionModel
from softdecode.llr import llr_ook_marginal

MC_SAMPLES = 4_000_000
SEED = 20261006
Y_GRID = np.array([-2.0, -0.5, 0.5, 1.5, 3.0, 5.0, 7.0, 9.0])


def banner(text: str) -> None:
    print()
    print(text)
    print("-" * len(text))


def moment_table() -> None:
    banner("1. Quadrature moments against closed-form moments")
    print(f"{'model':<26} {'nodes':>6} {'sum w':>12} {'E[h]':>12} {'E[h^2]':>12} "
          f"{'exact E[h^2]':>13} {'rel err':>11}")
    cases = [
        ("LognormalFading(0.1)", LognormalFading(0.1), 1.1),
        ("LognormalFading(0.3)", LognormalFading(0.3), 1.3),
        ("LognormalFading(1.0)", LognormalFading(1.0), 2.0),
        ("GammaGammaFading(4,2)", GammaGammaFading(4.0, 2.0), 1.25 * 1.5),
        ("GammaGammaFading(2.2,1.3)", GammaGammaFading(2.2, 1.3), (1 + 1 / 2.2) * (1 + 1 / 1.3)),
        ("GammaGammaFading(1,0.5)", GammaGammaFading(1.0, 0.5), (1 + 1 / 1.0) * (1 + 1 / 0.5)),
        ("GammaGammaFading(20,18)", GammaGammaFading(20.0, 18.0),
         (1 + 1 / 20.0) * (1 + 1 / 18.0)),
    ]
    for name, model, exact_m2 in cases:
        h, w = amplitude_quadrature(model)
        m1 = float((w * h).sum())
        m2 = float((w * h * h).sum())
        print(f"{name:<26} {h.size:>6} {w.sum():12.9f} {m1:12.9f} {m2:12.9f} "
              f"{exact_m2:13.9f} {abs(m2 - exact_m2) / exact_m2:11.3e}")


def likelihood_against_monte_carlo() -> None:
    banner("2. p(y | b=1) by quadrature against Monte Carlo over the same integral")
    det = DetectionModel()
    amplitude = det.ook_amplitude(10.0)
    print(f"a = {amplitude:.6f}, sigma = {det.sigma:.1f}, "
          f"Monte Carlo samples per model = {MC_SAMPLES}")
    for name, model in (
        ("lognormal sigma_I2=0.3", LognormalFading(0.3)),
        ("gamma-gamma (4, 2)", GammaGammaFading(4.0, 2.0)),
    ):
        rng = np.random.default_rng(SEED)
        h_mc = model.sample(MC_SAMPLES, rng)
        h_q, w_q = amplitude_quadrature(model)
        print(f"\n  {name}, quadrature nodes {h_q.size}")
        print(f"  {'y':>8} {'quadrature':>15} {'monte carlo':>15} {'mc se':>12} "
              f"{'rel diff':>11} {'z':>8}")
        worst_rel = 0.0
        worst_z = 0.0
        for y in Y_GRID:
            kern = np.exp(-((y - amplitude * h_mc) ** 2) / 2.0) / np.sqrt(2 * np.pi)
            mc = float(kern.mean())
            se = float(kern.std(ddof=1) / np.sqrt(MC_SAMPLES))
            quad = float(np.sum(w_q * np.exp(-((y - amplitude * h_q) ** 2) / 2.0)
                                / np.sqrt(2 * np.pi)))
            rel = abs(quad - mc) / mc
            z = (quad - mc) / se
            worst_rel = max(worst_rel, rel)
            worst_z = max(worst_z, abs(z))
            print(f"  {y:8.2f} {quad:15.9e} {mc:15.9e} {se:12.4e} {rel:11.3e} {z:8.3f}")
        print(f"  worst relative difference {worst_rel:.3e}, worst |z| {worst_z:.3f}")


def llr_against_monte_carlo() -> None:
    banner("3. The same comparison in LLR units, with the Monte Carlo error propagated")
    det = DetectionModel()
    amplitude = det.ook_amplitude(10.0)
    print("  se(LLR_mc) = se(p1)/p1, because LLR_mc = log p0 - log p1 and p0 is exact.")
    for name, model in (
        ("lognormal sigma_I2=0.3", LognormalFading(0.3)),
        ("gamma-gamma (4, 2)", GammaGammaFading(4.0, 2.0)),
    ):
        rng = np.random.default_rng(SEED + 1)
        h_mc = model.sample(MC_SAMPLES, rng)
        h_q, w_q = amplitude_quadrature(model)
        quad = llr_ook_marginal(Y_GRID, amplitude, h_q, w_q, det.sigma)
        print(f"\n  {name}, quadrature nodes {h_q.size}")
        print(f"  {'y':>8} {'quadrature LLR':>16} {'monte carlo LLR':>17} {'mc se':>10} "
              f"{'abs diff':>11} {'z':>8}")
        worst_z = 0.0
        for y, q in zip(Y_GRID, quad, strict=True):
            kern = np.exp(-((y - amplitude * h_mc) ** 2) / 2.0)
            p1 = float(kern.mean())
            se_llr = float(kern.std(ddof=1) / np.sqrt(MC_SAMPLES)) / p1
            mc = float(-(y**2) / 2.0 - np.log(p1))
            z = (q - mc) / se_llr
            worst_z = max(worst_z, abs(z))
            print(f"  {y:8.2f} {q:16.9f} {mc:17.9f} {se_llr:10.2e} "
                  f"{abs(q - mc):11.3e} {z:8.3f}")
        print(f"  worst |z| against the Monte Carlo standard error {worst_z:.3f}")


def node_convergence() -> None:
    banner("4. Node-count convergence of the quadrature (LLR, against the finest rule)")
    det = DetectionModel()
    amplitude = det.ook_amplitude(10.0)
    print("  Defaults: 80 Gauss-Hermite nodes (lognormal); for gamma-gamma an")
    print("  adaptive log-spaced trapezoid grid at a fixed log step of 0.08, which")
    print("  gives 265 points at (4, 2) and 379 at (2.2, 1.3). Both come from this")
    print("  table; the row whose node count equals the default is the shipped rule.")
    for name, model, counts, finest in (
        ("lognormal sigma_I2=0.3", LognormalFading(0.3), (10, 20, 40, 80, 120, 160), 300),
        ("lognormal sigma_I2=1.0", LognormalFading(1.0), (10, 20, 40, 80, 120, 160), 300),
        ("gamma-gamma (4, 2)", GammaGammaFading(4.0, 2.0), (33, 65, 129, 265, 513), 3201),
        ("gamma-gamma (2.2, 1.3)", GammaGammaFading(2.2, 1.3), (33, 65, 129, 379, 513), 3201),
    ):
        hf, wf = model.quadrature(finest)
        reference = llr_ook_marginal(Y_GRID, amplitude, hf, wf, det.sigma)
        print(f"\n  {name}, reference rule {hf.size} nodes")
        print(f"  {'nodes':>7} {'total nodes':>12} {'max |LLR - ref|':>17}")
        for n in counts:
            h, w = model.quadrature(n)
            got = llr_ook_marginal(Y_GRID, amplitude, h, w, det.sigma)
            print(f"  {n:7d} {h.size:12d} {np.max(np.abs(got - reference)):17.6e}")
        if isinstance(model, GammaGammaFading):
            print("  the tensor Gauss-Laguerre rule on the same reference, for comparison:")
            for n in (20, 40, 80):
                h, w = model.laguerre_quadrature(n)
                got = llr_ook_marginal(Y_GRID, amplitude, h, w, det.sigma)
                print(f"  {n:7d} {h.size:12d} {np.max(np.abs(got - reference)):17.6e}"
                      f"   (Gauss-Laguerre, {n} per factor)")


def posterior_against_monte_carlo() -> None:
    banner("5. Posterior quadrature of softdecode.csi against Monte Carlo")
    det = DetectionModel()
    amplitude = det.ook_amplitude(10.0)
    fading = LognormalFading(0.3)
    error = MultiplicativeCsiError(2.0, 1.5)
    rng = np.random.default_rng(SEED + 2)
    h_hat = np.array([0.4, 0.8, 1.0, 1.6, 2.5])
    y = np.array([0.5, 2.0, 4.0, 6.0, 7.5])
    quad = csi_aware_llr_ook(y, amplitude, h_hat, fading, error)
    m, s = error.log_posterior_moments(h_hat, fading)
    print(f"  bias {error.bias_db:+.2f} dB, jitter {error.jitter_db:.2f} dB, "
          f"posterior log-sd {float(s[0]):.6f}")
    print(f"  {'h_hat':>8} {'y':>7} {'quadrature':>15} {'monte carlo':>15} {'mc se':>10} "
          f"{'abs diff':>11} {'z':>7}")
    worst_z = 0.0
    draws_per_point = MC_SAMPLES // 4
    for i in range(h_hat.size):
        draws = np.exp(m[i] + s[i] * rng.standard_normal(draws_per_point))
        kern = np.exp(-((y[i] - amplitude * draws) ** 2) / 2.0)
        p1 = float(kern.mean())
        se = float(kern.std(ddof=1) / np.sqrt(draws_per_point)) / p1
        mc = float(-(y[i] ** 2) / 2.0 - np.log(p1))
        z = (float(quad[i]) - mc) / se
        worst_z = max(worst_z, abs(z))
        print(f"  {h_hat[i]:8.2f} {y[i]:7.2f} {float(quad[i]):15.9f} {mc:15.9f} {se:10.2e} "
              f"{abs(float(quad[i]) - mc):11.3e} {z:7.3f}")
    print(f"  Monte Carlo draws per point {draws_per_point}")
    print(f"  worst |z| against the Monte Carlo standard error {worst_z:.3f}")


def samplers() -> None:
    banner("6. Samplers: moments and goodness of fit")
    rng = np.random.default_rng(SEED + 3)
    n = 400_000
    for name, model in (
        ("LognormalFading(0.3)", LognormalFading(0.3)),
        ("GammaGammaFading(4,2)", GammaGammaFading(4.0, 2.0)),
    ):
        h = model.sample(n, rng)
        mean_se = float(h.std(ddof=1) / np.sqrt(n))
        print(f"  {name}")
        print(f"    sample mean     {h.mean():.6f}  (expected 1, se {mean_se:.2e}, "
              f"z {(h.mean() - 1.0) / mean_se:+.3f})")
        print(f"    sample variance {h.var(ddof=1):.6f}  "
              f"(expected {model.scintillation_index:.6f})")
    model = LognormalFading(0.3)
    print("    lognormal log-amplitude KS test, five independent draws of 400000")
    print("    (the sample is standardised first, then kstest(z, 'norm'))")
    pvalues = []
    for trial in range(5):
        h = model.sample(n, np.random.default_rng(SEED + 100 + trial))
        z = (np.log(h) - model.mu_x) / model.sigma_x
        ks = stats.kstest(z, "norm")
        pvalues.append(float(ks.pvalue))
        print(f"      draw {trial}: statistic {ks.statistic:.6f}, p = {ks.pvalue:.6f}")
    print(f"    draws with p > 0.01: {sum(p > 0.01 for p in pvalues)} of 5, "
          f"smallest p {min(pvalues):.6f}")
    gg = GammaGammaFading(4.0, 2.0)
    sample = gg.sample(200_000, rng)
    grid = np.exp(np.linspace(np.log(1e-4), np.log(30.0), 20001))
    pdf = gg.pdf(grid)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    ks_gg = stats.kstest(sample, lambda x: np.interp(x, grid, cdf))
    print(f"    gamma-gamma KS against the trapezoid CDF of its own pdf: "
          f"statistic {ks_gg.statistic:.6f}, p = {ks_gg.pvalue:.6f}")


def main() -> int:
    print("softdecode validation 1: fading quadrature and samplers")
    print("raw output committed as validation/quadrature_output.txt")
    print(f"seed {SEED}, Monte Carlo samples {MC_SAMPLES}")
    moment_table()
    likelihood_against_monte_carlo()
    llr_against_monte_carlo()
    node_convergence()
    posterior_against_monte_carlo()
    samplers()
    print("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
