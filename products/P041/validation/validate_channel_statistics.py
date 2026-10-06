"""Channel marginal and correlation: do the generated paths match the specification?

Checks
------
1. Lognormal marginal: sample mean irradiance against 1, sample scintillation
   index against the target, and a Kolmogorov-Smirnov test of ``ln I`` against
   ``N(-s^2/2, s^2)`` on thinned (roughly independent) samples.
2. AR(1) autocorrelation against ``R(k) = exp(-k/Lc)`` and the Gaussian kernel
   against ``exp(-(k/Lc)^2)``.
3. The measured 1/e correlation time against the specified one, for both kernels
   and both marginals. The gamma-gamma copula construction is expected to
   disagree; the size of the disagreement is the output.
4. The gamma-gamma scintillation-index identity: equation (6) applied to
   ``(alpha, beta)`` from equations (7)-(9) must equal ``exp(sx2+sy2)-1``
   exactly.

Run: ``PYTHONPATH=../src python3 validate_channel_statistics.py``
Runtime: about 30 s on one core.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from codedfade.channel import (
    ChannelConfig,
    autocorrelation,
    correlated_gaussian,
    gamma_gamma_parameters_from_si,
    gamma_gamma_scintillation_index,
    generate_irradiance,
    measured_correlation_time,
    rytov_to_gamma_gamma,
)

N = 400_000
FS = 1.0e6
TAU = 2.0e-4


def main() -> None:
    print("=" * 78)
    print("CHANNEL STATISTICS VALIDATION")
    print("=" * 78)
    print(f"samples per path         {N}")
    print(f"sample rate              {FS:.6e} Hz")
    print(f"correlation time         {TAU:.6e} s  (Lc = {TAU * FS:.1f} samples)")
    print()

    print("-- 1. lognormal marginal ---------------------------------------------")
    print(f"{'SI target':>10} {'mean I':>12} {'SI sample':>12} {'rel err':>10} "
          f"{'KS stat':>10} {'KS p':>10}")
    for row, si in enumerate((0.1, 0.3, 0.6, 1.0, 1.5)):
        # a different seed per row, so the five KS results are independent checks
        # rather than the same latent Gaussian seen through five transforms
        cfg = ChannelConfig(si, TAU, FS, "lognormal", "exp", seed=11 + row)
        i = generate_irradiance(cfg, N)
        si_hat = float(np.var(i) / np.mean(i) ** 2)
        s2 = cfg.log_irradiance_variance
        thin = int(4 * cfg.samples_per_correlation_time)
        # standardise with the model's own mean and variance, then test against the
        # standard normal: this checks the shape and the claimed parameters at once
        z = (np.log(i)[::thin] + 0.5 * s2) / np.sqrt(s2)
        ks = stats.kstest(z, "norm")
        print(
            f"{si:10.4f} {float(i.mean()):12.6f} {si_hat:12.6f} "
            f"{abs(si_hat - si) / si:10.6f} {ks.statistic:10.6f} {ks.pvalue:10.6f}"
        )
    print()

    print("-- 2. autocorrelation of the underlying Gaussian ---------------------")
    for kernel, model in (("exp", "exp(-k/Lc)"), ("gauss", "exp(-(k/Lc)^2)")):
        lc = 20.0
        g = correlated_gaussian(N, lc, np.random.default_rng(12), kernel)  # type: ignore[arg-type]
        acf = autocorrelation(g, 40)
        k = np.arange(41)
        expected = np.exp(-k / lc) if kernel == "exp" else np.exp(-((k / lc) ** 2))
        print(
            f"kernel={kernel:<6} model={model:<16} max |ACF - model| over lags 0..40 "
            f"= {float(np.max(np.abs(acf - expected))):.6f}"
        )
        print(
            f"              ACF at lag Lc={int(lc)} = {acf[int(lc)]:.6f}  "
            f"(1/e = {np.exp(-1):.6f})"
        )
    print()

    print("-- 3. measured 1/e correlation time of log-irradiance ----------------")
    print(f"{'marginal':<12} {'kernel':<7} {'target s':>12} {'measured s':>12} {'rel err':>10}")
    for marginal in ("lognormal", "gammagamma"):
        for kernel in ("exp", "gauss"):
            cfg = ChannelConfig(0.6, TAU, FS, marginal, kernel, seed=13)  # type: ignore[arg-type]
            i = generate_irradiance(cfg, N)
            tau_hat = measured_correlation_time(i, FS)
            print(
                f"{marginal:<12} {kernel:<7} {TAU:12.6e} {tau_hat:12.6e} "
                f"{abs(tau_hat - TAU) / TAU:10.6f}"
            )
    print()
    print("All four rows are measurements, not checks against a reference. The")
    print("gamma-gamma rows carry an additional caveat: the temporal correlation is")
    print("imposed through a Gaussian copula, so the specified tau governs the latent")
    print("Gaussian rather than the log-irradiance, and no claim is made that the")
    print("copula reproduces a physical two-scale temporal spectrum. At this")
    print("configuration the measured gamma-gamma deviation is of the same order as")
    print("the lognormal rows' own estimator bias, so nothing larger than the")
    print("estimator error was detected; that is reported, not asserted as agreement.")
    print()

    print("-- 4. gamma-gamma parameter relations --------------------------------")
    print(f"{'sigma_R^2':>10} {'alpha':>12} {'beta':>12} {'SI from (6)':>14} "
          f"{'exp(sx2+sy2)-1':>16} {'rel diff':>10}")
    for sr2 in (0.05, 0.3, 1.0, 4.0, 16.0):
        alpha, beta = rytov_to_gamma_gamma(sr2)
        sx2 = 0.49 * sr2 / (1.0 + 1.11 * sr2 ** (6 / 5)) ** (7 / 6)
        sy2 = 0.51 * sr2 / (1.0 + 0.69 * sr2 ** (6 / 5)) ** (5 / 6)
        si6 = gamma_gamma_scintillation_index(alpha, beta)
        ident = float(np.expm1(sx2 + sy2))
        print(
            f"{sr2:10.4f} {alpha:12.6f} {beta:12.6f} {si6:14.9f} {ident:16.9f} "
            f"{abs(si6 - ident) / ident:10.3e}"
        )
    print()
    print(f"{'SI target':>10} {'alpha':>12} {'beta':>12} {'SI recovered':>14} {'rel err':>10}")
    for si in (0.1, 0.5, 0.8, 1.0, 1.2):
        alpha, beta = gamma_gamma_parameters_from_si(si)
        back = gamma_gamma_scintillation_index(alpha, beta)
        print(f"{si:10.4f} {alpha:12.6f} {beta:12.6f} {back:14.9f} {abs(back - si) / si:10.3e}")
    print()

    print("-- 5. gamma-gamma sample marginal ------------------------------------")
    print(f"{'SI target':>10} {'mean I':>12} {'SI sample':>12} {'rel err':>10}")
    for si in (0.3, 0.8, 1.1):
        cfg = ChannelConfig(si, TAU, FS, "gammagamma", "exp", seed=14)
        i = generate_irradiance(cfg, N)
        si_hat = float(np.var(i) / np.mean(i) ** 2)
        print(f"{si:10.4f} {float(i.mean()):12.6f} {si_hat:12.6f} {abs(si_hat - si) / si:10.6f}")
    print()
    print("=" * 78)


if __name__ == "__main__":
    main()
