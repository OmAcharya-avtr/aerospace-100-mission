"""Cross-check X1 (binding): mean fade duration and level-crossing rate as sample statistics.

Batch 05 specification, cross-check X1: P041 CodedFade and P049 LinkOutage must
agree, to within 2 % relative difference on both quantities, on the mean fade
duration and the level-crossing rate computed from an **identical seeded
filtered-Gaussian amplitude series** with the same threshold and the same sample
count. Both are defined as sample statistics of that series; neither is compared
to any wall-clock measurement.

The configuration is printed in full alongside the numbers so that P049 can
reproduce the series exactly. The generator is specified here in words as well as
in the configuration block, because a reproduction has to match the construction
and not just the parameter values:

  1. ``g[0] = w[0]``, and ``g[i] = rho*g[i-1] + sqrt(1-rho^2)*w[i]`` for
     ``i >= 1``, with ``rho = exp(-1/(fs*tau))`` and ``w`` the first ``N`` draws
     of ``numpy.random.default_rng(seed).standard_normal(N)`` in one call.
  2. ``sigma_lnI^2 = ln(1 + SI)``;
     ``I[i] = exp(-sigma_lnI^2/2 + sigma_lnI*g[i])``.
  3. ``a[i] = sqrt(I[i])``.

Definitions, repeated here verbatim from ``codedfade.fade``:

  * ``b[i] = (a[i] < a_th)``.
  * A down-crossing is an index ``i >= 1`` with ``b[i-1] == False`` and
    ``b[i] == True``. ``LCR = (down-crossings) / (N / fs)`` in s^-1.
  * A fade is a maximal run of ``b == True``. A fade is complete if it is both
    preceded and followed by at least one sample with ``b == False``. Runs
    touching either end of the record are censored and excluded.
  * ``MFD = mean(complete run length in samples) / fs`` in s. A one-sample run has
    duration ``1/fs``.

Run: ``PYTHONPATH=../src python3 validate_cross_check_x1.py``
Runtime: about 10 s on one core.
"""

from __future__ import annotations

import numpy as np

from codedfade.channel import ChannelConfig, generate_amplitude
from codedfade.fade import (
    fade_statistics,
    lognormal_standard_level,
    markov_crossing_rate,
    markov_mean_fade_duration,
)

# ---- X1 CONFIGURATION, FROZEN ------------------------------------------------
X1 = {
    "distribution": "lognormal",
    "kernel": "exp (Gauss-Markov / AR(1), filtered Gaussian)",
    "scintillation_index": 0.6,
    "correlation_time_s": 2.0e-4,
    "sample_rate_hz": 1.0e6,
    "sample_count": 2_000_000,
    "threshold_amplitude": 0.6,
    "seed": 41,
    "rng": "numpy.random.default_rng(seed).standard_normal(sample_count)",
    "normalisation": "E[I] = 1, a = sqrt(I)",
}
# ------------------------------------------------------------------------------


def main() -> None:
    cfg = ChannelConfig(
        scintillation_index=X1["scintillation_index"],
        correlation_time_s=X1["correlation_time_s"],
        sample_rate_hz=X1["sample_rate_hz"],
        marginal="lognormal",
        kernel="exp",
        seed=X1["seed"],
    )
    n = int(X1["sample_count"])
    a_th = float(X1["threshold_amplitude"])

    print("=" * 78)
    print("CROSS-CHECK X1  (P041 CodedFade <-> P049 LinkOutage)")
    print("=" * 78)
    print("CONFIGURATION")
    for key, value in X1.items():
        print(f"  {key:24s} {value}")
    print(f"  rho = exp(-1/(fs*tau))   "
          f"{np.exp(-1.0 / (cfg.sample_rate_hz * cfg.correlation_time_s)):.15f}")
    print(f"  sigma_lnI^2 = ln(1+SI)   {cfg.log_irradiance_variance:.15f}")
    print(f"  Lc = tau*fs              {cfg.samples_per_correlation_time:.6f} samples")
    print()

    a = generate_amplitude(cfg, n)
    st = fade_statistics(a, a_th, cfg.sample_rate_hz)

    print("SERIES FINGERPRINT (for an exact reproduction check)")
    print(f"  amplitude[0:5]           {np.array2string(a[:5], precision=15)}")
    print(f"  amplitude[-5:]           {np.array2string(a[-5:], precision=15)}")
    print(f"  mean irradiance          {float(np.mean(a * a)):.15f}")
    print(f"  sample scintillation idx "
          f"{float(np.var(a * a) / np.mean(a * a) ** 2):.15f}")
    print()

    print("X1 QUANTITIES (sample statistics of the series above)")
    print(f"  sample_count             {st.n_samples}")
    print(f"  record duration          {st.n_samples / cfg.sample_rate_hz:.9f} s")
    print(f"  threshold_amplitude      {st.threshold:.6f}")
    print(f"  outage_fraction          {st.outage_fraction:.15f}")
    print(f"  down_crossings           {st.down_crossings}")
    print(f"  complete_fades           {st.complete_fades}")
    print(f"  censored_fades           {st.censored_fades}")
    print()
    print(f"  LEVEL_CROSSING_RATE_HZ   {st.level_crossing_rate_hz:.15f}")
    print(f"  MEAN_FADE_DURATION_S     {st.mean_fade_duration_s:.15e}")
    print()
    print("  (the two lines above are the X1 comparands; tolerance is 2 % relative")
    print("   difference on each, per the batch 05 specification)")
    print()

    u = lognormal_standard_level(a_th, cfg.scintillation_index)
    lcr_exact = markov_crossing_rate(u, cfg.correlation_time_s, cfg.sample_rate_hz)
    mfd_exact = markov_mean_fade_duration(u, cfg.correlation_time_s, cfg.sample_rate_hz)
    print("CONTEXT: the exact sampled-Gauss-Markov analytic result, equations (14)-(15)")
    print(f"  standard level u         {u:.15f}")
    print(f"  analytic LCR             {lcr_exact:.9f} s^-1   "
          f"(sample / analytic - 1 = {st.level_crossing_rate_hz / lcr_exact - 1:+.6f})")
    print(f"  analytic MFD             {mfd_exact:.9e} s     "
          f"(sample / analytic - 1 = {st.mean_fade_duration_s / mfd_exact - 1:+.6f})")
    print()
    print("  This block is context only. X1 compares the two sample statistics")
    print("  between products; it does not compare either to the analytic value, and")
    print("  no quantity here is a wall-clock measurement.")
    print()
    print("ADDITIONAL SAMPLE STATISTICS (not part of X1, supplied for diagnosis)")
    print(f"  median fade duration     {st.median_fade_duration_s:.9e} s")
    print(f"  max fade duration        {st.max_fade_duration_s:.9e} s")
    print(f"  mean fade duration       {st.mean_fade_duration_s * cfg.sample_rate_hz:.9f}"
          f" samples")
    quantiles = np.quantile(st.durations_s * cfg.sample_rate_hz, [0.5, 0.9, 0.99])
    print(f"  duration p50/p90/p99     {quantiles[0]:.1f} / {quantiles[1]:.1f} / "
          f"{quantiles[2]:.1f} samples")
    print("=" * 78)


if __name__ == "__main__":
    main()
