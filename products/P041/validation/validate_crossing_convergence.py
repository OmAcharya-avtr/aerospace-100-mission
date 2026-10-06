"""Level-crossing rate: which kernel converges, and does the analytic result hold?

This is the check that justifies having two correlation kernels. For the
Gauss-Markov (``exp``) kernel the continuous-time level-crossing rate of an
Ornstein-Uhlenbeck process is infinite, so the measured discrete rate must keep
rising as the sample rate rises; the exact sampled-process expression of
equation (14) must track it. For the Gaussian kernel the rate is finite and the
measured value must converge to Rice's (1945) formula, equation (12).

Also reports the convergence of the sample mean fade duration and level-crossing
rate to the exact discrete result as the record lengthens, which is the evidence
that the earlier short-record disagreement was sampling noise rather than a model
error.

Run: ``PYTHONPATH=../src python3 validate_crossing_convergence.py``
Runtime: about 60 s on one core.
"""

from __future__ import annotations

import numpy as np

from codedfade.channel import ChannelConfig, generate_amplitude
from codedfade.fade import (
    fade_statistics,
    lognormal_standard_level,
    markov_crossing_rate,
    markov_mean_fade_duration,
    rice_crossing_rate_gauss_kernel,
)

SI = 0.6
TAU = 2.0e-4
THRESHOLD = 0.6
#: Every row of sections 1 and 2 uses a record of this duration, so the number of
#: crossings stays comparable across sample rates and the convergence statement is
#: about the kernel rather than about how many crossings happened to be observed.
RECORD_S = 4.0
RATES = (1.0e5, 2.0e5, 5.0e5, 1.0e6)


def main() -> None:
    u = lognormal_standard_level(THRESHOLD, SI)
    print("=" * 78)
    print("LEVEL-CROSSING CONVERGENCE VALIDATION")
    print("=" * 78)
    print(f"marginal                 lognormal, SI = {SI}")
    print(f"correlation time         {TAU:.6e} s")
    print(f"amplitude threshold      {THRESHOLD}")
    print(f"standard level u         {u:.9f}")
    print()

    print(f"-- 1. exp kernel: the discrete rate must NOT converge ({RECORD_S:.0f} s records)")
    print(f"{'fs Hz':>12} {'Lc':>9} {'sample LCR':>13} {'eq(14) LCR':>13} {'rel diff':>10} "
          f"{'crossings':>10} {'1/sqrt(N)':>10}")
    for fs in RATES:
        cfg = ChannelConfig(SI, TAU, fs, "lognormal", "exp", seed=31)
        n = int(RECORD_S * fs)
        a = generate_amplitude(cfg, n)
        st = fade_statistics(a, THRESHOLD, fs)
        analytic = markov_crossing_rate(u, TAU, fs)
        print(
            f"{fs:12.3e} {TAU * fs:9.1f} {st.level_crossing_rate_hz:13.3f} "
            f"{analytic:13.3f} {abs(st.level_crossing_rate_hz - analytic) / analytic:10.6f} "
            f"{st.down_crossings:10d} {1 / np.sqrt(st.down_crossings):10.6f}"
        )
    print()
    print("Both columns rise with fs by a factor of about sqrt(fs). That is the")
    print("Ornstein-Uhlenbeck process having no finite mean-square derivative, not a")
    print("defect: the continuous-time crossing rate does not exist.")
    print()

    print(f"-- 2. gauss kernel: the rate MUST converge to Rice (1945) ({RECORD_S:.0f} s records)")
    rice = rice_crossing_rate_gauss_kernel(u, TAU)
    print(f"Rice eq(12) continuous-time rate = {rice:.6f} s^-1")
    print(f"{'fs Hz':>12} {'Lc':>9} {'sample LCR':>13} {'rel diff vs Rice':>18} "
          f"{'crossings':>10} {'in SE':>8}")
    for fs in RATES:
        cfg = ChannelConfig(SI, TAU, fs, "lognormal", "gauss", seed=32)
        n = int(RECORD_S * fs)
        a = generate_amplitude(cfg, n)
        st = fade_statistics(a, THRESHOLD, fs)
        rel = (st.level_crossing_rate_hz - rice) / rice
        se = 1.0 / np.sqrt(max(st.down_crossings, 1))
        print(
            f"{fs:12.3e} {TAU * fs:9.1f} {st.level_crossing_rate_hz:13.3f} "
            f"{rel:18.6f} {st.down_crossings:10d} {rel / se:8.2f}"
        )
    print()
    print("The 'in SE' column is the deviation in units of the Poisson standard error")
    print("of the crossing count, 1/sqrt(N). Rows within about 3 SE are consistent")
    print("with convergence; the discrete rate is biased slightly low at small Lc")
    print("because sampling misses short excursions, and that bias shrinks with Lc.")
    print("All four rows share one seed, so their offsets are correlated; the claim")
    print("being made is that the offset does not GROW with fs, which section 1 shows")
    print("it does for the exp kernel.")

    print("-- 3. convergence of the sample statistics with record length --------")
    fs = 1.0e6
    cfg = ChannelConfig(SI, TAU, fs, "lognormal", "exp", seed=33)
    lcr_exact = markov_crossing_rate(u, TAU, fs)
    mfd_exact = markov_mean_fade_duration(u, TAU, fs)
    print(f"exact eq(14) LCR = {lcr_exact:.6f} s^-1   exact eq(15) MFD = {mfd_exact:.9e} s")
    print(f"{'samples':>10} {'fades':>8} {'sample LCR':>13} {'LCR rel':>10} "
          f"{'sample MFD s':>14} {'MFD rel':>10}")
    big = generate_amplitude(cfg, 4_000_000)
    for n in (100_000, 400_000, 1_000_000, 4_000_000):
        st = fade_statistics(big[:n], THRESHOLD, fs)
        print(
            f"{n:10d} {st.complete_fades:8d} {st.level_crossing_rate_hz:13.3f} "
            f"{(st.level_crossing_rate_hz - lcr_exact) / lcr_exact:10.6f} "
            f"{st.mean_fade_duration_s:14.6e} "
            f"{(st.mean_fade_duration_s - mfd_exact) / mfd_exact:10.6f}"
        )
    print()
    print("=" * 78)


if __name__ == "__main__":
    main()
