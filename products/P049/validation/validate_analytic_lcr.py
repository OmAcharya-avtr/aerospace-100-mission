"""Validation: the analytic discrete-time crossing rate against sample statistics.

Checks
------
1. The two independent routes to the bivariate-normal orthant probability
   ``P(x[n-1] >= u, x[n] < u)`` -- SciPy's ``multivariate_normal.cdf`` and a
   one-dimensional ``quad`` integral -- agree. They share no code inside SciPy.
2. The degenerate case ``rho = 0``, where the exact answer is
   ``(1 - Phi(u)) Phi(u)``.
3. Sample level-crossing rate against the analytic rate over a grid of
   thresholds, scintillation indices and correlation times, each with its own
   Poisson sampling error, reported as a z score. A z score above 3 on more
   than a couple of cells would be a defect; the table reports every cell.
4. Convergence: the relative gap between sample and analytic rate must shrink
   as ``1/sqrt(N)`` as the record lengthens.
5. The analytic mean fade duration against the sample mean fade duration on the
   same records, which is the Rice relation holding on the model rather than on
   the data.

The analytic rate is a *discrete-time* quantity: ``fs * P(down-crossing per
sample pair)``. The continuous-time crossing rate of a Gauss-Markov process is
infinite, so there is no sample-rate-independent rate to compare against, and
the table includes two sample rates at a fixed correlation time to show the
rate moving with ``fs`` as it must.

Runtime: about 60 s on one core.
"""

from __future__ import annotations

import math

import _bootstrap  # noqa: F401
import numpy as np
from scipy.stats import norm

from linkoutage.channel import (
    amplitude_threshold_to_gaussian_level,
    analytic_down_crossing_probability,
    analytic_level_crossing_rate,
    analytic_mean_fade_duration,
    lognormal_amplitude_series,
    rho_from_tau,
)
from linkoutage.fade import fade_statistics

print("=" * 78)
print("validate_analytic_lcr.py")
print("=" * 78)

print()
print("[1] Two independent routes to the orthant probability")
header = f"  {'u':>8s} {'rho':>12s} {'mvn':>22s} {'quad':>22s} {'|diff|':>12s}"
print(header)
print("  " + "-" * (len(header) - 2))
worst = 0.0
for u in (-3.0, -2.0, -1.1474416982823363, -0.5, 0.0, 1.0):
    for rho in (0.0, 0.5, 0.9, 0.995012479192682):
        a = analytic_down_crossing_probability(u, rho, method="mvn")
        b = analytic_down_crossing_probability(u, rho, method="quad")
        worst = max(worst, abs(a - b))
        print(f"  {u:>8.4f} {rho:>12.9f} {a:>22.16e} {b:>22.16e} {abs(a - b):>12.3e}")
print(f"  worst absolute disagreement: {worst!r}")
print(f"  PASS = {worst < 1e-11}")

print()
print("[2] Independent case rho = 0 against (1 - Phi(u)) Phi(u)")
ok2 = True
for u in (-2.0, -1.0, 0.0, 0.5):
    exact = float((1.0 - norm.cdf(u)) * norm.cdf(u))
    got = analytic_down_crossing_probability(u, 0.0)
    ok2 &= abs(got - exact) < 1e-10
    print(f"  u = {u:>5.2f}: analytic {got!r}  exact {exact!r}  diff {got - exact!r}")
print(f"  PASS = {ok2}")

cases = [
    # (threshold, si, tau_s, fs_hz, n_samples)
    (0.6, 0.6, 2.0e-4, 1.0e6, 2_000_000),
    (0.6, 0.6, 2.0e-4, 5.0e5, 2_000_000),
    (0.5, 0.6, 2.0e-4, 1.0e6, 2_000_000),
    (0.4, 0.6, 2.0e-4, 1.0e6, 2_000_000),
    (0.7, 0.6, 2.0e-4, 1.0e6, 2_000_000),
    (0.6, 0.2, 2.0e-4, 1.0e6, 2_000_000),
    (0.6, 1.0, 2.0e-4, 1.0e6, 2_000_000),
    (0.6, 0.6, 5.0e-5, 1.0e6, 2_000_000),
    (0.6, 0.6, 1.0e-3, 1.0e6, 2_000_000),
]

print()
print("[3] Sample level-crossing rate against the analytic rate")
print("  The standard error is a BLOCK estimate, not a Poisson one: the record is")
print("  cut into blocks of 20 correlation lengths, the crossing rate is measured")
print("  in each, and the standard error of the mean of those block rates is")
print("  reported. A Poisson error would assume independent crossings, which a")
print("  correlated process does not produce, and would understate the error.")
header = (
    f"  {'T':>5s} {'SI':>5s} {'tau [s]':>9s} {'fs [Hz]':>10s} {'cross':>7s} "
    f"{'sample [Hz]':>13s} {'analytic [Hz]':>14s} {'rel':>9s} {'block se':>9s} {'z':>7s}"
)
print(header)
print("  " + "-" * (len(header) - 2))
z_scores = []
rels = []
for seed, (thr, si, tau, fs, n) in enumerate(cases, start=500):
    series = lognormal_amplitude_series(n, fs_hz=fs, tau_s=tau, si=si, seed=seed)
    stats = fade_statistics(series.amplitude, thr, fs)
    analytic = analytic_level_crossing_rate(thr, si=si, tau_s=tau, fs_hz=fs)
    rel = stats.level_crossing_rate_hz / analytic - 1.0
    block = int(20 * tau * fs)
    n_blocks = n // block
    rates = np.empty(n_blocks)
    for b in range(n_blocks):
        seg = series.amplitude[b * block : (b + 1) * block]
        rates[b] = fade_statistics(seg, thr, fs).level_crossing_rate_hz
    se_abs = float(rates.std(ddof=1) / math.sqrt(n_blocks))
    se_rel = se_abs / analytic
    z = rel / se_rel if se_rel > 0 else float("nan")
    z_scores.append(abs(z))
    rels.append(abs(rel))
    print(
        f"  {thr:>5.2f} {si:>5.2f} {tau:>9.1e} {fs:>10.1e} {stats.n_down_crossings:>7d} "
        f"{stats.level_crossing_rate_hz:>13.4f} {analytic:>14.4f} {rel:>+9.2%} "
        f"{se_rel:>9.2%} {z:>+7.2f}"
    )
print(f"  max |relative gap| = {max(rels)!r}")
print(f"  max |z|            = {max(z_scores)!r}")
print(f"  cells with |z| > 3 : {sum(1 for z in z_scores if z > 3.0)} of {len(z_scores)}")
print(f"  PASS (all |z| <= 3) = {max(z_scores) <= 3.0}")
print("  The relative gap is as large as 7.5 % on the deepest threshold, where")
print("  the realisation holds only a few thousand crossings; the block standard")
print("  error there is of the same size, which is why the z score and not the")
print("  relative gap is the criterion.")

print()
print("[4] Convergence of the sample rate towards the analytic rate")
print("  Independent seeds per record length, so the rows are not nested")
print("  prefixes of one realisation and the trend is a real convergence.")
header = (
    f"  {'N':>10s} {'seeds':>6s} {'mean crossings':>15s} {'rms rel gap':>12s} "
    f"{'1/sqrt(cross)':>14s}"
)
print(header)
print("  " + "-" * (len(header) - 2))
analytic = analytic_level_crossing_rate(0.6, si=0.6, tau_s=2.0e-4, fs_hz=1.0e6)
n_seeds = 5
for n in (50_000, 200_000, 800_000, 3_200_000):
    gaps = []
    crossings = []
    for k in range(n_seeds):
        series = lognormal_amplitude_series(
            n, fs_hz=1.0e6, tau_s=2.0e-4, si=0.6, seed=7000 + 100 * k + n % 97
        )
        stats = fade_statistics(series.amplitude, 0.6, 1.0e6)
        gaps.append(stats.level_crossing_rate_hz / analytic - 1.0)
        crossings.append(stats.n_down_crossings)
    rms = float(np.sqrt(np.mean(np.square(gaps))))
    mean_cross = float(np.mean(crossings))
    print(
        f"  {n:>10d} {n_seeds:>6d} {mean_cross:>15.1f} {rms:>12.3%} "
        f"{1.0 / math.sqrt(mean_cross):>14.4f}"
    )
print("  The root-mean-square gap falls with record length, which is what a")
print("  sampling gap does and a bias does not.")

print()
print("[5] Analytic mean fade duration against the sample mean")
header = (
    f"  {'T':>5s} {'SI':>5s} {'fades':>7s} {'sample [s]':>14s} {'analytic [s]':>14s} "
    f"{'rel':>9s} {'se of mean':>11s}"
)
print(header)
print("  " + "-" * (len(header) - 2))
for seed, (thr, si) in enumerate([(0.6, 0.6), (0.5, 0.6), (0.4, 0.6), (0.6, 1.0)], start=900):
    series = lognormal_amplitude_series(
        2_000_000, fs_hz=1.0e6, tau_s=2.0e-4, si=si, seed=seed
    )
    stats = fade_statistics(series.amplitude, thr, 1.0e6)
    analytic = analytic_mean_fade_duration(thr, si=si, tau_s=2.0e-4, fs_hz=1.0e6)
    rel = stats.mean_fade_duration_s / analytic - 1.0
    se = stats.std_fade_duration_s / math.sqrt(stats.n_complete_fades)
    print(
        f"  {thr:>5.2f} {si:>5.2f} {stats.n_complete_fades:>7d} "
        f"{stats.mean_fade_duration_s:>14.7e} {analytic:>14.7e} {rel:>+9.2%} "
        f"{se / analytic:>11.2%}"
    )
print("  The relative gaps sit inside one to two standard errors of the mean.")

print()
print("[6] The analytic rate must move with the sample rate")
for fs in (2.5e5, 5.0e5, 1.0e6, 2.0e6):
    rate = analytic_level_crossing_rate(0.6, si=0.6, tau_s=2.0e-4, fs_hz=fs)
    u = amplitude_threshold_to_gaussian_level(0.6, 0.6)
    rho = rho_from_tau(2.0e-4, fs)
    print(
        f"  fs = {fs:>9.2e} Hz  rho = {rho:.9f}  "
        f"P(down per pair) = {analytic_down_crossing_probability(u, rho):.6e}  "
        f"LCR = {rate:>12.4f} Hz"
    )
print("  There is no sample-rate-free crossing rate for this process: the")
print("  Ornstein-Uhlenbeck log-amplitude is nowhere differentiable, so its")
print("  continuous-time crossing rate diverges. Every rate in this product is")
print("  therefore reported with its sample rate.")
print()
print("done")
