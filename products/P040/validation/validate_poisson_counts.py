"""Validation: injected upset counts match the Poisson expectation.

Checks
------
1. The number of bits actually flipped in the parameter block equals the number
   of upsets the flux model drew, for every trial. This is the link between the
   rate model and the injector; without it the campaign could be reporting the
   degradation of a different number of upsets than it claims.
2. Sample mean and sample variance of the drawn counts against the Poisson
   prediction ``E[K] = Var[K] = mu``, each with its own sampling standard
   error:
       se(mean) = sqrt(mu / n)
       se(S2)   = sqrt((mu + 2 mu**2) / n)
   both derived in ``bitflipsim.flux.count_statistics``. Reported as z scores.
3. A chi-square goodness-of-fit test of the whole count histogram against the
   Poisson probability mass function, with the tail pooled so that every bin has
   an expectation of at least five.
4. Uniformity of the sampled bit positions over the 32 positions of a float32
   word, against the multinomial standard deviation.
5. The periodic-reload result ``E[live upsets] = lambda * T_s / 2`` against a
   Monte Carlo of the arrival-and-scrub process.

Campaign sizing: equation ``n = (s/e)**2`` from
``bitflipsim.campaign.trials_for_standard_error``. The counts here are cheap
(no inference), so n = 200000 is used for the count statistics, giving
se(mean) = sqrt(mu/200000). The degradation campaigns, which do run inference,
are sized in ``examples/degradation_vs_upset_rate.py`` and reported there.

Runtime: under 20 s on one core.
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import stats

from bitflipsim.flux import count_statistics, sample_upset_counts, upset_rate
from bitflipsim.injection import apply_upsets, net_flipped_bits, sample_upsets, upset_site_histogram
from bitflipsim.mitigation import expected_live_upsets

failures: list[str] = []


def report(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures.append(name)


N_PARAMETERS = 147
BITS = 32
POPULATION = N_PARAMETERS * BITS
FLUX = 1.0e3
CROSS_SECTION = 1.0e-14

rate = upset_rate(FLUX, CROSS_SECTION, POPULATION)
print("=" * 78)
print("Rate model under test")
print("=" * 78)
print(f"flux                      {rate.flux_per_cm2_s:.6e} particles cm^-2 s^-1 (illustrative)")
print(f"cross-section             {rate.cross_section_cm2_per_bit:.6e} cm^2 bit^-1 (illustrative)")
print(f"exposed bits              {rate.bit_count} ({N_PARAMETERS} float32 parameters)")
print(f"lambda = flux*sigma*N     {rate.rate_per_s:.9e} upsets s^-1")
print(f"                          {rate.rate_fit:.6e} FIT")
print(f"mean time between upsets  {rate.mean_time_between_upsets_s:.6e} s")
print()
print("Note: the flux and cross-section above are round illustrative numbers, not")
print("measured environment or part figures. The exposure times below are derived")
print("from them so that the expected counts are convenient; with a real")
print("cross-section the same equation gives different exposures.")

print()
print("=" * 78)
print("Check 1 - bits flipped equals upsets drawn, per trial")
print("=" * 78)
rng = np.random.default_rng(20261005)
block = np.linspace(-3.0, 3.0, N_PARAMETERS, dtype=np.float32)
trials = 500
mismatches = 0
total_drawn = 0
total_flipped = 0
counts = rng.poisson(lam=8.0, size=trials)
for count in counts:
    upsets = sample_upsets(N_PARAMETERS, BITS, int(count), rng)
    faulty = apply_upsets(block, upsets)
    flipped = net_flipped_bits(block, faulty)
    total_drawn += int(count)
    total_flipped += flipped
    if flipped != int(count):
        mismatches += 1
print(f"trials                    {trials}")
print(f"upsets drawn (total)      {total_drawn}")
print(f"bits flipped (total)      {total_flipped}")
print(f"trials with a mismatch    {mismatches}")
report("injected count equals drawn count", mismatches == 0 and total_drawn == total_flipped,
       f"{total_drawn} drawn, {total_flipped} flipped, {mismatches} mismatched trials")

print()
print("=" * 78)
print("Check 2 - sample mean and variance against the Poisson prediction")
print("=" * 78)
N_TRIALS = 200_000
print(f"trials per expectation    {N_TRIALS}")
print(f"{'mu':>8} {'exposure (s)':>16} {'mean':>12} {'se(mean)':>11} {'z':>7} "
      f"{'variance':>12} {'se(var)':>11} {'z':>7}")
worst_z = 0.0
for mu in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
    exposure = rate.exposure_for_expected_upsets(mu)
    sample = sample_upset_counts(rate, exposure, N_TRIALS, np.random.default_rng(int(mu * 1000)))
    stat = count_statistics(sample, mu)
    worst_z = max(worst_z, abs(stat.mean_z), abs(stat.variance_z))
    print(f"{mu:>8.2f} {exposure:>16.6e} {stat.sample_mean:>12.6f} "
          f"{stat.mean_standard_error:>11.6f} {stat.mean_z:>+7.2f} "
          f"{stat.sample_variance:>12.6f} {stat.variance_standard_error:>11.6f} "
          f"{stat.variance_z:>+7.2f}")
print()
one_tail = float(stats.norm.sf(4.0))
print("Gate: |z| < 4 on both statistics at every expectation. One tail of a")
print(f"standard normal beyond 4 sigma has probability {one_tail:.3e}, so over the")
print(f"12 statistics above the chance of a spurious failure is about "
      f"{1.0 - (1.0 - 2.0 * one_tail) ** 12:.3e}. The gate is deliberately wide; the")
print("seeds are fixed, so the numbers above are reproducible rather than lucky.")
report("Poisson mean and variance", worst_z < 4.0,
       f"worst |z| over 6 expectations x 2 statistics = {worst_z:.3f} (gate 4.0)")

print()
print("=" * 78)
print("Check 3 - chi-square goodness of fit of the count histogram")
print("=" * 78)
mu = 6.0
exposure = rate.exposure_for_expected_upsets(mu)
sample = sample_upset_counts(rate, exposure, N_TRIALS, np.random.default_rng(4242))
max_k = int(sample.max())
observed = np.bincount(sample, minlength=max_k + 1).astype(np.float64)
expected = stats.poisson.pmf(np.arange(max_k + 1), mu) * N_TRIALS
# Pool the tail so that every bin expects at least 5.
keep = expected >= 5.0
cut = int(np.argmin(keep)) if (~keep).any() else len(expected)
observed_pooled = np.append(observed[:cut], observed[cut:].sum())
expected_pooled = np.append(expected[:cut], N_TRIALS - expected[:cut].sum())
chi2 = float(((observed_pooled - expected_pooled) ** 2 / expected_pooled).sum())
dof = len(observed_pooled) - 1  # mu is the model's, not estimated from the sample
p_value = float(stats.chi2.sf(chi2, dof))
print(f"mu                        {mu}")
print(f"bins (tail pooled at k={cut}) {len(observed_pooled)}")
print(f"{'k':>4} {'observed':>10} {'expected':>12}")
for k in range(len(observed_pooled)):
    label = f">={cut}" if k == len(observed_pooled) - 1 else str(k)
    print(f"{label:>4} {observed_pooled[k]:>10.0f} {expected_pooled[k]:>12.2f}")
print(f"chi-square                {chi2:.6f} on {dof} degrees of freedom")
print(f"p-value                   {p_value:.6f}")
report("chi-square fit to the Poisson pmf", p_value > 0.01,
       f"chi2 = {chi2:.4f}, dof = {dof}, p = {p_value:.6f} (gate p > 0.01)")

print()
print("=" * 78)
print("Check 4 - uniformity of sampled bit positions")
print("=" * 78)
n_upsets = 320_000
upsets = sample_upsets(N_PARAMETERS, BITS, n_upsets, np.random.default_rng(777), allow_repeat=True)
histogram = upset_site_histogram(upsets, BITS)
expected_per_bit = n_upsets / BITS
sigma = np.sqrt(n_upsets * (1.0 / BITS) * (1.0 - 1.0 / BITS))
deviations = (histogram - expected_per_bit) / sigma
print(f"upsets drawn              {n_upsets} (with replacement, to sample positions freely)")
print(f"expected per bit position {expected_per_bit:.1f}")
print(f"multinomial sigma         {sigma:.3f}")
print(f"observed min / max        {histogram.min()} / {histogram.max()}")
print(f"worst |z| over 32 bins    {np.abs(deviations).max():.3f}")
report("bit-position uniformity", float(np.abs(deviations).max()) < 4.5,
       f"worst |z| = {np.abs(deviations).max():.3f} over 32 bins (gate 4.5)")

print()
print("=" * 78)
print("Check 5 - periodic reload: E[live upsets] = lambda * T_s / 2")
print("=" * 78)
print("Derivation: upsets arrive as a Poisson process of rate lambda and are")
print("cleared at each reload. Observing at a time tau uniform on [0, T_s) into")
print("the current interval, E[live | tau] = lambda tau, so E[live] = lambda T_s / 2.")
scrub_rng = np.random.default_rng(31337)
print(f"{'lambda (1/s)':>14} {'T_s (s)':>10} {'predicted':>12} {'simulated':>12} "
      f"{'se':>10} {'z':>7}")
worst_scrub_z = 0.0
for lam, interval in ((0.05, 20.0), (0.2, 10.0), (1.0, 4.0), (2.0, 3.0)):
    predicted = expected_live_upsets(lam, interval)
    n_obs = 200_000
    tau = scrub_rng.uniform(0.0, interval, size=n_obs)
    live = scrub_rng.poisson(lam=lam * tau)
    simulated = float(live.mean())
    standard_error = float(live.std(ddof=1) / np.sqrt(n_obs))
    z = (simulated - predicted) / standard_error if standard_error > 0 else 0.0
    worst_scrub_z = max(worst_scrub_z, abs(z))
    print(f"{lam:>14.4f} {interval:>10.2f} {predicted:>12.6f} {simulated:>12.6f} "
          f"{standard_error:>10.6f} {z:>+7.2f}")
report("expected live upsets under scrubbing", worst_scrub_z < 4.0,
       f"worst |z| over 4 configurations = {worst_scrub_z:.3f} (gate 4.0)")

print()
print("=" * 78)
if failures:
    print(f"RESULT: {len(failures)} FAILED check(s): {failures}")
    sys.exit(1)
print("RESULT: all checks PASSED")
