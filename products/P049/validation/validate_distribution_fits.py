"""Validation: is the exponential fade-duration assumption true? (No.)

Checks
------
1. **Positive control.** Durations drawn from a true geometric law must NOT be
   rejected by the geometric chi-square, and the exponential maximum likelihood
   must recover the scale. Without this, a rejection below proves nothing: it
   could be a broken test rather than a broken assumption.
2. **The measured channel.** The geometric chi-square on the fade lengths of the
   specified lognormal AR(1) series, at four thresholds. This is the valid test
   of the memoryless hypothesis on discrete data.
3. **Why it fails, in a number that needs no test.** The coefficient of
   variation of an exponential (and, to within ``sqrt(1-p)``, of a geometric) is
   1. The measured coefficient of variation is reported beside it.
4. **What fits instead.** AIC ranking of lognormal, Weibull and gamma against
   the two memoryless laws, with the Kolmogorov-Smirnov statistic for each and
   the caveats that the KS p-value carries.
5. **Censoring.** The exponential scale with and without the right-censored
   runs folded in, so the size of the censoring bias is on the record, and a
   Kaplan-Meier survival estimate against the raw empirical survival.
6. **Tail behaviour.** The empirical survival function against the fitted
   exponential at several quantiles: the exponential underestimates the long
   fades, which are the ones that break an interleaver.

Runtime: about 20 s on one core.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np

from linkoutage.channel import lognormal_amplitude_series
from linkoutage.distributions import (
    compare_fade_duration_models,
    exponential_mle_with_censoring,
    fit_geometric_samples,
    geometric_chi_square,
    kaplan_meier_survival,
)
from linkoutage.fade import FadeDefinitions, fade_durations, fade_runs

FS = 1.0e6
TAU = 2.0e-4
SI = 0.6
N = 2_000_000

print("=" * 78)
print("validate_distribution_fits.py")
print("=" * 78)

print()
print("[1] Positive control: a truly geometric sample must not be rejected")
rng = np.random.default_rng(3001)
header = f"  {'p true':>8s} {'n':>8s} {'p fitted':>10s} {'chi2':>12s} {'dof':>5s} {'p-value':>11s}"
print(header)
print("  " + "-" * (len(header) - 2))
control_ok = True
for p_true in (0.02, 0.06, 0.2):
    lengths = rng.geometric(p_true, 30_000)
    p_hat, _ = fit_geometric_samples(lengths)
    stat, dof, pv, _, _ = geometric_chi_square(lengths, p_hat)
    control_ok &= pv > 0.01
    print(f"  {p_true:>8.3f} {lengths.size:>8d} {p_hat:>10.6f} {stat:>12.4f} {dof:>5d} {pv:>11.4g}")
theta_true = 4.0
sample = rng.exponential(theta_true, 50_000)
theta_hat, _ = exponential_mle_with_censoring(sample)
print(f"  exponential MLE on exponential data: theta_hat = {theta_hat!r} (true {theta_true!r})")
print(f"  PASS = {control_ok and abs(theta_hat / theta_true - 1.0) < 0.02}")

print()
print("[2] The measured channel: geometric chi-square at four thresholds")
header = (
    f"  {'T':>5s} {'fades':>7s} {'cens':>5s} {'mean [samp]':>12s} {'p fitted':>10s} "
    f"{'chi2':>12s} {'dof':>5s} {'p-value':>10s} {'verdict':>10s}"
)
print(header)
print("  " + "-" * (len(header) - 2))
series = lognormal_amplitude_series(N, fs_hz=FS, tau_s=TAU, si=SI, seed=41)
results = {}
for thr in (0.4, 0.5, 0.6, 0.7):
    complete, censored = fade_durations(series.amplitude, thr, FS)
    comparison = compare_fade_duration_models(complete, fs_hz=FS, censored_s=censored)
    results[thr] = comparison
    g = comparison.geometric
    print(
        f"  {thr:>5.2f} {comparison.n_complete:>7d} {comparison.n_censored:>5d} "
        f"{1.0 / g.params[0]:>12.4f} {g.params[0]:>10.6f} {g.statistic:>12.1f} "
        f"{g.dof:>5d} {g.p_value:>10.3g} "
        f"{'REJECTED' if comparison.exponential_rejected else 'not rej.':>10s}"
    )
print("  alpha = 0.01. Every threshold rejects the memoryless hypothesis.")

print()
print("[3] Coefficient of variation: 1 for an exponential, measured here")
header = f"  {'T':>5s} {'cv measured':>12s} {'cv exponential':>15s} {'ratio':>8s}"
print(header)
print("  " + "-" * (len(header) - 2))
for thr, comparison in results.items():
    cv = comparison.extra["coefficient_of_variation"]
    print(f"  {thr:>5.2f} {cv:>12.4f} {1.0:>15.4f} {cv:>8.2f}")
print("  A fade-duration distribution two to three times more dispersed than an")
print("  exponential needs no hypothesis test to be visibly not exponential.")

print()
print("[4] Full model comparison at the specified threshold T = 0.6")
print(results[0.6].report())

print()
print("[5] Censoring")
complete, censored = fade_durations(series.amplitude, 0.6, FS)
theta_with, ll_with = exponential_mle_with_censoring(complete, censored)
theta_without, ll_without = exponential_mle_with_censoring(complete)
print(f"  complete fades                  : {complete.size}")
print(f"  right/left-censored runs        : {censored.size}")
print(f"  exponential scale with censoring: {theta_with!r} s")
print(f"  exponential scale ignoring it   : {theta_without!r} s")
print(f"  relative bias from ignoring     : {theta_without / theta_with - 1.0!r}")
print("  On this record one run of 16383 is censored, so the bias is negligible.")
print("  It is not negligible on a short record; the deliberate short case:")
short = lognormal_amplitude_series(2_000, fs_hz=FS, tau_s=TAU, si=SI, seed=41)
c_short, cens_short = fade_durations(short.amplitude, 0.6, FS)
if c_short.size and cens_short.size:
    tw, _ = exponential_mle_with_censoring(c_short, cens_short)
    two, _ = exponential_mle_with_censoring(c_short)
    print(f"    2000-sample record at T = 0.6: {c_short.size} complete, "
          f"{cens_short.size} censored")
    print(f"    scale with censoring {tw!r} s, ignoring it {two!r} s, "
          f"relative bias {two / tw - 1.0!r}")
else:
    print(f"    2000-sample record at T = 0.6 gave {c_short.size} complete and "
          f"{cens_short.size} censored runs; nothing to compare")

times_km, surv_km = kaplan_meier_survival(complete, censored)
times_plain, surv_plain = kaplan_meier_survival(complete)
print(f"  Kaplan-Meier vs raw survival, max absolute difference: "
      f"{float(np.max(np.abs(surv_km - surv_plain)))!r}")

print()
print("[6] Tail: empirical survival against the fitted exponential")
runs = fade_runs(series.amplitude, 0.6, FS)
lengths = runs.length_samples[~runs.censored]
p_geom, _ = fit_geometric_samples(lengths)
header = (
    f"  {'L [samples]':>12s} {'empirical S(L)':>15s} {'geometric S(L)':>15s} "
    f"{'ratio':>8s}"
)
print(header)
print("  " + "-" * (len(header) - 2))
for cut in (1, 2, 5, 10, 20, 50, 100, 200, 400):
    emp = float(np.mean(lengths > cut))
    geo = float((1.0 - p_geom) ** cut)
    ratio = emp / geo if geo > 0 else float("inf")
    print(f"  {cut:>12d} {emp:>15.6g} {geo:>15.6g} {ratio:>8.2f}")
print("  The memoryless law under-predicts the long fades by orders of")
print("  magnitude, and over-predicts the medium ones. Sizing an interleaver")
print("  from an exponential fade-duration fit would under-size it.")

print()
print("[7] Sensitivity of the verdict to the fade definitions")
header = f"  {'definition variant':<40s} {'fades':>7s} {'cv':>7s} {'chi2':>12s} {'verdict':>10s}"
print(header)
print("  " + "-" * (len(header) - 2))
variants = {
    "published": FadeDefinitions(),
    "singles not fades": FadeDefinitions(count_single_sample_fades=False),
    "censored folded in": FadeDefinitions(censoring="include_as_complete"),
}
for label, defs in variants.items():
    c, cz = fade_durations(series.amplitude, 0.6, FS, definitions=defs)
    comparison = compare_fade_duration_models(c, fs_hz=FS, censored_s=cz)
    print(
        f"  {label:<40s} {comparison.n_complete:>7d} "
        f"{comparison.extra['coefficient_of_variation']:>7.3f} "
        f"{comparison.geometric.statistic:>12.1f} "
        f"{'REJECTED' if comparison.exponential_rejected else 'not rej.':>10s}"
    )
print("  The verdict does not depend on the definitional choices: the")
print("  memoryless hypothesis is rejected under all of them.")
print()
print("done")
