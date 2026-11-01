# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-11-01

First release. Status: TESTING. Validation level 2, research grade.

### Added

- `transfergap.discrepancy`: the paired sim-to-field record, equation (1), and
  its validation. The declared range `R` is an argument with no default, from
  the instrument and never fitted, because a distribution-free bound computed
  with an `R` below the true support is not conservative, it is wrong.
  `clip_fraction()` reports how much of a sample the clip touched so a
  censored sample can be recognised instead of silently absorbed.
- `transfergap.bounds`: **baseline 1**, distribution-free. Hoeffding (1963)
  equation (3) and its exact closed-form inversion (10); the Maurer–Pontil
  (2009) empirical Bernstein bound (5) and its inversion (11), closed form in
  the `n−1 ≈ n` approximation and then corrected against the true forward
  expression; Wilks' (1941) nonparametric one-sided tolerance limit (6) with
  the exact closed-form requirement (7); and the exact normal-theory tolerance
  factor (9) from the non-central t, shipped as the parametric comparison it
  is. Every bound carries `target`, `confidence`, `proportion` and
  `distribution_free` in its result, because a bound printed without its
  standing is the thing this package exists to stop.
- `transfergap.resample`: **baseline 2**, tighter under exchangeability. Split
  conformal (12)–(13), with the quantile taken as an order statistic by
  integer index and never through `numpy.quantile`; the percentile and basic
  bootstrap (14)–(15), both marked not distribution-free.
- `transfergap.exchangeability`: the assumption that pays for baseline 2,
  **tested** rather than assumed. A permutation test against two named
  alternatives — Spearman's rank correlation with the index (16) and a
  half-split mean difference (17) — with the exactly valid p-value of (18)
  including the Phipson–Smyth `+1`. The verdict string says `not rejected`
  and never `exchangeable`.
- `transfergap.plan`: the sample-count inversions as a plan, with each row's
  `basis` in its own column — `exact`, `plug-in` or `existence` — so a
  guarantee and a prediction are never printed as the same kind of number.
- `transfergap.learned`: a gradient-boosted conditional quantile model with
  the pinball loss (23), a three-quantile uncertainty output with a reported
  crossing mask, conformalised quantile regression as the coverage repair, and
  the plug-in conditional-mean bound (29) that exists to be caught
  under-covering.
- `transfergap.scoring`: the coverage harness. Achieved coverage is the
  verdict (24)–(26); width is a separate column; the standard error of a
  prediction-target rate includes the held-out pool's own sampling error (27).
- `transfergap.synthetic`: equations (19)–(22), two campaign shapes — one
  exchangeable, one deliberately not — and the population truth with its
  Monte Carlo standard error.
- `transfergap.report`: plain-text rendering that puts the assumption before
  the number.
- A CLI with `bounds`, `plan`, `exchangeability`, `learned` and `truth`.
- Ten validation scripts with their raw output, six example figures, and 188
  tests including Hypothesis property tests and four pinned regressions.

### Measured and reported as found

- **The learned conditional quantile under-covers at every field sample size,
  and is the tightest limit at every one of them.** Over 80 independent
  campaigns: coverage 0.8761, 0.8754, 0.8769 and 0.8843 against a declared
  0.95 at n = 15, 25, 40 and 60, at widths of 10.233, 9.799, 9.539 and 9.433
  kN.m against 13.5 to 18.3 for the limits that hold. Its per-campaign
  tolerance rate falls to 0.0500 at n = 60. This is the result the
  specification stated in advance; it is confirmed, and nothing was retuned,
  softened or removed to change it.
- **The plug-in learned mean bound of equation (29) covers the true mean 57.5
  per cent of the time at n = 15** against a declared 0.95, while being 0.16
  kN.m above the truth where Hoeffding's bound is 12.59 above it.
- **Hoeffding's bound achieves coverage 1.0000 at every field sample size**
  over 300 campaigns, and its sample-count inversion hits its target slack on
  600 of 600 trials, identically to every printed digit, because equation (3)
  does not read the data.
- **The empirical Bernstein inversion hits its target 51.67 per cent of the
  time** at a 2 kN.m target with the TRUE population variance. The inversion
  is a median prediction, not a guarantee, and the bound itself still holds in
  every case — those are different claims and they are reported separately.
- **The older inequality wins at the sample sizes that matter.** The
  Hoeffding / empirical Bernstein crossover is at **n = 82**; at n = 15, 25, 40
  and 60 the ratio of slacks is 2.1592, 1.6788, 1.3542 and 1.1370 in
  Hoeffding's favour.
- **Neither prediction limit exists at the smallest sample sizes.** Wilks'
  limit needs n ≥ 59 and split conformal needs n ≥ 19 at `p = 1−δ = 0.95`.
  Both raise with the required `n` in the message rather than returning the
  sample maximum under a name it has not earned.
- **The exchangeability test has power 0.4300 at n = 15** against a swept Mach
  card, rising to 0.9967 at n = 120, with a size never above 0.0767 against a
  declared α of 0.05. A fifteen-point campaign cannot establish the assumption
  its tighter bounds need.
- **The default CLI invocation on a non-exchangeable campaign does not
  reject.** `exchangeability --n 40 --campaign swept` gives p = 0.3200 and
  0.1250 at a measured power of 0.8000 and 0.8867, and the validation output
  says so, because a build that tested one draw had a one-in-five chance of
  concluding the test does not work.
- **Both bootstrap limits and the normal-theory tolerance limit measurably
  under-cover** at most sample sizes measured. None is distribution-free and
  this is the expected result.
- **Split conformal is marginally valid and the measurement shows what that
  costs:** coverage at the declared 0.95 with a tolerance rate near 0.53 to
  0.64, meaning about half of individual campaigns produce a limit that
  under-covers. Wilks' limit is training-conditionally valid and its tolerance
  rate reaches 0.9750.
- **The conformal repair works and needs more data than the model does.**
  Coverage recovers to 0.9571 at n = 40 and 0.9721 at n = 60, width returns to
  14.6 and 14.4 kN.m, and the repair does not exist below n = 40 at a
  half-and-half split while the raw model returns a number from 15 points.
- **Hoeffding's exponential bound was checked against the exact binomial tail
  over 72 grid points with 0 violations** and a worst exact/bound ratio of
  0.460412: neither violated nor vacuous. The comparison is in log space with
  a slack following the conditioning of the bound expression.
- **The learned model ranks conditions usefully while failing to cover.** Rank
  correlation with the true conditional mean gap rises from 0.5371 at n = 15 to
  0.6793 at n = 60.
- **Nineteen negative controls, 11 fired, every one as declared.**

### Defects found during the build and pinned as regressions

- **`conformal_required_n` was wrong at α = 0.10 and α = 0.20.** It computed
  `ceil(1/(1−level)) − 1` with `level = 1−α`, reconstructing α from its own
  complement; `1 − (1 − 0.10)` is `0.09999999999999998`, whose reciprocal
  ceilings to 11, so the function returned 10 where the answer is 9, and 5
  where it is 4. The reciprocal now comes from α directly and is reconciled
  against `conformal_order_index` by a short walk. Pinned with literal inputs.
- **The trend statistic reported a perfect monotone drift on a constant
  sample.** Ranking by `argsort(argsort(x))` breaks ties by position, so
  `numpy.full(10, 3.0)` produced the rank vector `[0…9]` and a correlation with
  the index of exactly 1.0. Average ranks via `scipy.stats.rankdata` give a
  constant rank vector, a zero denominator and the documented 0.0. Pinned.
- **The rank-invariance property is false where the affine map underflows.**
  Hypothesis falsified it at 300 examples with `values = [5e-324, 0.0, 0.0]`
  and `scale = 0.5`: the multiplication underflows the denormal minimum to
  exactly 0.0, the values tie, and the statistic falls from
  −0.8660254037844387 to 0.0. The property test now assumes distinctness is
  preserved; the counterexample is pinned with literal inputs as a statement
  about the precondition rather than as a defect.
- **The coverage standard error omitted the held-out pool's own error.** At
  `m = 4000` and `p = 0.95` the pool term is 0.0034, larger than the repetition
  term at 300 repetitions. Reporting only the repetition term declared split
  conformal — exactly marginally valid by construction — to be measurably
  under-covering at n = 60 on 0.9454 against 0.95 with a stated error of
  0.0019. With the pool term of equation (27) the stated error is 0.0039 and
  the measurement is within one and a half standard errors, which is what the
  theory says.
- **A negative control that could not fire.** The first Hoeffding control
  corrupted the bound by scaling the slack down, at 0.5 and 0.25, and did not
  fire at either. The measured reason is now printed: at n = 40 the sample mean
  of 40 gaps has a sampling sd of 0.61 kN.m and is unbiased, so the bound
  reduced to the bare sample mean still covers on 48 per cent of draws and at
  factor 0.25 covers on 100 per cent. The must-fire controls are now sign
  errors whose margin below the truth, in standard deviations of the sample
  mean, is computed and printed, so the control fires by construction rather
  than by luck.
- **A prediction falsified by its own measurement, recorded rather than
  corrected.** The 15-point pilot campaign was expected to understate the
  variance and therefore to plan too few flights. On the declared seed it
  overstates — 20.446342 kN.m² against a true 14.826382 — so its plan asks for
  more flights than needed. The expectation is recorded in the script's
  docstring as falsified, and the direction of the effect is established by
  rows anchored on the true variance at 0.5×, 1.0× and 2.0× instead.
- **The permutation test was too slow to measure its own power.** A Python
  loop over `B` permutations made the size-and-power study exceed the declared
  three-minute budget. Both named statistics now have a closed-form
  vectorised permutation null over a `(B, n)` permutation matrix, which is the
  same distribution and one C-level call.

### Known limitations

See `README.md`. The dominant ones: every number is measured against synthetic
paired records generated by this package's own declared equations and not
against any real test article; the distribution-free bounds are very loose on
this population, loose enough that at n = 40 Hoeffding's entire slack could be
deleted without losing coverage; `R` is the foundation of every
distribution-free bound and the package can only catch a gross error in it;
independence is assumed by `transfergap.bounds` and is not tested, only
exchangeability is, and only against drift and a step change; the
exchangeability test has power 0.43 at n = 15; the conformal repair is marginal
and not conditional; and the declared dependency ranges were measured at one
version each except `ruff`, because the build environment is a shared
pre-pinned interpreter this product is forbidden to modify.

[0.1.0]: https://github.com/OmAcharya-avtr/transfergap/releases/tag/v0.1.0
