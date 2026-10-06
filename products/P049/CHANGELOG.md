# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-06

First release. Validation level 2 (research grade). Status: TESTING.

### Added

- `linkoutage.fade` - the deterministic core. Down-crossing and up-crossing
  indices, maximal below-threshold runs with left- and right-censoring flags,
  fade durations, level-crossing rate, mean fade duration, outage fraction and
  availability. Every definitional choice is a field of `FadeDefinitions`
  (strict or non-strict threshold comparison; sample-count, interval-count or
  linearly interpolated durations; censored runs excluded or folded in;
  single-sample excursions counted or not; `(N-1)/fs` or `N/fs` record
  duration), and `FadeStatistics.report()` prints the conventions beside the
  numbers. The Rice partition identity
  `mean_fade_duration = outage_fraction / level_crossing_rate` is carried as a
  built-in consistency residual.
- `linkoutage.channel` - the correlated lognormal fading model: scintillation
  index to log-irradiance variance, correlation time to AR(1) coefficient, a
  stationary AR(1) filter implemented with `scipy.signal.lfilter` plus a
  closed-form initial-condition correction, and the exact mapping from an
  amplitude threshold to a level of the underlying Gaussian. The analytic
  discrete-time down-crossing probability is an exact bivariate-normal orthant
  probability computed two independent ways (`multivariate_normal.cdf` and a
  one-dimensional `quad`), with the analytic level-crossing rate, outage
  fraction and mean fade duration derived from it, and a conditional onset
  probability for the analytic forecaster.
- `linkoutage.distributions` - fade-duration distribution fitting with right
  censoring: exponential maximum likelihood including censored observations,
  shifted-geometric maximum likelihood on sample counts, a Pearson chi-square
  of the sample-count histogram against the geometric (the valid test for
  discrete data), Kolmogorov-Smirnov via the probability-integral transform
  with its anticonservativeness and tie caveats attached, lognormal, Weibull
  and gamma fits, AIC ranking, Kaplan-Meier survival, and a single published
  verdict on the memoryless hypothesis.
- `linkoutage.markov` - amplitude quantisation into `K` channel states,
  first-order Markov maximum likelihood with stationary distribution and
  implied mean dwells, a per-state dwell-time chi-square against the geometric
  the fit implies, a conditional-independence test of the Markov property with
  Cramer's V as its effect size, and a semi-Markov fit with empirical dwell
  distributions and a simulator.
- `linkoutage.features` - leakage-controlled feature extraction (14 features
  from the observation window only), onset labels from the horizon only,
  decision-instant generation, and a temporal train/calibration/test split
  whose gap is sized to make the windows and horizons either side of a
  boundary disjoint.
- `linkoutage.predictors` - three baselines implemented and validated before
  the learned model: the constant base-rate forecaster, the analytic
  conditional level-crossing-rate forecaster (which fits the channel from the
  training amplitude record and uses no labels), and logistic regression; then
  a random forest with the spread of its per-tree votes as an uncertainty
  output, and a Platt recalibration wrapper fitted on the calibration split.
- `linkoutage.calibration` - Brier score, the Murphy reliability-resolution-
  uncertainty decomposition with its binning residual reported, expected and
  maximum calibration error on equal-count bins, reliability curves with
  Wilson intervals, ROC AUC, average precision, and the all-negative accuracy
  printed next to every accuracy so the reader can see what accuracy is worth
  at a 2.6 % base rate.
- CLI `python -m linkoutage` with `stats`, `fit`, `crosscheck` and `predict`
  subcommands, reading `.npy` or text amplitude files.
- 264 tests (pytest + Hypothesis), four examples producing the four
  screenshots, nine validation scripts with their raw output.

### Published results that went against the author

- The memoryless (exponential/geometric) fade-duration assumption is
  **rejected** at every threshold tested, with a geometric chi-square of
  62510.3 on 90 degrees of freedom at the specified threshold and a measured
  coefficient of variation of 2.571 against 1.0 for an exponential.
- The two-state Markov channel model fits the **mean** dwell of each state to
  four figures and the **dispersion** to a factor of about 2.7. Cramer's V on
  the triplet table is 0.264.
- In the outage-classifier benchmark **a baseline wins**: the analytic
  level-crossing-rate forecaster with a two-parameter Platt recalibration
  scores a held-out Brier of 0.0217557 against 0.0223498 for the random
  forest, 0.0222801 for the recalibrated forest and 0.0220142 for logistic
  regression. The forest is reported with those numbers and has not been
  retuned.
- The raw (uncalibrated) analytic forecaster has the **highest** discrimination
  of the four (AUC 0.873) and is **worse than doing nothing** on the Brier
  score (skill score -0.0043), because it over-forecasts by a factor of two.
  Both facts are published.

### Known limitations at this release

See the Limitations section of README.md. In short: one synthetic channel
model, no gamma-gamma distribution, no aperture averaging or pointing jitter,
no measured-spectrum fading, sample-rate-dependent crossing rates by
construction, and an uncertainty output whose usefulness is suggested but not
established by this experiment.
