# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-08

First release. Validation level 2 (research grade). Status: TESTING.
Contains a learned component, benchmarked against an analytic baseline that
was implemented first, and reported as winning on one instance and having
nothing to win on the other.

### Added

- `rareverify.intervals` — Clopper-Pearson and Wilson confidence intervals for
  a binomial proportion, one-sided and two-sided, with the `k = 0` and `k = n`
  boundaries set analytically rather than taken from a degenerate Beta
  quantile; the closed-form zero-failure bounds `1 - alpha**(1/n)` and
  `z^2/(n+z^2)`; the `3/n` rule-of-three approximation, provided so its error
  can be measured; and `exact_coverage`, which computes a method's coverage by
  summing binomial probability mass rather than by simulation.
- `rareverify.planner` — the two sample-size questions kept separate.
  Demonstration (`ceil(ln(alpha)/ln(1-p_target))`, exact and minimal) and
  estimation (search on the interval width, with the non-monotonicity of
  `k = round(n p)` documented and the back-scan window stated).
  `plan_campaign` reports the expected violation count and `P(k = 0)` alongside
  the size, because those are what tell a planner whether the plan is sane.
- `rareverify.limitstates` — three limit states on standard normal inputs, each
  with a reference probability accurate far beyond any Monte-Carlo estimate of
  it: linear Gaussian and lognormal ratio in closed form, rippled by 400-node
  Gauss-Hermite quadrature. The rippled state's `design_point()` deliberately
  returns the *smooth part's* design point, which is not the most probable
  failure point, because that is what an analyst has without a surrogate.
- `rareverify.montecarlo` — the crude estimator, batched to bound memory, with
  the arithmetic that motivates everything else (`required_samples_for_cov`).
- `rareverify.tilting` — importance sampling in an explicitly declared family:
  the mean-shift (translation) family on standard normal space, which for the
  Gaussian *is* exponential tilting, with the likelihood ratio in closed form.
  `analytic_mean_shift`, plus `scaled_mean_shift`, `orthogonal_mean_shift` and
  `oracle_mean_shift` so that bad tilts can be generated on purpose. Two
  independent design-point searches: SLSQP, and a vectorised ray search with an
  SLSQP polish step.
- `rareverify.subset` — subset simulation (Au and Beck 2001) with the modified
  Metropolis algorithm, per-level acceptance rates, and a standard error
  explicitly labelled `subset-independence-lower-bound`.
- `rareverify.estimate` — a single estimator result type and, more importantly,
  the dispatch that refuses to pair a binomial interval with a weighted
  estimator. `binomial_interval` raises on an importance-sampling or
  subset-simulation result; `interval_for` returns a central-limit interval
  instead and records why in the interval's own rationale string.
  `bootstrap_interval` resamples the stored non-zero contributions exactly in
  two stages.
- `rareverify.surrogate` — the learned component. A Gaussian-process surrogate
  of the limit-state function on a declared radial design, used in two ways
  that are kept apart: as the proposal (unbiased, every sample evaluated on the
  true limit state) and as the estimator (biased, implemented so the bias can
  be measured). The design point is located on the posterior mean and on the
  mean plus and minus two posterior standard deviations, so the model's output
  is a reliability-index band rather than a point.
- `rareverify.benchmark` — replication-based measurement of variance reduction
  on an equal true-evaluation basis, carrying both a variance reduction factor
  and a mean-squared-error reduction factor, because the two disagree for a
  biased estimator and only the second one is a quality metric.
- `rareverify.knownanswer` — known-answer checks whose tolerance is stated as a
  multiple of the counting-noise floor of the run that produced the estimate.
- `rareverify.campaign` — plan, run, and quote the one-sided bound the plan was
  sized from.
- CLI `python -m rareverify` with `plan`, `interval`, `coverage`, `mc`, `is`,
  `tilt-sweep`, `subset`, `surrogate`, `benchmark`, `known-answer` and
  `campaign`; exit 2 for invalid input and 3 for a failed check or an unmet
  target.
- 203 tests: hand-calculated known-answer tests with the arithmetic in the
  comments, Hypothesis property tests for the interval algebra (monotonicity in
  `k` and in `n`, containment of the point estimate, agreement of the
  zero-failure closed forms with the general path, and exact conservativeness
  of Clopper-Pearson coverage), input-validation and edge-case tests, an
  integration test that runs a campaign end to end, and a regression suite that
  pins every number this repository reports.
- Nine validation scripts with their committed raw output (92 checks, 0
  failures), five examples each producing a PNG in `screenshots/`,
  `MODEL_CARD.md` and `DATASET_CARD.md`.

### Measured negative results in this release

Published because they are the most informative part of the package.

- **The learned surrogate has nothing to win on a smooth analytic limit
  state.** It recovers the design point to within 5.890e-06 standard-normal
  units of the analytic one, so it is the same estimator, and it pays 100 of
  60 000 true evaluations for the privilege — a deterministic 0.167 % variance
  penalty plus fit and search time against a closed form that costs nothing.
  The measured variance ratio of 1.1006 is inside the ±3-sigma
  replication-noise band of 0.485 to 2.063. No retune was attempted.
- **The surrogate does win on a rough limit state**, by a measured factor of
  10.66 in variance against the analytic smooth-part tilt at equal budget, and
  still **falls 31 % short of the oracle tilt** (VRF 155.8 against 224.2).
- **26 of 40 swept importance-sampling tilts are worse than plain Monte Carlo**
  at the same cost. A sideways tilt at half the design-point magnitude is 108
  times worse in variance; at full magnitude, 3943 times worse. A tilt in the
  wrong direction returns exactly zero from every one of 40 replications.
- **An over-tilted sampler reports a variance reduction factor of 1.8e+08
  around an estimate that is 2.07e5 times too small.** The first version of
  the sweep called that "better". The mean-squared-error reduction factor was
  added because of it, and says 0.09998.
- **Subset simulation's own standard error understates its measured spread by
  a mean factor of 2.69**, worst 3.33, because it assumes independence within
  a level that Markov chains do not provide.
- **The crude estimator's plug-in standard error understates its spread by a
  factor of 3.6** at `p = 1e-6` with 100 000 runs, because most replications
  see no failure at all.
- **The Wilson interval under-covers**, reaching an exact coverage of 0.904610
  against a nominal 0.95, and its one-sided version reaches 0.930825.
- **The Wilson interval is not always narrower than Clopper-Pearson.** At
  `k = 0` the ordering reverses exactly once in `2 <= n <= 400`, at `n = 46`.
- **`3/n` does not converge to the exact zero-failure bound.** Its relative
  error tends to 1.4246021e-3 from above, not to zero.
- **The sample-efficiency curve is flat** on these two-dimensional problems, so
  there is no training-size trade-off to report. Said rather than dressed up.

### Known limitations in this release

- Standard normal inputs only; no Rosenblatt or Nataf transformation.
- One tilting family, mean shift with identity covariance. A multi-modal
  failure region is not served by it and that failure is not characterised.
- Independence of runs is assumed by every interval and is not checked.
- The relative-width planner's result is minimal only within its 512-step
  back-scan window, because the predicate is non-monotone in `n`.
- The ray-search design point is wrong by 8.7 % in six dimensions at 192
  directions without its polish step, and assumes the failure region is
  star-shaped along each sampled ray.
- Every surrogate result is from a two-dimensional instance with one connected
  failure region.
- Subset simulation is characterised only at `n_per_level` of 1000 to 10 000
  and `p0 = 0.1`.
- No interval here accounts for model error, a mis-specified input
  distribution, or a wrongly written requirement.

### Errors made during the build and corrected

Recorded rather than quietly fixed; full detail in `validation/VALIDATION.md`
section 10.

- The tilt sweep reported variance without bias and labelled a collapsed
  estimator "better". Fixed by adding the mean-squared-error reduction factor.
- `run_campaign` sized itself from the one-sided bound and judged itself
  against the two-sided one, so a campaign that met its target reported
  failure. Fixed by adding `CampaignReport.one_sided_upper`.
- The Gauss-Hermite reference defaulted to 200 nodes, converged to only
  5.709e-05 relative at ripple frequency 12. Raised to 400 after measuring it.
- The ray-search design point had no polish step and was 8.7 % wrong in six
  dimensions. Found by cross-checking against SLSQP.
- The surrogate's design-point search originally took 91 s because SLSQP called
  the Gaussian process one row at a time; replaced with a vectorised ray search
  plus one polish step, measured at 2.1 s for the same work.
- A placeholder check with a hardcoded `True` was written into
  `validate_surrogate.py` while it was being drafted, and replaced with the
  computed condition before the final run.

### Tool defects found and worked around

- `numpy.polynomial.hermite.hermgauss(n)` overflows and returns NaN weights for
  `n >= 400` on numpy 2.5.3 (finite at 200 and 300). The first implementation
  of the rippled reference used it and silently produced NaN.
  `scipy.special.roots_hermite` is finite to at least 3200 nodes and is what
  the package uses.
- A `WhiteKernel` lower bound of zero produces numerically negative posterior
  variances in `GaussianProcessRegressor` on scikit-learn 1.9.1, which
  propagate into the uncertainty output. The bound is set to `1e-8 var(g)`.
