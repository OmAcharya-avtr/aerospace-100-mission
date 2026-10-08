# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-10-08

First release. Status: TESTING. Validation level 3, research grade.

### Added

- `twininvalidate.twin`: a declared linear-Gaussian digital twin
  (`LinearGaussianTwin`) and its fixed-gain steady-state residual generator
  (`SteadyStateFilter`), with zero-order-hold discretisation and the
  continuous white-noise-acceleration process-noise model. A shipped
  illustrative single-axis attitude channel at 20 Hz.
- `twininvalidate.asset`: seeded injection of three asset-side changes — a
  parameter step, a slow ramp and a noise-variance change — plus a
  sensor-offset mechanism used only for the attribution demonstration.
- `twininvalidate.detectors`: the three declared analytic baselines — a
  two-sided tabular CUSUM (Page 1954), an EWMA (Roberts 1959) and a windowed
  generalised-likelihood-ratio test (Willsky & Jones 1976) — and a
  variance-CUSUM oracle used only to measure the baselines' mis-specification
  cost. All statistic paths are threshold-independent.
- `twininvalidate.thresholds`: threshold setting from a declared false-alarm
  target, with exact closed forms for GLR(window=1) and EWMA(λ=1) and
  bisection on a censored-MLE in-control ARL0 for everything else;
  `bracket_threshold` for statistics that take finitely many values.
- `twininvalidate.arl`: censoring-aware ARL0 and ARL1 estimators, the
  zero-state and steady-state delay conventions, and the delay-against-
  false-alarm curve.
- `twininvalidate.features` and `.classifier`: nine windowed residual
  features and an isotonic-calibrated random-forest drift classifier with a
  confidence output, benchmarked against the baselines at a declared
  false-alarm budget.
- `twininvalidate.ambiguity`: a paired-world simulation demonstrating that an
  asset fault and a twin configuration error can produce algebraically
  identical residual streams.
- `twininvalidate.monitor`: `InvalidationMonitor`, the user-facing object.
- CLI `python -m twininvalidate` with `twin`, `scenarios`, `calibrate`,
  `curve`, `benchmark` and `ambiguity` subcommands.
- Five plotting examples writing PNGs to `screenshots/`.
- Eight validation scripts with their raw output in `validation/*_output.txt`.
- 243 tests, including known-answer tests with the hand calculation in the
  test comments, 15 Hypothesis property tests, an integration test and a
  regression/benchmark test with locked numbers.
- `MODEL_CARD.md`, `DATASET_CARD.md`, `docs/REQUIREMENTS.md`,
  `validation/VALIDATION.md`, `CITATION.cff`.

### Measured results published as negative findings

- The CUSUM is faster than the windowed GLR on all three change types at a
  matched in-control ARL0, which is the opposite of the expectation this
  product was specified against.
- A correctly-specified variance CUSUM, handed the post-change residual
  variance in advance, is the **slowest** method on the noise-variance change
  (466.6 samples against the mis-specified mean CUSUM's 211.0).
- The learned drift classifier loses to the CUSUM on 2 of 3 in-distribution
  scenarios and 3 of 4 out-of-distribution ones at a comparable false-alarm
  budget, and detects 2 % of runs on a change of the same magnitude and the
  opposite sign where every baseline detects 100 %. It was not retuned.
- One validation check fails: the variance-CUSUM oracle's calibrated
  threshold transfers to fresh in-control banks with a bias of −8.6 % and
  +13.5 %, outside the declared 8.8 % tolerance. The tolerance was not
  widened.

### Known limitations

Scalar measurements only; linear and Gaussian throughout; the headline
false-alarm rate of 72 000 per 1000 h is far above any operational
requirement and the operating point a real requirement implies is outside the
compute budget; detectability requires excitation; the package cannot
attribute a residual to a cause. See the README.

[0.1.0]: https://github.com/OmAcharya-avtr/twininvalidate/releases/tag/v0.1.0
