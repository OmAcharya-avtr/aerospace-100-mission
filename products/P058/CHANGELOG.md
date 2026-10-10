# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-10

First release. Status: TESTING. Validation level 2.

### Added

- Five analytic streaming change detectors behind one interface, each with a
  single calibrated scalar and a stated threshold-setting procedure: `CUSUM`
  (two-sided tabular, Page 1954), `PageHinkley` (running-mean form),
  `EWMA` (exact time-varying control limits rather than the asymptotic ones),
  `WindowedKS` (fixed reference window against a sliding detection window), and
  `ADWIN` (exponential-histogram adaptive window, cut rule transcribed from the
  authors' own technical report).
- `ks_two_sample_statistic`, a direct two-sample KS statistic checked against
  `scipy.stats.ks_2samp` to machine precision and measured 30x cheaper per
  evaluation. `scipy.stats.kstest(x, "norm", args=(loc, scale))` is never
  called: it raises `TypeError` on the installed SciPy.
- `alarm_ratio()` on every detector: the current statistic divided by its own
  threshold, so five incomparable scales share one axis and 1.0 is the alarm
  line for all of them.
- `is_armed()` and `blind_fraction`, which measure the fraction of a stream on
  which a detector cannot raise an alarm at all. The windowed KS test is
  un-armed for 65.9 % of a stationary stream at its calibrated threshold, and
  that figure is invisible in an ARL0/ARL1 pair.
- `measure_arl0` (restart-after-alarm, with the right-censored tail reported
  rather than absorbed) and `measure_arl1` (steady-state convention, no
  replicate excluded, censoring flagged as a lower bound). Every figure carries
  a Monte Carlo standard error.
- `calibrate_threshold`: bisection on a declared target ARL0 over seeded
  stationary streams, with the achieved ARL0 reported on disjoint seeds and
  bracketing failure reported rather than raised.
- `tradeoff_curve` and `transient_response`, the two experiments that make the
  detectors comparable: the delay-versus-false-alarm curve, and the transient
  negative control measured against a matched stationary baseline.
- A learned detector: a random forest on eight causal windowed features, with
  an online path and a bit-identical batch path 1113x cheaper per sample.
- `telemdrift.reference`: the NIST handbook's CUSUM ARL table and Siegmund's
  closed form, used to validate the measurement machinery against a published
  value rather than only against itself.
- CLI `python -m telemdrift` with `detectors`, `defaults`, `calibrate`, `arl`,
  `tradeoff`, `transient` and `trace`, text and JSON output, a
  `--budget {full,quick}` switch that announces when it is reduced, and exit
  status 2 for a finding.
- Six runnable examples, each writing a PNG into `screenshots/`.
- Twelve validation scripts with their raw output committed under
  `validation/outputs/`, plus the junit XML the test count is read from (written to a temporary directory,
  not committed: see validation/VALIDATION.md).
- 302 tests: unit, input validation, known-answer with the hand arithmetic in
  the test comments, edge cases, 18 Hypothesis property tests including two
  deliberate *non*-properties, plotting regressions and CLI subprocess tests
  that assert on exit status.

### Measured results at this release

- The five shipped default thresholds span a **46.6x** range of measured ARL0
  (168 to 7819 samples). Calibration to a 500-sample target reduces the spread
  to **1.24x**.
- Measured two-sided CUSUM ARL0 agrees with the NIST handbook's halved
  one-sided value to **−2.0 %** at `h = 4` and **−1.4 %** at `h = 5`, and the
  zero-state ARL1 at a one-sigma shift to **−1.8 %** and **+0.9 %**.
- **The learned detector loses at equal ARL0** on three of four change types at
  more than two combined standard errors (1.88x, 1.45x and 1.31x slower than
  the best analytic detector), ties on the fourth, and wins on none.
- **All six detectors fire on the transient negative control**, with an
  attributable excess over a matched baseline of +32.8 to +88.8 percentage
  points. Training the learned detector explicitly on transients improves its
  rate by 3.6 percentage points.
- Autocorrelation destroys an i.i.d.-calibrated threshold: at lag-one 0.9, with
  the marginal distribution unchanged, measured ARL0 falls by up to **34.5x**.
- The windowed KS test's trade-off curve is **non-monotone**: its shortest
  measured delay is at ARL0 = 1243, not at its tightest threshold.

### Known limitations at this release

See the Limitations section of README.md. The ones that bite first: every
threshold is calibrated on an i.i.d. Gaussian stream and real telemetry is not
i.i.d.; the equal-ARL0 comparison is only equal to within 1.24x at a budget two
cores can afford; Page-Hinkley's `lambda` does not transfer between channels of
different scale; three of the five detectors cannot detect a mean step started
cold; and one validation check fails on purpose, recording that the learned
model overfits at the window level by +0.118 ROC-AUC against a pre-declared
expectation of 0.05.
