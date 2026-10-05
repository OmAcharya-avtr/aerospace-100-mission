# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-05

First release. Validation level 2 (research grade). Status: TESTING. Not
flight-qualified, not certified, not approved for operational aerospace use.

### Added

- `telemetryool.limits` — out-of-limit checking with the full operational
  semantics: soft and hard limits ordered
  `hard_low <= soft_low <= soft_high <= hard_high` with any bound optional;
  per-channel validity masks with three explicit policies (`HOLD`, `RESET`,
  `BREACH`) and no policy that silently treats an invalid sample as in-limit;
  separate persistence counts to raise a soft and a hard alarm; a separate clear
  count that steps the latch down one level at a time; mode-dependent limit
  tables with a `"*"` fallback, zeroing the counters on a mode change and
  keeping the latch by default. The per-sample update order is specified in the
  `OolChecker.update` docstring and pinned by two hand traces.
- `telemetryool.arl` — designed thresholds rather than tuned ones. Average run
  length and window false-alarm probability for the two-sided CUSUM (Brook &
  Evans 1972 Markov chain and the Siegmund 1985 closed form), the two-sided EWMA
  (Lucas & Saccucci 1990 Markov chain) and the persistence-counter limit check
  (an exact chain). `design_cusum_h`, `design_ewma_L` and `design_ool_limit`
  invert each one to a target operating point.
- `telemetryool.charts` — `EwmaChart` and `CusumChart` on standardised deviates,
  in both single-sequence and vectorised block form.
- `telemetryool.changepoint` — offline mean-shift change-point detection by
  binary segmentation over the maximised standardised mean-shift statistic
  (Hinkley 1970), with a Monte-Carlo-calibrated threshold reported with its own
  standard error.
- `telemetryool.novelty` — three one-class multivariate models: the classical
  Hotelling `T²`/`Q` PCA monitor (Hotelling 1947; Jackson & Mudholkar 1979), a
  Gaussian-mixture density model and an Isolation Forest (Liu, Ting & Zhou
  2008). Every one exposes `novelty_pvalue`, a calibrated tail probability with
  a Clopper-Pearson interval and an explicit resolution floor.
- `telemetryool.calibration` — the window trigger level, which makes threshold
  calibration an exact empirical quantile rather than a bisection, plus binomial
  standard errors, Wilson and Clopper-Pearson intervals, and
  `windows_for_precision` to size a Monte Carlo from the precision wanted.
- `telemetryool.metrics` — confusion matrices that are never reduced to a single
  score, window-level ROC, and detection-delay statistics that always carry the
  detection probability beside the delay.
- `telemetryool.harness` — the matched-false-alarm-rate comparison: train,
  calibrate on independent nominal data, measure the delivered rate on a third
  independent block, then evaluate.
- `telemetryool.synthetic` — deterministic seeded generators for nominal
  telemetry (AR(1) in time, Cholesky cross-channel correlation, unit marginals)
  and five anomaly kinds, including a decorrelation that leaves every marginal
  unchanged.
- CLI `python -m telemetryool` with `design`, `arl`, `check`, `far`, `compare`
  and `changepoint` subcommands.
- 241 tests, including two hand-traced persistence sequences asserted field by
  field, an exhaustive enumeration check on the persistence-counter chain, a
  Shewhart-limit check on the EWMA chain, and Hypothesis property tests for the
  algebraic identities.
- Five validation scripts with their committed raw output, and four examples
  each producing a PNG in `screenshots/`.

### Known limitations in this release

- **A designed false-alarm rate is a property of the nominal model.** Under
  AR(1) nominal data with a lag-1 coefficient of 0.3, the designed thresholds
  deliver 7.3× (CUSUM), 6.4× (EWMA) and 2.3× (limit check) their design value.
  Measured in `validation/validate_far_design.py`. No automatic prewhitening is
  provided.
- The Siegmund closed form for the CUSUM ARL is within 1.2 % of the Brook-Evans
  chain for shifts up to 1.5 sigma and degrades to **6.17 %** at 3 sigma. The
  2 % band stated before the measurement is **exceeded and reported as
  exceeded**, not widened.
- The two-sided charts combine their arms as independent competing risks, which
  is about a 1 % effect measured against Monte Carlo.
- The Isolation Forest shipped here loses on every scenario measured and is
  below chance (AUC 0.4253) on one. It is kept in the comparison rather than
  removed.
- Everything is measured on synthetic data from this repository's own generator.
  No real spacecraft telemetry was used. See `DATASET_CARD.md`.
- PyTorch is unavailable in the build environment, so there is no autoencoder or
  sequence model. No hyperparameter search was performed on the learned models.
- `telemetryool.limits` is consistent with the structure of the on-board
  monitoring service described in ECSS-E-ST-70-41C, but is **not** an
  implementation of that standard and makes no conformance claim.
