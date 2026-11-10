# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-11-10

First release. Status **TESTING**: built and validated, awaiting QA and batch
review.

### Added

- **`sitediverse.geometry`** — great-circle separation (equation 1), initial and
  midpoint azimuth (equation 2) and the along-axis / cross-axis decomposition
  (equation 3), plus a declared synthetic network of 20 stations in 5 groups of
  4 spanning 91.9 km to 7299.5 km.
- **`sitediverse.field`** — the synthetic concurrent cloud record: an
  exponentially correlated Gaussian field (equation 4) with AR(1) persistence
  (equation 5), thresholded against a seasonal and diurnal probit climatology
  (equation 6), in four declared seasonal regimes. Three declared records:
  `anisotropic` (the reference), `isotropic` (the control, on which the
  two-parameter baseline is correctly specified by construction) and `short`.
- **`sitediverse.climatology`** — the season-by-hour cell estimator (equation 12)
  with shrinkage, and the effective sample size for a serially dependent series
  (equation 13).
- **`sitediverse.independence`** — baseline 1: the product of marginals
  (equation 14) in log space, the epoch-conditional product (equation 15) and
  the exact overstatement decomposition (equation 16).
- **`sitediverse.copula`** — the Gaussian copula (equations 7 and 8), the
  bivariate orthant computed through the Plackett reduction formula (equation 9)
  so that the correlation excess never passes through a cancelling subtraction,
  the n-station Monte Carlo with a reported standard error (equation 10), and
  the implied-correlation inversion (equation 11).
- **`sitediverse.bounds`** — the Frechet-Hoeffding bounds (equation 17), a
  second-order Bonferroni bracket that is an exact identity at n = 2
  (equation 18), and a comparison slack that follows the conditioning of the
  bound expression (equation 19).
- **`sitediverse.empirical`** — baseline 3: the observed joint rate
  (equation 20), a moving-block bootstrap (equation 21) and a paired interval on
  a difference (equation 22).
- **`sitediverse.correlation`** — baseline 2: the specified two-parameter
  exponential (equation 23), a strengthened per-regime form (equation 24), and a
  correlation-matrix assembly that projects onto the nearest positive-definite
  matrix and reports the projection (equation 26).
- **`sitediverse.uncertainty`** — first-order propagation of the marginals'
  sampling uncertainty into the independence product (equation 27) and the
  climatology-uncertainty share with its bias-corrected form (equation 28).
- **`sitediverse.splits`, `.features`** — the held-out SITES and held-out TIME
  split (equations 29 to 31) and the pair-regime feature table (equations 32 to
  34), with the station identity, coordinates and altitude deliberately excluded.
- **`sitediverse.learned`** — a three-quantile gradient-boosted correlation model
  (equations 35 to 38) exposing an interval and no bare point-estimate method.
- **`sitediverse.study`** — the ordered comparison (equations 39 to 41):
  baselines fitted and validated first, then the learned model, then the paired
  bootstrap on each hold-out cell's margin.
- **A CLI**, `python -m sitediverse`, with eight subcommands: `sites`, `record`,
  `climatology`, `overstate`, `bounds`, `correlate`, `learned`, `network`.
- **433 tests**, counted from the junit XML; 7 examples producing 7 committed
  figures; 15 validation scripts with their raw output committed beside them.
- **`MODEL_CARD.md`**, **`DATASET_CARD.md`**, **`validation/VALIDATION.md`** and
  **`docs/REQUIREMENTS.md`**.

### Measured

- The independence assumption overstates availability by up to a factor of
  **22.98** in joint unavailability on the declared six-station two-group
  network, which is **1.3614 nines** of availability that are not there.
- The learned model's margin over the strongest baseline is **smaller than the
  bootstrap half-width on the empirical reference on every hold-out cell of both
  records**. On held-out TIME the baseline wins on both records; on the
  correctly-specified record the held-out-SITES margin is not distinguishable
  from zero.
- **0 of 433** assertions depend on the hidden `pytest.approx` absolute default.
- **26 negative controls**: 15 must fire and all 15 did; 11 must not and none
  did.
- **0 non-zero exits in 66** working-directory probe runs.

### Known limitations at this release

Listed in full in README.md. The headline ones: the record is synthetic and
nothing transfers to a real network; the block bootstrap's interval width does
not plateau, so every quoted interval is a lower bound; the learned model's
80 per cent interval covers 0.4643 on held-out TIME; and dependency ranges were
run at both ends for `ruff` only.

[0.1.0]: https://github.com/OmAcharya-avtr/sitediverse/releases/tag/v0.1.0
