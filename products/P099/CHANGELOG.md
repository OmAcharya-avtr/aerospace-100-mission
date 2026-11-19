# Changelog

All notable changes to this product are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) loosely, and the
versioning is [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 — 2026-11-19

First build. Status **TESTING**. Research-grade software: not flight-qualified,
not certified and not approved for operational aerospace use.

### Added

- `passyield.geometry` — ground-station pass geometry from an SGP4
  propagation of catalogue object 06251, taken verbatim from the official SGP4
  verification element set installed by the `sgp4` package. Mask crossings by
  coarse scan and bisection, IAU-1982 GMST, a pseudo-Earth-fixed rotation with
  its cost measured, and an independent spherical-Earth closed form used only
  as a cross-check.
- `passyield.link` — achievable downlink rate against elevation from a cited
  link budget: Friis (1946) free-space loss, the ITU-R P.618 secant slant-path
  law with its 10 degree validity floor **enforced by raising**, the standard
  Earth-space link equation, and a Shannon (1948) capacity envelope with an
  implementation margin and a modcod ceiling.
- `passyield.decompose` — the per-pass accounting identity, with the four
  terms integrated on their own composite-Simpson partitions and the closing
  residual **reported as a number on every result**. Declared interval
  priority so the three loss terms are pairwise disjoint. Three documented
  edge cases: a pass that never clears the mask, a zero-duration pass, and an
  acquisition longer than the whole pass.
- `passyield.distribution` — `ShortfallDistribution`, the product's headline
  output: a probability mass function on a uniform Mbit grid with its
  quantiles, its grid step, its quantile discretisation, its truncated mass
  and its resampling unit.
- `passyield.baselines` — baseline 1 (the booked figure, at full strength),
  baseline 2 (the analytic law by mass-preserving pushforward and
  compound-Poisson convolution), baseline 2b (the same with a backlog-aware
  idle term), baseline 3 (accounted ground truth with a bootstrap whose
  resampling unit is the **pass**), and a direct Monte Carlo of the same laws
  as a second method.
- `passyield.bounds` — two brackets derived from mathematics the estimators do
  not use, with a slack that follows the conditioning of the bound expression
  rather than an absolute constant.
- `passyield.predictor` — a random forest over 17 pre-pass features with two
  uncertainty outputs, the across-tree spread and the out-of-bag residual law,
  the second of which is the declared output.
- `passyield.scoring` — CRPS for an ensemble, PIT values and their Kolmogorov
  distance from uniform, interval coverage, MAE and RMSE.
- `passyield.dataset` — the station catalogue, the CSV schema, the loader with
  row invariants, and the two declared splits: held-out stations and held-out
  time.
- `passyield.report` and `python -m passyield {decompose,distribution,accounted}`.
- `data/generate_passes.py` and the committed 1990-pass campaign, with its
  manifest and transcript.
- 284 tests; 11 validation scripts with their committed output; 4 examples
  with their committed figures.

### Measured

- SGP4 against the official `tcppver.out` for catalogue 06251, 25 epochs over
  48 hours: worst position disagreement 7.266e-09 km.
- Decomposition identity residual over 1990 passes: median 1.464e-08, 95th
  percentile 3.889e-07, worst 8.706e-04 relative; concentrated 220.3 times on
  the passes where the rate profile saturates and the integrand loses its
  fourth derivative.
- Convolution against a direct Monte Carlo: 0 of 161 quantile comparisons
  outside a tolerance derived from the convolution stage count.
- Independent bracket: 0 of 1990 accounted shortfalls and 0 of 394 quantiles
  outside; 7 of 7 negative controls fired, 2 of 2 silent.
- `pytest.approx` hidden-default audit: 80 of 80 calls pass `abs=`, and 0
  assertions depend on the default under a plugin that forbids it.
- Working-directory probe: 14 of 14 scripts exit 0 from four directories.

### Published as measured, not retuned away

- The specified baseline 2 is **beaten by a trivial constant-fraction null on
  both splits** (CRPS 7014.63 against 4891.64 on held-out stations, 4102.21
  against 3685.53 on held-out time). The cause is diagnosed and the
  strengthened variant 2b is reported beside the specified form, not instead
  of it.
- On the harder split, the trivial null is **better calibrated than the
  learned model** (PIT 0.1961 against 0.2410, cov80 0.7185 against 0.6344).
  Calibration is what a scheduler needs in order to book a quantile.
- On the worst pass, **62.6 per cent of baseline 2's own law is physically
  impossible**, because the convolution assumes the three losses are
  independent. The figure is reported as `truncated_mass` on every returned
  distribution.

### Known limitations

See `README.md` under Limitations. The two that most constrain use: the
learned model's uncertainty output is not calibrated on an unseen station, and
the dependency ranges were not run at both ends because `pip` is forbidden to
build agents in this batch.
