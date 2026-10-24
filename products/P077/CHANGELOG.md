# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-10-24

First release. Status TESTING; validation level 2; research grade, not
flight-qualified, not certified, not approved for operational aerospace use.

### Added

- `relfiledelivery.link`: declared link conditions — forward and return bit
  rates, one-way light time, independent PDU loss, a continuous-time two-state
  Gilbert (1960) outage process with a lazily generated seeded trace, and a
  contact plan that places transmissions inside declared windows.
- `relfiledelivery.payload`: file segmentation, deterministic source payload
  generation and the SHA-256 identity the completion invariant is checked
  against.
- `relfiledelivery.analytic` (**baseline 1, written first**): the closed form.
  Expected transmissions `n/(1-p)`; the exact round-count CDF `(1-p**k)**n`;
  `E[R]` as a series summed to a stated tolerance with a rigorous Bernoulli
  truncation bound returned alongside; exact round-count and negative-binomial
  transmission-count quantiles; the expected deferred-NAK delivery time; the
  unacknowledged completion probability; and a declared extension past the
  no-outage assumption that divides by the outage duty cycle and lays the
  result over the declared contact plan. `analytic_delivery` raises rather than
  silently applying itself outside its assumptions.
- `relfiledelivery.simulate` (**baseline 2**): a seeded simulator for
  unacknowledged, deferred-NAK and immediate-NAK transfer over the link model,
  with seven declared modelling choices documented in the module docstring. The
  completion invariant — delivered bytes equal source bytes, verified by
  SHA-256 — is checked inside the simulator on every completed run, and
  `inject_corruption_at` exists so that it can be shown to fail when it should.
- `relfiledelivery.montecarlo`: ensembles reported as quantiles with the tail
  stated separately, per-contact completion probability, and the mean only
  beside the mean-over-median ratio that shows when it has stopped describing
  the distribution.
- `relfiledelivery.dataset`: a seeded scenario generator over two regimes,
  whose outage and contact time scales are drawn relative to the clean-pass
  time of the file so that the two regimes genuinely differ.
- `relfiledelivery.predictors`: four delivery-time predictors sharing one
  interval interface — the closed form scored as a median, the closed form
  scored as a mean, a measured-quantile lookup table, and seven
  gradient-boosted quantile regression models.
- `relfiledelivery.benchmark`: error, coverage, interval width, per regime,
  with the coverage standard error reported beside every coverage figure.
- `relfiledelivery.report` and a `python -m relfiledelivery` command line with
  `info`, `analytic`, `simulate`, `distribution`, `compare` and `benchmark`.
- 777 tests, counted from `junit.xml` and cross-checked against a literal count
  of `testcase` elements. Property-based tests with Hypothesis over the
  algebraic identities and the simulator invariants.
- Nine validation scripts with their committed raw output, flat in
  `validation/`.
- Six example scripts producing seven PNGs in `screenshots/`.

### Measured

- Simulated mean delivery time agrees with the exact closed-form expectation to
  within 1.9000 sampling standard errors over 12 grid cells, worst relative
  discrepancy 1.537594e-02.
- 0 digest mismatches over 1829 delivered runs of 2400; 15 of 15 injected
  single-bit corruptions caught.
- The closed form beats the learned quantile model by a factor of 2.0160 on
  median absolute relative error in the regime where the closed form is exact,
  using no training data. The learned model beats the closed form by a factor
  of 1.2623 in the correlated-outage regime. Both are published as measured.
- No uncertainty output is calibrated at nominal 0.90: learned 0.8146, closed
  form 0.5093, lookup 0.8784, over 1127 held-out scenarios.

### Fixed during the build

- `expected_rounds_detail` and `rounds_cdf` evaluated the survival term as
  `1 - (1 - p**k)**n`, which loses an absolute error of order `n * eps` to
  catastrophic cancellation — 2e-13 at `n = 1000`, a thousand times the stated
  truncation tolerance of the series. Found by
  `validation/validate_analytic.py` comparing the reported truncation bound
  against the measured truncation; the hand-written tests did not find it. Both
  are now `-expm1(n * log1p(-p**k))` and `exp(n * log1p(-p**k))`, and the worst
  slack over nine `(n, p)` pairs moved from -1.83e-13 to +7.30e-16.
- `LinkConditions` rejected a contiguous window list built by its own
  `uniform_windows` helper, because `first + 10 * period` and
  `first + 9 * period + duration` are not the same floating-point number and
  the overlap test was a strict comparison. Found by Hypothesis at
  `uniform_windows(11, 114.06854261351316, 114.06854261351316)`; now compared
  against a named relative tolerance and pinned as a regression test.
- `LinkConditions.contact_duty_cycle` returned 1.0000000000000002 for the same
  window list, which is not a probability, and is used as a divisor downstream.
  Now clamped, with the reason documented at the point of the clamp.
- The analytic extension divided the closed form by the product of the contact
  and outage duty cycles. On the worked-example scenario that gave 164.57 s
  against a measured median of 26.80 s — a factor of 6.14 — because the
  transfer fitted inside the first contact and the contact duty cycle never
  bit. The extension now lays the required contact time over the declared
  windows in order. The learned model's win in the correlated-outage regime
  fell from a factor of 1.9036 to 1.2623 as a result, and the smaller figure is
  the one reported.

[0.1.0]: https://github.com/OmAcharya-avtr/relfiledelivery/releases/tag/v0.1.0
