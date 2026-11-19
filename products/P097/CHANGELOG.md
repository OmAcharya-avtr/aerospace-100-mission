# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-19

First release. Research-grade; not flight-qualified, not certified and not
approved for operational aerospace use.

### Added

- `flakerank.geometric` -- the closed-form geometric re-run count and expected
  wasted runtime, derived in the module docstring, with the unbounded form
  `E[F] = p/(1-p)` and the capped form `E[F_m] = (p - p^m)/(1-p) + p^m`. The
  divergence at `p -> 1` raises `ValueError` with an actionable message rather
  than returning `inf`. Five validity limits are stated in the docstring.
- `flakerank.history` -- the `flakerank.history/1` schema, its loader and its
  validation. Durations are exact integer milliseconds. No module in the
  package reads a clock.
- `flakerank.baselines` -- baseline 1 (rank by flake rate, implemented at full
  strength as the thing being refuted), baseline 2 (the analytic expected
  re-run cost), baseline 3 (accumulated ground truth with percentile bootstrap
  intervals whose resampling unit is the session, declared on the returned
  object and asserted before return).
- `flakerank.disagreement` -- Kendall tau-b with the tie correction of Kendall
  (1945), Spearman rho from midranks, bootstrap intervals resampling tests, the
  tie structure, and the tests that move most between two rankings.
- `flakerank.bounds` -- a bracket on total wasted time derived from a failure
  count and order statistics, plus a Cauchy-Schwarz upper bound, neither of
  which uses the accumulator's mathematics. The slack follows the conditioning
  of the bound expression.
- `flakerank.resample` -- percentile bootstrap intervals that carry their
  resampling unit and refuse a caller who expects a different one.
- `flakerank.predictor` -- a random-forest intermittency-cost predictor with a
  confidence output from the spread across trees, scored on held-out TESTS and
  held-out TIME separately. No model is serialised.
- `flakerank.ranking` and `flakerank.report` -- the top-level `rank_tests` and
  its plain-text rendering.
- `python -m flakerank` with `rank`, `disagree` and `geometric` subcommands.
- `data/generate_history.py` and the committed `data/run_history.json`
  (402100 bytes, sha256
  `aa12f8cc861ba66008a0d3762cd68fa24a24b013f897da67c8151996c531b18a`).
- Nine validation scripts, three examples and their screenshots.

### Measured

- The analytic expected-cost baseline reproduces the ground-truth cost ranking
  at Kendall tau-b 0.9988479262672811 over the committed history, against
  0.8191244239631337 for the flake-rate ranking.
- A classical baseline matches or beats the learned forest on 6.67 of 8
  metric-split cells. The result is published as the finding.

### Known limitations

See README.md. The declared dependency ranges were not exercised at both ends:
the build agent is forbidden from running `pip`.
