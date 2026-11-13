# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-13

First release. Validation level: Level 4, measured on hardware, with the scope
recorded as part of the result: one power mode, one uninstrumented ambient, one
CPython CPU-only payload, one period, one shared board.

### Added

- `deadlinerisk.trace` -- loader for the campaign's `jobs_<run>.csv` and
  `mon_<run>.csv` files and its `manifest.json`, with schema validation and the
  measurement model stated in the module docstring.
- `deadlinerisk.accounting` -- reconstruction of `releases = sum(1 +
  dropped_before)` and `misses = sum(missed + dropped_before)` from the
  committed traces, the per-release miss indicator in release order, and the
  sample autocorrelation.
- `deadlinerisk.baselines` -- baseline 1, the Gaussian mean-plus-margin rule,
  reported both as a probability and as the `k` for which `mean + k*sd = D`;
  baseline 2, a generalised-Pareto peaks-over-threshold tail with a
  threshold-stability diagnostic over a declared candidate grid and a declared
  stability-window choice rule; baseline 3, the directly counted miss rate.
- `deadlinerisk.bootstrap` -- a moving-block bootstrap whose block length is
  selected from the measured autocorrelation of the miss indicator by a
  declared rule, and the headline understatement factor with its interval.
- `deadlinerisk.speed_scaling` -- the classical `min(1, m/(n+1))` speed-scaling
  predictor, fitted to nothing.
- `deadlinerisk.bounds` -- Markov, one-sided Chebyshev (Cantelli) and
  Bonferroni-bracket bounds, each from mathematics the estimators do not use,
  each with a slack that follows the conditioning of its own expression.
- `deadlinerisk.windows` -- fixed-length windowing, descriptor extraction, and
  the two held-out splits (grouped by run, and by time within every run).
- `deadlinerisk.predictor` -- a random-forest miss-rate predictor with an
  across-tree uncertainty output combined in quadrature with the binomial
  target noise, plus the constant-rate null model and the scoring functions.
- `deadlinerisk.__main__` -- `python -m deadlinerisk` with the `manifest`,
  `account`, `headline`, `evt` and `learned` subcommands.
- `tests/fixtures/synthetic_trace.py` -- a committed deterministic generator
  that reproduces the campaign loop's release and drop logic exactly.
- `validation/` -- evidence scripts, each anchored on `__file__`, each setting
  `sys.dont_write_bytecode = True` first, each runnable from any working
  directory, with its raw output committed beside it.
- `examples/` -- runnable scripts writing PNGs into `screenshots/`.
- The measurement script, verbatim as it ran on the board, at
  `data/campaign.py`, and the campaign it wrote, committed whole in
  `data/traces/`: 10 runs, 190 662 executed jobs over 195 010 releases, a 1 Hz
  contention monitor beside each run, `manifest.json` carrying the calibration
  history and the pre-declared probe table, and the `CAMPAIGN_COMPLETE` marker.
  The library, the tests and the prose were built before any trace existed, with
  every measured figure marked `MEASURED_PENDING`; the measured validation
  scripts and all four examples exited non-zero until the traces landed, rather
  than substituting the committed synthetic generator, and the tokens were then
  replaced from those scripts' output.
- `validation/validate_descriptor_leak.py`, added after the first measured
  scoring showed that the learned predictor's dominant descriptor is the task's
  own CPU occupancy, which the specification does not license. It refits the
  same forest with the same seed and the same splits on two narrower descriptor
  sets and reports what the restriction costs.
