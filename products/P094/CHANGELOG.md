# Changelog

All notable changes to this product are recorded here. The format follows
Keep a Changelog, and versions follow semantic versioning.

## 0.1.0 - 2026-11-16

First release. Status: TESTING.

### Added

- `src/hiltimingdiff/` -- per-stage timing divergence between the
  simulation-mode and hardware-mode execution of one six-stage PAT pipeline.
  `sources` locates, hashes and imports the committed board programs;
  `traces` loads the hardware campaign and refuses rather than substituting;
  `simulate` times the committed pipeline in this container with the committed
  instrument; `baselines` implements the three baselines of SPEC §4;
  `divergence` computes the five headline outputs of SPEC §5; `bootstrap`
  resamples at the declared pass unit; `drift` fits the board's drift and
  produces the drift-adjusted variant; `frequency` measures the clock
  co-variation SPEC §6 requires; `gate` counts the branch reachability SPEC §9
  requires; `report` assembles the headline; `__main__` is the CLI.
- `tests/` -- 1209 tests, counted from `junit.xml`.
- `validation/` -- fourteen re-runnable evidence scripts and their captured
  output, plus `VALIDATION.md`.
- `examples/` -- three runnable scripts, each writing a PNG to `screenshots/`
  and its stdout to a committed `_output.txt`.
- `data/` -- the four measurement programs byte-identical to the copies that
  executed on the Jetson Orin Nano, with their md5 sums recorded and asserted,
  and `data/traces/` with the completed campaign.

### Measured findings in this release

- SPEC §7's primary pre-registration is **CONFIRMED**: the divergence spread is
  1114.8633602088898 against a pre-registered 3.
- SPEC §7's secondary pre-registration is **REFUTED**: baseline 2's held-out
  transfer error is 0.1599 against a pre-registered above 0.5. The baseline
  that was expected to lose wins, and is published as winning.
- SPEC §7's stated mechanism is partly wrong about `threshold`, and the error
  is published.
- SPEC §4's stated reason for the pass resampling unit is falsified on this
  campaign while its choice of unit stands.
- SPEC §8's decorrelation of execution order from frame dimension was not
  achieved in this draw: `corr(order_index, W)` is 0.5011614417507341.
- SPEC §8 says ten calibration blocks; the committed campaign holds eleven.
- Board drift over the campaign: 1.0752408447278534, step-like rather than
  linear.
- Frequency co-variation: at most 0.008537090338330722 of per-stage variance,
  a measured null with a stated sampling limit.
- Gate branch reachability: all five branches fire in all nine cells.

### Corrected after coordinator review

- `tests/test_properties.py::test_ordering_is_invariant_under_a_positive_rescaling`
  asserted a property that is false in float64 and failed intermittently -- once
  in six full-suite runs in the coordinator's re-run, then five clean runs. A
  positive rescaling cannot invert an order but can destroy distinctness, and
  the `assume` guarded the input while the loss happens in the output. Three
  routes reach it: a plain rounding collision (the one that actually fired here,
  all values finite and ordinary), overflow to a shared `inf`, and underflow to
  a shared subnormal. Guards added on the rescaled array; all four
  counterexamples pinned; every other invariance property audited, one more of
  the same shape found safe and its reason pinned. No published figure is
  affected and two tests now assert that.
- `validation/probe_working_directories.py` was overwriting all twelve committed
  validation outputs with its own reduced `HILTIMINGDIFF_FAST=1` runs. Fixed
  with `HILTIMINGDIFF_OUTPUT_DIR`; the probe now verifies the redirection. Its
  wall time also fell from 493 s to 232 s at unchanged coverage, against a 600 s
  release-gate timeout.

- `drift.confounding` computed Spearman's rho with `numpy.argsort(numpy.argsort(x))`,
  an ordinal rank that breaks ties by position, on a series with five tied
  values. The published figure was 0.5333333333333333 and the correct
  tie-averaged-midrank value is 0.5809475019311126, matching
  `scipy.stats.spearmanr`. The ordinal form was also not invariant under a
  permutation of the rows: 0.85 in execution order against 0.5333333333333333
  in alphabetical order, from identical data. Fixed with dependency-free
  `midranks` and `spearman_rho` in `src/hiltimingdiff/drift.py`. The conclusion
  it supports is unchanged in direction and strengthened in magnitude.

### Fixed during the build

- `kendall_tau_b` decided concordance on the product of two differences, which
  underflows to exactly `0.0` in float64 for differences of order `1e-163`,
  returning `0.0` where the answer is `1.0`. Found by a property test, fixed to
  a product of signs, counterexample pinned as a regression test. The measured
  tau-b is unchanged.
