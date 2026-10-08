# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-10-08

First release. Status **TESTING**: the test suite and the validation scripts
pass in the build container and the numbers below are committed, but the
package has not been reviewed by anyone other than its author.

### Added

- **Requirement language and robustness semantics** (`requirements`,
  `traces`): bounded `Always` and `Eventually`, `And`, `Or`, and predicates on
  signals, their first differences and the magnitude of either. A second,
  independently implemented Boolean semantics, so that the sign agreement can be
  tested rather than assumed. No negation operator, deliberately, so the
  equivalence is exact at the boundary.
- **Simulator** (`systems`): a synthetic single-axis attitude-hold loop with a
  first-order actuator, a slew-rate limit, a deflection limit and a sinusoidal
  gust, integrated by explicit Euler, plus an exact zero-order-hold solution of
  the unsaturated linear loop for validation.
- **Benchmark suite** (`instances`): eight seeded instances sharing one declared
  six-dimensional box, spanning measured uniform-violation probabilities from
  0.321933 to 0.001267.
- **Five search strategies** (`search`): uniform random — the baseline, written
  first — Latin hypercube, simulated annealing, cross-entropy, and a
  random-forest surrogate-guided search.
- **Learned component** (`surrogate`): a random-forest regression of robustness
  with a tree-dispersion uncertainty output and a lower-confidence-bound
  acquisition.
- **Sample-efficiency curves** (`curves`, `benchmark`): right-censored empirical
  curves, pointwise and stratified percentile bootstrap bands, exact binomial
  intervals for instance difficulty, and a per-instance accounting that names
  the instances where the baseline wins.
- **CLI** (`python -m falsifyloop`): `instances`, `falsify`, `benchmark`,
  `evaluate`, `difficulty`. Exit status 0 means the command ran and never that
  anything was or was not found.
- **282 tests**, including a Hypothesis property suite for the sign agreement.
- **Seven validation scripts** with their raw output committed, and six
  plotting examples with their PNGs.
- `docs/REQUIREMENTS.md` with 73 numbered requirements, each mapped to the test
  or validation script that exercises it (validation level 3).
- `MODEL_CARD.md` and `DATASET_CARD.md`.

### Published negative results

Reported because they are the result, not in spite of it. Full detail in
`validation/VALIDATION.md`.

- **Latin hypercube never beats uniform random on any of the eight instances**,
  and loses resolvably on two.
- **Cross-entropy loses resolvably on six of eight** and found nothing at all in
  30 runs of 100 simulations on the hardest instance, with a best robustness of
  +0.0213 across all 3000 simulations.
- **Simulated annealing loses resolvably to uniform random on all four
  easy-to-moderate instances** and wins resolvably on only the hardest one.
- **The surrogate's advantage is confined to the hard instances.** It does not
  win resolvably on any of the four easiest, its one point-estimate loss is on
  `settling-band`, and on the easiest instance all thirty of its violations came
  during its unlearned warm start.
- **The surrogate's uncertainty output is not calibrated**: nominal-95 coverage
  0.9356, nominal-68 coverage 0.7631 against a Gaussian's 0.6827 — conservative
  in the body, thin in the tails.
- **7.2 % of counterexamples do not survive a four-times-finer integration
  step.**
- **14 of 32 per-instance comparisons are statistically undecided at 30 seeds**,
  so a single win/loss table is not a settled result.

### Known limitations

See the README. The largest are that the requirement language is a small
fragment and `rtamt`'s is far more complete; that the simulator is a synthetic
benchmark with no parameter identified from any vehicle; and that falsification
is one-sided, so finding no violation is not evidence of correctness.

[0.1.0]: https://github.com/OmAcharya-avtr/falsifyloop/releases/tag/v0.1.0
