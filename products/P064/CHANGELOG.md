# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-15

Initial release. Status: TESTING.

### Added

- `regressselect.importgraph` -- Baseline 1, dependency-based selection from an
  `ast`-parsed import graph, with hop distances, selection, and
  `with_conftest_coupling()` for the fixture dependency pytest imposes and a
  pure import graph cannot see.
- `regressselect.prioritise` -- Baseline 2, smoothed co-failure prioritisation
  over a commit history.
- `regressselect.selector` -- the learned selector: gradient-boosted trees over
  eleven change features, producing a ranked order with a per-test probability
  and a Brier score against held-out revisions.
- `regressselect.metrics` -- APFD exactly as published in Rothermel, Untch, Chu
  and Harrold 2001, safety loss, selection cost and detection curves. APFD
  raises rather than substituting a value when a fault is detected by no test.
- `regressselect.mutation` -- seeded-fault harness: applies each declared
  mutation to a throwaway copy of the bundled sample project, runs pytest, and
  reads the outcome from the junit XML with a `tests`-against-`testcase`
  cross-check.
- `regressselect.benchmark` -- three-strategy comparison on a committed history
  and on a measured fault matrix.
- `python -m regressselect` with `graph`, `select`, `benchmark`, `mutate` and
  `apfd` subcommands.
- Bundled sample project (`fixtures/sample_project`, 7 source modules, 24 test
  nodes) with a committed clean `junit.xml`, 14 declared seeded faults and the
  measured fault matrix.
- Committed 320-revision synthetic history (`fixtures/history/history.json`,
  seed 20261015) with its generation parameters beside it.
- Three examples writing PNGs to `screenshots/`; eight validation scripts with
  their committed output in `validation/`.

### Measured, and reported as measured

- The dependency baseline wins. On the held-out revisions it has the highest
  APFD (0.836149 against 0.814752 learned and 0.795608 history) and, at matched
  cost, the lowest safety loss (0.172072 against 0.282883 learned and 0.299399
  history). On the seeded faults it detects every detectable fault while
  running 0.2589 of the suite.
- The learned selector beats the history baseline on APFD and on safety loss at
  every budget, so it is not a no-op, but it does not beat Baseline 1.
- The full-suite seeded-fault detection upper bound is 0.714286: four of the
  fourteen seeded faults are detected by no test in the sample project.
