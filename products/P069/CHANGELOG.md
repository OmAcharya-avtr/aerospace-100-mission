# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-16

First release. Validation Level 2, status TESTING.

### Added

- `scenariocov.space`: declared scenario spaces with categorical and
  discretised continuous parameters, integer level encoding, a normalised
  embedding, and `edge_ambiguities` for records sitting on a bin edge.
- `scenariocov.combinatorial`: t-way coverage with **every uncovered tuple
  enumerated**, a closed form for `|T_t|`, a per-parameter-subset breakdown,
  and a `max_tuples` guard against combinatorial blow-up.
- `scenariocov.classes`: equivalence-class and boundary coverage against
  declared classes, with the nearest approach reported beside every boundary
  hit flag.
- `scenariocov.occupancy`: grid-cell occupancy, enumerated empty cells,
  connected empty blocks by flood fill, and the classical occupancy
  expectation as a uniform-random reference.
- `scenariocov.report`: the assembled campaign report, which states the
  discretisation beside every number it prints.
- `scenariocov.gapfinder`: the learned gap finder, a Gaussian kernel density
  estimate over executed scenarios, with a bootstrap uncertainty output
  (`log_density_sd`, `sparsity_agreement`).
- `scenariocov.strategies`: uniform random, greedy max-min distance, the
  learned gap finder, and a coverage-greedy reference ceiling.
- `scenariocov.benchmark`: the paired seeded coverage-per-scenario benchmark
  on two metrics, reporting the spread and the win/tie/loss split.
- `scenariocov.plotting`: three Agg figures, produced by the three examples.
- `python -m scenariocov` with `report`, `uncovered`, `occupancy`, `propose`
  and `benchmark`.
- Fixtures: a synthetic 240-scenario rendezvous campaign with five planted
  gaps and a 400-scenario candidate pool, regenerable at seed 20261016.
- 255 tests, counted from junit XML; eleven validation scripts with their raw
  output committed.

### Measured, and not changed afterwards

- The learned gap finder is **not separated** from the greedy max-min
  baseline: paired mean difference +0.004204 with a standard deviation of
  0.015119 over 60 seeds, losing on 20 of them.
- On the grid-occupancy metric the learned gap finder is **behind uniform
  random**, -0.001273, losing on 30 of 60 seeds.
- The deterministic coverage-greedy reference beats all three strategies
  under test on every one of 60 seeds, +0.038127 against the learned model.

### Known limitations

See README.md, Limitations. The principal ones: every figure is a figure
about the declared discretisation; the fixtures are synthetic; the confidence
output is not calibrated; and the dependency ceilings exclude the next minor
release of numpy, scikit-learn and matplotlib.
