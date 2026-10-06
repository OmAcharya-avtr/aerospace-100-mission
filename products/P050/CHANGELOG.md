# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-06

First release. Validation level 2 (research grade). Status: TESTING.
No machine learning: this is a deterministic optimisation library.

### Added

- `coderateopt.fade` - marginal fade models as availability functions:
  `LognormalFade` (weak turbulence, with both the `E[I] = 1` and
  `median(I) = 1` normalisations exposed rather than defaulted silently),
  `GammaGammaFade` with `from_scintillation` inverting
  `sigma_I^2 = 1/alpha + 1/beta + 1/(alpha beta)` and a survival function by
  adaptive quadrature taken over the shorter side of the mode, and
  `EmpiricalFade` with an explicit `1/n` resolution floor. Scintillation-index
  conversions, dB conversions and a bisection quantile shared by all three.
- `coderateopt.modcod` - `Modcod` and a canonically ordered `ModcodSet`
  (ascending threshold, then descending rate, then name), so every tie-break
  downstream is reproducible; plus an illustrative nine-entry table labelled
  as illustrative wherever it appears.
- `coderateopt.problem` - the optimisation written down: decision variables,
  objective, cardinality, minimum dwell, and both readings of the availability
  constraint, with the four assumptions that bound the answer and two
  structural results about when an optimiser is needed at all. `Solution`
  carries the worst-interval availability and the tied supports, not just the
  optimum.
- `coderateopt.lp` - the restricted linear programme on a fixed support,
  shared by both solver paths.
- `coderateopt.milp` - the MILP encoding with inspectable arrays and row
  labels, solved by `scipy.optimize.milp`. Sets `mip_rel_gap = 0.0`, and
  re-solves the continuous part on the chosen support so the answer sits
  exactly on the availability row.
- `coderateopt.exhaustive` - subset enumeration with a plain LP per subset, a
  solver-free closed form for two-entry mixes, canonical tie-breaking and
  explicit tie reporting.
- `coderateopt.select` - `select_rate`, which solves with the MILP,
  canonicalises by enumeration when that is cheap, and raises rather than
  choosing a side if the two disagree.
- `coderateopt.sensitivity` - stability intervals of the decision in the
  scintillation index, the margin or the target, found by a coarse scan then
  bisection, with knife-edge and flat verdicts and explicit flags for an
  endpoint that is only a scan bound.
- `coderateopt.heuristics` - the two rules of thumb this package competes
  with, and a comparison record that distinguishes a goodput loss from a
  constraint violation.
- CLI `python -m coderateopt` with `solve`, `sensitivity`, `compare`, `table`
  and `fade`, a CSV MODCOD table reader, and distinct exit statuses for bad
  input (2) and an infeasible instance (3).
- 152 tests including Hypothesis property tests, three-way solver agreement on
  random instances, and known-answer tests whose arithmetic is shown in the
  comments. Nine validation scripts with their committed raw output; four
  examples, each producing a PNG in `screenshots/`.

### Known limitations in this release

- The greedy "fastest MODCOD that meets the target" rule matched the optimum
  at 429 of 430 grid instances on the shipped table. The case for this package
  is the constraint check, the sensitivity interval, the availability-mode
  distinction and the degenerate-case handling, not the optimisation, and the
  README says so first.
- In `per_interval` mode the optimum is provably a single MODCOD, and in
  `long_run` mode with no minimum dwell provably at most two, so the integer
  variables do no work on the base formulation; 909 feasible random instances
  produced no counterexample. They matter only when `min_dwell_fraction > 0`,
  where 13 of 1922 targeted instances had a three-entry optimum.
- Goodput is the outage approximation: delivered above threshold, lost below.
  No waterfall, no finite-blocklength correction.
- Fade statistics are marginal only. No correlation time, fade duration,
  level-crossing rate, interleaver depth or feedback delay. ITU-R P.1623 is
  named as the methodology for fade dynamics and is not implemented.
- Switching between MODCODs is assumed free and instantaneous, and the channel
  state is assumed perfectly known.
- The sensitivity scan can miss a signature change narrower than one grid
  step; `grid_points` is reported so the step size is known.
- `is_knife_edge` uses a relative perturbation, which is natural for a
  scintillation index and weak for a margin in dB.
- `GammaGammaFade` costs an adaptive quadrature per evaluation, roughly four
  orders of magnitude more than the lognormal model.
- The shipped MODCOD thresholds are round illustrative constants, not
  measurements and not any standard's values.
