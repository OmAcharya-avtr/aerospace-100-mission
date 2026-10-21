# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-10-21

First release. Status: TESTING. Validation level 3, research grade.

### Added

- `downlinkpick.inventory`: `DataProduct` and `Inventory`, with size in bytes
  and on the link in bits, expiry as a hard deadline, dependency closures, and
  rejection of duplicate names, unresolved dependencies and dependency cycles.
- `downlinkpick.contacts`: `ContactWindow` with a piecewise-linear rate profile,
  the capacity integral in closed form, its inverse by bisection, and
  `ContactPlan`, which rejects overlapping windows because one transmit chain is
  assumed.
- `downlinkpick.selection`: the selection problem, the realised-value equation,
  and an invariant checker covering no double downlink, expiry, capacity,
  dependency closure, precedence by window and value conservation.
- `downlinkpick.greedy`: the operational baseline — bundle value-density
  ordering with two window-packing rules and whole-bundle rollback — plus
  Dantzig's fractional upper bound.
- `downlinkpick.exact`: the multiple-knapsack MILP, emitted both as a PuLP
  problem and as SciPy coefficient arrays from one model object, solved with
  `scipy.optimize.milp` (HiGHS), with exhaustive enumeration as a solver-free
  reference and an LP relaxation bound.
- `downlinkpick.marginal`: the marginal value of contact time by finite
  difference on the integral optimum, capacity sweeps, and the left-behind
  report with four reasons and a shortfall in bits and seconds.
- `downlinkpick.valuemodel` and `downlinkpick.compress`: two random-forest
  components with per-tree dispersion as an uncertainty output, each benchmarked
  against the classical baselines it has to beat.
- `downlinkpick.dataset`: the seeded synthetic generator, with its equations in
  the module docstring.
- `downlinkpick.benchmark`, `downlinkpick.stability`, `downlinkpick.report`,
  `downlinkpick.cases`, and a CLI with `select`, `compare`, `marginal`,
  `stability`, `lp` and `info`.
- Eleven validation scripts with their raw output, eight example figures, and
  300 tests including property-based tests and 15 pinned counterexamples.

### Measured and reported as found

- **The declared-value passthrough beats the learned value model when
  declarations are complete and fresh**, 0.000404 against 0.009582 in regret
  fraction. The learned model wins once declarations are stale or absent. This
  is the specification's prediction and it is published as the result.
- **The learned compressibility predictor beats every baseline**, delivering
  0.962760 of the oracle's value against 0.900303 for the per-instrument
  constant and 0.827558 for the entropy floor.
- **The marginal value of contact time can be negative.** A window's end is its
  delivery epoch, so extending a pass can push a product past its deadline; a
  12.89 per cent drop was measured and is pinned as a regression test.
- **Greedy matches the proved optimum on 17 per cent of the tight instance
  family** and is more than 20 per cent off on 2 per cent of it.

### Known limitations

See `README.md`. The dominant ones: every number is measured against a synthetic
generator and not against flight data; the rate profile is a shape and not a
link budget; PuLP has no solver in the measured environment, so the MILP is
modelled with PuLP and solved with HiGHS through SciPy.

[0.1.0]: https://github.com/OmAcharya-avtr/downlinkpick/releases/tag/v0.1.0
