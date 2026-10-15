# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-10-15

First release. Status: **TESTING**. Validation level 3, research grade.

### Added

- **Declarative problem model** (`activityplan.model`): activities with
  piecewise-constant power profiles, declared data volume, priority and time
  window; a resource model with battery capacity, a depth-of-discharge limit, an
  analytic eclipse-aware solar profile with a closed-form energy integral, a
  solid-state recorder and declared contact windows with rates and transmitter
  power; precedence and exclusion constraints; an optional
  `require_empty_recorder` operations-concept constraint.
- **Discretisation** (`activityplan.timeline`): one rasteriser consumed by both
  planners and the simulator, so they cannot disagree. Energy and data are exact
  per step; occupancy rounds up; windows tighten inward; the horizon must be an
  integer multiple of the step or the grid is refused.
- **Resource simulator** (`activityplan.simulate`): battery with a shunt clamp
  at capacity and no clamp at the floor, recorder with maximal drain, and a
  stated dominance lemma that lets a MILP plan be verified by the simulator.
- **The feasibility report** (`activityplan.feasibility`): slack on every
  declared constraint in its own unit and normalised by a declared scale, the
  binding constraint named, diagnostics that cannot be violated kept out of the
  comparison, and a closed-form energy audit.
- **Baseline 1** (`activityplan.greedy`): priority-ordered earliest-feasible-start
  insertion over a topological order of the precedence graph, with a rejection
  record naming the binding constraint of every refused start.
- **Baseline 2** (`activityplan.exact`): a time-indexed MILP over
  `scipy.optimize.milp` (HiGHS), plus an exhaustive enumerator used as an
  independent reference.
- **Learned component** (`activityplan.features`, `.predictor`), implemented
  last: a random-forest predictor of activity duration and energy with a
  split-conformal prediction interval in a constant-width and a locally weighted
  variant.
- **Warm start** (`activityplan.warmstart`): a verified primal-bound cut, with
  the simulator gating the bound, since `scipy.optimize.milp` exposes no MIP
  start.
- **Robust planning** (`activityplan.robust`): four planning modes — oracle,
  analytic, learned point, learned robust — scored on identical instances
  against a shared realisation.
- **Five hand-declared instances** and a seeded random family sized against the
  sampled demand and a computed worst-case drawdown.
- **CLI** `python -m activityplan` with `instances`, `declare`, `plan`,
  `compare`, `predict`, `robust` and `warmstart`.
- **217 tests**, including property-based tests with Hypothesis and pinned
  regression tests for every counterexample Hypothesis produced.
- **Seven validation scripts** with committed raw output, and seven examples
  producing committed figures.

### Measured, and reported as measured

- Greedy reaches **95.11 %** of the exact optimum on average over 24 seeded
  instances, **85.00 %** at worst, and ties it on **9 of 24**.
- The MILP optimum agrees with exhaustive enumeration on **9 of 9** small
  instances, worst objective difference **0.000e+00**.
- The battery energy balance closes to **2.274e-13 W\*h** over 21 traces.
- The learned predictor beats the analytic model (MAE ratio **0.9249** on
  duration, **0.4116** on energy) because the analytic model omits two terms the
  declared generator contains.
- The conformal interval attains marginal coverage on **12 of 12** combinations
  and **fails conditional coverage**, covering the long duration tercile
  **0.6782** against a nominal 0.90.
- **The learned warm start does not help the solver**: 31 of 34 cold solves have
  a node count of 1, so there is no search to prune.
- **The robust planning mode is worse than the point mode** (0.4167 against
  0.4722 realised feasibility), because booking a longer duration books a lower
  data production rate and the recorder then overflows.

### Fixed during the build

- Aggregate downlink volume was treated as a hard constraint in the feasibility
  report and not in the MILP, so the two disagreed by 1.5 priority units on a
  four-activity instance. It is now a diagnostic, and
  `Problem.require_empty_recorder` provides the stronger end-of-horizon
  constraint when an operations concept needs it. Pinned in
  `tests/test_regressions.py`.
- The first draft of the `energy-bound` instance did not admit the empty plan:
  its fixed loads alone broke the depth-of-discharge floor by 1.44 W\*h.
  `solve_exact` now names the empty-plan binding constraint when a model is
  infeasible, and the condition is tested for every shipped and sampled
  instance.
- The claim that the greedy planner always returns a feasible plan was
  unconditional and false; Hypothesis falsified it at `peak_power_w = 20.0`,
  `max_depth_of_discharge = 0.1`, `bus_load_w = 21.0`. The claim is now
  conditional on the empty plan being feasible, in the docstring and in a pinned
  regression test.

### Known limitations

See the Limitations section of `README.md`. The principal ones: start times are
restricted to grid points so the reported optimum is that of the discretised
problem; the MILP refuses a charge efficiency below one; there is no peak-power
constraint; and every dependency is pinned to the minor version actually
executed, which is a real cost recorded rather than resolved by removing the
ceiling.
