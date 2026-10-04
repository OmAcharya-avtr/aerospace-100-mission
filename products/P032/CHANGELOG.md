# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-10-04

First release. Status: **TESTING**. Validation level 3. Research-grade; not
flight-qualified, not certified, not approved for operational aerospace use.

### Added

- **Propagation and constellation definition** (`constellation.py`): TLE
  parsing with modulo-10 checksum validation, `Satellite` wrapping an `sgp4`
  `Satrec`, Walker delta generation (`i: T/P/F`), ground-station definition,
  vectorised ephemeris, and an analytic two-body `CircularOrbit` used only as
  an independent reference in validation.
- **Epoch guarding**: `TleEpochError` is raised when any sample of a requested
  window lies further from the element epoch than `max_epoch_age_days`
  (default 7). SGP4 returns a position with no error code however far from
  epoch it is asked, so the guard is the only thing preventing a silent wrong
  answer. `check_epoch=False` is an explicit escape hatch.
- **Geometry** (`geometry.py`): exact point-to-segment Earth-limb clearance,
  the closed-form maximum ISL central angle, the ground-access central angle
  and the slant range at a given elevation.
- **Contact windows** (`contacts.py`): coarse scan plus bisection refinement
  that re-propagates rather than interpolating, so edge accuracy is set by the
  tolerance and not by the grid step. Windows record their own grid step and
  whether they were clipped by the horizon.
- **Time-varying contact graph and its time expansion** (`graph.py`):
  per-slot adjacency, connected components, partition detection, mid-horizon
  node loss, and a layered DAG with hold and transmit edges whose capacity
  comes from a caller-supplied rate function.
- **Routing** (`routing.py`): Dijkstra over the time-expanded graph, plus an
  exhaustive enumerator kept as an independent reference for small instances.
- **Scheduling** (`flow.py`): integer maximum flow on the time-expanded
  network, solved through `scipy.optimize.milp` (HiGHS), with the same program
  also built as a `pulp` model and checked structurally against it, and a
  brute-force enumerator as the cross-check.
- **Capacity** (`capacity.py`): an RF chain (Friis, C/N0, Shannon bound and an
  Eb/N0-driven achievable rate) and a free-space-optical chain (Gaussian
  far-field capture, pointing loss, photons-per-bit sensitivity), plus the
  Kim-Kruse visibility model and a plane-parallel slant-path scaling that
  refuses below its validity elevation.
- **Queueing and delay accounting** (`queueing.py`): M/M/1 and M/D/1 mean
  delay, and a per-hop end-to-end delay breakdown over a route.
- **Link-availability prediction** (`synthdata.py`, `availability.py`,
  `metrics.py`): a deterministic synthetic dataset, a climatology baseline, a
  logistic-regression baseline, a bagged and Platt-calibrated gradient
  boosting model with an ensemble uncertainty output, and a verification
  harness with the Brier score, the Murphy decomposition, ECE, reliability
  curves with Wilson intervals and bootstrap intervals.
- **Uncertainty analysis** (`uncertainty.py`): along-track perturbation to
  window-edge shift, grid-step convergence, and Monte Carlo over terminal
  parameters.
- **Benchmark harness** (`benchmark.py`): wall-clock timing with the
  measurement method and environment recorded in the output.
- **CLI**: `python -m constellink {contacts,route,flow,capacity,predict,benchmark}`.
- 387 tests, 10 validation scripts with raw output committed, 5 runnable
  examples producing 5 figures, `docs/REQUIREMENTS.md` with 18 numbered
  requirements and a verification matrix.

### Measured results worth stating in a changelog

- SGP4 reproduces the verification data shipped with the `sgp4` package to
  6.819e-09 km in position over the full 4320 min span of satellite 00005.
- Dijkstra agrees with exhaustive enumeration to 0.000e+00 s over 40 random
  instances; the ILP agrees with brute force on all 27 enumerable instances.
- **The logistic-regression baseline is better calibrated than the learned
  model** (reliability 0.00260 ± 0.00151 against 0.00516 ± 0.00284 over five
  grouped splits) and has a better Brier score. On the smaller pinned
  regression split the climatology baseline has the lowest reliability term.
  Both results are kept and both are asserted by tests.

### Known limitations at this release

- **No `pulp` objective value was produced.** `pulp` 4.0.0 ships no bundled
  CBC binary and no external MILP solver is installed in the build
  environment, so the `pulp` model is verified structurally only and the
  solved objective comes from HiGHS. Documented in README.md,
  `docs/REQUIREMENTS.md` (REQ-09) and `validation/VALIDATION.md` section 5.3.
- Walker generation sets a Kepler mean motion where SGP4 expects a Kozai mean
  motion, so the realised mean radius is 0.113-0.127 km below the requested
  one. Measured, documented, not corrected.
- All machine-learning results are self-consistency against this package's own
  capacity models, not agreement with measured link outages.
- Every timing figure is a single-core container measurement.
