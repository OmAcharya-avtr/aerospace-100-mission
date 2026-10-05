# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-05

First release. Status: TESTING.

### Added

- Fault taxonomy with sixteen kinds across five classes (sensor, actuator, bus,
  timing, numerical), each with units, declared parameter ranges, a coverage
  binning and a literature reference. 248 coverage cells in the cross product.
- Executable handler for every kind, with all randomness drawn from an injected
  `numpy.random.Generator` so that a case replays bit-identically from its seed.
- `InjectionWrapper`: injection around a target through its own call interface,
  with no modification to the target. Bit-transparent when no fault is active.
- `NumericalMonitor`: boundary surveillance for NaN, infinity, subnormal and
  large magnitudes, reporting `propagated`, `absorbed`, `emitted` or `absent`,
  so a target that silently swallows a NaN is reported rather than passed.
- Reference GNC target: double-integrator plant, fixed-gain Kalman filter at the
  Riccati fixed point, saturating PD controller.
- Severity scoring from the target response, relative to the fault-free run of
  the same seed, with the weights and reference scales stated as module
  constants.
- Coverage accounting over the taxonomy cross product, with per-kind breakdown
  and rejection of out-of-subset cases.
- Campaign cases with a content-hash identifier, canonical JSON serialisation
  and tamper detection, plus deterministic case pools covering every cell.
- Two classical search baselines (uniform random, coverage greedy), a non-learned
  kind-mean ablation, and a learned random-forest campaign prioritiser with an
  ensemble-spread uncertainty output.
- Same-budget benchmark with percentile-bootstrap and paired-bootstrap intervals
  and the overlap decision rule.
- Command-line interface: `taxonomy`, `cells`, `run`, `replay`, `campaign`,
  `benchmark`.
- 202 tests, including Hypothesis property tests and recorded-severity
  regression tests; six validation scripts with committed raw output.

### Known limitations

See the Limitations section of README.md. In short: one fault per case, one
reference target, a finite enumerable benchmark pool, a severity function whose
weights are a design choice, and an uncertainty output that is wide rather than
calibrated.
