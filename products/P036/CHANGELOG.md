# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-04

First release. Validation level 1 (educational to research grade). Status:
TESTING.

### Added

- `rtclock.units` - seconds as the single canonical unit, with exact decimal
  conversions to and from ms, us and ns, and period/frequency reciprocals.
- `rtclock.timebase` - `MonotonicTimebase` over `time.monotonic_ns`,
  `SimulatedTimebase` (deterministic), `SkewedSimulatedTimebase`
  (deterministic with an injected sleep-primitive rate error and wake-up
  delay), and `measure_clock_resolution`, which measures the observable tick
  and derives the duration error term `q` (worst case) and `q/sqrt(6)`
  (standard, per JCGM 100:2008).
- `rtclock.taskset` - `PeriodicTask` and `TaskSet` with rate-monotonic and
  deadline-monotonic priority assignment, utilization and density, and an
  integer-grid hyperperiod.
- `rtclock.schedulability` - Liu & Layland RM utilization bound
  `n(2^(1/n)-1)`, the hyperbolic bound, the EDF test (exact for implicit
  deadlines, density-only for constrained), response-time analysis by the
  Audsley et al. fixed-point recurrence with the full iterate sequence
  retained, and priority-ceiling blocking per Sha, Rajkumar & Lehoczky.
  Every result carries `strength` in `{sufficient, necessary, exact}`.
- `rtclock.histogram` - `LatencyHistogram` storing every sample so that
  percentiles are exact, with two explicitly named definitions
  (`nearest_rank` and `linear`/Hyndman & Fan type 7) and no default that
  could change between releases; strict-inequality overrun accounting with
  index list, longest consecutive run and worst overshoot.
- `rtclock.loop` - `FixedRateLoop` with absolute and relative wait
  computation, reporting per-iteration drift against the ideal schedule
  rather than absorbing it, plus the closed-form drift expressions for both
  modes.
- `rtclock.budget` - stage-chain composition: worst-case sum, mean sum, the
  independent-stage quadrature sigma, the fully-correlated upper bound, and
  the clock-quantization term reported separately from real jitter.
- CLI `python -m rtclock` with `resolution`, `bound`, `analyse`,
  `percentile` and `drift` subcommands.
- 252 tests including Hypothesis property tests for the algebraic
  identities; six validation scripts with their committed raw output.
- Four examples, each producing a PNG in `screenshots/`.

### Known limitations in this release

- The response-time recurrence is the exact worst case only while
  `R_i <= T_i`; beyond that the level-i busy period must be examined
  (Lehoczky 1990). The package reports the recurrence value, flags the
  deadline miss, and says so.
- The EDF test for constrained deadlines is the density test, which is
  sufficient only. The exact processor-demand criterion is not implemented.
- The nearest-rank percentile inherits the binary64 representation of `p`:
  `p = 99.9` with `N = 20000` yields rank 19981 rather than 19980. No
  tolerance is applied to hide this; it is documented and pinned by a test.
- The P031/P036 overrun cross-check is over a synthetic injected trace, not
  a hardware measurement. It agrees on both published definitions (176 direct
  and 307 cascade overruns over P031's 1200-sample trace, with identical
  index lists), which establishes implementation agreement and nothing about
  hardware.
