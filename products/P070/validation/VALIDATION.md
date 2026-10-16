# Validation evidence — watchdogspec 0.1.0

Validation level 2 (research grade). Every number in this document came from
running a script in this directory on **2026-10-16** on the machine described
in section 7, and the raw output of each run is committed beside the script
as `<script-name>_output.txt`. Nothing here was transcribed by hand.

**What this validates, and what it cannot.** Every check below is about the
*internal consistency of the arithmetic*: that the closed form agrees with an
independent simulation of the same model, that the known answers are the
hand-calculated ones, that the parser rejects what it should. **None of it
validates the model against a real supervisor, because no real supervisor was
measured.** This package reads a declaration and checks it against itself. A
clean report is not evidence that a flight computer behaves as declared, and
no amount of validation of this kind could make it one.

## 0. What was run

| script | what it establishes | wall clock |
|---|---|---|
| `validate_environment.py` | the exact interpreter and library versions | 0.84 s |
| `validate_known_answers.py` | hand-calculated answers for the escalation arithmetic, the planted defect catalogue, the planted tolerance-exceeds-deadline spec | 0.29 s |
| `validate_parser_rejections.py` | every malformed-document path rejected by key path | 0.59 s |
| `validate_properties.py` | algebraic identities over a deterministic sweep of 8100 to 40500 parameter sets | 3.10 s |
| `validate_latency_vs_simulation.py` | closed-form worst case against the seeded discrete-event simulation | 7.97 s |
| `validate_false_trip.py` | the Gaussian tail against an event-driven count, with Wilson intervals | 2.52 s |
| `worked_example.py` | the worked example exactly as the README quotes it | 0.61 s |
| `validate_cli.py` | the exit-code contract, as real subprocesses, and the gitignore sweep | 5.98 s |

Total 22.3 s wall clock for the whole sweep, measured end to end on the two
cores described in section 7.

Every script exits 0 when every check in it passed and 1 otherwise. All eight
exited 0 on the run recorded here.

## 1. The equations, their sources and their validity

All times in milliseconds. Symbols are fixed in
`src/watchdogspec/model.py`: `P` heartbeat period, `J` bound on the deviation
of the receipt-to-receipt gap from `P`, `sigma` standard deviation of the
emission deviation, `k` missed beats tolerated, `D` liveness deadline, `Q`
supervisor poll period, `Jq` poll release jitter, `O` declaration overhead.

| equation | statement | source of the model | validity |
|---|---|---|---|
| (1) | `W = P + J` | definition | `P > 0`, `J >= 0`; meaningless once `J >= P`, which is check `WS008` |
| (2) | last internal expiry at `t_ref + (k+1) W` | the deadline-chain supervisor described in `latency.py` | `k >= 0` integer |
| (3) | `WCDL = (k+1)(P+J) + Q + Jq + O` | elementary composition of declared bounds; the bounded-release model is that of Audsley et al. 1993 and Tindell and Clark 1994 | upper bound everywhere; **tight only while `Jq <= Q`** (section 5) |
| (4) | `WCRL(r) = WCDL + sum of rung latencies 1..r` | definition of a sequential ladder | each rung latency `>= 0`, ladder executed strictly in order |
| (5) | gap `= P + Delta`, `Delta ~ N(0, 2 sigma^2)` | difference of two independent `N(0, sigma^2)` emission deviations | independence and normality assumed, not measured |
| (6) | a false trip occurs when `Delta > k P + (k+1) J` | equations (2) and (5) | the deadline chain resets on every receipt |
| (7) | `p = 0.5 erfc((k P + (k+1) J) / (2 sigma))` | the standard identity `Phi_bar(x) = 0.5 erfc(x / sqrt 2)`, Abramowitz and Stegun 1964 section 26.2 | `sigma > 0`; underflows to 0 beyond about 26.6 sigma |
| (8) | `false trips per hour = p * 3600000 / P` | beats per hour | rare-event approximation, relative error `O(p)` |
| (9) | `MTBFT = 1 / (8)` | definition | reported as `inf` when (7) underflows |

Full references, with DOIs where they exist, are in `CITATION.cff` and in the
README's Citation section. The schedulability arithmetic that decides whether
the supervised tasks can meet their own periods in the first place is a
different question and is **not** addressed here (Liu and Layland 1973; P036
RtClock in this portfolio).

**The two jitter models are deliberately inconsistent.** Equation (3) assumes
the declared bound `J` holds. Equation (7) computes the rate at which an
unbounded Gaussian violates it. A specification that declares both `J` and
`sigma` is declaring a bound and a violation rate for that bound, and this
package reports both numbers side by side rather than pretending one
supersedes the other.

## 2. Known answers (`validate_known_answers.py`)

Every expected number below was hand-calculated before the code was run, and
the calculation is written out in the test comments and in the shipped
specification files.

| case | expected | measured | result |
|---|---|---|---|
| Planted fixture: findings raised | 12, one per check | 12 | pass |
| Planted fixture: code to subject mapping | hand-written table of 12 pairs | identical | pass |
| Planted fixture: severity split | 7 error, 5 warning | 7 error, 5 warning | pass |
| Planted fixture: exit code | 1 | 1 | pass |
| Planted fixture: the clean task carries no finding | 0 findings on `T-CLEAN` | 0 | pass |
| `T-CLEAN` worst-case detection latency | `3 * 22 + 13 = 79 ms` | 79.0000 ms | pass |
| `T-CLEAN` ladder completion | `79 + 172 = 251 ms` | 251.0000 ms | pass |
| Worked example `TASK-GNC` WCDL | `3 * 23 + 6.5 = 75.5 ms` | 75.5000 ms | pass |
| Worked example `TASK-GNC` ladder completion | `75.5 + 612 = 687.5 ms` | 687.5000 ms | pass |
| Worked example `TASK-IMU` WCDL | `2 * 6 + 6.5 = 18.5 ms` | 18.5000 ms | pass |
| Worked example `TASK-TLM` WCDL | `1 * 1250 + 6.5 = 1256.5 ms` | 1256.5000 ms | pass |
| Worked example `TASK-FDIR` WCDL | `4 * 115 + 6.5 = 466.5 ms` | 466.5000 ms | pass |
| Worked example `TASK-FDIR` deepest rung | 1 of 4 (`468.5 ms` fits, `528.5 ms` does not, against `D = 500 ms`) | 1/4 | pass |
| Minimal specification | 0 findings, exit 0 | 0, 0 | pass |
| Worked example | exactly `WS003` on `TASK-FDIR` and `WS005` on `TASK-TLM`, exit 1 | as expected | pass |

### 2.1 The planted tolerance-exceeds-deadline specification

This is the defect the batch specification names, and it is planted in
`examples/specs/planted_defects.yaml` as task `T-WS002`.

| quantity | value |
|---|---|
| heartbeat period `P` | 50 ms |
| jitter bound `J` | 5 ms |
| beat window `W = P + J` | 55.000000 ms |
| missed beats tolerated `k` | 4 |
| tolerated silence `k W` | 220.000000 ms |
| liveness deadline `D` | 200.000000 ms |
| slack `D - k W` | **-20.000000 ms** |
| margin | **missed** |
| finding | **`WS002` error on `T-WS002`** |

Detected. The message names the task, prints the arithmetic and says that the
tolerance exceeds the deadline it is declared to protect. The defect cannot be
repaired by a faster escalation ladder, and the check costs nothing because
both quantities are already declared.

### 2.2 The planted floating-point boundary case

`T-WS004` of the same fixture has a liveness deadline of exactly 229.0 ms,
which is exactly its worst-case ladder completion time.

| quantity | value |
|---|---|
| worst-case completion | 229.0 ms |
| declared deadline | 229.0 ms |
| slack | 0.0 ms |
| comparison tolerance | 1.000e-09 ms |
| margin | **marginal** |
| finding | `WS004` **warning**, not an error |

The package declines to answer whether the ladder completes in time, and says
so. See section 4 for why.

## 3. Closed form against the discrete-event simulation (`validate_latency_vs_simulation.py`)

The simulation generates heartbeat arrival times, places the fault at a
continuous instant inside a beat interval, and steps the supervisor deadline
chain and poll loop event by event. It never consults equation (3) while
running.

### 3.1 The worked example, per task

| task | closed form (ms) | adversarial simulation (ms) | difference (ms) | ulp | stochastic max over 19200 trials (ms) | shortfall (ms) | bound violations |
|---|---|---|---|---|---|---|---|
| `TASK-GNC` | 75.500000000 | 75.500000000 | -1.14e-13 | 8.0 | 75.370093974 | 0.129906 | 0 |
| `TASK-IMU` | 18.500000000 | 18.500000000 | -4.26e-14 | 12.0 | 18.370093974 | 0.129906 | 0 |
| `TASK-TLM` | 1256.500000000 | 1256.500000000 | -3.64e-12 | 16.0 | 1256.370093974 | 0.129906 | 0 |
| `TASK-FDIR` | 466.500000000 | 466.500000000 | -4.55e-13 | 8.0 | 466.370093974 | 0.129906 | 0 |

The adversarial trial agrees with the closed form to between 8 and 16 units
in the last place of a sum of six doubles. It does not agree bit for bit and
it is not claimed to: the simulation accumulates the same sum through a
different sequence of additions.

The shortfall column is the honest one. Over 19200 random trials per task the
largest observed latency is about 0.13 ms **below** the bound, identically for
every task, because the detection latency measured from the last accepted
heartbeat depends on the beat jitter only through `W` and its spread is set by
the supervisor poll phase, which is the same for all four tasks. Attaining the
bound needs the fault to land exactly at a receipt and the poll phase to be
adversarial; both are measure-zero events.

### 3.2 A grid of synthetic declarations

| quantity | value |
|---|---|
| configurations (period x jitter x tolerance x supervisor, with `J < P`) | 400 |
| of which `Jq <= Q`, where the bound is claimed tight | 320 |
| of which `Jq > Q`, where the bound is only an upper bound | 80 |
| seeds per configuration | 4 |
| trials per seed | 200 |
| total stochastic trials | 320000 |
| **bound violations** | **0** |
| largest adversarial difference where `Jq <= Q` | -1.819e-11 ms (10.0 ulp), at `P=1000 J=0.1 k=9 Q=1 Jq=0 O=0` |
| largest shortfall where `Jq > Q` | 4.000000 ms |

## 4. Floating-point discipline (`validate_properties.py`)

Equation (3) can be summed in two algebraically identical orders. They are not
the same number.

| quantity | value |
|---|---|
| parameter sets swept | 8100 |
| largest absolute difference between the two orders | 4.547e-13 ms |
| largest difference in units in the last place | 1.0 |
| **two-valued `slack > 0` booleans that flip under re-association** | **137 of 8100** |
| three-valued verdicts that flip, absolute band only | 0 of 8100 |
| three-valued verdicts that flip, scale-aware band | **0 of 8100** |

At larger magnitudes the absolute part of the band is not sufficient on its
own, and that is measured separately in the same script:

| quantity | value |
|---|---|
| parameter sets swept, periods 1e5 to 1e6 ms, `k` up to 90 | 135 |
| largest re-association error | 1.0 ulp |
| **three-valued verdicts that flip, absolute band only** | **14 of 135** |
| three-valued verdicts that flip, scale-aware band | **0 of 135** |

The comparison band is therefore
`max(tol_ms, 16 * eps * max(|latency|, |deadline|))`, not `tol_ms` alone. See
section 5.3.

A deadline comparison is a step function of a rounded sum, and at the
threshold it is decided by the last bit rather than by the specification. That
is why `watchdogspec.latency.verdict` returns three values and not two, and
why no test in this repository asserts the equality of a boolean produced by a
strict threshold comparison.

The pinned counterexample, with literal inputs, is
`tests/test_regression.py::test_summation_order_flips_a_strict_deadline_comparison`:
`k = 2, P = 3.7, J = 0.1, Q = 2.3` gives `13.7` one way and
`13.700000000000003` the other.

### 4.1 Other identities over the deterministic sweep

| identity | parameter sets | failures |
|---|---|---|
| rung finish times are non-decreasing | 40500 | 0 |
| rung 1 starts at the detection instant | 40500 | 0 |
| the ladder total equals the last rung finish | 40500 | 0 |
| the deepest reachable rung lies in `0..len(rungs)` | 40500 | 0 |
| the deepest reachable rung never falls when the deadline widens | 40500 | 0 |
| `max_tolerance_for_deadline` returns a `k` that fits | 6750 | 0 |
| `max_tolerance_for_deadline` returns the largest `k` that fits | 6750 | 0 |
| every false-trip probability lies in `[0, 0.5]` | 720 | 0 |
| the rate never rises with the tolerance, never rises with the jitter bound, never falls with sigma | 720 each | 0 |
| the verdict agrees with its definition | 36 | 0 |

## 5. What was falsified

Three claims were falsified and all three are reported rather than removed.

### 5.1 Equation (3) is not attained when `Jq > Q`

Found by Hypothesis while checking that the adversarial simulation attains the
closed form. Falsifying inputs, pinned in
`tests/test_regression.py::test_poll_jitter_above_the_poll_period_makes_the_bound_loose`:

```
P = 1.0 ms, J = 0.0 ms, k = 0, Q = 1.0 ms, Jq = 2.0 ms, O = 0.0 ms
equation (3):  (0+1)*1 + 1 + 2 + 0 = 4.0 ms
simulation:                           3.0 ms
```

With a poll release jitter larger than the poll period, the poll released one
period later executes **earlier** than the delayed next poll, so the worst
observation delay is `2Q` and not `Q + Jq`. The bound still holds everywhere;
it is simply no longer attainable, and the tightness claim is therefore
restricted to `Jq <= Q` in the module docstring, in the README limitations and
in the grid of section 3.2, where 80 of 400 configurations sit in that regime
and the largest shortfall is 4.000000 ms.

### 5.2 An absolute comparison tolerance alone is too narrow above 4.5e6 ms

Found by Hypothesis **while running the test suite at the declared dependency
floor**, which is the only place it appeared: hypothesis 6.100.0 explores a
different part of the input space from 6.170.0, and the ceiling run had passed
this property for the whole build. Falsifying inputs, pinned in
`tests/test_regression.py::test_absolute_tolerance_alone_is_too_narrow_at_large_latencies`:

```
P = 100000.0 ms, J = 86413.50000000362 ms, k = 44, Q = 1.0 ms
(k+1)(P+J) + Q        = 8388608.500000162 ms
(k+1)P + (k+1)J + Q   = 8388608.500000164 ms
difference            = 1.862645149230957e-09 ms  = exactly one ulp
absolute tolerance    = 1.0e-09 ms
```

One ulp at that magnitude is nearly twice the absolute tolerance, so the
three-valued verdict flipped from `MET` to `MARGINAL` under a re-association
of the same sum — the behaviour the band exists to prevent. The fix is
`watchdogspec.latency.comparison_band_ms`, which adds a relative part of
16 units in the last place of the larger quantity compared. A 5000-example
Hypothesis run over periods to 1e7 ms and tolerances to 200, at both the
floor and the ceiling hypothesis, found a worst re-association error of
**2.0 ulp**, so the 16-ulp guard has a factor of eight of margin. Both numbers
in the band are chosen, not derived, and the README says so under Limitations.

This defect was in the package's own first draft. It is recorded here because
it is the exact class the portfolio build guide warns about, found by the exact
method the build guide prescribes, and only because both ends of the declared
dependency range were actually run.

### 5.3 The Wilson lower bound at zero successes is not exactly zero

`wilson_interval(0, 1000)` returns a lower bound of `2.168404344971009e-19`,
not `0.0`. It is zero in exact arithmetic; the centre and the half-width are
two separately rounded doubles and cancel only to within rounding. Pinned in
`tests/test_regression.py::test_wilson_lower_bound_at_zero_successes_is_not_exactly_zero`.
It is harmless, and it is in this document because it is the same class of
error as section 4 and was found the same way.

## 6. False-trip rate against the simulation (`validate_false_trip.py`)

Equation (7) is an analytic Gaussian tail; the simulation counts declarations
made by the deadline chain against a task that never fails. Agreement is
assessed with a Wilson score interval (Wilson 1927), chosen over Wald because
the proportions are small.

| configuration | `P` (ms) | `J` (ms) | `sigma` (ms) | `k` | gaps | closed-form `p` | sampled `p` (first seed) | inside 95 % interval |
|---|---|---|---|---|---|---|---|---|
| p ~ 2.4e-1 | 20.0 | 1.0 | 1.00 | 0 | 200000 | 0.239750 | 0.239405 | yes (3/3 seeds) |
| p ~ 4.5e-2 | 20.0 | 2.4 | 1.00 | 0 | 200000 | 0.0448430 | 0.044530 | yes (3/3) |
| p ~ 6.7e-3 | 20.0 | 3.5 | 1.00 | 0 | 300000 | 0.00666416 | 0.00648667 | yes (3/3) |
| p ~ 9.3e-4 | 20.0 | 4.4 | 1.00 | 0 | 400000 | 0.000931423 | 0.000910 | yes (3/3) |
| p ~ 2.0e-4 | 20.0 | 5.0 | 1.00 | 0 | 900000 | 0.000203476 | 0.000207778 | yes (3/3) |
| k = 1 | 10.0 | 1.0 | 5.00 | 1 | 300000 | 0.0448430 | 0.044680 | yes (3/3) |
| k = 2 | 10.0 | 1.0 | 9.00 | 2 | 300000 | 0.0353772 | 0.0351633 | yes (3/3) |

**21 of 21 comparisons placed the closed form inside the 95 % Wilson
interval.** The sweep spans three decades of rate, from one trip in four
heartbeat gaps to one in five thousand. 7.3 million heartbeat gaps were
simulated in total.

The worked example's telemetry task, measured the same way:

| quantity | value |
|---|---|
| trip threshold `k P + (k+1) J` | 250 ms |
| `z = threshold / (sigma sqrt 2)` | 1.964186 |
| closed-form `p` per gap | 0.02475429 |
| closed-form rate | 89.1155 per hour |
| mean time between false trips | 0.011221 hours |
| declared budget | 1 per hour |
| simulated `p` over 400000 gaps | 0.02444750 |
| Wilson 95 % interval | [0.02397346, 0.02493067] |
| simulated rate | 88.0110 per hour over 111.11 simulated hours |

**Below about `2e-4` per gap the rate is not validated by sampling.** It is
the value the Gaussian model gives, and the Gaussian model is an assumption
about the declaration, not a measurement of a system. Real heartbeat jitter is
usually neither independent nor normal: a scheduler overload delays several
beats together, which makes the true rate **higher** than equation (7) at the
same `sigma`. Equation (7) is not a bound.

A declared `sigma` of zero returns exactly `0.0`, which is a statement about
the declaration ("the bound is hard"), not about the system. A tail beyond the
double range, such as `TASK-GNC` at 38.50 sigma, is reported as `0` and should
be read as "below what this arithmetic can represent".

## 7. Parser, command line and environment

| check | reference | result |
|---|---|---|
| malformed-document corpus | 73 hand-written schema rejection cases | 73/73 raised `SpecFormatError`, 73/73 named the offending key path, 0 accepted |
| input and output paths | missing file, directory, invalid UTF-8 | 3/3 raised `SpecFormatError` |
| shipped specifications parse | 3 files | 3/3 |
| exit-code contract | 26 real `python -m watchdogspec` subprocesses | 0 mismatches |
| gitignore sweep | every file this product ships | 0 excluded by a gitignore pattern |

Environment of the recorded run (`validate_environment_output.txt`):

```
python            : 3.13.16
implementation    : CPython
platform          : Linux-6.18.44-fc-v139-x86_64-with-glibc2.39
machine           : x86_64
watchdogspec      : 0.1.0
numpy             : 2.5.0
yaml              : 6.0.3
matplotlib        : 3.11.2
pytest            : 9.1.1
hypothesis        : 6.170.0
```

**Compute budget.** Two cores. The whole validation set runs in 22.3 s and
the test suite in 8 to 13 s depending on contention. The largest single piece
of work is 7.3 million simulated heartbeat gaps in `validate_false_trip.py`,
which takes about 2 s.
Nothing here approaches the three-minute limit, and the two simulation scripts
report their own wall clock and check it against that limit.

## 8. Dependency ranges, measured at both ends

Every range in `pyproject.toml` is bounded on both sides, and **both ends were
installed and the whole suite plus all eight validation scripts run against
them**. A floor with no ceiling broke five published products in this
portfolio on 2026-10-13, and one of the five passed its tests while its
validation numbers moved, so the test verdict alone is not the check: the
committed validation output was diffed between the two configurations.

| package | declared range | floor measured | ceiling measured |
|---|---|---|---|
| numpy | `>=2.1.0,<2.6` | 2.1.0 | 2.5.0 |
| pyyaml | `>=6.0.1,<6.1` | 6.0.1 | 6.0.3 |
| matplotlib (plot, test) | `>=3.10.7,<3.12` | 3.10.7 | 3.11.2 |
| pytest (test) | `>=8.3.5,<9.2` | 8.3.5 | 9.1.1 |
| hypothesis (test) | `>=6.100.0,<6.171` | 6.100.0 | 6.170.0 |
| ruff (test) | `>=0.16.0,<0.18` | 0.16.0 | 0.17.1 |

Result at both ends: **300 tests passed, 0 failed, 0 errors, 0 skipped**, and
`ruff check src/ tests/ examples/ validation/` clean at both ruff versions (0.16.0 and 0.17.1).

Diff of the committed validation output between the two configurations:

| file | difference |
|---|---|
| `validate_known_answers_output.txt` | identical |
| `validate_parser_rejections_output.txt` | identical |
| `validate_properties_output.txt` | identical |
| `worked_example_output.txt` | identical |
| `validate_environment_output.txt` | only the recorded version numbers |
| `validate_latency_vs_simulation_output.txt` | only the wall-clock line |
| `validate_false_trip_output.txt` | only the wall-clock line |
| `validate_cli_output.txt` | identical |

No validation **number** moved between the floor and the ceiling. The only
differences in the whole committed output set are the recorded library
versions and two wall-clock lines.

**The floor run was not ceremony.** The first floor run of the final code
*failed one test*, and that failure is section 5.2: a defect in the comparison
band that the ceiling run had not found in the entire build. Running both ends
is what produced it.

### 8.1 Two floors are set by measurement, not by preference

- **matplotlib 3.10.0 fails this suite.** `tests/test_diagram.py::test_png_is_written_and_is_a_png`
  errors with `PyparsingDeprecationWarning: 'oneOf' deprecated - use 'one_of'`,
  because matplotlib 3.10.0 calls pyparsing's deprecated `oneOf`, pyparsing
  3.3.3 warns, and `[tool.pytest.ini_options] filterwarnings` turns a
  `DeprecationWarning` into an error. matplotlib 3.10.7 does not, so 3.10.7 is
  the declared floor.
- **PyYAML 6.0 cannot be installed here.** It has no cp313 wheel and its sdist
  fails to build against Cython 3 on Python 3.13 with
  `AttributeError: 'build_ext' object has no attribute 'cython_sources'`.
  6.0.1 builds from its sdist and the suite passes on it, so 6.0.1 is the
  declared floor.

### 8.2 The ceilings are a cost, not free safety

`numpy<2.6`, `pyyaml<6.1`, `matplotlib<3.12`, `pytest<9.2`,
`hypothesis<6.171` and `ruff<0.18` mean this package cannot take a future
release above any of those without a port and a re-measurement, including a
security release. That is recorded here and in the README limitations rather
than resolved by deleting the ceilings.

## 9. Scope, once more

This package evaluates a declaration. It does not observe heartbeats, it does
not instrument a supervisor, and it has no way to tell whether the numbers in
a specification describe the system the specification is about. A `CONSISTENT`
verdict means the declared numbers do not contradict each other. Research
grade; not flight-qualified, not certified, not approved for operational
aerospace use.
