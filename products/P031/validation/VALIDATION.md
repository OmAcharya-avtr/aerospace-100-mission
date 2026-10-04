# Validation evidence — HilForge 0.1.0

**Validation level: 3, hardware-pending.** Every number in this document was
produced by a script in this directory, run in the 0.1.0 build session on
2026-10-04, with its raw stdout captured in the `*_output.txt` file named
beside it. Nothing here was typed in by hand, and nothing here came from a
board.

**Environment for every run below:** Python 3.13.16, numpy 2.5.3, scipy
1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, Linux 6.18.44 on x86_64, **one
CPU allowed to the process**, 7.8 GiB RAM, four other build agents running
concurrently in the same container. `perf_counter` reported resolution
1.0e-09 s; measured smallest non-zero step 1.16e-07 s to 1.17e-07 s depending
on the run.

**What Level 3, hardware-pending means here.** All five items of the Batch 04
Level 4 entry plan are implemented and exercised: the HAL with two backends, a
deterministic seeded simulation mode, a dry-run mode, deployment and recovery
procedures as executable checks, and a benchmark harness that records its own
measurement method. What is missing is the only thing that would make it Level
4: **measured timing and resource use from a Jetson Orin Nano.** No number in
this repository comes from one. Every benchmark record carries
`is_hardware: false`, every backend's `BackendInfo.is_hardware` is `False`, and
the device backend's only driver is a loopback stub that wraps the plant model.
Until the harness has been run on the board and its raw output captured, this
product is Level 3 and is labelled Level 3.

---

## How to re-run everything

```bash
cd products/P031
python -m pytest tests/ -q                       # 264 passed in 16.5 s
ruff check src/ tests/ examples/ validation/     # All checks passed
for f in validation/*.py; do PYTHONPATH=src python3 "$f" > "${f%.py}_output.txt"; done
for f in examples/*.py;    do PYTHONPATH=src python3 "$f"; done
```

Every validation script exits 0 on success and non-zero on any failed check, so
the loop above is also the gate.

---

## 1. Simulation/device parity, bit-identical

`validation/parity_simdev.py` → `parity_simdev_output.txt`

The same `LoopConfig` run against the simulated backend and against the device
backend pointed at `LoopbackDriver`, for three injected timing profiles, 1500
iterations each, seed 20261004, period 10 ms.

| Case | signal bytes identical | max abs difference | data digests equal | full digests equal | overrun accounts equal |
|---|---|---|---|---|---|
| constant 0.40 T | yes | 0.000e+00 | yes | yes | yes |
| ramp 0.30 T → 1.35 T | yes | 0.000e+00 | yes | yes | yes |
| sawtooth with overruns | yes | 0.000e+00 | yes | yes | yes |

Tolerance: **exactly zero**, compared as `bytes == bytes` on the float64
signal matrix, not as a numerical closeness.

The data digest is the same string in all three cases
(`ab86a11d…c16aa04f`) and the full digests differ. That is the design working:
the data path does not depend on the injected timing, and the full digest does.

**What this does and does not prove.** It proves that the HAL path, the
complementary filter, the control law, the dry-run guard and both overrun
accounts behave identically whichever backend they are handed, and that the
device path through the driver shim introduces no numerical difference. It
proves nothing about timing on hardware: both backends were driven by a virtual
clock and the loopback driver is the plant model, not a board. Parity is a
statement about the harness, and it is the statement that makes a later
hardware run interpretable — a difference found then will be a hardware
difference, because this check rules out a software one.

---

## 2. Latency percentiles against an injected known distribution

`validation/latency_percentiles.py` → `latency_percentiles_output.txt`

Injected law: shifted exponential, offset `a` = 1.200e-03 s, scale `b` =
9.00e-04 s, 200 000 samples, PCG64 seed 7310. Analytic quantile
`q(p) = a - b ln(1-p)`; density at the quantile `f(q(p)) = (1-p)/b`.

**Stated sampling error**, the tolerance used:
`se(p) = sqrt(p(1-p)/n) / f(q(p)) = b sqrt(p / ((1-p) n))` (Serfling 1980
§2.3.3, Corollary 2.3.3B). The band is **3 se**, computed from the
distribution and `n` before the comparison.

| p | analytic [s] | measured [s] | deviation [s] | 3 se [s] | abs dev / se | result |
|---|---|---|---|---|---|---|
| 0.500 | 1.823832e-03 | 1.822562e-03 | −1.271e-06 | 6.037e-06 | 0.632 | PASS |
| 0.900 | 3.272327e-03 | 3.268935e-03 | −3.391e-06 | 1.811e-05 | 0.562 | PASS |
| 0.990 | 5.344653e-03 | 5.391810e-03 | +4.716e-05 | 6.007e-05 | 2.355 | PASS |
| 0.999 | 7.416980e-03 | 7.426653e-03 | +9.673e-06 | 1.908e-04 | 0.152 | PASS |

The p99 deviation is 2.36 se — inside the 3 se band but not comfortably. With
four quantiles checked, seeing one at 2.4 se is unremarkable; it is reported
rather than smoothed because a reader comparing against their own run should
expect that scale of scatter, not the 0.6 se of the median.

Sample mean 2.099946e-03 s against the analytic `a + b` = 2.100000e-03 s.

**Through the loop.** The first 20 000 of the same samples injected as
iteration durations: the loop's own `total` histogram reproduces a directly
built histogram's percentiles with a difference of **exactly 0.000e+00** at
every p, and records exactly 20 000 samples. The loop neither loses nor
distorts samples.

**Stage split.** `max |sum(stages) − total|` = 8.674e-19 s against a
round-off bound of `8 eps max(total)` = 1.33e-17 s.

**Binned mode.** Bin width 2.000e-05 s; percentile error against exact mode
was +1.744e-05, +1.107e-05, +8.200e-06 and +1.335e-05 s at p50/p90/p99/p99.9 —
all positive and all below one bin width, as the one-sided bound requires. Zero
overflow samples.

**Honest scope:** these are injected synthetic latencies. The check validates
the histogram's percentile arithmetic and the loop's sample handling. It is not
a measurement of any machine's timing.

---

## 3. Overrun accounting against a hand count

`validation/overrun_handcount.py` → `overrun_handcount_output.txt`

Trace (13 iterations, `T = D = 0.010 s`):
`[0.004, 0.012, 0.003, 0.011, 0.011, 0.002, 0.009, 0.010, 0.0101, 0.005, 0.016, 0.006, 0.003]`

The hand count is worked out iteration by iteration in the 30-line comment
above `tests/test_timing.py::test_overrun_hand_counted_sequence` and repeated
in `validation/overrun_handcount.py`. It was done before the code was run.

| Quantity | hand | code | result |
|---|---|---|---|
| direct overrun indices | `[1, 3, 4, 8, 10]` | `[1, 3, 4, 8, 10]` | PASS |
| direct overrun count | 5 | 5 | PASS |
| cascade overrun indices | `[1, 3, 4, 8, 10, 11]` | `[1, 3, 4, 8, 10, 11]` | PASS |
| cascade overrun count | 6 | 6 | PASS |
| max consecutive cascade | 2 | 2 | PASS |
| first cascade run start | 3 | 3 | PASS |
| completion times | 13 values | max difference 1.388e-17 s (tol 1e-15) | PASS |

The same trace through the actual loop, not just through the accounting
function, reproduces the same four lists.

Two iterations are the reason both definitions exist. **Iteration 7** takes
exactly 0.010 s and is **not** an overrun: a deadline met exactly at the
deadline is met. **Iteration 11** takes 0.006 s, well inside the deadline, and
*is* a cascade overrun, because iteration 10 ran 0.016 s and pushed it late.
An implementation that reports one number for "overruns" will disagree with
another implementation about this trace, and the disagreement will be about
the definition rather than the arithmetic.

### Cross-check record for P036 RtClock

`validation/crosscheck_overruns.json`, written by the same script.

| Field | Value |
|---|---|
| samples | 1200 (limit 2000) |
| period | 1.000000e-02 s |
| mean duration | 6.909200e-03 s |
| utilisation `E[d]/T` | 0.690920 |
| `overrun_count` (direct) | **176** |
| `cascade_overrun_count` | 307 |
| `max_consecutive_cascade` | 16 |
| trace units | seconds, rounded to whole nanoseconds |
| generator | PCG64 seed 36031: `gamma(5, 0.0062/5)` + `Bernoulli(0.09) × exponential(0.0075)`, rounded to 1 ns |

`overrun_count` and `overrun_indices` use the **direct** definition, stated
verbatim in the file's `definition` field. The cascade count and indices are
also in the file under their own keys with the recursion written out, so a
disagreement can be localised to a definition rather than left ambiguous. The
trace is rounded to whole nanoseconds and the script verifies that the JSON
round-trips it bit-exactly, so a disagreement cannot be an artefact of decimal
printing.

---

## 4. Dry-run mode issues no write

`validation/dryrun_nowrite.py` → `dryrun_nowrite_output.txt`

5000 iterations, period 10 ms, seed 20261004. Verified by counters read after
the run, not by reading the code.

| Check | expected | measured | result |
|---|---|---|---|
| dry run: actuator `write_count` | 0 | 0 | PASS |
| dry run: record `writes_issued` | 0 | 0 | PASS |
| dry run: record `rehearsed_writes` | 5000 | 5000 | PASS |
| dry run: plant held torque unchanged | 0.0 | 0.0 | PASS |
| **control** — real run: `write_count` | 5000 | 5000 | PASS |
| **control** — real run: `rehearsed_writes` | 0 | 0 | PASS |
| forbidding stub: `write_count` | 0 | 0 | PASS |
| forbidding stub: `attempts` | 0 | 0 | PASS |
| forbidding stub: `DryRunViolationError` raised | never | never | PASS |
| stub raises on a *direct* write | yes | yes | PASS |
| dry-run wrapper still clips 9.0 N·m to the 1.5 N·m limit | 1.500, saturated | 1.500, saturated | PASS |
| dry-run wrapper still rejects NaN and wrong-length commands | 2 rejections | 2 | PASS |

The second-to-last two rows matter: the stub does raise when written to
directly, so a zero `attempts` under the dry-run wrapper is a real zero rather
than a stub that cannot raise. And `attempts` counts *attempted* writes, so the
write is not merely refused — it is never issued.

`tests/test_dryrun.py::test_dry_run_trajectory_diverges_because_nothing_was_actuated`
adds a third angle: the dry run's commands separate from the real run's after
the first iteration, by more than 0.1 N·m by iteration 60, precisely because no
torque was applied. If a write had leaked through, the two trajectories would
agree.

---

## 5. Recovery after an induced mid-run abort

`validation/recovery_abort.py` → `recovery_abort_output.txt`

Every case warms the rig up for 25 iterations first, so the pre-run state is
non-trivial (write count 25, held torque −0.6736660941755666 N·m, θ =
0.077735046510207323 rad), then injects the fault and verifies the restoration
field by field over a flattened snapshot that includes the plant's PCG64 bit
state.

| Induced fault | signal | fields compared | fields differing | recovery verified |
|---|---|---|---|---|
| device disconnects mid-run | `DeviceDisconnectedError` | 14 | 0 | yes |
| sample dropped, policy=abort | `SampleDroppedError` | 14 | 0 | yes |
| actuator write rejected, policy=abort | `WriteRejectedError` | 14 | 0 | yes |
| timebase steps backwards | `TimebaseRegressionError` | 14 | 0 | yes |
| loop overrun cascade | `OverrunCascadeError` | 13 | 0 | yes |

5/5 recovered exactly, with write count, held torque, θ and ω identical to 17
significant figures before and after.

### The defect this check found

**The first run of this script reported 4/5.** A mid-run device disconnect did
not raise: the loop caught `DeviceDisconnectedError`, recorded
`aborted=True` in the record and returned it. `RunGuard` therefore saw a clean
exit from its `with` block and never ran the recovery, leaving the rig mid-run
with 11 extra writes issued and the plant 2.2 mrad away from where the snapshot
had it.

This was a design error, not a test artefact: a returned record looks like a
completed run to anything that only checks for an exception, and that is most
callers. The fix was to make raising the default —
`LoopConfig.on_disconnect="raise"`, with the partial record attached to the
exception as its `record` attribute so nothing is lost — and to keep
`"record"` as an explicit opt-in for a caller that really does want the partial
record returned. The same pass added the partial record to every other aborting
exception, and `TimebaseRegressionError` to the set that attaches it (before
that, the failure-mode example reported 0 completed iterations for the
timebase case because the record was never attached).

It is written up here because a validation suite in which nothing ever fails is
not evidence that there is nothing wrong; it is evidence that the checks are
too weak. See `CHANGELOG.md` and `docs/REQUIREMENTS.md` §5.

**Honest scope:** recovery is verified against a simulated rig and a loopback
stub, where the whole state is readable and restorable. A real board's state is
not: a motor that has moved has moved. The procedure and its verification would
have to be re-run, and probably narrowed, on hardware.

---

## 6. Timing-measurement uncertainty, with clock resolution as a stated term

`validation/timing_uncertainty.py` → `timing_uncertainty_output.txt`

### 6a — the arithmetic against a hand-computed case

Four identical 1 ms durations, `delta` = 1.0e-06 s. Then `s = 0`, so
`u_A = 0`, and `u_B = delta/sqrt(6n) = 1e-6/sqrt(24)`.

| Quantity | hand | code | result |
|---|---|---|---|
| `u_statistical_s` | 0.000000000e+00 | 0.000000000e+00 | PASS |
| `u_resolution_s` | 2.041241452e-07 | 2.041241452e-07 | PASS |
| `combined_s` | 2.041241452e-07 | 2.041241452e-07 | PASS |
| `expanded_k2_s` | 4.082482905e-07 | 4.082482905e-07 | PASS |
| `u` for a single duration, `delta/sqrt(6)` | 4.082482905e-07 | 4.082482905e-07 | PASS |

Scaling laws: `u_A` fell by a factor 1.978 and `u_B` by exactly 2.000 when `n`
went from 10 000 to 40 000 (both expected 2); `u_B` rose by exactly 1.0000e+04
when `delta` rose by 1e4.

### 6b — the measured clock on this host

| Quantity | Value |
|---|---|
| reported resolution (`time.get_clock_info`) | 1.000e-09 s |
| **measured** smallest non-zero step | 1.130e-07 s |
| measured timing-call-pair overhead (median) | 5.900e-08 s |
| `u(duration)` from resolution alone, `delta/sqrt(6)` | 4.613e-08 s |
| samples | 20 000 |

The reported and measured resolutions differ by two orders of magnitude. The
platform claims nanoseconds; what a Python loop can actually distinguish on
this container is ~113 ns (it measured between 112 and 117 ns across the runs
of this session). The budget uses the measured value, which is why it
is measured rather than read off `get_clock_info`.

### 6c — the budget applied to a measured 20 000-iteration run

**This table and the one in §7 are a snapshot of the run captured in the
adjacent `_output.txt`.** They are the only numbers in this document that move
between runs, and they move by factors of a few — see the note under the
interpretation. If a re-run disagrees with the values printed here, the file is
right and this table is stale; that is a property of the environment and not a
finding.

| Stage | n | mean [s] | stdev [s] | u_A [s] | u_B [s] | u_c [s] | u_B/u_c |
|---|---|---|---|---|---|---|---|
| sense | 20000 | 1.782987e-05 | 6.089207e-05 | 4.306e-07 | 3.262e-10 | 4.306e-07 | 7.6e-04 |
| estimate | 20000 | 1.125255e-06 | 3.970149e-06 | 2.807e-08 | 3.262e-10 | 2.808e-08 | 1.2e-02 |
| control | 20000 | 8.342501e-06 | 3.595532e-05 | 2.542e-07 | 3.262e-10 | 2.542e-07 | 1.3e-03 |
| actuate | 20000 | 1.529286e-05 | 5.859171e-05 | 4.143e-07 | 3.262e-10 | 4.143e-07 | 7.9e-04 |
| **total** | 20000 | 4.259049e-05 | 9.261932e-05 | 6.549e-07 | 3.262e-10 | 6.549e-07 | 5.0e-04 |

Full budget for the mean total iteration duration:

```
mean                      : 4.259049395e-05 s
sample stdev              : 9.261932165e-05 s
clock resolution (delta)  : 1.130000000e-07 s
u_A statistical  s/sqrt(n): 6.549175040e-07 s
u_B resolution d/sqrt(6n) : 3.262029021e-10 s
u_c combined (quadrature) : 6.549175853e-07 s
U expanded (k=2)          : 1.309835171e-06 s
bias: timing-call overhead: 5.900000000e-08 s   (reported, not subtracted)
```

**Interpretation.** The resolution term contributes 5.0e-04 of the combined
uncertainty on the total — it is negligible here. It is carried anyway because
on a coarse clock it would dominate, and a budget that silently drops it would
be wrong there without saying so. What is *not* negligible is the bias: four
stages at one timing-call pair each is 2.360e-07 s, **0.55 %** of the mean
iteration. That is reported and never subtracted, because subtracting a median
from individual samples would make the shortest durations negative.

The stdev is about twice the mean. That is not jitter in the work; it is
preemption on a shared single-core container. It is the reason none of these
numbers is usable as a hardware figure, and the reason the harness reports
percentiles and maxima rather than means.

**How much these move.** Across three runs of this same script in this same
session, with the same code and the same seed, the mean total iteration
duration was 1.32e-04 s, 4.43e-05 s and 4.26e-05 s — a factor of three between
the first and the others, from nothing but how much of the core the four other
agents wanted at the time. Every other number in this document is either exact
or bounded by a stated tolerance and is reproducible anywhere. That contrast is
the whole argument for recording the method and the environment with the
measurement, and it is why the only honest use of this table is as a coarse
tripwire.

---

## 7. Performance benchmark

`validation/performance_benchmark.py` → `performance_benchmark_output.txt`,
`validation/benchmark_record.txt`, `validation/benchmark_record.json`

Period 10 ms, seed 20261004, no pacing (the loop runs flat out and deadlines
are accounted against the period).

Snapshot of the run captured in `performance_benchmark_output.txt`; see the
note in §6c about how far these move between runs.

| Case | n | p50 [s] | p99 [s] | max [s] | iterations/s | tracemalloc peak [MB] |
|---|---|---|---|---|---|---|
| simulated, write path | 20000 | 1.338760e-04 | 3.823830e-04 | 8.227209e-03 | 4153.7 | 16.608 |
| loopback device, write path | 20000 | 1.367330e-04 | 3.892580e-04 | 9.166359e-03 | 4164.9 | 16.601 |
| simulated, dry-run path | 20000 | 1.194720e-04 | 4.036930e-04 | 6.219441e-03 | 4384.6 | 16.597 |
| simulated, short run | 1000 | 1.332270e-04 | 4.750300e-04 | 1.277627e-03 | 3989.0 | 0.982 |

Process high-water memory 89.412 MB (`ru_maxrss`, interpreted as kibibytes on
Linux; the record states which conversion it applied and derives it from
`sys.platform`).

`is_hardware` across all four cases: `[False, False, False, False]`.

**Read this table with its environment.** The p50 column is consistent across
cases to within 15 % and is the only part of the table that is about the code:
the loopback device path has one more layer of indirection than the simulated
path and its p50 is 2.1 % higher, which is the right sign and within the noise;
the dry-run path is 11 % faster, which is the suppressed write. Everything to
the right of p50 is the container. The max column spans a factor of 7 between
cases that differ by nothing relevant, and across three runs of this script in
this session the simulated write path's p99 was 4.29e-03 s, 3.28e-04 s and
3.82e-04 s — a factor of 13 on the same code and the same seed. These are not
hardware numbers, they are not stable numbers, and the only honest use for them
is as a coarse regression tripwire. They are published because the measurement
method and the environment are published with them.

**What a Level 4 record would need instead:** this same harness, this same
`BenchmarkRecord` format, run on a Jetson Orin Nano with nothing else on the
cores it uses, with its raw `.txt` and `.json` output captured and
`is_hardware: true` because a real driver shim answered. That does not exist.

---

## 8. Deployment pre-run checks

`validation/deployment_checks.py` → `deployment_checks_output.txt`

Nine named checks, each with a measured value and the value required, run
against all three backends this repository can build.

| Backend | verdict | checks passed |
|---|---|---|
| simulated plant | GO | 9/9 |
| loopback device stub | GO | 9/9 |
| absent device | **NO-GO** | 1/1 attempted (the report stops at `device_present`) |

The ninth check, `state_unchanged`, reports `0 of 14 differing  (restored on
exit)` for both working backends. It exists because of the second defect this
validation round found — see §11.

A deliberately impossible requirement — a 10 ns period against a 1 ns clock,
where the 1 % rule requires 0.1 ns — fails `clock_resolution` and **only**
`clock_resolution`. A check suite in which nothing can fail is a tick-list, and
this one is not.

The script also prints the captured-field list read from the dataclasses
themselves (`IterationRecord`: 18 fields, `RunRecord`: 22 fields) rather than
from a docstring, so the documented capture cannot drift from the code, and
prints the full flattened recovery snapshot including the plant's PCG64 state.

---

## 9. Baselines against the learned overrun predictor

`validation/predictor_benchmark.py` → `predictor_benchmark_output.txt`

Two trace families × three seeds (4242, 909090, 31337) × two operating points ×
three predictors. Horizon 3 iterations, 40 000 iterations per trace,
chronological 60/40 split (a random split would leak the autocorrelation the
predictors exist to exploit). Total fitting time across every case: **4.2 s** on
one core.

The two trace families are built to differ in exactly one way that matters.
Measured lag-1 autocorrelation of the overrun indicator: **0.02** for
`jittery`, **0.85** for `bursty`, at similar per-iteration overrun rates
(~0.15 both) and utilisations 0.70 and 0.79.

### Mean over three seeds, F1-maximising operating point

| family | predictor | false-alarm | missed | lead [it] | F1 | Brier | AUC | flag rate |
|---|---|---|---|---|---|---|---|---|
| jittery | fixed_threshold | 0.9887 | 0.0095 | 1.886 | 0.5604 | 0.23796 | **0.5041** | 0.9894 |
| jittery | queueing_markov | 0.2867 | 0.7100 | 1.875 | 0.3337 | 0.24117 | **0.5016** | 0.2880 |
| jittery | learned_hgb | 0.9935 | 0.0069 | 1.884 | 0.5604 | 0.23853 | **0.4966** | 0.9933 |
| bursty | fixed_threshold | 0.0156 | 0.2930 | 1.019 | 0.7967 | 0.06306 | 0.8558 | 0.1474 |
| bursty | queueing_markov | 0.0106 | 0.3219 | 1.017 | 0.7865 | 0.06983 | 0.8337 | 0.1378 |
| bursty | learned_hgb | 0.0113 | 0.2820 | 1.017 | **0.8123** | **0.05775** | **0.8583** | 0.1460 |

### Mean over three seeds, matched flag rate (target 0.15)

| family | predictor | false-alarm | missed | lead [it] | F1 | Brier | AUC | achieved flag rate |
|---|---|---|---|---|---|---|---|---|
| jittery | fixed_threshold | 0.1483 | 0.8441 | 1.876 | 0.2247 | 0.23834 | 0.5041 | 0.1513 |
| jittery | queueing_markov | 0.2867 | 0.7100 | 1.875 | 0.3337 | 0.24117 | 0.5016 | 0.2880 |
| jittery | learned_hgb | 0.0511 | 0.9502 | 1.861 | 0.0801 | 0.23853 | 0.4966 | 0.0506 |
| bursty | fixed_threshold | 0.0127 | 0.3377 | 1.019 | 0.7701 | 0.06862 | 0.8558 | 0.1364 |
| bursty | queueing_markov | 0.0106 | 0.3219 | 1.017 | 0.7865 | 0.06983 | 0.8337 | 0.1378 |
| bursty | learned_hgb | 0.0098 | **0.3202** | 1.017 | 0.7892 | **0.05775** | 0.8583 | 0.1372 |

Two of those rows need their achieved flag rate read before their metrics. The
Markov baseline's probability output takes only **two** distinct values, one
per fitted regime, so no cut-off gives it a 15 % flag rate; the nearest
achievable is 28.8 % on the jittery family. The learned model's isotonically
calibrated probability on the jittery family is nearly as coarse, and lands at
5.1 %. Both are reported with the rate they actually achieved rather than the
rate they were asked for.

### The result, stated plainly

**On the jittery family, nothing works, and the fixed threshold is not beaten
because there is nothing to beat it with.** All three ROC AUCs are within 0.01
of 0.50: `fixed_threshold` 0.5041, `queueing_markov` 0.5016, `learned_hgb`
0.4966 — the learned model is, by a hair, the worst of the three and all three
are indistinguishable from guessing. That is the correct answer for a trace
whose overruns have lag-1 autocorrelation 0.02: the recent past does not
contain the information, so no amount of modelling extracts it. The
F1-maximising operating point on this family degenerates to flagging ~99 % of
iterations, because with a 39 % positive rate under a horizon of 3, flagging
everything maximises F1. That is reported rather than hidden; it is what an F1
criterion does on a high base rate with no signal.

**On the bursty family the learned model wins, narrowly and unevenly.** At the
matched flag rate its missed-overrun rate is 0.3202 against the threshold's
0.3377, a difference of 0.0175, against a typical 95 % Wilson half-width of
**0.0168** on that rate. The difference is marginally larger than the interval
around either number: a real but small improvement, the same order as its own
uncertainty, and not a decisive one. Its clearer advantages are calibration —
Brier 0.05775 against 0.06862, a **15.8 % reduction** — and ranking, AUC 0.8583
against 0.8558. So the honest summary is: *the learned model's probability is
the only usable confidence output of the three, and its decisions are not
reliably better than a one-parameter threshold's.*

**The queueing/Markov baseline is the most conservative.** It flags least and
misses most on the bursty family, which is its stated independence
approximation doing exactly what `MODEL_CARD.md` §2 says it does: multiplying
per-step survival probabilities over the horizon treats successive durations as
independent given their regimes, which they are not, so the model
underestimates the probability of a run of overruns.

**Lead time** is ~1.02 iterations on the bursty family and ~1.88 on the
jittery one. That is not the bursty predictors being slow to warn; it is where
the overruns are. Inside a burst the next iteration is usually the one that
misses, so a correct flag has a lead time of 1 by construction. A lead time
near the horizon mean, as on the jittery family, is what a flag looks like when
it carries no information about *which* iteration will miss — the flag is
right that something will overrun in the window and has no idea when.

**Conclusion, with the recommendation it implies.** None of this is a reason to
put the learned model in a loop in place of the threshold. An 18-feature
boosted ensemble is a large amount of machinery for a within-interval win on
one synthetic trace family; the threshold has one parameter and can be read in
an afternoon. The defensible use of the learned model is its calibrated
probability, where it is clearly the best of the three — if what you need is
"how likely is an overrun in the next three iterations", it answers; if what you
need is "flag it or not", use the threshold.

### The learned model's uncertainty output

Reliability table on the bursty held-out set, seed 4242, in
`predictor_benchmark_output.txt`. Permutation importance on the same set is
dominated by one feature: `total_max8_s` at +0.044492 Brier, an order of
magnitude above the next (`total_last_s`, +0.003730), then `control_last_s`
(+0.001819) and `headroom_frac` (+0.001202). The rolling maximum over the last
eight iterations is where the autocorrelation lives, which is what the bursty
family was built to have — and it means seventeen of the eighteen features are
nearly idle, so a two-feature model would probably do almost as well. That is
another reason the learned model is hard to justify over the threshold. `examples/overrun_predictor.py` plots the reliability curve for both
families; on the jittery family it is flat, which is the same negative result
seen from a different angle.

---

## 10. Regression pins

`tests/test_regression.py`, 11 parametrised tests over 13 pinned values for
fixed seeds. These are change detectors, not correctness checks — the
correctness checks are §1–§9 above. If a pin fails after a deliberate change it
is updated in the same commit as the change, and said so in `CHANGELOG.md`.

| Pin | Value |
|---|---|
| data digest (both backends, 500 iterations, seed 20261004) | `c955385f9f0522551f857b4527f3353b41cc001b574ea36351a8a17e1dad716d` |
| full digest (same run) | `6233d8afe161c13bacca2647cf7556ba678d908bdb00e320aaed3e407499c750` |
| direct / cascade / max-run overruns (same run) | 167 / 167 / 167 |
| first sample `[θ, ω]` | `(0.07995176313764274, −0.0019565700959251068)` |
| final estimate θ̂ [rad] | −0.0032182389227649764 |
| final command [N·m] | −0.0038191376287159198 |
| signal-matrix sum | 18.659889514420314 |
| plant state after 300 PD steps, seed 777 | `(0.0025907643443245317, −0.011771693226175889)` |
| jittery trace, 5000 iterations, seed 99: mean / direct / cascade / max-run | 7.0087972905e-03 s / 764 / 1109 / 10 |
| bursty trace, same: mean / direct / cascade / max-run | 7.7282903634e-03 s / 684 / 1854 / 413 |

---

## 11. What failed, what is weak, and what is not claimed

**Failed and then fixed, within this session: two defects.**

1. The mid-run-disconnect recovery gap in §5 — a disconnect returned a record
   instead of raising, so `RunGuard` never recovered. Found by
   `validation/recovery_abort.py`, which reported 4/5 on its first run. Fixed
   by making raising the default and attaching the partial record to the
   exception. Covered by two new tests.
2. **`preflight` left the rig one sensor sample ahead of where it found it.**
   Found while writing the README's worked example: a backend that had been
   pre-flighted no longer agreed bit for bit with one that had not, because the
   `sensor_read` check draws a sample and that advances the plant's noise
   stream. `preflight` now snapshots on entry, restores on exit, and reports the
   comparison as a ninth check (`state_unchanged`). Covered by
   `tests/test_deploy.py::test_preflight_leaves_the_backend_state_untouched`,
   which asserts both the restoration and the consequence that matters — that
   parity survives a one-sided preflight.

Both were design errors rather than test artefacts, both were fixed by changing
the code rather than the tolerance, and both are in `CHANGELOG.md`.

**Weak, and reported as weak:**

1. The p99 latency deviation in §2 is 2.36 standard errors. Inside the band,
   not comfortably inside it.
2. Every timing number in §6 and §7 comes from a shared, contended,
   single-core container. The iteration-duration standard deviation is six
   times its mean. The p99 and max columns are not reproducible on this machine
   between runs, let alone on another machine.
3. The learned predictor's win in §9 is within interval on the metric that
   would justify deploying it, and its only clear win is on calibration.
4. The Markov baseline cannot be put on an arbitrary operating point, because
   its score has two levels. That limits what the matched-flag-rate comparison
   can say about it.
5. All the predictor evidence is from synthetic traces generated by a model in
   this repository. A trace from a real loop on real hardware could have
   structure neither family captures, and the ranking of the three predictors
   on it is not predicted by anything here.

**Not claimed, anywhere:**

* No hardware measurement. Every backend in this repository has
  `is_hardware == False`.
* No Level 4. Level 4 requires measured timing and resource use from a Jetson
  Orin Nano, and that measurement does not exist. Reporting Level 4 as 0 is the
  correct reporting.
* Not flight-qualified, not certified, not approved for operational aerospace
  use. Research-grade.
