# HilForge — Requirements and Verification Matrix

**Product:** P031 HilForge · **Version:** 0.1.0 · **Status:** TESTING
**Validation level:** 3, hardware-pending · **Licence:** AGPL-3.0-or-later
**Date of this revision:** 2026-10-04

This document states what `hilforge` is required to do, in numbered,
individually verifiable terms, and names for each requirement the test or
validation script that verifies it. Requirements are written so that a
verification can **fail**: "shall reproduce X exactly" or "shall agree to
within Y" rather than "shall be correct".

Paths are relative to `products/P031/`. Test identifiers are pytest node ids;
validation identifiers are scripts in `validation/` whose captured stdout is
stored alongside them as `*_output.txt`.

---

## 1. Scope and definitions

HilForge is a hardware-in-the-loop harness for a single periodic control loop.
It contains a hardware abstraction layer with two interchangeable backends, a
fixed-rate loop with deadline accounting, dry-run rehearsal, deterministic
replay, executable pre-run and recovery checks, a benchmark harness, and a
deadline-overrun predictor with two deterministic baselines.

It does **not** contain a board driver, a real-time operating system, a
scheduler, a board-farm provisioner or an instrument-control layer. See
`README.md` §Alternatives.

| Term | Meaning in this document |
|---|---|
| *backend* | an implementation of the four HAL protocols: timebase, sensor, actuator, container |
| *driver shim* | the five-method interface between the HAL and a board (`hilforge.backends.device.DriverShim`) |
| *loopback stub* | `LoopbackDriver`, a driver shim backed by the plant model; **not hardware** |
| *model clock* | the backend's timebase; drives the plant, timestamps iterations |
| *measurement clock* | the host clock that times stage execution; never touches the data path |
| *period* `T` | the loop period [s] |
| *deadline* `D` | the relative deadline [s], `0 < D <= T`, default `D = T` |
| *direct overrun* | iteration `i` with `d[i] > D` |
| *cascade overrun* | iteration `i` with `c[i] > i T + D`, where `c[i] = max(i T, c[i-1]) + d[i]`, `c[-1] = 0` |
| *data path* | the sequence of sensor samples, estimates and commands; contains no timing |
| *data digest* | SHA-256 over the data path, the per-iteration flags and the counters |
| *full digest* | SHA-256 over the data digest plus every stage duration and completion time |
| *horizon* `H` | the look-ahead of the overrun label, in iterations |
| *is_hardware* | `True` only for a device backend whose driver reports itself as physical; `False` for every backend in this repository |

**Equality is not an overrun.** `d[i] == D` and `c[i] == D[i]` are met
deadlines. This is stated here because it is the single most likely source of
an off-by-one disagreement with another implementation.

---

## 2. Functional requirements

### R-01 — One HAL contract, two backends
The package shall define sensor, actuator, timebase and backend protocols, and
shall provide a simulated backend and a device backend that both satisfy them.
The same test suite shall run against both, parametrised rather than
duplicated.

*Verification:* `tests/test_backends.py::TestBothBackends` (nine assertions,
each run twice), `tests/test_hal.py::test_both_backends_satisfy_the_protocols`.

### R-02 — Simulation/device parity, bit-identical
On a seeded deterministic case with the device backend pointed at the loopback
stub, the two backends' signal matrices shall be **byte-identical** and their
data and full digests shall be equal, for at least three distinct injected
timing profiles.

*Verification:* `validation/parity_simdev.py` (three cases),
`tests/test_regression.py::test_the_two_backends_agree_on_every_pinned_value`,
`tests/test_backends.py::test_simulated_and_device_reads_are_bit_identical`.

*Tolerance:* exactly zero. Not "within 1e-12".

### R-03 — Deterministic seeded replay
A recorded run shall replay through a replay backend and reproduce the original
run's data digest exactly, and the replay shall reproduce the original overrun
accounting index for index.

*Verification:* `tests/test_replay.py::test_replay_reproduces_the_data_digest_exactly`,
`::test_replay_reproduces_the_overrun_accounting`,
`::test_replay_of_a_measured_run_reinjects_its_durations`.

### R-04 — Both overrun definitions, computed and reported
The package shall compute and report the direct and cascade overrun counts and
index lists for every run, and shall never report one as if it were the other.

*Verification:* `validation/overrun_handcount.py` §3a and §3b,
`tests/test_timing.py::test_overrun_hand_counted_sequence`,
`::test_cascade_diverges_from_direct`,
`tests/test_properties.py::test_cascade_overruns_are_a_superset_of_direct_overruns`.

### R-05 — Overrun accounting against a hand count
The accounting shall reproduce a hand-counted 13-iteration sequence exactly:
direct indices `[1, 3, 4, 8, 10]`, cascade indices `[1, 3, 4, 8, 10, 11]`,
longest consecutive cascade run 2 starting at index 3, and completion times to
within 1e-15 s. The hand count shall be written out in the source, not
generated by the code under test.

*Verification:* `tests/test_timing.py::test_overrun_hand_counted_sequence` (the
hand count is the 30-line comment above it),
`::test_overrun_hand_counted_completions`, `validation/overrun_handcount.py` §3a.

### R-06 — Latency percentiles with a stated convention
The latency histogram shall implement both the nearest-rank (Hyndman & Fan
type 1) and linear-interpolation (type 7) quantile conventions, shall name them
in its API, and shall agree with `numpy.percentile` to a relative 1e-12 in
linear mode.

*Verification:* `tests/test_timing.py::test_histogram_percentiles_known_answer`
(hand-computed on 1..10), `::test_histogram_matches_numpy_linear_percentile`,
`tests/test_properties.py::test_nearest_rank_percentile_is_always_an_observed_sample`.

### R-07 — Percentiles reproduce an injected known distribution
On 200 000 samples drawn from a shifted exponential, the measured p50, p90, p99
and p99.9 shall lie within **3 standard errors** of the analytic quantile,
where the standard error is `sqrt(p(1-p)/n) / f(q_p)` (Serfling 1980 §2.3.3)
computed from the distribution before the comparison.

*Verification:* `validation/latency_percentiles.py` §2a. Measured: deviations
of 0.63, 0.56, 2.36 and 0.15 standard errors respectively.

### R-08 — Binned-mode error bounded by one bin
In binned mode the percentile error relative to exact mode shall lie in
`[0, one bin width]`, i.e. one-sided and bounded, and the bound shall be
exposed as an API value rather than documented only in prose.

*Verification:* `validation/latency_percentiles.py` §2d,
`tests/test_timing.py::test_binned_histogram_error_is_bounded_by_one_bin`.

### R-09 — Dry-run mode issues no write
With `dry_run=True` the wrapped actuator's `write_count` shall be zero after a
full run, the plant's held command shall be unchanged, and a counting stub
configured to forbid writes shall record zero attempts. The same run with
`dry_run=False` shall write once per iteration, so the zero is not vacuous.

*Verification:* `validation/dryrun_nowrite.py` §4a–4e (5000 iterations),
`tests/test_dryrun.py` (nine tests),
`tests/test_dryrun.py::test_dry_run_trajectory_diverges_because_nothing_was_actuated`.

### R-10 — Pre-run checks as executable assertions
The package shall provide a pre-run check report of at least eight named
checks, each carrying a measured value and the value required, and shall return
GO/NO-GO rather than raising, so the caller decides. A check shall be able to
fail: the report for an absent device shall be NO-GO, and a period finer than
the clock shall fail the resolution check and nothing else. Running the checks
shall leave the backend in exactly the state it was found in, and that shall
itself be one of the checks.

*Verification:* `validation/deployment_checks.py` (nine checks, three
backends), `tests/test_deploy.py` (sixteen tests, parametrised over both
backends), `tests/test_deploy.py::test_preflight_leaves_the_backend_state_untouched`.

### R-11 — Recovery after an induced mid-run abort
After each of the five abort-capable failure modes, the recovery procedure
shall restore the pre-run state with **zero** differing fields over a flattened
snapshot that includes the plant's RNG bit state, and shall report the
comparison rather than asserting success.

*Verification:* `validation/recovery_abort.py` (five cases, 13–14 fields each,
all zero differences), `tests/test_failure_modes.py::TestMidRunDisconnect::test_recovery_restores_the_pre_run_state`,
`::TestOverrunCascade::test_recovery_after_a_cascade_abort`,
`tests/test_deploy.py::test_run_guard_reraises_the_failure`.

### R-12 — Six failure modes, each distinctly signalled
The six required failure modes shall each raise or record a distinct exception
type, injected at the driver so the loop takes its normal path: device absent,
device disconnects mid-run, sample dropped, timebase steps backwards, actuator
write rejected, loop-overrun cascade. Every aborting exception shall carry the
partial run record as its `record` attribute.

*Verification:* `tests/test_failure_modes.py` (twenty-six tests in six
classes), `examples/failure_modes.py`,
`tests/test_loop.py::test_every_aborting_exception_carries_the_partial_record`.

### R-13 — Benchmark record states its own method and environment
Every benchmark record shall carry the clock used, its reported and measured
resolution, the measured timing-call overhead, the resolution-derived duration
uncertainty, the host, a caller-supplied environment note (with no default),
the backend's `is_hardware` flag, and a caveat naming what is still required
for Level 4. A record shall be refused if the environment note is blank.

*Verification:* `validation/performance_benchmark.py`, `tests/test_bench.py`
(nine tests), `validation/benchmark_record.json`.

### R-14 — Timing-measurement uncertainty with clock resolution as a term
The package shall report a combined standard uncertainty on a mean duration
with a Type A statistical term `s/sqrt(n)` and a Type B resolution term
`delta/sqrt(6 n)`, combined in quadrature per JCGM 100:2008 §5.1.2, and shall
report the timing-call overhead separately as a bias rather than folding it in.

*Verification:* `validation/timing_uncertainty.py` §6a (hand-computed case),
§6b (measured clock), §6c (applied to a 20 000-iteration run),
`tests/test_timing.py::test_timing_uncertainty_known_answer`,
`tests/test_properties.py::test_combined_uncertainty_is_at_least_each_component`.

### R-15 — Two deterministic baselines, implemented before the learned model
The package shall implement a fixed-threshold-on-previous-iteration predictor
and a Markov/queueing stage-latency predictor, both with their fitting
procedure auditable (threshold sweep and probability sweep retained as
attributes), before and independently of any learned model.

*Verification:* `tests/test_predict.py::test_fixed_threshold_fits_and_describes`,
`::test_queueing_baseline_fits_a_two_state_chain`,
`::test_md1_pollaczek_khinchine_known_answer`,
`validation/predictor_benchmark.py` (the `describe()` block for each fit).

### R-16 — Learned predictor benchmarked on the three required metrics
The learned predictor shall be benchmarked against both baselines on lead
time, false-alarm rate and missed-overrun rate, on held-out data from a
chronological split, at two operating points, over at least three seeds and
two trace families, with 95 % Wilson intervals on the two rates.

*Verification:* `validation/predictor_benchmark.py` (2 families × 3 seeds × 2
operating points × 3 predictors).

**Measured outcome, kept as it came out:** on the jittery family every
predictor has ROC AUC within 0.01 of 0.50 — none of them can rank the
iterations, including the learned one. On the bursty family the learned model
has the best Brier score (0.0578 against 0.0686 for the threshold, a 15.8 %
reduction) and the best AUC (0.8583 against 0.8558), while its missed-overrun
advantage at a matched flag rate (0.3202 against 0.3377) is the same order as
the 95 % interval half-width on that rate (0.0168). See
`MODEL_CARD.md` §7 and `README.md` §Validation evidence.

### R-17 — The learned model exposes a calibrated uncertainty output
The learned predictor shall output a probability, not only a decision, shall
calibrate it on a held-back chronological slice of the training data, and shall
provide a reliability table on held-out data so the probability claim can be
falsified.

*Verification:* `tests/test_predict.py::test_learned_model_fits_and_produces_calibrated_probabilities`,
`::test_learned_model_beats_a_constant_on_the_bursty_trace`,
`validation/predictor_benchmark.py` (reliability table),
`examples/overrun_predictor.py` (reliability curve, bottom right panel).

### R-18 — Regression suite with pinned seeded outputs
The package shall pin, for fixed seeds, the data and full digests, both overrun
counts, the first sample, the final estimate and command, the signal-matrix
sum, a plant trajectory, and the trace statistics of both presets, so that any
numerical change surfaces as a test failure.

*Verification:* `tests/test_regression.py` (eleven parametrised tests,
thirteen pinned values).

---

## 3. Interface and robustness requirements

### R-19 — CLI
`python -m hilforge --help` shall exit 0, and each of the nine subcommands
shall exit 0 for `--help`. `preflight` shall exit non-zero on NO-GO, `dryrun`
non-zero if any write occurred, `parity` non-zero on a digest mismatch, and
`replay` non-zero on a digest mismatch.

*Verification:* `tests/test_cli.py` (eighteen tests, nine of them parametrised
over the subcommands).

### R-20 — Input validation with actionable messages
Every public constructor and function shall reject invalid input with
`ValueError`/`TypeError` (or the `ConfigurationError` subclass of `ValueError`)
naming the offending parameter and its value. No silent clamping of an
out-of-range configuration.

*Verification:* validation tests throughout — `tests/test_timing.py` (five),
`tests/test_plant.py` (three), `tests/test_hal.py` (four),
`tests/test_loop.py::test_loop_config_validation` (eleven cases),
`tests/test_predict.py` (five).

### R-21 — Units in every public docstring
Every public function and dataclass field whose value has a physical dimension
shall state that dimension in its docstring, in SI.

*Verification:* by review; mechanically, the suffix convention `_s`, `_rad`,
`_rad_s`, `_nm`, `_kgm2`, `_hz` on every such name.

---

## 4. Deliberate non-requirements

These are the things HilForge does **not** do. Each is a decision, not an
oversight, and each is the reason some reader should use something else.

| Ref | Not implemented | Consequence |
|---|---|---|
| N-01 | A real board driver | `DriverShim` is a five-method contract with two stub implementations. Running on hardware requires writing one; nothing in this repository has touched a board. |
| N-02 | Board provisioning, power control, flashing, serial-console capture | `labgrid` does this. HilForge assumes a board is already up and reachable. |
| N-03 | Instrument control (SCPI, VISA) | `pyvisa` does this. HilForge has no instrument layer. |
| N-04 | HDL simulation or testbenches | `cocotb` is a different layer entirely. |
| N-05 | Real-time scheduling, priority assignment, CPU pinning, `SCHED_FIFO` | The loop measures and accounts; it does not try to make Python real-time, which it is not. |
| N-06 | Pacing the loop to the period | The benchmark harness runs flat out and accounts deadlines against `T`. Sleeping would measure the sleep. A paced driver belongs in P036 RtClock. |
| N-07 | Schedulability analysis — RM/EDF bounds, response-time analysis, blocking under a priority ceiling | P036 RtClock's scope. HilForge quotes the M/D/1 Pollaczek-Khinchine wait as a reference and does nothing with it. |
| N-08 | Multi-rate or multi-task loops | One periodic task, four stages. No task set, no shared resources, no priority inversion. |
| N-09 | Multi-axis attitude, flexible modes, disturbance torques | The plant is a single-axis rigid body. It exists to give the loop something deterministic to drive, not to be a vehicle model. |
| N-10 | Gyro bias estimation | The complementary filter inherits the bias; the residual attitude offset in the examples is that bias. |
| N-11 | Distributed or networked HIL | One process, one backend. No time synchronisation between hosts, no IEEE 1588. |
| N-12 | Any Level 4 claim | Level 4 requires measured timing and resource use from a Jetson Orin Nano. No such measurement exists. Every benchmark record in this repository has `is_hardware: false`. |

---

## 5. Verification status summary

Run in the 0.1.0 build session on 2026-10-04 (Python 3.13.16, numpy 2.5.3,
scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, **one** shared CPU core,
four other build agents running concurrently):

* `python -m pytest tests/ -q` → **264 passed, 0 failed, 0 errors, 0 skipped**
  in 16.5 s (counts from the junit XML, not from stdout)
* `ruff check src/ tests/ examples/ validation/` → **All checks passed**
  (line-length 100)
* nine validation scripts executed, each exiting 0, raw stdout stored in
  `validation/*_output.txt`
* four examples executed, PNGs stored in `screenshots/`

**Two requirements were changed during verification rather than after it.**

1. R-11: `validation/recovery_abort.py` initially reported 4/5 cases recovered.
   A mid-run device disconnect *returned* a partial run record instead of
   raising, so `RunGuard` saw a clean exit and never ran the recovery. The
   default was changed to raise (`LoopConfig.on_disconnect="raise"`, with the
   partial record attached to the exception), `"record"` was kept as an
   explicit opt-in, and the case now passes.
2. R-10: `preflight` left the backend one sensor sample ahead of where it found
   it, because the `sensor_read` check draws a sample and that advances the
   plant's noise stream. The consequence showed up in the README's worked
   example as a parity failure between a pre-flighted backend and an untouched
   one. `preflight` now snapshots on entry, restores on exit, and reports the
   field-by-field comparison as a ninth check (`state_unchanged`).

Both are recorded in `validation/VALIDATION.md` and `CHANGELOG.md`. A
validation suite that finds nothing is not evidence that there is nothing.

**One requirement records a measured result that is weaker than the obvious
hope.** R-16: the learned overrun predictor does not beat the fixed threshold
on the jittery family, because nothing does there; on the bursty family it wins
clearly on calibration and narrowly, within interval, on missed-overrun rate.
The numbers are kept as measured and are repeated in `MODEL_CARD.md` and the
README.
