# Changelog

All notable changes to HilForge are documented in this file.

## [0.1.0] - 2026-10-04

Initial release. Validation level **3, hardware-pending**.

### Added

- `hilforge.hal`: the hardware abstraction layer — `ChannelSpec` (name,
  length, units, inclusive range), `SensorChannel`, `ActuatorChannel`,
  `WriteAck` (what was applied, whether it saturated, a per-channel sequence
  number), `Backend`, and `BackendInfo` whose `is_hardware` property is
  `True` only for a device backend with a non-stub driver. `validate_sample`
  and `validate_command` reject the wrong length, a non-finite element, and an
  out-of-range value, each with the channel name and units in the message.
- `hilforge.timebase`: `MonotonicTimebase` over `time.perf_counter_ns` with an
  optional monotonicity guard; `VirtualTimebase` counting integer ticks so a
  seeded run reads no wall clock; `clock_report` which **measures** the
  resolution and the timing-call-pair overhead rather than reading
  `get_clock_info` and trusting it; `duration_resolution_uncertainty`
  (`delta/sqrt(6)`, from the uniform-quantisation variance `delta^2/12`,
  Bennett 1948).
- `hilforge.timing`: `PeriodSpec` with `0 < D <= T` enforced; `MonotonicGuard`;
  `LatencyHistogram` with exact and binned modes and both Hyndman & Fan
  quantile conventions named in the API (`"nearest_rank"` = type 1,
  `"linear"` = type 7), plus `binned_percentile_error_bound_s` so the binning
  error is a value and not a sentence; `overrun_report`, which computes the
  **direct** and **cascade** overrun definitions for every trace because they
  disagree and reporting one as the other is how two tools end up arguing;
  `quantile_standard_error` (Serfling 1980) and `timing_uncertainty` (Type A
  plus a clock-resolution Type B, combined per JCGM 100:2008).
- `hilforge.plant`: `AttitudePlant`, a single-axis rigid body with the **exact**
  ZOH double-integrator transition (Franklin, Powell & Workman 1998 §4.3) and a
  gyro model with bias and angle random walk (IEEE Std 952-2020);
  `snapshot`/`restore` including the PCG64 bit state, which is what lets
  recovery be verified rather than asserted. `PDGains`/`PDController` with the
  closed-loop `wn` and `zeta` exposed (Wie 2008 §7.2).
- `hilforge.backends`: `SimulatedBackend` (plant plus virtual clock);
  `DeviceBackend` over a five-method `DriverShim`; `LoopbackDriver`, which
  wraps the same plant class and is **not hardware**; `AbsentDriver`, which
  always refuses to connect; `make_backend_pair`, which builds one of each on
  one seed so parity is testable. Fault-injection stubs for the five injectable
  failure modes, plus `CountingActuator` with a `forbid_writes` mode.
- `hilforge.loop`: `HilLoop` with four stages (`sense`, `estimate` — a
  single-pole complementary filter, Higgins 1975 — `control`, `actuate`), two
  separate clocks (a model clock that drives the plant and a measurement clock
  that times stages and never touches the data path), per-iteration deadline
  accounting under both definitions, four configurable fault policies, dry-run
  mode, an optional overrun flagger with load shedding, and `RunRecord` with a
  `data_digest` over the data path alone and a `full_digest` that also covers
  timing.
- `hilforge.dryrun`: `DryRunActuator`, which runs the real command path —
  validation and range check — and then discards instead of writing, leaving
  the wrapped channel's `write_count` at zero as the proof.
- `hilforge.replay`: `ReplayBackend` as a third backend over stored samples,
  `save_run`/`load_samples`, and `replay_run`, which reproduces a recorded
  run's data digest exactly.
- `hilforge.deploy`: `preflight` with nine named checks, each carrying a
  measured value and the value required, returning GO/NO-GO rather than
  raising; `snapshot_backend`/`restore_backend`/`verify_restored`; `RunGuard`,
  a context manager that snapshots before a run and, if the run raises,
  restores and verifies field by field without swallowing the exception.
- `hilforge.bench`: `BenchmarkRecord` and `run_benchmark`, which record the
  clock used, its reported **and** measured resolution, the measured
  timing-call bias, the resolution-derived duration uncertainty, the host, a
  **mandatory** environment note, the backend's `is_hardware` flag, and a
  caveat naming what is still required for Level 4. `write_record` emits `.txt`
  for a reader and `.json` for a diff.
- `hilforge.predict`: the AI element and its two baselines, in that order of
  implementation — `FixedThresholdPredictor` (one parameter, auditable F1
  sweep), `QueueingOverrunPredictor` (Otsu 1979 regime split, counted
  transition matrix, per-regime method-of-moments gamma, Chapman-Kolmogorov
  horizon propagation with its independence approximation stated), then
  `LearnedOverrunPredictor` (`HistGradientBoostingClassifier` with isotonic
  calibration on a chronologically held-back slice). `TraceConfig`/
  `generate_trace` for two deliberately contrasting synthetic families,
  18 causal features with a test that proves causality, and
  `evaluate_predictor` reporting lead time, false-alarm rate, missed-overrun
  rate, precision, recall, F1, Brier and AUC with Wilson intervals.
- `hilforge.cli` and `__main__`: nine subcommands — `info`, `preflight`, `run`,
  `dryrun`, `parity`, `replay`, `bench`, `overruns`, `predict` — with non-zero
  exit codes on a failed check, a leaked write or a digest mismatch.
- 264 tests, including 26 failure-mode tests across the six required modes, 13
  property-based tests (Hypothesis) over the timing algebra, and 11 regression
  tests pinning 13 seeded outputs.
- Nine validation scripts with their raw stdout captured, `docs/REQUIREMENTS.md`
  with 21 numbered requirements and a verification matrix, `MODEL_CARD.md`,
  `DATASET_CARD.md`, and four examples each producing a PNG.

### Fixed during the 0.1.0 validation pass

Both of these were found by the validation suite, both were design errors, and
both were fixed by changing the code rather than the tolerance. They are listed
because a validation suite that finds nothing is not evidence that there is
nothing.

- **A mid-run device disconnect returned a partial record instead of raising,
  so `RunGuard` never recovered.** `validation/recovery_abort.py` reported
  4/5 cases recovered on its first run: the loop caught
  `DeviceDisconnectedError`, recorded `aborted=True` and returned, which looks
  like a clean exit to anything that only checks for an exception. Raising is
  now the default (`LoopConfig.on_disconnect="raise"`) with the partial record
  attached to the exception as `.record`; `"record"` is kept as an explicit
  opt-in. The same change attached the partial record to every other aborting
  exception and added `TimebaseRegressionError` to the set that carries it.
  Written up in `validation/VALIDATION.md` §5.
- **`preflight` left the backend one sensor sample ahead of where it found
  it.** The `sensor_read` check draws a sample, which advances the plant's
  noise stream, so a backend that had been pre-flighted no longer agreed bit
  for bit with one that had not — which showed up as a parity failure in the
  README's own worked example. `preflight` now snapshots on entry, restores on
  exit, and reports the field-by-field comparison as a ninth check
  (`state_unchanged`). Written up in `validation/VALIDATION.md` §8 and §11.

### Known at release

- **No hardware measurement exists.** Every backend reports
  `is_hardware == False`; the device backend's only driver is a loopback stub.
  Level 4 requires measured timing and resource use from a Jetson Orin Nano and
  is not claimed anywhere.
- **The learned overrun predictor does not beat the fixed threshold on the
  jittery trace family** — all three predictors sit at ROC AUC 0.50 there — and
  on the bursty family its win on missed-overrun rate (0.0175) is the same
  order as the 95 % interval half-width on that rate (0.0168). Its clear win is
  calibration: Brier 0.05775 against 0.06862, a 15.8 % reduction. The result is
  kept as measured; see `MODEL_CARD.md` §7 and `README.md`.
- Every latency and throughput number was measured on a shared, contended,
  single-core container and is not reproducible between runs on that machine,
  let alone portable. The measurement method and the environment are printed in
  every record for that reason.
