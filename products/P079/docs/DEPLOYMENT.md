# Deployment and recovery on a Jetson Orin Nano

This document is the procedure the Level 4 measurement will follow. It is
written now, before the board is reachable, because the procedure is part of
the product and because writing it afterwards would let the measurement decide
what the procedure was.

**Nothing in this repository has run on a board.** Every backend reports
`is_hardware == False`, the device backend's only drivers are a loopback stub
and a recording replayer, and the Level 4 table in
`validation/benchmark_record.json` is empty with every status `unmeasured`.
This product is labelled **Level 3, hardware-pending**, and it is not relabelled
on simulated numbers.

---

## 0. What Level 4 needs, exactly

Five figures, measured on the board, with this harness and this record format:

| figure | units | what it means |
|---|---|---|
| `exchange_round_trip_s` | s | wall-clock time for one two-way exchange driven by a real timestamping device |
| `timestamp_resolution_ns` | ns | resolution the board's timestamping path actually resolves, measured on the board |
| `estimator_latency_s` | s | time to run the full estimation path on the board's CPU for one window |
| `peak_rss_bytes` | B | process high-water memory on the board under the acquisition workload |
| `cpu_utilisation_fraction` | 1 | fraction of one Orin Nano core consumed by a continuous 1 Hz acquisition |

The list is in `timetransfer.bench.HARDWARE_STAGES` and is the single source of
truth for it. Nothing else is missing.

**Blocked on owner action:** the board powered on and reachable, its hostname
or IP, and passwordless SSH from the Mac. No cloud session can supply any of
those.

---

## 1. Before the board

Run the whole thing against the simulated backend first. If it does not pass
here it will not pass there, and a failure found on the bench costs an
acquisition window.

```bash
python -m pytest tests/ -q --junitxml=junit.xml
python -m timetransfer preflight
python -m timetransfer parity --n 500
python -m timetransfer dryrun fixtures/recorded_exchange.json
```

`preflight` must print `10/10 passed`. `parity` must print
`timestamp bytes   : equal` and `offset max |diff| : 0.000e+00 s`. `dryrun`
must print `hardware operations  : 0 (must be 0)` and `passed     : True`.

## 2. Board bring-up

1. **Power and network.** Power the Orin Nano, confirm it has an address, and
   confirm passwordless SSH from the Mac. This package does nothing about
   power, flashing or console capture; if you need those, use `labgrid`.
2. **Performance mode.** Put the board in its maximum-clock mode and pin the
   fan, so that a timing measurement is not a measurement of the governor:

   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```

   Record the `nvpmodel` mode and the `jetson_clocks --show` output into the
   benchmark record's `environment_note`. A benchmark without it is not
   interpretable and `BenchmarkRecord` refuses an empty note.
3. **Quiesce.** Stop anything else on the board. Record what was running.
4. **Clock source.** Decide and write down what the remote timescale is. Two
   nodes exchanging timestamps need two clocks; this package estimates the
   difference between them and does not care which is which, but the record
   has to say.

## 3. Install

```bash
git clone https://github.com/OmAcharya-avtr/timetransfer.git
cd timetransfer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -q --junitxml=junit.xml
```

The dependency ranges in `pyproject.toml` are bounded on both sides at the
versions every number in this repository was measured against. If pip resolves
something outside them, the install is not the configuration this product was
validated in and the numbers do not carry over. Install both ends of each range
and diff the committed validation outputs before trusting anything.

## 4. Write the driver

`timetransfer.hal.TimestampDriver` is five methods: `open`, `close`, `probe`,
`exchange`, and the `name` and `touch_count` properties. `exchange(seq)`
returns `(t1, t2, t3, t4)` as **integer nanoseconds** and raises
`ExchangeLostError` when the exchange does not complete.

Three rules, all of which the rest of the package depends on:

- **Integer nanoseconds, not floats.** The estimators difference these four
  numbers four ways and the parity check compares them byte for byte.
- **`name` must not be a stub name.** `timetransfer.hal.STUB_DRIVERS` holds
  the names that mean "not hardware", and a `guarded-` prefix is stripped
  before the comparison. A driver called `loopback` will report
  `is_hardware == False` whatever it is talking to, which is the correct
  failure direction.
- **A lost exchange raises; a missing device raises something else.**
  `ExchangeLostError` is counted by the acquisition and the run continues.
  `DeviceAbsentError` stops the run, because a missing device is a fault and
  counting it as link loss would hide it.

Run the parity check against the real driver as soon as it exists:

```bash
python -m timetransfer parity --n 500
```

Parity between a simulated backend and a *real* device backend will not be
byte-identical -- a real clock is not a seeded model. What the check is for at
that point is that the code path runs, the accounting agrees and the estimates
are the same shape.

## 5. Pre-flight on the board

```bash
python - <<'PY'
from timetransfer.backends import DeviceBackend
from timetransfer.deploy import preflight
from mydriver import OrinNanoDriver          # yours, not this package's

backend = DeviceBackend(OrinNanoDriver(...), name="orin-nano")
report = preflight(backend, expect_hardware=True, probe_attempts=60)
print(report.as_text())
report.raise_if_failed()
PY
```

`expect_hardware=True` is the declaration. It fails if the backend's own
`is_hardware` says otherwise, which is the only guard between a simulated run
and a written-up board measurement.

The ten checks, and what a failure of each means:

| check | a failure means |
|---|---|
| `backend_opens` | the driver could not acquire the device; nothing below is meaningful |
| `hardware_declaration` | you said hardware and the driver says stub, or the reverse; **stop** |
| `timebase_monotone` | the clock went backwards; every duration is invalid |
| `timebase_resolution` | the timebase is coarser than the ceiling you set; the quantisation term `delta/sqrt(12)` will dominate |
| `probe_acquires` | fewer than eight exchanges completed; the link is not usable |
| `timestamp_ordering` | `t4 < t1` or `t3 < t2`; the driver is not returning what it claims |
| `round_trip_positive` | a non-positive round trip; the same |
| `loss_budget` | the link lost more than you allowed; fix the link, do not lower the budget |
| `estimate_finite` | the estimation path did not run on the probe data |
| `state_unchanged` | pre-flight perturbed the backend; the run that follows will not be the run you planned |

`state_unchanged` exists because the probe acquisition advances the models'
generators. It is a check, not an assumption, because it was wrong once.

## 6. The run

Wrap it. `RunGuard` snapshots the backend on entry and, if the body raises,
restores the snapshot and verifies it field by field.

```bash
python - <<'PY'
from timetransfer.acquire import acquire
from timetransfer.deploy import RunGuard, recover_truncated_estimate
from timetransfer.uncertainty import AsymmetrySource
from mydriver import make_backend

PLANNED = 3600
backend = make_backend()
backend.open()
with RunGuard(backend) as guard:
    report = acquire(
        backend,
        PLANNED,
        min_exchanges=60,
        max_consecutive_losses=10,
    )
backend.close()
print(report.as_text())
print(guard.recovery_report())
PY
```

`max_consecutive_losses=10` is the deliberate choice: ten consecutive losses at
1 Hz means the link is gone, not that this exchange was unlucky, and hammering
a dead link wastes the window.

## 7. When the exchange is lost mid-estimate

This is the case the procedure exists for, and it has three outcomes.

### 7a. Isolated losses

Counted, the sequence number is skipped, the run continues. A gap in
`ExchangeSet.seq` is a visible loss. Nothing to do.

### 7b. The link dies part way through

`acquire` stops on the loss run and returns what arrived, with
`stopped_early=True` and `stopped_at_seq` naming the attempt it stopped on.
**Do not throw the data away and do not present it as a complete window.**

```bash
python - <<'PY'
from timetransfer.deploy import recover_truncated_estimate
from timetransfer.uncertainty import AsymmetrySource

result = recover_truncated_estimate(
    report.exchanges,
    PLANNED,
    order=1,
    resolution_ns=backend.timebase.resolution_ns,
    asymmetry_source=AsymmetrySource.DECLARED_BOUND,
    asymmetry_bound_s=YOUR_CALIBRATED_BOUND,
)
print(result.as_text())
PY
```

What comes back is an estimate with `truncated=True`, the fraction of the
planned window that arrived, the uncertainty computed from the exchanges that
actually arrived (serial correlation included, measured from the residuals),
and the independence-model expectation `sqrt(n_planned / n_used)` printed
beside it so the gap between the two is visible.

Measured on the simulated backend in
`validation/validate_deployment_output.txt`: a 600-exchange window truncated at
120 exchanges reports a statistical term 2.15 times the complete window's,
against an independence expectation of 2.24.

Then decide, in this order:

1. **Is `n_used` above your `min_exchanges`?** If not, there is no estimate.
   `acquire` has already raised `InsufficientDataError` carrying `n_usable` and
   `n_required`.
2. **Does the truncated uncertainty still meet your requirement?** If it does,
   the window is usable and the estimate is labelled truncated. If it does not,
   extend or repeat.
3. **Did the window end before the drift was identifiable?** An order-2 fit
   over a short span will absorb low-frequency noise into the drift term and
   understate its own uncertainty. Drop to `order=1` and say so.

### 7c. The run raised

`RunGuard` restores the backend and `recovery_report()` prints what was put
back, field by field. Measured on the simulated backend: five injected faults,
five clean restorations, zero differing snapshot fields
(`validation/validate_deployment_output.txt`). Check the report before
starting the next run; a restoration that did not verify means the next run
starts from an unknown state.

## 8. The benchmark, and the one thing not to do

```bash
python -m timetransfer bench --n 600 --repeats 9 \
    --note "Orin Nano, nvpmodel 0, jetson_clocks on, nothing else running" \
    --json validation/benchmark_record_orin.json
```

The host table is measured. **The Level 4 table stays empty until the five
figures in section 0 are measured on the board.** Three guards enforce it and
all three are exercised in `validation/validate_benchmark_table.py`:

- `BenchmarkRecord(is_hardware=True)` raises `HardwareClaimError`;
- a hardware row carrying a value or a status other than `unmeasured` raises;
- `promote_to_hardware()` always raises on a record from a non-hardware
  backend.

Do not edit the JSON. A number that was not measured where it says it was
measured is worse than no number.

## 9. Afterwards

- Capture the raw `.txt` and `.json` output, unedited.
- Re-run every validation script on the board and diff the outputs against the
  committed ones. A suite that passes while the validation numbers move is the
  failure mode no test catches.
- Only then is the label `Level 4` available, and only for the rows that were
  actually measured.
