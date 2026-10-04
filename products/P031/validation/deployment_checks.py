"""Check 9 — the deployment procedure as executable checks.

Level 4 entry plan item 4: "deployment and recovery procedures, written as
executable checks rather than prose: what is verified before a run, what is
captured during it, and how a half-finished run is backed out."

This script runs the pre-run checks against every backend this repository can
produce and prints the reports verbatim, including the NO-GO case. Recovery —
the backing-out half — is ``validation/recovery_abort.py``.

What is verified before a run (nine checks, in order)
-----------------------------------------------------
1. ``device_present``      the backend opens and a driver answers
2. ``channels_present``    the expected channel names exist
3. ``channel_specs_match`` lengths and units match what the loop expects
4. ``timebase_monotonic``  200 reads with no regression
5. ``clock_resolution``    resolution <= 1 % of the intended period
6. ``dry_run_no_write``    a rehearsal leaves the write count unchanged
7. ``state_round_trip``    the backend can be snapshotted and restored
8. ``sensor_read``         one finite sample of the right length
9. ``state_unchanged``     the checks themselves left the rig as they found it

What is captured during a run
-----------------------------
Per iteration: model timestamp, four stage durations, total, completion,
absolute deadline, lateness, both overrun flags, the dropped/rejected/
saturated flags, the estimate, and (optionally) the sample, command and
applied command. Per run: per-stage latency histograms, both overrun accounts,
the counters, the abort reason and index, and two digests. This script prints
the field list from the record itself rather than from a docstring, so the
list cannot drift from the code.

Run: ``python validation/deployment_checks.py``
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hilforge.backends import make_backend_pair
from hilforge.backends.device import AbsentDriver, DeviceBackend
from hilforge.deploy import preflight, snapshot_backend
from hilforge.loop import HilLoop, IterationRecord, LoopConfig, RunRecord
from hilforge.timing import PeriodSpec

PERIOD = 0.010
SEED = 20261004


def main() -> int:
    print("HilForge validation 9 — deployment pre-run checks")
    print("=" * 72)
    print(f"intended period {PERIOD:.6e} s")
    print()
    sim, dev = make_backend_pair(seed=SEED, sample_dt_s=PERIOD)
    absent = DeviceBackend(AbsentDriver(detail="no device on the configured bus"))
    expectations = {
        "simulated plant": (sim, True),
        "loopback device stub": (dev, True),
        "absent device": (absent, False),
    }
    failures = 0
    for label, (backend, should_pass) in expectations.items():
        report = preflight(backend, period_s=PERIOD)
        ok = report.passed == should_pass
        failures += 0 if ok else 1
        print(f"backend: {label}")
        print(f"  is_hardware : {backend.info.is_hardware}")
        for line in report.as_text().splitlines():
            print(f"  {line}")
        print(f"  expected verdict: {'GO' if should_pass else 'NO-GO'}  "
              f"-> {'PASS' if ok else 'FAIL'}")
        print()

    print("a deliberately impossible requirement, to show the checks can fail")
    report = preflight(sim, period_s=1.0e-8)
    failed_names = [r.name for r in report.failures]
    expected_failure = failed_names == ["clock_resolution"]
    failures += 0 if expected_failure else 1
    print(f"  period 1e-8 s against a 1e-9 s clock: failures = {failed_names}")
    print(f"  (1 % of 1e-8 s is 1e-10 s, finer than the clock)  "
          f"{'PASS' if expected_failure else 'FAIL'}")
    print()

    print("what is captured during a run — field list read from the code")
    sim2, _ = make_backend_pair(seed=SEED, sample_dt_s=PERIOD)
    cfg = LoopConfig(
        period=PeriodSpec(period_s=PERIOD),
        n_iterations=200,
        injected_durations_s=tuple(np.full(200, 0.004)),
    )
    record = HilLoop(sim2, cfg).run()
    print("  per iteration (IterationRecord):")
    for f in fields(IterationRecord):
        print(f"    {f.name}")
    print("  per run (RunRecord):")
    for f in fields(RunRecord):
        print(f"    {f.name}")
    print()
    print("  example run summary")
    for key, value in record.summary().items():
        print(f"    {key:<26}: {value}")
    print(f"    {'data_digest':<26}: {record.data_digest()}")
    print(f"    {'full_digest':<26}: {record.full_digest()}")
    print()
    print("state captured for backing out a half-finished run")
    sim2.open()
    snap = snapshot_backend(sim2)
    def walk(d, prefix=""):
        for key, value in d.items():
            if isinstance(value, dict):
                walk(value, f"{prefix}{key}.")
            else:
                shown = repr(value)
                if len(shown) > 60:
                    shown = shown[:57] + "..."
                print(f"    {prefix}{key:<20}: {shown}")
    walk(snap)
    sim2.close()
    print()
    print(f"verdict: {'PASS' if failures == 0 else f'{failures} FAILURES'}")
    print("note: these checks verify the harness against a simulated rig and a")
    print("      loopback stub. Against real hardware they would verify the")
    print("      rig; nothing here has been run against real hardware.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
