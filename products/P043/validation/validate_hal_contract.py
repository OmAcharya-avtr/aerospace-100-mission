"""Validation: the HAL contract and the operational procedures, executed.

This is the Level-3 hardware-pending evidence. It does not and cannot show that
a detector works; it shows that the interface, the preflight checks, the run
capture and the backout path all execute, that the simulated and device backends
satisfy the same contract, and that a simulated number can never be mistaken for
a measurement.

Checks
------
1. **One contract, both backends.** Every clause of
   :class:`photoncount.hal.PhotonCountingBackend` applied to the simulated and
   device backends, with the device expected to refuse by raising
   ``NotImplementedError`` naming what is missing.
2. **Provenance.** ``Acquisition.is_measurement`` is False for every simulated
   and every dry-run result, and the device backend never returns counts.
3. **Simulation and dry-run modes.** Both execute the same command path; the dry
   run reports real wall-clock time and no counts.
4. **Preflight** on a sane configuration, a window too short against the dead
   time, and a rate past the paralyzable maximum.
5. **Capture and backout.** A committed run leaves a record; a crashed run
   leaves an in-progress journal that :func:`backout_incomplete_runs` converts to
   an aborted record without deleting anything.
6. **No absolute paths.** Every filename this package returns or writes into a
   record is checked for a path separator.

**What is still missing for Level 4:** measured latency, memory and throughput
from a Jetson Orin Nano. Nothing below is such a measurement. The numbers in
``benchmark/`` come from a shared cloud container and say so.

Runtime: under 5 s on one core. Writes into a temporary directory, not the repo.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    _SRC = Path(__file__).resolve().parents[1] / "src"
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from photoncount.hal import (  # noqa: E402
    REQUIRED_DESCRIPTION_KEYS,
    AcquisitionRequest,
    BackendMode,
    DeviceBackend,
    SimulatedBackend,
)
from photoncount.ops import (  # noqa: E402
    RunJournal,
    RunRecord,
    backout_incomplete_runs,
    environment_summary,
    read_run_record,
    run_preflight,
)
from photoncount.simulate import DetectorSpec  # noqa: E402

SPEC = DetectorSpec(
    dead_time_s=1e-7,
    model="paralyzable",
    afterpulse_probability=0.04,
    afterpulse_mean_delay_s=4e-7,
)


def _backends():
    return [
        ("simulated", SimulatedBackend(SPEC, 2e5, np.random.default_rng(20261006))),
        (
            "device",
            DeviceBackend(
                dead_time_s=1e-7,
                dead_time_model="paralyzable",
                afterpulse_probability=0.04,
                afterpulse_mean_delay_s=4e-7,
                max_sustained_rate_hz=3.68e6,
                timestamp_resolution_s=1e-10,
            ),
        ),
    ]


def check_contract() -> bool:
    print("1. One contract, both backends")
    ok = True
    for name, backend in _backends():
        desc = backend.describe()
        missing = [k for k in REQUIRED_DESCRIPTION_KEYS if k not in desc]
        print(f"   {name:10s} describe() keys present: {not missing}  "
              f"simulated={desc['simulated']}  hardware_required={desc['hardware_required']}")
        ok &= not missing
        # Clause 2: a malformed request is a ValueError, not a hardware error.
        try:
            backend.acquire("not a request", BackendMode.SIMULATION)  # type: ignore[arg-type]
            verdict = "no exception (FAIL)"
            ok = False
        except ValueError:
            verdict = "ValueError (correct: validated before hardware)"
        except NotImplementedError:
            verdict = "NotImplementedError (FAIL: hardware touched before validation)"
            ok = False
        print(f"   {name:10s} malformed request -> {verdict}")
        # Clause 4: hardware operations refuse by name.
        try:
            backend.open()
            print(f"   {name:10s} open() succeeded; is_open={backend.is_open}")
            backend.close()
        except NotImplementedError as exc:
            named = any(w in str(exc) for w in ("hardware", "physical", "device"))
            print(f"   {name:10s} open() raised NotImplementedError; names what is "
                  f"missing: {named}")
            ok &= named
        backend.close()
        backend.close()  # idempotent
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def check_provenance_and_modes() -> bool:
    print("\n2 & 3. Provenance and the two modes")
    backend = SimulatedBackend(SPEC, 2e5, np.random.default_rng(1))
    ok = True
    with backend:
        sim = backend.acquire(AcquisitionRequest(1e-3, 50, "sim"), BackendMode.SIMULATION)
        dry = backend.acquire(AcquisitionRequest(1e-3, 50, "dry"), BackendMode.DRY_RUN)
    print(f"   simulation: counts={sim.counts.size} rate={sim.observed_rate_hz:.1f} "
          f"fano={sim.fano_factor:.4f} wall={sim.wall_seconds * 1e3:.2f} ms "
          f"is_measurement={sim.is_measurement}")
    print(f"   dry run:    counts={dry.counts.size} rate={dry.observed_rate_hz} "
          f"wall={dry.wall_seconds * 1e3:.2f} ms is_measurement={dry.is_measurement}")
    ok &= sim.is_measurement is False and dry.is_measurement is False
    ok &= dry.counts.size == 0 and dry.wall_seconds > 0.0
    device = DeviceBackend(dead_time_s=1e-7)
    try:
        device.acquire(AcquisitionRequest(1e-3, 10, "dev"), BackendMode.SIMULATION)
        print("   device backend returned counts (FAIL)")
        ok = False
    except NotImplementedError as exc:
        print(f"   device backend refused: {str(exc).splitlines()[0][:70]}...")
    print("   a simulated number can never report is_measurement=True; that predicate")
    print("   is the only thing downstream code needs to consult")
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def check_preflight(tmp: Path) -> bool:
    print("\n4. Preflight")
    ok = True
    cases = (
        ("sane", AcquisitionRequest(1e-3, 50, "sane"), 2e5, True),
        ("window too short", AcquisitionRequest(1e-6, 50, "short"), 2e5, False),
        ("past the paralyzable maximum", AcquisitionRequest(1e-3, 50, "sat"), 2e7, False),
    )
    for label, request, rate, expect_pass in cases:
        backend = SimulatedBackend(SPEC, rate, np.random.default_rng(2))
        report = run_preflight(backend, request, rate, tmp)
        print(f"   {label:30s} passed={report.passed}  "
              f"failures={[c.name for c in report.failures]}")
        for check in report.checks:
            print(f"      {check.status:4s} {check.name:30s} {check.detail}")
        ok &= report.passed is expect_pass
    device_report = run_preflight(
        DeviceBackend(dead_time_s=45e-9, afterpulse_probability=0.01),
        AcquisitionRequest(1e-3, 50, "dev"),
        1e6,
        tmp,
    )
    self_test = next(c for c in device_report.checks if c.name == "backend_self_test")
    print(f"   device backend self-test status: {self_test.status} "
          "(WARN, not FAIL: no hardware is not a broken detector)")
    ok &= self_test.status == "WARN" and device_report.passed is True
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def check_capture_and_backout(tmp: Path) -> bool:
    print("\n5 & 6. Capture, backout, and no absolute paths")
    ok = True
    backend = SimulatedBackend(SPEC, 2e5, np.random.default_rng(3))
    request = AcquisitionRequest(1e-3, 40, "night01")
    report = run_preflight(backend, request, 2e5, tmp)
    journal = RunJournal(request.label, tmp)
    journal_name = journal.begin(
        {"window_s": request.window_s, "n_windows": int(request.n_windows)}
    )
    with backend:
        acq = backend.acquire(request, BackendMode.SIMULATION)
    record = RunRecord.from_acquisition(acq, request, backend, report, seed=3)
    record_name = journal.commit(record)
    print(f"   journal file written: {journal_name}")
    print(f"   record file written:  {record_name}")
    loaded = read_run_record(record_name, tmp)
    print(f"   record fields: {sorted(loaded)}")
    print(f"   record says simulated={loaded['simulated']} mode={loaded['mode']} "
          f"windows={len(loaded['counts'])}")
    ok &= loaded["simulated"] is True and len(loaded["counts"]) == 40

    RunJournal("crashed", tmp).begin({"window_s": 1e-3, "n_windows": 40})
    backed = backout_incomplete_runs(tmp)
    print(f"   backout: {backed}")
    again = backout_incomplete_runs(tmp)
    print(f"   backout is idempotent: second call returned {again}")
    ok &= len(backed) == 1 and again == []
    aborted = json.loads((tmp / "crashed.aborted.json").read_text(encoding="utf-8"))
    print(f"   aborted record state={aborted['state']} (nothing deleted)")
    ok &= aborted["state"] == "aborted" and (tmp / "night01.json").exists()

    strings = [journal_name, record_name, *(v for row in backed for v in row.values())]
    strings += list(environment_summary().values())
    offenders = [s for s in strings if "/" in s or "\\" in s]
    print(f"   filenames and environment strings containing a path separator: "
          f"{offenders}")
    ok &= offenders == []
    print(f"   verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("validate_hal_contract.py")
    print("Level 3, hardware-pending. Nothing here is a hardware measurement.")
    print("=" * 78)
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        results = [
            check_contract(),
            check_provenance_and_modes(),
            check_preflight(tmp),
            check_capture_and_backout(tmp),
        ]
    print("\n" + "=" * 78)
    print(f"checks passed: {sum(results)} of {len(results)}")
    print("What Level 4 still requires: benchmark/run_benchmark.py executed on a")
    print("Jetson Orin Nano, with its raw output file kept. No simulated backend,")
    print("extrapolation or datasheet substitutes for that run.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
