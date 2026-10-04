"""Check 5 — the recovery procedure restores pre-run state after an abort.

Requirement: "recovery procedure restores the pre-run state after an induced
mid-run abort".

Four induced aborts, one per abort-capable failure mode, each wrapped in a
:class:`~hilforge.deploy.RunGuard`. For each, the script reports:

* how far the run got before aborting,
* how many state fields the snapshot covers,
* how many of them differ after recovery (must be zero),
* the pre-run and post-recovery values of the three fields a HIL operator
  actually cares about: the actuator write count, the last command issued, and
  the plant's held torque and attitude.

The comparison is field by field over a flattened snapshot that includes the
plant's RNG bit state, so a recovery that put the model back in the right
place but left the noise stream advanced would be reported as a failure.

Run: ``python validation/recovery_abort.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hilforge.backends import make_backend_pair
from hilforge.backends.device import DeviceBackend
from hilforge.backends.stubs import (
    DisconnectingDriver,
    DroppingDriver,
    RejectingDriver,
    SteppingBackDriver,
)
from hilforge.deploy import RunGuard, snapshot_backend
from hilforge.loop import HilLoop, LoopConfig
from hilforge.timing import PeriodSpec

PERIOD = 0.010
N = 400
SEED = 20261004
WARMUP = 25


def _cfg(n: int = N, **kwargs) -> LoopConfig:
    kwargs.setdefault("injected_durations_s", tuple(np.full(n, 0.004)))
    kwargs.setdefault("period", PeriodSpec(period_s=PERIOD))
    return LoopConfig(n_iterations=n, **kwargs)


def _flat_len(snapshot) -> int:
    from hilforge.deploy import _flatten

    return len(_flatten(snapshot))


def _case_disconnect():
    backend = DeviceBackend(
        DisconnectingDriver(seed=SEED, fail_at_read=WARMUP + 12), sample_dt_s=PERIOD
    )
    return (
        "device disconnects mid-run",
        backend,
        _cfg(),
        "DeviceDisconnectedError",
    )


def _case_dropped():
    backend = DeviceBackend(
        DroppingDriver(seed=SEED, drop_at_reads=(WARMUP + 7,)), sample_dt_s=PERIOD
    )
    return (
        "sample dropped, policy=abort",
        backend,
        _cfg(on_dropped_sample="abort"),
        "SampleDroppedError",
    )


def _case_rejected():
    backend = DeviceBackend(
        RejectingDriver(seed=SEED, reject_at_writes=(WARMUP + 5,)), sample_dt_s=PERIOD
    )
    return (
        "actuator write rejected, policy=abort",
        backend,
        _cfg(on_write_rejected="abort"),
        "WriteRejectedError",
    )


def _case_timebase():
    backend = DeviceBackend(
        SteppingBackDriver(seed=SEED, step_back_at_read=WARMUP + 9, step_back_s=0.05),
        sample_dt_s=PERIOD,
    )
    return "timebase steps backwards", backend, _cfg(), "TimebaseRegressionError"


def _case_cascade():
    sim, _ = make_backend_pair(seed=SEED, sample_dt_s=PERIOD)
    cfg = _cfg(
        period=PeriodSpec(period_s=PERIOD, cascade_limit=4),
        injected_durations_s=tuple(np.full(N, 1.3 * PERIOD)),
        on_cascade="abort",
    )
    return "loop overrun cascade", sim, cfg, "OverrunCascadeError"


CASES = (_case_disconnect, _case_dropped, _case_rejected, _case_timebase, _case_cascade)


def main() -> int:
    print("HilForge validation 5 — recovery after an induced mid-run abort")
    print("=" * 74)
    print(f"period {PERIOD:.6e} s   iterations requested {N}   seed {SEED}")
    print("every case warms the rig up first, so the pre-run state is non-trivial")
    print()
    failures = 0
    for maker in CASES:
        label, backend, cfg, expected_exc = maker()
        backend.open()
        # Warm up: the snapshot must capture a rig that has already been used,
        # otherwise "restored" could mean "never touched".
        warm = LoopConfig(
            period=PeriodSpec(period_s=PERIOD),
            n_iterations=WARMUP,
            injected_durations_s=tuple(np.full(WARMUP, 0.004)),
        )
        HilLoop(backend, warm).run()
        before = snapshot_backend(backend)
        guard = RunGuard(backend)
        raised = ""
        completed = -1
        try:
            with guard:
                record = HilLoop(backend, cfg).run()
                completed = record.n_completed
                if record.aborted:
                    raised = "(aborted and recorded, no exception)"
        except Exception as exc:  # noqa: BLE001 - the case is the exception
            raised = type(exc).__name__
        after = snapshot_backend(backend)
        n_fields = _flat_len(before)
        ok_recovered = guard.recovered and not guard.diffs
        ok_exc = (expected_exc is None) or (raised == expected_exc)
        failures += 0 if (ok_recovered and ok_exc) else 1
        print(f"case: {label}")
        print(f"  abort signal            : {raised or 'none'}"
              + (f"  (expected {expected_exc})" if expected_exc else ""))
        print(f"  iterations completed    : {completed}")
        print(f"  snapshot fields compared: {n_fields}")
        print(f"  fields differing after  : {len(guard.diffs)}  (required 0)")
        print(f"  recovery verified       : {guard.recovered}")
        print(f"  write_count  before/after: {before['write_count']} / "
              f"{after['write_count']}")
        plant_before = before.get("plant", before.get("driver", {}).get("plant", {}))
        plant_after = after.get("plant", after.get("driver", {}).get("plant", {}))
        print(f"  held torque  before/after: {plant_before.get('torque')} / "
              f"{plant_after.get('torque')}")
        print(f"  theta [rad]  before/after: {plant_before.get('theta'):.17g} / "
              f"{plant_after.get('theta'):.17g}")
        print(f"  omega [rad/s] before/after: {plant_before.get('omega'):.17g} / "
              f"{plant_after.get('omega'):.17g}")
        print(f"  RESULT                  : "
              f"{'PASS' if (ok_recovered and ok_exc) else 'FAIL'}")
        if guard.diffs:
            for key, (want, got) in sorted(guard.diffs.items()):
                print(f"    DIFF {key}: expected {want!r}, found {got!r}")
        print()
        backend.close()
    print(f"verdict: {len(CASES) - failures}/{len(CASES)} cases recovered exactly")
    print("note: recovery is verified against a simulated rig and a loopback")
    print("      stub. On real hardware the same procedure would have to be")
    print("      re-verified, because a board's state is not all readable.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
