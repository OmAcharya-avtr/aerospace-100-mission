"""Check 4 — dry-run mode provably issues no write.

Requirement: "dry-run mode provably issues no write (verified by a counting
stub, not by inspection)".

The proof is three independent counters, all read after the run rather than
reasoned about before it:

1. ``CountingActuator.write_count`` under the dry-run wrapper, which must be
   0 after a full-length run.
2. ``CountingActuator.attempts``, which counts *attempted* writes including
   refused ones, and must also be 0: the write does not merely fail, it is
   never attempted.
3. The stub is configured with ``forbid_writes=True``, so if anything did
   reach it the run would raise :class:`DryRunViolationError` and this script
   would exit non-zero.

The control is the same run with ``dry_run=False``, whose counters must equal
the iteration count. A zero that is zero in both cases proves nothing.

Run: ``python validation/dryrun_nowrite.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hilforge.backends import make_backend_pair
from hilforge.backends.stubs import CountingActuator
from hilforge.dryrun import DryRunActuator
from hilforge.errors import DryRunViolationError
from hilforge.hal import ChannelSpec
from hilforge.loop import HilLoop, LoopConfig
from hilforge.timing import PeriodSpec

PERIOD = 0.010
N = 5000
SEED = 20261004


class _StubBackend:
    """A backend whose actuator is a forbidding counting stub.

    Everything else comes from the simulated backend, so the loop runs its
    normal path; only the actuator is swapped for the stub that refuses to be
    written to.
    """

    def __init__(self, inner, stub) -> None:
        self._inner = inner
        self._stub = stub

    @property
    def info(self):
        return self._inner.info

    @property
    def timebase(self):
        return self._inner.timebase

    @property
    def sensors(self):
        return self._inner.sensors

    @property
    def actuators(self):
        return {"torque": self._stub}

    @property
    def is_open(self):
        return self._inner.is_open

    def open(self) -> None:
        self._inner.open()

    def close(self) -> None:
        self._inner.close()

    def advance(self, dt_s: float) -> None:
        self._inner.advance(dt_s)


def main() -> int:
    print("HilForge validation 4 — dry-run mode issues no write")
    print("=" * 66)
    print(f"period {PERIOD:.6e} s   iterations {N}   seed {SEED}")
    print()
    failures = 0

    print("4a. dry run against a real simulated actuator")
    sim, _ = make_backend_pair(seed=SEED, sample_dt_s=PERIOD)
    sim.open()
    actuator = sim.actuators["torque"]
    cfg_dry = LoopConfig(
        period=PeriodSpec(period_s=PERIOD),
        n_iterations=N,
        dry_run=True,
        injected_durations_s=tuple(np.full(N, 0.004)),
    )
    torque_before = sim.plant.snapshot()["torque"]
    dry = HilLoop(sim, cfg_dry).run()
    torque_after = sim.plant.snapshot()["torque"]
    for label, want, got in (
        ("actuator write_count", 0, actuator.write_count),
        ("record writes_issued", 0, dry.writes_issued),
        ("record rehearsed_writes", N, dry.rehearsed_writes),
        ("iterations completed", N, dry.n_completed),
        ("plant held torque unchanged", torque_before, torque_after),
    ):
        ok = want == got
        failures += 0 if ok else 1
        print(f"  {label:<30} expected={want!s:<8} measured={got!s:<8} "
              f"{'PASS' if ok else 'FAIL'}")
    sim.close()
    print()

    print("4b. the control: the same run with dry_run=False")
    sim2, _ = make_backend_pair(seed=SEED, sample_dt_s=PERIOD)
    sim2.open()
    actuator2 = sim2.actuators["torque"]
    cfg_wet = LoopConfig(
        period=PeriodSpec(period_s=PERIOD),
        n_iterations=N,
        dry_run=False,
        injected_durations_s=tuple(np.full(N, 0.004)),
    )
    wet = HilLoop(sim2, cfg_wet).run()
    for label, want, got in (
        ("actuator write_count", N, actuator2.write_count),
        ("record writes_issued", N, wet.writes_issued),
        ("record rehearsed_writes", 0, wet.rehearsed_writes),
    ):
        ok = want == got
        failures += 0 if ok else 1
        print(f"  {label:<30} expected={want!s:<8} measured={got!s:<8} "
              f"{'PASS' if ok else 'FAIL'}")
    sim2.close()
    print()

    print("4c. dry run against a counting stub configured to forbid writes")
    sim3, _ = make_backend_pair(seed=SEED, sample_dt_s=PERIOD)
    spec = ChannelSpec(name="torque", length=1, units="N*m", lower=-1.5, upper=1.5)
    stub = CountingActuator(spec, forbid_writes=True)
    backend = _StubBackend(sim3, stub)
    raised = ""
    try:
        record = HilLoop(backend, cfg_dry).run()
        completed = record.n_completed
        rehearsed = record.rehearsed_writes
    except DryRunViolationError as exc:
        raised = str(exc)
        completed = -1
        rehearsed = -1
    for label, want, got in (
        ("DryRunViolationError raised", "", raised),
        ("stub write_count", 0, stub.write_count),
        ("stub attempts", 0, stub.attempts),
        ("iterations completed", N, completed),
        ("rehearsed writes", N, rehearsed),
    ):
        ok = want == got
        failures += 0 if ok else 1
        print(f"  {label:<30} expected={want!r:<10} measured={got!r:<10} "
              f"{'PASS' if ok else 'FAIL'}")
    print()

    print("4d. the stub does raise when written to directly")
    direct_stub = CountingActuator(spec, forbid_writes=True)
    try:
        direct_stub.write(np.array([0.0]))
        caught = False
    except DryRunViolationError:
        caught = True
    failures += 0 if caught else 1
    print(f"  {'raises on a direct write':<30} expected={True!s:<8} "
          f"measured={caught!s:<8} {'PASS' if caught else 'FAIL'}")
    print(f"  {'attempts after the raise':<30} expected={1!s:<8} "
          f"measured={direct_stub.attempts!s:<8} "
          f"{'PASS' if direct_stub.attempts == 1 else 'FAIL'}")
    failures += 0 if direct_stub.attempts == 1 else 1
    print()

    print("4e. the dry-run wrapper still validates and clips")
    guard = DryRunActuator(CountingActuator(spec, forbid_writes=True))
    ack = guard.write(np.array([9.0]))
    clipped_ok = abs(float(ack.applied[0]) - 1.5) < 1e-15 and ack.saturated
    failures += 0 if clipped_ok else 1
    print(f"  {'command 9.0 N*m clipped to':<30} {float(ack.applied[0]):.3f} N*m, "
          f"saturated={ack.saturated}  {'PASS' if clipped_ok else 'FAIL'}")
    rejected = 0
    for bad in (np.array([np.nan]), np.array([0.1, 0.2])):
        try:
            guard.write(bad)
        except (ValueError, TypeError):
            rejected += 1
    failures += 0 if rejected == 2 else 1
    print(f"  {'invalid commands rejected':<30} expected=2        measured={rejected}"
          f"        {'PASS' if rejected == 2 else 'FAIL'}")
    print(f"  {'inner stub attempts':<30} expected=0        "
          f"measured={guard.inner.attempts}        "
          f"{'PASS' if guard.inner.attempts == 0 else 'FAIL'}")
    failures += 0 if guard.inner.attempts == 0 else 1
    print()
    print(f"verdict: {'PASS' if failures == 0 else f'{failures} FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
