"""The six failure modes required at Level 3, one test class each.

=============================  =======================================
Failure mode                   Class
=============================  =======================================
device absent                  :class:`TestDeviceAbsent`
device disconnects mid-run     :class:`TestMidRunDisconnect`
sample dropped                 :class:`TestDroppedSample`
timebase steps backwards       :class:`TestTimebaseStepsBack`
actuator write rejected        :class:`TestWriteRejected`
loop overrun cascade           :class:`TestOverrunCascade`
=============================  =======================================
"""

from __future__ import annotations

import numpy as np
import pytest

from hilforge.backends.device import AbsentDriver, DeviceBackend
from hilforge.backends.stubs import (
    DisconnectingDriver,
    DroppingDriver,
    RejectingDriver,
    SteppingBackDriver,
)
from hilforge.deploy import RunGuard, preflight
from hilforge.errors import (
    DeviceAbsentError,
    DeviceDisconnectedError,
    OverrunCascadeError,
    SampleDroppedError,
    TimebaseRegressionError,
    WriteRejectedError,
)
from hilforge.loop import HilLoop, LoopConfig
from hilforge.timing import PeriodSpec

PERIOD = 0.010


def _cfg(n: int, **kwargs) -> LoopConfig:
    kwargs.setdefault("injected_durations_s", tuple(np.full(n, 0.004)))
    kwargs.setdefault("period", PeriodSpec(period_s=PERIOD))
    return LoopConfig(n_iterations=n, **kwargs)


class TestDeviceAbsent:
    def test_open_raises_device_absent(self):
        backend = DeviceBackend(AbsentDriver(detail="nothing on I2C 0x68"))
        with pytest.raises(DeviceAbsentError, match="0x68"):
            backend.open()

    def test_run_propagates_device_absent(self):
        backend = DeviceBackend(AbsentDriver())
        with pytest.raises(DeviceAbsentError):
            HilLoop(backend, _cfg(10)).run()

    def test_preflight_reports_no_go_without_raising(self):
        report = preflight(DeviceBackend(AbsentDriver()), period_s=PERIOD)
        assert report.passed is False
        assert report.failures[0].name == "device_present"
        assert "NO-GO" in report.as_text()

    def test_absent_driver_rejects_every_operation(self):
        driver = AbsentDriver()
        for call in (
            lambda: driver.read_channel("ahrs", 0.01),
            lambda: driver.write_channel("torque", np.array([0.0])),
            driver.read_clock,
        ):
            with pytest.raises(DeviceAbsentError):
                call()
        assert driver.channel_specs() == {}
        driver.disconnect()
        driver.advance(0.01)
        assert driver.snapshot() == {}


class TestMidRunDisconnect:
    def test_default_policy_raises_and_carries_the_partial_record(self):
        """A disconnect raises by default, so a RunGuard recovers.

        Found by ``validation/recovery_abort.py``: when the loop *returned* a
        record instead of raising, a guard saw a clean exit and left the rig
        mid-run. The record is attached to the exception so nothing is lost.
        """
        backend = DeviceBackend(DisconnectingDriver(fail_at_read=8), sample_dt_s=PERIOD)
        with pytest.raises(DeviceDisconnectedError) as exc:
            HilLoop(backend, _cfg(40)).run()
        record = exc.value.record
        assert record.aborted is True
        assert record.n_completed == 7
        assert record.abort_reason == "device disconnected mid-run"

    def test_record_policy_returns_the_partial_run(self):
        backend = DeviceBackend(DisconnectingDriver(fail_at_read=8), sample_dt_s=PERIOD)
        record = HilLoop(backend, _cfg(40, on_disconnect="record")).run()
        assert record.aborted is True
        assert record.abort_reason == "device disconnected mid-run"
        assert record.n_completed == 7
        assert record.abort_index == 7
        assert record.overruns is not None
        assert record.overruns.n_iterations == 7

    def test_recovery_restores_the_pre_run_state(self):
        driver = DisconnectingDriver(fail_at_read=6)
        backend = DeviceBackend(driver, sample_dt_s=PERIOD)
        backend.open()
        before = backend.snapshot()
        guard = RunGuard(backend)
        with pytest.raises(DeviceDisconnectedError), guard:
            HilLoop(backend, _cfg(30)).run()
        assert guard.recovered is True
        assert guard.diffs == {}
        after = backend.snapshot()
        assert after["write_count"] == before["write_count"]
        assert after["driver"]["plant"]["theta"] == before["driver"]["plant"]["theta"]
        assert "recovery verified  : True" in guard.recovery_report()

    def test_disconnect_marks_the_link_down(self):
        driver = DisconnectingDriver(fail_at_read=2)
        backend = DeviceBackend(driver, sample_dt_s=PERIOD)
        backend.open()
        backend.sensors["ahrs"].read()
        with pytest.raises(DeviceDisconnectedError):
            backend.sensors["ahrs"].read()
        assert driver.reads == 2
        with pytest.raises(DeviceDisconnectedError):
            backend.sensors["ahrs"].read()

    def test_disconnecting_driver_validates_its_argument(self):
        with pytest.raises(ValueError):
            DisconnectingDriver(fail_at_read=0)


class TestDroppedSample:
    def test_hold_policy_counts_the_drop_and_continues(self):
        driver = DroppingDriver(drop_at_reads=(5, 6, 17))
        backend = DeviceBackend(driver, sample_dt_s=PERIOD)
        record = HilLoop(backend, _cfg(40, on_dropped_sample="hold")).run()
        assert record.aborted is False
        assert record.n_completed == 40
        assert record.dropped_samples == 3
        dropped = [it.index for it in record.iterations if it.dropped]
        assert dropped == [4, 5, 16]
        # The held sample equals the previous iteration's sample, exactly.
        held = record.iterations[4]
        previous = record.iterations[3]
        assert np.array_equal(held.sample, previous.sample)

    def test_abort_policy_raises(self):
        backend = DeviceBackend(DroppingDriver(drop_at_reads=(4,)), sample_dt_s=PERIOD)
        with pytest.raises(SampleDroppedError):
            HilLoop(backend, _cfg(20, on_dropped_sample="abort")).run()

    def test_drop_on_the_first_iteration_always_aborts(self):
        # Nothing to hold, so the hold policy cannot apply.
        backend = DeviceBackend(DroppingDriver(drop_at_reads=(1,)), sample_dt_s=PERIOD)
        with pytest.raises(SampleDroppedError):
            HilLoop(backend, _cfg(20, on_dropped_sample="hold")).run()

    def test_dropping_driver_validates_its_argument(self):
        with pytest.raises(ValueError):
            DroppingDriver(drop_at_reads=(0,))


class TestTimebaseStepsBack:
    def test_loop_raises_on_a_backwards_model_clock(self):
        driver = SteppingBackDriver(step_back_at_read=5, step_back_s=0.05)
        backend = DeviceBackend(driver, sample_dt_s=PERIOD)
        with pytest.raises(TimebaseRegressionError, match="stepped backwards"):
            HilLoop(backend, _cfg(40)).run()
        assert driver.stepped is True

    def test_tolerance_does_not_mask_a_large_step(self):
        driver = SteppingBackDriver(step_back_at_read=3, step_back_s=0.02)
        backend = DeviceBackend(driver, sample_dt_s=PERIOD)
        with pytest.raises(TimebaseRegressionError):
            HilLoop(backend, _cfg(30, guard_tolerance_s=1e-6)).run()

    def test_a_step_smaller_than_the_tolerance_is_clamped(self):
        driver = SteppingBackDriver(step_back_at_read=3, step_back_s=5e-7)
        backend = DeviceBackend(driver, sample_dt_s=PERIOD)
        record = HilLoop(backend, _cfg(20, guard_tolerance_s=1e-6)).run()
        assert record.n_completed == 20
        assert record.aborted is False

    def test_stepping_driver_validates_its_arguments(self):
        with pytest.raises(ValueError):
            SteppingBackDriver(step_back_at_read=0)
        with pytest.raises(ValueError):
            SteppingBackDriver(step_back_s=0.0)


class TestWriteRejected:
    def test_hold_policy_counts_the_refusal_and_continues(self):
        driver = RejectingDriver(reject_at_writes=(3, 9), reason="bus_nak")
        backend = DeviceBackend(driver, sample_dt_s=PERIOD)
        record = HilLoop(backend, _cfg(30, on_write_rejected="hold")).run()
        assert record.aborted is False
        assert record.rejected_writes == 2
        assert record.writes_issued == 28
        rejected = [it.index for it in record.iterations if it.rejected]
        assert rejected == [2, 8]
        assert record.iterations[2].applied is None

    def test_abort_policy_raises_with_the_reason(self):
        driver = RejectingDriver(reject_at_writes=(5,), reason="over_current")
        backend = DeviceBackend(driver, sample_dt_s=PERIOD)
        with pytest.raises(WriteRejectedError) as exc:
            HilLoop(backend, _cfg(20, on_write_rejected="abort")).run()
        assert exc.value.reason == "over_current"

    def test_rejecting_driver_validates_its_argument(self):
        with pytest.raises(ValueError):
            RejectingDriver(reject_at_writes=(0,))


class TestOverrunCascade:
    def test_cascade_limit_aborts_the_run(self):
        # Every iteration takes 1.4 T, so every iteration is late and the
        # lateness compounds: the cascade run reaches the limit at index 2.
        n = 20
        durations = tuple(np.full(n, 1.4 * PERIOD))
        cfg = _cfg(
            n,
            period=PeriodSpec(period_s=PERIOD, cascade_limit=3),
            injected_durations_s=durations,
            on_cascade="abort",
        )
        with pytest.raises(OverrunCascadeError) as exc:
            HilLoop(_make_sim(), cfg).run()
        assert exc.value.run_length == 3
        assert exc.value.first_index == 0

    def test_record_policy_counts_without_aborting(self):
        n = 20
        durations = tuple(np.full(n, 1.4 * PERIOD))
        cfg = _cfg(
            n,
            period=PeriodSpec(period_s=PERIOD, cascade_limit=3),
            injected_durations_s=durations,
            on_cascade="record",
        )
        record = HilLoop(_make_sim(), cfg).run()
        assert record.aborted is False
        assert record.overruns.cascade_count == n
        assert record.overruns.max_consecutive_cascade == n

    def test_cascade_limit_zero_never_aborts(self):
        n = 30
        durations = tuple(np.full(n, 2.0 * PERIOD))
        cfg = _cfg(n, injected_durations_s=durations, on_cascade="abort")
        record = HilLoop(_make_sim(), cfg).run()
        assert record.aborted is False

    def test_an_isolated_overrun_does_not_trip_the_limit(self):
        n = 30
        durations = np.full(n, 0.004)
        durations[10] = 0.012
        cfg = _cfg(
            n,
            period=PeriodSpec(period_s=PERIOD, cascade_limit=2),
            injected_durations_s=tuple(durations),
            on_cascade="abort",
        )
        record = HilLoop(_make_sim(), cfg).run()
        assert record.aborted is False
        assert record.overruns.cascade_count == 1

    def test_recovery_after_a_cascade_abort(self):
        n = 20
        durations = tuple(np.full(n, 1.4 * PERIOD))
        cfg = _cfg(
            n,
            period=PeriodSpec(period_s=PERIOD, cascade_limit=2),
            injected_durations_s=durations,
            on_cascade="abort",
        )
        sim = _make_sim()
        sim.open()
        before = sim.snapshot()
        guard = RunGuard(sim)
        with pytest.raises(OverrunCascadeError), guard:
            HilLoop(sim, cfg).run()
        assert guard.recovered is True
        assert sim.snapshot()["write_count"] == before["write_count"]


def _make_sim():
    from hilforge.backends import make_backend_pair

    sim, _ = make_backend_pair(seed=20261004, sample_dt_s=PERIOD)
    return sim


def test_timebase_regression_carries_the_partial_record():
    """The regression exception carries the run up to the bad timestamp."""
    driver = SteppingBackDriver(step_back_at_read=6, step_back_s=0.05)
    backend = DeviceBackend(driver, sample_dt_s=PERIOD)
    with pytest.raises(TimebaseRegressionError) as exc:
        HilLoop(backend, _cfg(40)).run()
    record = exc.value.record
    assert record.n_completed == 6
    assert record.overruns is not None
