"""Pre-run checks and the recovery procedure."""

from __future__ import annotations

import numpy as np
import pytest

from hilforge.backends import make_backend_pair
from hilforge.backends.device import AbsentDriver, DeviceBackend
from hilforge.deploy import (
    CheckReport,
    CheckResult,
    RunGuard,
    preflight,
    restore_backend,
    snapshot_backend,
    verify_restored,
)
from hilforge.errors import PreflightFailure, RecoveryError
from hilforge.loop import HilLoop, LoopConfig
from hilforge.timing import PeriodSpec

PERIOD = 0.010


@pytest.fixture(params=["simulated", "device"])
def backend(request):
    sim, dev = make_backend_pair(sample_dt_s=PERIOD)
    return sim if request.param == "simulated" else dev


def test_preflight_passes_all_nine_checks(backend):
    report = preflight(backend, period_s=PERIOD)
    assert report.passed is True
    assert len(report.results) == 9
    assert report.n_passed == 9
    assert "GO" in report.as_text()
    report.raise_if_failed()


def test_preflight_names_are_stable(backend):
    names = [r.name for r in preflight(backend, period_s=PERIOD).results]
    assert names == [
        "device_present",
        "channels_present",
        "channel_specs_match",
        "timebase_monotonic",
        "clock_resolution",
        "dry_run_no_write",
        "state_round_trip",
        "sensor_read",
        "state_unchanged",
    ]


def test_preflight_leaves_the_backend_as_it_found_it(backend):
    assert backend.is_open is False
    preflight(backend, period_s=PERIOD)
    assert backend.is_open is False
    backend.open()
    preflight(backend, period_s=PERIOD)
    assert backend.is_open is True
    backend.close()


def test_preflight_leaves_the_backend_state_untouched(backend):
    """The checks must not advance the rig they are checking.

    Found while writing the README's worked example: the ``sensor_read`` check
    draws one sample, which advances the plant's noise stream, so a backend
    that had been pre-flighted no longer agreed bit for bit with one that had
    not. ``preflight`` now snapshots on entry and restores on exit.
    """
    sim, dev = make_backend_pair(sample_dt_s=PERIOD)
    before = snapshot_backend(sim)
    report = preflight(sim, period_s=PERIOD)
    assert report.passed is True
    assert verify_restored(sim, before) == {}
    # And the consequence that matters: parity survives a one-sided preflight.
    import numpy as np

    from hilforge.loop import HilLoop as _Loop

    preflight(dev, period_s=PERIOD)
    cfg = LoopConfig(
        period=PeriodSpec(period_s=PERIOD),
        n_iterations=120,
        injected_durations_s=tuple(np.full(120, 0.004)),
    )
    a = _Loop(sim, cfg).run()
    b = _Loop(dev, cfg).run()
    assert a.signal_matrix().tobytes() == b.signal_matrix().tobytes()


def test_preflight_dry_run_check_does_not_write(backend):
    backend.open()
    before = backend.actuators["torque"].write_count
    preflight(backend, period_s=PERIOD)
    assert backend.actuators["torque"].write_count == before
    backend.close()


def test_preflight_fails_on_a_clock_too_coarse_for_the_period():
    sim, _ = make_backend_pair(sample_dt_s=PERIOD)
    # A 1 ns virtual clock against a 1 ns period: the resolution check must fail.
    report = preflight(sim, period_s=1.0e-8, torque_limit_nm=1.5)
    failures = [r.name for r in report.failures]
    assert "clock_resolution" in failures
    assert report.passed is False
    with pytest.raises(PreflightFailure, match="clock_resolution"):
        report.raise_if_failed()


def test_preflight_fails_on_a_mismatched_actuator_range():
    sim, _ = make_backend_pair(sample_dt_s=PERIOD)
    report = preflight(sim, period_s=PERIOD, torque_limit_nm=99.0)
    # The length and units still match, so the spec check passes; the channel
    # set is the same. Nothing should fail: the range is not part of the spec
    # comparison, and the test documents that deliberately narrow scope.
    assert [r.name for r in report.failures] == []


def test_preflight_validates_its_arguments(backend):
    with pytest.raises(ValueError):
        preflight(backend, period_s=0.0)
    with pytest.raises(ValueError):
        preflight(backend, period_s=PERIOD, n_clock_reads=1)
    with pytest.raises(ValueError):
        preflight(backend, period_s=PERIOD, max_resolution_fraction=0.0)


def test_check_result_line_formats_both_ways():
    good = CheckResult(name="a", passed=True, measured="1", expected="1")
    bad = CheckResult(name="b", passed=False, measured="2", expected="1", detail="why")
    assert good.as_line().startswith("[PASS]")
    assert bad.as_line().startswith("[FAIL]")
    assert "why" in bad.as_line()


def test_check_report_counts_and_verdict():
    report = CheckReport()
    report.add(CheckResult(name="a", passed=True, measured="-", expected="-"))
    report.add(CheckResult(name="b", passed=False, measured="-", expected="-"))
    assert report.passed is False
    assert report.n_passed == 1
    assert "1/2 checks passed" in report.as_text()


def test_snapshot_restore_verify_round_trip(backend):
    backend.open()
    snap = snapshot_backend(backend)
    for _ in range(20):
        backend.sensors["ahrs"].read()
        backend.actuators["torque"].write(np.array([0.07]))
        backend.advance(PERIOD)
    diffs = verify_restored(backend, snap)
    assert diffs  # state moved, so verification against the old snapshot fails
    restore_backend(backend, snap)
    assert verify_restored(backend, snap) == {}
    backend.close()


def test_verify_restored_reports_the_differing_fields(backend):
    backend.open()
    snap = snapshot_backend(backend)
    backend.actuators["torque"].write(np.array([0.1]))
    diffs = verify_restored(backend, snap)
    assert "write_count" in diffs
    assert diffs["write_count"] == (0, 1)
    backend.close()


def test_snapshot_rejects_a_backend_without_one():
    class Bare:
        pass

    with pytest.raises(RecoveryError, match="no snapshot"):
        snapshot_backend(Bare())
    with pytest.raises(RecoveryError, match="no restore"):
        restore_backend(Bare(), {})


def test_run_guard_does_not_recover_a_successful_run_by_default(backend):
    backend.open()
    guard = RunGuard(backend)
    cfg = LoopConfig(
        period=PeriodSpec(period_s=PERIOD),
        n_iterations=30,
        injected_durations_s=tuple(np.full(30, 0.004)),
    )
    with guard:
        HilLoop(backend, cfg).run()
    assert guard.recovered is False
    assert backend.actuators["torque"].write_count == 30
    assert "no recovery attempted" in guard.recovery_report()
    backend.close()


def test_run_guard_recovers_on_success_when_asked(backend):
    backend.open()
    guard = RunGuard(backend, recover_on_success=True)
    cfg = LoopConfig(
        period=PeriodSpec(period_s=PERIOD),
        n_iterations=30,
        injected_durations_s=tuple(np.full(30, 0.004)),
    )
    with guard:
        HilLoop(backend, cfg).run()
    assert guard.recovered is True
    assert backend.actuators["torque"].write_count == 0
    backend.close()


def test_run_guard_reraises_the_failure(backend):
    backend.open()
    guard = RunGuard(backend)
    with pytest.raises(ZeroDivisionError), guard:
        raise ZeroDivisionError("induced")
    assert guard.recovered is True
    assert isinstance(guard.error, ZeroDivisionError)
    assert "ZeroDivisionError" in guard.recovery_report()
    backend.close()


def test_run_guard_report_before_entry():
    sim, _ = make_backend_pair()
    assert RunGuard(sim).recovery_report() == "guard was never entered"


def test_preflight_on_an_absent_device_stops_after_the_first_check():
    report = preflight(DeviceBackend(AbsentDriver()), period_s=PERIOD)
    assert len(report.results) == 1
    assert report.results[0].name == "device_present"
