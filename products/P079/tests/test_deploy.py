"""Pre-flight, recovery after an abort, and recovery after losing the link."""

from __future__ import annotations

import numpy as np
import pytest

from timetransfer.acquire import acquire
from timetransfer.backends import AbsentDriver, DeviceBackend, SimulatedBackend
from timetransfer.deploy import RunGuard, preflight, recover_truncated_estimate
from timetransfer.errors import ConfigurationError, PreflightFailure
from timetransfer.uncertainty import AsymmetrySource

from conftest import make_spec


def test_preflight_passes_on_the_simulated_backend() -> None:
    report = preflight(SimulatedBackend(make_spec()), expect_hardware=False)
    assert report.passed
    assert len(report.checks) == 10
    assert report.failures == ()
    assert "10/10 passed" in report.as_text()
    report.raise_if_failed()


def test_preflight_leaves_the_backend_exactly_as_it_found_it() -> None:
    spec = make_spec(q1=1e-20, q2=1e-26, loss_probability=0.1, seed=71)
    backend = SimulatedBackend(spec)
    before = backend.snapshot()
    preflight(backend, expect_hardware=False)
    after = backend.snapshot()
    assert after["engine"]["clock"]["bit_generator_state"] == (
        before["engine"]["clock"]["bit_generator_state"]
    )
    assert after["engine"]["attempts"] == before["engine"]["attempts"]
    # And the acquisition afterwards must be identical to one on a fresh backend.
    backend.open()
    after_preflight = acquire(backend, 40, min_exchanges=8).exchanges
    backend.close()
    fresh = SimulatedBackend(spec)
    fresh.open()
    untouched = acquire(fresh, 40, min_exchanges=8).exchanges
    fresh.close()
    assert after_preflight.timestamp_bytes() == untouched.timestamp_bytes()


def test_preflight_fails_the_hardware_declaration_when_it_is_wrong() -> None:
    report = preflight(SimulatedBackend(make_spec()), expect_hardware=True)
    assert not report.passed
    names = [c.name for c in report.failures]
    assert "hardware_declaration" in names
    with pytest.raises(PreflightFailure, match="hardware_declaration"):
        report.raise_if_failed()


def test_preflight_reports_an_absent_device_rather_than_raising() -> None:
    report = preflight(DeviceBackend(AbsentDriver()), expect_hardware=False)
    assert not report.passed
    names = [c.name for c in report.failures]
    assert "backend_opens" in names
    assert "probe_acquires" in names
    assert "estimate_finite" in names


def test_preflight_fails_a_loss_budget_it_cannot_meet() -> None:
    report = preflight(
        SimulatedBackend(make_spec(loss_probability=0.4, seed=5)),
        expect_hardware=False,
        probe_attempts=60,
        max_loss_fraction=0.05,
    )
    assert not report.passed
    assert "loss_budget" in [c.name for c in report.failures]


def test_preflight_fails_a_resolution_ceiling_it_cannot_meet() -> None:
    report = preflight(
        SimulatedBackend(make_spec(resolution_ns=1_000_000)),
        expect_hardware=False,
        max_resolution_ns=1000,
    )
    assert not report.passed
    assert "timebase_resolution" in [c.name for c in report.failures]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"probe_attempts": 4}, "probe_attempts"), ({"monotone_reads": 2}, "monotone_reads")],
)
def test_preflight_validates(kwargs: dict[str, int], match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        preflight(SimulatedBackend(make_spec()), expect_hardware=False, **kwargs)


def test_run_guard_restores_after_an_injected_abort() -> None:
    backend = SimulatedBackend(make_spec(q1=1e-20, q2=1e-26, loss_probability=0.05, seed=12))
    backend.open()
    before = backend.snapshot()
    with pytest.raises(RuntimeError, match="injected"):
        with RunGuard(backend) as guard:
            acquire(backend, 30, min_exchanges=8)
            raise RuntimeError("injected mid-acquisition fault")
    assert guard.restored
    assert guard.differing_fields == ()
    assert guard.exception_type == "RuntimeError"
    assert guard.n_snapshot_fields >= 8
    assert backend.snapshot() == before
    assert "restored            : True" in guard.recovery_report()


def test_run_guard_is_inert_when_nothing_raises() -> None:
    backend = SimulatedBackend(make_spec())
    backend.open()
    with RunGuard(backend) as guard:
        acquire(backend, 20, min_exchanges=8)
    assert not guard.restored
    assert guard.exception_type == ""
    assert guard.recovery_report() == "no exception; nothing to recover"


def test_run_guard_restores_five_different_injected_faults() -> None:
    faults: list[type[BaseException]] = [
        RuntimeError,
        ValueError,
        KeyError,
        ZeroDivisionError,
        TimeoutError,
    ]
    restored = 0
    for fault in faults:
        backend = SimulatedBackend(make_spec(q1=1e-20, seed=hash(fault.__name__) % 10000))
        backend.open()
        before = backend.snapshot()
        with pytest.raises(fault):
            with RunGuard(backend) as guard:
                acquire(backend, 25, min_exchanges=8)
                raise fault("injected")
        restored += int(guard.restored and backend.snapshot() == before)
    assert restored == len(faults)


def test_recover_truncated_estimate_labels_the_truncation() -> None:
    backend = SimulatedBackend(make_spec(q1=1e-20, seed=23))
    backend.open()
    partial = acquire(backend, 60, min_exchanges=8).exchanges
    backend.close()
    result = recover_truncated_estimate(partial, 300, order=1)
    assert result.truncated
    assert result.n_used == 60
    assert result.n_planned == 300
    assert result.completed_fraction == pytest.approx(0.2)
    assert result.uncertainty_ratio_to_plan == pytest.approx(np.sqrt(5.0), rel=1e-9)
    assert "window ended early" in result.note
    assert "truncated           : True" in result.as_text()


def test_truncated_estimate_has_a_larger_uncertainty_than_a_full_window() -> None:
    spec = make_spec(q1=1e-20, timestamp_noise_s=2e-8, seed=29)
    backend = SimulatedBackend(spec)
    backend.open()
    full = acquire(backend, 300, min_exchanges=8).exchanges
    backend.close()
    short = full.subset(slice(0, 40))
    long_u = recover_truncated_estimate(full, 300, order=1).uncertainty.u_combined_s
    short_u = recover_truncated_estimate(short, 300, order=1).uncertainty.u_combined_s
    assert short_u > long_u


def test_recover_truncated_estimate_on_a_complete_window_is_not_truncated() -> None:
    backend = SimulatedBackend(make_spec())
    backend.open()
    full = acquire(backend, 50, min_exchanges=8).exchanges
    backend.close()
    result = recover_truncated_estimate(full, 50, order=1)
    assert not result.truncated
    assert result.note == ""
    assert result.uncertainty_ratio_to_plan == pytest.approx(1.0)


def test_recover_truncated_estimate_accepts_an_asymmetry_term() -> None:
    backend = SimulatedBackend(make_spec(asymmetry_s=8e-4))
    backend.open()
    partial = acquire(backend, 30, min_exchanges=8).exchanges
    backend.close()
    result = recover_truncated_estimate(
        partial,
        200,
        asymmetry_source=AsymmetrySource.ESTIMATED,
        asymmetry_stderr_s=1.0e-4,
    )
    assert result.uncertainty.u_asymmetry_s == pytest.approx(1.0e-4)


def test_recover_truncated_estimate_validates() -> None:
    backend = SimulatedBackend(make_spec())
    backend.open()
    partial = acquire(backend, 20, min_exchanges=8).exchanges
    backend.close()
    with pytest.raises(ConfigurationError, match="n_planned"):
        recover_truncated_estimate(partial, 0)
