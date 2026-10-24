"""Dry-run mode: the full path over a recording, and the negative control."""

from __future__ import annotations

from pathlib import Path

import pytest

from timetransfer.acquire import acquire
from timetransfer.backends import SimulatedBackend
from timetransfer.dryrun import (
    dry_run,
    dry_run_negative_control,
    record_exchanges,
)
from timetransfer.errors import ConfigurationError, InsufficientDataError
from timetransfer.exchange import ExchangeSet
from timetransfer.uncertainty import AsymmetrySource

from conftest import make_spec


def test_committed_fixture_is_present_and_not_hardware(recorded: ExchangeSet) -> None:
    assert recorded.n > 100
    assert recorded.backend_kind == "simulated"
    assert recorded.is_hardware is False
    assert recorded.n_lost > 0


def test_dry_run_touches_no_hardware(recorded: ExchangeSet) -> None:
    result = dry_run(recorded)
    assert result.hardware_operations == 0
    assert result.is_hardware is False
    assert result.backend_driver == "guarded-recorded"
    assert result.driver_operations > 0
    assert result.refused_operations == 0
    assert "hardware operations  : 0 (must be 0)" in result.as_text()


def test_dry_run_exercises_the_whole_estimation_path(recorded: ExchangeSet) -> None:
    result = dry_run(recorded, order=2)
    assert result.report.exchanges.n == recorded.n
    assert result.estimate.n == recorded.n
    assert result.fit.order == 2
    assert result.uncertainty.u_combined_s > 0.0
    assert result.uncertainty.n == recorded.n


def test_dry_run_from_a_path(recorded_path: Path) -> None:
    result = dry_run(recorded_path)
    assert result.report.exchanges.n > 100


def test_dry_run_accepts_an_asymmetry_term(recorded: ExchangeSet) -> None:
    result = dry_run(
        recorded,
        asymmetry_source=AsymmetrySource.DECLARED_BOUND,
        asymmetry_bound_s=1.6e-3,
    )
    assert result.uncertainty.accounted
    assert result.uncertainty.u_asymmetry_s > 0.0


def test_dry_run_refuses_a_recording_marked_hardware(recorded: ExchangeSet) -> None:
    recorded.is_hardware = True
    with pytest.raises(ConfigurationError, match="is_hardware=true cannot be replayed"):
        dry_run(recorded)


def test_negative_control_fires(recorded: ExchangeSet) -> None:
    control = dry_run_negative_control(recorded)
    assert control.completed == 0
    assert control.refused > 0
    assert control.raised_insufficient
    assert control.passed


def test_negative_control_refusal_count_equals_the_attempts(recorded: ExchangeSet) -> None:
    control = dry_run_negative_control(recorded)
    # every recorded sequence number, plus the acquire() path's first attempt
    assert control.refused == int(recorded.seq.max()) + 2


def test_record_exchanges_writes_a_replayable_file(tmp_path: Path) -> None:
    backend = SimulatedBackend(make_spec(loss_probability=0.1, seed=55))
    backend.open()
    path, report = record_exchanges(backend, 60, tmp_path / "rec.json", min_exchanges=8)
    backend.close()
    assert path.exists()
    replayed = dry_run(path)
    assert replayed.report.exchanges.n == report.exchanges.n
    assert replayed.report.exchanges.timestamp_bytes() == report.exchanges.timestamp_bytes()


def test_dry_run_raises_when_the_recording_is_too_short() -> None:
    backend = SimulatedBackend(make_spec())
    backend.open()
    short = acquire(backend, 10, min_exchanges=8).exchanges
    backend.close()
    with pytest.raises(InsufficientDataError):
        dry_run(short, min_exchanges=50)
