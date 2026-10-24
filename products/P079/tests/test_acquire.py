"""Acquisition: loss accounting, early stop, and the insufficient-data path."""

from __future__ import annotations

import numpy as np
import pytest

from timetransfer.acquire import acquire
from timetransfer.backends import SimulatedBackend, SimulationSpec
from timetransfer.errors import ConfigurationError, InsufficientDataError
from timetransfer.linkmodel import LinkSpec

from conftest import make_spec


def _open(**kwargs) -> SimulatedBackend:
    backend = SimulatedBackend(make_spec(**kwargs))
    backend.open()
    return backend


def test_clean_link_completes_every_attempt() -> None:
    report = acquire(_open(), 100, min_exchanges=8)
    assert report.exchanges.n == 100
    assert report.n_lost == 0
    assert report.longest_loss_run == 0
    assert report.completion_fraction == 1.0
    assert not report.stopped_early
    assert report.stopped_at_seq == -1


def test_lost_exchanges_leave_a_gap_in_the_sequence_numbers() -> None:
    report = acquire(_open(loss_probability=0.3, seed=19), 300, min_exchanges=8)
    assert report.n_lost > 0
    assert report.exchanges.n + report.n_lost == 300
    gaps = np.diff(report.exchanges.seq)
    assert int(np.max(gaps)) > 1
    assert report.exchanges.loss_fraction == pytest.approx(report.n_lost / 300.0)


def _outage_backend(window: tuple[float, float], *, seed: int = 3) -> SimulatedBackend:
    """A backend whose link is in a scheduled outage over ``window``."""
    base = make_spec(seed=seed)
    spec = SimulationSpec(
        clock=base.clock,
        link=LinkSpec(
            floor_ab_s=base.link.floor_ab_s,
            floor_ba_s=base.link.floor_ba_s,
            jitter_scale_ab_s=base.link.jitter_scale_ab_s,
            jitter_scale_ba_s=base.link.jitter_scale_ba_s,
            outage_windows=(window,),
        ),
        period_s=base.period_s,
        timestamp_noise_s=base.timestamp_noise_s,
        resolution_ns=base.resolution_ns,
        seed=base.seed,
    )
    backend = SimulatedBackend(spec)
    backend.open()
    return backend


def test_early_stop_on_a_loss_run() -> None:
    # An outage window from 20 s onward guarantees a long loss run at 1 Hz.
    backend = _outage_backend((20.0, 1000.0))
    report = acquire(backend, 200, min_exchanges=8, max_consecutive_losses=5)
    assert report.stopped_early
    assert report.stopped_at_seq == 24
    assert report.longest_loss_run == 5
    assert report.exchanges.n == 20


def test_insufficient_data_carries_both_counts() -> None:
    backend = _outage_backend((0.0, 1000.0))
    with pytest.raises(InsufficientDataError) as excinfo:
        acquire(backend, 50, min_exchanges=8)
    assert excinfo.value.n_usable == 0
    assert excinfo.value.n_required == 8


def test_truth_columns_are_present_for_a_simulated_backend() -> None:
    report = acquire(_open(asymmetry_s=8e-4, jitter_scale_s=0.0), 40, min_exchanges=8)
    assert report.exchanges.true_offset_s is not None
    assert report.exchanges.true_asymmetry_s is not None
    assert report.exchanges.true_asymmetry_s.size == report.exchanges.n
    assert float(np.mean(report.exchanges.true_asymmetry_s)) == pytest.approx(8e-4, rel=1e-6)


def test_truth_columns_are_absent_when_not_requested() -> None:
    report = acquire(_open(), 40, min_exchanges=8, keep_truth=False)
    assert report.exchanges.true_offset_s is None


def test_acquire_requires_an_open_backend() -> None:
    backend = SimulatedBackend(make_spec())
    with pytest.raises(ConfigurationError, match="open backend"):
        acquire(backend, 10)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_attempts": 0}, "n_attempts"),
        ({"n_attempts": 10, "min_exchanges": 0}, "min_exchanges"),
        ({"n_attempts": 10, "max_consecutive_losses": 0}, "max_consecutive_losses"),
    ],
)
def test_acquire_validates(kwargs: dict[str, int], match: str) -> None:
    backend = _open()
    attempts = kwargs.pop("n_attempts")
    with pytest.raises(ConfigurationError, match=match):
        acquire(backend, attempts, **kwargs)


def test_report_text_names_the_backend() -> None:
    report = acquire(_open(), 20, min_exchanges=8)
    text = report.as_text()
    assert "is_hardware        : False" in text
    assert "backend            : simulated" in text
