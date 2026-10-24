"""Timebase behaviour and the quantisation error term."""

from __future__ import annotations

import math

import pytest

from timetransfer.errors import ConfigurationError, TimebaseRegressionError
from timetransfer.timebase import (
    NS_PER_S,
    MonotonicTimebase,
    Timebase,
    VirtualTimebase,
    clock_report,
    measure_call_overhead,
    measure_resolution_ns,
    offset_quantisation_uncertainty,
)


def test_ns_per_s_is_exact() -> None:
    assert NS_PER_S == 1_000_000_000


def test_quantisation_uncertainty_known_answer() -> None:
    # Hand calculation: delta = 1 ns = 1e-9 s; u = 1e-9 / sqrt(12).
    # sqrt(12) = 3.4641016151377544, so u = 2.8867513459481288e-10 s.
    assert offset_quantisation_uncertainty(1.0) == pytest.approx(2.8867513459481288e-10, rel=1e-12)


def test_quantisation_uncertainty_scales_linearly() -> None:
    assert offset_quantisation_uncertainty(100.0) == pytest.approx(
        100.0 * offset_quantisation_uncertainty(1.0), rel=1e-12
    )


@pytest.mark.parametrize("bad", [0.0, -1.0, -1e-9])
def test_quantisation_uncertainty_rejects_non_positive(bad: float) -> None:
    with pytest.raises(ConfigurationError, match="resolution_ns"):
        offset_quantisation_uncertainty(bad)


def test_virtual_timebase_is_exact_in_ticks() -> None:
    tb = VirtualTimebase(resolution_ns=10)
    assert tb.now_ns() == 0
    assert tb.advance_ns(25) == 20  # 25/10 = 2.5 ticks; round(2.5) == 2 in Python
    assert tb.advance_ns(30) == 50
    assert tb.resolution_ns == 10
    assert tb.is_virtual


def test_virtual_timebase_rounding_is_bankers() -> None:
    # round(2.5) == 2 in Python: stated here so the behaviour is pinned rather
    # than discovered.
    tb = VirtualTimebase(resolution_ns=10)
    tb.advance_ns(25)
    assert tb.now_ns() == 20


def test_virtual_timebase_refuses_negative_advance() -> None:
    tb = VirtualTimebase()
    with pytest.raises(TimebaseRegressionError, match="negative"):
        tb.advance_ns(-1)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"resolution_ns": 0}, "resolution_ns"), ({"start_ns": -5}, "start_ns")],
)
def test_virtual_timebase_validates(kwargs: dict[str, int], match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        VirtualTimebase(**kwargs)


def test_virtual_timebase_set_ns_allows_a_backwards_step() -> None:
    tb = VirtualTimebase(resolution_ns=1)
    tb.advance_ns(1000)
    tb.set_ns(10)
    assert tb.now_ns() == 10


def test_monotonic_timebase_is_non_decreasing() -> None:
    tb = MonotonicTimebase()
    reads = [tb.now_ns() for _ in range(500)]
    assert all(b >= a for a, b in zip(reads, reads[1:], strict=False))
    assert not tb.is_virtual
    assert tb.resolution_ns >= 1


def test_both_timebases_satisfy_the_protocol() -> None:
    assert isinstance(VirtualTimebase(), Timebase)
    assert isinstance(MonotonicTimebase(), Timebase)


def test_measure_resolution_is_positive() -> None:
    assert measure_resolution_ns(200) >= 1


def test_measure_call_overhead_is_non_negative() -> None:
    assert measure_call_overhead(200) >= 0.0


def test_clock_report_is_self_consistent() -> None:
    report = clock_report(400)
    assert report.n_samples == 400
    assert report.measured_resolution_ns >= 1
    expected = offset_quantisation_uncertainty(report.measured_resolution_ns)
    assert report.offset_quantisation_uncertainty_s == pytest.approx(expected, rel=1e-12)
    assert "Bennett 1948" in report.as_text()
    assert math.isfinite(report.call_overhead_ns)


@pytest.mark.parametrize("bad", [1, 0, -3])
def test_measure_resolution_rejects_small_n(bad: int) -> None:
    with pytest.raises(ConfigurationError, match="n must be"):
        measure_resolution_ns(bad)
