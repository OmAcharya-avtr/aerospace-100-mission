"""Unit conversions: known answers and round trips."""

from __future__ import annotations

import pytest

from latencynet.units import ms_to_s, ns_to_s, s_to_ms, s_to_ns, s_to_us, us_to_s


def test_known_answers():
    # Hand-calculated: 1 ms = 1e-3 s, 250 us = 2.5e-4 s, 1500 ns = 1.5e-6 s.
    assert ms_to_s(1.0) == pytest.approx(1.0e-3, rel=0, abs=0)
    assert us_to_s(250.0) == pytest.approx(2.5e-4, rel=1e-15)
    assert ns_to_s(1500.0) == pytest.approx(1.5e-6, rel=1e-15)
    # 0.002 s = 2 ms = 2000 us = 2_000_000 ns.
    assert s_to_ms(0.002) == pytest.approx(2.0, rel=1e-15)
    assert s_to_us(0.002) == pytest.approx(2000.0, rel=1e-15)
    assert s_to_ns(0.002) == pytest.approx(2.0e6, rel=1e-15)


@pytest.mark.parametrize("value", [0.0, 1.0, 123.456, 1e-9, 1e9])
def test_round_trips(value):
    assert s_to_ms(ms_to_s(value)) == pytest.approx(value, rel=1e-12, abs=1e-300)
    assert s_to_us(us_to_s(value)) == pytest.approx(value, rel=1e-12, abs=1e-300)
    assert s_to_ns(ns_to_s(value)) == pytest.approx(value, rel=1e-12, abs=1e-300)


def test_accepts_integers():
    assert us_to_s(1) == pytest.approx(1e-6, rel=1e-15)
