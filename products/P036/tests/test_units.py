"""Unit-conversion tests, including exact decimal cases done by hand."""

from __future__ import annotations

import math

import pytest

from rtclock.units import (
    MS_PER_S,
    NS_PER_S,
    US_PER_S,
    frequency_to_period,
    ms_to_s,
    ns_to_s,
    period_to_frequency,
    s_to_ms,
    s_to_ns,
    s_to_us,
    us_to_s,
)


def test_constants_are_exact_decimal_powers():
    assert NS_PER_S == 1_000_000_000
    assert US_PER_S == 1_000_000
    assert MS_PER_S == 1_000


# Hand computation: 1 ms = 1e-3 s = 1e6 ns exactly; 1e-3 * 1e9 = 1e6 is exact in
# binary64 because 1e6 < 2**53 and the product rounds to the nearest
# representable value, which is 1e6 itself.
def test_known_answer_millisecond_to_nanosecond_chain():
    assert s_to_ns(ms_to_s(1.0)) == 1_000_000.0
    assert s_to_us(ms_to_s(1.0)) == 1_000.0
    assert s_to_ms(ms_to_s(1.0)) == 1.0


# Hand computation: 400 Hz -> T = 1/400 = 0.0025 s = 2.5 ms = 2500 us.
def test_known_answer_400_hz_period():
    t = frequency_to_period(400.0)
    assert t == 0.0025
    assert s_to_ms(t) == 2.5
    assert s_to_us(t) == 2500.0
    assert period_to_frequency(t) == 400.0


# Hand computation: 1/3 s period -> 3 Hz exactly? 1/3 is not representable, so
# the round trip is only exact to one ulp. Assert the ulp bound, not equality.
def test_non_representable_period_round_trip_within_one_ulp():
    t = 1.0 / 3.0
    f = period_to_frequency(t)
    assert abs(f - 3.0) <= math.ulp(3.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, -1e-30])
def test_frequency_to_period_rejects_non_positive(bad):
    with pytest.raises(ValueError, match="must be > 0 Hz"):
        frequency_to_period(bad)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_period_to_frequency_rejects_non_positive(bad):
    with pytest.raises(ValueError, match="must be > 0 s"):
        period_to_frequency(bad)


@pytest.mark.parametrize("fn", [s_to_ns, ns_to_s, s_to_us, us_to_s, s_to_ms, ms_to_s])
def test_conversions_reject_nan_and_inf(fn):
    with pytest.raises(ValueError, match="NaN"):
        fn(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        fn(float("inf"))


@pytest.mark.parametrize("fn", [s_to_ns, ns_to_s, frequency_to_period])
def test_conversions_reject_non_numeric(fn):
    with pytest.raises(TypeError, match="real number"):
        fn("1.0")
    with pytest.raises(TypeError, match="real number"):
        fn(True)
