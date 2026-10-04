"""Property-based tests (Hypothesis) for the algebraic identities.

The identities under test are the ones that must hold for *every* input, not
just the hand-checked cases:

1. Range-rate is invariant under reversing the link direction.
2. Doppler is exactly linear in the carrier frequency.
3. Zero relative velocity gives exactly zero range-rate and zero Doppler.
4. Two-way is exactly twice one-way at unit turnaround ratio.
5. Pre-compensation is exactly the negative of the Doppler shift.
6. Range and range-rate are invariant under a rigid rotation of the frame.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from dopplerkit.doppler import (
    doppler_rate_hz_per_s,
    one_way_doppler_hz,
    precompensation_offset_hz,
    relativistic_fraction_second_order,
    two_way_doppler_hz,
)
from dopplerkit.geometry import State, range_rate_mps, slant_range_m

SETTINGS = settings(max_examples=150, deadline=None)

# allow_subnormal=False throughout: the bit-exact identities below hold in
# the normal float64 range but lose a bit in the subnormal range, where
# 2*x is not exactly representable. That is a property of IEEE-754 gradual
# underflow, not of this package, and it is demonstrated explicitly in
# test_doubling_identity_degrades_in_the_subnormal_range below rather than
# hidden. No physical range-rate is 1e-308 m/s.
_F = {"allow_nan": False, "allow_infinity": False, "allow_subnormal": False}

positions = st.floats(min_value=-1.0e8, max_value=1.0e8, **_F)
velocities = st.floats(min_value=-2.0e4, max_value=2.0e4, **_F)
carriers = st.floats(min_value=1.0e6, max_value=1.0e12, **_F)



def _signed(lo: float, hi: float) -> st.SearchStrategy[float]:
    """Exactly zero, or a magnitude in [lo, hi] of either sign.

    The magnitude floor keeps ``carrier * rate / c`` out of the subnormal
    range (f_c as low as 1e6 Hz and rate as low as 1e-9 m/s gives 3e-12 Hz,
    comfortably normal), while still exercising the exact-zero case.
    """
    return st.one_of(
        st.just(0.0),
        st.floats(min_value=lo, max_value=hi, **_F),
        st.floats(min_value=-hi, max_value=-lo, **_F),
    )


rates = _signed(1.0e-9, 5.0e4)
accels = _signed(1.0e-9, 1.0e4)
vec3 = st.lists(positions, min_size=3, max_size=3)
vel3 = st.lists(velocities, min_size=3, max_size=3)


def _pair(pa, pb, va, vb) -> tuple[State, State]:
    return (
        State(position_m=pa, velocity_mps=va),
        State(position_m=pb, velocity_mps=vb),
    )


@SETTINGS
@given(pa=vec3, pb=vec3, va=vel3, vb=vel3)
def test_range_rate_invariant_under_link_reversal(pa, pb, va, vb):
    """Property 1: rho_dot(A->B) == rho_dot(B->A), bit-exact."""
    assume(np.linalg.norm(np.array(pa) - np.array(pb)) > 1.0)
    a, b = _pair(pa, pb, va, vb)
    assert range_rate_mps(a, b) == range_rate_mps(b, a)
    assert slant_range_m(a, b) == slant_range_m(b, a)


@SETTINGS
@given(pa=vec3, pb=vec3, va=vel3, vb=vel3, carrier=carriers)
def test_doppler_invariant_under_link_reversal_at_equal_carrier(pa, pb, va, vb, carrier):
    """Property 1, consequence: reversing the link does not flip the Doppler sign."""
    assume(np.linalg.norm(np.array(pa) - np.array(pb)) > 1.0)
    a, b = _pair(pa, pb, va, vb)
    assert one_way_doppler_hz(range_rate_mps(a, b), carrier) == one_way_doppler_hz(
        range_rate_mps(b, a), carrier
    )


@SETTINGS
@given(rate=rates, carrier=carriers, scale=st.floats(min_value=0.01, max_value=100.0, **_F))
def test_doppler_is_linear_in_the_carrier(rate, carrier, scale):
    """Property 2: Delta_f(k * f_c) == k * Delta_f(f_c)."""
    assume(carrier * scale <= 1.0e14)
    base = one_way_doppler_hz(rate, carrier)
    scaled = one_way_doppler_hz(rate, carrier * scale)
    assert scaled == pytest.approx(scale * base, rel=1e-12, abs=1e-12)


@SETTINGS
@given(rate=rates, c1=carriers, c2=carriers)
def test_doppler_ratio_equals_carrier_ratio(rate, c1, c2):
    """Property 2, restated: the shift ratio is exactly the carrier ratio."""
    assume(abs(rate) > 1e-6)
    assert one_way_doppler_hz(rate, c1) / one_way_doppler_hz(rate, c2) == pytest.approx(
        c1 / c2, rel=1e-12
    )


@SETTINGS
@given(pa=vec3, pb=vec3, v=vel3, carrier=carriers)
def test_zero_relative_velocity_gives_exactly_zero_doppler(pa, pb, v, carrier):
    """Property 3: both endpoints sharing one velocity gives no Doppler at all."""
    assume(np.linalg.norm(np.array(pa) - np.array(pb)) > 1.0)
    a, b = _pair(pa, pb, v, v)
    assert range_rate_mps(a, b) == 0.0
    assert one_way_doppler_hz(range_rate_mps(a, b), carrier) == 0.0
    assert two_way_doppler_hz(range_rate_mps(a, b), carrier) == 0.0
    assert precompensation_offset_hz(range_rate_mps(a, b), carrier) == 0.0


@SETTINGS
@given(rate=rates, carrier=carriers)
def test_two_way_is_exactly_twice_one_way(rate, carrier):
    """Property 4: bit-exact doubling at unit turnaround ratio."""
    assert two_way_doppler_hz(rate, carrier, 1.0) == 2.0 * one_way_doppler_hz(rate, carrier)


@SETTINGS
@given(accel=accels, carrier=carriers)
def test_two_way_doppler_rate_is_exactly_twice_one_way(accel, carrier):
    """Property 4, for the rate."""
    assert doppler_rate_hz_per_s(accel, carrier, ways=2) == 2.0 * doppler_rate_hz_per_s(
        accel, carrier, ways=1
    )


@SETTINGS
@given(rate=rates, carrier=carriers)
def test_precompensation_is_minus_the_doppler(rate, carrier):
    """Property 5."""
    assert precompensation_offset_hz(rate, carrier) == pytest.approx(
        -one_way_doppler_hz(rate, carrier), rel=1e-15, abs=1e-12
    )


@SETTINGS
@given(
    pa=vec3, pb=vec3, va=vel3, vb=vel3,
    angle=st.floats(min_value=-np.pi, max_value=np.pi, **_F),
)
def test_range_and_range_rate_are_rotation_invariant(pa, pb, va, vb, angle):
    """Property 6: a rigid rotation of the frame changes neither scalar."""
    assume(np.linalg.norm(np.array(pa) - np.array(pb)) > 1.0e3)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    a, b = _pair(pa, pb, va, vb)
    ar = State(position_m=rot @ a.position_m, velocity_mps=rot @ a.velocity_mps)
    br = State(position_m=rot @ b.position_m, velocity_mps=rot @ b.velocity_mps)
    assert slant_range_m(ar, br) == pytest.approx(slant_range_m(a, b), rel=1e-12)
    assert range_rate_mps(ar, br) == pytest.approx(
        range_rate_mps(a, b), rel=1e-9, abs=1e-6
    )


@SETTINGS
@given(rate=rates, speed=st.floats(min_value=0.0, max_value=5.0e4, **_F))
def test_relativistic_term_is_always_second_order_small(rate, speed):
    """The reported correction must never exceed O(beta^2) in magnitude."""
    c = 299792458.0
    frac = relativistic_fraction_second_order(rate, speed)
    bound = (rate / c) ** 2 + speed**2 / (2.0 * c**2)
    assert abs(frac) <= bound + 1e-300


@SETTINGS
@given(speed=st.floats(min_value=1.0, max_value=5.0e4, **_F))
def test_relativistic_term_at_zero_range_rate_is_negative(speed):
    """Pure time dilation always slows the transmitter clock."""
    assert relativistic_fraction_second_order(0.0, speed) < 0.0


def test_doubling_identity_degrades_in_the_subnormal_range():
    """Documented exception to property 4, found by Hypothesis.

    At rho_dot = 2.2250738585072014e-308 m/s and f_c = 1 MHz the one-way shift
    lands in the subnormal range, where 2*x is not exactly representable, so
    the two-way result differs from twice the one-way result in the last bit.
    This is IEEE-754 gradual underflow. It is recorded here because the test
    suite asserts bit-exactness elsewhere and a reader deserves to know where
    that stops.
    """
    rate = 2.2250738585072014e-308
    carrier = 1.0e6
    one = one_way_doppler_hz(rate, carrier)
    two = two_way_doppler_hz(rate, carrier, 1.0)
    assert two != 2.0 * one
    assert two == pytest.approx(2.0 * one, rel=1e-3)
    assert abs(one) < 2.3e-308  # subnormal
