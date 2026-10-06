"""Pulse shapes: normalisation, symmetry, support, and input validation."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from slotsync.pulses import (
    PulseShape,
    half_sine,
    nyquist_raised_cosine,
    pulse_by_name,
    raised_cosine_time,
    rectangular,
    triangular,
)

ALL_SHAPES = [
    rectangular(1.0),
    triangular(1.0),
    raised_cosine_time(1.0),
    half_sine(1.0),
    nyquist_raised_cosine(0.0, 4.0),
    nyquist_raised_cosine(0.5, 4.0),
    nyquist_raised_cosine(1.0, 4.0),
]


@pytest.mark.parametrize("pulse", ALL_SHAPES, ids=lambda p: p.name)
def test_unit_peak_at_the_centre(pulse: PulseShape) -> None:
    assert pulse.amplitude(np.array([0.0]))[0] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("pulse", ALL_SHAPES, ids=lambda p: p.name)
def test_even_symmetry(pulse: PulseShape) -> None:
    t = np.linspace(0.0, pulse.half_support, 257)
    assert np.allclose(pulse.amplitude(t), pulse.amplitude(-t), atol=1e-14)


@pytest.mark.parametrize("pulse", ALL_SHAPES, ids=lambda p: p.name)
def test_zero_outside_the_support(pulse: PulseShape) -> None:
    outside = np.array(
        [pulse.half_support * 1.0001, pulse.half_support + 1.0, -pulse.half_support - 3.5]
    )
    assert np.all(pulse.amplitude(outside) == 0.0)


def test_triangle_is_exactly_linear_hand_computed() -> None:
    # p(t) = 1 - |t| for half_width 1, so p(0.25) = 0.75 and p(0.75) = 0.25 exactly.
    pulse = triangular(1.0)
    assert pulse.amplitude(np.array([0.25]))[0] == pytest.approx(0.75, abs=1e-15)
    assert pulse.amplitude(np.array([0.75]))[0] == pytest.approx(0.25, abs=1e-15)
    # Energy of a unit-peak triangle of half-width 1 is 2 * int_0^1 (1-t)^2 dt = 2/3.
    assert pulse.energy(200001) == pytest.approx(2.0 / 3.0, rel=1e-8)


def test_half_sine_energy_hand_computed() -> None:
    # p(t) = cos(pi t) on |t| <= 1/2, so int p^2 = int_-1/2^1/2 cos^2(pi t) dt = 1/2.
    assert half_sine(1.0).energy(200001) == pytest.approx(0.5, rel=1e-8)


def test_raised_cosine_in_time_energy_hand_computed() -> None:
    # p(t) = (1 + cos(pi t))/2 on |t| <= 1; int p^2 dt = (1/4) int (1 + cos)^2
    # = (1/4)[2 + 0 + 1] = 3/4 using int_-1^1 cos^2(pi t) dt = 1.
    assert raised_cosine_time(1.0).energy(200001) == pytest.approx(0.75, rel=1e-8)


def test_nyquist_shape_has_zero_crossings_at_integer_symbols() -> None:
    for rolloff in (0.0, 0.25, 0.5, 0.75, 1.0):
        pulse = nyquist_raised_cosine(rolloff, 6.0)
        samples = pulse.amplitude(np.array([1.0, 2.0, 3.0, -1.0, -2.0]))
        assert np.max(np.abs(samples)) < 1e-9, rolloff


def test_nyquist_rolloff_one_half_is_finite_at_the_removable_singularity() -> None:
    # At t = +-1/(2a) the closed form is 0/0; the implementation averages across it.
    pulse = nyquist_raised_cosine(0.5, 4.0)
    value = pulse.amplitude(np.array([1.0]))[0]
    assert np.isfinite(value)
    assert abs(value) < 1e-9


def test_isi_span_rounds_up() -> None:
    assert rectangular(1.0).isi_span_symbols == 1
    assert triangular(1.0).isi_span_symbols == 1
    assert nyquist_raised_cosine(0.5, 4.0).isi_span_symbols == 4


@pytest.mark.parametrize(
    ("factory", "argument"),
    [
        (rectangular, 0.0),
        (rectangular, -1.0),
        (triangular, 0.0),
        (raised_cosine_time, -2.0),
        (half_sine, 0.0),
    ],
)
def test_non_positive_widths_are_rejected(factory, argument: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        factory(argument)


def test_nyquist_argument_validation() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        nyquist_raised_cosine(1.5, 4.0)
    with pytest.raises(ValueError, match="at least 1 symbol"):
        nyquist_raised_cosine(0.5, 0.5)


def test_pulse_shape_rejects_a_non_positive_support() -> None:
    with pytest.raises(ValueError, match="half_support"):
        PulseShape("bad", 0.0, lambda t: np.ones_like(t))


def test_pulse_by_name_round_trip_and_rejection() -> None:
    for name in ("rect", "tri", "rc-time", "half-sine", "nyq-rc"):
        assert pulse_by_name(name).amplitude(np.array([0.0]))[0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unknown pulse shape"):
        pulse_by_name("gaussian")


def test_energy_rejects_a_coarse_grid() -> None:
    with pytest.raises(ValueError, match="at least 16"):
        triangular(1.0).energy(8)


@settings(max_examples=60, deadline=None)
@given(st.floats(min_value=-3.0, max_value=3.0))
def test_every_shape_is_bounded_by_its_peak(t: float) -> None:
    for pulse in ALL_SHAPES:
        value = pulse.amplitude(np.array([t]))[0]
        assert abs(value) <= 1.0 + 1e-12, (pulse.name, t, value)
