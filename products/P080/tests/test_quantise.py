"""Quantiser unit tests, including hand-calculated known answers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from boundcompress import quantise as Q


def test_known_answer_absolute_step_and_symbols() -> None:
    """Hand-calculated. bound=0.25 -> step=0.5; x=[0.0, 0.24, 0.26, 1.0, -0.8].

    q = round(x/0.5) = [0, 0, 1, 2, -2]  (0.24/0.5 = 0.48 -> 0; 0.26/0.5 = 0.52 -> 1)
    xhat = q*0.5  = [0.0, 0.0, 0.5, 1.0, -1.0]
    |err|         = [0.0, 0.24, 0.24, 0.0, 0.2]  -> all <= 0.25
    """
    values = np.array([0.0, 0.24, 0.26, 1.0, -0.8])
    spec = Q.make_spec(values, "abs", 0.25)
    assert spec.step == 0.5
    assert spec.absolute_bound == 0.25
    symbols, reconstruction, corrections = Q.quantise(values, spec)
    assert symbols.tolist() == [0, 0, 1, 2, -2]
    assert reconstruction.tolist() == [0.0, 0.0, 0.5, 1.0, -1.0]
    assert corrections == 0
    assert np.max(np.abs(values - reconstruction)) <= 0.25


def test_known_answer_relative_step_is_two_log1p() -> None:
    """mode='rel' with r=0.01 gives step = 2*ln(1.01).

    ``2 * math.log1p(0.01)`` is 0.019900661706336167 in float64. The first
    version of this test asserted 0.01990049875623903, which is wrong in the
    seventh significant figure; the test caught the test.
    """
    values = np.array([1.0, 2.0, 100.0])
    spec = Q.make_spec(values, "rel", 0.01)
    assert spec.step == pytest.approx(0.019900661706336167, rel=1e-15)
    assert spec.transform == Q.TRANSFORM_LOG
    assert spec.sign == 1
    assert math.isnan(spec.absolute_bound)


def test_known_answer_range_relative_derives_the_absolute_bound() -> None:
    """range=4.0, r=0.005 -> effective absolute bound 0.02, step 0.04."""
    values = np.array([-1.0, 0.0, 3.0])
    spec = Q.make_spec(values, "range_rel", 0.005)
    assert spec.step == pytest.approx(0.04, rel=1e-15)
    assert spec.absolute_bound == pytest.approx(0.02, rel=1e-15)


def test_relative_mode_sign_is_recorded_for_negative_arrays() -> None:
    values = np.array([-5.0, -2.0, -9.0])
    spec = Q.make_spec(values, "rel", 0.01)
    assert spec.sign == -1
    symbols, reconstruction, _ = Q.quantise(values, spec)
    assert np.all(reconstruction < 0.0)
    assert np.all(np.abs(values - reconstruction) <= 0.01 * np.abs(values))


@pytest.mark.parametrize(
    ("mode", "bound", "message"),
    [
        ("nope", 1.0, "mode must be one of"),
        ("abs", 0.0, "strictly positive"),
        ("abs", -1.0, "strictly positive"),
        ("abs", float("inf"), "finite"),
        ("abs", float("nan"), "finite"),
        ("rel", 1.0, "100 per cent"),
        ("range_rel", 2.0, "100 per cent"),
    ],
)
def test_invalid_bounds_are_refused(mode: str, bound: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Q.validate_bound(mode, bound)


def test_relative_mode_refuses_a_zero_sample() -> None:
    with pytest.raises(ValueError, match="undefined at zero"):
        Q.make_spec(np.array([1.0, 0.0, 2.0]), "rel", 0.01)


def test_relative_mode_refuses_a_sign_change() -> None:
    with pytest.raises(ValueError, match="share one sign"):
        Q.make_spec(np.array([1.0, -2.0]), "rel", 0.01)


def test_range_relative_refuses_a_constant_array() -> None:
    with pytest.raises(ValueError, match="non-zero dynamic range"):
        Q.make_spec(np.full(16, 3.0), "range_rel", 0.01)


@pytest.mark.parametrize(
    ("array", "message"),
    [
        (np.zeros((2, 2)), "1-D array"),
        (np.array([]), "at least one sample"),
        (np.array([1.0, float("nan")]), "non-finite"),
        (np.array([float("inf")]), "non-finite"),
    ],
)
def test_invalid_arrays_are_refused(array: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Q.check_array(array)


def test_corrections_are_counted_and_the_bound_still_holds() -> None:
    """A bound so fine that double rounding bites, on a large-magnitude array.

    Measured here rather than asserted to be zero: the point of the correction
    loop is that it is sometimes needed, and the count is reported in the
    certificate for exactly that reason.
    """
    values = np.array([1e9 + index * 0.5 for index in range(64)])
    spec = Q.make_spec(values, "abs", 1e-7)
    symbols, reconstruction, corrections = Q.quantise(values, spec)
    assert corrections >= 0
    assert np.all(np.abs(values - reconstruction) <= 1e-7)
    assert symbols.dtype == np.int64


def test_dequantise_is_the_only_decoder_arithmetic() -> None:
    """``dequantise(symbols)`` reproduces the encoder's reconstruction exactly."""
    values = np.linspace(2.0, 9.0, 300)
    spec = Q.make_spec(values, "abs", 1e-3)
    symbols, reconstruction, _ = Q.quantise(values, spec)
    assert np.array_equal(Q.dequantise(symbols, spec), reconstruction)


def test_bound_not_achievable_is_raised_rather_than_certified() -> None:
    """A bound too fine for the array's magnitude must refuse, not wrap.

    Constructed by hand: a sample at 1e300 with an absolute bound of 1e-300
    needs an integer symbol near 5e599, which does not fit int64. The first
    version of this code cast it anyway and ``OverflowError`` escaped from the
    correction loop; see tests/test_regressions.py.
    """
    values = np.array([1e300, 1e300])
    spec = Q.make_spec(values, "abs", 1e-300)
    with pytest.raises(Q.BoundNotAchievable, match="does not fit int64"):
        Q.quantise(values, spec)


def test_admissible_has_no_tolerance() -> None:
    """A reconstruction exactly at the bound is admissible; one ulp past is not."""
    values = np.array([1.0])
    spec = Q.make_spec(values, "abs", 0.25)
    assert bool(Q.admissible(values, np.array([1.25]), spec)[0])
    assert not bool(Q.admissible(values, np.array([np.nextafter(1.25, 2.0)]), spec)[0])
