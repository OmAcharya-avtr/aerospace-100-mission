"""Metric tests with hand-calculated answers."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from softdecode.metrics import (
    BitErrorRate,
    bit_error_rate,
    generalised_mutual_information,
    llr_error,
)


def test_bit_error_rate_hand_calculation():
    # 3 errors in 12 trials: p = 0.25, se = sqrt(0.25*0.75/12) = 0.125
    result = BitErrorRate(3, 12)
    assert result.rate == pytest.approx(0.25)
    assert result.standard_error == pytest.approx(0.125, rel=1e-12)
    assert "3/12" in str(result)


def test_bit_error_rate_zero_errors():
    result = BitErrorRate(0, 100)
    assert result.rate == 0.0
    assert result.standard_error == 0.0


def test_bit_error_rate_from_arrays():
    a = np.array([[0, 1, 1], [0, 0, 1]])
    b = np.array([[0, 0, 1], [1, 0, 1]])
    result = bit_error_rate(a, b)
    assert (result.errors, result.trials) == (2, 6)


def test_bit_error_rate_validation():
    with pytest.raises(ValueError, match="trials"):
        BitErrorRate(0, 0)
    with pytest.raises(ValueError, match="errors"):
        BitErrorRate(5, 3)
    with pytest.raises(ValueError, match="shape mismatch"):
        bit_error_rate(np.zeros(3), np.zeros(4))


def test_llr_error_hand_calculation():
    # errors (-1, +1, +2): rmse = sqrt(6/3) = sqrt(2), max 2, mean signed 2/3
    reference = np.array([1.0, 2.0, 3.0])
    got = np.array([0.0, 3.0, 5.0])
    err = llr_error(got, reference)
    assert err.rmse == pytest.approx(np.sqrt(2.0))
    assert err.max_abs == pytest.approx(2.0)
    assert err.mean_signed == pytest.approx(2.0 / 3.0)
    assert err.reference_rms == pytest.approx(np.sqrt(14.0 / 3.0))
    assert err.relative_rmse == pytest.approx(np.sqrt(2.0) / np.sqrt(14.0 / 3.0))
    assert "rmse" in str(err)


def test_llr_error_shape_validation():
    with pytest.raises(ValueError, match="shape mismatch"):
        llr_error(np.zeros(2), np.zeros(3))


def test_gmi_of_perfect_llrs_is_one():
    bits = np.array([0, 1, 0, 1])
    llr = np.where(bits == 0, 200.0, -200.0)
    assert generalised_mutual_information(llr, bits) == pytest.approx(1.0, abs=1e-12)


def test_gmi_of_zero_llrs_is_zero():
    bits = np.array([0, 1, 0, 1])
    assert generalised_mutual_information(np.zeros(4), bits) == pytest.approx(0.0, abs=1e-12)


def test_gmi_of_confidently_wrong_llrs_is_negative():
    bits = np.array([0, 1])
    llr = np.array([-20.0, 20.0])
    value = generalised_mutual_information(llr, bits)
    # 1 - log2(1 + e**20) = 1 - 20/ln2 = -27.853...
    assert value == pytest.approx(1.0 - 20.0 / np.log(2.0), rel=1e-9)
    assert value < 0.0


def test_gmi_validation():
    with pytest.raises(ValueError, match="shape mismatch"):
        generalised_mutual_information(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match="0/1"):
        generalised_mutual_information(np.zeros(2), np.array([0, 2]))


@given(st.floats(min_value=0.1, max_value=30.0))
@settings(max_examples=40, deadline=None)
def test_gmi_is_monotone_in_llr_magnitude_when_correct(magnitude):
    # Algebraic identity: with the sign always correct, GMI increases with |L|.
    bits = np.array([0, 1, 0, 1])
    sign = 1.0 - 2.0 * bits
    low = generalised_mutual_information(sign * magnitude, bits)
    high = generalised_mutual_information(sign * (magnitude + 0.5), bits)
    assert high >= low
