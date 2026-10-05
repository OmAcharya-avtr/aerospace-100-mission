"""Layout discovery and single-bit-flip prediction.

Known-answer values are hand-computed from the IEEE 754 formula
``(-1)**s * 2**(E - bias) * (1 + m / 2**nmant)`` and written out in the test
comments, not taken from any library.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bitflipsim.bitlayout import (
    bit_site_count,
    exponent_field,
    exponent_scale_factor,
    flip_bit,
    float_layout,
    from_bits,
    int_layout,
    mantissa_field,
    predict_flip,
    to_bits,
    verify_float_layout,
)

FLOATS = ("float16", "float32", "float64")


@pytest.mark.parametrize(
    ("name", "total", "nmant", "nexp", "bias"),
    [
        # IEEE 754-2019 binary16 / binary32 / binary64 interchange formats.
        ("float16", 16, 10, 5, 15),
        ("float32", 32, 23, 8, 127),
        ("float64", 64, 52, 11, 1023),
    ],
)
def test_layout_matches_the_standard(name, total, nmant, nexp, bias):
    layout = float_layout(name)
    assert layout.total_bits == total
    assert layout.mantissa_bits == nmant
    assert layout.exponent_bits == nexp
    # bias = 2**(nexp - 1) - 1: 2**4-1=15, 2**7-1=127, 2**10-1=1023
    assert layout.exponent_bias == bias == (1 << (nexp - 1)) - 1
    assert layout.sign_bit == total - 1
    assert layout.exponent_lsb == nmant
    assert layout.exponent_msb == total - 2


@pytest.mark.parametrize("name", FLOATS)
def test_independent_probe_rederivation_agrees(name):
    report = verify_float_layout(name)
    assert report["agree"] is True, report


@pytest.mark.parametrize("name", FLOATS)
def test_roles_partition_the_word(name):
    layout = float_layout(name)
    counts = {"sign": 0, "exponent": 0, "mantissa": 0}
    for b in range(layout.total_bits):
        counts[layout.role(b)] += 1
    assert counts == {
        "sign": 1,
        "exponent": layout.exponent_bits,
        "mantissa": layout.mantissa_bits,
    }


def test_exponent_weights_are_powers_of_two():
    layout = float_layout("float32")
    weights = [layout.exponent_weight(b) for b in range(23, 31)]
    assert weights == [1, 2, 4, 8, 16, 32, 64, 128]
    with pytest.raises(ValueError, match="not an exponent bit"):
        layout.exponent_weight(31)
    with pytest.raises(ValueError, match="not an exponent bit"):
        layout.exponent_weight(0)


def test_known_bit_pattern_of_one_and_two():
    # 1.0 = 2**0 * 1.0 -> s=0, E=bias=127, m=0 -> 0x3F800000
    assert to_bits(np.float32(1.0), "float32") == 0x3F800000
    # 2.0 = 2**1 * 1.0 -> E=128 -> 0x40000000
    assert to_bits(np.float32(2.0), "float32") == 0x40000000
    # -1.0 sets the sign bit -> 0xBF800000
    assert to_bits(np.float32(-1.0), "float32") == 0xBF800000
    assert exponent_field(np.float32(1.0), "float32") == 127
    assert mantissa_field(np.float32(1.5), "float32") == 1 << 22  # 1.5 = 1 + 2**-1


def test_sign_bit_flip_negates_exactly():
    for value in (1.0, -3.25, 1e-30, 7.5e20):
        v = np.float32(value)
        assert float(flip_bit(v, 31, "float32")) == -float(v)


def test_exponent_msb_flip_scales_by_two_to_the_128():
    # float32 bit 30 has exponent weight 2**(30-23) = 128, so a 1 -> 0 flip
    # divides a normal number by 2**128 when the result is still normal.
    # 5.0 = 1.25 * 2**2 -> E = 129; 129 - 128 = 1 > 0 so the result is normal.
    value = np.float32(5.0)
    flipped = float(flip_bit(value, 30, "float32"))
    assert flipped == pytest.approx(5.0 * 2.0**-128, rel=0.0, abs=0.0)
    assert exponent_scale_factor(30, -1, "float32") == 2.0**-128


def test_exponent_msb_flip_overflows_to_infinity_when_predicted():
    # 1.0 has E = 127; +128 -> 255, the reserved all-ones field, mantissa 0,
    # therefore +inf, not 2**128.
    prediction = predict_flip(np.float32(1.0), 30, "float32")
    assert prediction.regime == "to_infinity"
    assert math.isinf(prediction.predicted_value)
    assert math.isinf(float(flip_bit(np.float32(1.0), 30, "float32")))


def test_mantissa_flip_is_additive_and_hand_computable():
    # 1.0 with mantissa bit 0 set is 1 + 2**-23 = 1.00000011920928955078125
    flipped = float(flip_bit(np.float32(1.0), 0, "float32"))
    assert flipped == 1.0 + 2.0**-23
    prediction = predict_flip(np.float32(1.0), 0, "float32")
    assert prediction.regime == "additive"
    assert prediction.predicted_delta == 2.0**-23


@pytest.mark.parametrize("name", FLOATS)
def test_prediction_matches_actual_flip_over_a_grid(name):
    layout = float_layout(name)
    dtype = layout.dtype
    values = np.array(
        [1.0, -1.0, 0.5, 3.5, 5.0, 123.25, -0.001953125, 1024.0, 1e-5, -77.0], dtype=dtype
    )
    mismatches = []
    for value in values:
        for b in range(layout.total_bits):
            prediction = predict_flip(value, b, name)
            actual = float(flip_bit(value, b, name))
            if math.isnan(prediction.predicted_value):
                ok = math.isnan(actual)
            else:
                ok = actual == prediction.predicted_value
            if not ok:
                mismatches.append((float(value), b, actual, prediction.predicted_value))
    assert mismatches == []


def test_double_flip_of_the_same_bit_is_the_identity():
    value = np.float32(-6.125)
    once = flip_bit(value, 17, "float32")
    twice = flip_bit(once, 17, "float32")
    assert to_bits(twice, "float32") == to_bits(value, "float32")


def test_int8_two_complement_flip_delta_exhaustive():
    layout = int_layout("int8")
    for raw in range(-128, 128):
        value = np.int8(raw)
        for b in range(8):
            expected = layout.flip_delta(int(value), b)
            actual = int(flip_bit(value, b, "int8"))
            assert actual - int(value) == expected, (raw, b, actual, expected)


def test_int8_sign_bit_changes_value_by_128():
    # two's complement: value = -128*b7 + sum_{k<7} 2**k b_k
    assert int(flip_bit(np.int8(0), 7, "int8")) == -128
    assert int(flip_bit(np.int8(-128), 7, "int8")) == 0
    assert int(flip_bit(np.int8(-1), 7, "int8")) == 127


def test_bits_roundtrip():
    for name in FLOATS:
        layout = float_layout(name)
        for pattern in (0, 1, (1 << layout.total_bits) - 1, 0x1234 % (1 << layout.total_bits)):
            value = from_bits(pattern, name)
            assert to_bits(value, name) == pattern


def test_bit_site_count():
    assert bit_site_count(147, 32) == 4704
    assert bit_site_count(0, 8) == 0
    with pytest.raises(ValueError):
        bit_site_count(-1, 8)
    with pytest.raises(ValueError):
        bit_site_count(10, 0)


def test_input_validation():
    with pytest.raises(ValueError, match="unsupported float dtype"):
        float_layout("int8")
    with pytest.raises(ValueError, match="unsupported integer dtype"):
        int_layout("float32")
    with pytest.raises(ValueError, match="outside"):
        flip_bit(np.float32(1.0), 32, "float32")
    with pytest.raises(ValueError, match="outside"):
        flip_bit(np.float32(1.0), -1, "float32")
    with pytest.raises(TypeError, match="must be an integer"):
        float_layout("float32").role(1.5)
    with pytest.raises(ValueError, match="does not fit"):
        from_bits(1 << 32, "float32")
    with pytest.raises(ValueError, match="scalar"):
        to_bits(np.zeros(3, dtype=np.float32), "float32")
    with pytest.raises(ValueError, match=r"\+1 or -1"):
        exponent_scale_factor(30, 0, "float32")


@given(
    value=st.floats(min_value=1e-20, max_value=1e20, allow_nan=False, allow_infinity=False),
    bit=st.integers(min_value=0, max_value=31),
)
@settings(max_examples=250, deadline=None)
def test_property_flip_is_an_involution(value, bit):
    v = np.float32(value)
    once = flip_bit(v, bit, "float32")
    assert to_bits(flip_bit(once, bit, "float32"), "float32") == to_bits(v, "float32")


@given(
    value=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
    bit=st.integers(min_value=0, max_value=22),
)
@settings(max_examples=200, deadline=None)
def test_property_mantissa_flip_changes_magnitude_by_less_than_the_value(value, bit):
    v = np.float32(value)
    flipped = flip_bit(v, bit, "float32")
    # A mantissa flip changes the significand by 2**(bit-23) <= 2**-1, and the
    # significand is in [1, 2), so the relative change is at most 0.5.
    assert abs(float(flipped) - float(v)) <= 0.5 * abs(float(v)) + 1e-30
