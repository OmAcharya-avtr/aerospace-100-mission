"""Property-based tests for the algebraic identities in this package."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from framesync.asm import bits_from_int, false_sync_probability, int_from_bits
from framesync.channel import bpsk_ber, ebn0_to_esn0_db
from framesync.conv import ConvCode
from framesync.crc import crc16, crc16_batch
from framesync.fer import uncoded_fer
from framesync.rs import codeword_failure_probability, symbol_error_probability

SETTINGS = settings(max_examples=40, deadline=None)


@given(st.integers(min_value=0, max_value=2**32 - 1))
@SETTINGS
def test_bit_expansion_round_trips(value):
    assert int_from_bits(bits_from_int(value, 32)) == value


@given(st.integers(min_value=0, max_value=32))
@SETTINGS
def test_false_sync_probability_is_a_cdf(t):
    p = false_sync_probability(t, 32)
    assert 0.0 < p <= 1.0
    if t > 0:
        assert p > false_sync_probability(t - 1, 32)


@given(st.floats(min_value=-5.0, max_value=16.0), st.floats(min_value=0.05, max_value=1.0))
@SETTINGS
def test_rate_loss_identity(ebn0_db, rate):
    """bpsk_ber(x, R) == bpsk_ber(x + 10 log10 R, 1): a pure dB offset.

    Equal to floating-point round-off only: the two paths compute
    10**((x + 10 log10 R)/10) and R * 10**(x/10), which differ in the last
    ulp or two, so the comparison is relative rather than exact.
    """
    shifted = float(ebn0_to_esn0_db(ebn0_db, rate)[0])
    left = float(bpsk_ber(ebn0_db, rate)[0])
    right = float(bpsk_ber(shifted, 1.0)[0])
    assert abs(left - right) <= 1e-12 * max(left, right, 1e-300)


@given(st.floats(min_value=-5.0, max_value=20.0), st.integers(min_value=1, max_value=20000))
@SETTINGS
def test_uncoded_fer_never_below_the_bit_error_rate(ebn0_db, n):
    """A frame of n >= 1 bits cannot be more reliable than one bit."""
    fer = float(uncoded_fer(ebn0_db, n)[0])
    ber = float(bpsk_ber(ebn0_db)[0])
    assert ber - 1e-18 <= fer <= 1.0


@given(st.floats(min_value=-5.0, max_value=20.0), st.integers(min_value=1, max_value=5000))
@SETTINGS
def test_uncoded_fer_is_non_decreasing_in_frame_length(ebn0_db, n):
    assert float(uncoded_fer(ebn0_db, n)[0]) <= float(uncoded_fer(ebn0_db, n + 1)[0]) + 1e-18


@given(st.floats(min_value=0.0, max_value=1.0), st.integers(min_value=1, max_value=16))
@SETTINGS
def test_symbol_error_probability_bounds(p, m):
    ps = float(symbol_error_probability(p, m))
    assert p - 1e-15 <= ps <= min(1.0, m * p) + 1e-12


@given(st.floats(min_value=0.0, max_value=1.0))
@SETTINGS
def test_codeword_failure_is_monotone_in_the_symbol_rate(ps):
    a = float(codeword_failure_probability(ps))
    b = float(codeword_failure_probability(min(1.0, ps + 0.01)))
    assert a <= b + 1e-15


@given(st.binary(min_size=0, max_size=80))
@SETTINGS
def test_crc_batch_matches_scalar(data):
    arr = np.frombuffer(data, dtype=np.uint8).reshape(1, -1)
    if arr.shape[1] == 0:
        return
    assert int(crc16_batch(arr)[0]) == crc16(data)


@given(st.lists(st.integers(0, 1), min_size=1, max_size=60))
@SETTINGS
def test_convolutional_code_is_linear(bits):
    """A linear code satisfies enc(a XOR b) = enc(a) XOR enc(b).

    Checked on the non-inverted generators; the CCSDS G2 inversion is an
    affine offset and breaks linearity by exactly that constant, which this
    test also pins.
    """
    c = ConvCode(invert_g2=False)
    a = np.array(bits, dtype=np.uint8)
    b = np.roll(a, 1)
    assert np.array_equal(c.encode(a ^ b), c.encode(a) ^ c.encode(b))


@given(st.lists(st.integers(0, 1), min_size=1, max_size=40))
@SETTINGS
def test_viterbi_recovers_a_noiseless_codeword(bits):
    c = ConvCode()
    a = np.array(bits, dtype=np.uint8)
    assert np.array_equal(c.decode(c.encode(a)), a)
