"""ASM pattern, correlation detector, and the false-sync expression (3)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import binom

from framesync.asm import (
    ASM_32_BITS,
    ASM_32_HEX,
    asm_autocorrelation,
    bits_from_int,
    detect_asm,
    expected_false_syncs,
    false_sync_probability,
    hamming_distances,
    int_from_bits,
)


def test_marker_value_and_bit_expansion():
    assert ASM_32_HEX == 0x1ACFFC1D
    assert ASM_32_BITS.size == 32
    assert int_from_bits(ASM_32_BITS) == 0x1ACFFC1D
    # Hand check: 0x1A = 0001 1010 is the first octet, MSB first.
    assert list(ASM_32_BITS[:8]) == [0, 0, 0, 1, 1, 0, 1, 0]
    # Population count of 0x1ACFFC1D, by hand from the four octets:
    # 0x1A -> 3, 0xCF -> 6, 0xFC -> 6, 0x1D -> 4, total 19.
    assert int(ASM_32_BITS.sum()) == 19


def test_bits_from_int_validation():
    with pytest.raises(ValueError, match="does not fit"):
        bits_from_int(0x100, 8)
    with pytest.raises(ValueError, match="length must be positive"):
        bits_from_int(1, 0)


def test_hamming_distance_of_marker_with_itself_is_zero():
    d = hamming_distances(ASM_32_BITS, ASM_32_BITS)
    assert d.size == 1 and d[0] == 0


def test_detector_finds_an_embedded_marker_at_the_right_offset():
    rng = np.random.default_rng(5)
    stream = rng.integers(0, 2, 500, dtype=np.uint8)
    stream[137:169] = ASM_32_BITS
    assert 137 in detect_asm(stream, 0).tolist()


def test_detector_tolerance_admits_exactly_that_many_errors():
    stream = ASM_32_BITS.copy()
    stream[3] ^= 1
    stream[17] ^= 1
    assert detect_asm(stream, 0).size == 0
    assert detect_asm(stream, 1).size == 0
    assert detect_asm(stream, 2).tolist() == [0]


def test_false_sync_probability_hand_values():
    # T = 0: a single pattern out of 2^32 -> 2^-32.
    assert false_sync_probability(0) == pytest.approx(2.0**-32, rel=1e-15)
    # T = 1: the pattern plus its 32 single-bit neighbours -> 33 / 2^32.
    assert false_sync_probability(1) == pytest.approx(33.0 / 2**32, rel=1e-15)
    # T = 2: 1 + 32 + C(32,2) = 1 + 32 + 496 = 529 patterns.
    assert false_sync_probability(2) == pytest.approx(529.0 / 2**32, rel=1e-15)
    # T = L: every pattern matches.
    assert false_sync_probability(32) == pytest.approx(1.0, rel=1e-15)


def test_false_sync_probability_equals_binomial_cdf():
    """Equation (3) is the Binomial(L, 1/2) CDF, by an independent path."""
    for t in range(0, 33):
        assert false_sync_probability(t, 32) == pytest.approx(
            float(binom.cdf(t, 32, 0.5)), rel=1e-12
        )


def test_false_sync_probability_other_lengths():
    for L in (8, 16, 24, 64):
        for t in (0, 1, L // 2, L):
            assert false_sync_probability(t, L) == pytest.approx(
                float(binom.cdf(t, L, 0.5)), rel=1e-11
            )


def test_expected_false_syncs_is_linear_in_window_count():
    n = 10**6
    assert expected_false_syncs(n, 3) == pytest.approx(
        (n - 31) * false_sync_probability(3), rel=1e-15
    )
    with pytest.raises(ValueError, match="at least the marker length"):
        expected_false_syncs(10, 0, 32)


@pytest.mark.parametrize("bad", [-1, 33])
def test_tolerance_validation(bad):
    with pytest.raises(ValueError, match="tolerance"):
        false_sync_probability(bad, 32)
    with pytest.raises(ValueError, match="tolerance"):
        detect_asm(np.zeros(64, dtype=np.uint8), bad)


def test_stream_validation():
    with pytest.raises(ValueError, match="shorter than"):
        hamming_distances(np.zeros(8, dtype=np.uint8))
    with pytest.raises(ValueError, match="0/1"):
        hamming_distances(np.full(64, 3, dtype=np.uint8))


def test_marker_autocorrelation_minimum_shift_distance():
    """The marker stays far from every cyclic shift of itself.

    Measured, not quoted: the minimum Hamming distance over the 31 non-zero
    cyclic shifts of 0x1ACFFC1D is 12 bits, so a detector at tolerance
    T <= 5 cannot lock onto a shift of the marker (it would need
    T >= ceil(12/2) overlap errors on both sides to be ambiguous).
    """
    ac = asm_autocorrelation()
    assert ac[0] == 0
    assert int(ac[1:].min()) == 12


def test_monte_carlo_false_sync_rate_matches_eq3_at_large_tolerance():
    """Small, fast confirmation of Eq. (3); the full sweep is in validation/.

    T = 12 gives P_fa = 0.0811, which 200k windows resolve to ~2% relative.
    """
    rng = np.random.default_rng(31337)
    n = 200_000
    stream = rng.integers(0, 2, n + 31, dtype=np.uint8)
    d = hamming_distances(stream)
    measured = float(np.count_nonzero(d <= 12)) / d.size
    expected = false_sync_probability(12)
    se = math.sqrt(expected * (1 - expected) / d.size)
    # Overlapping windows are correlated, so the effective variance is larger
    # than binomial; 6 sigma keeps this from flaking while still being a test.
    assert abs(measured - expected) < 6 * se
