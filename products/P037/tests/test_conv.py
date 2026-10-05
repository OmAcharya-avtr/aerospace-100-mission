"""CCSDS rate-1/2 K=7 convolutional code: generators, trellis, Viterbi."""

from __future__ import annotations

import numpy as np
import pytest

from framesync.channel import awgn_bpsk_samples
from framesync.conv import CCSDS_G1, CCSDS_G2, ConvCode, free_distance


def test_generators_are_the_ccsds_octal_values():
    assert CCSDS_G1 == 0o171 == 121
    assert CCSDS_G2 == 0o133 == 91
    # 0o171 = 1111001b, 0o133 = 1011011b, both 7 bits for K = 7.
    assert f"{CCSDS_G1:07b}" == "1111001"
    assert f"{CCSDS_G2:07b}" == "1011011"


def test_structure():
    c = ConvCode()
    assert c.k == 7 and c.n_states == 64 and c.rate == 0.5 and c.tail_bits() == 6


def test_free_distance_is_computed_and_equals_the_literature_value():
    """d_free of the (171, 133) rate-1/2 K=7 code is 10.

    Computed here from the trellis by shortest-weight search, not quoted.
    The literature value for this code is 10; the computation agrees.
    """
    assert free_distance(ConvCode()) == 10
    # The G2 inversion is a fixed XOR and must not change the distance.
    assert free_distance(ConvCode(invert_g2=False)) == 10


def test_free_distance_of_a_small_hand_checkable_code():
    """K = 3, G1 = 0o7 (111), G2 = 0o5 (101): the textbook d_free = 5 code.

    By hand: input 1 followed by zeros gives outputs 11, 10, 11 -> weight 5,
    and no shorter returning path has lower weight.
    """
    assert free_distance(ConvCode(k=3, g1=0o7, g2=0o5, invert_g2=False)) == 5


def test_encode_length_and_termination():
    c = ConvCode()
    bits = np.ones(10, dtype=np.uint8)
    assert c.encode(bits).size == 2 * (10 + 6)
    assert c.encode(bits, terminate=False).size == 2 * 10


def test_encode_hand_check_all_zero_input_with_g2_inverted():
    """All-zero input: G1 output is 0, inverted G2 output is 1, every step.

    So the channel bits are 0,1 repeated, which is exactly the transition
    density the CCSDS inversion convention exists to guarantee.
    """
    c = ConvCode(invert_g2=True)
    out = c.encode(np.zeros(5, dtype=np.uint8))
    assert out.size == 2 * (5 + 6)
    assert np.array_equal(out, np.tile(np.array([0, 1], dtype=np.uint8), out.size // 2))
    # Without the inversion the all-zero input gives the all-zero codeword.
    assert not np.any(ConvCode(invert_g2=False).encode(np.zeros(5, dtype=np.uint8)))


def test_encode_hand_check_single_one_without_inversion():
    """K=3, G1=111, G2=101, input 1 then two tail zeros.

    Register states (current bit leftmost): 100 -> 010 -> 001.
      step 0: reg=100 -> G1 parity(100 & 111)=1, G2 parity(100 & 101)=1 -> 11
      step 1: reg=010 -> G1 parity(010 & 111)=1, G2 parity(010 & 101)=0 -> 10
      step 2: reg=001 -> G1 parity(001 & 111)=1, G2 parity(001 & 101)=1 -> 11
    Codeword 11 10 11, weight 5 = d_free.
    """
    c = ConvCode(k=3, g1=0o7, g2=0o5, invert_g2=False)
    out = c.encode(np.array([1], dtype=np.uint8))
    assert out.tolist() == [1, 1, 1, 0, 1, 1]


def test_noiseless_roundtrip():
    c = ConvCode()
    rng = np.random.default_rng(17)
    info = rng.integers(0, 2, (8, 200), dtype=np.uint8)
    enc = c.encode_batch(info)
    assert np.array_equal(c.decode_batch(enc), info)


def test_viterbi_corrects_up_to_errors_within_the_free_distance():
    """Fewer than d_free/2 = 5 channel errors in a window must be corrected."""
    c = ConvCode()
    rng = np.random.default_rng(23)
    info = rng.integers(0, 2, 300, dtype=np.uint8)
    enc = c.encode(info)
    for n_err in (1, 2, 3, 4):
        rx = enc.copy()
        pos = rng.choice(enc.size, size=n_err, replace=False)
        rx[pos] ^= 1
        assert np.array_equal(c.decode(rx), info), f"{n_err} spread errors must be corrected"


def test_soft_decision_beats_hard_decision_at_the_same_ebn0():
    """Measured, not asserted from theory: soft decision wins at 3 dB.

    Small sample sizes, so the test only requires soft <= hard rather than a
    particular margin; the quantified comparison is in
    validation/validate_conv_fer.py.
    """
    c = ConvCode()
    rng = np.random.default_rng(777)
    info = rng.integers(0, 2, (60, 200), dtype=np.uint8)
    enc = c.encode_batch(info)
    y = awgn_bpsk_samples(enc, 3.0, c.rate, rng)
    hard = (y < 0).astype(np.uint8)
    e_hard = np.count_nonzero(c.decode_batch(hard, soft=False) != info)
    e_soft = np.count_nonzero(c.decode_batch(y, soft=True) != info)
    assert e_soft <= e_hard


def test_predecessor_structure_is_consistent_with_the_trellis():
    c = ConvCode()
    nxt, out = c._trellis()
    prev_state, prev_input, prev_out = c._predecessors()
    for ns in range(c.n_states):
        for j in range(2):
            s, u = int(prev_state[ns, j]), int(prev_input[ns, j])
            assert int(nxt[s, u]) == ns
            assert int(out[s, u]) == int(prev_out[ns, j])


def test_validation_errors():
    with pytest.raises(ValueError, match="constraint length"):
        ConvCode(k=1)
    with pytest.raises(ValueError, match="does not fit"):
        ConvCode(k=3, g1=0o171)
    c = ConvCode()
    with pytest.raises(ValueError, match="even number of columns"):
        c.decode_batch(np.zeros((2, 7), dtype=np.uint8))
    with pytest.raises(ValueError, match="n_info_bits"):
        c.decode_batch(np.zeros((2, 20), dtype=np.uint8), n_info_bits=99)
    with pytest.raises(ValueError, match="0/1"):
        c.decode_batch(np.full((1, 20), 5, dtype=np.uint8))
    with pytest.raises(ValueError, match="0/1 bits"):
        c.encode(np.array([0, 3], dtype=np.uint8))
    with pytest.raises(ValueError, match="no returning path"):
        free_distance(ConvCode(), max_weight=2)
