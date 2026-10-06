"""Extended Hamming (8,4) tests and the soft-versus-hard ordering."""

from __future__ import annotations

import numpy as np
import pytest

from softdecode.codes import EXTENDED_HAMMING_84, ExtendedHamming84
from softdecode.detection import DetectionModel
from softdecode.llr import llr_ook_known_csi
from softdecode.metrics import bit_error_rate


def test_parameters():
    code = EXTENDED_HAMMING_84
    assert (code.length, code.dimension, code.rate, code.min_distance) == (8, 4, 0.5, 4)


def test_weight_distribution_known_answer():
    # The (8,4) extended Hamming code has weight enumerator 1 + 14 x**4 + x**8,
    # which fixes d_min = 4.
    assert EXTENDED_HAMMING_84.weight_distribution().tolist() == [1, 0, 0, 0, 14, 0, 0, 0, 1]


def test_generator_is_systematic():
    g = EXTENDED_HAMMING_84.generator
    assert np.array_equal(g[:, :4], np.eye(4, dtype=np.int8))


def test_all_codewords_have_zero_syndrome():
    code = EXTENDED_HAMMING_84
    assert not np.any(code.syndrome(code.codewords))


def test_codewords_are_distinct():
    assert len({tuple(c) for c in EXTENDED_HAMMING_84.codewords}) == 16


def test_encode_then_project_round_trip():
    code = EXTENDED_HAMMING_84
    messages = code.messages
    words = code.encode(messages)
    assert np.array_equal(words[:, :4], messages)


def test_noiseless_soft_and_hard_both_exact():
    code = EXTENDED_HAMMING_84
    llr = (1 - 2 * code.codewords.astype(float)) * 10.0
    assert np.array_equal(code.decode_soft(llr), code.messages)
    assert np.array_equal(code.decode_hard(llr), code.messages)


def test_single_error_corrected_by_both():
    code = EXTENDED_HAMMING_84
    for flip in range(8):
        llr = np.full((16, 8), 10.0) * (1 - 2 * code.codewords.astype(float))
        llr[:, flip] *= -1.0
        assert np.array_equal(code.decode_soft(llr), code.messages)
        assert np.array_equal(code.decode_hard(llr), code.messages)


def test_soft_uses_magnitude_where_hard_cannot():
    # Two sign errors, one of them weak: soft decoding recovers the all-zero
    # codeword, hard decoding is at minimum distance 2 from it and from a
    # weight-4 codeword, so it cannot.
    code = EXTENDED_HAMMING_84
    llr = np.array([[5.0, 5.0, 5.0, 5.0, -0.2, -0.2, 5.0, 5.0]])
    assert np.array_equal(code.decode_soft(llr), np.zeros((1, 4), dtype=np.int8))


def test_soft_decoding_is_scale_invariant():
    # Documented property: maximum-likelihood block decoding depends only on
    # the sign pattern and the relative magnitudes, so a scalar LLR error is
    # invisible to it. This is why the mismatch study uses sum-product.
    rng = np.random.default_rng(5)
    llr = rng.standard_normal((500, 8)) * 3.0
    base = EXTENDED_HAMMING_84.decode_soft(llr)
    for alpha in (0.01, 0.5, 2.0, 100.0):
        assert np.array_equal(EXTENDED_HAMMING_84.decode_soft(llr * alpha), base)


def test_soft_never_worse_than_hard_over_the_fading_channel():
    # Same realisations for both decoders: the LLR array is computed once.
    code = EXTENDED_HAMMING_84
    det = DetectionModel()
    from softdecode.channel import LognormalFading

    fading = LognormalFading(0.3)
    rng = np.random.default_rng(20261006)
    for ebn0 in (2.0, 6.0, 10.0):
        a = det.ook_amplitude(ebn0, code.rate)
        messages = rng.integers(0, 2, size=(4000, 4)).astype(np.int8)
        words = code.encode(messages)
        h = fading.sample(words.shape, rng)
        y = a * h * words + rng.standard_normal(words.shape)
        llr = llr_ook_known_csi(y, a, h)
        soft = bit_error_rate(code.decode_soft(llr), messages)
        hard = bit_error_rate(code.decode_hard(llr), messages)
        assert soft.rate <= hard.rate, f"ordering violated at {ebn0} dB"


def test_input_validation():
    code = ExtendedHamming84()
    with pytest.raises(ValueError, match="last dimension 4"):
        code.encode(np.zeros((2, 5), dtype=np.int8))
    with pytest.raises(ValueError, match="0/1"):
        code.encode(np.full((2, 4), 2, dtype=np.int8))
    with pytest.raises(ValueError, match="last dimension 8"):
        code.decode_soft(np.zeros((2, 7)))
    with pytest.raises(ValueError, match="last dimension 8"):
        code.decode_hard(np.zeros((2, 7)))
    with pytest.raises(ValueError, match="last dimension 8"):
        code.syndrome(np.zeros((2, 7), dtype=np.int8))
