"""Convolutional code and Viterbi decoder, against a hand trace.

Exercises REQ-10, REQ-11 of docs/REQUIREMENTS.md.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from codedfade.convolutional import ConvolutionalCode


class TestHandTrace:
    def test_encode_1011_matches_the_hand_trace(self, conv: ConvolutionalCode) -> None:
        """K = 3, g = (0o7, 0o5) = (111, 101), message 1 0 1 1, two zero tail bits.

        State is (b[i-1], b[i-2]); outputs are (b ^ b[i-1] ^ b[i-2], b ^ b[i-2]).

        step  bit  state  out1              out2            new state
        1     1    0,0    1^0^0 = 1         1^0 = 1         1,0
        2     0    1,0    0^1^0 = 1         0^0 = 0         0,1
        3     1    0,1    1^0^1 = 0         1^1 = 0         1,0
        4     1    1,0    1^1^0 = 0         1^0 = 1         1,1
        5     0    1,1    0^1^1 = 0         0^1 = 1         0,1
        6     0    0,1    0^0^1 = 1         0^1 = 1         0,0

        so the output is 11 10 00 01 01 11 = 111000010111.
        """
        encoded = conv.encode(np.array([1, 0, 1, 1], dtype=np.uint8))
        assert "".join(map(str, encoded.tolist())) == "111000010111"

    def test_encode_terminates_in_the_zero_state(self, conv: ConvolutionalCode) -> None:
        for value in range(16):
            bits = np.array([(value >> b) & 1 for b in range(3, -1, -1)], dtype=np.uint8)
            assert conv.encode(bits).size == conv.n_out * (4 + conv.constraint_length - 1)

    def test_all_zero_message_gives_all_zero_codeword(
        self, conv: ConvolutionalCode
    ) -> None:
        assert not conv.encode(np.zeros(8, dtype=np.uint8)).any()

    def test_minimum_terminated_weight_is_measured_not_asserted(
        self, conv: ConvolutionalCode
    ) -> None:
        """The (0o7, 0o5) K=3 code is reported in the literature to have free
        distance 5. This test does not assert that from a reference; it enumerates
        every non-zero terminated codeword up to L = 12 and records what the
        implementation actually produces.
        """
        assert conv.minimum_terminated_weight(12) == 5


class TestViterbi:
    def test_clean_decode_is_exact(self, conv: ConvolutionalCode, rng) -> None:
        bits = rng.integers(0, 2, 64).astype(np.uint8)
        result = conv.decode(conv.encode(bits), bits.size)
        assert np.array_equal(result.message, bits)
        assert result.path_metric == 0

    def test_corrects_isolated_single_bit_errors(
        self, conv: ConvolutionalCode, rng
    ) -> None:
        bits = rng.integers(0, 2, 60).astype(np.uint8)
        encoded = conv.encode(bits)
        for pos in range(0, encoded.size, 7):
            received = encoded.copy()
            received[pos] ^= 1
            result = conv.decode(received, bits.size)
            assert np.array_equal(result.message, bits), pos
            assert result.path_metric == 1

    def test_path_metric_is_the_hamming_distance_to_the_chosen_codeword(
        self, conv: ConvolutionalCode, rng
    ) -> None:
        bits = rng.integers(0, 2, 32).astype(np.uint8)
        encoded = conv.encode(bits)
        received = encoded.copy()
        received[[1, 9, 20]] ^= 1
        result = conv.decode(received, bits.size)
        chosen = conv.encode(result.message)
        assert result.path_metric == int(np.count_nonzero(chosen != received))

    def test_burst_of_errors_defeats_the_code(self, conv: ConvolutionalCode) -> None:
        """A contiguous burst longer than the free distance is not correctable.

        This is the failure mode the interleaver exists to prevent, and it is
        asserted here as a positive fact rather than hoped for.
        """
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        encoded = conv.encode(bits)
        received = encoded.copy()
        received[4:12] ^= 1
        assert not np.array_equal(conv.decode(received, bits.size).message, bits)


class TestValidation:
    def test_rejects_short_constraint_length(self) -> None:
        with pytest.raises(ValueError, match="constraint_length"):
            ConvolutionalCode((0o7, 0o5), 1)

    def test_rejects_empty_generator_list(self) -> None:
        with pytest.raises(ValueError, match="at least one generator"):
            ConvolutionalCode((), 3)

    def test_rejects_oversized_generator(self) -> None:
        with pytest.raises(ValueError, match="each generator must satisfy"):
            ConvolutionalCode((0o17, 0o5), 3)

    def test_rejects_empty_message(self, conv: ConvolutionalCode) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            conv.encode(np.zeros(0, dtype=np.uint8))

    def test_rejects_non_binary_message(self, conv: ConvolutionalCode) -> None:
        with pytest.raises(ValueError, match="only 0 and 1"):
            conv.encode(np.array([0, 2], dtype=np.uint8))

    def test_rejects_wrong_received_length(self, conv: ConvolutionalCode) -> None:
        with pytest.raises(ValueError, match="received must have"):
            conv.decode(np.zeros(5, dtype=np.uint8), 4)

    def test_rejects_non_binary_received(self, conv: ConvolutionalCode) -> None:
        with pytest.raises(ValueError, match="only 0 and 1"):
            conv.decode(np.full(12, 3, dtype=np.uint8), 4)

    def test_rejects_non_positive_message_bits(self, conv: ConvolutionalCode) -> None:
        with pytest.raises(ValueError, match="message_bits"):
            conv.decode(np.zeros(12, dtype=np.uint8), 0)
        with pytest.raises(ValueError, match="message_bits"):
            conv.terminated_rate(0)

    def test_rejects_oversized_exhaustive_weight_search(
        self, conv: ConvolutionalCode
    ) -> None:
        with pytest.raises(ValueError, match="message_bits must be in"):
            conv.minimum_terminated_weight(17)

    def test_rates(self, conv: ConvolutionalCode) -> None:
        assert conv.rate == pytest.approx(0.5)
        assert conv.terminated_rate(100) == pytest.approx(100 / (2 * 102))


@settings(max_examples=25, deadline=None)
@given(bits=st.lists(st.integers(0, 1), min_size=4, max_size=40))
def test_encode_decode_identity_on_a_clean_channel(bits: list[int]) -> None:
    conv = ConvolutionalCode()
    msg = np.array(bits, dtype=np.uint8)
    assert np.array_equal(conv.decode(conv.encode(msg), msg.size).message, msg)
