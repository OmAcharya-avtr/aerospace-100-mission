"""Reed-Solomon: known answers at the correction radius and immediately beyond it.

Exercises REQ-07, REQ-08, REQ-09 of docs/REQUIREMENTS.md.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from codedfade.reedsolomon import (
    ReedSolomon,
    bits_to_symbols,
    symbols_to_bits,
)


class TestConstruction:
    def test_generator_degree_is_two_t(self, rs_small: ReedSolomon) -> None:
        assert rs_small.generator.size - 1 == 2 * rs_small.t

    def test_generator_roots_are_consecutive_alpha_powers(
        self, rs_small: ReedSolomon
    ) -> None:
        f = rs_small.field
        for i in range(2 * rs_small.t):
            assert f.poly_eval(rs_small.generator, f.alpha_power(i)) == 0

    def test_rate_and_bits_per_symbol(self, rs_work: ReedSolomon) -> None:
        assert rs_work.rate == pytest.approx(21 / 31)
        assert rs_work.bits_per_symbol == 5

    @pytest.mark.parametrize(
        "n,k,m,msg",
        [
            (16, 11, 4, "n must satisfy"),
            (15, 15, 4, "k must satisfy"),
            (15, 0, 4, "k must satisfy"),
            (15, 12, 4, "n - k must be even"),
        ],
    )
    def test_rejects_invalid_parameters(self, n: int, k: int, m: int, msg: str) -> None:
        with pytest.raises(ValueError, match=msg):
            ReedSolomon(n, k, m)


class TestEncode:
    def test_is_systematic(self, rs_small: ReedSolomon, rng) -> None:
        msg = rng.integers(0, 16, rs_small.k)
        assert np.array_equal(rs_small.encode(msg)[: rs_small.k], msg)

    def test_codeword_has_zero_syndromes(self, rs_small: ReedSolomon, rng) -> None:
        msg = rng.integers(0, 16, rs_small.k)
        assert not np.any(rs_small.syndromes(rs_small.encode(msg)))

    def test_all_zero_message_gives_all_zero_codeword(self, rs_small: ReedSolomon) -> None:
        assert not np.any(rs_small.encode(np.zeros(rs_small.k, dtype=np.int64)))

    def test_rejects_wrong_shape(self, rs_small: ReedSolomon) -> None:
        with pytest.raises(ValueError, match="message must have shape"):
            rs_small.encode(np.zeros(3, dtype=np.int64))

    def test_rejects_out_of_range_symbols(self, rs_small: ReedSolomon) -> None:
        with pytest.raises(ValueError, match="field elements must lie in"):
            rs_small.encode(np.full(rs_small.k, 16, dtype=np.int64))

    def test_rejects_wrong_received_shape(self, rs_small: ReedSolomon) -> None:
        with pytest.raises(ValueError, match="received must have shape"):
            rs_small.decode(np.zeros(3, dtype=np.int64))
        with pytest.raises(ValueError, match="received must have shape"):
            rs_small.syndromes(np.zeros(3, dtype=np.int64))


class TestKnownAnswersAtTheRadius:
    """RS(15,11), t = 2. The whole point of the code is this boundary."""

    def test_clean_codeword_decodes_with_zero_corrections(
        self, rs_small: ReedSolomon, rng
    ) -> None:
        msg = rng.integers(0, 16, rs_small.k)
        result = rs_small.decode(rs_small.encode(msg))
        assert result.success
        assert result.corrected_symbols == 0
        assert np.array_equal(result.message, msg)

    def test_every_single_error_is_corrected(self, rs_small: ReedSolomon) -> None:
        msg = np.arange(rs_small.k, dtype=np.int64)
        cw = rs_small.encode(msg)
        for pos in range(rs_small.n):
            for val in range(1, 16):
                received = cw.copy()
                received[pos] ^= val
                result = rs_small.decode(received)
                assert result.success, (pos, val)
                assert np.array_equal(result.message, msg), (pos, val)

    def test_every_double_error_is_corrected_exhaustively(
        self, rs_small: ReedSolomon
    ) -> None:
        """All C(15,2) * 15^2 = 23625 two-symbol error patterns.

        t = (15-11)/2 = 2, so bounded-distance decoding must correct every one of
        them; Reed-Solomon codes are maximum distance separable, so this is a
        guarantee and not a statistical claim.
        """
        msg = np.arange(rs_small.k, dtype=np.int64)
        cw = rs_small.encode(msg)
        failures = 0
        for positions in itertools.combinations(range(rs_small.n), 2):
            for values in itertools.product(range(1, 16), repeat=2):
                received = cw.copy()
                for p, v in zip(positions, values, strict=True):
                    received[p] ^= v
                result = rs_small.decode(received)
                if not (result.success and np.array_equal(result.message, msg)):
                    failures += 1
        assert failures == 0

    def test_t_plus_one_errors_are_never_decoded_successfully_to_the_original(
        self, rs_small: ReedSolomon
    ) -> None:
        """At t+1 = 3 errors the code is beyond its radius.

        The honest statement needs four buckets, not two, because "the decoded
        message equals the transmitted message" is not the same as "decoding
        succeeded":

        * ``recovered``: the decoder reported success **and** the message is right.
          Must be 0: no bounded-distance decoder can do this at t+1 errors.
        * ``miscorrected``: reported success, message wrong. Must be > 0, because a
          bounded-distance decoder does miscorrect.
        * ``failed_message_wrong``: reported failure, message wrong.
        * ``failed_message_intact``: reported failure, but all t+1 errors happened
          to land in the ``n-k`` parity symbols, so the untouched message symbols
          are returned unchanged. This is not a recovery and must not be counted as
          one; it is why the assertion is on ``recovered`` and not on message
          equality alone. The expected share is C(n-k, t+1)/C(n, t+1) =
          C(4,3)/C(15,3) = 4/455.
        """
        msg = np.arange(rs_small.k, dtype=np.int64)
        cw = rs_small.encode(msg)
        rng = np.random.default_rng(99)
        recovered = miscorrected = failed_wrong = failed_intact = 0
        trials = 3000
        for _ in range(trials):
            received = cw.copy()
            for p in rng.choice(rs_small.n, rs_small.t + 1, replace=False):
                received[p] ^= int(rng.integers(1, 16))
            result = rs_small.decode(received)
            right = bool(np.array_equal(result.message, msg))
            if result.success and right:
                recovered += 1
            elif result.success:
                miscorrected += 1
            elif right:
                failed_intact += 1
            else:
                failed_wrong += 1
        assert recovered == 0
        assert recovered + miscorrected + failed_wrong + failed_intact == trials
        assert miscorrected > 0, "a bounded-distance decoder must miscorrect sometimes"
        # 4/455 of 3000 is about 26; allow a wide band, this is a sanity check
        assert 5 <= failed_intact <= 60, failed_intact


class TestWorkingCode:
    def test_t_errors_corrected_on_rs_31_21(self, rs_work: ReedSolomon) -> None:
        rng = np.random.default_rng(4)
        for _ in range(60):
            msg = rng.integers(0, 32, rs_work.k)
            received = rs_work.encode(msg)
            for p in rng.choice(rs_work.n, rs_work.t, replace=False):
                received[p] ^= int(rng.integers(1, 32))
            result = rs_work.decode(received)
            assert result.success
            assert np.array_equal(result.message, msg)

    def test_t_plus_one_errors_never_decode_successfully_on_rs_31_21(
        self, rs_work: ReedSolomon
    ) -> None:
        """Same four-bucket accounting as the small code, on the working code."""
        rng = np.random.default_rng(5)
        recovered = 0
        for _ in range(200):
            msg = rng.integers(0, 32, rs_work.k)
            received = rs_work.encode(msg)
            for p in rng.choice(rs_work.n, rs_work.t + 1, replace=False):
                received[p] ^= int(rng.integers(1, 32))
            result = rs_work.decode(received)
            if result.success and np.array_equal(result.message, msg):
                recovered += 1
        assert recovered == 0


class TestBitPacking:
    def test_round_trip(self, rng) -> None:
        sym = rng.integers(0, 32, 100)
        assert np.array_equal(bits_to_symbols(symbols_to_bits(sym, 5), 5), sym)

    def test_most_significant_bit_first(self) -> None:
        assert symbols_to_bits(np.array([0b10110]), 5).tolist() == [1, 0, 1, 1, 0]

    def test_rejects_ragged_bit_count(self) -> None:
        with pytest.raises(ValueError, match="not a multiple of"):
            bits_to_symbols(np.zeros(7, dtype=np.uint8), 5)


@settings(max_examples=40, deadline=None)
@given(data=st.data())
def test_encode_decode_identity_under_t_errors(data) -> None:
    n, k, m = 15, 11, 4
    code = ReedSolomon(n, k, m)
    msg = np.array(
        data.draw(st.lists(st.integers(0, 15), min_size=k, max_size=k)), dtype=np.int64
    )
    positions = data.draw(
        st.lists(st.integers(0, n - 1), min_size=code.t, max_size=code.t, unique=True)
    )
    values = data.draw(
        st.lists(st.integers(1, 15), min_size=code.t, max_size=code.t)
    )
    received = code.encode(msg)
    for p, v in zip(positions, values, strict=True):
        received[p] ^= v
    result = code.decode(received)
    assert result.success
    assert np.array_equal(result.message, msg)
