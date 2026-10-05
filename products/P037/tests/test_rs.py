"""RS(255,223): the 16/17 correction boundary and the analytic expressions."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binom

from framesync.rs import (
    RS_E,
    RS_K,
    RS_N,
    RS_RATE,
    ReedSolomonLink,
    codeword_failure_probability,
    rs_frame_error_rate,
    rs_output_bit_error_rate,
    symbol_error_probability,
)


def test_code_parameters():
    assert (RS_N, RS_K, RS_E) == (255, 223, 16)
    # d_min = n - k + 1 = 33 for an MDS code; E = (d_min - 1) / 2 = 16.
    assert RS_N - RS_K + 1 == 33
    assert RS_E == (RS_N - RS_K) // 2
    assert RS_RATE == pytest.approx(223 / 255, rel=1e-15)


def test_codeword_roundtrip_is_exact():
    link = ReedSolomonLink(interleave=1)
    msg = bytes(range(RS_K))
    cw = link.encode_codeword(msg)
    assert len(cw) == RS_N
    res = link.decode_codeword(cw)
    assert res.corrected and res.message == msg


@pytest.mark.parametrize("n_err", [0, 1, 8, 15, 16])
def test_corrects_up_to_16_symbol_errors(n_err):
    """E = 16 symbols is the guaranteed correction radius."""
    link = ReedSolomonLink(interleave=1)
    msg = bytes((7 * i + 13) % 256 for i in range(RS_K))
    cw = bytearray(link.encode_codeword(msg))
    rng = np.random.default_rng(1000 + n_err)
    pos = rng.choice(RS_N, size=n_err, replace=False)
    for p in pos:
        cw[int(p)] ^= 0xA5 or 1
    res = link.decode_codeword(bytes(cw))
    assert res.corrected, f"{n_err} symbol errors must be correctable"
    assert res.message == msg


def test_fails_at_17_symbol_errors_on_seeded_patterns():
    """17 symbol errors exceed the radius: decoding must not return the message.

    The exhaustive seeded sweep over many patterns is in
    validation/validate_rs_correction.py; this is the unit-level case.
    """
    link = ReedSolomonLink(interleave=1)
    msg = bytes((11 * i + 5) % 256 for i in range(RS_K))
    cw0 = link.encode_codeword(msg)
    recovered = 0
    for seed in range(12):
        cw = bytearray(cw0)
        rng = np.random.default_rng(5000 + seed)
        pos = rng.choice(RS_N, size=17, replace=False)
        for p in pos:
            cw[int(p)] ^= int(rng.integers(1, 256))
        res = link.decode_codeword(bytes(cw))
        if res.corrected and res.message == msg:
            recovered += 1
    assert recovered == 0, "17 symbol errors must never be corrected to the original"


def test_interleaved_frame_roundtrip():
    link = ReedSolomonLink(interleave=5)
    assert link.frame_data_octets == 1115
    assert link.codeblock_octets == 1275
    rng = np.random.default_rng(42)
    data = rng.integers(0, 256, link.frame_data_octets, dtype=np.uint8).tobytes()
    block = link.encode_frame(data)
    out, failed = link.decode_frame(block)
    assert failed == 0 and out == data


def test_interleaved_frame_survives_16_errors_in_each_codeword():
    link = ReedSolomonLink(interleave=3)
    rng = np.random.default_rng(314)
    data = rng.integers(0, 256, link.frame_data_octets, dtype=np.uint8).tobytes()
    arr = np.frombuffer(link.encode_frame(data), dtype=np.uint8).reshape(RS_N, 3).copy()
    for j in range(3):
        pos = rng.choice(RS_N, size=16, replace=False)
        arr[pos, j] ^= 0x5A
    out, failed = link.decode_frame(arr.tobytes())
    assert failed == 0 and out == data


def test_symbol_error_probability_hand_values():
    # p = 0 -> 0; p = 1 -> 1; p = 0.5, m = 8 -> 1 - 0.5^8 = 255/256.
    assert float(symbol_error_probability(0.0)) == pytest.approx(0.0, abs=1e-18)
    assert float(symbol_error_probability(1.0)) == pytest.approx(1.0)
    assert float(symbol_error_probability(0.5)) == pytest.approx(255 / 256, rel=1e-14)
    # small-p limit: p_s ~ 8p
    assert float(symbol_error_probability(1e-9)) == pytest.approx(8e-9, rel=1e-6)


def test_codeword_failure_equals_binomial_upper_tail():
    for ps in (1e-4, 1e-2, 0.05, 0.1):
        direct = sum(
            float(binom.pmf(i, RS_N, ps)) for i in range(RS_E + 1, RS_N + 1)
        )
        assert float(codeword_failure_probability(ps)) == pytest.approx(direct, rel=1e-8)


def test_codeword_failure_endpoints():
    assert float(codeword_failure_probability(0.0)) == pytest.approx(0.0, abs=1e-18)
    assert float(codeword_failure_probability(1.0)) == pytest.approx(1.0)


def test_frame_error_rate_interleaving_relation():
    p = 2e-3
    pcw = float(codeword_failure_probability(symbol_error_probability(p)))
    for i in (1, 2, 5, 8):
        expected = 1.0 - (1.0 - pcw) ** i
        assert float(np.atleast_1d(rs_frame_error_rate(p, i))[0]) == pytest.approx(
            expected, rel=1e-9
        )


def test_frame_error_rate_saturates_without_warning():
    with np.errstate(all="raise"):
        assert float(np.atleast_1d(rs_frame_error_rate(0.5, 5))[0]) == pytest.approx(1.0)


def test_output_bit_error_rate_monotone_and_below_input_in_coding_regime():
    p = np.array([1e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    out = rs_output_bit_error_rate(p)
    assert np.all(np.diff(out) > 0)
    # Coding regime: well below the code threshold the decoder helps, by many
    # decades.
    assert np.all(out[:3] < p[:3])


def test_output_bit_error_rate_amplifies_above_the_code_threshold():
    """Documented non-monotonicity: Eq. (9) exceeds p above threshold.

    Eq. (9) models a decoder that emits the uncorrected word on failure, so
    near and above p ~ 1e-2 (expected symbol errors approaching E = 16) it
    returns an output bit error rate above the channel rate. Asserted so the
    behaviour is pinned rather than discovered later; see rs.py docstring.
    """
    p = np.array([1e-2, 3e-2])
    assert np.all(rs_output_bit_error_rate(p) > p)


def test_validation_errors():
    with pytest.raises(ValueError, match="interleave"):
        ReedSolomonLink(interleave=0)
    link = ReedSolomonLink(interleave=2)
    with pytest.raises(ValueError, match="223 octets"):
        link.encode_codeword(b"short")
    with pytest.raises(ValueError, match="255 octets"):
        link.decode_codeword(b"short")
    with pytest.raises(ValueError, match="frame data must be"):
        link.encode_frame(b"short")
    with pytest.raises(ValueError, match="codeblock must be"):
        link.decode_frame(b"short")
    with pytest.raises(ValueError, match=r"p_bit must lie"):
        symbol_error_probability(1.5)
    with pytest.raises(ValueError, match=r"p_sym must lie"):
        codeword_failure_probability(-0.1)
    with pytest.raises(ValueError, match="interleave depth"):
        rs_frame_error_rate(1e-3, 0)
