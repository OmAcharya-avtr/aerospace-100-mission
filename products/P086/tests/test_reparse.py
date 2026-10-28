"""The reinterpretation test: known answers, the (7a)/(7b) conditions, vector agreement."""

from __future__ import annotations

import numpy as np
import pytest

from telemgap.record import ChannelSpec
from telemgap.reparse import (
    ALTERNATIVES,
    DEFAULT_CONSISTENCY_TAU,
    alternative_value,
    alternative_value_array,
    as_signed,
    best_rescue,
    byte_reverse,
    byte_reverse_array,
    raw_as_read,
    reinterpret,
    reinterpret_array,
    value_as_read,
)


@pytest.mark.parametrize(
    ("word", "bits", "expected"),
    [
        # Hand-calculated: 0x1234 -> 0x3412 = 13330; 0x00FF -> 0xFF00 = 65280.
        (0x1234, 16, 0x3412),
        (0x00FF, 16, 0xFF00),
        (0xFF00, 16, 0x00FF),
        (0x0000, 16, 0x0000),
        (0xFFFF, 16, 0xFFFF),
        (0x12345678, 32, 0x78563412),
    ],
)
def test_byte_reverse_known_answers(word: int, bits: int, expected: int) -> None:
    assert byte_reverse(word, bits) == expected


def test_byte_reverse_is_an_involution() -> None:
    for word in (0, 1, 255, 256, 4660, 65535):
        assert byte_reverse(byte_reverse(word, 16), 16) == word


@pytest.mark.parametrize("bits", [0, 7, 12, -8])
def test_byte_reverse_rejects_bad_width(bits: int) -> None:
    with pytest.raises(ValueError, match="multiple of 8"):
        byte_reverse(1, bits)


def test_byte_reverse_rejects_oversized_word() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        byte_reverse(1 << 16, 16)


@pytest.mark.parametrize(
    ("word", "bits", "expected"),
    [
        # Two's complement: 0x8000 is -32768, 0xFFFF is -1, 0x7FFF is 32767.
        (0x0000, 16, 0),
        (0x7FFF, 16, 32767),
        (0x8000, 16, -32768),
        (0xFFFF, 16, -1),
        (0xFFFFFFFF, 32, -1),
    ],
)
def test_as_signed_known_answers(word: int, bits: int, expected: int) -> None:
    assert as_signed(word, bits) == expected


def test_raw_as_read_respects_declared_signedness(small_channel: ChannelSpec) -> None:
    import dataclasses

    signed = dataclasses.replace(small_channel, signed=True)
    assert raw_as_read(0xFFFF, small_channel) == 65535
    assert raw_as_read(0xFFFF, signed) == -1
    assert value_as_read(0xFFFF, signed) == pytest.approx(9.5, rel=1e-15, abs=0.0)


def test_alternative_value_known_answers(small_channel: ChannelSpec) -> None:
    # scale 0.5, offset 10.0, 16-bit unsigned.
    # word 0x0100 = 256: as read 0.5*256+10 = 138.0
    # byteswap -> 0x0001 = 1 -> 0.5*1+10 = 10.5
    # signedness -> 256 is below 0x8000, so unchanged -> 138.0
    # scale_decade_up -> 0.5*10*256+10 = 1290.0
    # scale_decade_down -> 0.5/10*256+10 = 22.8
    # offset_omitted -> 0.5*256 = 128.0
    expected = {
        "byteswap": 10.5,
        "signedness": 138.0,
        "scale_decade_up": 1290.0,
        "scale_decade_down": 22.8,
        "offset_omitted": 128.0,
    }
    for name, value in expected.items():
        _, got = alternative_value(0x0100, small_channel, name)
        assert got == pytest.approx(value, rel=1e-12, abs=0.0), name


def test_alternative_value_rejects_unknown(small_channel: ChannelSpec) -> None:
    with pytest.raises(ValueError, match="unknown alternative"):
        alternative_value(1, small_channel, "nonsense")
    with pytest.raises(ValueError, match="unknown alternative"):
        alternative_value_array(np.zeros(1, dtype=np.uint64), small_channel, "nonsense")


def test_alternatives_tuple_is_the_documented_five() -> None:
    assert ALTERNATIVES == (
        "byteswap",
        "signedness",
        "scale_decade_up",
        "scale_decade_down",
        "offset_omitted",
    )


def test_vector_and_scalar_alternatives_agree_bitwise(small_channel: ChannelSpec) -> None:
    import dataclasses

    words = np.array([0, 1, 255, 256, 32767, 32768, 65535, 4660], dtype=np.uint64)
    for channel in (small_channel, dataclasses.replace(small_channel, signed=True)):
        for name in ALTERNATIVES:
            vector = alternative_value_array(words, channel, name)
            scalar = np.array(
                [alternative_value(int(word), channel, name)[1] for word in words]
            )
            assert np.array_equal(vector, scalar), (channel.signed, name)


def test_byte_reverse_array_matches_scalar() -> None:
    words = np.arange(0, 65536, 997, dtype=np.uint64)
    vector = byte_reverse_array(words, 16)
    scalar = np.array([byte_reverse(int(word), 16) for word in words], dtype=np.int64)
    assert np.array_equal(vector, scalar)


def test_byte_reverse_array_rejects_bad_width() -> None:
    with pytest.raises(ValueError, match="multiple of 8"):
        byte_reverse_array(np.zeros(1, dtype=np.uint64), 7)


def test_reinterpret_requires_both_conditions(small_channel: ChannelSpec) -> None:
    # word 0x2800 = 10240: as read 0.5*10240+10 = 5130.0, far out of [12, 60].
    # byteswap -> 0x0028 = 40 -> 30.0, inside the range.
    # With a local median of 30.0 the byteswap both fits (7a) and (7b).
    candidates = reinterpret(0x2800, small_channel, local_median=30.0, scaled_mad=0.5)
    rescue = best_rescue(candidates)
    assert rescue is not None
    assert rescue.alternative == "byteswap"
    assert rescue.value_alternative == pytest.approx(30.0, rel=1e-12, abs=0.0)
    # With a local median of 50.0 the byteswap still fits (7a) but not (7b),
    # so nothing rescues: condition (7a) alone is not evidence of a misparse.
    far = reinterpret(0x2800, small_channel, local_median=50.0, scaled_mad=0.5)
    assert best_rescue(far) is None
    assert any(candidate.in_range and not candidate.consistent for candidate in far)


def test_reinterpret_reports_every_alternative(small_channel: ChannelSpec) -> None:
    candidates = reinterpret(0x2800, small_channel, local_median=30.0, scaled_mad=0.5)
    assert [candidate.alternative for candidate in candidates] == list(ALTERNATIVES)
    assert all(np.isfinite(candidate.value_alternative) for candidate in candidates)


def test_reinterpret_rejects_bad_arguments(small_channel: ChannelSpec) -> None:
    with pytest.raises(ValueError, match="tau must be positive"):
        reinterpret(1, small_channel, 30.0, 0.5, tau=0.0)
    with pytest.raises(ValueError, match="scaled_mad"):
        reinterpret(1, small_channel, 30.0, 0.0)
    with pytest.raises(ValueError, match="scaled_mad"):
        reinterpret(1, small_channel, 30.0, float("nan"))


def test_default_tau_is_four_sigma() -> None:
    assert DEFAULT_CONSISTENCY_TAU == pytest.approx(4.0, rel=0.0, abs=0.0)


def test_reinterpret_array_matches_scalar(small_channel: ChannelSpec) -> None:
    words = np.array([0x2800, 0x0028, 0xFFFF, 0x1234], dtype=np.uint64)
    medians = np.full(words.size, 30.0)
    scaled = np.full(words.size, 0.5)
    bulk = reinterpret_array(words, small_channel, medians, scaled)
    for position, word in enumerate(words):
        candidates = reinterpret(int(word), small_channel, 30.0, 0.5)
        rescue = best_rescue(candidates)
        assert bool(bulk["any_rescue"][position]) == (rescue is not None)
        if rescue is not None:
            assert ALTERNATIVES[int(bulk["best_index"][position])] == rescue.alternative
            assert bulk["best_deviation"][position] == pytest.approx(
                rescue.deviation_sigmas, rel=1e-12, abs=0.0
            )


def test_reinterpret_array_validates_shapes(small_channel: ChannelSpec) -> None:
    words = np.zeros(4, dtype=np.uint64)
    with pytest.raises(ValueError, match="same shape"):
        reinterpret_array(words, small_channel, np.zeros(3), np.ones(4))
    with pytest.raises(ValueError, match="positive and finite"):
        reinterpret_array(words, small_channel, np.zeros(4), np.zeros(4))
    with pytest.raises(ValueError, match="tau must be positive"):
        reinterpret_array(words, small_channel, np.zeros(4), np.ones(4), tau=-1.0)


def test_rescue_probability_is_not_negligible(small_channel: ChannelSpec) -> None:
    # The reason condition (7b) exists: with (7a) alone, a uniformly random
    # word is rescued often enough on a wide channel to be useless as
    # evidence. Measured here on the whole 16-bit range rather than asserted.
    words = np.arange(1 << 16, dtype=np.uint64)
    medians = np.full(words.size, 30.0)
    scaled = np.full(words.size, 0.5)
    bulk = reinterpret_array(words, small_channel, medians, scaled)
    in_range_only = float(bulk["in_range"].any(axis=1).mean())
    both = float(bulk["any_rescue"].mean())
    assert in_range_only > both
    assert in_range_only > 0.001
