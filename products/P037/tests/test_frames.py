"""Frame geometry, FECF checking, and marker-prefixed stream construction."""

from __future__ import annotations

import numpy as np
import pytest

from framesync.asm import ASM_32_BITS, detect_asm
from framesync.frames import CCSDS_PRIMARY_HEADER_OCTETS, FrameGeometry, build_stream


def test_ccsds_i5_geometry_arithmetic():
    """I = 5 RS codeblock carries 223*5 = 1115 octets of transfer frame.

    With the 2-octet Frame Error Control Field that is 1117 octets = 8936
    bits, and with the 32-bit marker the frame period is 8968 bits.
    """
    g = FrameGeometry(data_octets=1115, fecf=True, asm_bits=32)
    assert 223 * 5 == 1115
    assert g.frame_octets == 1117
    assert g.frame_bits == 8936
    assert g.period_bits == 8968


def test_geometry_without_fecf_or_marker():
    g = FrameGeometry(data_octets=223, fecf=False, asm_bits=0)
    assert g.frame_octets == 223 and g.frame_bits == 1784 and g.period_bits == 1784


def test_primary_header_length():
    assert CCSDS_PRIMARY_HEADER_OCTETS == 6


def test_random_frames_carry_a_valid_fecf():
    g = FrameGeometry(data_octets=64)
    rng = np.random.default_rng(11)
    frames = g.random_frames(50, rng)
    assert frames.shape == (50, 66)
    assert np.all(g.fecf_ok(frames))
    assert all(g.check_single(frames[i]) for i in range(5))


def test_any_single_bit_flip_in_a_frame_is_caught_by_the_fecf():
    g = FrameGeometry(data_octets=16)
    rng = np.random.default_rng(12)
    frame = g.random_frames(1, rng)[0]
    for octet in range(g.frame_octets):
        for bit in range(8):
            bad = frame.copy()
            bad[octet] ^= 1 << bit
            assert not g.fecf_ok(bad[None, :])[0]


def test_fecf_check_rejects_wrong_row_length_and_missing_fecf():
    g = FrameGeometry(data_octets=16)
    with pytest.raises(ValueError, match="octets per row"):
        g.fecf_ok(np.zeros((2, 5), dtype=np.uint8))
    with pytest.raises(ValueError, match="no Frame Error Control Field"):
        FrameGeometry(data_octets=16, fecf=False).fecf_ok(np.zeros((1, 16), dtype=np.uint8))


def test_geometry_validation():
    with pytest.raises(ValueError, match="data_octets must be positive"):
        FrameGeometry(data_octets=0)
    with pytest.raises(ValueError, match="asm_bits must be non-negative"):
        FrameGeometry(asm_bits=-1)
    with pytest.raises(ValueError, match="n_frames must be positive"):
        FrameGeometry(data_octets=8).random_frames(0, np.random.default_rng(1))


def test_build_stream_places_markers_at_the_frame_period():
    g = FrameGeometry(data_octets=32, fecf=True, asm_bits=32)
    rng = np.random.default_rng(13)
    stream, offsets = build_stream(g, 7, rng)
    assert stream.size == 7 * g.period_bits
    assert offsets.tolist() == [k * g.period_bits for k in range(7)]
    found = detect_asm(stream, 0)
    for o in offsets.tolist():
        assert o in found.tolist()
    assert np.array_equal(stream[: ASM_32_BITS.size], ASM_32_BITS)


def test_build_stream_rejects_a_pattern_length_mismatch():
    g = FrameGeometry(data_octets=8, asm_bits=16)
    with pytest.raises(ValueError, match="does not match"):
        build_stream(g, 3, np.random.default_rng(1))
