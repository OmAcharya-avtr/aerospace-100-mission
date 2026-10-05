"""CRC-16 Frame Error Control Field: catalogue check value and batch path."""

from __future__ import annotations

import numpy as np
import pytest

from framesync.crc import CRC16_INIT, CRC16_POLY, crc16, crc16_batch


def test_catalogue_check_value():
    """CRC-16/IBM-3740 ("CRC-16/CCITT-FALSE"): check("123456789") = 0x29B1.

    poly 0x1021, init 0xFFFF, no reflection, no final XOR. This is the
    published check value of that catalogue entry and is the known-answer
    test for the implementation.
    """
    assert crc16(b"123456789") == 0x29B1


def test_empty_input_is_the_initial_register():
    assert crc16(b"") == CRC16_INIT
    assert crc16(b"", init=0x0000) == 0x0000


def test_poly_and_init_are_the_ccsds_values():
    # g(x) = x^16 + x^12 + x^5 + 1 -> 0x1021 in normal notation; all-ones preload.
    assert CRC16_POLY == 0x1021
    assert CRC16_INIT == 0xFFFF


def test_single_bit_error_is_always_detected():
    """A CRC with a non-trivial generator detects every single-bit error."""
    data = bytearray(b"framesync frame error control field")
    base = crc16(bytes(data))
    for byte_i in range(len(data)):
        for bit in range(8):
            flipped = bytearray(data)
            flipped[byte_i] ^= 1 << bit
            assert crc16(bytes(flipped)) != base


def test_batch_matches_scalar_row_by_row():
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, (40, 97), dtype=np.uint8)
    batch = crc16_batch(arr)
    for i in range(arr.shape[0]):
        assert int(batch[i]) == crc16(arr[i])


def test_batch_rejects_bad_shapes_and_init():
    with pytest.raises(ValueError, match="16-bit"):
        crc16_batch(np.zeros((2, 3), dtype=np.uint8), init=0x1FFFF)
    with pytest.raises(ValueError, match="16-bit"):
        crc16(b"x", init=-1)


def test_hand_calculation_single_zero_octet():
    """Hand calculation, init 0x0000, one 0x80 octet, g(x) = 0x1021.

    The register loads byte << 8 = 0x8000 and runs eight shift steps; a step
    XORs in 0x1021 after the shift when the pre-shift top bit was set.

        start 0x8000  top set -> (0x8000<<1 & 0xFFFF) ^ 0x1021 = 0x1021
        1     0x1021  clear   -> 0x2042
        2     0x2042  clear   -> 0x4084
        3     0x4084  clear   -> 0x8108
        4     0x8108  top set -> (0x0210) ^ 0x1021             = 0x1231
        5     0x1231  clear   -> 0x2462
        6     0x2462  clear   -> 0x48C4
        7     0x48C4  clear   -> 0x9188

    so the CRC is 0x9188.
    """
    assert crc16(bytes([0x80]), init=0x0000) == 0x9188
