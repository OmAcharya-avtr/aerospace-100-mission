"""CRC: catalogue check values, two independent references, and error handling."""

from __future__ import annotations

import zlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from arqlonghaul.crc import (
    CATALOGUE,
    CRC8,
    CRC16_ARC,
    CRC16_CCITT_FALSE,
    CRC32,
    CRC32C,
    CrcSpec,
    append_fcs,
    check_fcs,
    crc,
    crc_bitwise,
)


@pytest.mark.parametrize("spec", list(CATALOGUE.values()), ids=list(CATALOGUE))
def test_catalogue_check_value(spec: CrcSpec) -> None:
    # Reference: crcmod 1.7 python3/crcmod/predefined.py, read 2026-10-06.
    assert crc(b"123456789", spec) == spec.check


def test_specific_check_values_written_out() -> None:
    # Hand-copied from the crcmod catalogue so a typo in CATALOGUE is caught.
    assert crc(b"123456789", CRC32) == 0xCBF43926
    assert crc(b"123456789", CRC32C) == 0xE3069283
    assert crc(b"123456789", CRC16_CCITT_FALSE) == 0x29B1
    assert crc(b"123456789", CRC16_ARC) == 0xBB3D
    assert crc(b"123456789", CRC8) == 0xF4


def test_crc32_matches_zlib_on_fixed_strings() -> None:
    for data in (b"", b"a", b"abc", b"The quick brown fox", bytes(range(256))):
        assert crc(data, CRC32) == zlib.crc32(data)


def test_empty_message_gives_init_xor_xorout() -> None:
    # With no data the register is untouched, so the result is init ^ xorout
    # (reflected if refout differs from refin). For CRC-32 that is 0.
    assert crc(b"", CRC32) == 0
    assert crc(b"", CRC16_CCITT_FALSE) == 0xFFFF


@pytest.mark.parametrize("spec", list(CATALOGUE.values()), ids=list(CATALOGUE))
def test_table_matches_bitwise(spec: CrcSpec) -> None:
    for data in (b"", b"x", b"12345678901234567890", bytes(range(64))):
        assert crc(data, spec) == crc_bitwise(data, spec)


@pytest.mark.parametrize("spec", list(CATALOGUE.values()), ids=list(CATALOGUE))
def test_fcs_roundtrip(spec: CrcSpec) -> None:
    frame = append_fcs(b"payload bytes here", spec)
    assert len(frame) == len(b"payload bytes here") + spec.width // 8
    assert check_fcs(frame, spec)


def test_fcs_detects_a_single_bit_flip() -> None:
    frame = bytearray(append_fcs(b"payload", CRC32))
    frame[2] ^= 0x01
    assert not check_fcs(bytes(frame), CRC32)


def test_fcs_too_short_raises() -> None:
    with pytest.raises(ValueError, match="too short"):
        check_fcs(b"abc", CRC32)


def test_str_input_raises_typeerror() -> None:
    with pytest.raises(TypeError, match="bytes"):
        crc("123456789", CRC32)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        crc_bitwise("123456789", CRC32)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 12},
        {"poly": 1 << 40},
        {"init": 1 << 40},
        {"xorout": 1 << 40},
    ],
)
def test_bad_spec_raises(kwargs: dict) -> None:
    base = {
        "name": "x",
        "width": 32,
        "poly": 1,
        "init": 0,
        "refin": False,
        "refout": False,
        "xorout": 0,
        "check": 0,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        CrcSpec(**base)  # type: ignore[arg-type]


@given(data=st.binary(min_size=0, max_size=300))
@settings(max_examples=150, deadline=None)
def test_crc32_equals_zlib_property(data: bytes) -> None:
    assert crc(data, CRC32) == zlib.crc32(data)


@given(data=st.binary(min_size=1, max_size=120))
@settings(max_examples=120, deadline=None)
def test_fcs_accepts_untouched_frames(data: bytes) -> None:
    assert check_fcs(append_fcs(data, CRC16_CCITT_FALSE), CRC16_CCITT_FALSE)
