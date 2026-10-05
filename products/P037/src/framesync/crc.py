"""CRC-16 Frame Error Control Field, implemented internally.

Definition
----------
CCSDS 132.0-B (TM Space Data Link Protocol) places an optional two-octet
Frame Error Control Field at the end of the transfer frame, computed with
the generator polynomial

    g(x) = x^16 + x^12 + x^5 + 1                                        (4)

over the whole frame preceding the field, with the shift register
pre-loaded to all ones. In the usual CRC-catalogue parameterisation this is
width 16, polynomial 0x1021, init 0xFFFF, non-reflected input and output,
final XOR 0x0000 -- the parameter set commonly catalogued as
CRC-16/IBM-3740, historically also called "CRC-16/CCITT-FALSE".

Known-answer: the check value of that parameter set over the ASCII string
``"123456789"`` is 0x29B1. ``tests/test_crc.py`` asserts it, and
``validation/validate_uncoded_fer.py`` prints it.

Why this is implemented here rather than taken from ``crcmod``
--------------------------------------------------------------
``crcmod`` is the mature choice and is named in the README alternatives
table. It cannot be installed in this build container (no wheel, and the
source build of its C extension fails), so the 30 lines below stand in.
A reader with a working toolchain should prefer ``crcmod``.

Undetected error probability
----------------------------
A 16-bit CRC maps 2^16 syndromes onto error patterns; for error patterns
distributed uniformly over the non-zero patterns the undetected fraction
tends to 2^-16 = 1.526e-5. That limit is the reason this package reports
*both* the true frame error rate (payload comparison) and the
CRC-detected frame error rate, and reports the gap; see
``validation/validate_uncoded_fer.py``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CRC16_POLY", "CRC16_INIT", "crc16", "crc16_batch", "FECF_BITS"]

CRC16_POLY = 0x1021
"""Generator polynomial of Eq. (4) in Koopman-free 'normal' notation."""

CRC16_INIT = 0xFFFF
"""All-ones pre-load, as CCSDS 132.0-B specifies."""

FECF_BITS = 16
"""Width of the Frame Error Control Field, in bits."""


def _build_table(poly: int) -> np.ndarray:
    table = np.zeros(256, dtype=np.uint16)
    for byte in range(256):
        crc = byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
        table[byte] = crc
    return table


_TABLE = _build_table(CRC16_POLY)


def crc16(data: bytes | bytearray | np.ndarray, init: int = CRC16_INIT) -> int:
    """CRC-16 of a byte string under Eq. (4). Returns a 16-bit integer.

    Parameters
    ----------
    data : bytes-like or ndarray of uint8
        The octets to protect, in transmission order.
    init : int
        Shift-register pre-load, 0 <= init <= 0xFFFF. Defaults to all ones.
    """
    if not 0 <= int(init) <= 0xFFFF:
        raise ValueError(f"init must be a 16-bit value, got {init!r}")
    buf = np.asarray(bytearray(data) if not isinstance(data, np.ndarray) else data, dtype=np.uint8)
    if buf.ndim != 1:
        raise ValueError("crc16 expects a 1-D sequence of octets")
    crc = int(init)
    for byte in buf.tolist():
        crc = (int(_TABLE[((crc >> 8) ^ byte) & 0xFF]) ^ ((crc << 8) & 0xFFFF)) & 0xFFFF
    return crc


def crc16_batch(data: np.ndarray, init: int = CRC16_INIT) -> np.ndarray:
    """CRC-16 of every row of a (B, n_octets) uint8 array, vectorised.

    Same polynomial and pre-load as :func:`crc16`; the loop runs over octet
    index rather than over rows, so cost is ``n_octets`` NumPy operations on
    length-B arrays instead of B*n_octets Python steps. ``tests/test_crc.py``
    asserts row-by-row agreement with :func:`crc16`.

    Returns
    -------
    ndarray of uint16
        One CRC per row.
    """
    arr = np.atleast_2d(np.asarray(data, dtype=np.uint8))
    if arr.ndim != 2:
        raise ValueError("crc16_batch expects a 2-D (n_frames, n_octets) uint8 array")
    if not 0 <= int(init) <= 0xFFFF:
        raise ValueError(f"init must be a 16-bit value, got {init!r}")
    crc = np.full(arr.shape[0], int(init), dtype=np.uint16)
    for j in range(arr.shape[1]):
        idx = ((crc >> 8).astype(np.uint16) ^ arr[:, j].astype(np.uint16)) & 0xFF
        crc = (_TABLE[idx] ^ ((crc << np.uint16(8)) & np.uint16(0xFFFF))).astype(np.uint16)
    return crc
