"""Attached Sync Marker (ASM) patterns and the correlation detector.

The marker
----------
CCSDS 131.0-B (TM Synchronization and Channel Coding) prefixes each
transfer frame or codeblock on the channel with an Attached Sync Marker.
For all non-turbo-coded transfer frames the marker is the 32-bit pattern

    0x1ACFFC1D   =   0001 1010 1100 1111 1111 1100 0001 1101

transmitted most-significant bit first. CCSDS defines longer markers for
turbo-coded frames; this package implements the 32-bit marker only, and the
detector is written against an arbitrary bit pattern so a different marker
can be supplied.

The detector
------------
A correlation detector slides the 32-bit reference over the bit stream and
declares a marker at offset ``i`` when the Hamming distance between the
reference and the 32-bit window starting at ``i`` is at most a tolerance T:

    D(i) = sum_{j=0}^{L-1} ( r[i+j] XOR asm[j] )   <= T                 (2)

For hard-decision input this correlation is exactly ``L - 2*D``, so a
Hamming-distance threshold and a correlation threshold are the same test.

False-sync probability
----------------------
For a stream of independent, uniformly distributed bits each of the L
comparisons in (2) mismatches with probability 1/2, independently, so the
window Hamming distance is Binomial(L, 1/2) and

    P_fa(T) = P[D <= T] = 2^-L * sum_{k=0}^{T} C(L, k)                  (3)

This is the combinatorial expression validated in
``validation/validate_false_sync.py``. For L = 32:
T = 0 gives 2^-32; T = 1 gives 33 * 2^-32.

Equation (3) is the probability **per window position**. Over a stream of N
bits there are N - L + 1 window positions, so by linearity of expectation
the expected number of false declarations is (N - L + 1) * P_fa(T), exactly,
even though overlapping windows are not independent events. Converting that
expectation into a distribution would require the overlap structure and this
package does not attempt it; only the expectation is claimed.

Assumptions and validity: (3) holds for an i.i.d. uniform bit stream, which
is the usual model for a randomised channel (CCSDS pseudo-randomiser) or for
the search state before acquisition. It does **not** hold inside a payload
with structure, nor for the self-correlation of the marker with shifts of
itself, which is why ``asm_autocorrelation`` is provided separately.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "ASM_32_HEX",
    "ASM_32_BITS",
    "bits_from_int",
    "int_from_bits",
    "hamming_distances",
    "detect_asm",
    "false_sync_probability",
    "expected_false_syncs",
    "asm_autocorrelation",
]

ASM_32_HEX = 0x1ACFFC1D
"""CCSDS 131.0-B attached sync marker for non-turbo-coded transfer frames."""


def bits_from_int(value: int, length: int) -> np.ndarray:
    """Most-significant-bit-first 0/1 array of ``length`` bits from an integer."""
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if value < 0 or value >> length:
        raise ValueError(f"value 0x{value:X} does not fit in {length} bits")
    shifts = np.arange(length - 1, -1, -1, dtype=np.int64)
    return ((int(value) >> shifts) & 1).astype(np.uint8)


def int_from_bits(bits: np.ndarray) -> int:
    """Integer from a most-significant-bit-first 0/1 array."""
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    out = 0
    for b in arr:
        out = (out << 1) | int(b)
    return out


ASM_32_BITS = bits_from_int(ASM_32_HEX, 32)
"""The 32-bit marker as a 0/1 uint8 array, MSB first."""


def hamming_distances(stream: np.ndarray, pattern: np.ndarray | None = None) -> np.ndarray:
    """Hamming distance of ``pattern`` against every window of ``stream``.

    Implements D(i) of Eq. (2) for every valid offset i, vectorised with a
    sliding window view.

    Parameters
    ----------
    stream : ndarray of uint8
        1-D array of 0/1 bits, length >= len(pattern).
    pattern : ndarray of uint8, optional
        The marker bits. Defaults to the 32-bit CCSDS marker.

    Returns
    -------
    ndarray of int32
        Length ``len(stream) - len(pattern) + 1``; element i is D(i), in bits.
    """
    pat = ASM_32_BITS if pattern is None else np.asarray(pattern, dtype=np.uint8).ravel()
    s = np.asarray(stream, dtype=np.uint8).ravel()
    if pat.size == 0:
        raise ValueError("pattern must be non-empty")
    if s.size < pat.size:
        raise ValueError(f"stream of {s.size} bits is shorter than the {pat.size}-bit pattern")
    if (s.size and s.max() > 1) or pat.max() > 1:
        raise ValueError("stream and pattern must contain only 0/1 bit values")
    win = np.lib.stride_tricks.sliding_window_view(s, pat.size)
    return np.count_nonzero(win ^ pat, axis=1).astype(np.int32)


def detect_asm(
    stream: np.ndarray, tolerance: int = 0, pattern: np.ndarray | None = None
) -> np.ndarray:
    """Bit offsets at which the marker is declared under Eq. (2).

    Parameters
    ----------
    stream : ndarray of uint8
        1-D array of 0/1 bits.
    tolerance : int
        Maximum tolerated Hamming distance T, in bits, 0 <= T <= len(pattern).
    pattern : ndarray of uint8, optional
        Marker bits; defaults to the 32-bit CCSDS marker.

    Returns
    -------
    ndarray of int64
        Sorted bit offsets where D(i) <= T. Overlapping declarations are not
        suppressed; the caller decides what to do with them.
    """
    pat = ASM_32_BITS if pattern is None else np.asarray(pattern, dtype=np.uint8).ravel()
    t = int(tolerance)
    if t < 0 or t > pat.size:
        raise ValueError(f"tolerance must lie in [0, {pat.size}] bits, got {tolerance!r}")
    return np.flatnonzero(hamming_distances(stream, pat) <= t).astype(np.int64)


def false_sync_probability(tolerance: int, length: int = 32) -> float:
    """Per-window false-sync probability P_fa(T) of Eq. (3), dimensionless.

    Parameters
    ----------
    tolerance : int
        Hamming-distance threshold T in bits, 0 <= T <= length.
    length : int
        Marker length L in bits, positive.

    Returns
    -------
    float
        2^-L * sum_{k=0}^{T} C(L, k), exact rational arithmetic then one
        float division, so no accumulated rounding.
    """
    L = int(length)
    t = int(tolerance)
    if L <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if t < 0 or t > L:
        raise ValueError(f"tolerance must lie in [0, {L}] bits, got {tolerance!r}")
    numer = sum(math.comb(L, k) for k in range(t + 1))
    return numer / float(2**L)


def expected_false_syncs(n_bits: int, tolerance: int, length: int = 32) -> float:
    """Expected count of false declarations in an i.i.d. uniform stream.

    ``(n_bits - length + 1) * P_fa(T)``. Exact as an expectation by linearity
    even though overlapping windows are dependent (see module docstring).
    """
    n = int(n_bits)
    L = int(length)
    if n < L:
        raise ValueError(f"n_bits ({n}) must be at least the marker length ({L})")
    return (n - L + 1) * false_sync_probability(tolerance, L)


def asm_autocorrelation(pattern: np.ndarray | None = None) -> np.ndarray:
    """Hamming distance between the marker and each cyclic shift of itself.

    Index s holds the distance at shift s, in bits; index 0 is 0 by
    construction. A marker is chosen so that the non-zero shifts stay far
    from 0, which is what keeps the sliding detector from locking onto the
    marker's own tail. Reported, not asserted: this package measures the
    property rather than claiming a design criterion from the standard.
    """
    pat = ASM_32_BITS if pattern is None else np.asarray(pattern, dtype=np.uint8).ravel()
    L = pat.size
    return np.array([np.count_nonzero(pat ^ np.roll(pat, s)) for s in range(L)], dtype=np.int32)
