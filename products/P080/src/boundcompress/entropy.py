"""Residual-to-symbol mapping, frequency tables, and the entropy reference figure.

The predictive coder in :mod:`boundcompress.codec` produces signed integer
residuals of unbounded magnitude. The rANS coder in :mod:`boundcompress.rans`
wants a small alphabet with a shipped frequency table. This module is the
bridge, and it is where the accounting of "what did the guarantee cost" is
actually done, byte by byte.

Mapping
-------

1. **Zigzag.** A signed residual is folded onto the non-negative integers by
   :math:`z = 2e` for :math:`e \\ge 0` and :math:`z = -2e - 1` for
   :math:`e < 0`, the mapping used by Protocol Buffers and by LEB128 signed
   varints. It is a bijection on the integers, which is asserted by a
   Hypothesis property test, and it puts small-magnitude residuals -- the ones
   a working predictor produces -- on small symbols.
2. **Escape.** Symbols :math:`0 \\ldots 254` are coded directly by rANS.
   Anything larger is coded as symbol :data:`ESCAPE` and the full zigzag value
   is appended to a side stream as an unsigned LEB128 varint. The escape
   stream is *not* entropy coded; its length is counted in the rate in full.
   This keeps the alphabet at 256 and the shipped table small, and makes the
   cost of a prediction failure explicit rather than hidden in a wider
   alphabet.

Entropy as a reference figure, never as a floor
-----------------------------------------------

:func:`shannon_entropy_bits` returns the first-order empirical entropy of the
symbol stream in bits per symbol -- Shannon 1948, "A mathematical theory of
communication", the source-coding theorem. Two things are true about it and
both are stated wherever it appears in this repository:

* It is a reference figure for a *memoryless* source. The symbol stream here
  is the output of a predictor and is correlated, so a coder with memory can
  and does go below it. It is therefore **not a lower bound on the achievable
  rate**, and this package does not present it as one.
* It ignores every byte of header, frequency table and escape stream. The
  realised rate includes all of them, which is why the realised rate can also
  be *above* the entropy by a wide margin on a short array.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Size of the rANS alphabet. 255 direct symbols plus one escape.
ALPHABET = 256

#: Symbol index reserved for "the zigzag value does not fit; read a varint".
ESCAPE = 255


class SymbolError(ValueError):
    """Raised when a symbol stream or varint block is malformed."""


#: Largest absolute residual :func:`zigzag` can fold without wrapping int64.
MAX_FOLDABLE = 2**62


def zigzag(values: np.ndarray) -> np.ndarray:
    """Fold signed integers onto the non-negative integers. Bijective.

    Raises
    ------
    SymbolError
        If any value's fold would exceed ``int64``. The quantiser's
        :data:`boundcompress.quantise.MAX_SYMBOL_MAGNITUDE` guard makes this
        unreachable through :func:`boundcompress.codec.encode`; it is kept
        because the function is public and a silent wrap here produced a
        negative symbol index and an ``IndexError`` three frames away.
    """
    array = np.asarray(values, dtype=np.int64)
    if array.size and int(np.max(np.abs(array))) >= MAX_FOLDABLE:
        worst = int(np.argmax(np.abs(array)))
        raise SymbolError(
            f"value {int(array[worst])} at index {worst} cannot be zigzag folded "
            f"without exceeding int64; the limit is {MAX_FOLDABLE}"
        )
    return np.where(array >= 0, 2 * array, -2 * array - 1).astype(np.int64)


def unzigzag(values: np.ndarray) -> np.ndarray:
    """Inverse of :func:`zigzag`."""
    array = np.asarray(values, dtype=np.int64)
    return np.where(array % 2 == 0, array // 2, -(array + 1) // 2).astype(np.int64)


def put_varint(value: int, out: bytearray) -> None:
    """Append ``value`` to ``out`` as an unsigned LEB128 varint.

    Raises
    ------
    SymbolError
        If ``value`` is negative. Signed values are zigzagged first.
    """
    if value < 0:
        raise SymbolError(f"varints are unsigned; zigzag first (got {value})")
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return


def get_varint(data: bytes, position: int) -> tuple[int, int]:
    """Read one unsigned LEB128 varint. Returns ``(value, next_position)``.

    Raises
    ------
    SymbolError
        If the varint runs off the end of ``data``.
    """
    value = 0
    shift = 0
    while True:
        if position >= len(data):
            raise SymbolError("varint block truncated")
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, position
        shift += 7


@dataclass(frozen=True)
class SymbolStream:
    """A residual stream split into rANS symbols and an escape varint block.

    Attributes
    ----------
    symbols:
        Alphabet indices, one per residual.
    escapes:
        LEB128 varint block holding the full zigzag value of every residual
        that did not fit in a direct symbol, in order of occurrence.
    escape_count:
        Number of escaped residuals. Equal to ``symbols.count(ESCAPE)`` and
        carried separately so the container can be validated without
        re-scanning.
    """

    symbols: list[int]
    escapes: bytes
    escape_count: int


def to_symbols(residuals: np.ndarray) -> SymbolStream:
    """Map signed residuals to a :class:`SymbolStream`."""
    folded = zigzag(residuals)
    symbols: list[int] = []
    escapes = bytearray()
    escape_count = 0
    for value in folded.tolist():
        if value < ESCAPE:
            symbols.append(int(value))
        else:
            symbols.append(ESCAPE)
            put_varint(int(value), escapes)
            escape_count += 1
    return SymbolStream(symbols, bytes(escapes), escape_count)


def from_symbols(symbols: list[int], escapes: bytes) -> np.ndarray:
    """Inverse of :func:`to_symbols`. Returns signed ``int64`` residuals.

    Raises
    ------
    SymbolError
        If a symbol is outside the alphabet, or if the escape block is
        truncated or has bytes left over -- either of which means the
        container is inconsistent with itself.
    """
    folded: list[int] = []
    position = 0
    for symbol in symbols:
        if not 0 <= symbol < ALPHABET:
            raise SymbolError(f"symbol {symbol} is outside the alphabet of {ALPHABET}")
        if symbol == ESCAPE:
            value, position = get_varint(escapes, position)
            folded.append(value)
        else:
            folded.append(symbol)
    if position != len(escapes):
        raise SymbolError(
            f"escape block has {len(escapes) - position} unread byte(s); the container "
            "does not describe its own payload"
        )
    return unzigzag(np.asarray(folded, dtype=np.int64))


def symbol_counts(symbols: list[int]) -> list[int]:
    """Occurrence counts over the full alphabet."""
    counts = [0] * ALPHABET
    for symbol in symbols:
        counts[symbol] += 1
    return counts


def shannon_entropy_bits(symbols: list[int]) -> float:
    """First-order empirical entropy of ``symbols``, in bits per symbol.

    This is a reference figure for a memoryless source (Shannon 1948). It is
    **not** a lower bound on the achievable rate of this coder, because the
    symbol stream is correlated; and it excludes every header, table and
    escape byte that the realised rate includes. Returns 0.0 for an empty
    stream.
    """
    if not symbols:
        return 0.0
    total = len(symbols)
    bits = 0.0
    for count in symbol_counts(symbols):
        if count:
            probability = count / total
            bits -= probability * math.log2(probability)
    return bits


def value_entropy_bits(residuals: np.ndarray) -> float:
    """First-order empirical entropy over the distinct residual *values*, in bits.

    :func:`shannon_entropy_bits` is computed over the 256-symbol alphabet the
    rANS coder sees, in which every residual too large for a direct symbol has
    been collapsed onto :data:`ESCAPE`. On a stream where most residuals
    escape, that figure collapses towards zero while the information has moved
    into the varint block, and it is then not comparable with a byte count.
    Measured on this repository's own signals: the symbol-alphabet entropy of
    the ``white_noise`` residual stream at the median swept bound is 0.4546
    bits per symbol while the realised payload is thousands of bytes, because
    almost every residual escapes.

    This function has no escape, so it is the figure to compare against a
    byte count. It is still a **memoryless reference and not a floor** -- a
    coder with memory goes below it, which
    ``validation/validate_rate_distortion.py`` measures directly -- and it
    still excludes every header byte a realised rate includes. Returns 0.0 for
    an empty stream.
    """
    array = np.asarray(residuals, dtype=np.int64)
    if array.size == 0:
        return 0.0
    _, counts = np.unique(array, return_counts=True)
    probabilities = counts / array.size
    return float(-np.sum(probabilities * np.log2(probabilities)))


def table_bytes(frequencies: list[int]) -> bytes:
    """Serialise a normalised frequency table as ``count``, then symbol/freq pairs.

    Only symbols with a non-zero frequency are written: one byte for the
    symbol index and a varint for the frequency. The table is part of the
    rate and is counted in it.
    """
    used = [(symbol, freq) for symbol, freq in enumerate(frequencies) if freq > 0]
    out = bytearray()
    put_varint(len(used), out)
    for symbol, freq in used:
        out.append(symbol)
        put_varint(freq, out)
    return bytes(out)


def table_from_bytes(data: bytes, position: int) -> tuple[list[int], int]:
    """Inverse of :func:`table_bytes`. Returns ``(frequencies, next_position)``.

    Raises
    ------
    SymbolError
        If the table is truncated, repeats a symbol, or names a symbol outside
        the alphabet.
    """
    used, position = get_varint(data, position)
    frequencies = [0] * ALPHABET
    for _ in range(used):
        if position >= len(data):
            raise SymbolError("frequency table truncated")
        symbol = data[position]
        position += 1
        if symbol >= ALPHABET:  # pragma: no cover - a byte cannot exceed 255
            raise SymbolError(f"frequency table names symbol {symbol}")
        if frequencies[symbol]:
            raise SymbolError(f"frequency table repeats symbol {symbol}")
        frequency, position = get_varint(data, position)
        frequencies[symbol] = frequency
    return frequencies, position
