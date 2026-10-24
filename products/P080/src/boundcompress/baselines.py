"""Baseline coders, measured in actual bytes and verified by round-trip.

What the specification asked for, and what was available
--------------------------------------------------------

The batch specification for this product names ``numcodecs`` (0.17.0) and
``zfpy`` (1.0.1) as baseline 2 and requires a rate-distortion comparison
against them at matched distortion. **Neither package is installed in the
pinned environment this product was measured in**, and no package was
installed to get them, so **no rate-distortion number anywhere in this
repository is taken from either of them.** They are named in the README's
alternatives table as the right tools for compressing an array, and that is
all they are: named.

In their place the comparison is run against coders that *are* available and
that a reader can therefore reproduce exactly: ``zlib``, ``lzma`` and ``bz2``
from the Python standard library. The substitution is recorded as a deviation
in ``validation/VALIDATION.md``. It is not a like-for-like replacement --
zfp is a transform codec with its own quantiser and ``numcodecs`` wraps
Blosc, Zstandard and friends -- but it does give a real, measured,
reproducible comparison point for the question that matters here, which is
whether the package's own entropy coder is doing anything a general-purpose
compressor would not.

Matched distortion, exactly
---------------------------

Every coder in this module compresses the **same integer symbol stream** the
package's own coder compresses, produced by the same quantiser at the same
bound. The reconstruction is therefore bit-identical across all of them and
the distortion is matched exactly rather than approximately. The only thing
that differs is the number of bytes.

Accounting
----------

Each baseline is charged the same fixed header
(:data:`boundcompress.container.FIXED_HEADER_BYTES`) as the package's own
container, because a decoder needs the bound, the step, the sample count and
the mode whatever the entropy coder is. Nothing else is added and nothing is
subtracted. Every size reported here is ``len()`` of a byte string that
:func:`roundtrip` has decompressed back to the original symbol stream.

A lossless reference is included: ``lzma`` and ``zlib`` over the raw float64
bytes of the input array, at zero distortion. It is the honest upper anchor of
the rate-distortion plot and answers "what am I actually buying by accepting
an error bound at all".
"""

from __future__ import annotations

import bz2
import lzma
import zlib
from dataclasses import dataclass

import numpy as np

from .container import FIXED_HEADER_BYTES
from .entropy import get_varint, put_varint, unzigzag, zigzag

#: Coders applied to the quantised symbol or residual stream, at matched distortion.
LOSSY_METHODS: tuple[str, ...] = (
    "zlib_residual",
    "lzma_residual",
    "bz2_residual",
    "zlib_symbols",
    "lzma_symbols",
    "bz2_symbols",
)

#: Coders applied to the untouched float64 input, at zero distortion.
LOSSLESS_METHODS: tuple[str, ...] = ("zlib_float64", "lzma_float64", "bz2_float64")

_COMPRESSORS = {
    "zlib": lambda data: zlib.compress(data, 9),
    "lzma": lambda data: lzma.compress(data, preset=6),
    "bz2": lambda data: bz2.compress(data, 9),
}
_DECOMPRESSORS = {
    "zlib": zlib.decompress,
    "lzma": lzma.decompress,
    "bz2": bz2.decompress,
}


class BaselineError(ValueError):
    """Raised for an unknown baseline method name."""


@dataclass(frozen=True)
class BaselineResult:
    """One baseline measurement.

    Attributes
    ----------
    method:
        Name from :data:`LOSSY_METHODS` or :data:`LOSSLESS_METHODS`.
    payload_bytes:
        Length of the compressed byte string alone.
    total_bytes:
        ``payload_bytes`` plus the fixed header every decoder needs.
    bits_per_sample:
        ``8 * total_bytes / samples``.
    roundtrip_ok:
        Whether decompressing reproduced the input exactly. A baseline whose
        round-trip fails is reported, not dropped.
    """

    method: str
    payload_bytes: int
    total_bytes: int
    bits_per_sample: float
    roundtrip_ok: bool


def varint_stream(residuals: np.ndarray) -> bytes:
    """Serialise signed integers as zigzag LEB128 varints, the dense byte form."""
    out = bytearray()
    for value in zigzag(np.asarray(residuals, dtype=np.int64)).tolist():
        put_varint(int(value), out)
    return bytes(out)


def varint_unstream(data: bytes, count: int) -> np.ndarray:
    """Inverse of :func:`varint_stream`. Reads exactly ``count`` values."""
    folded: list[int] = []
    position = 0
    for _ in range(count):
        value, position = get_varint(data, position)
        folded.append(value)
    return unzigzag(np.asarray(folded, dtype=np.int64))


def symbol_stream(symbols: np.ndarray) -> bytes:
    """Serialise integer symbols as little-endian int64, the naive byte form."""
    return np.asarray(symbols, dtype=np.int64).tobytes()


def _family(method: str) -> str:
    for name in _COMPRESSORS:
        if method.startswith(name + "_"):
            return name
    raise BaselineError(f"unknown baseline method {method!r}")


def measure(
    method: str,
    *,
    values: np.ndarray,
    symbols: np.ndarray,
    residuals: np.ndarray,
) -> BaselineResult:
    """Compress with ``method``, verify the round-trip, and report actual bytes.

    Parameters
    ----------
    method:
        One of :data:`LOSSY_METHODS` or :data:`LOSSLESS_METHODS`.
    values:
        The original float64 array; used only by the lossless methods.
    symbols:
        Quantised integer symbols at the declared bound.
    residuals:
        Predictor residuals of ``symbols``, which the ``*_residual`` methods
        compress.

    Raises
    ------
    BaselineError
        For an unknown method name.
    """
    family = _family(method)
    compress = _COMPRESSORS[family]
    decompress = _DECOMPRESSORS[family]
    samples = int(np.asarray(values).size)

    if method.endswith("_residual"):
        raw = varint_stream(residuals)
        blob = compress(raw)
        ok = decompress(blob) == raw
    elif method.endswith("_symbols"):
        raw = symbol_stream(symbols)
        blob = compress(raw)
        ok = decompress(blob) == raw
    elif method.endswith("_float64"):
        raw = np.ascontiguousarray(values, dtype=np.float64).tobytes()
        blob = compress(raw)
        ok = decompress(blob) == raw
    else:
        raise BaselineError(f"unknown baseline method {method!r}")

    total = len(blob) + FIXED_HEADER_BYTES
    return BaselineResult(
        method=method,
        payload_bytes=len(blob),
        total_bytes=total,
        bits_per_sample=8.0 * total / samples if samples else 0.0,
        roundtrip_ok=bool(ok),
    )


def roundtrip(
    method: str, *, values: np.ndarray, symbols: np.ndarray, residuals: np.ndarray
) -> bool:
    """Whether ``method`` decompresses back to exactly what it was given."""
    return measure(method, values=values, symbols=symbols, residuals=residuals).roundtrip_ok


def measure_all(
    *, values: np.ndarray, symbols: np.ndarray, residuals: np.ndarray
) -> dict[str, BaselineResult]:
    """Every baseline in :data:`LOSSY_METHODS` + :data:`LOSSLESS_METHODS`."""
    return {
        method: measure(method, values=values, symbols=symbols, residuals=residuals)
        for method in LOSSY_METHODS + LOSSLESS_METHODS
    }


def entropy_floor_bytes(entropy_bits: float, samples: int) -> float:
    """Bytes the first-order entropy figure corresponds to, header excluded.

    Reported in the rate-distortion tables as a reference line and never as a
    floor: the symbol stream is correlated, so a coder with memory can go
    below it, and this figure also ignores the header bytes every realised
    rate includes. See :func:`boundcompress.entropy.shannon_entropy_bits`.
    """
    return entropy_bits * samples / 8.0
