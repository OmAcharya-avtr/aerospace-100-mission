"""The container format. Every rate figure in this repository is ``len()`` of one.

Layout, little-endian, no padding
---------------------------------

=========================  ======  ===============================================
field                      bytes   meaning
=========================  ======  ===============================================
``magic``                  4       ``b"BCMP"``
``version``                1       :data:`VERSION`
``mode``                   1       0 ``abs``, 1 ``rel``, 2 ``range_rel``
``transform``              1       0 identity, 1 log magnitude
``sign``                   1       0 negative, 1 positive (log transform only)
``predictor``              1       :data:`boundcompress.predictors.PREDICTOR_IDS`
``order``                  1       warm-up length in samples
``declared_bound``         8       float64, as declared by the caller
``step``                   8       float64, quantiser step in the transformed domain
``count``                  4       uint32, number of samples
``escape_count``           4       uint32, number of escaped residuals
``state``                  4       uint32, final rANS state
``warmup``                 var     ``order`` zigzag varints
``table``                  var     normalised frequency table
``escapes``                var     ``escape_length`` varint then that many bytes
``payload``                var     ``payload_length`` varint then the rANS words
=========================  ======  ===============================================

The quantiser step is carried as a float64 rather than recomputed from the
declared bound, so a decoder reproduces the reconstruction bit-for-bit without
re-deriving ``2 * log1p(r)`` on a possibly different libm. The declared bound
travels with it because the certificate has to be able to state what was
promised, not only what was delivered.

Nothing in the header is compressed and nothing is omitted from the rate. The
fixed header is :data:`FIXED_HEADER_BYTES` bytes; on a 4096-sample array that
is well under one per cent of the container, and on a 64-sample array it is
most of it. Both regimes are measured in
``validation/validate_rate_distortion_output.txt``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from .entropy import (
    SymbolError,
    get_varint,
    put_varint,
    table_bytes,
    table_from_bytes,
    zigzag,
)
from .quantise import MODES

MAGIC = b"BCMP"
VERSION = 1

#: ``magic`` through ``state`` inclusive.
FIXED_HEADER_BYTES = 4 + 1 + 1 + 1 + 1 + 1 + 1 + 8 + 8 + 4 + 4 + 4

_FIXED = struct.Struct("<4sBBBBBBddIII")
assert _FIXED.size == FIXED_HEADER_BYTES

MODE_IDS: dict[str, int] = {name: index for index, name in enumerate(MODES)}
MODE_NAMES: dict[int, str] = {index: name for name, index in MODE_IDS.items()}


class ContainerError(ValueError):
    """Raised when a byte string is not a container this version can read."""


@dataclass(frozen=True)
class Container:
    """Decoded header fields plus the three variable-length blocks."""

    mode: str
    transform: int
    sign: int
    predictor: int
    order: int
    declared_bound: float
    step: float
    count: int
    escape_count: int
    state: int
    warmup: list[int]
    frequencies: list[int]
    escapes: bytes
    payload: bytes


def pack(container: Container) -> bytes:
    """Serialise a :class:`Container`. The returned length is the rate."""
    out = bytearray(
        _FIXED.pack(
            MAGIC,
            VERSION,
            MODE_IDS[container.mode],
            container.transform,
            1 if container.sign >= 0 else 0,
            container.predictor,
            container.order,
            container.declared_bound,
            container.step,
            container.count,
            container.escape_count,
            container.state,
        )
    )
    for value in zigzag(np.asarray(container.warmup, dtype=np.int64)).tolist():
        put_varint(int(value), out)
    out += table_bytes(container.frequencies)
    put_varint(len(container.escapes), out)
    out += container.escapes
    put_varint(len(container.payload), out)
    out += container.payload
    return bytes(out)


def unpack(blob: bytes) -> Container:
    """Parse a container.

    Raises
    ------
    ContainerError
        On a short buffer, a wrong magic, an unsupported version, an unknown
        mode, trailing bytes, or any block that runs off the end. A container
        that does not describe itself is rejected rather than decoded on a
        best-effort basis; this codec exists to make a guarantee, and a
        best-effort parse is incompatible with one.
    """
    if len(blob) < FIXED_HEADER_BYTES:
        raise ContainerError(
            f"buffer of {len(blob)} byte(s) is shorter than the {FIXED_HEADER_BYTES}-byte header"
        )
    (
        magic,
        version,
        mode_id,
        transform,
        sign_bit,
        predictor,
        order,
        declared_bound,
        step,
        count,
        escape_count,
        state,
    ) = _FIXED.unpack(blob[:FIXED_HEADER_BYTES])
    if magic != MAGIC:
        raise ContainerError(f"magic is {magic!r}, expected {MAGIC!r}")
    if version != VERSION:
        raise ContainerError(f"container version {version} is not readable by version {VERSION}")
    if mode_id not in MODE_NAMES:
        raise ContainerError(f"unknown bound mode id {mode_id}")

    position = FIXED_HEADER_BYTES
    # Every parse failure below becomes a ContainerError. A caller handling a
    # malformed buffer should need one exception type, not two: the first
    # version of this function let entropy.SymbolError escape from a truncated
    # varint, which tests/test_container.py caught.
    try:
        warmup: list[int] = []
        for _ in range(order):
            folded, position = get_varint(blob, position)
            warmup.append(int(folded // 2) if folded % 2 == 0 else int(-(folded + 1) // 2))
        frequencies, position = table_from_bytes(blob, position)
        escape_length, position = get_varint(blob, position)
        if position + escape_length > len(blob):
            raise ContainerError("escape block runs past the end of the container")
        escapes = blob[position : position + escape_length]
        position += escape_length
        payload_length, position = get_varint(blob, position)
        if position + payload_length > len(blob):
            raise ContainerError("rANS payload runs past the end of the container")
        payload = blob[position : position + payload_length]
        position += payload_length
    except SymbolError as error:
        raise ContainerError(f"container body is malformed: {error}") from error
    if position != len(blob):
        raise ContainerError(
            f"{len(blob) - position} trailing byte(s) after the payload; the container "
            "does not account for its own length"
        )
    return Container(
        mode=MODE_NAMES[mode_id],
        transform=int(transform),
        sign=1 if sign_bit else -1,
        predictor=int(predictor),
        order=int(order),
        declared_bound=float(declared_bound),
        step=float(step),
        count=int(count),
        escape_count=int(escape_count),
        state=int(state),
        warmup=warmup,
        frequencies=frequencies,
        escapes=escapes,
        payload=payload,
    )


def block_sizes(container: Container) -> dict[str, int]:
    """Byte cost of each part of the container, summing to ``len(pack(container))``.

    Used by ``validation/validate_rate_distortion.py`` and by
    ``examples/rate_breakdown.py`` to show where the bytes went, because
    "the header is negligible" is a claim and not an observation.
    """
    blob = pack(container)
    warmup = bytearray()
    for value in zigzag(np.asarray(container.warmup, dtype=np.int64)).tolist():
        put_varint(int(value), warmup)
    table = table_bytes(container.frequencies)
    escape_prefix = bytearray()
    put_varint(len(container.escapes), escape_prefix)
    payload_prefix = bytearray()
    put_varint(len(container.payload), payload_prefix)
    sizes = {
        "fixed_header": FIXED_HEADER_BYTES,
        "warmup": len(warmup),
        "frequency_table": len(table),
        "escapes": len(escape_prefix) + len(container.escapes),
        "payload": len(payload_prefix) + len(container.payload),
    }
    assert sum(sizes.values()) == len(blob)
    return sizes
