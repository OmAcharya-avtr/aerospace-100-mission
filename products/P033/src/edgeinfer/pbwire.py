"""Minimal protobuf wire-format codec, sufficient for ONNX ModelProto.

Why this exists: `onnxruntime` is installed in this environment but the `onnx`
package (which provides the generated protobuf bindings and the graph helper
API) is not. `onnxruntime.InferenceSession` accepts a serialised ModelProto as
`bytes`, so a small encoder is all that is needed to build models for
characterisation, and a small decoder is all that is needed to read one back.

Wire format reference: Protocol Buffers encoding specification, Google,
"Encoding" (protobuf.dev/programming-guides/encoding). Only the four wire
types used by ONNX are implemented:

==========  ====  ===================================
Wire type   Code  Payload
==========  ====  ===================================
VARINT      0     base-128 varint
FIXED64     1     8 bytes little-endian
LEN         2     varint length then that many bytes
FIXED32     5     4 bytes little-endian
==========  ====  ===================================

Field numbers are not hard-coded here; they live in :mod:`edgeinfer.onnx_io`,
where they are verified empirically against a model shipped inside
``onnxruntime`` (see ``tests/test_onnx_io.py``).

Signed/zigzag varints, groups (wire types 3 and 4) and unknown-field
preservation are not implemented because ONNX ModelProto does not need them
for the subset of operators this package characterises.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator

__all__ = [
    "WIRE_FIXED32",
    "WIRE_FIXED64",
    "WIRE_LEN",
    "WIRE_VARINT",
    "encode_varint",
    "field_bytes",
    "field_message",
    "field_string",
    "field_varint",
    "iter_fields",
    "packed_varints",
    "read_varint",
]

WIRE_VARINT = 0
WIRE_FIXED64 = 1
WIRE_LEN = 2
WIRE_FIXED32 = 5

_MAX_VARINT_BYTES = 10  # 64-bit value, 7 payload bits per byte


def encode_varint(value: int) -> bytes:
    """Encode a non-negative integer as a base-128 varint.

    Parameters
    ----------
    value
        Non-negative integer, dimensionless.

    Raises
    ------
    ValueError
        If ``value`` is negative. Negative field values do not occur in the
        ONNX subset used here, and encoding them would require either
        ten-byte two's-complement or zigzag encoding depending on the declared
        protobuf type, which cannot be inferred from the value alone.
    """
    if value < 0:
        raise ValueError(f"encode_varint accepts non-negative values only, got {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _tag(field: int, wire_type: int) -> bytes:
    if field < 1:
        raise ValueError(f"protobuf field numbers start at 1, got {field}")
    return encode_varint((field << 3) | wire_type)


def field_varint(field: int, value: int) -> bytes:
    """Serialise one VARINT field."""
    return _tag(field, WIRE_VARINT) + encode_varint(value)


def field_bytes(field: int, payload: bytes) -> bytes:
    """Serialise one length-delimited field from raw bytes."""
    return _tag(field, WIRE_LEN) + encode_varint(len(payload)) + payload


def field_string(field: int, text: str) -> bytes:
    """Serialise one length-delimited field from a UTF-8 string."""
    return field_bytes(field, text.encode("utf-8"))


def field_message(field: int, message: bytes) -> bytes:
    """Serialise one embedded message (identical wire form to bytes)."""
    return field_bytes(field, message)


def packed_varints(field: int, values: list[int]) -> bytes:
    """Serialise a repeated varint field in packed form."""
    body = b"".join(encode_varint(v) for v in values)
    return field_bytes(field, body)


def read_varint(buf: bytes, offset: int) -> tuple[int, int]:
    """Decode one varint.

    Returns
    -------
    (value, new_offset)
    """
    result = 0
    shift = 0
    start = offset
    while True:
        if offset >= len(buf):
            raise ValueError(f"truncated varint starting at byte {start}")
        if offset - start >= _MAX_VARINT_BYTES:
            raise ValueError(f"varint at byte {start} exceeds 64 bits")
        byte = buf[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, offset
        shift += 7


def iter_fields(buf: bytes) -> Iterator[tuple[int, int, object]]:
    """Iterate ``(field_number, wire_type, value)`` over a serialised message.

    ``value`` is an ``int`` for VARINT, ``bytes`` for LEN and FIXED64, and a
    ``float`` for FIXED32. Fields are yielded in the order they appear; a
    repeated field therefore yields once per element.
    """
    offset = 0
    while offset < len(buf):
        key, offset = read_varint(buf, offset)
        field, wire_type = key >> 3, key & 0x07
        if wire_type == WIRE_VARINT:
            value, offset = read_varint(buf, offset)
            yield field, wire_type, value
        elif wire_type == WIRE_LEN:
            length, offset = read_varint(buf, offset)
            if offset + length > len(buf):
                raise ValueError(f"length-delimited field {field} runs past end of buffer")
            yield field, wire_type, buf[offset : offset + length]
            offset += length
        elif wire_type == WIRE_FIXED32:
            yield field, wire_type, struct.unpack_from("<f", buf, offset)[0]
            offset += 4
        elif wire_type == WIRE_FIXED64:
            yield field, wire_type, buf[offset : offset + 8]
            offset += 8
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type} on field {field}")
