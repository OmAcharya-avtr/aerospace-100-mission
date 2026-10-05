"""Minimal ONNX reader, writer and in-place initializer patcher.

Why this module exists
----------------------
A stored model on a spacecraft is a file in flash, and an upset in that file is
an upset in the bytes of a tensor initializer. To model that honestly the
injection has to act on the real serialised bytes, not on a numpy array that
stands in for them. This module therefore reads and writes ONNX files directly.

The ``onnx`` Python package is not installed in this build environment, so the
protobuf encoding of exactly the fields needed is implemented here against the
field numbers of ``onnx.proto3``:

    ModelProto   : ir_version = 1, producer_name = 2, graph = 7,
                   opset_import = 8
    GraphProto   : node = 1, name = 2, initializer = 5, input = 11, output = 12
    NodeProto    : input = 1, output = 2, name = 3, op_type = 4
    TensorProto  : dims = 1, data_type = 2, name = 8, raw_data = 9
    ValueInfoProto: name = 1, type = 2
    TypeProto    : tensor_type = 1
    TypeProto.Tensor: elem_type = 1, shape = 2
    TensorShapeProto: dim = 1
    TensorShapeProto.Dimension: dim_value = 1, dim_param = 2
    OperatorSetIdProto: domain = 1, version = 2

with ``TensorProto.DataType.FLOAT = 1``. Rather than trusting that list, the
module round-trips every file it writes through ``onnxruntime``, and
``validation/validate_onnx_path.py`` additionally checks that onnxruntime's
output on the written graph equals the numpy forward pass of
:mod:`bitflipsim.network` to float32 round-off. If the encoding were wrong,
onnxruntime would reject the file.

Scope: ``float32`` initializers stored in ``raw_data``, which is what this
module writes and what an exporter produces for a dense MLP. A model whose
weights are in ``float_data`` rather than ``raw_data``, or in any other dtype,
is reported as unsupported rather than silently skipped.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from .network import MlpParameters

FLOAT32_DATA_TYPE = 1


def _varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("negative varints are not emitted by this encoder")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _tag(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _field_varint(field: int, value: int) -> bytes:
    return _tag(field, 0) + _varint(value)


def _field_bytes(field: int, payload: bytes) -> bytes:
    return _tag(field, 2) + _varint(len(payload)) + payload


def _field_string(field: int, text: str) -> bytes:
    return _field_bytes(field, text.encode("utf-8"))


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint in protobuf stream")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint longer than 64 bits")


def _iter_fields(buf: bytes):
    """Yield ``(field_number, wire_type, payload_or_value, next_pos)``."""
    pos = 0
    while pos < len(buf):
        key, pos = _read_varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _read_varint(buf, pos)
            yield field, wire, value
        elif wire == 2:
            length, pos = _read_varint(buf, pos)
            payload = buf[pos : pos + length]
            if len(payload) != length:
                raise ValueError("truncated length-delimited protobuf field")
            pos += length
            yield field, wire, payload
        elif wire == 5:
            payload = buf[pos : pos + 4]
            pos += 4
            yield field, wire, payload
        elif wire == 1:
            payload = buf[pos : pos + 8]
            pos += 8
            yield field, wire, payload
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")


@dataclass(frozen=True)
class Initializer:
    """One float32 initializer located inside a serialised ONNX model.

    Attributes
    ----------
    name:
        Tensor name in the graph.
    dims:
        Shape.
    raw_offset, raw_length:
        Byte range of the tensor's ``raw_data`` inside the whole model file, so
        that a bit can be flipped in the file bytes without re-encoding.
    """

    name: str
    dims: tuple[int, ...]
    raw_offset: int
    raw_length: int

    @property
    def n_elements(self) -> int:
        return self.raw_length // 4

    def array(self, model_bytes: bytes) -> np.ndarray:
        """Decode this initializer from the model bytes as float32."""
        chunk = model_bytes[self.raw_offset : self.raw_offset + self.raw_length]
        return np.frombuffer(chunk, dtype="<f4").reshape(self.dims if self.dims else ())


def _find_subfield(buf: bytes, base: int, field: int, wire: int = 2) -> list[tuple[int, int]]:
    """Return ``(absolute_offset, length)`` for every occurrence of ``field``."""
    found: list[tuple[int, int]] = []
    pos = 0
    while pos < len(buf):
        key, next_pos = _read_varint(buf, pos)
        f, w = key >> 3, key & 7
        if w == 0:
            _, next_pos = _read_varint(buf, next_pos)
        elif w == 2:
            length, data_start = _read_varint(buf, next_pos)
            if f == field and w == wire:
                found.append((base + data_start, length))
            next_pos = data_start + length
        elif w == 5:
            next_pos += 4
        elif w == 1:
            next_pos += 8
        else:
            raise ValueError(f"unsupported protobuf wire type {w}")
        pos = next_pos
    return found


def list_initializers(model_bytes: bytes) -> list[Initializer]:
    """Locate every float32 ``raw_data`` initializer in a serialised ONNX model.

    Raises
    ------
    ValueError
        If the model has no graph, or if an initializer is not float32 or does
        not store its values in ``raw_data``. Reported rather than skipped: an
        upset model that silently ignored half the weights would understate the
        degradation.
    """
    graphs = _find_subfield(model_bytes, 0, field=7)
    if not graphs:
        raise ValueError("no GraphProto (field 7) in this ModelProto")
    g_off, g_len = graphs[0]
    graph = model_bytes[g_off : g_off + g_len]
    out: list[Initializer] = []
    for t_off, t_len in _find_subfield(graph, g_off, field=5):
        tensor = model_bytes[t_off : t_off + t_len]
        dims: list[int] = []
        data_type: int | None = None
        name = ""
        raw: tuple[int, int] | None = None
        pos = 0
        while pos < len(tensor):
            key, next_pos = _read_varint(tensor, pos)
            f, w = key >> 3, key & 7
            if w == 0:
                value, next_pos = _read_varint(tensor, next_pos)
                if f == 1:
                    dims.append(value)
                elif f == 2:
                    data_type = value
            elif w == 2:
                length, data_start = _read_varint(tensor, next_pos)
                payload = tensor[data_start : data_start + length]
                if f == 8:
                    name = payload.decode("utf-8")
                elif f == 9:
                    raw = (t_off + data_start, length)
                elif f == 1:
                    cursor = 0
                    while cursor < len(payload):
                        value, cursor = _read_varint(payload, cursor)
                        dims.append(value)
                next_pos = data_start + length
            elif w == 5:
                next_pos += 4
            elif w == 1:
                next_pos += 8
            else:
                raise ValueError(f"unsupported protobuf wire type {w}")
            pos = next_pos
        if data_type != FLOAT32_DATA_TYPE:
            raise ValueError(
                f"initializer {name!r} has data_type {data_type}, not float32 "
                f"({FLOAT32_DATA_TYPE}); "
                "this module only handles float32 raw_data initializers"
            )
        if raw is None:
            raise ValueError(
                f"initializer {name!r} stores no raw_data; float_data-encoded tensors are "
                "not supported by this module"
            )
        out.append(Initializer(name=name, dims=tuple(dims), raw_offset=raw[0], raw_length=raw[1]))
    return out


def flip_initializer_bit(
    model_bytes: bytes, initializer: Initializer, element_index: int, bit_position: int
) -> bytes:
    """Return a copy of the model with one bit of one initializer element flipped.

    The flip is applied to the file bytes themselves, at
    ``raw_offset + 4 * element_index``, little-endian, which is how the IEEE 754
    word is stored in an ONNX ``raw_data`` field.
    """
    if not 0 <= element_index < initializer.n_elements:
        raise ValueError(
            f"element_index {element_index} outside [0, {initializer.n_elements}) "
            f"for initializer {initializer.name!r}"
        )
    if not 0 <= bit_position < 32:
        raise ValueError(f"bit_position {bit_position} outside [0, 32) for float32")
    offset = initializer.raw_offset + 4 * element_index
    word = struct.unpack_from("<I", model_bytes, offset)[0]
    patched = bytearray(model_bytes)
    struct.pack_into("<I", patched, offset, word ^ (1 << bit_position))
    return bytes(patched)


def build_mlp_onnx(
    params: MlpParameters,
    input_name: str = "x",
    output_name: str = "logits",
    model_name: str = "bitflipsim_reference_mlp",
    opset: int = 13,
    ir_version: int = 8,
) -> bytes:
    """Serialise the reference MLP as an ONNX graph producing the logits.

    Graph: ``MatMul(x, W1) -> Add(b1) -> Relu -> MatMul(W2) -> Add(b2)``. The
    softmax is deliberately left out so that the ONNX output is the quantity the
    clamping bound of :mod:`bitflipsim.mitigation` is stated for.

    Returns
    -------
    bytes
        A serialised ``ModelProto``. onnxruntime loads it; see
        ``validation/validate_onnx_path.py``.
    """
    tensors = params.tensors()
    lay = params.layout

    def tensor_proto(name: str, array: np.ndarray) -> bytes:
        arr = np.ascontiguousarray(np.asarray(array, dtype="<f4"))
        body = b"".join(_field_varint(1, int(d)) for d in arr.shape)
        body += _field_varint(2, FLOAT32_DATA_TYPE)
        body += _field_string(8, name)
        body += _field_bytes(9, arr.tobytes())
        return _field_bytes(5, body)

    def value_info_body(name: str, dims: list[int | str]) -> bytes:
        """ValueInfoProto body (unwrapped), so the caller picks field 11 or 12."""
        dim_entries = b""
        for d in dims:
            if isinstance(d, str):
                dim_entries += _field_bytes(1, _field_string(2, d))
            else:
                dim_entries += _field_bytes(1, _field_varint(1, int(d)))
        shape = _field_bytes(2, dim_entries)
        tensor_type = _field_varint(1, FLOAT32_DATA_TYPE) + shape
        type_proto = _field_bytes(1, tensor_type)
        return _field_string(1, name) + _field_bytes(2, type_proto)

    def node(op_type: str, inputs: list[str], outputs: list[str], name: str) -> bytes:
        body = b"".join(_field_string(1, i) for i in inputs)
        body += b"".join(_field_string(2, o) for o in outputs)
        body += _field_string(3, name)
        body += _field_string(4, op_type)
        return _field_bytes(1, body)

    graph = b""
    graph += node("MatMul", [input_name, "W1"], ["h_pre"], "matmul1")
    graph += node("Add", ["h_pre", "b1"], ["h_bias"], "add1")
    graph += node("Relu", ["h_bias"], ["h"], "relu1")
    graph += node("MatMul", ["h", "W2"], ["o_pre"], "matmul2")
    graph += node("Add", ["o_pre", "b2"], [output_name], "add2")
    graph += _field_string(2, model_name)
    graph += tensor_proto("W1", tensors["W1"])
    graph += tensor_proto("b1", tensors["b1"])
    graph += tensor_proto("W2", tensors["W2"])
    graph += tensor_proto("b2", tensors["b2"])
    graph += _field_bytes(11, value_info_body(input_name, ["N", lay.n_in]))
    graph += _field_bytes(12, value_info_body(output_name, ["N", lay.n_out]))

    model = _field_varint(1, ir_version)
    model += _field_string(2, "bitflipsim")
    model += _field_string(3, "0.1.0")
    model += _field_bytes(7, graph)
    model += _field_bytes(8, _field_string(1, "") + _field_varint(2, opset))
    return model
