"""ONNX serialisation: field numbers verified against a shipped model.

The field numbers in :mod:`edgeinfer.onnx_io` are taken from the ONNX
Standard's ``onnx.proto``. Because the ``onnx`` package is not installed in
this environment, they cannot be checked against generated bindings; they are
checked instead against ``onnxruntime/datasets/mul_1.onnx``, a model produced
by someone else and shipped inside the installed ``onnxruntime`` wheel. If a
field number were wrong, the parse below would read the wrong value.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgeinfer import pbwire as pw
from edgeinfer.graph import GraphError, ModelGraph, Node, TensorSpec
from edgeinfer.onnx_io import (
    DEFAULT_IR_VERSION,
    DEFAULT_OPSET,
    IR_ONLY_ATTRIBUTES,
    build_model,
    parse_model,
    shipped_model_path,
)


class TestWireFormat:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, b"\x00"),
            (1, b"\x01"),
            (127, b"\x7f"),
            (128, b"\x80\x01"),
            (300, b"\xac\x02"),
            (16384, b"\x80\x80\x01"),
        ],
    )
    def test_varint_known_answers(self, value: int, expected: bytes) -> None:
        """Hand-encoded against the protobuf base-128 varint specification.

        300 = 0b100101100. Low seven bits 0101100 = 0x2c with the
        continuation bit set gives 0xac; remaining bits 10 = 0x02. So
        300 -> b"\\xac\\x02".
        """
        assert pw.encode_varint(value) == expected

    def test_varint_round_trips(self) -> None:
        for value in (0, 1, 63, 64, 127, 128, 255, 65535, 2**31, 2**53):
            assert pw.read_varint(pw.encode_varint(value), 0) == (value, len(
                pw.encode_varint(value)
            ))

    def test_negative_varint_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            pw.encode_varint(-1)

    def test_field_zero_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="start at 1"):
            pw.field_varint(0, 1)

    def test_truncated_varint_is_detected(self) -> None:
        with pytest.raises(ValueError, match="truncated varint"):
            pw.read_varint(b"\x80", 0)

    def test_overlong_varint_is_detected(self) -> None:
        with pytest.raises(ValueError, match="exceeds 64 bits"):
            pw.read_varint(b"\x80" * 12, 0)

    def test_length_running_past_the_buffer_is_detected(self) -> None:
        payload = pw.field_bytes(1, b"abcdef")[:-3]
        with pytest.raises(ValueError, match="past end of buffer"):
            list(pw.iter_fields(payload))

    def test_unsupported_wire_type_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="wire type 3"):
            list(pw.iter_fields(pw.encode_varint((1 << 3) | 3)))

    def test_packed_varints_round_trip(self) -> None:
        buf = pw.packed_varints(5, [1, 2, 300])
        (field, wire_type, payload), = list(pw.iter_fields(buf))
        assert (field, wire_type) == (5, pw.WIRE_LEN)
        values = []
        offset = 0
        while offset < len(payload):
            value, offset = pw.read_varint(payload, offset)
            values.append(value)
        assert values == [1, 2, 300]

    def test_iter_fields_round_trips_a_mixed_message(self) -> None:
        buf = pw.field_varint(1, 9) + pw.field_string(2, "edgeinfer")
        fields = list(pw.iter_fields(buf))
        assert fields[0] == (1, pw.WIRE_VARINT, 9)
        assert fields[1][0] == 2
        assert bytes(fields[1][2]) == b"edgeinfer"


class TestFieldNumbersAgainstShippedModel:
    """Parse a model this package did not write, and check what comes out."""

    def test_shipped_model_is_present(self) -> None:
        assert shipped_model_path("mul_1.onnx").endswith("mul_1.onnx")

    def test_a_missing_shipped_model_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            shipped_model_path("not_a_real_model.onnx")

    def test_parsing_mul_1_reads_the_expected_structure(self) -> None:
        """``mul_1.onnx`` is a single ``Mul`` of a (3, 2) input with a (3, 2)
        initialiser, producing a (3, 2) output. Everything asserted here would
        come out wrong if a field number in ``onnx_io`` were wrong."""
        raw = open(shipped_model_path("mul_1.onnx"), "rb").read()
        graph, arrays = parse_model(raw)
        assert graph.op_types == ("Mul",)
        assert [spec.name for spec in graph.inputs] == ["X"]
        assert graph.inputs[0].shape == (3, 2)
        assert graph.inputs[0].dtype == "float32"
        assert graph.outputs == ("Y",)
        assert list(arrays) == ["W"]
        assert arrays["W"].shape == (3, 2)
        assert arrays["W"].dtype == np.float32

    def test_mul_1_initialiser_values_are_read_correctly(self) -> None:
        """The initialiser is stored as packed ``float_data``, not
        ``raw_data``; reading field 4 rather than field 9 is what is being
        checked. Its values are 1..6."""
        raw = open(shipped_model_path("mul_1.onnx"), "rb").read()
        _graph, arrays = parse_model(raw)
        assert np.allclose(arrays["W"].ravel(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_parsed_shipped_model_costs_match_a_hand_count(self) -> None:
        """``Mul`` over (3, 2): 6 flops; read 2 x 24 B, write 24 B = 72 B."""
        from edgeinfer.ops import node_cost

        raw = open(shipped_model_path("mul_1.onnx"), "rb").read()
        graph, _arrays = parse_model(raw)
        graph.infer_shapes()
        cost = node_cost(graph.nodes[0], graph)
        assert cost.flops == 6
        assert cost.bytes_total == 72

    def test_sigmoid_model_parses(self) -> None:
        raw = open(shipped_model_path("sigmoid.onnx"), "rb").read()
        graph, _ = parse_model(raw)
        assert "Sigmoid" in graph.op_types


class TestBuildModel:
    def test_build_then_parse_round_trips(self, hand_cnn) -> None:
        graph, arrays = parse_model(hand_cnn.model_bytes)
        assert graph.op_types == hand_cnn.graph.op_types
        for spec in hand_cnn.graph.initializers:
            assert np.array_equal(arrays[spec.name], hand_cnn.initializer_arrays[spec.name])

    def test_attributes_round_trip(self, hand_cnn) -> None:
        graph, _ = parse_model(hand_cnn.model_bytes)
        conv = next(n for n in graph.nodes if n.op_type == "Conv")
        assert conv.attrs["kernel_shape"] == [3, 3]
        assert conv.attrs["pads"] == [1, 1, 1, 1]
        assert conv.attrs["group"] == 1

    def test_ir_only_attributes_are_not_serialised(self, hand_cnn) -> None:
        """The ``shape`` attribute must not reach the wire (onnxruntime rejects
        an unrecognised attribute), and must come back on parse, recovered
        from the constant target-shape initialiser."""
        assert "shape" in IR_ONLY_ATTRIBUTES["Reshape"]
        wire_attribute_names: list[str] = []
        for field, _wt, payload in pw.iter_fields(hand_cnn.model_bytes):
            if field != 7:  # ModelProto.graph
                continue
            for gfield, _gwt, node_buf in pw.iter_fields(bytes(payload)):
                if gfield != 1:  # GraphProto.node
                    continue
                for nfield, _nwt, attr_buf in pw.iter_fields(bytes(node_buf)):
                    if nfield != 5:  # NodeProto.attribute
                        continue
                    for afield, _awt, value in pw.iter_fields(bytes(attr_buf)):
                        if afield == 1:  # AttributeProto.name
                            wire_attribute_names.append(bytes(value).decode())
        assert "shape" not in wire_attribute_names
        assert "kernel_shape" in wire_attribute_names

        graph, _ = parse_model(hand_cnn.model_bytes)
        reshape = next(n for n in graph.nodes if n.op_type == "Reshape")
        assert reshape.attrs["shape"] == [1, 18]

    def test_build_parse_build_is_byte_identical(self, hand_cnn) -> None:
        """Recovering the Reshape target shape on parse is what makes this
        round trip exact; without it the re-parsed graph cannot be
        shape-inferred at all."""
        graph, arrays = parse_model(hand_cnn.model_bytes)
        graph.infer_shapes()
        rebuilt = build_model(graph, arrays)
        assert rebuilt.model_bytes == hand_cnn.model_bytes

    def test_a_reshape_with_no_static_target_shape_is_refused(self) -> None:
        graph = ModelGraph(
            name="dyn",
            inputs=(TensorSpec("X", (2, 6)), TensorSpec("S", (2,), "int64")),
            outputs=("Y",),
            nodes=(Node("r", "Reshape", ("X", "S"), ("Y",)),),
        )
        with pytest.raises(GraphError, match="needs a static target shape"):
            graph.infer_shapes()

    def test_onnxruntime_accepts_the_default_ir_and_opset(self, hand_mlp) -> None:
        import onnxruntime as ort

        assert DEFAULT_IR_VERSION >= 3
        assert DEFAULT_OPSET >= 7
        session = ort.InferenceSession(
            hand_mlp.model_bytes, providers=["CPUExecutionProvider"]
        )
        assert [o.name for o in session.get_outputs()] == ["Y"]

    def test_a_missing_initialiser_array_is_rejected(self) -> None:
        graph = ModelGraph(
            name="g",
            inputs=(TensorSpec("X", (1, 4)),),
            outputs=("Y",),
            nodes=(Node("g0", "Gemm", ("X", "W"), ("Y",)),),
            initializers=(TensorSpec("W", (4, 3)),),
        )
        with pytest.raises(GraphError, match="no array supplied"):
            build_model(graph, {})

    def test_a_dtype_mismatch_is_rejected(self) -> None:
        graph = ModelGraph(
            name="g",
            inputs=(TensorSpec("X", (1, 4)),),
            outputs=("Y",),
            nodes=(Node("g0", "Gemm", ("X", "W"), ("Y",)),),
            initializers=(TensorSpec("W", (4, 3), "float32"),),
        )
        with pytest.raises(GraphError, match="dtype"):
            build_model(graph, {"W": np.zeros((4, 3), dtype=np.float64)})

    def test_a_non_int_attribute_is_rejected(self, rng) -> None:
        graph = ModelGraph(
            name="g",
            inputs=(TensorSpec("X", (1, 4)),),
            outputs=("Y",),
            nodes=(Node("r", "Relu", ("X",), ("Y",), {"alpha": 0.1}),),
        )
        with pytest.raises(GraphError, match="only int and list-of-int"):
            build_model(graph, {})

    def test_a_bool_attribute_is_rejected(self) -> None:
        graph = ModelGraph(
            name="g",
            inputs=(TensorSpec("X", (1, 4)),),
            outputs=("Y",),
            nodes=(Node("r", "Relu", ("X",), ("Y",), {"flag": True}),),
        )
        with pytest.raises(GraphError, match="not a bool"):
            build_model(graph, {})

    def test_input_feed_is_deterministic_in_the_seed(self, hand_cnn) -> None:
        a = hand_cnn.input_feed(5)
        b = hand_cnn.input_feed(5)
        c = hand_cnn.input_feed(6)
        assert np.array_equal(a["X"], b["X"])
        assert not np.array_equal(a["X"], c["X"])

    def test_input_feed_matches_the_declared_shapes_and_dtypes(self, hand_cnn) -> None:
        feed = hand_cnn.input_feed(0)
        for spec in hand_cnn.graph.inputs:
            assert feed[spec.name].shape == spec.shape
            assert str(feed[spec.name].dtype) == spec.dtype


class TestParserRefusals:
    def test_a_model_without_a_graph_is_rejected(self) -> None:
        with pytest.raises(GraphError, match="no graph field"):
            parse_model(pw.field_varint(1, 9))

    def test_a_symbolic_axis_is_rejected(self) -> None:
        """A dim_param instead of a dim_value: shapes must be made static."""
        dim = pw.field_message(1, pw.field_string(2, "batch"))
        tensor_type = pw.field_varint(1, 1) + pw.field_message(2, dim)
        value_info = pw.field_string(1, "X") + pw.field_message(
            2, pw.field_message(1, tensor_type)
        )
        node = (
            pw.field_string(1, "X")
            + pw.field_string(2, "Y")
            + pw.field_string(3, "r")
            + pw.field_string(4, "Relu")
        )
        graph = pw.field_message(1, node) + pw.field_message(11, value_info)
        model = pw.field_varint(1, 9) + pw.field_message(7, graph)
        with pytest.raises(GraphError, match="symbolic"):
            parse_model(model)

    def test_an_unsupported_element_type_is_rejected(self) -> None:
        dim = pw.field_message(1, pw.field_varint(1, 3))
        # elem_type 8 is STRING in the ONNX DataType enum.
        tensor_type = pw.field_varint(1, 8) + pw.field_message(2, dim)
        value_info = pw.field_string(1, "X") + pw.field_message(
            2, pw.field_message(1, tensor_type)
        )
        node = (
            pw.field_string(1, "X")
            + pw.field_string(2, "Y")
            + pw.field_string(3, "r")
            + pw.field_string(4, "Relu")
        )
        graph = pw.field_message(1, node) + pw.field_message(11, value_info)
        model = pw.field_varint(1, 9) + pw.field_message(7, graph)
        with pytest.raises(GraphError, match="unsupported ONNX elem_type 8"):
            parse_model(model)
