"""Graph IR: shapes, validation, liveness inputs, ONNX conv output formula."""

from __future__ import annotations

import pytest

from edgeinfer.graph import (
    DTYPE_BYTES,
    GraphError,
    ModelGraph,
    Node,
    TensorSpec,
    conv_output_spatial,
)


class TestTensorSpec:
    def test_elements_and_bytes(self) -> None:
        # (2, 3, 4) float32: 24 elements, 24 * 4 = 96 B.
        spec = TensorSpec("t", (2, 3, 4), "float32")
        assert spec.elements == 24
        assert spec.nbytes == 96

    def test_every_declared_dtype_has_a_size(self) -> None:
        for dtype, size in DTYPE_BYTES.items():
            assert TensorSpec("t", (10,), dtype).nbytes == 10 * size

    @pytest.mark.parametrize(
        "shape", [(0, 3), (-1, 2), (2, 0)], ids=["zero-first", "negative", "zero-last"]
    )
    def test_non_positive_dimension_is_rejected(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(GraphError, match="positive int"):
            TensorSpec("t", shape)

    def test_rank_zero_is_rejected(self) -> None:
        with pytest.raises(GraphError, match="rank-0"):
            TensorSpec("t", ())

    def test_unknown_dtype_names_the_supported_set(self) -> None:
        with pytest.raises(GraphError, match="float32"):
            TensorSpec("t", (2,), "bfloat16")

    def test_empty_name_is_rejected(self) -> None:
        with pytest.raises(GraphError, match="non-empty"):
            TensorSpec("", (2,))


class TestConvOutputSpatial:
    def test_known_answer_no_padding(self) -> None:
        # 8x8 input, 3x3 kernel, stride 1, no pad, no dilation:
        # floor((8 + 0 + 0 - 1*(3-1) - 1) / 1) + 1 = floor(5/1) + 1 = 6.
        assert conv_output_spatial((8, 8), (3, 3), (1, 1), (0, 0, 0, 0), (1, 1)) == (6, 6)

    def test_known_answer_same_padding(self) -> None:
        # pad 1 each side restores the input size for a 3x3 stride-1 kernel:
        # floor((8 + 1 + 1 - 2 - 1) / 1) + 1 = 8.
        assert conv_output_spatial((8, 8), (3, 3), (1, 1), (1, 1, 1, 1), (1, 1)) == (8, 8)

    def test_known_answer_stride_two(self) -> None:
        # 2x2 kernel, stride 2, no pad: floor((8 - 1 - 1)/2) + 1 = floor(6/2)+1 = 4.
        assert conv_output_spatial((8, 8), (2, 2), (2, 2), (0, 0, 0, 0), (1, 1)) == (4, 4)

    def test_known_answer_dilation_two(self) -> None:
        # effective extent = 2*(3-1)+1 = 5; floor((9 - 5)/1) + 1 = 5.
        assert conv_output_spatial((9,), (3,), (1,), (0, 0), (2,)) == (5,)

    def test_kernel_larger_than_padded_input_is_an_error(self) -> None:
        with pytest.raises(GraphError, match="exceeds padded input"):
            conv_output_spatial((2, 2), (5, 5), (1, 1), (0, 0, 0, 0), (1, 1))

    def test_pads_length_must_be_twice_the_rank(self) -> None:
        with pytest.raises(GraphError, match="pads must have length 4"):
            conv_output_spatial((8, 8), (3, 3), (1, 1), (0, 0), (1, 1))

    def test_inconsistent_attribute_lengths_are_rejected(self) -> None:
        with pytest.raises(GraphError, match="must each have length 2"):
            conv_output_spatial((8, 8), (3,), (1, 1), (0, 0, 0, 0), (1, 1))

    @pytest.mark.parametrize("bad", ["stride", "dilation", "kernel"])
    def test_non_positive_attributes_are_rejected(self, bad: str) -> None:
        kernel, strides, dilations = (3,), (1,), (1,)
        if bad == "stride":
            strides = (0,)
        elif bad == "dilation":
            dilations = (0,)
        else:
            kernel = (0,)
        with pytest.raises(GraphError, match="strictly positive"):
            conv_output_spatial((8,), kernel, strides, (0, 0), dilations)


def _two_node_graph() -> ModelGraph:
    return ModelGraph(
        name="g",
        inputs=(TensorSpec("X", (1, 4)),),
        outputs=("Y",),
        nodes=(
            Node("fc", "Gemm", ("X", "W", "B"), ("H",)),
            Node("act", "Relu", ("H",), ("Y",)),
        ),
        initializers=(TensorSpec("W", (4, 3)), TensorSpec("B", (3,))),
    )


class TestModelGraphValidation:
    def test_shape_inference_fills_every_output(self) -> None:
        graph = _two_node_graph().infer_shapes()
        assert graph.spec("H").shape == (1, 3)
        assert graph.spec("Y").shape == (1, 3)

    def test_weight_bytes_is_the_initialiser_total(self) -> None:
        # W (4, 3) float32 = 48 B, B (3,) float32 = 12 B, total 60 B.
        assert _two_node_graph().weight_bytes == 60

    def test_out_of_order_nodes_are_rejected(self) -> None:
        with pytest.raises(GraphError, match="topological order"):
            ModelGraph(
                name="g",
                inputs=(TensorSpec("X", (1, 4)),),
                outputs=("Y",),
                nodes=(
                    Node("act", "Relu", ("H",), ("Y",)),
                    Node("fc", "Gemm", ("X", "W", "B"), ("H",)),
                ),
                initializers=(TensorSpec("W", (4, 3)), TensorSpec("B", (3,))),
            )

    def test_unproduced_output_is_rejected(self) -> None:
        with pytest.raises(GraphError, match="never produced"):
            ModelGraph(
                name="g",
                inputs=(TensorSpec("X", (1, 4)),),
                outputs=("Z",),
                nodes=(Node("r", "Relu", ("X",), ("Y",)),),
            )

    def test_duplicate_node_names_are_rejected(self) -> None:
        with pytest.raises(GraphError, match="duplicate node name"):
            ModelGraph(
                name="g",
                inputs=(TensorSpec("X", (1, 4)),),
                outputs=("Y",),
                nodes=(
                    Node("r", "Relu", ("X",), ("A",)),
                    Node("r", "Relu", ("A",), ("Y",)),
                ),
            )

    def test_empty_graph_is_rejected(self) -> None:
        with pytest.raises(GraphError, match="no nodes"):
            ModelGraph("g", (TensorSpec("X", (1,)),), ("X",), ())

    def test_graph_without_inputs_is_rejected(self) -> None:
        with pytest.raises(GraphError, match="no inputs"):
            ModelGraph("g", (), ("Y",), (Node("r", "Relu", ("X",), ("Y",)),))

    def test_node_without_output_is_rejected(self) -> None:
        with pytest.raises(GraphError, match="produces no output"):
            Node("r", "Relu", ("X",), ())

    def test_node_without_op_type_is_rejected(self) -> None:
        with pytest.raises(GraphError, match="empty op_type"):
            Node("r", "", ("X",), ("Y",))

    def test_spec_for_unknown_tensor_says_to_infer_shapes(self) -> None:
        graph = _two_node_graph()
        with pytest.raises(GraphError, match="infer_shapes"):
            graph.spec("H")

    def test_consumers_counts_graph_outputs_once(self) -> None:
        graph = _two_node_graph().infer_shapes()
        counts = graph.consumers()
        assert counts["X"] == 1  # consumed by fc
        assert counts["H"] == 1  # consumed by act
        assert counts["Y"] == 1  # graph output
        assert counts["W"] == 1
