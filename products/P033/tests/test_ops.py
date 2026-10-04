"""Operation counts against hand counts, and the unsupported-operator refusal.

The two hand-counted networks required by the batch specification are
``edgeinfer.dataset.hand_counted_mlp`` and
``edgeinfer.dataset.hand_counted_cnn``. Their full hand counts are written out
in the comments of :class:`TestHandCountedMlp` and :class:`TestHandCountedCnn`
below, so that the analytic model is checked against arithmetic a reader can
redo on paper rather than against another implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgeinfer.graph import GraphError, ModelGraph, Node, TensorSpec
from edgeinfer.onnx_io import build_model
from edgeinfer.ops import (
    SOFTMAX_FLOPS_PER_ELEMENT,
    SUPPORTED_OPS,
    TRANSCENDENTAL_FLOPS_PER_ELEMENT,
    UnsupportedOperatorError,
    node_cost,
)


def _totals(model) -> tuple[int, int, int]:
    graph = model.graph
    graph.infer_shapes()
    macs = flops = traffic = 0
    for node in graph.nodes:
        cost = node_cost(node, graph)
        macs += cost.macs
        flops += cost.flops
        traffic += cost.bytes_total
    return macs, flops, traffic


class TestHandCountedMlp:
    """``X (1, 4) -> Gemm(4, 6)+bias -> Relu -> Gemm(6, 3)+bias -> Y``, float32.

    HAND COUNT, written out in full.

    Tensor sizes (float32, 4 B per element):
        X  (1, 4)  =  4 elem =  16 B
        W0 (4, 6)  = 24 elem =  96 B
        B0 (6,)    =  6 elem =  24 B
        G0 (1, 6)  =  6 elem =  24 B      (Gemm 0 output)
        H0 (1, 6)  =  6 elem =  24 B      (Relu output)
        W1 (6, 3)  = 18 elem =  72 B
        B1 (3,)    =  3 elem =  12 B
        Y  (1, 3)  =  3 elem =  12 B

    MACs (Golub & Van Loan 2013 section 1.1.11: an (M, K) x (K, N) product is
    M*K*N multiply-accumulates):
        gemm0: M=1, K=4, N=6  ->  1*4*6  = 24
        gemm1: M=1, K=6, N=3  ->  1*6*3  = 18
        Relu, being elementwise, performs no multiply-accumulate -> 0
        TOTAL MACs = 24 + 18 = 42

    FLOPs (2 per MAC, plus 1 per output element for a bias add, plus 1 per
    element for Relu):
        gemm0: 2*24 = 48, bias over 6 outputs = 6   -> 54
        act0 : Relu over 6 elements                 ->  6
        gemm1: 2*18 = 36, bias over 3 outputs = 3   -> 39
        TOTAL FLOPs = 54 + 6 + 39 = 99

    Compulsory traffic (each input read once, each output written once):
        gemm0: read X 16 + W0 96 + B0 24 = 136; write G0 24   -> 160 B
        act0 : read G0 24;                       write H0 24  ->  48 B
        gemm1: read H0 24 + W1 72 + B1 12 = 108; write Y 12   -> 120 B
        TOTAL traffic = 160 + 48 + 120 = 328 B

    Weight bytes resident = W0 96 + B0 24 + W1 72 + B1 12 = 204 B

    Peak live activations, liveness analysis (Aho et al. 2006 section 8.4),
    outputs allocated before inputs are released:
        entry : X 16                              = 16 B
        gemm0 : X 16 + G0 24                      = 40 B   <- X dies here
        act0  : G0 24 + H0 24                     = 48 B   <- peak; G0 dies
        gemm1 : H0 24 + Y 12                      = 36 B
        PEAK activations = 48 B
    """

    EXPECTED_MACS = 42
    EXPECTED_FLOPS = 99
    EXPECTED_TRAFFIC = 328
    EXPECTED_WEIGHT_BYTES = 204
    EXPECTED_PEAK_ACTIVATIONS = 48

    def test_shape_is_the_one_the_hand_count_assumes(self, hand_mlp) -> None:
        graph = hand_mlp.graph
        graph.infer_shapes()
        assert graph.inputs[0].shape == (1, 4)
        assert graph.op_types == ("Gemm", "Relu", "Gemm")
        assert graph.spec("Y").shape == (1, 3)

    def test_macs_match_the_hand_count(self, hand_mlp) -> None:
        macs, _, _ = _totals(hand_mlp)
        assert macs == self.EXPECTED_MACS

    def test_flops_match_the_hand_count(self, hand_mlp) -> None:
        _, flops, _ = _totals(hand_mlp)
        assert flops == self.EXPECTED_FLOPS

    def test_traffic_matches_the_hand_count(self, hand_mlp) -> None:
        _, _, traffic = _totals(hand_mlp)
        assert traffic == self.EXPECTED_TRAFFIC

    def test_weight_bytes_match_the_hand_count(self, hand_mlp) -> None:
        assert hand_mlp.graph.weight_bytes == self.EXPECTED_WEIGHT_BYTES

    def test_peak_activations_match_the_hand_count(self, hand_mlp) -> None:
        from edgeinfer.analytic import peak_activation_bytes

        peak, _ = peak_activation_bytes(hand_mlp.graph.infer_shapes())
        assert peak == self.EXPECTED_PEAK_ACTIVATIONS


class TestHandCountedCnn:
    """``X (1,1,6,6) -> Conv(1->2, 3x3, pad 1) -> Relu -> MaxPool(2x2, s2)
    -> Reshape(1,18) -> Gemm(18,2)+bias -> Y``, float32.

    HAND COUNT, written out in full.

    Shapes (ONNX Conv formula, floor((6 + 1 + 1 - 3)/1) + 1 = 6):
        X  (1, 1, 6, 6) =  36 elem = 144 B
        CW (2, 1, 3, 3) =  18 elem =  72 B
        C0 (1, 2, 6, 6) =  72 elem = 288 B
        R0 (1, 2, 6, 6) =  72 elem = 288 B
        P0 (1, 2, 3, 3) =  18 elem =  72 B     (MaxPool 2x2 stride 2)
        RS (2,) int64   =   2 elem =  16 B     (Reshape target-shape input)
        F  (1, 18)      =  18 elem =  72 B
        FW (18, 2)      =  36 elem = 144 B
        FB (2,)         =   2 elem =   8 B
        Y  (1, 2)       =   2 elem =   8 B

    MACs (Sze et al. 2017 section II-A:
    N * M * prod(S_out) * (C/group) * prod(K)):
        conv0: 1 * 2 * (6*6) * (1/1) * (3*3) = 1*2*36*1*9 = 648
        fc   : 1 * 18 * 2                                 =  36
        TOTAL MACs = 648 + 36 = 684

    FLOPs:
        conv0: 2*648 = 1296, no bias input                 -> 1296
        crelu0: Relu over 72 elements                      ->   72
        pool0: MaxPool 2x2 -> 18 outputs * 4 comparisons   ->   72
        flat : Reshape, metadata only                      ->    0
        fc   : 2*36 = 72, bias over 2 outputs = 2          ->   74
        TOTAL FLOPs = 1296 + 72 + 72 + 0 + 74 = 1514

    Compulsory traffic:
        conv0 : read X 144 + CW 72 = 216; write C0 288     -> 504 B
        crelu0: read C0 288;              write R0 288     -> 576 B
        pool0 : read R0 288;              write P0 72      -> 360 B
        flat  : read P0 72 + RS 16 = 88;  write F 72       -> 160 B
        fc    : read F 72 + FW 144 + FB 8 = 224; write Y 8 -> 232 B
        TOTAL traffic = 504 + 576 + 360 + 160 + 232 = 1832 B

    Weight bytes resident = CW 72 + RS 16 + FW 144 + FB 8 = 240 B

    Peak live activations:
        entry  : X 144                      = 144 B
        conv0  : X 144 + C0 288             = 432 B   <- X dies
        crelu0 : C0 288 + R0 288            = 576 B   <- peak; C0 dies
        pool0  : R0 288 + P0 72             = 360 B   <- R0 dies
        flat   : P0 72 + F 72               = 144 B   <- P0 dies
        fc     : F 72 + Y 8                 =  80 B
        PEAK activations = 576 B
    """

    EXPECTED_MACS = 684
    EXPECTED_FLOPS = 1514
    EXPECTED_TRAFFIC = 1832
    EXPECTED_WEIGHT_BYTES = 240
    EXPECTED_PEAK_ACTIVATIONS = 576

    def test_shapes_are_the_ones_the_hand_count_assumes(self, hand_cnn) -> None:
        graph = hand_cnn.graph
        graph.infer_shapes()
        assert graph.inputs[0].shape == (1, 1, 6, 6)
        assert graph.spec("C0").shape == (1, 2, 6, 6)
        assert graph.spec("P0").shape == (1, 2, 3, 3)
        assert graph.spec("Y").shape == (1, 2)

    def test_macs_match_the_hand_count(self, hand_cnn) -> None:
        macs, _, _ = _totals(hand_cnn)
        assert macs == self.EXPECTED_MACS

    def test_flops_match_the_hand_count(self, hand_cnn) -> None:
        _, flops, _ = _totals(hand_cnn)
        assert flops == self.EXPECTED_FLOPS

    def test_traffic_matches_the_hand_count(self, hand_cnn) -> None:
        _, _, traffic = _totals(hand_cnn)
        assert traffic == self.EXPECTED_TRAFFIC

    def test_weight_bytes_match_the_hand_count(self, hand_cnn) -> None:
        assert hand_cnn.graph.weight_bytes == self.EXPECTED_WEIGHT_BYTES

    def test_peak_activations_match_the_hand_count(self, hand_cnn) -> None:
        from edgeinfer.analytic import peak_activation_bytes

        peak, _ = peak_activation_bytes(hand_cnn.graph.infer_shapes())
        assert peak == self.EXPECTED_PEAK_ACTIVATIONS

    def test_conv_node_mac_count_is_the_product_in_sze_et_al(self, hand_cnn) -> None:
        graph = hand_cnn.graph.infer_shapes()
        conv = next(n for n in graph.nodes if n.op_type == "Conv")
        assert node_cost(conv, graph).macs == 648


class TestPerOperatorCosts:
    def _single(self, op: str, in_shape: tuple[int, ...], attrs=None, dtype="float32"):
        graph = ModelGraph(
            name="single",
            inputs=(TensorSpec("X", in_shape, dtype),),
            outputs=("Y",),
            nodes=(Node("n", op, ("X",), ("Y",), dict(attrs or {})),),
        ).infer_shapes()
        return node_cost(graph.nodes[0], graph), graph

    def test_relu_is_one_flop_per_element(self) -> None:
        cost, _ = self._single("Relu", (3, 4))
        assert cost.flops == 12
        assert cost.macs == 0

    def test_sigmoid_is_charged_the_transcendental_rate(self) -> None:
        cost, _ = self._single("Sigmoid", (3, 4))
        assert cost.flops == TRANSCENDENTAL_FLOPS_PER_ELEMENT * 12

    def test_softmax_is_charged_the_declared_rate(self) -> None:
        cost, _ = self._single("Softmax", (2, 5))
        assert cost.flops == SOFTMAX_FLOPS_PER_ELEMENT * 10

    def test_batchnorm_is_two_flops_per_element(self) -> None:
        cost, _ = self._single("BatchNormalization", (1, 3, 4, 4))
        assert cost.flops == 2 * 48

    def test_maxpool_charges_one_comparison_per_kernel_element(self) -> None:
        # (1, 1, 4, 4) with 2x2 stride 2 -> (1, 1, 2, 2) = 4 outputs, 4 each.
        cost, graph = self._single(
            "MaxPool",
            (1, 1, 4, 4),
            {"kernel_shape": [2, 2], "strides": [2, 2], "pads": [0, 0, 0, 0]},
        )
        assert graph.spec("Y").shape == (1, 1, 2, 2)
        assert cost.flops == 16

    def test_global_average_pool_sums_inputs_and_divides_outputs(self) -> None:
        # 1*3*4*4 = 48 adds, output (1, 3, 1, 1) = 3 divides.
        cost, graph = self._single("GlobalAveragePool", (1, 3, 4, 4))
        assert graph.spec("Y").shape == (1, 3, 1, 1)
        assert cost.flops == 48 + 3

    def test_reshape_is_free_in_flops(self) -> None:
        cost, graph = self._single("Reshape", (2, 6), {"shape": [3, 4]})
        assert graph.spec("Y").shape == (3, 4)
        assert cost.flops == 0

    def test_reshape_that_changes_element_count_is_rejected(self) -> None:
        with pytest.raises(GraphError, match="changes element count"):
            self._single("Reshape", (2, 6), {"shape": [5, 5]})

    def test_arithmetic_intensity_is_flops_over_traffic(self) -> None:
        cost, _ = self._single("Relu", (10,))
        # 10 flops; read 40 B, write 40 B -> 10 / 80 = 0.125 FLOP/B
        assert cost.arithmetic_intensity == pytest.approx(0.125)

    def test_matmul_counts_the_full_product(self) -> None:
        graph = ModelGraph(
            name="mm",
            inputs=(TensorSpec("X", (2, 3)),),
            outputs=("Y",),
            nodes=(Node("mm", "MatMul", ("X", "W"), ("Y",)),),
            initializers=(TensorSpec("W", (3, 5)),),
        ).infer_shapes()
        # (2,3) x (3,5) -> (2,5); MACs = 2*5*3 = 30; flops = 60.
        assert graph.spec("Y").shape == (2, 5)
        cost = node_cost(graph.nodes[0], graph)
        assert (cost.macs, cost.flops) == (30, 60)

    def test_grouped_conv_divides_input_channels(self) -> None:
        graph = ModelGraph(
            name="gc",
            inputs=(TensorSpec("X", (1, 4, 4, 4)),),
            outputs=("Y",),
            nodes=(
                Node(
                    "c",
                    "Conv",
                    ("X", "W"),
                    ("Y",),
                    {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "group": 2},
                ),
            ),
            initializers=(TensorSpec("W", (4, 2, 3, 3)),),
        ).infer_shapes()
        # out (1, 4, 4, 4); MACs = 1*4*16*(4/2)*9 = 1152.
        assert graph.spec("Y").shape == (1, 4, 4, 4)
        assert node_cost(graph.nodes[0], graph).macs == 1152

    def test_group_that_does_not_divide_channels_is_rejected(self) -> None:
        with pytest.raises(GraphError, match="does not divide"):
            ModelGraph(
                name="gc",
                inputs=(TensorSpec("X", (1, 3, 4, 4)),),
                outputs=("Y",),
                nodes=(
                    Node(
                        "c",
                        "Conv",
                        ("X", "W"),
                        ("Y",),
                        {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "group": 2},
                    ),
                ),
                initializers=(TensorSpec("W", (4, 2, 3, 3)),),
            ).infer_shapes()

    def test_binary_op_broadcasts(self) -> None:
        graph = ModelGraph(
            name="b",
            inputs=(TensorSpec("X", (3, 4)), TensorSpec("Z", (1, 4))),
            outputs=("Y",),
            nodes=(Node("a", "Add", ("X", "Z"), ("Y",)),),
        ).infer_shapes()
        assert graph.spec("Y").shape == (3, 4)
        assert node_cost(graph.nodes[0], graph).flops == 12

    def test_incompatible_broadcast_is_rejected(self) -> None:
        with pytest.raises(GraphError, match="cannot broadcast"):
            ModelGraph(
                name="b",
                inputs=(TensorSpec("X", (3, 4)), TensorSpec("Z", (2, 4))),
                outputs=("Y",),
                nodes=(Node("a", "Add", ("X", "Z"), ("Y",)),),
            ).infer_shapes()

    def test_concat_sums_the_axis(self) -> None:
        graph = ModelGraph(
            name="c",
            inputs=(TensorSpec("X", (2, 3)), TensorSpec("Z", (2, 5))),
            outputs=("Y",),
            nodes=(Node("cc", "Concat", ("X", "Z"), ("Y",), {"axis": 1}),),
        ).infer_shapes()
        assert graph.spec("Y").shape == (2, 8)

    def test_gemm_transb_is_honoured(self) -> None:
        graph = ModelGraph(
            name="t",
            inputs=(TensorSpec("X", (1, 4)),),
            outputs=("Y",),
            nodes=(Node("g", "Gemm", ("X", "W"), ("Y",), {"transB": 1}),),
            initializers=(TensorSpec("W", (3, 4)),),
        ).infer_shapes()
        assert graph.spec("Y").shape == (1, 3)
        assert node_cost(graph.nodes[0], graph).macs == 12

    def test_gemm_with_mismatched_inner_dimension_is_rejected(self) -> None:
        with pytest.raises(GraphError, match="inner dimensions disagree"):
            ModelGraph(
                name="t",
                inputs=(TensorSpec("X", (1, 4)),),
                outputs=("Y",),
                nodes=(Node("g", "Gemm", ("X", "W"), ("Y",)),),
                initializers=(TensorSpec("W", (5, 3)),),
            ).infer_shapes()


class TestUnsupportedOperator:
    def test_cost_of_an_unknown_operator_is_refused_not_zero(self) -> None:
        graph = ModelGraph.__new__(ModelGraph)
        object.__setattr__(graph, "name", "u")
        with pytest.raises(UnsupportedOperatorError) as exc:
            node_cost(Node("n", "GridSample", ("X",), ("Y",)), graph)
        assert exc.value.op_type == "GridSample"
        assert "will not charge an unknown operator zero cost" in str(exc.value)

    def test_the_error_lists_the_supported_operators(self) -> None:
        graph = ModelGraph.__new__(ModelGraph)
        with pytest.raises(UnsupportedOperatorError, match="Conv"):
            node_cost(Node("n", "Einsum", ("X",), ("Y",)), graph)

    def test_unsupported_operator_is_a_graph_error_subclass(self) -> None:
        assert issubclass(UnsupportedOperatorError, GraphError)

    def test_shape_inference_also_refuses(self) -> None:
        with pytest.raises(UnsupportedOperatorError, match="Loop"):
            ModelGraph(
                name="u",
                inputs=(TensorSpec("X", (2,)),),
                outputs=("Y",),
                nodes=(Node("n", "Loop", ("X",), ("Y",)),),
            ).infer_shapes()

    def test_supported_ops_is_non_empty_and_contains_the_core_set(self) -> None:
        for op in ("Conv", "Gemm", "Relu", "MaxPool", "Reshape"):
            assert op in SUPPORTED_OPS


class TestAgainstOnnxRuntimeNumerics:
    """The hand-counted graphs must also be graphs onnxruntime will run.

    A cost model for a graph the runtime rejects is a cost model for nothing,
    so both hand-counted networks are executed here and their outputs checked
    against a NumPy recomputation.
    """

    def test_hand_mlp_runs_and_matches_numpy(self, hand_mlp) -> None:
        import onnxruntime as ort

        session = ort.InferenceSession(
            hand_mlp.model_bytes, providers=["CPUExecutionProvider"]
        )
        feed = hand_mlp.input_feed(7)
        got = session.run(None, feed)[0]
        arrays = hand_mlp.initializer_arrays
        hidden = np.maximum(feed["X"] @ arrays["W0"] + arrays["B0"], 0.0)
        expected = hidden @ arrays["W1"] + arrays["B1"]
        assert np.allclose(got, expected, atol=1e-5)

    def test_hand_cnn_runs_and_produces_the_inferred_shape(self, hand_cnn) -> None:
        import onnxruntime as ort

        session = ort.InferenceSession(
            hand_cnn.model_bytes, providers=["CPUExecutionProvider"]
        )
        got = session.run(None, hand_cnn.input_feed(7))[0]
        assert got.shape == hand_cnn.graph.spec("Y").shape

    def test_build_model_rejects_a_mismatched_initialiser(self, rng) -> None:
        graph = ModelGraph(
            name="g",
            inputs=(TensorSpec("X", (1, 4)),),
            outputs=("Y",),
            nodes=(Node("g0", "Gemm", ("X", "W"), ("Y",)),),
            initializers=(TensorSpec("W", (4, 3)),),
        )
        with pytest.raises(GraphError, match="does not match declared shape"):
            build_model(graph, {"W": rng.standard_normal((4, 5)).astype(np.float32)})
