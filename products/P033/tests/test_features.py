"""Graph features: the quantities, their ordering and their invariants."""

from __future__ import annotations

import numpy as np
import pytest

from edgeinfer.features import FEATURE_NAMES, OP_GROUPS, graph_features, population_features
from edgeinfer.graph import ModelGraph, Node, TensorSpec
from edgeinfer.ops import SUPPORTED_OPS, UnsupportedOperatorError


class TestFeatureVector:
    def test_length_matches_the_declared_names(self, hand_mlp) -> None:
        assert graph_features(hand_mlp.graph).shape == (len(FEATURE_NAMES),)

    def test_known_answers_for_the_hand_counted_mlp(self, hand_mlp) -> None:
        """From the hand count in ``test_ops.py``: 99 flops, 328 B traffic,
        42 MACs, 3 nodes, 2 distinct op types, 204 B weights, 48 B peak
        activations."""
        values = dict(zip(FEATURE_NAMES, graph_features(hand_mlp.graph), strict=True))
        assert values["log10_flops"] == pytest.approx(np.log10(100))
        assert values["log10_bytes"] == pytest.approx(np.log10(329))
        assert values["log10_macs"] == pytest.approx(np.log10(43))
        assert values["n_nodes"] == 3
        assert values["n_distinct_ops"] == 2
        assert values["log10_weight_bytes"] == pytest.approx(np.log10(205))
        assert values["log10_peak_act_bytes"] == pytest.approx(np.log10(49))

    def test_flop_group_fractions_sum_to_one(self, hand_cnn) -> None:
        values = dict(zip(FEATURE_NAMES, graph_features(hand_cnn.graph), strict=True))
        total = sum(values[f"frac_{group}"] for group in OP_GROUPS)
        assert total == pytest.approx(1.0)

    def test_matmul_group_holds_all_the_flops_of_a_pure_gemm(self, rng) -> None:
        from edgeinfer.dataset import random_mlp

        model = random_mlp(rng, "pure", n_in=8, widths=(), n_out=4)
        values = dict(zip(FEATURE_NAMES, graph_features(model.graph), strict=True))
        assert values["frac_matmul"] == pytest.approx(1.0)

    def test_a_zero_flop_graph_gives_finite_features(self) -> None:
        """All-reshape graph: log10(0 + 1) = 0, not -inf."""
        graph = ModelGraph(
            name="shapeonly",
            inputs=(TensorSpec("X", (2, 6)),),
            outputs=("Y",),
            nodes=(Node("r", "Reshape", ("X",), ("Y",), {"shape": [3, 4]}),),
        )
        values = graph_features(graph)
        assert np.all(np.isfinite(values))
        assert values[FEATURE_NAMES.index("log10_flops")] == 0.0

    def test_every_supported_operator_belongs_to_a_group(self) -> None:
        grouped = {op for ops in OP_GROUPS.values() for op in ops}
        assert SUPPORTED_OPS <= grouped

    def test_an_unsupported_operator_is_refused_not_featurised_as_zero(self) -> None:
        with pytest.raises(UnsupportedOperatorError):
            graph_features(
                ModelGraph(
                    name="u",
                    inputs=(TensorSpec("X", (2,)),),
                    outputs=("Y",),
                    nodes=(Node("n", "Scan", ("X",), ("Y",)),),
                )
            )

    def test_features_are_monotone_in_graph_size(self, rng) -> None:
        from edgeinfer.dataset import random_mlp

        small = graph_features(random_mlp(rng, "s", n_in=8, widths=(8,), n_out=4).graph)
        large = graph_features(
            random_mlp(rng, "l", n_in=256, widths=(256,), n_out=64).graph
        )
        i = FEATURE_NAMES.index("log10_flops")
        assert large[i] > small[i]


class TestPopulationFeatures:
    def test_matrix_shape(self, hand_mlp, hand_cnn) -> None:
        matrix = population_features([hand_mlp.graph, hand_cnn.graph])
        assert matrix.shape == (2, len(FEATURE_NAMES))

    def test_rows_match_the_single_graph_call(self, hand_mlp, hand_cnn) -> None:
        matrix = population_features([hand_mlp.graph, hand_cnn.graph])
        assert np.allclose(matrix[0], graph_features(hand_mlp.graph))
        assert np.allclose(matrix[1], graph_features(hand_cnn.graph))

    def test_empty_list_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one graph"):
            population_features([])
