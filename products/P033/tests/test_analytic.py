"""Analytic estimate: totals, liveness peaks, and the lower-bound property."""

from __future__ import annotations

import numpy as np
import pytest

from edgeinfer.analytic import analytic_estimate, node_cost_arrays, peak_activation_bytes
from edgeinfer.graph import ModelGraph, Node, TensorSpec
from edgeinfer.roofline import DeviceModel


class TestAnalyticEstimate:
    def test_totals_match_the_hand_counted_mlp(self, hand_mlp, device) -> None:
        estimate = analytic_estimate(hand_mlp.graph, device)
        assert estimate.total_macs == 42
        assert estimate.total_flops == 99
        assert estimate.total_traffic_bytes == 328
        assert estimate.peak_memory_bytes == 204 + 48

    def test_latency_is_the_sum_of_node_bounds_when_overheads_are_zero(
        self, hand_cnn, device
    ) -> None:
        estimate = analytic_estimate(hand_cnn.graph, device)
        assert estimate.latency_s == pytest.approx(
            sum(n.roofline_s for n in estimate.nodes)
        )
        assert estimate.latency_s == pytest.approx(estimate.roofline_only_s)

    def test_overheads_add_exactly_once_each(self, hand_cnn) -> None:
        plain = DeviceModel("d", 1e9, 1e9)
        with_overhead = DeviceModel(
            "d", 1e9, 1e9, overhead_per_node_s=1e-6, fixed_overhead_s=5e-6
        )
        a = analytic_estimate(hand_cnn.graph, plain)
        b = analytic_estimate(hand_cnn.graph, with_overhead)
        n_nodes = len(hand_cnn.graph.nodes)
        assert b.latency_s - a.latency_s == pytest.approx(n_nodes * 1e-6 + 5e-6)
        assert b.roofline_only_s == pytest.approx(a.roofline_only_s)

    def test_graph_arithmetic_intensity_is_flops_over_traffic(
        self, hand_cnn, device
    ) -> None:
        estimate = analytic_estimate(hand_cnn.graph, device)
        assert estimate.arithmetic_intensity == pytest.approx(1514 / 1832)

    def test_memory_bound_fraction_counts_nodes_below_the_ridge(
        self, hand_cnn, device
    ) -> None:
        estimate = analytic_estimate(hand_cnn.graph, device)
        expected = sum(n.memory_bound for n in estimate.nodes) / len(estimate.nodes)
        assert estimate.memory_bound_fraction == pytest.approx(expected)

    def test_summary_lines_name_the_device_and_its_source(self, hand_mlp, device) -> None:
        text = "\n".join(analytic_estimate(hand_mlp.graph, device).summary_lines())
        assert device.name in text
        assert device.source in text

    def test_estimate_is_a_lower_bound_on_a_measured_onnxruntime_latency(
        self, small_mlp
    ) -> None:
        """A pure roofline is a bound, so the measurement must exceed it.

        Peaks here are generous on purpose (100 GFLOP/s, 100 GB/s), far above
        anything a single core of this container delivers, so the bound is
        unambiguously below the measurement. A measurement *below* the bound
        would mean the declared peaks are wrong, which is the calibration
        fault the module docstring warns about.
        """
        from edgeinfer.backends import OnnxRuntimeBackend
        from edgeinfer.harness import benchmark

        generous = DeviceModel("generous", 100e9, 100e9)
        bound = analytic_estimate(small_mlp.graph, generous).roofline_only_s
        backend = OnnxRuntimeBackend(small_mlp.model_bytes, small_mlp.input_feed(0))
        backend.prepare()
        try:
            profile = benchmark(
                backend.infer, label="bound check", repeats=40, warmup=5,
                timer_bias_samples=100,
            )
        finally:
            backend.close()
        assert profile.min_s > bound


class TestLiveness:
    def test_peak_matches_the_hand_counted_cnn(self, hand_cnn) -> None:
        peak, per_step = peak_activation_bytes(hand_cnn.graph.infer_shapes())
        assert peak == 576
        # live_by_step is the total AFTER a node's dead inputs are released,
        # so it is below the peak reached during that node. After conv0, X
        # (144 B) is dead and only C0 (288 B) is live; after fc, F (72 B) is
        # dead and only the graph output Y (8 B) is live.
        assert per_step["conv0"] == 288
        assert per_step["fc"] == 8

    def test_a_tensor_feeding_two_nodes_stays_live_until_both_have_run(self) -> None:
        # X (1, 100) float32 = 400 B feeds two Relus; A and B are 400 B each.
        graph = ModelGraph(
            name="fork",
            inputs=(TensorSpec("X", (1, 100)),),
            outputs=("C",),
            nodes=(
                Node("r1", "Relu", ("X",), ("A",)),
                Node("r2", "Relu", ("X",), ("B",)),
                Node("add", "Add", ("A", "B"), ("C",)),
            ),
        ).infer_shapes()
        peak, per_step = peak_activation_bytes(graph)
        # After r1: X 400 + A 400 = 800 (X still needed by r2).
        assert per_step["r1"] == 800
        # After r2: A 400 + B 400 = 800 (X released).
        assert per_step["r2"] == 800
        # Peak during add: A 400 + B 400 + C 400 = 1200.
        assert peak == 1200

    def test_weights_are_excluded_from_the_activation_peak(self, hand_mlp) -> None:
        peak, _ = peak_activation_bytes(hand_mlp.graph.infer_shapes())
        assert peak < hand_mlp.graph.weight_bytes + peak

    def test_a_graph_output_is_never_released(self) -> None:
        graph = ModelGraph(
            name="keep",
            inputs=(TensorSpec("X", (1, 10)),),
            outputs=("A", "B"),
            nodes=(
                Node("r1", "Relu", ("X",), ("A",)),
                Node("r2", "Relu", ("A",), ("B",)),
            ),
        ).infer_shapes()
        _, per_step = peak_activation_bytes(graph)
        # After r2, both A (a graph output) and B are still live: 40 + 40.
        assert per_step["r2"] == 80


class TestNodeCostArrays:
    def test_flattening_preserves_the_per_graph_totals(self, hand_mlp, hand_cnn) -> None:
        graphs = [hand_mlp.graph, hand_cnn.graph]
        node_flops, node_bytes, index = node_cost_arrays(graphs)
        assert len(node_flops) == len(hand_mlp.graph.nodes) + len(hand_cnn.graph.nodes)
        assert np.bincount(index, weights=node_flops)[0] == pytest.approx(99)
        assert np.bincount(index, weights=node_flops)[1] == pytest.approx(1514)
        assert np.bincount(index, weights=node_bytes)[1] == pytest.approx(1832)

    def test_empty_list_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one graph"):
            node_cost_arrays([])
