"""Property-based tests for the algebraic identities in the cost model."""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from edgeinfer.analytic import analytic_estimate, peak_activation_bytes
from edgeinfer.budget import Budget
from edgeinfer.graph import ModelGraph, Node, TensorSpec, conv_output_spatial
from edgeinfer.ops import node_cost
from edgeinfer.roofline import DeviceModel, roofline_time_s
from edgeinfer.uncertainty import clock_quantisation_uncertainty_s, combine_standard_uncertainties

FAST = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _gemm_graph(m: int, k: int, n: int) -> ModelGraph:
    return ModelGraph(
        name="g",
        inputs=(TensorSpec("X", (m, k)),),
        outputs=("Y",),
        nodes=(Node("g0", "Gemm", ("X", "W"), ("Y",)),),
        initializers=(TensorSpec("W", (k, n)),),
    ).infer_shapes()


class TestCostIdentities:
    @FAST
    @given(
        m=st.integers(1, 16),
        k=st.integers(1, 32),
        n=st.integers(1, 32),
    )
    def test_gemm_flops_are_twice_its_macs(self, m: int, k: int, n: int) -> None:
        graph = _gemm_graph(m, k, n)
        cost = node_cost(graph.nodes[0], graph)
        assert cost.flops == 2 * cost.macs
        assert cost.macs == m * k * n

    @FAST
    @given(m=st.integers(1, 8), k=st.integers(1, 16), n=st.integers(1, 16), s=st.integers(2, 5))
    def test_gemm_macs_scale_linearly_in_each_dimension(
        self, m: int, k: int, n: int, s: int
    ) -> None:
        base = node_cost(_gemm_graph(m, k, n).nodes[0], _gemm_graph(m, k, n)).macs
        scaled_graph = _gemm_graph(m * s, k, n)
        scaled = node_cost(scaled_graph.nodes[0], scaled_graph).macs
        assert scaled == s * base

    @FAST
    @given(
        n=st.integers(1, 4),
        c=st.integers(1, 8),
        h=st.integers(3, 16),
        cout=st.integers(1, 8),
        kernel=st.sampled_from([1, 3]),
    )
    def test_conv_macs_match_the_closed_form(
        self, n: int, c: int, h: int, cout: int, kernel: int
    ) -> None:
        pad = kernel // 2
        graph = ModelGraph(
            name="c",
            inputs=(TensorSpec("X", (n, c, h, h)),),
            outputs=("Y",),
            nodes=(
                Node(
                    "c0",
                    "Conv",
                    ("X", "W"),
                    ("Y",),
                    {"kernel_shape": [kernel, kernel], "pads": [pad, pad, pad, pad]},
                ),
            ),
            initializers=(TensorSpec("W", (cout, c, kernel, kernel)),),
        ).infer_shapes()
        out = graph.spec("Y")
        expected = n * cout * out.shape[2] * out.shape[3] * c * kernel * kernel
        assert node_cost(graph.nodes[0], graph).macs == expected

    @FAST
    @given(
        flops=st.floats(0.0, 1e9, allow_nan=False),
        traffic=st.floats(0.0, 1e9, allow_nan=False),
        pf=st.floats(1e6, 1e12, allow_nan=False),
        bw=st.floats(1e6, 1e12, allow_nan=False),
    )
    def test_roofline_time_is_monotone_non_increasing_in_both_peaks(
        self, flops: float, traffic: float, pf: float, bw: float
    ) -> None:
        slow = DeviceModel("d", pf, bw)
        fast = DeviceModel("d", pf * 2.0, bw * 2.0)
        assert roofline_time_s(flops, traffic, fast) <= roofline_time_s(flops, traffic, slow)

    @FAST
    @given(
        flops=st.floats(0.0, 1e9, allow_nan=False),
        traffic=st.floats(0.0, 1e9, allow_nan=False),
        pf=st.floats(1e6, 1e12, allow_nan=False),
        bw=st.floats(1e6, 1e12, allow_nan=False),
    )
    def test_roofline_time_is_never_below_either_single_roof(
        self, flops: float, traffic: float, pf: float, bw: float
    ) -> None:
        device = DeviceModel("d", pf, bw)
        t = roofline_time_s(flops, traffic, device)
        assert t >= flops / pf - 1e-18
        assert t >= traffic / bw - 1e-18


class TestLivenessProperties:
    @FAST
    @given(depth=st.integers(1, 8), width=st.integers(1, 64))
    def test_a_chain_peak_is_the_largest_adjacent_pair(
        self, depth: int, width: int
    ) -> None:
        """In a straight chain of shape-preserving ops, exactly two tensors are
        live at the peak, so the peak is twice one tensor's size."""
        nodes = []
        current = "X"
        for i in range(depth):
            out = f"T{i}"
            nodes.append(Node(f"r{i}", "Relu", (current,), (out,)))
            current = out
        graph = ModelGraph(
            name="chain",
            inputs=(TensorSpec("X", (1, width)),),
            outputs=(current,),
            nodes=tuple(nodes),
        ).infer_shapes()
        peak, _ = peak_activation_bytes(graph)
        assert peak == 2 * width * 4

    @FAST
    @given(depth=st.integers(1, 6), width=st.integers(1, 32))
    def test_peak_never_exceeds_the_sum_of_every_tensor(
        self, depth: int, width: int
    ) -> None:
        nodes = []
        current = "X"
        for i in range(depth):
            nodes.append(Node(f"r{i}", "Relu", (current,), (f"T{i}",)))
            current = f"T{i}"
        graph = ModelGraph(
            name="chain",
            inputs=(TensorSpec("X", (1, width)),),
            outputs=(current,),
            nodes=tuple(nodes),
        ).infer_shapes()
        peak, _ = peak_activation_bytes(graph)
        total = sum(
            spec.nbytes
            for name, spec in graph.tensors.items()
            if name not in {s.name for s in graph.initializers}
        )
        assert peak <= total


class TestConvFormulaProperties:
    @FAST
    @given(
        size=st.integers(1, 64),
        kernel=st.integers(1, 7),
        stride=st.integers(1, 4),
        pad=st.integers(0, 4),
    )
    def test_output_never_exceeds_the_padded_input(
        self, size: int, kernel: int, stride: int, pad: int
    ) -> None:
        padded = size + 2 * pad
        if kernel > padded:
            return
        out = conv_output_spatial((size,), (kernel,), (stride,), (pad, pad), (1,))[0]
        assert 1 <= out <= padded

    @FAST
    @given(size=st.integers(1, 64), kernel=st.integers(1, 7))
    def test_same_padding_preserves_the_size_for_an_odd_kernel(
        self, size: int, kernel: int
    ) -> None:
        if kernel % 2 == 0:
            return
        pad = kernel // 2
        if kernel > size + 2 * pad:
            return
        assert conv_output_spatial((size,), (kernel,), (1,), (pad, pad), (1,))[0] == size


class TestBudgetProperties:
    @FAST
    @given(
        latency_ms=st.floats(0.01, 100.0, allow_nan=False),
        duty=st.floats(0.01, 1.0, allow_nan=False),
        period_ms=st.floats(0.01, 1000.0, allow_nan=False),
    )
    def test_effective_ceiling_is_never_above_either_declared_ceiling(
        self, latency_ms: float, duty: float, period_ms: float
    ) -> None:
        budget = Budget(
            "b",
            latency_s=latency_ms * 1e-3,
            peak_memory_bytes=1 << 20,
            duty_cycle=duty,
            period_s=period_ms * 1e-3,
        )
        assert budget.effective_latency_s <= budget.latency_s
        implied = budget.duty_cycle_latency_s
        assert implied is not None and budget.effective_latency_s <= implied

    @FAST
    @given(
        latency_ms=st.floats(0.01, 100.0, allow_nan=False),
        duty=st.floats(0.01, 1.0, allow_nan=False),
        period_ms=st.floats(0.01, 1000.0, allow_nan=False),
    )
    def test_feasibility_is_exactly_the_absence_of_a_tighter_duty_ceiling(
        self, latency_ms: float, duty: float, period_ms: float
    ) -> None:
        budget = Budget(
            "b",
            latency_s=latency_ms * 1e-3,
            peak_memory_bytes=1 << 20,
            duty_cycle=duty,
            period_s=period_ms * 1e-3,
        )
        implied = budget.duty_cycle_latency_s
        assert implied is not None
        assert budget.feasibility().feasible == (implied >= budget.latency_s)


class TestUncertaintyProperties:
    @FAST
    @given(
        a=st.floats(0.0, 1e3, allow_nan=False),
        b=st.floats(0.0, 1e3, allow_nan=False),
    )
    def test_combination_is_symmetric_and_at_least_each_component(
        self, a: float, b: float
    ) -> None:
        combined = combine_standard_uncertainties(a, b)
        assert combined == combine_standard_uncertainties(b, a)
        assert combined >= max(a, b) - 1e-12

    @FAST
    @given(resolution=st.floats(0.0, 1.0, allow_nan=False), reads=st.integers(1, 8))
    def test_clock_term_grows_as_root_of_the_read_count(
        self, resolution: float, reads: int
    ) -> None:
        one = clock_quantisation_uncertainty_s(resolution, 1)
        many = clock_quantisation_uncertainty_s(resolution, reads)
        assert many == np.float64(one * np.sqrt(reads)) or abs(
            many - one * np.sqrt(reads)
        ) < 1e-18


class TestAnalyticProperties:
    @FAST
    @given(width=st.integers(1, 64), depth=st.integers(1, 5))
    def test_analytic_latency_is_non_negative_and_monotone_in_depth(
        self, width: int, depth: int
    ) -> None:
        device = DeviceModel("d", 1e9, 1e9)

        def chain(n: int) -> ModelGraph:
            nodes = []
            current = "X"
            for i in range(n):
                nodes.append(Node(f"r{i}", "Relu", (current,), (f"T{i}",)))
                current = f"T{i}"
            return ModelGraph(
                name="c",
                inputs=(TensorSpec("X", (1, width)),),
                outputs=(current,),
                nodes=tuple(nodes),
            )

        shallow = analytic_estimate(chain(depth), device).latency_s
        deeper = analytic_estimate(chain(depth + 1), device).latency_s
        assert 0.0 <= shallow <= deeper
