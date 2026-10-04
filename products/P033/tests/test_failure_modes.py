"""The four failure modes the batch specification names, plus their edges.

1. a model exceeding the memory budget
2. an unsupported operator
3. a thermally throttled device
4. a budget that is infeasible by construction
"""

from __future__ import annotations

import numpy as np
import pytest

from edgeinfer.analytic import analytic_estimate
from edgeinfer.budget import Budget, Verdict, build_report
from edgeinfer.dataset import random_mlp
from edgeinfer.graph import ModelGraph, Node, TensorSpec
from edgeinfer.ops import UnsupportedOperatorError, node_cost
from edgeinfer.roofline import DeviceModel
from edgeinfer.thermal import ThrottleState, throttled_device


class TestModelExceedsMemoryBudget:
    def test_an_oversized_model_fails_the_memory_row_not_the_latency_row(
        self, rng, device
    ) -> None:
        """A 256 -> 1024 -> 1024 -> 64 MLP in float32 holds roughly 5.4 MB of
        weights, against a 1 MB declared ceiling."""
        model = random_mlp(rng, "fat", n_in=256, widths=(1024, 1024), n_out=64)
        estimate = analytic_estimate(model.graph, device)
        budget = Budget("tight-memory", latency_s=10.0, peak_memory_bytes=1024 * 1024)
        assert estimate.peak_memory_bytes > budget.peak_memory_bytes
        report = build_report(
            budget,
            candidate=model.graph.name,
            environment="test",
            worst_case_latency_s=1e-4,
            worst_case_uncertainty_s=1e-9,
            peak_memory_bytes=float(estimate.peak_memory_bytes),
            peak_memory_uncertainty_bytes=0.0,
        )
        memory_row = next(r for r in report.rows if r.quantity == "peak memory")
        latency_row = report.rows[0]
        assert memory_row.verdict is Verdict.FAIL
        assert latency_row.verdict is Verdict.PASS
        assert report.overall is Verdict.FAIL

    def test_the_failing_row_states_by_how_much(self, rng, device) -> None:
        model = random_mlp(rng, "fat", n_in=256, widths=(1024,), n_out=64)
        estimate = analytic_estimate(model.graph, device)
        budget = Budget("tight", latency_s=10.0, peak_memory_bytes=1024)
        report = build_report(
            budget,
            candidate="fat",
            environment="test",
            peak_memory_bytes=float(estimate.peak_memory_bytes),
        )
        memory_row = next(r for r in report.rows if r.quantity == "peak memory")
        assert memory_row.margin is not None and memory_row.margin < 0
        assert memory_row.utilisation is not None and memory_row.utilisation > 1.0

    def test_onnxruntime_still_runs_the_oversized_model(self, rng) -> None:
        """A model over budget is not a model that fails to execute: the point
        of the budget check is that it catches what execution does not."""
        import onnxruntime as ort

        model = random_mlp(rng, "fat", n_in=256, widths=(512,), n_out=32)
        session = ort.InferenceSession(
            model.model_bytes, providers=["CPUExecutionProvider"]
        )
        assert np.all(np.isfinite(session.run(None, model.input_feed(0))[0]))


class TestUnsupportedOperator:
    def _graph_with(self, op: str) -> ModelGraph:
        return ModelGraph(
            name="unsupported",
            inputs=(TensorSpec("X", (1, 8)),),
            outputs=("Y",),
            nodes=(Node("n", op, ("X",), ("Y",)),),
        )

    @pytest.mark.parametrize(
        "op",
        ["GRU", "NonMaxSuppression", "Einsum", "If", "Loop", "Transpose", "Squeeze"],
    )
    def test_characterisation_is_refused_for_an_unknown_operator(self, op: str) -> None:
        with pytest.raises(UnsupportedOperatorError) as exc:
            self._graph_with(op).infer_shapes()
        assert exc.value.op_type == op

    def test_the_analytic_estimate_refuses_rather_than_undercounting(
        self, device
    ) -> None:
        graph = ModelGraph.__new__(ModelGraph)
        object.__setattr__(graph, "name", "u")
        object.__setattr__(graph, "nodes", (Node("n", "GRU", ("X",), ("Y",)),))
        object.__setattr__(graph, "tensors", {})
        object.__setattr__(graph, "inputs", ())
        object.__setattr__(graph, "initializers", ())
        object.__setattr__(graph, "outputs", ("Y",))
        with pytest.raises(UnsupportedOperatorError):
            analytic_estimate(graph, device)

    def test_a_mixed_graph_fails_on_the_unsupported_node_only(self) -> None:
        graph = ModelGraph(
            name="mixed",
            inputs=(TensorSpec("X", (1, 8)),),
            outputs=("Y",),
            nodes=(
                Node("ok", "Relu", ("X",), ("A",)),
                Node("bad", "GRU", ("A",), ("Y",)),
            ),
        )
        graph.tensors["A"] = TensorSpec("A", (1, 8))
        assert node_cost(graph.nodes[0], graph).flops == 8
        with pytest.raises(UnsupportedOperatorError, match="GRU"):
            node_cost(graph.nodes[1], graph)

    def test_the_message_tells_the_caller_what_to_do(self) -> None:
        with pytest.raises(UnsupportedOperatorError, match="Add a cost rule"):
            self._graph_with("Einsum").infer_shapes()


class TestThermallyThrottledDevice:
    def test_a_throttled_device_produces_a_longer_analytic_latency(
        self, hand_cnn
    ) -> None:
        nominal = DeviceModel("orin-like-declared", 2e9, 8e9)
        state = ThrottleState(
            "sustained load, declared 50 % clock",
            compute_factor=0.5,
            bandwidth_factor=0.6,
            basis="declared for this test; no device measurement exists",
        )
        throttled = throttled_device(nominal, state)
        fast = analytic_estimate(hand_cnn.graph, nominal).latency_s
        slow = analytic_estimate(hand_cnn.graph, throttled).latency_s
        assert slow > fast

    def test_a_model_that_passes_nominally_can_fail_when_throttled(
        self, hand_cnn
    ) -> None:
        nominal = DeviceModel("d", 2e9, 8e9)
        throttled = throttled_device(
            nominal,
            ThrottleState("declared 10 % clock", 0.1, 0.1, basis="declared for this test"),
        )
        nominal_latency = analytic_estimate(hand_cnn.graph, nominal).latency_s
        throttled_latency = analytic_estimate(hand_cnn.graph, throttled).latency_s
        budget = Budget("thermal", latency_s=nominal_latency * 2.0, peak_memory_bytes=1 << 20)
        nominal_report = build_report(
            budget, candidate="c", environment="test",
            worst_case_latency_s=nominal_latency, worst_case_uncertainty_s=0.0,
            peak_memory_bytes=1.0,
        )
        throttled_report = build_report(
            budget, candidate="c", environment="test",
            worst_case_latency_s=throttled_latency, worst_case_uncertainty_s=0.0,
            peak_memory_bytes=1.0,
        )
        assert nominal_report.overall is Verdict.PASS
        assert throttled_report.overall is Verdict.FAIL

    def test_the_throttled_device_name_and_source_say_so(self) -> None:
        throttled = throttled_device(
            DeviceModel("orin-like-declared", 2e9, 8e9, source="declared"),
            ThrottleState("hot", 0.4, 0.7, basis="declared, not measured"),
        )
        assert "hot" in throttled.name
        assert "THROTTLED" in throttled.source
        assert "declared, not measured" in throttled.source

    def test_bandwidth_is_not_derated_unless_asked(self) -> None:
        """Memory clocks throttle on their own schedule, so the default
        bandwidth factor is 1.0 rather than a guess."""
        nominal = DeviceModel("d", 2e9, 8e9)
        throttled = throttled_device(
            nominal, ThrottleState("compute only", 0.5, basis="declared")
        )
        assert throttled.peak_bandwidth_bytes_s == nominal.peak_bandwidth_bytes_s
        assert throttled.peak_flops == pytest.approx(1e9)

    def test_a_nominal_state_changes_nothing_numerically(self) -> None:
        nominal = DeviceModel("d", 2e9, 8e9)
        state = ThrottleState("nominal", 1.0, 1.0, basis="declared")
        assert state.is_nominal
        throttled = throttled_device(nominal, state)
        assert throttled.peak_flops == nominal.peak_flops
        assert throttled.peak_bandwidth_bytes_s == nominal.peak_bandwidth_bytes_s

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"compute_factor": 0.0}, "compute_factor"),
            ({"compute_factor": 1.5}, "compute_factor"),
            ({"bandwidth_factor": -0.1}, "bandwidth_factor"),
            ({"label": ""}, "label"),
            ({"basis": ""}, "basis"),
        ],
    )
    def test_invalid_throttle_states_are_rejected(self, kwargs: dict, match: str) -> None:
        base = {"label": "t", "compute_factor": 0.5, "bandwidth_factor": 0.5,
                "basis": "declared"}
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            ThrottleState(**base)

    def test_overheads_survive_throttling_unchanged(self) -> None:
        nominal = DeviceModel("d", 2e9, 8e9, overhead_per_node_s=1e-6, fixed_overhead_s=4e-6)
        throttled = throttled_device(
            nominal, ThrottleState("hot", 0.3, 0.3, basis="declared")
        )
        assert throttled.overhead_per_node_s == 1e-6
        assert throttled.fixed_overhead_s == 4e-6


class TestInfeasibleBudget:
    def test_duty_cycle_contradicting_the_latency_ceiling(self) -> None:
        budget = Budget(
            "contradictory", latency_s=2e-3, peak_memory_bytes=1 << 20,
            duty_cycle=0.1, period_s=10e-3,
        )
        result = budget.feasibility()
        assert not result.feasible
        assert len(result.reasons) == 1
        assert "contradict" in result.reasons[0]

    def test_median_ceiling_above_the_worst_case_ceiling(self) -> None:
        budget = Budget(
            "contradictory", latency_s=1e-3, peak_memory_bytes=1 << 20,
            median_latency_s=2e-3,
        )
        assert not budget.feasibility().feasible

    def test_an_infeasible_budget_fails_even_with_a_perfect_candidate(self) -> None:
        budget = Budget(
            "contradictory", latency_s=2e-3, peak_memory_bytes=1 << 20,
            duty_cycle=0.01, period_s=10e-3,
        )
        report = build_report(
            budget,
            candidate="instantaneous",
            environment="test",
            worst_case_latency_s=1e-12,
            worst_case_uncertainty_s=0.0,
            peak_memory_bytes=1.0,
            peak_memory_uncertainty_bytes=0.0,
        )
        assert report.overall is Verdict.FAIL
        assert not report.feasibility.feasible

    def test_feasibility_is_decidable_without_touching_a_model(self) -> None:
        """No graph, no backend, no measurement: the contradiction is in the
        declaration."""
        budget = Budget("c", 1e-3, 1 << 20, median_latency_s=5e-3)
        assert not budget.feasibility()

    def test_a_self_consistent_budget_is_feasible_even_if_nothing_meets_it(self) -> None:
        """Feasibility is about the declaration, not about achievability: a
        1 ns ceiling is self-consistent and unachievable, and the two are
        reported separately."""
        budget = Budget("absurd-but-consistent", 1e-9, 1, median_latency_s=1e-10)
        assert budget.feasibility().feasible
