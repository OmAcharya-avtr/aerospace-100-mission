"""End-to-end paths: declare a budget, characterise a candidate, decide."""

from __future__ import annotations

import json

import numpy as np
import pytest

from edgeinfer.analytic import analytic_estimate, node_cost_arrays
from edgeinfer.backends import OnnxRuntimeBackend, PipelineStage, SimulatedBackend
from edgeinfer.budget import Budget, Verdict, build_report
from edgeinfer.dataset import generate_population
from edgeinfer.features import population_features
from edgeinfer.harness import benchmark
from edgeinfer.predictor import LatencyPredictor, compare_predictors, split_indices
from edgeinfer.report import (
    CLOUD_ENVIRONMENT_NOTE,
    ResultRow,
    crosscheck_payload,
    write_crosscheck_json,
    write_results_file,
)
from edgeinfer.roofline import DeviceModel, calibrate_device
from edgeinfer.thermal import ThrottleState, throttled_device


class TestFullCandidateCheck:
    def test_analytic_then_measured_then_verdict(self, small_cnn) -> None:
        budget = Budget(
            name="integration",
            latency_s=50e-3,
            peak_memory_bytes=8 * 1024 * 1024,
            median_latency_s=25e-3,
            power_w=7.0,
            duty_cycle=0.5,
            period_s=200e-3,
        )
        assert budget.feasibility().feasible

        device = DeviceModel("declared-target", 2e9, 8e9)
        estimate = analytic_estimate(small_cnn.graph, device)
        assert estimate.total_flops > 0

        backend = OnnxRuntimeBackend(small_cnn.model_bytes, small_cnn.input_feed(0))
        backend.prepare()
        try:
            profile = benchmark(
                backend.infer,
                label=small_cnn.graph.name,
                repeats=60,
                warmup=8,
                timer_bias_samples=200,
                environment_note=CLOUD_ENVIRONMENT_NOTE,
            )
        finally:
            backend.close()

        tail = profile.uncertainty("p99")
        median = profile.uncertainty("p50")
        report = build_report(
            budget,
            candidate=small_cnn.graph.name,
            environment=profile.environment.one_line(),
            worst_case_latency_s=tail.value,
            worst_case_uncertainty_s=tail.combined_s,
            median_latency_s=median.value,
            median_uncertainty_s=median.combined_s,
            peak_memory_bytes=float(estimate.peak_memory_bytes),
            declared_power_w=7.0,
            latency_method=profile.method_line(),
            memory_method="analytic liveness analysis",
        )
        assert report.overall.is_pass
        assert tail.value >= median.value
        assert any("power" in u for u in report.undecided)

    def test_the_same_candidate_can_fail_a_throttled_device_budget(
        self, small_cnn
    ) -> None:
        nominal = DeviceModel("declared-target", 2e9, 8e9)
        throttled = throttled_device(
            nominal,
            ThrottleState("declared 5 % clock", 0.05, 0.05, basis="declared for a test"),
        )
        fast = analytic_estimate(small_cnn.graph, nominal).latency_s
        slow = analytic_estimate(small_cnn.graph, throttled).latency_s
        budget = Budget("throttle", latency_s=fast * 3.0, peak_memory_bytes=1 << 24)
        assert slow > budget.latency_s
        report = build_report(
            budget,
            candidate="throttled",
            environment="test",
            worst_case_latency_s=slow,
            worst_case_uncertainty_s=0.0,
            peak_memory_bytes=1.0,
        )
        assert report.overall is Verdict.FAIL

    def test_results_file_is_written_with_an_empty_device_column(
        self, small_mlp, tmp_path
    ) -> None:
        backend = OnnxRuntimeBackend(small_mlp.model_bytes, small_mlp.input_feed(0))
        backend.prepare()
        try:
            profile = benchmark(
                backend.infer, label="x", repeats=40, warmup=5, timer_bias_samples=100
            )
        finally:
            backend.close()
        rows = (
            ResultRow(
                "p99 latency", "s", profile.p99_s,
                profile.uncertainty("p99").combined_s, profile.method_line()
            ),
            ResultRow(
                "p50 latency", "s", profile.p50_s,
                profile.uncertainty("p50").combined_s, profile.method_line()
            ),
        )
        path = write_results_file(tmp_path / "res.md", "Integration", rows)
        text = path.read_text()
        assert "(not measured)" in text
        assert "Jetson Orin Nano" in text


class TestPredictorPipeline:
    def test_population_to_comparison_runs_inside_the_compute_budget(self) -> None:
        """Generate, measure, calibrate the analytic baseline, fit the learned
        model, compare. 40 models at 25 repeats keeps this test under a few
        seconds on one core; the full campaign is in
        ``validation/validate_predictor.py``."""
        models, _summaries = generate_population(40, seed=4242)
        graphs = [m.graph for m in models]

        measured = np.empty(len(models))
        for i, model in enumerate(models):
            backend = OnnxRuntimeBackend(model.model_bytes, model.input_feed(0))
            backend.prepare()
            try:
                profile = benchmark(
                    backend.infer, label=model.graph.name, repeats=25, warmup=5,
                    timer_bias_samples=50,
                )
            finally:
                backend.close()
            measured[i] = profile.p50_s

        node_flops, node_bytes, graph_index = node_cost_arrays(graphs)
        train, test = split_indices(len(models), 0.3, seed=4242)

        train_mask = np.isin(graph_index, train)
        remap = {g: i for i, g in enumerate(train)}
        fitted = calibrate_device(
            node_flops[train_mask],
            node_bytes[train_mask],
            np.asarray([remap[g] for g in graph_index[train_mask]]),
            measured[train],
            "container-fit",
            grid=10,
        )
        assert fitted.source == "calibrated-fit"

        analytic_test = np.asarray(
            [analytic_estimate(graphs[i], fitted).latency_s for i in test]
        )
        features = population_features(graphs)
        learned = LatencyPredictor(n_estimators=40, seed=4242).fit(
            features[train], measured[train]
        )
        learned_test = learned.predict(features[test])

        metrics_a, metrics_l, verdict = compare_predictors(
            analytic_test, learned_test, measured[test]
        )
        assert metrics_a.n_test == metrics_l.n_test == len(test)
        assert verdict
        assert np.all(analytic_test > 0)
        assert np.all(learned_test > 0)

    def test_an_uncertainty_accompanies_every_learned_prediction(self) -> None:
        models, _ = generate_population(20, seed=77)
        graphs = [m.graph for m in models]
        features = population_features(graphs)
        rng = np.random.default_rng(0)
        target = np.power(10.0, -5.0 + 0.2 * features[:, 0] + rng.normal(0, 0.05, 20))
        model = LatencyPredictor(n_estimators=30, seed=1).fit(features, target)
        for result in model.predict_with_uncertainty(features):
            assert result.value > 0
            assert result.uncertainty >= 0


class TestCrosscheckExport:
    def test_the_crosscheck_file_is_reproducible_from_its_own_definition(
        self, tmp_path
    ) -> None:
        """Write the payload, read it back, rebuild the pipeline from the
        stated stages and seed, and require the same injected mean. This is
        what makes the file usable by an independent implementation."""
        stages = (
            PipelineStage("preprocess", 60e-6, 12e-6, "lognormal"),
            PipelineStage("inference", 180e-6, 40e-6, "lognormal"),
            PipelineStage("postprocess", 30e-6, 6e-6, "lognormal"),
        )
        backend = SimulatedBackend(stages, seed=20260401, consume_time=False)
        backend.prepare()
        draws = np.array([backend.infer() for _ in range(4000)])
        payload = crosscheck_payload(
            stages=tuple(s.as_dict() for s in stages),
            n_samples=4000,
            seed=20260401,
            mean_s=float(draws.mean()),
            p50_s=float(np.quantile(draws, 0.5)),
            p99_s=float(np.quantile(draws, 0.99)),
        )
        path = write_crosscheck_json(tmp_path / "c.json", payload)
        loaded = json.loads(path.read_text())

        rebuilt = SimulatedBackend(
            tuple(
                PipelineStage(s["name"], s["mean_s"], s["std_s"], s["dist"])
                for s in loaded["stages"]
            ),
            seed=loaded["seed"],
            consume_time=False,
        )
        rebuilt.prepare()
        redraws = np.array([rebuilt.infer() for _ in range(loaded["n_samples"])])
        assert np.allclose(redraws, draws)
        assert loaded["measured"]["mean_s"] == pytest.approx(float(draws.mean()))

    def test_the_injected_mean_is_recovered_within_the_sampling_error(self) -> None:
        stages = (
            PipelineStage("a", 60e-6, 12e-6, "lognormal"),
            PipelineStage("b", 180e-6, 40e-6, "lognormal"),
        )
        backend = SimulatedBackend(stages, seed=1, consume_time=False)
        backend.prepare()
        n = 20_000
        draws = np.array([backend.infer() for _ in range(n)])
        injected = backend.injected_mean_s
        standard_error = backend.injected_std_s / np.sqrt(n)
        assert abs(draws.mean() - injected) < 4.0 * standard_error
