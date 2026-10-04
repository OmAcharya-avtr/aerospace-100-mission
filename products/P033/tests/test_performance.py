"""Performance benchmark: the package's own cost, with budgets asserted.

These are not latency figures for a model: they are the cost of *using*
`edgeinfer`. They exist because an analytic estimator that takes longer than
running the model is useless, and because this repository runs on one shared
CPU core with a stated compute budget that the test suite itself must respect.

The asserted ceilings are loose on purpose --- a factor of several above what
was observed --- because the host is shared and a tight ceiling would make the
suite flaky rather than informative. The observed figures are recorded by
``validation/validate_performance.py``.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from edgeinfer.analytic import analytic_estimate, node_cost_arrays
from edgeinfer.dataset import generate_population
from edgeinfer.features import population_features
from edgeinfer.predictor import LatencyPredictor
from edgeinfer.roofline import DeviceModel, calibrate_device

DEVICE = DeviceModel("bench", 2e9, 8e9)


@pytest.fixture(scope="module")
def models():
    return generate_population(40, seed=777)[0]


class TestAnalyticThroughput:
    def test_analytic_estimation_is_fast_enough_to_screen_a_population(
        self, models
    ) -> None:
        """40 graphs estimated in under two seconds on one core."""
        graphs = [m.graph for m in models]
        t0 = time.perf_counter()
        estimates = [analytic_estimate(g, DEVICE) for g in graphs]
        elapsed = time.perf_counter() - t0
        assert len(estimates) == 40
        assert elapsed < 2.0, f"analytic estimation took {elapsed:.3f} s for 40 graphs"

    def test_generating_a_population_is_fast_enough(self) -> None:
        t0 = time.perf_counter()
        generate_population(40, seed=778)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"population generation took {elapsed:.3f} s"

    def test_featurising_a_population_is_fast_enough(self, models) -> None:
        t0 = time.perf_counter()
        features = population_features([m.graph for m in models])
        elapsed = time.perf_counter() - t0
        assert features.shape[0] == 40
        assert elapsed < 3.0, f"featurisation took {elapsed:.3f} s for 40 graphs"


class TestCalibrationCost:
    def test_calibration_at_the_default_grid_stays_inside_the_budget(
        self, models
    ) -> None:
        """A 20x20 grid with an NNLS solve per point: the dominant cost of the
        analytic baseline's four-parameter fit."""
        node_flops, node_bytes, graph_index = node_cost_arrays([m.graph for m in models])
        rng = np.random.default_rng(0)
        measured = rng.uniform(1e-5, 1e-3, size=40)
        t0 = time.perf_counter()
        fitted = calibrate_device(
            node_flops, node_bytes, graph_index, measured, "bench", grid=20
        )
        elapsed = time.perf_counter() - t0
        assert fitted.peak_flops > 0
        assert elapsed < 30.0, f"calibration took {elapsed:.3f} s at grid=20"


class TestPredictorCost:
    def test_training_the_default_forest_stays_inside_the_budget(self, models) -> None:
        features = population_features([m.graph for m in models])
        rng = np.random.default_rng(1)
        target = np.power(10.0, -5.0 + 0.2 * features[:, 0] + rng.normal(0, 0.05, 40))
        t0 = time.perf_counter()
        LatencyPredictor(n_estimators=120, seed=1).fit(features, target)
        elapsed = time.perf_counter() - t0
        assert elapsed < 20.0, f"forest training took {elapsed:.3f} s for 120 trees"

    def test_prediction_is_fast_enough_to_screen_many_candidates(self, models) -> None:
        features = population_features([m.graph for m in models])
        rng = np.random.default_rng(1)
        target = np.power(10.0, -5.0 + 0.2 * features[:, 0] + rng.normal(0, 0.05, 40))
        model = LatencyPredictor(n_estimators=60, seed=1).fit(features, target)
        t0 = time.perf_counter()
        for _ in range(20):
            model.predict(features)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"800 predictions took {elapsed:.3f} s"


class TestAnalyticIsCheaperThanMeasuring:
    def test_an_analytic_estimate_costs_less_than_one_measured_profile(
        self, models
    ) -> None:
        """The reason an analytic estimate is worth having at all.

        Compared against a 60-repeat measured profile of the same graph,
        including session construction, which is what a real screening run
        would have to pay.
        """
        from edgeinfer.backends import OnnxRuntimeBackend
        from edgeinfer.harness import benchmark

        model = models[0]
        t0 = time.perf_counter()
        for _ in range(10):
            analytic_estimate(model.graph, DEVICE)
        analytic_cost = (time.perf_counter() - t0) / 10

        t0 = time.perf_counter()
        backend = OnnxRuntimeBackend(model.model_bytes, model.input_feed(0))
        backend.prepare()
        try:
            benchmark(
                backend.infer, label="x", repeats=60, warmup=8, timer_bias_samples=200
            )
        finally:
            backend.close()
        measured_cost = time.perf_counter() - t0

        assert analytic_cost < measured_cost
