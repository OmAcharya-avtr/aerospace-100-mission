"""Pinned seeded outputs.

Every value below was produced by running this repository's own code and is
pinned so that an unintended change to the cost model, the generator, the
feature set, the calibration or the predictor shows up as a test failure
rather than as a quietly different number in the README.

Nothing here is a timing. Timings vary with the host and are checked
statistically in ``test_harness.py`` and ``validation/``; what is pinned here
is the deterministic part: operation counts, byte counts, generated model
bytes, features, the calibration fit on fixed synthetic data, the split, and
the predictor's output on a fixed synthetic target.

A value changing is not automatically a defect --- but it must be a change
someone made on purpose, and the new value has to be re-derived and the
README's numbers updated with it.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from edgeinfer.analytic import analytic_estimate
from edgeinfer.backends import PipelineStage, SimulatedBackend
from edgeinfer.dataset import generate_population, hand_counted_cnn, hand_counted_mlp
from edgeinfer.features import FEATURE_NAMES, population_features
from edgeinfer.predictor import LatencyPredictor, split_indices
from edgeinfer.roofline import DeviceModel, calibrate_device
from edgeinfer.uncertainty import bootstrap_quantile_uncertainty

POPULATION_SIZE = 16
POPULATION_SEED = 20260401

PINNED_MODEL_BYTES_SHA256 = (
    "4c61fd3d8f3e6d9fa4a1278e5531621edb1ec15d7bb7f44bd45b0b5cac32c48f"
)
PINNED_HAND_MLP_SHA256 = (
    "e8e29a9704703730398ccf7a8adcc140081866026f83132b30f359e4fe315db2"
)
PINNED_HAND_CNN_SHA256 = (
    "f7aac13b3751e101fdf432b13a36b073dd8e8d29e2a9ee40b866fb23f9f7f171"
)

PINNED_FLOPS = [
    622127, 164750, 42385, 648004, 383708, 5896805, 10829, 6149383,
    448707, 3191806, 689505, 4816, 186195, 20688223, 102360, 351706,
]
PINNED_MACS = [
    310169, 67392, 20880, 285552, 190641, 2904498, 5264, 3034098,
    223667, 1583463, 343856, 1858, 92698, 10285344, 50826, 152076,
]
PINNED_TRAFFIC_BYTES = [
    1259492, 511720, 86740, 730128, 774108, 882476, 22636, 810668,
    910104, 431104, 1394796, 17676, 379592, 2007184, 207712, 1001172,
]
PINNED_PEAK_MEMORY_BYTES = [
    1248140, 329648, 85168, 223808, 767524, 372664, 21880, 278908,
    901016, 244448, 1382540, 9808, 375060, 1145364, 205912, 705656,
]
PINNED_NODE_COUNTS = [5, 5, 3, 11, 5, 11, 3, 11, 5, 8, 7, 5, 5, 8, 3, 5]

PINNED_FEATURE_ROW_0 = [
    5.793879748, 6.1001957581, 5.49159979, 0.1743362765, 5.0, 5.5781530782,
    6.094939411, 3.5798978696, 2.0, 0.0, 0.9985967495, 0.0, 0.0014032505,
    0.0, 0.0, 0.0, 0.0,
]
PINNED_FEATURE_SUM = 679.1918120549

PINNED_CALIBRATION = {
    "peak_flops": 2410461622.181291,
    "peak_bandwidth_bytes_s": 5022311469.513855,
    "overhead_per_node_s": 1.7653892994040943e-06,
    "fixed_overhead_s": 0.0,
}

PINNED_PREDICTIONS_FIRST_3 = [
    0.001087802683355342, 0.004458433298412043, 0.0013620882741616624,
]
PINNED_PREDICTION_SUM = 0.04408259410216306
PINNED_PREDICTION_UNCERTAINTY = 0.0005804256610794272
PINNED_TRAIN_INDEX_FIRST_5 = [0, 2, 3, 5, 8]
PINNED_TEST_INDEX_FIRST_5 = [1, 4, 6, 7, 10]

PINNED_BOOTSTRAP_P99_UNCERTAINTY = 0.2784005443313197
PINNED_SIMULATED_MEAN_S = 0.00023983766748178842
PINNED_SIMULATED_P99_S = 0.0003590881942075898

PINNED_DEVICE = DeviceModel("pin", 1e9, 1e9)


def _population():
    return generate_population(POPULATION_SIZE, seed=POPULATION_SEED)[0]


class TestPinnedGeneratedModels:
    def test_concatenated_model_bytes_hash_is_unchanged(self) -> None:
        models = _population()
        digest = hashlib.sha256(b"".join(m.model_bytes for m in models)).hexdigest()
        assert digest == PINNED_MODEL_BYTES_SHA256

    def test_hand_counted_model_hashes_are_unchanged(self) -> None:
        assert (
            hashlib.sha256(hand_counted_mlp().model_bytes).hexdigest()
            == PINNED_HAND_MLP_SHA256
        )
        assert (
            hashlib.sha256(hand_counted_cnn().model_bytes).hexdigest()
            == PINNED_HAND_CNN_SHA256
        )

    def test_node_counts_are_unchanged(self) -> None:
        assert [len(m.graph.nodes) for m in _population()] == PINNED_NODE_COUNTS


class TestPinnedAnalyticTotals:
    @pytest.fixture
    def estimates(self):
        return [analytic_estimate(m.graph, PINNED_DEVICE) for m in _population()]

    def test_flop_counts_are_unchanged(self, estimates) -> None:
        assert [e.total_flops for e in estimates] == PINNED_FLOPS

    def test_mac_counts_are_unchanged(self, estimates) -> None:
        assert [e.total_macs for e in estimates] == PINNED_MACS

    def test_traffic_counts_are_unchanged(self, estimates) -> None:
        assert [e.total_traffic_bytes for e in estimates] == PINNED_TRAFFIC_BYTES

    def test_peak_memory_is_unchanged(self, estimates) -> None:
        assert [e.peak_memory_bytes for e in estimates] == PINNED_PEAK_MEMORY_BYTES

    def test_flops_are_at_least_twice_the_macs(self, estimates) -> None:
        """A structural invariant, not a pin: every multiply-accumulate is two
        flops, and elementwise work adds more."""
        for estimate in estimates:
            assert estimate.total_flops >= 2 * estimate.total_macs


class TestPinnedFeatures:
    def test_first_row_is_unchanged(self) -> None:
        features = population_features([m.graph for m in _population()])
        assert features.shape == (POPULATION_SIZE, len(FEATURE_NAMES))
        assert np.allclose(features[0], PINNED_FEATURE_ROW_0, rtol=0, atol=1e-9)

    def test_matrix_sum_is_unchanged(self) -> None:
        features = population_features([m.graph for m in _population()])
        assert float(features.sum()) == pytest.approx(PINNED_FEATURE_SUM, abs=1e-8)


class TestPinnedCalibration:
    def _synthetic(self):
        rng = np.random.default_rng(31337)
        node_flops = 10.0 ** rng.uniform(2, 6, size=200)
        node_bytes = 10.0 ** rng.uniform(2, 6, size=200)
        graph_index = np.repeat(np.arange(40), 5)
        truth = DeviceModel(
            "t", 2e9, 6e9, overhead_per_node_s=3e-7, fixed_overhead_s=5e-6
        )
        per_node = np.maximum(
            node_flops / truth.peak_flops, node_bytes / truth.peak_bandwidth_bytes_s
        )
        measured = np.bincount(graph_index, weights=per_node) + 3e-7 * 5 + 5e-6
        return node_flops, node_bytes, graph_index, measured

    def test_the_fit_on_fixed_synthetic_data_is_unchanged(self) -> None:
        node_flops, node_bytes, graph_index, measured = self._synthetic()
        fitted = calibrate_device(
            node_flops, node_bytes, graph_index, measured, "pin", grid=12
        )
        assert fitted.peak_flops == pytest.approx(
            PINNED_CALIBRATION["peak_flops"], rel=1e-9
        )
        assert fitted.peak_bandwidth_bytes_s == pytest.approx(
            PINNED_CALIBRATION["peak_bandwidth_bytes_s"], rel=1e-9
        )
        assert fitted.overhead_per_node_s == pytest.approx(
            PINNED_CALIBRATION["overhead_per_node_s"], rel=1e-9
        )
        assert fitted.fixed_overhead_s == pytest.approx(
            PINNED_CALIBRATION["fixed_overhead_s"], abs=1e-12
        )


class TestPinnedPredictor:
    def _fixture(self):
        rng = np.random.default_rng(4242)
        features = rng.uniform(0.0, 6.0, size=(120, len(FEATURE_NAMES)))
        target = np.power(
            10.0,
            -5.0 + 0.4 * features[:, 0] + 0.1 * features[:, 1]
            + rng.normal(0.0, 0.03, size=120),
        )
        return features, target

    def test_the_split_is_unchanged(self) -> None:
        train, test = split_indices(120, 0.3, seed=4242)
        assert train[:5].tolist() == PINNED_TRAIN_INDEX_FIRST_5
        assert test[:5].tolist() == PINNED_TEST_INDEX_FIRST_5

    def test_predictions_on_a_fixed_target_are_unchanged(self) -> None:
        features, target = self._fixture()
        train, test = split_indices(120, 0.3, seed=4242)
        model = LatencyPredictor(n_estimators=50, seed=4242).fit(
            features[train], target[train]
        )
        predicted = model.predict(features[test])
        assert np.allclose(
            predicted[:3], PINNED_PREDICTIONS_FIRST_3, rtol=1e-9, atol=0
        )
        assert float(predicted.sum()) == pytest.approx(PINNED_PREDICTION_SUM, rel=1e-9)

    def test_the_uncertainty_output_is_unchanged(self) -> None:
        features, target = self._fixture()
        train, test = split_indices(120, 0.3, seed=4242)
        model = LatencyPredictor(n_estimators=50, seed=4242).fit(
            features[train], target[train]
        )
        result = model.predict_with_uncertainty(features[test][:1])[0]
        assert result.uncertainty == pytest.approx(
            PINNED_PREDICTION_UNCERTAINTY, rel=1e-9
        )


class TestPinnedStochasticHelpers:
    def test_the_bootstrap_uncertainty_is_unchanged(self) -> None:
        samples = np.random.default_rng(5).lognormal(0.0, 0.4, size=300)
        value = bootstrap_quantile_uncertainty(samples, 0.99, 400, seed=9)
        assert value == pytest.approx(PINNED_BOOTSTRAP_P99_UNCERTAINTY, rel=1e-9)

    def test_the_simulated_pipeline_draws_are_unchanged(self) -> None:
        backend = SimulatedBackend(
            (PipelineStage("a", 60e-6, 12e-6), PipelineStage("b", 180e-6, 40e-6)),
            seed=20260401,
            consume_time=False,
        )
        backend.prepare()
        draws = np.array([backend.infer() for _ in range(2000)])
        assert float(draws.mean()) == pytest.approx(PINNED_SIMULATED_MEAN_S, rel=1e-9)
        assert float(np.quantile(draws, 0.99)) == pytest.approx(
            PINNED_SIMULATED_P99_S, rel=1e-9
        )
