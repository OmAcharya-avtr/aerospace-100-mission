"""Learned fade-exceedance predictor: features, uncertainty, metrics, calibration.

Exercises REQ-28, REQ-29, REQ-30, REQ-31 of docs/REQUIREMENTS.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from codedfade.channel import ChannelConfig, generate_amplitude
from codedfade.predictor import (
    FEATURE_NAMES,
    WINDOW,
    FadeExceedancePredictor,
    brier_score,
    build_dataset,
    crossing_features,
    empirical_exceedance_baseline,
    expected_calibration_error,
    log_loss_score,
    reliability_table,
    roc_auc,
)

SI, TAU, FS, THR, TARGET = 0.6, 2.0e-4, 1.0e6, 0.6, 14


@pytest.fixture(scope="module")
def dataset():
    return build_dataset([0, 1, 2, 3], 60_000, THR, TARGET, SI, TAU, FS)


@pytest.fixture(scope="module")
def heldout():
    return build_dataset([100, 101], 60_000, THR, TARGET, SI, TAU, FS)


@pytest.fixture(scope="module")
def fitted(dataset):
    return FadeExceedancePredictor(80, 8, 0).fit(dataset.features, dataset.labels)


class TestFeatures:
    def test_feature_count_matches_the_name_list(self, dataset) -> None:
        assert dataset.features.shape[1] == len(FEATURE_NAMES)

    def test_features_are_finite(self, dataset) -> None:
        assert np.all(np.isfinite(dataset.features))

    def test_labels_are_binary_and_both_classes_present(self, dataset) -> None:
        assert set(np.unique(dataset.labels).tolist()) == {0, 1}

    def test_label_matches_the_duration_definition(self, dataset) -> None:
        assert np.array_equal(
            dataset.labels, (dataset.durations_samples > TARGET).astype(np.uint8)
        )

    def test_features_use_only_causal_information(self) -> None:
        """Truncating the series after a crossing must not change that crossing's
        features. The test finds the first complete fade, cuts the record a few
        samples into it, and compares the first feature row."""
        cfg = ChannelConfig(SI, TAU, FS, seed=0)
        a = generate_amplitude(cfg, 60_000)
        rows_full, durations, starts = crossing_features(a, THR, TARGET)
        assert rows_full.shape[0] > 0
        s = int(starts[0])
        truncated = a[: s + 1]
        # append a stretch above threshold so the run closes and is not censored
        truncated = np.concatenate([truncated, np.full(50, 2.0)])
        rows_cut, _, _ = crossing_features(truncated, THR, TARGET)
        assert rows_cut.shape[0] >= 1
        assert np.allclose(rows_cut[0], rows_full[0])

    def test_window_is_respected(self) -> None:
        """A crossing closer to the start than WINDOW is dropped."""
        a = np.concatenate([np.full(3, 2.0), np.full(5, 0.1), np.full(10, 2.0)])
        rows, _, _ = crossing_features(a, THR, TARGET)
        assert rows.shape[0] == 0
        assert WINDOW == 32

    def test_groups_track_the_source_seed(self, dataset) -> None:
        assert set(np.unique(dataset.group).tolist()) == {0, 1, 2, 3}

    @pytest.mark.parametrize(
        "amp,thr,target,msg",
        [
            (np.array([1.0, -1.0]), 0.6, 4, "strictly positive"),
            (np.ones(100), 0.0, 4, "threshold must be > 0"),
            (np.ones(100), 0.6, 0, "target_samples must be >= 1"),
        ],
    )
    def test_rejects_bad_inputs(self, amp, thr, target, msg: str) -> None:
        with pytest.raises(ValueError, match=msg):
            crossing_features(amp, thr, target)

    def test_rejects_non_1d(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            crossing_features(np.ones((4, 4)), 0.6, 4)

    def test_build_dataset_rejects_empty_seed_list(self) -> None:
        with pytest.raises(ValueError, match="seeds must be non-empty"):
            build_dataset([], 1000, THR, TARGET)

    def test_build_dataset_raises_when_no_fade_is_found(self) -> None:
        with pytest.raises(RuntimeError, match="no complete fades"):
            build_dataset([0], 2000, 1e-6, TARGET, SI, TAU, FS)


class TestPredictor:
    def test_predict_returns_a_probability_and_an_uncertainty(self, fitted, heldout) -> None:
        out = fitted.predict(heldout.features)
        assert out.probability.shape == (heldout.labels.size,)
        assert out.uncertainty.shape == (heldout.labels.size,)
        assert np.all((out.probability >= 0) & (out.probability <= 1))
        assert np.all(out.uncertainty >= 0)

    def test_uncertainty_is_not_identically_zero(self, fitted, heldout) -> None:
        """A point estimate dressed up as an interval is worse than no interval."""
        assert float(fitted.predict(heldout.features).uncertainty.mean()) > 0.01

    def test_unfitted_predictor_raises(self) -> None:
        p = FadeExceedancePredictor(8, 4, 0)
        with pytest.raises(RuntimeError, match="not fitted"):
            p.predict(np.zeros((2, len(FEATURE_NAMES))))
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = p.feature_importances

    def test_rejects_single_class_labels(self) -> None:
        p = FadeExceedancePredictor(8, 4, 0)
        with pytest.raises(ValueError, match="both classes"):
            p.fit(np.zeros((10, len(FEATURE_NAMES))), np.zeros(10, dtype=np.uint8))

    def test_rejects_shape_mismatch(self) -> None:
        p = FadeExceedancePredictor(8, 4, 0)
        with pytest.raises(ValueError, match="shape mismatch"):
            p.fit(np.zeros((10, len(FEATURE_NAMES))), np.zeros(9, dtype=np.uint8))

    def test_feature_importances_align_with_the_names(self, fitted) -> None:
        assert fitted.feature_importances.shape == (len(FEATURE_NAMES),)
        assert float(fitted.feature_importances.sum()) == pytest.approx(1.0, abs=1e-6)

    def test_is_reproducible(self, dataset) -> None:
        a = FadeExceedancePredictor(40, 6, 3).fit(dataset.features, dataset.labels)
        b = FadeExceedancePredictor(40, 6, 3).fit(dataset.features, dataset.labels)
        assert np.allclose(
            a.predict(dataset.features[:50]).probability,
            b.predict(dataset.features[:50]).probability,
        )


class TestMetrics:
    def test_brier_known_answers(self) -> None:
        assert brier_score(np.array([1.0, 0.0]), np.array([1, 0])) == pytest.approx(0.0)
        assert brier_score(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(0.25)

    def test_brier_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            brier_score(np.zeros(3), np.zeros(2))

    def test_log_loss_known_answer(self) -> None:
        assert log_loss_score(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(
            np.log(2.0)
        )

    def test_auc_known_answers(self) -> None:
        assert roc_auc(np.array([0.9, 0.1]), np.array([1, 0])) == pytest.approx(1.0)
        assert roc_auc(np.array([0.1, 0.9]), np.array([1, 0])) == pytest.approx(0.0)
        assert roc_auc(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(0.5)

    def test_auc_is_nan_with_one_class(self) -> None:
        assert np.isnan(roc_auc(np.array([0.5, 0.6]), np.array([1, 1])))

    def test_auc_handles_ties_by_averaging_ranks(self) -> None:
        p = np.array([0.5, 0.5, 0.5, 0.5])
        y = np.array([1, 1, 0, 0])
        assert roc_auc(p, y) == pytest.approx(0.5)

    def test_reliability_table_shapes_and_counts(self) -> None:
        p = np.linspace(0.0, 1.0, 100)
        y = (p > 0.5).astype(np.uint8)
        mean_p, obs, count = reliability_table(p, y, bins=10)
        assert mean_p.shape == obs.shape == count.shape == (10,)
        assert int(count.sum()) == 100

    def test_reliability_table_rejects_too_few_bins(self) -> None:
        with pytest.raises(ValueError, match="bins must be >= 2"):
            reliability_table(np.zeros(4), np.zeros(4), bins=1)

    def test_ece_is_zero_for_a_perfectly_calibrated_forecast(self) -> None:
        rng = np.random.default_rng(0)
        p = np.full(20_000, 0.3)
        y = (rng.random(20_000) < 0.3).astype(np.uint8)
        assert expected_calibration_error(p, y, bins=10) < 0.02

    def test_ece_is_large_for_a_miscalibrated_forecast(self) -> None:
        p = np.full(1000, 0.9)
        y = np.zeros(1000, dtype=np.uint8)
        assert expected_calibration_error(p, y, bins=10) == pytest.approx(0.9, abs=0.01)

    def test_ece_is_nan_on_empty_input(self) -> None:
        assert np.isnan(expected_calibration_error(np.zeros(0), np.zeros(0)))

    def test_empirical_baseline_is_the_training_frequency(self, dataset) -> None:
        assert empirical_exceedance_baseline(dataset.labels) == pytest.approx(
            float(dataset.labels.mean())
        )

    def test_empirical_baseline_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            empirical_exceedance_baseline(np.zeros(0))


class TestBaselineComparison:
    def test_learned_model_beats_the_exponential_closure_on_brier(
        self, fitted, heldout
    ) -> None:
        learned = brier_score(fitted.predict(heldout.features).probability, heldout.labels)
        analytic = brier_score(heldout.analytic_configured, heldout.labels)
        assert learned < analytic

    def test_the_exponential_closure_is_miscalibrated(self, heldout) -> None:
        """Published as a measured defect of equation (16), not hidden: the
        memoryless closure of the level-crossing result over-predicts
        P(T > MFD)."""
        predicted = float(heldout.analytic_configured[0])
        observed = float(heldout.labels.mean())
        assert predicted > 1.5 * observed

    def test_state_carries_information(self, fitted, heldout) -> None:
        """AUC above 0.5 on held-out seeds is the only evidence that the state at
        the crossing predicts the fade length at all."""
        auc = roc_auc(fitted.predict(heldout.features).probability, heldout.labels)
        assert auc > 0.55
