"""Learned slope estimator: API, uncertainty output, reproducibility, benchmark.

The model trained here is deliberately small (2 members, 1500 stamps, one
hidden layer of 48 units, flux 30-300 e-) so the suite stays fast. It is NOT
the configuration characterized in validation/VALIDATION.md — see
`validation/run_validation.py` for that. The benchmark regression test below
pins the *small* model's behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from shacksim import (
    LensletArray,
    MLSlopeEstimator,
    cog_displacement,
    generate_subaperture_dataset,
    simulate_frame,
    tilt_slopes,
)

ARRAY = LensletArray()
BACKGROUND = 1.0
READ_NOISE = 3.0
TRAIN_SEED = 100
TEST_SEED = 9030


@pytest.fixture(scope="module")
def trained_model() -> MLSlopeEstimator:
    stamps, slopes = generate_subaperture_dataset(
        ARRAY, 1500, photons=(30.0, 300.0), background=BACKGROUND, read_noise=READ_NOISE,
        elongation=(1.0, 3.0), seed=TRAIN_SEED,
    )
    return MLSlopeEstimator(
        ARRAY, n_estimators=2, hidden_layer_sizes=(48,), max_iter=400, random_state=0
    ).fit(stamps, slopes)


@pytest.fixture(scope="module")
def test_set() -> tuple[np.ndarray, np.ndarray]:
    return generate_subaperture_dataset(
        ARRAY, 600, photons=30.0, background=BACKGROUND, read_noise=READ_NOISE,
        seed=TEST_SEED,
    )


class TestApi:
    def test_predict_shape(self, trained_model, test_set):
        stamps, slopes = test_set
        pred = trained_model.predict(stamps)
        assert pred.shape == slopes.shape
        assert np.all(np.isfinite(pred))

    def test_single_stamp_accepted(self, trained_model, test_set):
        stamps, _ = test_set
        assert trained_model.predict(stamps[0]).shape == (1, 2)

    def test_uncertainty_output(self, trained_model, test_set):
        stamps, _ = test_set
        pred, std = trained_model.predict(stamps, return_std=True)
        assert pred.shape == std.shape == (len(stamps), 2)
        assert np.all(std >= 0.0)
        assert std.mean() > 0.0

    def test_predict_frame(self, trained_model):
        truth = tilt_slopes(ARRAY, 5e-4, -5e-4)
        frame = simulate_frame(
            ARRAY, truth, photons=100.0, background=BACKGROUND, read_noise=READ_NOISE, seed=3
        )
        pred, std = trained_model.predict_frame(frame, return_std=True)
        assert pred.shape == (ARRAY.n_valid, 2)
        assert std.shape == (ARRAY.n_valid, 2)

    def test_features_shape_and_normalization(self, trained_model, test_set):
        stamps, _ = test_set
        feats = trained_model.features(stamps)
        p = ARRAY.pixels_per_sub
        assert feats.shape == (len(stamps), p * p + 1)
        # the shape block sums to 1 for every stamp with positive total counts
        assert np.allclose(feats[:, : p * p].sum(axis=1), 1.0)

    def test_features_handle_an_all_negative_stamp(self, trained_model):
        feats = trained_model.features(-np.ones((ARRAY.pixels_per_sub, ARRAY.pixels_per_sub)))
        assert np.all(feats == 0.0)

    def test_fitted_flag(self):
        model = MLSlopeEstimator(ARRAY, n_estimators=2)
        assert not model.fitted_


class TestReproducibility:
    def test_same_seed_gives_identical_predictions(self, test_set):
        stamps, slopes = generate_subaperture_dataset(
            ARRAY, 400, photons=(30.0, 300.0), background=BACKGROUND,
            read_noise=READ_NOISE, seed=55,
        )
        kw = {"n_estimators": 2, "hidden_layer_sizes": (16,), "max_iter": 60}
        a = MLSlopeEstimator(ARRAY, random_state=1, **kw).fit(stamps, slopes)
        b = MLSlopeEstimator(ARRAY, random_state=1, **kw).fit(stamps, slopes)
        c = MLSlopeEstimator(ARRAY, random_state=2, **kw).fit(stamps, slopes)
        x, _ = test_set
        assert np.array_equal(a.predict(x), b.predict(x))
        assert not np.array_equal(a.predict(x), c.predict(x))

    def test_dataset_seed_reproducibility(self):
        a = generate_subaperture_dataset(ARRAY, 30, photons=(30.0, 300.0), seed=7)
        b = generate_subaperture_dataset(ARRAY, 30, photons=(30.0, 300.0), seed=7)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


class TestValidation:
    def test_predict_before_fit_raises(self):
        model = MLSlopeEstimator(ARRAY, n_estimators=2)
        with pytest.raises(RuntimeError, match="before fit"):
            model.predict(np.zeros((1, ARRAY.pixels_per_sub, ARRAY.pixels_per_sub)))

    def test_too_few_estimators(self):
        with pytest.raises(ValueError, match=">= 2"):
            MLSlopeEstimator(ARRAY, n_estimators=1)

    def test_bad_estimator_type(self):
        with pytest.raises(TypeError):
            MLSlopeEstimator(ARRAY, n_estimators=2.5)

    def test_bad_array_type(self):
        with pytest.raises(TypeError, match="LensletArray"):
            MLSlopeEstimator("not-an-array")

    def test_wrong_stamp_size_rejected(self, trained_model):
        with pytest.raises(ValueError, match="shape"):
            trained_model.predict(np.zeros((3, 8, 8)))

    def test_nonfinite_stamps_rejected(self, trained_model):
        bad = np.full((2, ARRAY.pixels_per_sub, ARRAY.pixels_per_sub), np.nan)
        with pytest.raises(ValueError, match="non-finite"):
            trained_model.predict(bad)

    def test_mismatched_label_count(self):
        stamps, _ = generate_subaperture_dataset(ARRAY, 20, photons=100.0, seed=1)
        with pytest.raises(ValueError, match="disagree"):
            MLSlopeEstimator(ARRAY, n_estimators=2).fit(stamps, np.zeros((5, 2)))

    def test_bad_label_shape(self):
        stamps, _ = generate_subaperture_dataset(ARRAY, 20, photons=100.0, seed=1)
        with pytest.raises(ValueError, match="shape"):
            MLSlopeEstimator(ARRAY, n_estimators=2).fit(stamps, np.zeros((20, 3)))

    def test_too_few_training_samples(self):
        stamps, slopes = generate_subaperture_dataset(ARRAY, 5, photons=100.0, seed=1)
        with pytest.raises(ValueError, match="at least 10"):
            MLSlopeEstimator(ARRAY, n_estimators=2).fit(stamps, slopes)

    def test_nonfinite_labels_rejected(self):
        stamps, slopes = generate_subaperture_dataset(ARRAY, 20, photons=100.0, seed=1)
        slopes = slopes.copy()
        slopes[0, 0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            MLSlopeEstimator(ARRAY, n_estimators=2).fit(stamps, slopes)


class TestBenchmarkRegression:
    """Pinned benchmark: the learned estimator must keep beating the thresholded
    CoG in the low-flux regime it was trained for, and must keep losing outside it.

    Values pinned from a seeded run in this build session (numpy 2.4.4,
    scikit-learn 1.8.0). Tolerances are wide enough to absorb BLAS-level
    numerical drift but narrow enough to catch a genuine regression.
    """

    def _rms_px(self, est_slopes: np.ndarray, true_slopes: np.ndarray) -> float:
        d = ARRAY.slope_to_displacement(est_slopes - true_slopes)
        return float(np.sqrt(np.mean(d**2)))

    def test_ml_beats_thresholded_cog_at_low_flux(self, trained_model, test_set):
        stamps, slopes = test_set
        d_true = ARRAY.slope_to_displacement(slopes)
        cog_rms = float(np.sqrt(np.mean((cog_displacement(stamps, threshold=7.0) - d_true) ** 2)))
        ml_rms = self._rms_px(trained_model.predict(stamps), slopes)
        # measured in this session: CoG 2.744 px, ML 2.266 px, ratio 0.826
        assert cog_rms == pytest.approx(2.744, rel=0.05)
        assert ml_rms == pytest.approx(2.266, rel=0.15)
        assert ml_rms / cog_rms < 0.95

    def test_classical_wins_far_outside_the_training_flux(self, trained_model):
        stamps, slopes = generate_subaperture_dataset(
            ARRAY, 400, photons=3000.0, background=BACKGROUND, read_noise=READ_NOISE,
            seed=9300,
        )
        d_true = ARRAY.slope_to_displacement(slopes)
        cog_rms = float(np.sqrt(np.mean((cog_displacement(stamps, threshold=10.0) - d_true) ** 2)))
        ml_rms = self._rms_px(trained_model.predict(stamps), slopes)
        # The model was trained on 30-300 e- only; at 3000 e- it extrapolates
        # and is far worse than the analytic estimator. This is a documented
        # failure mode, pinned so it cannot be silently "fixed" by luck.
        assert cog_rms < 0.05
        assert ml_rms > 10 * cog_rms

    def test_uncertainty_is_not_calibrated(self, trained_model, test_set):
        stamps, slopes = test_set
        pred, std = trained_model.predict(stamps, return_std=True)
        err = np.abs(ARRAY.slope_to_displacement(pred - slopes))
        spread = ARRAY.slope_to_displacement(std)
        ratio = float(spread.mean() / err.mean())
        # measured ~0.05: the ensemble spread massively under-states the error
        assert ratio < 0.5, f"ensemble spread/error ratio {ratio:.3f} — update MODEL_CARD"
