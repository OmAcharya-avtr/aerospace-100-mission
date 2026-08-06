"""Tests for the MLCentroider ensemble: benchmark vs baseline at low SNR,
uncertainty output, reproducibility, input validation.

The trained model is small (5 x MLP(32,), ~1500 training frames) to keep the
test suite fast; the production-scale benchmark lives in validation/.
"""

import numpy as np
import pytest

from centroidnet import MLCentroider, cog_centroid, generate_spots

# Low-SNR regime: S=300 e-, B=2 e-/px, R=3 e- -> SNR ~ 5 (snr_estimate).
GEN_KWARGS = dict(grid_size=16, sigma=1.5, signal=300.0, background=2.0, read_noise=3.0)


@pytest.fixture(scope="module")
def trained():
    imgs, truths = generate_spots(1500, seed=2024, **GEN_KWARGS)
    model = MLCentroider(
        n_estimators=5, hidden_layer_sizes=(32,), max_iter=200, random_state=0
    ).fit(imgs, truths)
    test_imgs, test_truths = generate_spots(300, seed=777, **GEN_KWARGS)  # held out
    return model, test_imgs, test_truths


def _rms(pred, truth):
    return float(np.sqrt(np.mean(np.sum((pred - truth) ** 2, axis=1))))


class TestBenchmark:
    def test_ml_beats_or_ties_plain_cog_at_low_snr(self, trained):
        # Seeded, held-out low-SNR data (SNR ~ 5). Result from this exact
        # configuration during development: ML RMS ~ 0.5 px vs plain CoG
        # RMS ~ 1.2 px (plain CoG is strongly degraded by background/read
        # noise, cf. Thomas et al. 2006). Honest note: a well-tuned
        # *thresholded* CoG is competitive with the ML model except at the
        # lowest SNR -- see validation/VALIDATION.md for the full comparison.
        model, test_imgs, test_truths = trained
        pred = model.predict(test_imgs)
        rms_ml = _rms(pred, test_truths)
        rms_cog = _rms(np.array([cog_centroid(im) for im in test_imgs]), test_truths)
        assert rms_ml <= rms_cog * 1.02, (
            f"ML RMS {rms_ml:.3f} px worse than plain CoG {rms_cog:.3f} px"
        )

    def test_ml_absolute_accuracy_reasonable(self, trained):
        model, test_imgs, test_truths = trained
        rms = _rms(model.predict(test_imgs), test_truths)
        assert rms < 1.0, f"ML RMS {rms:.3f} px unexpectedly large at SNR ~ 5"


class TestUncertainty:
    def test_return_std_shapes_and_positivity(self, trained):
        model, test_imgs, _ = trained
        mean, std = model.predict(test_imgs, return_std=True)
        assert mean.shape == (len(test_imgs), 2)
        assert std.shape == (len(test_imgs), 2)
        assert np.all(std >= 0.0)
        assert np.all(np.isfinite(std))
        # Ensemble members differ, so the spread cannot be identically zero.
        assert std.mean() > 0.0

    def test_single_image_accepted(self, trained):
        model, test_imgs, _ = trained
        mean, std = model.predict(test_imgs[0], return_std=True)
        assert mean.shape == (1, 2)
        assert std.shape == (1, 2)


class TestReproducibility:
    def test_same_seed_same_predictions(self, trained):
        # Refit with the identical seed and identical seeded data: predictions
        # must match the fixture model exactly (deterministic pipeline).
        model, test_imgs, _ = trained
        imgs, truths = generate_spots(1500, seed=2024, **GEN_KWARGS)
        model2 = MLCentroider(
            n_estimators=5, hidden_layer_sizes=(32,), max_iter=200, random_state=0
        ).fit(imgs, truths)
        a = model.predict(test_imgs)
        b = model2.predict(test_imgs)
        assert np.allclose(a, b, atol=1e-10)


class TestInputValidation:
    def test_predict_before_fit(self):
        with pytest.raises(RuntimeError):
            MLCentroider().predict(np.ones((1, 16, 16)))

    def test_bad_constructor(self):
        with pytest.raises(ValueError):
            MLCentroider(n_estimators=1)

    def test_fit_shape_mismatch(self):
        imgs = np.ones((10, 16, 16))
        with pytest.raises(ValueError):
            MLCentroider().fit(imgs, np.zeros((9, 2)))
        with pytest.raises(ValueError):
            MLCentroider().fit(imgs, np.zeros((10, 3)))
        with pytest.raises(ValueError):
            MLCentroider().fit(np.ones((16, 16)), np.zeros((1, 2)))

    def test_predict_wrong_image_shape(self, trained):
        model, _, _ = trained
        with pytest.raises(ValueError):
            model.predict(np.ones((2, 8, 8)))  # shape differs from fitted 16x16

    def test_zero_flux_image_rejected(self, trained):
        model, _, _ = trained
        with pytest.raises(ValueError):
            model.predict(np.zeros((1, 16, 16)))

    def test_nonfinite_rejected(self, trained):
        model, _, _ = trained
        bad = np.ones((1, 16, 16))
        bad[0, 0, 0] = np.inf
        with pytest.raises(ValueError):
            model.predict(bad)
