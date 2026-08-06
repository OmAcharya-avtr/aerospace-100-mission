"""Tests for the ML fade-probability surrogate (beamtwin.surrogate)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from beamtwin.budget import LinkParams
from beamtwin.channel import ChannelParams
from beamtwin.surrogate import (
    FEATURE_NAMES,
    P_FLOOR,
    FadeSurrogate,
    default_model_path,
    features_from_params,
    generate_dataset,
    in_training_domain,
)

MODEL_AVAILABLE = default_model_path().exists()


class TestFeatures:
    def test_feature_vector_length(self):
        x = features_from_params(LinkParams(), ChannelParams())
        assert x.shape == (len(FEATURE_NAMES),)

    def test_features_are_finite(self):
        x = features_from_params(LinkParams(range_m=12_000.0), ChannelParams(cn2=1e-15))
        assert np.all(np.isfinite(x))

    def test_log_range_feature(self):
        x = features_from_params(LinkParams(range_m=10_000.0), ChannelParams())
        assert x[0] == pytest.approx(4.0)

    def test_log_cn2_feature(self):
        x = features_from_params(LinkParams(), ChannelParams(cn2=1e-15))
        assert x[1] == pytest.approx(-15.0)

    def test_zero_cn2_is_floored_not_infinite(self):
        x = features_from_params(LinkParams(), ChannelParams(cn2=0.0))
        assert math.isfinite(x[1]) and x[1] == pytest.approx(-18.0)

    def test_jitter_ratio_feature(self):
        # jitter_ratio = jitter / divergence half-angle.
        link = LinkParams(beam_waist_radius_m=0.02, wavelength_m=1550e-9)
        theta = 1550e-9 / (math.pi * 0.02)
        x = features_from_params(link, ChannelParams(pointing_jitter_rad=0.5 * theta))
        assert x[2] == pytest.approx(0.5)

    def test_in_training_domain_true_for_typical(self):
        x = features_from_params(
            LinkParams(range_m=10_000.0, attenuation_db_per_km=2.5, rx_sensitivity_dbm=-30.0),
            ChannelParams(cn2=5e-16, pointing_jitter_rad=5e-6),
        )
        assert in_training_domain(x) is True

    def test_in_training_domain_false_for_short_range(self):
        x = features_from_params(LinkParams(range_m=100.0), ChannelParams())
        assert in_training_domain(x) is False

    def test_in_training_domain_rejects_wrong_length(self):
        with pytest.raises(ValueError):
            in_training_domain(np.zeros(3))


class TestDatasetGeneration:
    def test_small_dataset_shapes(self):
        x, y = generate_dataset(n_scenarios=12, seed=1, mc_samples=800)
        assert x.shape == (12, len(FEATURE_NAMES))
        assert y.shape == (12,)

    def test_targets_are_log10_probabilities(self):
        _, y = generate_dataset(n_scenarios=12, seed=1, mc_samples=800)
        assert np.all(y <= 0.0)
        assert np.all(y >= math.log10(P_FLOOR) - 1e-12)

    def test_deterministic_for_fixed_seed(self):
        a = generate_dataset(n_scenarios=8, seed=5, mc_samples=500)
        b = generate_dataset(n_scenarios=8, seed=5, mc_samples=500)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    def test_different_seed_gives_different_data(self):
        a = generate_dataset(n_scenarios=8, seed=5, mc_samples=500)
        b = generate_dataset(n_scenarios=8, seed=6, mc_samples=500)
        assert not np.array_equal(a[0], b[0])

    def test_rejects_zero_scenarios(self):
        with pytest.raises(ValueError):
            generate_dataset(n_scenarios=0)


class TestSurrogateFitting:
    @pytest.fixture(scope="class")
    @staticmethod
    def trained():
        x, y = generate_dataset(n_scenarios=120, seed=3, mc_samples=3000)
        return FadeSurrogate(n_members=3, random_state=1).fit(x, y), x, y

    def test_requires_at_least_two_members(self):
        with pytest.raises(ValueError, match="n_members"):
            FadeSurrogate(n_members=1)

    def test_unfitted_reports_not_fitted(self):
        assert FadeSurrogate().is_fitted is False

    def test_unfitted_predict_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            FadeSurrogate().predict_log10(np.zeros((1, len(FEATURE_NAMES))))

    def test_unfitted_save_raises(self, tmp_path):
        with pytest.raises(RuntimeError):
            FadeSurrogate().save(tmp_path / "m.joblib")

    def test_fit_sets_is_fitted(self, trained):
        model, _, _ = trained
        assert model.is_fitted is True

    def test_fit_rejects_wrong_feature_count(self):
        with pytest.raises(ValueError, match="shape"):
            FadeSurrogate().fit(np.zeros((10, 3)), np.zeros(10))

    def test_fit_rejects_mismatched_y(self):
        with pytest.raises(ValueError, match="shape"):
            FadeSurrogate().fit(np.zeros((10, len(FEATURE_NAMES))), np.zeros(9))

    def test_predict_log10_shapes(self, trained):
        model, x, _ = trained
        mean, std = model.predict_log10(x[:5])
        assert mean.shape == (5,) and std.shape == (5,)

    def test_predict_accepts_single_row(self, trained):
        model, x, _ = trained
        mean, std = model.predict_log10(x[0])
        assert mean.shape == (1,) and std.shape == (1,)

    def test_predict_rejects_wrong_width(self, trained):
        model, _, _ = trained
        with pytest.raises(ValueError, match="columns"):
            model.predict_log10(np.zeros((2, 3)))

    def test_ensemble_std_non_negative(self, trained):
        model, x, _ = trained
        _, std = model.predict_log10(x[:20])
        assert np.all(std >= 0.0)

    def test_training_fit_is_better_than_predicting_the_mean(self, trained):
        model, x, y = trained
        mean, _ = model.predict_log10(x)
        assert np.mean(np.abs(mean - y)) < np.mean(np.abs(y - y.mean()))

    def test_prediction_object_fields(self, trained):
        model, _, _ = trained
        pred = model.predict(
            LinkParams(range_m=10_000.0, attenuation_db_per_km=2.5, rx_sensitivity_dbm=-30.0),
            ChannelParams(cn2=5e-16, pointing_jitter_rad=5e-6),
        )
        assert P_FLOOR <= pred.probability <= 1.0
        assert pred.p_low <= pred.probability <= pred.p_high
        assert pred.log10_std >= 0.0
        assert isinstance(pred.extrapolating, bool)

    def test_extrapolation_flag_set_outside_domain(self, trained):
        model, _, _ = trained
        pred = model.predict(LinkParams(range_m=100.0), ChannelParams())
        assert pred.extrapolating is True

    def test_fit_is_deterministic(self):
        x, y = generate_dataset(n_scenarios=60, seed=2, mc_samples=1500)
        a = FadeSurrogate(n_members=2, random_state=4).fit(x, y).predict_log10(x[:10])[0]
        b = FadeSurrogate(n_members=2, random_state=4).fit(x, y).predict_log10(x[:10])[0]
        assert np.allclose(a, b)

    def test_save_and_load_roundtrip(self, trained, tmp_path):
        model, x, _ = trained
        path = tmp_path / "m.joblib"
        model.save(path)
        loaded = FadeSurrogate.load(path)
        assert loaded.is_fitted
        assert np.allclose(model.predict_log10(x[:10])[0], loaded.predict_log10(x[:10])[0])

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="train_surrogate"):
            FadeSurrogate.load(tmp_path / "absent.joblib")


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="committed surrogate model not present")
class TestCommittedModel:
    @pytest.fixture(scope="class")
    @staticmethod
    def model():
        return FadeSurrogate.load(default_model_path())

    def test_loads(self, model):
        assert model.is_fitted

    def test_predicts_reference_case_within_an_order_of_magnitude(self, model):
        # Reference 10 km case: Monte Carlo truth ~1.08e-3 (examples output).
        pred = model.predict(
            LinkParams(range_m=10_000.0, attenuation_db_per_km=2.5, rx_sensitivity_dbm=-30.0),
            ChannelParams(cn2=5e-16, pointing_jitter_rad=5e-6),
        )
        assert 1e-4 <= pred.probability <= 1e-2

    def test_monotone_increasing_with_range(self, model):
        probs = [
            model.predict(
                LinkParams(range_m=r, attenuation_db_per_km=2.5, rx_sensitivity_dbm=-30.0),
                ChannelParams(cn2=5e-16, pointing_jitter_rad=5e-6),
            ).probability
            for r in (9000.0, 11_000.0, 13_000.0)
        ]
        assert all(b >= a for a, b in zip(probs, probs[1:]))

    def test_uncertainty_band_is_ordered(self, model):
        pred = model.predict(
            LinkParams(range_m=12_000.0, attenuation_db_per_km=2.0, rx_sensitivity_dbm=-32.0),
            ChannelParams(cn2=1e-15, pointing_jitter_rad=6e-6),
        )
        assert pred.p_low <= pred.probability <= pred.p_high
