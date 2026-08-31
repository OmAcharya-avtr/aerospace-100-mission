"""Learned gain scheduler: fitting, uncertainty and the safety clamp."""

from __future__ import annotations

import numpy as np
import pytest

from detumblesim.features import N_FEATURES
from detumblesim.scheduler import GainScheduler


def toy_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, N_FEATURES))
    y = 0.4 * x[:, 5] - 0.2 * x[:, 6]
    return x, y


class TestConstruction:
    def test_rejects_too_few_trees(self):
        with pytest.raises(ValueError, match="n_estimators"):
            GainScheduler(n_estimators=1)

    @pytest.mark.parametrize("kw", [{"max_log_adjust": 0.0}, {"confidence_scale": -1.0}])
    def test_rejects_bad_parameters(self, kw):
        with pytest.raises(ValueError, match="must be positive"):
            GainScheduler(**kw)

    def test_unfitted_raises(self):
        s = GainScheduler()
        assert not s.fitted
        with pytest.raises(ValueError, match="not fitted"):
            s.predict_with_uncertainty(np.zeros((1, N_FEATURES)))
        with pytest.raises(ValueError, match="not fitted"):
            s.feature_importances()


class TestFit:
    def test_fits_and_predicts(self):
        x, y = toy_data()
        s = GainScheduler(n_estimators=40, random_state=0).fit(x, y)
        assert s.fitted
        mean, spread = s.predict_with_uncertainty(x[:10])
        assert mean.shape == (10,) and spread.shape == (10,)
        assert np.all(spread >= 0.0)
        assert float(np.corrcoef(mean, y[:10])[0, 1]) > 0.8

    def test_importances_sum_to_one(self):
        x, y = toy_data()
        s = GainScheduler(n_estimators=30).fit(x, y)
        imp = s.feature_importances()
        assert imp.shape == (N_FEATURES,)
        assert np.isclose(float(imp.sum()), 1.0)

    @pytest.mark.parametrize(
        "x,y,msg",
        [
            (np.zeros((5, 3)), np.zeros(5), "shape"),
            (np.zeros((5, N_FEATURES)), np.zeros(4), "entries"),
            (np.full((5, N_FEATURES), np.nan), np.zeros(5), "finite"),
            (np.zeros((1, N_FEATURES)), np.zeros(1), "at least two"),
        ],
    )
    def test_rejects_bad_training_data(self, x, y, msg):
        with pytest.raises(ValueError, match=msg):
            GainScheduler(n_estimators=10).fit(x, y)

    def test_predict_rejects_wrong_width(self):
        x, y = toy_data()
        s = GainScheduler(n_estimators=10).fit(x, y)
        with pytest.raises(ValueError, match="columns"):
            s.predict_with_uncertainty(np.zeros((2, 3)))

    def test_is_reproducible_for_a_fixed_seed(self):
        x, y = toy_data()
        a = GainScheduler(n_estimators=25, random_state=3).fit(x, y)
        b = GainScheduler(n_estimators=25, random_state=3).fit(x, y)
        ma, _ = a.predict_with_uncertainty(x[:20])
        mb, _ = b.predict_with_uncertainty(x[:20])
        assert np.array_equal(ma, mb)


class TestConfidenceAndClamp:
    def test_confidence_is_one_at_zero_spread(self):
        assert np.isclose(GainScheduler().confidence([0.0])[0], 1.0)

    def test_confidence_halves_at_the_scale(self):
        s = GainScheduler(confidence_scale=0.25)
        assert np.isclose(s.confidence([0.25])[0], 0.5)

    def test_confidence_is_monotone_decreasing(self):
        s = GainScheduler()
        c = s.confidence([0.0, 0.1, 0.5, 2.0])
        assert np.all(np.diff(c) < 0.0)
        assert np.all((c > 0.0) & (c <= 1.0))

    def test_confidence_rejects_negative_spread(self):
        with pytest.raises(ValueError, match="non-negative"):
            GainScheduler().confidence([-0.1])

    def test_gain_clamp_bounds_the_correction(self):
        # A model trained on a huge constant label must still respect the
        # +/- max_log_adjust safety clamp.
        x = np.random.default_rng(0).normal(size=(50, N_FEATURES))
        y = np.full(50, 9.0)
        s = GainScheduler(n_estimators=20, max_log_adjust=0.5).fit(x, y)
        gain, conf = s.predict_gain(x[0], 1.0e5)
        assert gain <= 1.0e5 * 10.0**0.5 + 1e-6
        assert 0.0 < conf <= 1.0

    def test_zero_label_returns_the_base_gain(self):
        x = np.random.default_rng(1).normal(size=(50, N_FEATURES))
        s = GainScheduler(n_estimators=20).fit(x, np.zeros(50))
        gain, conf = s.predict_gain(x[3], 7.0e4)
        assert np.isclose(gain, 7.0e4)
        assert np.isclose(conf, 1.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
    def test_rejects_bad_base_gain(self, bad):
        x, y = toy_data(60)
        s = GainScheduler(n_estimators=10).fit(x, y)
        with pytest.raises(ValueError, match="base_gain"):
            s.predict_gain(x[0], bad)
