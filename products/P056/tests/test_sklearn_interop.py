"""Cross-checks against scikit-learn, including the API change in 1.9.

``sklearn`` is already a runtime dependency of this package (isotonic
regression comes from it), so these are real agreement checks against a
mature implementation, not an optional extra.
"""

from __future__ import annotations

import numpy as np
import pytest
import sklearn
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.exceptions import NotFittedError
from sklearn.frozen import FrozenEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.utils._param_validation import InvalidParameterError

from calibaudit.recalibration import IsotonicCalibration, PlattScaling
from calibaudit.reliability import reliability_curve
from calibaudit.synthetic import get_spec, sample_forecast


class TestPrefitRemoval:
    """``cv="prefit"`` was removed; this pins the current behaviour."""

    def test_sklearn_version_is_at_least_1_9(self):
        major, minor = (int(p) for p in sklearn.__version__.split(".")[:2])
        assert (major, minor) >= (1, 9)

    def test_cv_prefit_is_rejected(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 2))
        y = (x[:, 0] + rng.normal(scale=0.5, size=200) > 0).astype(int)
        base = LogisticRegression().fit(x, y)
        with pytest.raises(InvalidParameterError, match="'cv' parameter"):
            CalibratedClassifierCV(base, cv="prefit", method="sigmoid").fit(x, y)

    def test_frozen_estimator_is_the_replacement(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(200, 2))
        y = (x[:, 0] + rng.normal(scale=0.5, size=200) > 0).astype(int)
        base = LogisticRegression().fit(x, y)
        model = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(x, y)
        p = model.predict_proba(x)[:, 1]
        assert np.all((p >= 0.0) & (p <= 1.0))

    def test_frozen_estimator_requires_a_fitted_base(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(60, 2))
        y = (x[:, 0] > 0).astype(int)
        with pytest.raises(NotFittedError):
            CalibratedClassifierCV(
                FrozenEstimator(LogisticRegression()), method="sigmoid"
            ).fit(x, y)


class TestAgreementOnTheCalibrationMaps:
    def test_platt_agrees_with_an_unregularised_logistic_regression_on_the_logit(self):
        # Platt scaling is logistic regression on a single feature, logit(f).
        # sklearn's LogisticRegression regularises by default, so C must be
        # large for the comparison to be meaningful; this is why the package
        # fits the two parameters directly instead of delegating.
        s = sample_forecast(get_spec("overconfident"), 20000, seed=56900)
        z = np.log(s.forecasts / (1.0 - s.forecasts)).reshape(-1, 1)
        theirs = LogisticRegression(C=1e8, tol=1e-10, max_iter=2000).fit(z, s.outcomes)
        ours = PlattScaling().fit(s.forecasts, s.outcomes)
        assert ours.a_ == pytest.approx(float(theirs.coef_[0, 0]), rel=2e-3)
        assert ours.b_ == pytest.approx(float(theirs.intercept_[0]), abs=2e-3)

    def test_isotonic_predictions_match_a_direct_sklearn_fit(self):
        s = sample_forecast(get_spec("overconfident"), 4000, seed=56901)
        ours = IsotonicCalibration().fit(s.forecasts, s.outcomes)
        theirs = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        ).fit(s.forecasts, s.outcomes)
        grid = np.linspace(0.0, 1.0, 301)
        assert np.allclose(ours.predict(grid), theirs.predict(grid), atol=0)


class TestAgreementOnTheReliabilityCurve:
    def test_uniform_strategy_matches_sklearn_calibration_curve(self):
        # sklearn returns (prob_true, prob_pred) per non-empty bin, which are
        # exactly this package's observed_frequency and mean_forecast.
        s = sample_forecast(get_spec("overconfident"), 4000, seed=56902)
        prob_true, prob_pred = calibration_curve(
            s.outcomes, s.forecasts, n_bins=10, strategy="uniform"
        )
        ours = reliability_curve(s.forecasts, s.outcomes, n_bins=10, strategy="equal_width")
        assert ours.n_occupied == prob_true.size
        assert np.allclose(ours.observed_frequency, prob_true, atol=1e-12)
        assert np.allclose(ours.mean_forecast, prob_pred, atol=1e-12)

    def test_quantile_strategy_bin_count_can_differ_from_sklearn(self):
        # sklearn's "quantile" strategy and this package's "equal_mass" both
        # put edges at forecast quantiles, but sklearn drops duplicate edges
        # while this package keeps the requested bin count and lets bins be
        # empty. The observed frequencies therefore need not align one to one,
        # and the README does not claim they do.
        s = sample_forecast(get_spec("calibrated_rare"), 4000, seed=56903)
        prob_true, _ = calibration_curve(
            s.outcomes, s.forecasts, n_bins=10, strategy="quantile"
        )
        ours = reliability_curve(s.forecasts, s.outcomes, n_bins=10, strategy="equal_mass")
        assert prob_true.size <= 10
        assert ours.n_occupied <= 10
