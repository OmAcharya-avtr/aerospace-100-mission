"""Learned quantile forecaster."""

from __future__ import annotations

import numpy as np
import pytest

from degradepredict.features import FEATURE_NAMES
from degradepredict.forecast import QuantileForecast
from degradepredict.generate import DEFAULT_LIMIT, generate_population
from degradepredict.learned import DEFAULT_LEVELS, LearnedCrossingForecaster

SMALL = {"n_estimators": 40, "max_depth": 2}


@pytest.fixture(scope="module")
def fitted():
    train = generate_population(220, seed=2026)
    return LearnedCrossingForecaster(limit=DEFAULT_LIMIT, **SMALL).fit(train)


def test_default_levels_bracket_the_ninety_percent_interval():
    assert 0.05 in DEFAULT_LEVELS
    assert 0.95 in DEFAULT_LEVELS
    assert tuple(sorted(DEFAULT_LEVELS)) == tuple(DEFAULT_LEVELS)


def test_fit_uses_only_the_uncensored_units_as_targets(fitted):
    assert fitted.fitted
    assert 0 < fitted.n_train_used_ < fitted.n_train_


def test_predict_returns_a_quantile_forecast_with_the_fitted_levels(fitted):
    unit = generate_population(1, seed=7)[0]
    f = fitted.predict(unit)
    assert isinstance(f, QuantileForecast)
    np.testing.assert_allclose(f.levels, np.asarray(DEFAULT_LEVELS))
    assert np.all(np.diff(f.values_days) >= 0.0)
    assert np.all(f.values_days > 0.0)


def test_predict_many_agrees_with_predict_one_at_a_time(fitted):
    pop = generate_population(5, seed=8)
    batch = fitted.predict_many(pop)
    for tr, f in zip(pop, batch, strict=True):
        np.testing.assert_allclose(f.values_days, fitted.predict(tr).values_days, rtol=1e-12)


def test_refitting_with_the_same_seed_reproduces_the_predictions():
    """Determinism is required to tolerance, not bit-exactly.

    The boosting subsample draws are seeded, so two fits on identical data
    produce the same trees; the predictions are compared with a relative
    tolerance because the arithmetic path through NumPy is not guaranteed to be
    reproduced bit for bit across builds.
    """
    train = generate_population(160, seed=55)
    a = LearnedCrossingForecaster(limit=DEFAULT_LIMIT, **SMALL).fit(train)
    b = LearnedCrossingForecaster(limit=DEFAULT_LIMIT, **SMALL).fit(train)
    unit = generate_population(1, seed=3)[0]
    np.testing.assert_allclose(a.predict(unit).values_days, b.predict(unit).values_days,
                               rtol=1e-10)


def test_ipcw_and_complete_case_fits_differ():
    """The weighting is not a no-op: dropping it changes the forecasts."""
    train = generate_population(220, seed=2026)
    ipcw = LearnedCrossingForecaster(limit=DEFAULT_LIMIT, weighting="ipcw", **SMALL).fit(train)
    plain = LearnedCrossingForecaster(limit=DEFAULT_LIMIT, weighting="none", **SMALL).fit(train)
    unit = generate_population(1, seed=4)[0]
    assert not np.allclose(ipcw.predict(unit).values_days, plain.predict(unit).values_days)


def test_feature_importance_sums_to_one_and_names_every_feature(fitted):
    imp = fitted.feature_importance()
    assert set(imp) == set(FEATURE_NAMES)
    assert sum(imp.values()) == pytest.approx(1.0)


def test_predicting_before_fitting_raises():
    model = LearnedCrossingForecaster(limit=DEFAULT_LIMIT)
    with pytest.raises(ValueError, match="call fit"):
        model.predict_matrix(np.zeros((1, len(FEATURE_NAMES))))
    with pytest.raises(ValueError, match="call fit"):
        model.feature_importance()


def test_predict_matrix_rejects_the_wrong_number_of_features(fitted):
    with pytest.raises(ValueError, match="expected .* features"):
        fitted.predict_matrix(np.zeros((2, 3)))


def test_fit_refuses_a_training_set_with_too_few_uncensored_units():
    train = generate_population(220, seed=2026)[:6]
    with pytest.raises(ValueError, match="uncensored training units"):
        LearnedCrossingForecaster(limit=DEFAULT_LIMIT, **SMALL).fit(train)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"levels": (0.5,)}, "at least two"),
        ({"levels": (0.6, 0.4)}, "strictly increasing"),
        ({"levels": (0.0, 0.5)}, "strictly increasing inside"),
        ({"weighting": "kaplan"}, "weighting must be"),
        ({"subsample": 0.0}, "subsample"),
    ],
)
def test_constructor_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LearnedCrossingForecaster(limit=DEFAULT_LIMIT, **kwargs)
