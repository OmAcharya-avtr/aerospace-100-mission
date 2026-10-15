"""The learned component: conformal calibration, coverage and its failure modes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from activityplan.features import feature_matrix, realise, sample_requests
from activityplan.predictor import (
    ConformalRegressor,
    DurationEnergyPredictor,
    Interval,
    analytic_predictions,
    coverage,
    mean_interval_width,
)


def _dataset(n: int = 500, seed: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    requests = sample_requests(n, rng)
    realised = [realise(request, rng) for request in requests]
    features = feature_matrix(list(requests))
    durations = np.array([item.duration_s for item in realised], dtype=float)
    energies = np.array([item.energy_wh for item in realised], dtype=float)
    return features, durations, energies


def test_interval_shape_validation_and_width() -> None:
    point = np.array([1.0, 2.0])
    interval = Interval(point=point, lower=point - 1.0, upper=point + 2.0)
    assert interval.width.tolist() == [3.0, 3.0]
    with pytest.raises(ValueError, match="same shape"):
        Interval(point=point, lower=np.array([0.0]), upper=point)


def test_conformal_quantile_index_is_the_documented_rank() -> None:
    # alpha = 0.1, n = 19 calibration points: rank = ceil(20 * 0.9) = 18, so the
    # quantile is the 18th smallest of 19 absolute residuals. With residuals
    # constructed to be 0..18 exactly the quantile must be 17.0.
    regressor = ConformalRegressor(alpha=0.1, n_estimators=5, random_state=0)
    x_train = np.linspace(0.0, 1.0, 40).reshape(-1, 1)
    y_train = np.zeros(40)
    x_cal = np.zeros((19, 1))
    y_cal = np.arange(19, dtype=float)
    regressor.fit(x_train, y_train, x_cal, y_cal)
    assert regressor.n_calibration_ == 19
    assert regressor.level_attainable
    assert regressor.quantile_ == pytest.approx(17.0, abs=1e-9)


def test_conformal_interval_is_infinite_when_the_level_is_unattainable() -> None:
    # alpha = 0.1 needs ceil((n+1) * 0.9) <= n, which fails for n = 5:
    # ceil(5.4) = 6 > 5. The honest answer is an infinite interval.
    regressor = ConformalRegressor(alpha=0.1, n_estimators=5, random_state=0)
    x_train = np.linspace(0.0, 1.0, 20).reshape(-1, 1)
    y_train = np.zeros(20)
    regressor.fit(x_train, y_train, np.zeros((5, 1)), np.arange(5, dtype=float))
    assert not regressor.level_attainable
    assert math.isinf(regressor.quantile_)
    interval = regressor.predict(np.zeros((3, 1)))
    assert np.all(np.isinf(interval.width))


def test_conformal_absolute_mode_attains_marginal_coverage_on_held_out_data() -> None:
    features, durations, _ = _dataset(600, seed=9)
    regressor = ConformalRegressor(alpha=0.1, mode="absolute", n_estimators=60, random_state=1)
    regressor.fit(features[:300], durations[:300], features[300:450], durations[300:450])
    interval = regressor.predict(features[450:])
    measured = coverage(durations[450:], interval)
    # The guarantee is marginal and finite-sample, and the test split has 150
    # points, so the band is wide. Asserted as a band, never as an equality.
    assert 0.80 <= measured <= 1.0
    assert mean_interval_width(interval) > 0.0


def test_normalised_mode_gives_a_narrower_mean_interval_here() -> None:
    features, durations, _ = _dataset(700, seed=12)
    widths = {}
    for mode in ("absolute", "normalised"):
        regressor = ConformalRegressor(
            alpha=0.1, mode=mode, n_estimators=80, random_state=2
        )
        regressor.fit(features[:360], durations[:360], features[360:520], durations[360:520])
        widths[mode] = mean_interval_width(regressor.predict(features[520:]))
    # Measured on this dataset and seed; the normalised variant spends its width
    # where the error is, so the mean is smaller. This is a measurement, not a
    # theorem, and it is pinned so a regression in the scale model is visible.
    assert widths["normalised"] < widths["absolute"]


def test_point_prediction_lies_inside_the_interval() -> None:
    features, durations, _ = _dataset(300, seed=7)
    for mode in ("absolute", "normalised"):
        regressor = ConformalRegressor(alpha=0.2, mode=mode, n_estimators=40, random_state=3)
        regressor.fit(features[:150], durations[:150], features[150:240], durations[150:240])
        interval = regressor.predict(features[240:])
        assert np.all(interval.lower <= interval.point + 1e-12)
        assert np.all(interval.point <= interval.upper + 1e-12)
        assert np.allclose(
            regressor.predict_point(features[240:]), interval.point, rtol=1e-12
        )


def test_conformal_input_validation() -> None:
    with pytest.raises(ValueError, match="alpha"):
        ConformalRegressor(alpha=0.0)
    with pytest.raises(ValueError, match="mode"):
        ConformalRegressor(mode="bayesian")
    with pytest.raises(ValueError, match="beta"):
        ConformalRegressor(beta=0.0)
    regressor = ConformalRegressor(n_estimators=5)
    with pytest.raises(ValueError, match="before fit"):
        regressor.predict(np.zeros((2, 1)))
    with pytest.raises(ValueError, match="2-D"):
        regressor.fit(np.zeros(5), np.zeros(5), np.zeros((2, 1)), np.zeros(2))
    with pytest.raises(ValueError, match="same number of rows"):
        regressor.fit(np.zeros((5, 1)), np.zeros(4), np.zeros((2, 1)), np.zeros(2))
    with pytest.raises(ValueError, match="at least 2 training rows"):
        regressor.fit(np.zeros((1, 1)), np.zeros(1), np.zeros((2, 1)), np.zeros(2))


def test_duration_energy_predictor_splits_and_fits_both_targets() -> None:
    features, durations, energies = _dataset(400, seed=15)
    predictor = DurationEnergyPredictor(
        alpha=0.1, mode="absolute", calibration_fraction=0.4, n_estimators=40, random_state=5
    ).fit(features[:300], durations[:300], energies[:300])
    assert predictor.n_calibration_ == 120
    assert predictor.n_train_ == 180
    duration_interval, energy_interval = predictor.predict_requests(
        sample_requests(25, np.random.default_rng(1))
    )
    assert duration_interval.point.shape == (25,)
    assert energy_interval.point.shape == (25,)


def test_duration_energy_predictor_validation() -> None:
    with pytest.raises(ValueError, match="calibration_fraction"):
        DurationEnergyPredictor(calibration_fraction=0.0)
    features, durations, energies = _dataset(40, seed=1)
    predictor = DurationEnergyPredictor(n_estimators=5, calibration_fraction=0.5)
    with pytest.raises(ValueError, match="same number of rows"):
        predictor.fit(features, durations[:10], energies)
    tiny = DurationEnergyPredictor(n_estimators=5, calibration_fraction=0.99)
    with pytest.raises(ValueError, match="cannot be split"):
        tiny.fit(features[:3], durations[:3], energies[:3])


def test_learned_model_beats_the_analytic_model_on_energy_held_out() -> None:
    """The benchmark against the baseline, on held-out activities.

    It wins because the analytic model omits a heater term the generator
    contains; see activityplan.features. The assertion is a direction, not a
    magnitude, so it survives a reseeding.
    """
    rng = np.random.default_rng(61)
    requests = sample_requests(700, rng)
    realised = [realise(request, rng) for request in requests]
    features = feature_matrix(list(requests))
    energies = np.array([item.energy_wh for item in realised], dtype=float)
    predictor = DurationEnergyPredictor(
        alpha=0.1, mode="normalised", n_estimators=80, random_state=61
    ).fit(
        features[:500],
        np.array([item.duration_s for item in realised[:500]], dtype=float),
        energies[:500],
    )
    learned = predictor.energy.predict_point(features[500:])
    _, analytic = analytic_predictions(requests[500:])
    learned_mae = float(np.mean(np.abs(energies[500:] - learned)))
    analytic_mae = float(np.mean(np.abs(energies[500:] - analytic)))
    assert learned_mae < analytic_mae


def test_coverage_and_width_helpers_validate_shapes() -> None:
    point = np.array([1.0, 2.0, 3.0])
    interval = Interval(point=point, lower=point - 1.0, upper=point + 1.0)
    assert coverage(np.array([1.0, 2.5, 5.0]), interval) == pytest.approx(2.0 / 3.0, rel=1e-13)
    assert mean_interval_width(interval) == pytest.approx(2.0, rel=1e-14)
    with pytest.raises(ValueError, match="does not match"):
        coverage(np.array([1.0]), interval)


def test_coverage_is_inclusive_at_the_interval_edge() -> None:
    point = np.array([0.0])
    interval = Interval(point=point, lower=np.array([-1.0]), upper=np.array([1.0]))
    assert coverage(np.array([1.0]), interval) == pytest.approx(1.0, abs=1e-15)
    assert coverage(np.array([-1.0]), interval) == pytest.approx(1.0, abs=1e-15)
