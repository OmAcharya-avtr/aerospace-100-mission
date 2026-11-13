"""The learned predictor, its uncertainty output and the null model."""

from __future__ import annotations

import numpy as np
import pytest

from deadlinerisk.predictor import (
    ConstantRatePredictor,
    ContentionMissPredictor,
    score,
    target_noise_sd,
)
from deadlinerisk.windows import extract_windows, grouped_splits, time_split


def test_known_answer_target_noise_sd() -> None:
    # HAND CALCULATION. p = 0.25, N = 100 -> sqrt(0.25*0.75/100)
    #   = sqrt(0.001875) = 0.04330127018922193
    value = target_noise_sd(np.asarray([0.25]), np.asarray([100]))
    assert value[0] == pytest.approx(0.04330127018922193, rel=1e-12, abs=0.0)


def test_target_noise_is_zero_at_the_endpoints() -> None:
    value = target_noise_sd(np.asarray([0.0, 1.0]), np.asarray([50, 50]))
    assert value.tolist() == [0.0, 0.0]


def test_target_noise_rejects_a_non_positive_release_count() -> None:
    with pytest.raises(ValueError, match="releases must be positive"):
        target_noise_sd(np.asarray([0.5]), np.asarray([0]))


def test_constant_predictor_returns_the_training_mean() -> None:
    model = ConstantRatePredictor().fit(np.zeros((4, 2)), np.asarray([0.1, 0.2, 0.3, 0.4]))
    prediction = model.predict(np.zeros((3, 2)))
    assert prediction.rate.tolist() == pytest.approx(
        [0.25, 0.25, 0.25], rel=1e-12, abs=0.0
    )
    assert prediction.sigma[0] > 0.0


def test_constant_predictor_refuses_before_fit() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        ConstantRatePredictor().predict(np.zeros((1, 2)))


def test_constant_predictor_refuses_an_empty_target() -> None:
    with pytest.raises(ValueError, match="empty target"):
        ConstantRatePredictor().fit(np.zeros((0, 2)), np.zeros(0))


def test_forest_predicts_within_the_training_target_range(campaign) -> None:
    # A forest averages leaf means, so it cannot extrapolate past the training
    # targets. This is a property of the estimator's implementation, which is
    # why scikit-learn is bounded to its minor version in pyproject.toml.
    windows = extract_windows(campaign)
    model = ContentionMissPredictor().fit(
        windows.features, windows.targets, windows.feature_names
    )
    prediction = model.predict(windows.features)
    assert float(prediction.rate.min()) >= float(windows.targets.min()) - 1e-12
    assert float(prediction.rate.max()) <= float(windows.targets.max()) + 1e-12


def test_forest_uncertainty_has_both_components(campaign) -> None:
    windows = extract_windows(campaign)
    model = ContentionMissPredictor().fit(
        windows.features, windows.targets, windows.feature_names
    )
    prediction = model.predict(windows.features, windows.releases)
    assert np.all(prediction.sigma >= prediction.sigma_model - 1e-15)
    assert np.all(prediction.sigma_target >= 0.0)
    # Quadrature: sigma^2 = sigma_model^2 + sigma_target^2, exact in exact
    # arithmetic and to rounding here.
    assert np.allclose(
        prediction.sigma**2,
        prediction.sigma_model**2 + prediction.sigma_target**2,
        rtol=1e-12,
        atol=0.0,
    )


def test_forest_without_release_counts_reports_only_the_model_component(
    campaign,
) -> None:
    windows = extract_windows(campaign)
    model = ContentionMissPredictor().fit(windows.features, windows.targets)
    prediction = model.predict(windows.features)
    assert prediction.sigma_target.tolist() == [0.0] * len(windows)
    # sigma = sqrt(sigma_model^2 + 0) equals sigma_model in exact arithmetic and
    # to within one rounding here, because squaring and taking the square root
    # are each a rounded operation. A bit-exact assertion would be a claim about
    # those two roundings cancelling, which floating point does not promise.
    assert np.allclose(prediction.sigma, prediction.sigma_model, rtol=1e-12, atol=0.0)


def test_importances_are_named_and_sum_to_one(campaign) -> None:
    windows = extract_windows(campaign)
    model = ContentionMissPredictor().fit(
        windows.features, windows.targets, windows.feature_names
    )
    importances = model.importances()
    assert set(importances) == set(windows.feature_names)
    assert sum(importances.values()) == pytest.approx(1.0, rel=1e-9, abs=0.0)


def test_forest_refuses_before_fit() -> None:
    model = ContentionMissPredictor()
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict(np.zeros((1, 3)))
    with pytest.raises(RuntimeError, match="before fit"):
        model.importances()


@pytest.mark.parametrize(
    ("features", "targets", "pattern"),
    [
        (np.zeros(5), np.zeros(5), "two-dimensional"),
        (np.zeros((5, 2)), np.zeros(4), "does not match"),
        (np.zeros((1, 2)), np.zeros(1), "at least 2 training"),
        (np.zeros((3, 2)), np.asarray([0.1, 1.4, 0.2]), "miss RATES"),
        (np.zeros((3, 2)), np.asarray([0.1, -0.4, 0.2]), "miss RATES"),
    ],
)
def test_forest_fit_input_validation(
    features: np.ndarray, targets: np.ndarray, pattern: str
) -> None:
    with pytest.raises(ValueError, match=pattern):
        ContentionMissPredictor().fit(features, targets)


@pytest.mark.parametrize(("estimators", "depth"), [(1, 6), (300, 0), (300, -2)])
def test_forest_constructor_validation(estimators: int, depth: int) -> None:
    with pytest.raises(ValueError):
        ContentionMissPredictor(n_estimators=estimators, max_depth=depth)


def test_score_is_zero_on_a_perfect_prediction() -> None:
    truth = np.asarray([0.1, 0.2, 0.3])
    result = score("perfect", truth, truth)
    assert result.mae == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert result.rmse == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert result.bias == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert np.isnan(result.coverage)


def test_known_answer_score() -> None:
    # HAND CALCULATION. truth = [0.0, 0.0, 0.0], prediction = [0.1, -0.2, 0.3]
    #   residuals = [0.1, -0.2, 0.3]
    #   MAE  = (0.1 + 0.2 + 0.3)/3 = 0.2
    #   RMSE = sqrt((0.01 + 0.04 + 0.09)/3) = sqrt(0.14/3)
    #        = sqrt(0.04666666666666667) = 0.21602468994692867
    #   bias = (0.1 - 0.2 + 0.3)/3 = 0.06666666666666667
    result = score("hand", np.zeros(3), np.asarray([0.1, -0.2, 0.3]))
    assert result.mae == pytest.approx(0.2, rel=1e-12, abs=0.0)
    assert result.rmse == pytest.approx(0.21602468994692867, rel=1e-12, abs=0.0)
    assert result.bias == pytest.approx(0.06666666666666667, rel=1e-12, abs=0.0)


def test_score_coverage_counts_residuals_inside_one_sigma() -> None:
    # HAND CALCULATION. residuals [0.01, 0.5], sigma [0.1, 0.1]:
    #   |0.01| <= 0.1 -> inside ; |0.5| <= 0.1 -> outside ; coverage = 0.5
    result = score(
        "cover",
        np.zeros(2),
        np.asarray([0.01, 0.5]),
        sigma=np.asarray([0.1, 0.1]),
    )
    assert result.coverage == pytest.approx(0.5, rel=1e-12, abs=0.0)


def test_score_rejects_a_shape_mismatch_and_an_empty_split() -> None:
    with pytest.raises(ValueError, match="does not match"):
        score("bad", np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError, match="empty split"):
        score("bad", np.zeros(0), np.zeros(0))


def test_fit_is_reproducible_at_the_declared_seed(campaign) -> None:
    windows = extract_windows(campaign)
    first = ContentionMissPredictor().fit(windows.features, windows.targets)
    second = ContentionMissPredictor().fit(windows.features, windows.targets)
    # This IS a bit-exact comparison, deliberately: it asserts DETERMINISM at a
    # fixed random_state, which is a property of the estimator and not a
    # floating-point identity over two different pipelines. Two fits of the same
    # data with the same seed execute the same operations in the same order.
    assert np.allclose(
        first.predict(windows.features).rate,
        second.predict(windows.features).rate,
        rtol=0.0,
        atol=0.0,
    )


def test_both_splits_are_scoreable_on_the_fixture(campaign) -> None:
    windows = extract_windows(campaign)
    assert len(grouped_splits(windows)) >= 2
    train, test = time_split(windows)
    model = ContentionMissPredictor().fit(windows.features[train], windows.targets[train])
    result = score(
        "time", windows.targets[test], model.predict(windows.features[test])
    )
    assert result.n_test == int(test.size)
    assert result.mae >= 0.0
