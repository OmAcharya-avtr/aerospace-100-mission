"""The learned estimator and the trivial baseline, including the overconfidence check."""

from __future__ import annotations

import numpy as np
import pytest

from aligncal import dataset, features, learned, observability


@pytest.fixture(scope="module")
def population():
    return dataset.training_population(count=240, samples=120, seed=31)


@pytest.fixture(scope="module")
def training(population):
    return features.build_population(population)


@pytest.fixture(scope="module")
def forest(training):
    matrix, targets, _ = training
    return learned.LearnedEstimator(seed=20261029, **learned.CONFIGURATIONS["forest"]).fit(
        matrix, targets
    )


@pytest.fixture(scope="module")
def ridge(training):
    matrix, targets, _ = training
    return learned.LearnedEstimator(
        seed=20261029, **learned.CONFIGURATIONS["bagged_ridge"]
    ).fit(matrix, targets)


def test_configurations_are_the_declared_three():
    assert sorted(learned.CONFIGURATIONS) == ["bagged_ridge", "forest", "forest_deep"]


def test_prediction_shapes(forest, training):
    matrix, _, _ = training
    prediction = forest.predict(matrix[:7])
    assert prediction.theta_rad.shape == (7, 3)
    assert prediction.sigma_rad.shape == (7, 3)
    assert prediction.members.shape == (200, 7, 3)


def test_ridge_member_count(ridge, training):
    matrix, _, _ = training
    assert ridge.predict(matrix[:4]).members.shape[0] == 40


def test_point_estimate_is_the_member_mean(forest, training):
    matrix, _, _ = training
    prediction = forest.predict(matrix[:5])
    assert np.max(np.abs(prediction.members.mean(axis=0) - prediction.theta_rad)) < 1.0e-18


def test_sigma_is_the_member_spread(forest, training):
    matrix, _, _ = training
    prediction = forest.predict(matrix[:5])
    assert np.max(np.abs(prediction.members.std(axis=0) - prediction.sigma_rad)) < 1.0e-18


def test_sigma_in_direction_matches_a_canonical_axis(forest, training):
    matrix, _, _ = training
    prediction = forest.predict(matrix[:6])
    axis = np.array([1.0, 0.0, 0.0])
    assert np.max(
        np.abs(prediction.sigma_in_direction(axis) - prediction.sigma_rad[:, 0])
    ) < 1.0e-18


def test_component_in_direction_matches_a_canonical_axis(forest, training):
    matrix, _, _ = training
    prediction = forest.predict(matrix[:6])
    axis = np.array([0.0, 0.0, 1.0])
    assert np.max(
        np.abs(prediction.component_in_direction(axis) - prediction.theta_rad[:, 2])
    ) < 1.0e-18


def test_sigma_in_direction_accepts_a_direction_per_row(forest, training):
    matrix, _, _ = training
    prediction = forest.predict(matrix[:4])
    directions = np.eye(3)[[0, 1, 2, 0]]
    sigmas = prediction.sigma_in_direction(directions)
    assert sigmas.shape == (4,)
    assert sigmas[0] == pytest.approx(prediction.sigma_rad[0, 0], rel=1.0e-12, abs=0.0)


def test_sigma_in_direction_normalises(forest, training):
    matrix, _, _ = training
    prediction = forest.predict(matrix[:3])
    unit = prediction.sigma_in_direction(np.array([0.0, 1.0, 0.0]))
    scaled = prediction.sigma_in_direction(np.array([0.0, 7.0, 0.0]))
    assert np.max(np.abs(unit - scaled)) < 1.0e-20


def test_sigma_in_direction_rejects_bad_directions(forest, training):
    matrix, _, _ = training
    prediction = forest.predict(matrix[:3])
    with pytest.raises(ValueError, match="3 components"):
        prediction.sigma_in_direction(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="non-zero"):
        prediction.sigma_in_direction(np.zeros(3))
    with pytest.raises(ValueError, match="directions for"):
        prediction.sigma_in_direction(np.eye(3)[[0, 1]])


def test_component_in_direction_rejects_zero(forest, training):
    matrix, _, _ = training
    with pytest.raises(ValueError, match="non-zero"):
        forest.predict(matrix[:2]).component_in_direction(np.zeros(3))


def test_learned_beats_the_trivial_baseline_on_held_out_records(training):
    """Measured, not assumed. If this ever fails the learned component has learned nothing."""
    matrix, targets, _ = training
    model = learned.LearnedEstimator(seed=20261029, **learned.CONFIGURATIONS["forest"]).fit(
        matrix, targets
    )
    trivial = learned.TrivialEstimator().fit(matrix, targets)
    held_out = dataset.training_population(count=120, samples=120, seed=98765)
    test_matrix, test_targets, _ = features.build_population(held_out)
    learned_rmse = float(
        np.sqrt(np.mean((model.predict(test_matrix).theta_rad - test_targets) ** 2))
    )
    trivial_rmse = float(
        np.sqrt(np.mean((trivial.predict(test_matrix).theta_rad - test_targets) ** 2))
    )
    assert learned_rmse < trivial_rmse


def test_fit_rejects_bad_shapes(training):
    matrix, targets, _ = training
    model = learned.LearnedEstimator()
    with pytest.raises(ValueError, match=r"targets \(m, 3\)"):
        model.fit(matrix, targets[:, :2])
    with pytest.raises(ValueError, match="feature rows"):
        model.fit(matrix[:10], targets)
    with pytest.raises(ValueError, match="at least 4 records"):
        model.fit(matrix[:3], targets[:3])


def test_fit_rejects_non_finite(training):
    matrix, targets, _ = training
    broken = matrix.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        learned.LearnedEstimator().fit(broken, targets)


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown kind"):
        learned.LearnedEstimator(kind="transformer")


def test_predict_before_fit():
    with pytest.raises(RuntimeError, match="before fit"):
        learned.LearnedEstimator().predict(np.zeros((1, 24)))


def test_predict_rejects_wrong_feature_count(forest):
    with pytest.raises(ValueError, match="fitted on"):
        forest.predict(np.zeros((1, 5)))


def test_predict_rejects_non_finite(forest):
    with pytest.raises(ValueError, match="finite"):
        forest.predict(np.full((1, 24), np.nan))


def test_fit_is_deterministic_under_a_fixed_seed(training):
    matrix, targets, _ = training
    first = learned.LearnedEstimator(seed=7, **learned.CONFIGURATIONS["bagged_ridge"]).fit(
        matrix, targets
    )
    second = learned.LearnedEstimator(seed=7, **learned.CONFIGURATIONS["bagged_ridge"]).fit(
        matrix, targets
    )
    assert np.array_equal(first.predict(matrix[:5]).theta_rad, second.predict(matrix[:5]).theta_rad)


def test_different_seeds_give_different_ensembles(training):
    matrix, targets, _ = training
    first = learned.LearnedEstimator(seed=1, **learned.CONFIGURATIONS["bagged_ridge"]).fit(
        matrix, targets
    )
    second = learned.LearnedEstimator(seed=2, **learned.CONFIGURATIONS["bagged_ridge"]).fit(
        matrix, targets
    )
    assert not np.array_equal(
        first.predict(matrix[:5]).theta_rad, second.predict(matrix[:5]).theta_rad
    )


# ---------------------------------------------------------------------------
# The trivial baseline.
# ---------------------------------------------------------------------------


def test_trivial_predicts_the_training_mean(training):
    matrix, targets, _ = training
    trivial = learned.TrivialEstimator().fit(matrix, targets)
    prediction = trivial.predict(matrix[:9])
    assert np.max(np.abs(prediction.theta_rad - targets.mean(axis=0))) < 1.0e-18
    assert prediction.theta_rad.shape == (9, 3)


def test_trivial_mean_is_near_zero_by_construction(training):
    """A misalignment direction uniform on the sphere has a mean within rounding of zero."""
    matrix, targets, _ = training
    trivial = learned.TrivialEstimator().fit(matrix, targets)
    assert float(np.linalg.norm(trivial.mean_rad)) < 0.4 * float(np.std(targets))


def test_trivial_sigma_is_the_training_spread(training):
    matrix, targets, _ = training
    trivial = learned.TrivialEstimator().fit(matrix, targets)
    prediction = trivial.predict(matrix[:4])
    assert np.max(np.abs(prediction.sigma_rad[0] - targets.std(axis=0, ddof=1))) < 1.0e-18


def test_trivial_spread_in_an_arbitrary_direction(training):
    matrix, targets, _ = training
    trivial = learned.TrivialEstimator().fit(matrix, targets)
    prediction = trivial.predict(matrix[:3])
    direction = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    expected = float(np.std(targets @ direction))
    assert prediction.sigma_in_direction(direction)[0] == pytest.approx(
        expected, rel=1.0e-9, abs=0.0
    )


def test_trivial_rejects_bad_input():
    with pytest.raises(ValueError, match=r"targets \(m, 3\)"):
        learned.TrivialEstimator().fit(np.zeros((5, 24)), np.zeros((5, 2)))
    with pytest.raises(ValueError, match="at least 2 targets"):
        learned.TrivialEstimator().fit(np.zeros((1, 24)), np.zeros((1, 3)))
    with pytest.raises(RuntimeError, match="before fit"):
        learned.TrivialEstimator().predict(np.zeros((1, 24)))


# ---------------------------------------------------------------------------
# The headline: behaviour in the unobservable subspace.
# ---------------------------------------------------------------------------


def test_learned_does_not_inflate_its_uncertainty_where_the_bound_is_infinite(training):
    """The product's central measurement, as a test rather than only as a report.

    A misalignment is injected entirely along a single-axis record's own
    rotation axis -- the direction the record cannot support at any data volume
    -- and the learned estimator is asked for it. The analytic ratio of the
    uncertainty along that direction to the uncertainty along the best-observed
    one is infinite, because the batch solution withholds the direction and
    reports no number. The learned ratio is measured here and the assertion is
    that it is small. If a future configuration ever learns to refuse, this
    test fails and that is the right outcome.
    """
    matrix, targets, _ = training
    model = learned.LearnedEstimator(
        seed=20261029, **learned.CONFIGURATIONS["bagged_ridge"]
    ).fit(matrix, targets)
    ratios = []
    for index in range(25):
        seed = 880000 + index
        probe = dataset.generate(profile="single_axis", samples=120, seed=seed)
        axis = observability.dominant_rotation_axis(probe.record)
        generated = dataset.generate(
            profile="single_axis", samples=120, seed=seed, theta_rad=2.0e-3 * axis
        )
        verdict = observability.assess(generated.record)
        sensor = verdict["secondary_1"]
        assert sensor.rank_deficient
        vector = features.build(generated.record, "secondary_1", verdict)[None, :]
        prediction = model.predict(vector)
        worst = sensor.degrees_of_freedom[-1].direction[None, :]
        best = sensor.degrees_of_freedom[0].direction[None, :]
        ratios.append(
            float(prediction.sigma_in_direction(worst)[0])
            / float(prediction.sigma_in_direction(best)[0])
        )
    assert float(np.median(ratios)) < 5.0


def test_learned_interval_misses_the_truth_in_the_unobservable_direction(training):
    """Measured two-sigma coverage along the unobservable direction, as a test."""
    matrix, targets, _ = training
    model = learned.LearnedEstimator(
        seed=20261029, **learned.CONFIGURATIONS["bagged_ridge"]
    ).fit(matrix, targets)
    covered = 0
    total = 25
    for index in range(total):
        seed = 990000 + index
        probe = dataset.generate(profile="single_axis", samples=120, seed=seed)
        axis = observability.dominant_rotation_axis(probe.record)
        generated = dataset.generate(
            profile="single_axis", samples=120, seed=seed, theta_rad=2.0e-3 * axis
        )
        verdict = observability.assess(generated.record)
        sensor = verdict["secondary_1"]
        worst = sensor.degrees_of_freedom[-1].direction[None, :]
        prediction = model.predict(
            features.build(generated.record, "secondary_1", verdict)[None, :]
        )
        claim = float(prediction.component_in_direction(worst)[0])
        sigma = float(prediction.sigma_in_direction(worst)[0])
        truth = float(np.dot(generated.truth["secondary_1"], worst[0]))
        covered += abs(claim - truth) <= 2.0 * sigma
    assert covered / total < 0.2
