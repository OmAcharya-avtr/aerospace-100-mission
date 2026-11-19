"""The learned predictor: its surface, its two uncertainty outputs, its limits."""

from __future__ import annotations

import numpy as np
import pytest

from passyield.dataset import (
    feature_matrix,
    station_split,
    target_vector,
    time_split,
)
from passyield.predictor import (
    DECLARED_UNCERTAINTY_MODE,
    FEATURE_NAMES,
    UNCERTAINTY_MODES,
    PassYieldPredictor,
    PredictorSettings,
)

#: A small forest for the unit tests: the build environment has two cores
#: shared with four other builds, so the tests use 40 trees and the published
#: figures use 300. The settings are declared here so the difference is
#: visible rather than implicit.
TEST_SETTINGS = PredictorSettings(n_estimators=40, max_depth=8, random_state=99099)


@pytest.fixture(scope="module")
def trained(record):
    """A predictor fitted on the time split's training side."""
    matrix = feature_matrix(record)
    target = target_vector(record)
    split = time_split(record)
    train = list(split.train)
    model = PassYieldPredictor(TEST_SETTINGS).fit(matrix[train], target[train])
    return model, matrix, target, split


def test_feature_names_are_all_pre_pass():
    """No feature names a quantity only knowable after the contact.

    The acquisition time, the outage count, the outage durations and the idle
    time are what the pass DID; a model given any of them is a measurement
    and not a predictor.
    """
    forbidden = ("acquisition_s", "resync", "idle", "realised", "shortfall")
    #: The one exception, and it is a pre-pass quantity: the outage count of
    #: the PREVIOUS contact at the same station. The next test checks that it
    #: is the only outage-flavoured feature and that its name says which pass
    #: it belongs to.
    allowed = {"prior_pass_outage_count"}
    for name in FEATURE_NAMES:
        if name in allowed:
            continue
        assert not any(token in name for token in forbidden), name
        assert "outage" not in name, name
    assert "backlog_mbit" in FEATURE_NAMES


def test_prior_pass_outage_count_is_the_only_outage_feature_and_is_pre_pass():
    """The one outage-flavoured feature refers to the PREVIOUS pass.

    It is knowable before this contact starts, which is the test, and the
    name says which pass it belongs to so that it cannot be mistaken for
    this one's.
    """
    assert "prior_pass_outage_count" in FEATURE_NAMES
    assert sum("outage" in name for name in FEATURE_NAMES) == 1


def test_unfitted_predictor_refuses_to_predict():
    """Calling predict before fit raises, with the remedy in the message."""
    model = PassYieldPredictor(TEST_SETTINGS)
    assert model.fitted is False
    with pytest.raises(ValueError, match="not fitted"):
        model.predict_fraction(np.zeros((1, len(FEATURE_NAMES))))
    with pytest.raises(ValueError, match="not fitted"):
        _ = model.out_of_bag_residuals
    with pytest.raises(ValueError, match="not fitted"):
        model.feature_importance()


def test_fit_rejects_a_feature_matrix_of_the_wrong_width():
    """Input validation: the matrix must match FEATURE_NAMES."""
    model = PassYieldPredictor(TEST_SETTINGS)
    with pytest.raises(ValueError, match="FEATURE_NAMES"):
        model.fit(np.zeros((20, 3)), np.zeros(20))


def test_fit_rejects_misaligned_targets():
    """Input validation."""
    model = PassYieldPredictor(TEST_SETTINGS)
    with pytest.raises(ValueError, match="targets"):
        model.fit(np.zeros((20, len(FEATURE_NAMES))), np.zeros(19))


def test_fit_rejects_a_sample_too_small_for_the_leaf_size():
    """Input validation: a forest needs more rows than two leaves."""
    model = PassYieldPredictor(PredictorSettings(min_samples_leaf=10))
    with pytest.raises(ValueError, match="at least 20 passes"):
        model.fit(np.zeros((5, len(FEATURE_NAMES))), np.zeros(5))


def test_fit_rejects_non_finite_inputs():
    """Input validation."""
    matrix = np.zeros((20, len(FEATURE_NAMES)))
    matrix[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        PassYieldPredictor(TEST_SETTINGS).fit(matrix, np.zeros(20))


def test_point_prediction_is_a_fraction(trained):
    """The target is a fraction, so the prediction lies in ``[0, 1]``."""
    model, matrix, _target, split = trained
    predicted = model.predict_fraction(matrix[list(split.test)])
    assert predicted.shape == (len(split.test),)
    assert float(predicted.min()) >= 0.0
    assert float(predicted.max()) <= 1.0


def test_tree_predictions_have_one_column_per_tree(trained):
    """The tree-spread ensemble is one member per tree."""
    model, matrix, _target, split = trained
    spread = model.tree_predictions(matrix[list(split.test)[:5]])
    assert spread.shape == (5, TEST_SETTINGS.n_estimators)


def test_both_uncertainty_modes_return_an_ensemble_in_range(trained):
    """Both modes return a ``(n, members)`` ensemble clipped to ``[0, 1]``."""
    model, matrix, _target, split = trained
    rows = list(split.test)[:8]
    for mode in UNCERTAINTY_MODES:
        ensemble = model.predict_ensemble(matrix[rows], mode)
        assert ensemble.shape[0] == len(rows)
        assert ensemble.shape[1] >= 2
        assert float(ensemble.min()) >= 0.0
        assert float(ensemble.max()) <= 1.0


def test_declared_uncertainty_mode_is_the_residual_ensemble():
    """The product's declared uncertainty output is the out-of-bag residual law.

    The tree spread measures disagreement about the conditional mean and omits
    the irreducible noise, so it is structurally over-confident; it is kept and
    reported rather than removed.
    """
    assert DECLARED_UNCERTAINTY_MODE == "residual"
    assert set(UNCERTAINTY_MODES) == {"tree-spread", "residual"}


def test_residual_ensemble_is_deterministic(trained):
    """Two calls agree exactly: the ensemble is a quantile grid, not a draw."""
    model, matrix, _target, split = trained
    rows = list(split.test)[:6]
    first = model.predict_ensemble(matrix[rows], "residual")
    second = model.predict_ensemble(matrix[rows], "residual")
    assert np.array_equal(first, second)


def test_residual_ensemble_is_wider_than_the_tree_spread(trained):
    """The out-of-bag residual law is wider than the across-tree spread.

    This is the measurable form of the statement that the tree spread is
    over-confident: it omits the irreducible noise, which the out-of-bag
    residuals contain. Asserted on the median width across the held-out rows,
    not row by row, because a single row can go either way.
    """
    model, matrix, _target, split = trained
    rows = list(split.test)
    spread = model.predict_ensemble(matrix[rows], "tree-spread")
    residual = model.predict_ensemble(matrix[rows], "residual")
    spread_width = np.quantile(spread, 0.9, axis=1) - np.quantile(spread, 0.1, axis=1)
    residual_width = np.quantile(residual, 0.9, axis=1) - np.quantile(residual, 0.1, axis=1)
    assert float(np.median(residual_width)) > float(np.median(spread_width))


def test_unknown_uncertainty_mode_is_refused(trained):
    """Input validation."""
    model, matrix, _target, split = trained
    with pytest.raises(ValueError, match="mode must be one of"):
        model.predict_ensemble(matrix[list(split.test)[:2]], "bayesian")


def test_shortfall_ensemble_is_nominal_minus_fraction_times_capacity(trained, record):
    """The transform to Mbit is applied member by member, exactly.

    Checked against a hand-applied transform of the fraction ensemble, so a
    broadcasting error cannot hide.
    """
    model, matrix, _target, split = trained
    rows = list(split.test)[:5]
    nominal = np.asarray([record[index].nominal_mbit for index in rows])
    capacity = np.asarray([record[index].capacity_mbit for index in rows])
    fraction = model.predict_ensemble(matrix[rows], "residual")
    shortfall = model.predict_shortfall_ensemble(matrix[rows], nominal, capacity, "residual")
    expected = nominal[:, None] - fraction * capacity[:, None]
    assert np.allclose(shortfall, expected, rtol=1.0e-12, atol=0.0)


def test_shortfall_ensemble_rejects_misaligned_volumes(trained):
    """Input validation."""
    model, matrix, _target, split = trained
    rows = list(split.test)[:5]
    with pytest.raises(ValueError, match="must have 5 entries"):
        model.predict_shortfall_ensemble(matrix[rows], np.zeros(4), np.zeros(5))


def test_feature_importance_sums_to_one_and_is_named(trained):
    """Importances are keyed by feature name and total 1."""
    model, _matrix, _target, _split = trained
    importance = model.feature_importance()
    assert set(importance) == set(FEATURE_NAMES)
    assert sum(importance.values()) == pytest.approx(1.0, rel=1.0e-9, abs=1.0e-12)


def test_out_of_bag_residuals_are_centred_near_zero(trained):
    """An unbiased fit leaves out-of-bag residuals with a small mean.

    Checked against three standard errors of the residual mean, a declared
    band from the sample itself rather than an absolute constant.
    """
    model, _matrix, _target, _split = trained
    residuals = model.out_of_bag_residuals
    assert residuals.size > 100
    error = float(np.std(residuals, ddof=1) / np.sqrt(residuals.size))
    assert abs(float(np.mean(residuals))) < 3.0 * error


def test_the_same_seed_gives_the_same_fit(record):
    """Two fits at the same random_state predict identically."""
    matrix = feature_matrix(record)
    target = target_vector(record)
    rows = list(station_split(record).train)
    first = PassYieldPredictor(TEST_SETTINGS).fit(matrix[rows], target[rows])
    second = PassYieldPredictor(TEST_SETTINGS).fit(matrix[rows], target[rows])
    probe = matrix[list(station_split(record).test)[:10]]
    assert np.array_equal(first.predict_fraction(probe), second.predict_fraction(probe))
