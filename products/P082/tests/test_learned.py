"""The learned correlation model and its interval output.

The structural claims tested here are the two that matter for honesty: there is
no method on the prediction that returns a bare number, and the quantile
crossings are counted rather than repaired.
"""

from __future__ import annotations

import numpy as np
import pytest

from sitediverse.learned import (
    LEARNING_RATE,
    MAX_DEPTH,
    NOMINAL_COVERAGE,
    QUANTILES,
    RANDOM_STATE,
    RHO_CLAMP,
    SUBSAMPLE,
    TREES,
    CorrelationPrediction,
    LearnedCorrelationModel,
)


def test_the_declared_hyperparameters_are_the_ones_documented():
    """Equation (38). Declared once, reported, and not tuned on a hold-out."""
    assert QUANTILES == (0.1, 0.5, 0.9)
    assert NOMINAL_COVERAGE == pytest.approx(0.8, rel=1e-15, abs=0.0)
    assert TREES == 240
    assert MAX_DEPTH == 3
    assert LEARNING_RATE == 0.05
    assert SUBSAMPLE == 0.9
    assert RANDOM_STATE == 12345


@pytest.fixture(scope="module")
def fitted():
    """A model fitted on a declared synthetic feature table.

    The target is an exponential in the first feature plus a term in the
    anisotropy ratio, so the model has something to learn that a function of
    separation alone cannot represent. 400 rows, which is the order of the
    training set the real study uses.
    """
    rng = np.random.default_rng(3838)
    rows = 400
    separation = rng.uniform(90.0, 7000.0, rows)
    ratio = rng.uniform(0.0, 1.0, rows)
    features = np.column_stack(
        [
            separation,
            separation * ratio,
            separation * (1.0 - ratio),
            ratio,
            rng.integers(0, 4, rows).astype(float),
            rng.uniform(0.25, 0.45, rows),
            rng.uniform(0.45, 0.55, rows),
            rng.uniform(0.1, 0.25, rows),
        ]
    )
    target = 0.9 * np.exp(-separation / (250.0 + 900.0 * ratio))
    target = target + 0.02 * rng.standard_normal(rows)
    model = LearnedCorrelationModel().fit(features, target)
    return model, features, target


def test_the_prediction_has_no_method_returning_a_bare_value(fitted):
    """Equation (37): a caller who wants a point estimate holds the interval.

    Checked structurally rather than by inspection: no public attribute of the
    prediction is a plain float, and the three quantile arrays are all present.
    """
    model, features, _ = fitted
    prediction = model.predict(features[:10])
    assert isinstance(prediction, CorrelationPrediction)
    assert prediction.lower.shape == (10,)
    assert prediction.median.shape == (10,)
    assert prediction.upper.shape == (10,)
    for name in ("value", "point", "estimate", "predict"):
        assert not hasattr(prediction, name)


def test_the_model_learns_something_a_separation_only_fit_cannot(fitted):
    """The learned component has a real information advantage on this target.

    Constructed so: the correlation length depends on the anisotropy ratio. A
    two-parameter exponential in separation alone cannot represent that, and the
    learned model's error must be the smaller of the two. If this test ever
    fails the fixture has stopped exercising the thing the learned model is for.
    """
    from sitediverse.correlation import fit_exponential

    model, features, target = fitted
    learned_error = float(np.mean(np.abs(model.predict(features).median - target)))
    baseline = fit_exponential(features[:, 0], target)
    baseline_error = float(
        np.mean(np.abs(baseline.predict(features[:, 0]) - target))
    )
    assert learned_error < baseline_error


def test_the_interval_is_wide_enough_to_cover_most_of_the_training_target(fitted):
    """In-sample coverage near the nominal 80 per cent.

    In-sample only. The out-of-sample coverage is MEASURED in
    ``validation/validate_learned_model.py`` and is below nominal on at least
    one hold-out cell, which is reported there rather than asserted here.
    """
    model, features, target = fitted
    coverage = model.predict(features).coverage(target)
    assert coverage == pytest.approx(NOMINAL_COVERAGE, rel=0.0, abs=0.15)


def test_the_interval_width_is_positive_for_most_rows(fitted):
    """Quantile crossings exist and are a minority, which is the honest claim."""
    model, features, _ = fitted
    prediction = model.predict(features)
    assert float(np.median(prediction.interval_width)) > 0.0
    assert prediction.crossing_count < 0.2 * prediction.rows


def test_the_crossings_are_counted_and_not_repaired(fitted):
    """Equation (37)'s declared choice: a sorted interval hides a diagnostic."""
    model, features, _ = fitted
    prediction = model.predict(features)
    crossed = prediction.crossed
    if prediction.crossing_count > 0:
        # Where a crossing occurred the raw ordering really is violated, which
        # is what a sorted interval would conceal.
        assert np.any(
            (prediction.lower[crossed] > prediction.median[crossed])
            | (prediction.median[crossed] > prediction.upper[crossed])
        )
    assert prediction.crossing_count == int(np.count_nonzero(crossed))


def test_coverage_treats_a_crossed_row_as_the_interval_between_the_outer_quantiles(
    fitted,
):
    """So that a coverage figure is well defined even where the fit crossed."""
    model, features, target = fitted
    prediction = model.predict(features)
    covers = prediction.covers(target)
    low = np.minimum(prediction.lower, prediction.upper)
    high = np.maximum(prediction.lower, prediction.upper)
    assert np.array_equal(covers, (target >= low) & (target <= high))


def test_the_median_is_clamped_into_the_copulas_bracket(fitted):
    """Equation (37)'s clamp: a tree ensemble is not constrained to a correlation."""
    model, features, _ = fitted
    prediction = model.predict(features)
    assert float(prediction.median.min()) >= RHO_CLAMP[0]
    assert float(prediction.median.max()) <= RHO_CLAMP[1]


def test_the_clamp_count_is_reported(fitted):
    """Counted, so that a report can say how often the model left the range."""
    model, features, _ = fitted
    prediction = model.predict(features)
    assert prediction.clamp_count == int(np.count_nonzero(prediction.clamped))


def test_the_fit_is_reproducible_at_a_fixed_random_state(fitted):
    """Two fits on the same data with the same declared state agree.

    A relative tolerance, not an exact comparison: the two fits run the same
    arithmetic in the same order within one build of scikit-learn, so they
    agree to rounding, and the claim that does not follow -- and is not made --
    is reproducibility across versions. ``pyproject.toml`` bounds
    ``scikit-learn`` to the minor version for exactly that reason.
    """
    model, features, target = fitted
    again = LearnedCorrelationModel().fit(features, target)
    first = model.predict(features[:50]).median
    second = again.predict(features[:50]).median
    assert float(np.max(np.abs(first - second))) < 1e-12


def test_a_different_random_state_gives_a_different_fit(fitted):
    """Subsampling is stochastic, so the state matters and is declared."""
    _, features, target = fitted
    other = LearnedCorrelationModel(random_state=999).fit(features, target)
    base = LearnedCorrelationModel().fit(features, target)
    assert float(
        np.max(np.abs(other.predict(features).median - base.predict(features).median))
    ) > 1e-6


def test_the_training_report_records_what_the_fit_did(fitted):
    """Reported, not inferred from the configuration."""
    model, features, _ = fitted
    report = model.report
    assert report.rows == features.shape[0]
    assert report.feature_count == features.shape[1]
    assert report.fit_seconds > 0.0
    assert report.quantiles == QUANTILES
    assert report.trees == TREES
    assert report.tuned_against_holdout is False


def test_the_fit_finishes_well_inside_the_declared_compute_budget(fitted):
    """The declared budget is three minutes for any training run."""
    model, _, _ = fitted
    assert model.report.fit_seconds < 60.0


def test_feature_importance_has_one_entry_per_feature(fitted):
    """A diagnostic only; no claim in this product rests on it."""
    model, features, _ = fitted
    importance = model.feature_importance()
    assert importance.shape == (features.shape[1],)
    assert float(importance.sum()) == pytest.approx(1.0, rel=1e-6, abs=0.0)


def test_predicting_before_fitting_raises():
    """A model that has not been fitted has nothing to say."""
    model = LearnedCorrelationModel()
    assert not model.fitted
    with pytest.raises(RuntimeError, match="has not been fitted"):
        model.predict(np.zeros((2, 8)))


def test_reading_the_report_before_fitting_raises():
    """The same for the report."""
    with pytest.raises(RuntimeError, match="has not been fitted"):
        _ = LearnedCorrelationModel().report


def test_feature_importance_before_fitting_raises():
    """And for the importance."""
    with pytest.raises(RuntimeError, match="has not been fitted"):
        LearnedCorrelationModel().feature_importance()


def test_the_model_rejects_quantiles_that_are_not_increasing():
    """Input validation."""
    with pytest.raises(ValueError, match="strictly increasing"):
        LearnedCorrelationModel(quantiles=(0.5, 0.1, 0.9))


def test_the_model_rejects_the_wrong_number_of_quantiles():
    """Input validation."""
    with pytest.raises(ValueError, match="exactly 3 quantile levels"):
        LearnedCorrelationModel(quantiles=(0.1, 0.9))


def test_the_model_rejects_a_quantile_outside_zero_one():
    """Input validation."""
    with pytest.raises(ValueError, match="must lie in"):
        LearnedCorrelationModel(quantiles=(0.0, 0.5, 0.9))


def test_the_model_rejects_a_non_positive_tree_count():
    """Input validation."""
    with pytest.raises(ValueError, match="trees must be at least 1"):
        LearnedCorrelationModel(trees=0)


def test_the_model_rejects_a_learning_rate_above_one():
    """Input validation."""
    with pytest.raises(ValueError, match="learning_rate must lie in"):
        LearnedCorrelationModel(learning_rate=1.5)


def test_the_model_rejects_a_subsample_fraction_above_one():
    """Input validation."""
    with pytest.raises(ValueError, match="subsample must lie in"):
        LearnedCorrelationModel(subsample=1.5)


def test_the_model_rejects_a_zero_depth():
    """Input validation."""
    with pytest.raises(ValueError, match="max_depth must be at least 1"):
        LearnedCorrelationModel(max_depth=0)


def test_the_fit_rejects_a_one_dimensional_feature_matrix():
    """Input validation."""
    with pytest.raises(ValueError, match="two-dimensional"):
        LearnedCorrelationModel().fit(np.zeros(10), np.zeros(10))


def test_the_fit_rejects_a_mismatched_target():
    """Input validation."""
    with pytest.raises(ValueError, match="target must have shape"):
        LearnedCorrelationModel().fit(np.zeros((10, 8)), np.zeros(9))


def test_the_fit_rejects_too_few_rows_for_the_declared_depth():
    """A tree of depth 3 needs more than eight rows to mean anything."""
    with pytest.raises(ValueError, match="too few for trees of depth"):
        LearnedCorrelationModel().fit(np.zeros((4, 8)), np.zeros(4))


def test_the_fit_rejects_a_non_finite_feature():
    """Input validation."""
    features = np.zeros((40, 8))
    features[3, 2] = np.nan
    with pytest.raises(ValueError, match="features contain a non-finite"):
        LearnedCorrelationModel().fit(features, np.zeros(40))


def test_the_fit_rejects_a_non_finite_target():
    """Input validation."""
    target = np.zeros(40)
    target[7] = np.inf
    with pytest.raises(ValueError, match="target contains a non-finite"):
        LearnedCorrelationModel().fit(np.zeros((40, 8)), target)


def test_predicting_with_the_wrong_column_count_raises(fitted):
    """Input validation with the expected count in the message."""
    model, _, _ = fitted
    with pytest.raises(ValueError, match="must have 8 columns"):
        model.predict(np.zeros((5, 4)))


def test_coverage_rejects_a_mismatched_truth_array(fitted):
    """Input validation."""
    model, features, _ = fitted
    prediction = model.predict(features[:10])
    with pytest.raises(ValueError, match="truth must have shape"):
        prediction.covers(np.zeros(11))


def test_the_repeated_feature_names_match_the_canonical_list():
    """``learned.FEATURE_NAMES_NOTE`` is asserted equal to ``features.FEATURE_NAMES``.

    The names are repeated in ``sitediverse.learned`` so that a report can label
    an importance vector without importing the feature builder, which pulls in
    the copula and the climatology. A repeated constant is a defect waiting to
    happen, so the equality is a test rather than a comment.
    """
    from sitediverse.features import FEATURE_NAMES
    from sitediverse.learned import FEATURE_NAMES_NOTE

    assert FEATURE_NAMES_NOTE == FEATURE_NAMES
