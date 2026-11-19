"""The learned component: row construction, the two splits, the confidence output.

The baselines were implemented and validated first; these tests check that the
learned component is wired correctly and that the two held-out splits are
genuinely held out. The measured comparison -- which predictor wins, with the
numbers -- is in ``validation/validate_learned_model.py`` and ``MODEL_CARD.md``.
Nothing here asserts that the learned model beats a baseline, because on the
committed history it does not on most metrics, and a test asserting otherwise
would have to be retuned away.
"""

from __future__ import annotations

import numpy as np
import pytest

from flakerank.history import RunHistory
from flakerank.predictor import (
    FEATURE_NAMES,
    IntermittencyCostPredictor,
    baseline_scores,
    build_rows,
    evaluate_split,
    make_splits,
)

LOOKBACK = 120
HORIZON = 30
TRAIN_ORIGINS = (120, 150, 180, 210)
TIME_ORIGINS = (240, 270)
ALL_ORIGINS = TRAIN_ORIGINS + TIME_ORIGINS


@pytest.fixture(scope="module")
def rows(history: RunHistory):
    """The full row set over the committed history."""
    return build_rows(
        history,
        origins=ALL_ORIGINS,
        lookback=LOOKBACK,
        horizon=HORIZON,
        minimum_attempts=8,
    )


@pytest.fixture(scope="module")
def splits(rows):
    """The training set and the two held-out sets."""
    return make_splits(rows, train_origins=TRAIN_ORIGINS, time_origins=TIME_ORIGINS)


@pytest.fixture(scope="module")
def fitted(splits):
    """One fit on the training rows, reused by both evaluations."""
    return IntermittencyCostPredictor().fit(splits["train"])


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def test_row_count_and_feature_width(rows) -> None:
    """432 rows: 72 tests at 6 origins, all surviving the attempt threshold."""
    assert rows.n_rows == 432
    assert rows.features.shape == (432, len(FEATURE_NAMES))
    assert len(FEATURE_NAMES) == 17


def test_no_feature_is_nan_or_infinite(rows) -> None:
    """A NaN feature would be imputed silently by some estimators."""
    assert np.all(np.isfinite(rows.features))


def test_target_is_log1p_of_the_seconds(rows) -> None:
    """``target`` and ``target_seconds`` are the same quantity in two scales."""
    np.testing.assert_allclose(
        rows.target, np.log1p(rows.target_seconds), rtol=1e-14, atol=0.0
    )


def test_features_cannot_see_their_own_target(history: RunHistory) -> None:
    """A row's features come from sessions strictly before its origin.

    Checked structurally rather than by inspection: the wasted-seconds feature
    at origin ``o`` equals the accumulated wasted time of the window
    ``[o - L, o)``, which is disjoint from the target window ``[o, o + H)``.
    """
    origin = 150
    single = build_rows(
        history, origins=(origin,), lookback=LOOKBACK, horizon=HORIZON, minimum_attempts=8
    )
    past = history.session_window(origin - LOOKBACK, origin)
    column = FEATURE_NAMES.index("wasted_seconds")
    expected = {
        summary.name: summary.wasted_seconds
        for summary in past.summaries(minimum_attempts=8)
    }
    for position, name in enumerate(single.test_name):
        assert single.features[position, column] == pytest.approx(
            expected[name], rel=1e-12, abs=1e-12
        )


def test_minimum_attempts_threshold_removes_rows(history: RunHistory) -> None:
    """Raising the threshold cannot add rows."""
    low = build_rows(
        history, origins=(150,), lookback=LOOKBACK, horizon=HORIZON, minimum_attempts=1
    )
    high = build_rows(
        history, origins=(150,), lookback=LOOKBACK, horizon=HORIZON, minimum_attempts=100
    )
    assert high.n_rows <= low.n_rows


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"origins": ()}, "at least one session index"),
        ({"lookback": 0}, "must be >= 1"),
        ({"horizon": 0}, "must be >= 1"),
        ({"minimum_attempts": 0}, "minimum_attempts must be >= 1"),
        ({"origins": (5,)}, "falls outside a history"),
        ({"origins": (325,)}, "falls outside a history"),
    ],
)
def test_build_rows_validates_its_windows(
    history: RunHistory, kwargs: dict, match: str
) -> None:
    """Every window argument is checked with a message naming the problem."""
    arguments = {
        "origins": (150,),
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "minimum_attempts": 8,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        build_rows(history, **arguments)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The two splits
# ---------------------------------------------------------------------------


def test_split_sizes(splits) -> None:
    """200 training rows, 88 held-out-test rows, 100 held-out-time rows.

    Hand arithmetic: 72 tests, 30 per cent withheld gives 22 withheld and 50
    kept. Training is 50 tests at 4 origins = 200 rows; held-out TESTS is 22
    tests at the same 4 origins = 88 rows; held-out TIME is the 50 training
    tests at 2 later origins = 100 rows.
    """
    assert splits["train"].n_rows == 200
    assert splits["heldout_tests"].n_rows == 88
    assert splits["heldout_time"].n_rows == 100


def test_heldout_tests_shares_no_test_with_training(splits) -> None:
    """The held-out TESTS split is held out BY TEST, which is the point."""
    train = set(splits["train"].test_index.tolist())
    heldout = set(splits["heldout_tests"].test_index.tolist())
    assert train and heldout
    assert not (train & heldout)


def test_heldout_time_shares_no_origin_with_training(splits) -> None:
    """The held-out TIME split is held out BY TIME, which is a different cut."""
    train = set(splits["train"].origin.tolist())
    heldout = set(splits["heldout_time"].origin.tolist())
    assert train == set(TRAIN_ORIGINS)
    assert heldout == set(TIME_ORIGINS)
    assert not (train & heldout)


def test_the_two_splits_are_different_questions(splits) -> None:
    """Held-out TESTS and held-out TIME overlap in neither test nor origin.

    If they did overlap on both axes they would be the same split under two
    names, and reporting two numbers would be misleading.
    """
    tests_split = splits["heldout_tests"]
    time_split = splits["heldout_time"]
    assert not (
        set(tests_split.test_index.tolist()) & set(time_split.test_index.tolist())
    )
    assert not (set(tests_split.origin.tolist()) & set(time_split.origin.tolist()))


def test_group_assignment_is_deterministic(rows) -> None:
    """The same seed gives the same withheld tests."""
    first = make_splits(rows, train_origins=TRAIN_ORIGINS, time_origins=TIME_ORIGINS, seed=5)
    second = make_splits(rows, train_origins=TRAIN_ORIGINS, time_origins=TIME_ORIGINS, seed=5)
    assert first["heldout_tests"].test_index.tolist() == (
        second["heldout_tests"].test_index.tolist()
    )


def test_overlapping_origin_sets_rejected(rows) -> None:
    """A held-out TIME split sharing an origin with training is not held out."""
    with pytest.raises(ValueError, match="train and time origins overlap"):
        make_splits(rows, train_origins=(120, 150), time_origins=(150, 240))


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.2, 1.4])
def test_invalid_holdout_fraction_rejected(rows, fraction: float) -> None:
    """A fraction of 0 or 1 leaves one of the two splits empty."""
    with pytest.raises(ValueError, match="holdout_fraction"):
        make_splits(
            rows,
            train_origins=TRAIN_ORIGINS,
            time_origins=TIME_ORIGINS,
            holdout_fraction=fraction,
        )


def test_empty_origin_set_rejected(rows) -> None:
    """Both origin sets are required."""
    with pytest.raises(ValueError, match="must be non-empty"):
        make_splits(rows, train_origins=(), time_origins=TIME_ORIGINS)


def test_select_rejects_an_empty_mask(rows) -> None:
    """A split with no rows cannot be scored."""
    with pytest.raises(ValueError, match="would be empty"):
        rows.select(np.zeros(rows.n_rows, dtype=bool), "nothing")


def test_select_rejects_a_mismatched_mask(rows) -> None:
    """The mask must be parallel to the rows."""
    with pytest.raises(ValueError, match="mask must have shape"):
        rows.select(np.zeros(3, dtype=bool), "wrong")


# ---------------------------------------------------------------------------
# The model and its confidence output
# ---------------------------------------------------------------------------


def test_predict_before_fit_raises() -> None:
    """The predictor holds no state until fitted, and says why."""
    with pytest.raises(RuntimeError, match="serialises no model"):
        IntermittencyCostPredictor().predict(None)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="before fit"):
        _ = IntermittencyCostPredictor().feature_importances


def test_fit_rejects_a_split_smaller_than_a_leaf(splits) -> None:
    """A training split below ``min_samples_leaf`` cannot be fitted."""
    tiny_split = splits["train"].select(
        np.arange(splits["train"].n_rows) < 2, "two rows"
    )
    with pytest.raises(ValueError, match="below min_samples_leaf"):
        IntermittencyCostPredictor(min_samples_leaf=5).fit(tiny_split)


def test_prediction_shape_and_tree_count(fitted, splits) -> None:
    """One mean, one sd and one interval per row, over 300 trees."""
    prediction = fitted.predict(splits["heldout_tests"])
    rows = splits["heldout_tests"].n_rows
    assert prediction.mean.shape == (rows,)
    assert prediction.sd.shape == (rows,)
    assert prediction.low.shape == (rows,)
    assert prediction.high.shape == (rows,)
    assert prediction.n_trees == 300
    assert prediction.nominal_coverage == pytest.approx(0.90, rel=0.0, abs=0.0)


def test_confidence_interval_is_well_ordered(fitted, splits) -> None:
    """``low <= high`` on every row, which is what a quantile pair guarantees.

    A first draft of this test asserted that the ensemble MEAN lies inside the
    central quantile interval, and it failed on the held-out TESTS split. That
    is not a defect, it is a property of the ensemble distribution, and it is
    worth stating because it bears directly on how the confidence output should
    be read.

    For a test that almost never fails, nearly every tree predicts exactly
    ``log1p(0) = 0`` and a handful predict a positive value. The distribution
    across trees is then a point mass at zero with a thin right tail, so the
    95th percentile is still 0 while the mean is small and positive -- measured
    here, a mean of 1.00719352e-05 against a 95th percentile of 0. A central
    quantile interval of a distribution that skewed does not contain its own
    mean, and nothing is wrong with either number.

    What follows for the reader: the interval is an interval on the TREES'
    answers, not a predictive interval for the test's next-window cost, and for
    a rarely failing test it can collapse to ``[0, 0]`` while the point
    prediction is positive. ``validation/validate_learned_model.py`` reports
    how many rows that happens on, and ``MODEL_CARD.md`` records it under
    failure cases.
    """
    prediction = fitted.predict(splits["heldout_tests"])
    assert np.all(prediction.low <= prediction.high)
    assert np.all(prediction.interval_width >= 0.0)
    assert np.all(prediction.sd >= 0.0)


def test_mean_escapes_the_interval_only_where_it_is_near_degenerate(
    fitted, splits
) -> None:
    """Where the mean escapes the interval, that interval is vanishingly narrow.

    This is the finding above turned into a check, and the first form of the
    check was also wrong: asserting that every escaping row has a width of
    EXACTLY zero failed, because 2 of 55 escaping rows on the held-out TIME
    split have a width of 7.58412808e-04 against a mean of 2.60599686e-03.
    What is true, and is what the check now asserts, is that the escape only
    happens where the interval is near-degenerate.

    Measured on the committed history: 47 of 88 held-out-TESTS rows and 55 of
    100 held-out-TIME rows have the mean outside the interval, and the widest
    such interval anywhere is 7.58412808e-04 in ``log1p`` seconds, against a
    mean interval width of about 0.84 to 0.96 across the splits. So the
    threshold below is about one per cent of a typical width, which keeps the
    assertion from being vacuous. The counts are reported by
    ``validation/validate_learned_model.py``.
    """
    widest = 0.0
    for key in ("heldout_tests", "heldout_time"):
        prediction = fitted.predict(splits[key])
        outside = (prediction.mean < prediction.low) | (prediction.mean > prediction.high)
        typical = float(prediction.interval_width.mean())
        assert typical > 0.1, "a split with no spread at all would make this vacuous"
        widest = max(widest, float(prediction.interval_width[outside].max(initial=0.0)))
        assert np.all(prediction.interval_width[outside] <= 1e-2)
    assert widest == pytest.approx(7.58412808e-04, rel=1e-6, abs=0.0)


def test_confidence_output_is_not_a_constant(fitted, splits) -> None:
    """A confidence output identical on every row would carry no information."""
    prediction = fitted.predict(splits["heldout_tests"])
    assert prediction.sd.std() > 0.0
    assert prediction.interval_width.std() > 0.0


def test_seconds_property_is_monotone_in_the_mean(fitted, splits) -> None:
    """``expm1`` is strictly increasing, so the ranking is unchanged by it."""
    prediction = fitted.predict(splits["heldout_tests"])
    order_log = np.argsort(prediction.mean, kind="stable")
    order_seconds = np.argsort(prediction.seconds, kind="stable")
    assert order_log.tolist() == order_seconds.tolist()


def test_fit_is_deterministic_at_a_fixed_random_state(splits) -> None:
    """Two fits at the same ``random_state`` predict the same values."""
    first = IntermittencyCostPredictor(random_state=11).fit(splits["train"])
    second = IntermittencyCostPredictor(random_state=11).fit(splits["train"])
    np.testing.assert_allclose(
        first.predict(splits["heldout_time"]).mean,
        second.predict(splits["heldout_time"]).mean,
        rtol=0.0,
        atol=0.0,
    )


def test_feature_importances_sum_to_one(fitted) -> None:
    """Impurity importances are a normalised decomposition."""
    importances = fitted.feature_importances
    assert importances.shape == (len(FEATURE_NAMES),)
    assert importances.sum() == pytest.approx(1.0, rel=1e-10, abs=0.0)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_baseline_scores_cover_the_three_comparators(splits) -> None:
    """The flake rate, the analytic cost and the accumulated persistence."""
    scores = baseline_scores(splits["heldout_tests"], horizon=HORIZON, lookback=LOOKBACK)
    assert set(scores) == {"flake rate", "analytic expected cost", "accumulated persistence"}
    for vector in scores.values():
        assert vector.shape == (splits["heldout_tests"].n_rows,)
        assert np.all(np.isfinite(vector))


def test_flake_rate_gets_no_error_metric(splits) -> None:
    """The rate has no seconds interpretation, so its MAE is reported as ``nan``.

    Reporting a number there would invite a comparison the units do not
    support, which is the product's whole argument in miniature.
    """
    scores = baseline_scores(splits["heldout_tests"], horizon=HORIZON, lookback=LOOKBACK)
    score = evaluate_split(
        "flake rate", splits["heldout_tests"], scores["flake rate"], seconds_scale=False
    )
    assert np.isnan(score.mae_log1p)
    assert np.isnan(score.median_absolute_error_log1p)
    assert not np.isnan(score.spearman)
    assert not np.isnan(score.captured_fraction)


def test_perfect_score_is_recognised(splits) -> None:
    """Scoring the target against itself gives rho = 1 and full capture."""
    split = splits["heldout_time"]
    score = evaluate_split("oracle", split, split.target, seconds_scale=True)
    assert score.spearman == pytest.approx(1.0, rel=1e-12, abs=0.0)
    assert score.mae_log1p == pytest.approx(0.0, rel=0.0, abs=1e-12)
    assert score.captured_fraction == pytest.approx(
        score.oracle_captured_fraction, rel=1e-12, abs=0.0
    )


def test_capture_is_a_fraction(fitted, splits) -> None:
    """Captured fraction lies in [0, 1] and never exceeds the oracle's."""
    for key in ("heldout_tests", "heldout_time"):
        split = splits[key]
        prediction = fitted.predict(split)
        score = evaluate_split(
            "learned", split, prediction.mean, seconds_scale=True, prediction=prediction
        )
        assert 0.0 <= score.captured_fraction <= 1.0
        assert score.captured_fraction <= score.oracle_captured_fraction + 1e-12


def test_interval_coverage_is_a_fraction(fitted, splits) -> None:
    """Coverage is reported whatever it is, and is a fraction."""
    for key in ("heldout_tests", "heldout_time"):
        split = splits[key]
        prediction = fitted.predict(split)
        score = evaluate_split(
            "learned", split, prediction.mean, seconds_scale=True, prediction=prediction
        )
        assert 0.0 <= score.interval_coverage <= 1.0
        assert score.mean_interval_width_log1p > 0.0


def test_evaluate_split_validates_the_score_shape(splits) -> None:
    """A score vector must be parallel to the rows."""
    with pytest.raises(ValueError, match="score must have shape"):
        evaluate_split(
            "x", splits["heldout_tests"], np.zeros(3), seconds_scale=True
        )
