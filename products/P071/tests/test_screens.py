"""Analytic screens, which are implemented before the learned model."""

from __future__ import annotations

import numpy as np
import pytest

from conjprob.analytic import pc_small_body_approximation
from conjprob.features import features_from_arrays
from conjprob.screens import (
    SCREEN_NAMES,
    all_screen_scores,
    mahalanobis_score,
    miss_distance_score,
    small_body_score,
)


def _features():
    return features_from_arrays(
        [300.0, 100.0, 50.0],
        [54.0, 100.0, 20.0],
        [120.0, 10.0, 200.0],
        [30.0, 0.0, 0.0],
        [12.0, 5.0, 3.0],
        [5740.0, 900.0, 1200.0],
        [80.0, 60.0, 30.0],
    )


def test_all_screens_are_present_and_sized():
    scores = all_screen_scores(_features())
    assert set(scores) == set(SCREEN_NAMES)
    for values in scores.values():
        assert values.shape == (3,)


def test_miss_distance_score_is_the_negated_log_distance():
    features = _features()
    np.testing.assert_allclose(
        miss_distance_score(features),
        -np.log10([np.hypot(120.0, 30.0), 10.0, 200.0]),
        rtol=1e-13,
    )


def test_mahalanobis_score_is_the_negated_mahalanobis_distance():
    features = _features()
    expected = -np.array(
        [
            np.hypot(120.0 / 300.0, 30.0 / 54.0),
            np.hypot(10.0 / 100.0, 0.0),
            np.hypot(200.0 / 50.0, 0.0),
        ]
    )
    np.testing.assert_allclose(mahalanobis_score(features), expected, rtol=1e-13)


def test_small_body_score_is_the_log_of_the_closed_form():
    features = _features()
    expected = np.log10(
        [
            pc_small_body_approximation(300.0, 54.0, 120.0, 30.0, 12.0),
            pc_small_body_approximation(100.0, 100.0, 10.0, 0.0, 5.0),
            pc_small_body_approximation(50.0, 20.0, 200.0, 0.0, 3.0),
        ]
    )
    np.testing.assert_allclose(small_body_score(features), expected, rtol=1e-12)


def test_scores_are_larger_for_a_closer_miss():
    near = features_from_arrays(300.0, 54.0, 10.0, 0.0, 12.0, 5740.0, 80.0)
    far = features_from_arrays(300.0, 54.0, 900.0, 0.0, 12.0, 5740.0, 80.0)
    for score in (miss_distance_score, mahalanobis_score, small_body_score):
        assert score(near)[0] > score(far)[0]


def test_mahalanobis_score_ignores_the_hard_body_radius_by_construction():
    small = features_from_arrays(300.0, 54.0, 120.0, 30.0, 1.0, 5740.0, 80.0)
    large = features_from_arrays(300.0, 54.0, 120.0, 30.0, 30.0, 5740.0, 80.0)
    assert mahalanobis_score(small)[0] == pytest.approx(mahalanobis_score(large)[0], rel=1e-15)
    assert small_body_score(large)[0] > small_body_score(small)[0]


def test_accepts_a_single_row_as_a_one_dimensional_array():
    row = _features()[0]
    assert miss_distance_score(row).shape == (1,)


def test_rejects_the_wrong_feature_width():
    with pytest.raises(ValueError, match=r"shape \(n, 10\)"):
        miss_distance_score(np.zeros((4, 3)))
