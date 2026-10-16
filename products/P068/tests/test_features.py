"""Window features: shape, finiteness and the absence of label leakage."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from degradepredict.features import (
    FEATURE_NAMES,
    FEATURE_UNITS,
    feature_matrix,
    window_features,
)
from degradepredict.generate import DEFAULT_LIMIT, generate_population


def _window(n=40, slope=-0.02, noise=0.2, seed=0):
    t = np.arange(n, dtype=float) * 7.0
    y = 100.0 + slope * t + np.random.default_rng(seed).normal(0.0, noise, size=n)
    return t, y


def test_every_feature_name_has_a_declared_unit():
    assert set(FEATURE_NAMES) == set(FEATURE_UNITS)
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


def test_feature_vector_shape_and_finiteness():
    t, y = _window()
    x = window_features(t, y, DEFAULT_LIMIT)
    assert x.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(x))


def test_linear_extrapolation_feature_is_the_hand_calculated_time_to_limit():
    # Noise-free y = 100 - 0.02 t at t0 = 273 d: the smoothed margin is
    # 100 - 0.02*266.. averaged over the last three samples minus 80, and the
    # fitted decline is 0.02 percent per day, so log_ttl_linear is the log of
    # their ratio. Check against the same arithmetic done by hand.
    t = np.arange(40, dtype=float) * 7.0
    y = 100.0 - 0.02 * t
    x = window_features(t, y, DEFAULT_LIMIT)
    idx = FEATURE_NAMES.index("log_ttl_linear")
    margin_smoothed = float(np.mean(y[-3:])) - 80.0
    assert x[idx] == pytest.approx(np.log(margin_smoothed / 0.02), rel=1e-6)


def test_fraction_consumed_is_between_zero_and_one_for_a_degrading_window():
    t, y = _window(noise=0.0)
    x = window_features(t, y, DEFAULT_LIMIT)
    idx = FEATURE_NAMES.index("fraction_consumed")
    assert 0.0 < x[idx] < 1.0


def test_features_do_not_depend_on_the_label():
    """No crossing time, follow-up or event flag may reach the feature vector."""
    tr = generate_population(1, seed=99)[0]
    base = window_features(tr.times_days, tr.values_percent, DEFAULT_LIMIT)
    tampered = dataclasses.replace(
        tr,
        crossing_time_days=tr.crossing_time_days * 3.0,
        followup_days=1.0,
        remaining_true_days=1.0,
        remaining_observed_days=1.0,
        event=not tr.event,
    )
    after = window_features(tampered.times_days, tampered.values_percent, DEFAULT_LIMIT)
    np.testing.assert_array_equal(base, after)


def test_feature_matrix_stacks_rows_in_order():
    pop = generate_population(6, seed=12)
    mat = feature_matrix(pop, DEFAULT_LIMIT)
    assert mat.shape == (6, len(FEATURE_NAMES))
    np.testing.assert_allclose(
        mat[3], window_features(pop[3].times_days, pop[3].values_percent, DEFAULT_LIMIT)
    )


def test_feature_matrix_rejects_an_empty_population():
    with pytest.raises(ValueError, match="no trajectories"):
        feature_matrix([], DEFAULT_LIMIT)


def test_features_survive_an_improving_window_without_producing_infinities():
    # A window drifting the wrong way makes the linear time-to-limit undefined;
    # the guard must clamp rather than emit an infinity.
    t, y = _window(slope=+0.01, noise=0.1, seed=5)
    x = window_features(t, y, DEFAULT_LIMIT)
    assert np.all(np.isfinite(x))


@pytest.mark.parametrize(
    ("times", "values", "match"),
    [
        (np.arange(2.0), np.arange(2.0), "at least three"),
        (np.arange(5.0), np.arange(4.0), "same length"),
        (np.array([0.0, 0.0, 1.0]), np.array([1.0, 2.0, 3.0]), "strictly increasing"),
        (np.array([0.0, 1.0, np.nan]), np.array([1.0, 2.0, 3.0]), "finite"),
    ],
)
def test_window_features_validation(times, values, match):
    with pytest.raises(ValueError, match=match):
        window_features(times, values, DEFAULT_LIMIT)


def test_short_window_still_fills_the_long_lag_features():
    # With only five samples the eight-step lag falls back to the longest
    # available span rather than indexing out of the array.
    t, y = _window(n=5, seed=7)
    x = window_features(t, y, DEFAULT_LIMIT)
    assert np.isfinite(x[FEATURE_NAMES.index("lag8_rate")])
