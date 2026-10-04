"""Forecast-verification metrics."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from constellink.metrics import (
    bootstrap_ci,
    brier_decomposition,
    brier_score,
    expected_calibration_error,
    log_loss_safe,
    reliability_curve,
    wilson_interval,
)


def test_brier_score_hand_values():
    # Perfect forecasts score 0; maximally wrong forecasts score 1.
    assert brier_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
    assert brier_score([0.0, 1.0], [1, 0]) == pytest.approx(1.0)
    # A constant 0.5 forecast scores 0.25 whatever the outcomes.
    assert brier_score([0.5] * 4, [1, 0, 1, 0]) == pytest.approx(0.25)


def test_brier_score_is_the_mean_squared_error():
    p = np.array([0.1, 0.4, 0.9])
    o = np.array([0, 1, 1])
    assert brier_score(p, o) == pytest.approx(float(np.mean((p - o) ** 2)))


@pytest.mark.parametrize("p,o", [([1.5], [1]), ([-0.1], [0]), ([0.5], [2]),
                                  ([], []), ([0.5, 0.5], [1])])
def test_brier_score_validation(p, o):
    with pytest.raises(ValueError):
        brier_score(p, o)


def test_log_loss_hand_value():
    # A single forecast of 0.5 with any outcome gives -ln(0.5) = 0.693147.
    assert log_loss_safe([0.5], [1]) == pytest.approx(np.log(2.0), rel=1e-12)


def test_log_loss_clips_rather_than_diverging():
    assert np.isfinite(log_loss_safe([0.0], [1]))
    assert log_loss_safe([0.0], [1]) > 20.0


def test_log_loss_validation():
    with pytest.raises(ValueError):
        log_loss_safe([0.5], [1], eps=0.6)


def test_decomposition_identity_exact_for_constant_forecast():
    # A constant forecast at the base rate: REL = 0, RES = 0, BS = UNC.
    o = np.array([1, 1, 1, 0])
    p = np.full(4, 0.75)
    d = brier_decomposition(p, o, n_bins=10)
    assert d.reliability == pytest.approx(0.0, abs=1e-15)
    assert d.resolution == pytest.approx(0.0, abs=1e-15)
    assert d.uncertainty == pytest.approx(0.75 * 0.25)
    assert d.brier == pytest.approx(d.uncertainty)
    assert d.identity_residual == pytest.approx(0.0, abs=1e-15)


def test_decomposition_identity_exact_by_distinct_value():
    rng = np.random.default_rng(0)
    p = rng.random(200)
    o = (rng.random(200) < p).astype(int)
    d = brier_decomposition(p, o, by_distinct_value=True)
    assert d.by_distinct_value
    assert abs(d.identity_residual) < 1e-12


def test_equal_width_residual_is_the_within_bin_component():
    # With continuous forecasts the three-term identity is NOT exact; the
    # residual is the Stephenson-Coelho-Jolliffe within-bin component and is
    # genuinely nonzero. This test asserts it is nonzero, i.e. that the code
    # is not quietly pretending otherwise.
    rng = np.random.default_rng(1)
    p = rng.random(500)
    o = (rng.random(500) < p).astype(int)
    d = brier_decomposition(p, o, n_bins=10)
    assert abs(d.identity_residual) > 1e-6
    # And it must vanish on distinct-value bins for the same data.
    e = brier_decomposition(p, o, by_distinct_value=True)
    assert abs(e.identity_residual) < 1e-12


def test_perfect_discrimination_gives_resolution_equal_to_uncertainty():
    p = np.array([0.0, 0.0, 1.0, 1.0])
    o = np.array([0, 0, 1, 1])
    d = brier_decomposition(p, o, n_bins=10)
    assert d.brier == pytest.approx(0.0)
    assert d.reliability == pytest.approx(0.0, abs=1e-15)
    assert d.resolution == pytest.approx(d.uncertainty, rel=1e-12)


def test_decomposition_counts_occupied_bins():
    p = np.array([0.05, 0.05, 0.95])
    o = np.array([0, 0, 1])
    d = brier_decomposition(p, o, n_bins=10)
    assert d.n_bins == 10
    assert d.n_occupied_bins == 2


def test_decomposition_rejects_too_few_bins():
    with pytest.raises(ValueError):
        brier_decomposition([0.5], [1], n_bins=1)


def test_ece_zero_for_a_calibrated_constant_forecast():
    o = np.array([1, 1, 1, 0])
    assert expected_calibration_error(np.full(4, 0.75), o) == pytest.approx(
        0.0, abs=1e-15)


def test_ece_hand_value():
    # One bin at p = 0.9 with observed frequency 0.5: ECE = |0.9 - 0.5| = 0.4.
    p = np.array([0.9, 0.9])
    o = np.array([1, 0])
    assert expected_calibration_error(p, o, n_bins=10) == pytest.approx(0.4)


def test_ece_is_not_a_proper_score():
    # The documented pathology: a base-rate constant forecast has ECE 0 with
    # no skill, while a sharp and nearly perfect forecast has ECE > 0.
    o = np.array([1, 1, 1, 0])
    flat = expected_calibration_error(np.full(4, 0.75), o)
    sharp = expected_calibration_error(np.array([0.95, 0.95, 0.95, 0.05]), o)
    assert flat < sharp
    assert brier_score(np.full(4, 0.75), o) > brier_score(
        np.array([0.95, 0.95, 0.95, 0.05]), o)


def test_ece_rejects_too_few_bins():
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [1], n_bins=1)


def test_wilson_interval_brackets_the_proportion():
    for successes, trials in ((0, 10), (5, 10), (10, 10), (1, 3), (3, 3)):
        lo, hi = wilson_interval(successes, trials)
        phat = successes / trials
        assert lo <= phat <= hi
        assert 0.0 <= lo <= hi <= 1.0


def test_wilson_interval_narrows_with_sample_size():
    lo_s, hi_s = wilson_interval(5, 10)
    lo_l, hi_l = wilson_interval(500, 1000)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_wilson_interval_zero_trials():
    assert wilson_interval(0, 0) == (0.0, 1.0)


@pytest.mark.parametrize("args", [(-1, 10), (11, 10), (5, -1)])
def test_wilson_interval_validation(args):
    with pytest.raises(ValueError):
        wilson_interval(*args)


def test_reliability_curve_shapes_and_empty_bins():
    p = np.array([0.05, 0.05, 0.95])
    o = np.array([0, 1, 1])
    rc = reliability_curve(p, o, n_bins=10)
    assert rc.bin_centre.shape == (10,)
    assert rc.count.sum() == 3
    assert np.isnan(rc.mean_forecast[5])
    assert not np.isnan(rc.mean_forecast[0])
    assert rc.observed_frequency[0] == pytest.approx(0.5)


def test_reliability_curve_intervals_bracket_the_frequency():
    rng = np.random.default_rng(2)
    p = rng.random(300)
    o = (rng.random(300) < p).astype(int)
    rc = reliability_curve(p, o, n_bins=10)
    m = rc.count > 0
    assert np.all(rc.ci_low[m] <= rc.observed_frequency[m] + 1e-15)
    assert np.all(rc.observed_frequency[m] <= rc.ci_high[m] + 1e-15)


def test_reliability_curve_rejects_too_few_bins():
    with pytest.raises(ValueError):
        reliability_curve([0.5], [1], n_bins=1)


def test_bootstrap_ci_contains_the_point_estimate():
    rng = np.random.default_rng(3)
    p = rng.random(300)
    o = (rng.random(300) < p).astype(int)
    pt, lo, hi = bootstrap_ci(p, o, n_boot=400, seed=5)
    assert lo <= pt <= hi
    assert pt == pytest.approx(brier_score(p, o))


def test_bootstrap_ci_is_deterministic_given_a_seed():
    rng = np.random.default_rng(4)
    p = rng.random(100)
    o = (rng.random(100) < p).astype(int)
    assert bootstrap_ci(p, o, n_boot=200, seed=1) == bootstrap_ci(
        p, o, n_boot=200, seed=1)


@pytest.mark.parametrize("kwargs", [{"n_boot": 5}, {"alpha": 0.0},
                                    {"alpha": 0.9}])
def test_bootstrap_ci_validation(kwargs):
    with pytest.raises(ValueError):
        bootstrap_ci([0.5, 0.5], [1, 0], **kwargs)


@given(n=st.integers(20, 200), seed=st.integers(0, 1000))
@settings(max_examples=20, deadline=None)
def test_brier_is_bounded(n, seed):
    # Algebraic bound: 0 <= BS <= 1 for probabilities in [0, 1].
    rng = np.random.default_rng(seed)
    p = rng.random(n)
    o = rng.integers(0, 2, size=n)
    assert 0.0 <= brier_score(p, o) <= 1.0
