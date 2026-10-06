"""Poisson counting statistics: known answers, limits and input validation."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from photoncount import poisson as pc


def test_photon_rate_known_answer():
    # Hand calculation: 1 W at 1550 nm.
    # rate = P * lambda / (h c) = 1 * 1.55e-6 / (6.62607015e-34 * 2.99792458e8)
    #      = 1.55e-6 / 1.98644586e-25 = 7.8029e18 photons/s
    rate = pc.photons_per_second(1.0, 1550e-9)
    assert rate == pytest.approx(7.8029e18, rel=1e-4)


def test_mean_counts_scales_with_efficiency_and_time():
    base = pc.mean_counts_from_power(1e-15, 1550e-9, 1e-6, 1.0)
    assert pc.mean_counts_from_power(1e-15, 1550e-9, 1e-6, 0.5) == pytest.approx(base / 2)
    assert pc.mean_counts_from_power(1e-15, 1550e-9, 2e-6, 1.0) == pytest.approx(2 * base)


def test_threshold_detection_probability_known_answers():
    # P(K >= 1) = 1 - exp(-mean) for a Poisson count.
    assert pc.threshold_detection_probability(1.0, 1) == pytest.approx(1.0 - np.exp(-1.0))
    # P(K >= 0) is 1 for any mean, including zero.
    assert pc.threshold_detection_probability(0.0, 0) == 1.0
    # A zero-mean slot never produces a count.
    assert pc.threshold_detection_probability(0.0, 1) == 0.0


def test_missed_and_detection_sum_to_one():
    for mean in (0.1, 1.0, 10.0):
        for thr in (1, 2, 5):
            assert pc.threshold_detection_probability(mean, thr) + pc.missed_detection_probability(
                mean, thr
            ) == pytest.approx(1.0)


def test_counting_snr_known_answer():
    # n_s = 10, n_b = 2: SNR = 10 / sqrt(12) = 2.8867513
    assert pc.counting_snr(10.0, 2.0) == pytest.approx(2.8867513, rel=1e-6)


def test_fano_factor_is_one_for_poisson_samples():
    rng = np.random.default_rng(0)
    draws = rng.poisson(50.0, size=200_000)
    f = pc.fano_factor(float(draws.mean()), float(draws.var(ddof=1)))
    # Standard error of a Fano estimate at n = 2e5 is about sqrt(2/n) = 0.0032.
    assert abs(f - 1.0) < 5 * np.sqrt(2.0 / 200_000)


def test_exact_count_interval_matches_published_garwood_values():
    # Garwood exact 95 % interval for an observed count of 5 is (1.623, 11.668).
    lo, hi = pc.exact_count_interval(5, 0.95)
    assert lo == pytest.approx(1.6235, abs=1e-3)
    assert hi == pytest.approx(11.6683, abs=1e-3)


def test_exact_count_interval_zero_count_has_zero_lower_bound():
    lo, hi = pc.exact_count_interval(0, 0.95)
    assert lo == 0.0
    # Upper bound is chi2.ppf(0.975, 2)/2 = 3.6889
    assert hi == pytest.approx(3.6889, abs=1e-3)


@pytest.mark.parametrize(
    ("func", "args"),
    [
        (pc.photons_per_second, (-1.0, 1550e-9)),
        (pc.photons_per_second, (1.0, 0.0)),
        (pc.photons_per_second, (1.0, -1.0)),
        (pc.mean_counts_from_power, (1.0, 1550e-9, 1e-6, 1.5)),
        (pc.mean_counts_from_power, (1.0, 1550e-9, 0.0, 1.0)),
        (pc.slot_mean, (-1.0,)),
        (pc.threshold_detection_probability, (1.0, -1)),
        (pc.counting_snr, (0.0, 1.0)),
        (pc.exact_count_interval, (-1,)),
        (pc.exact_count_interval, (3, 1.0)),
        (pc.fano_factor, (0.0, 1.0)),
    ],
)
def test_invalid_inputs_raise_value_error(func, args):
    with pytest.raises(ValueError):
        func(*args)


def test_non_finite_input_raises():
    with pytest.raises(ValueError):
        pc.slot_mean(float("nan"))


@given(
    st.floats(min_value=1e-3, max_value=50.0),
    st.integers(min_value=1, max_value=20),
)
@settings(max_examples=50, deadline=None)
def test_detection_probability_agrees_with_scipy(mean, threshold):
    assert pc.threshold_detection_probability(mean, threshold) == pytest.approx(
        float(stats.poisson.sf(threshold - 1, mean)), abs=1e-14
    )


@given(st.integers(min_value=0, max_value=400))
@settings(max_examples=40, deadline=None)
def test_exact_interval_brackets_the_count(count):
    lo, hi = pc.exact_count_interval(count, 0.95)
    assert lo <= count <= hi
    assert lo < hi
