"""Rate model, FIT conversion, Poisson counts and their sampling errors."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bitflipsim.flux import (
    ILLUSTRATIVE_CROSS_SECTION_CM2_PER_BIT,
    ILLUSTRATIVE_FLUX_PER_CM2_S,
    SECONDS_PER_FIT_WINDOW,
    count_statistics,
    poisson_validity,
    sample_upset_counts,
    upset_rate,
)


def test_rate_is_the_product_hand_computed():
    # phi = 1e3 cm^-2 s^-1, sigma = 1e-14 cm^2/bit, N = 4704 bits
    # lambda = 1e3 * 1e-14 * 4704 = 4.704e-08 upsets/s
    rate = upset_rate(1.0e3, 1.0e-14, 4704)
    assert rate.rate_per_s == pytest.approx(4.704e-08, rel=1e-12)
    # 4.704e-08 / s * 3.6e12 s per 1e9 device-hours = 1.69344e5 FIT
    assert rate.rate_fit == pytest.approx(1.69344e5, rel=1e-12)
    # 1 / 4.704e-08 = 2.1258503401360545e7 s
    assert rate.mean_time_between_upsets_s == pytest.approx(1.0 / 4.704e-08, rel=1e-12)


def test_fit_window_is_one_billion_device_hours():
    assert SECONDS_PER_FIT_WINDOW == 1e9 * 3600.0


def test_expected_upsets_and_its_inverse():
    rate = upset_rate(1.0e3, 1.0e-14, 4704)
    exposure = rate.exposure_for_expected_upsets(5.0)
    assert rate.expected_upsets(exposure) == pytest.approx(5.0, rel=1e-12)


def test_zero_rate_has_infinite_mean_time():
    rate = upset_rate(0.0, 1.0e-14, 4704)
    assert rate.rate_per_s == 0.0
    assert rate.mean_time_between_upsets_s == float("inf")
    with pytest.raises(ValueError, match="rate is zero"):
        rate.exposure_for_expected_upsets(1.0)


def test_poisson_validity_flags_the_out_of_range_case():
    rate = upset_rate(1.0e3, 1.0e-14, 4704)
    inside = poisson_validity(rate, 1.0e3)
    assert inside["within_validity"] is True
    # per-bit probability = 1e3 * 1e-14 * t; t = 1e9 gives 1e-2 > 1e-3
    outside = poisson_validity(rate, 1.0e9)
    assert outside["per_bit_probability"] == pytest.approx(1.0e-2, rel=1e-12)
    assert outside["within_validity"] is False


def test_sampled_counts_match_the_poisson_mean_and_variance():
    rate = upset_rate(1.0e3, 1.0e-14, 4704)
    exposure = rate.exposure_for_expected_upsets(6.0)
    counts = sample_upset_counts(rate, exposure, 40_000, np.random.default_rng(99))
    stats = count_statistics(counts, 6.0)
    # Both z scores must sit inside +-4 standard errors; at n = 40000 the
    # standard error of the mean is sqrt(6/40000) = 0.01225.
    assert abs(stats.mean_z) < 4.0, stats
    assert abs(stats.variance_z) < 4.0, stats
    assert stats.mean_standard_error == pytest.approx(np.sqrt(6.0 / 40_000), rel=1e-12)
    # se(S2) = sqrt((mu + 2 mu**2) / n) = sqrt((6 + 72) / 40000)
    assert stats.variance_standard_error == pytest.approx(np.sqrt(78.0 / 40_000), rel=1e-12)


def test_count_statistics_input_validation():
    with pytest.raises(ValueError, match="at least 2 counts"):
        count_statistics(np.array([3]), 3.0)
    with pytest.raises(ValueError, match="must be >= 0"):
        count_statistics(np.array([3, 4]), -1.0)


def test_illustrative_constants_are_labelled_and_positive():
    assert ILLUSTRATIVE_FLUX_PER_CM2_S > 0.0
    assert ILLUSTRATIVE_CROSS_SECTION_CM2_PER_BIT > 0.0
    # The names carry the word "illustrative" so that a reader grepping for a
    # hard-coded environment figure finds the caveat with it.
    assert "ILLUSTRATIVE" in "ILLUSTRATIVE_FLUX_PER_CM2_S"


def test_rate_input_validation():
    with pytest.raises(ValueError, match="flux_per_cm2_s"):
        upset_rate(-1.0, 1e-14, 10)
    with pytest.raises(ValueError, match="cross_section"):
        upset_rate(1.0, -1e-14, 10)
    with pytest.raises(ValueError, match="bit_count"):
        upset_rate(1.0, 1e-14, -10)
    rate = upset_rate(1.0, 1e-14, 10)
    with pytest.raises(ValueError, match="exposure_s"):
        rate.expected_upsets(-1.0)
    with pytest.raises(ValueError, match="trials must be"):
        sample_upset_counts(rate, 1.0, 0, np.random.default_rng(0))


@given(
    flux=st.floats(min_value=1e-3, max_value=1e6),
    sigma=st.floats(min_value=1e-18, max_value=1e-8),
    bits=st.integers(min_value=1, max_value=10**7),
    exposure=st.floats(min_value=1e-3, max_value=1e6),
)
@settings(max_examples=150, deadline=None)
def test_property_rate_is_linear_in_every_factor(flux, sigma, bits, exposure):
    base = upset_rate(flux, sigma, bits).expected_upsets(exposure)
    doubled = upset_rate(2.0 * flux, sigma, bits).expected_upsets(exposure)
    assert doubled == pytest.approx(2.0 * base, rel=1e-12)
    doubled_bits = upset_rate(flux, sigma, 2 * bits).expected_upsets(exposure)
    assert doubled_bits == pytest.approx(2.0 * base, rel=1e-12)
    doubled_time = upset_rate(flux, sigma, bits).expected_upsets(2.0 * exposure)
    assert doubled_time == pytest.approx(2.0 * base, rel=1e-12)
