"""Channel model and analytic crossing-rate tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from linkoutage.channel import (
    amplitude_threshold_to_gaussian_level,
    analytic_down_crossing_probability,
    analytic_level_crossing_rate,
    analytic_mean_fade_duration,
    analytic_outage_fraction,
    ar1_unit_variance,
    conditional_onset_probability,
    correlation_length_samples,
    gaussian_level_to_amplitude_threshold,
    lognormal_amplitude_series,
    rho_from_tau,
    sigma_ln_i_from_si,
)
from linkoutage.fade import below_threshold, level_crossing_rate


def test_rho_from_tau_known_value():
    # tau * fs = 200 samples -> rho = exp(-1/200) = exp(-0.005)
    assert rho_from_tau(2.0e-4, 1.0e6) == pytest.approx(math.exp(-0.005), rel=0, abs=1e-16)
    assert rho_from_tau(2.0e-4, 1.0e6) == pytest.approx(0.995012479192682, abs=1e-15)


def test_correlation_length():
    assert correlation_length_samples(2.0e-4, 1.0e6) == pytest.approx(200.0)


def test_sigma_from_si_known_value():
    # sigma**2 = ln(1 + 0.6) = ln(1.6) = 0.470003629245736
    assert sigma_ln_i_from_si(0.6) ** 2 == pytest.approx(0.470003629245736, abs=1e-15)


def test_sigma_from_si_zero():
    assert sigma_ln_i_from_si(0.0) == 0.0


def test_ar1_matches_direct_recursion():
    rng = np.random.default_rng(5)
    z = rng.standard_normal(5000)
    rho = 0.995012479192682
    x = ar1_unit_variance(z, rho)
    direct = np.empty_like(z)
    direct[0] = z[0]
    s = math.sqrt(1.0 - rho * rho)
    for i in range(1, z.size):
        direct[i] = rho * direct[i - 1] + s * z[i]
    assert np.max(np.abs(x - direct)) < 1e-13


def test_ar1_first_sample_is_stationary_initial_condition():
    z = np.random.default_rng(0).standard_normal(100)
    x = ar1_unit_variance(z, 0.9)
    assert x[0] == pytest.approx(z[0], abs=1e-15)


def test_ar1_rho_zero_is_identity():
    z = np.random.default_rng(1).standard_normal(50)
    assert np.allclose(ar1_unit_variance(z, 0.0), z)


def test_ar1_is_unit_variance():
    z = np.random.default_rng(2).standard_normal(400_000)
    x = ar1_unit_variance(z, 0.99)
    assert x.var() == pytest.approx(1.0, rel=0.03)


def test_ar1_autocorrelation_is_rho():
    z = np.random.default_rng(3).standard_normal(400_000)
    rho = 0.98
    x = ar1_unit_variance(z, rho)
    c = x - x.mean()
    lag1 = float(np.dot(c[:-1], c[1:]) / np.dot(c, c))
    assert lag1 == pytest.approx(rho, abs=0.003)


@pytest.mark.parametrize("rho", [-0.1, 1.0, 1.5, float("nan")])
def test_ar1_rejects_bad_rho(rho):
    with pytest.raises(ValueError, match="rho"):
        ar1_unit_variance(np.zeros(5), rho)


def test_series_has_unit_mean_irradiance(short_channel):
    assert float(short_channel.irradiance.mean()) == pytest.approx(1.0, rel=0.02)


def test_series_amplitude_is_sqrt_irradiance(short_channel):
    assert np.allclose(short_channel.amplitude**2, short_channel.irradiance)


def test_series_describe_names_the_generator(short_channel):
    text = short_channel.describe()
    assert "default_rng(2049)" in text
    assert "standard_normal(200000)" in text


def test_series_duration(short_channel):
    assert short_channel.duration_s == pytest.approx(0.2)


def test_threshold_level_round_trip():
    u = amplitude_threshold_to_gaussian_level(0.6, 0.6)
    assert gaussian_level_to_amplitude_threshold(u, 0.6) == pytest.approx(0.6, rel=1e-12)


def test_threshold_level_known_value():
    # u = (2 ln 0.6 + ln(1.6)/2) / sqrt(ln 1.6)
    sigma = math.sqrt(math.log(1.6))
    expected = (2.0 * math.log(0.6) + sigma * sigma / 2.0) / sigma
    assert amplitude_threshold_to_gaussian_level(0.6, 0.6) == pytest.approx(expected, abs=1e-15)


def test_amplitude_below_threshold_iff_gaussian_below_level(short_channel):
    u = amplitude_threshold_to_gaussian_level(0.6, short_channel.si)
    by_amplitude = below_threshold(short_channel.amplitude, 0.6)
    by_level = short_channel.gaussian < u
    # The map x -> a is strictly increasing, so the two masks must agree.
    assert int(np.count_nonzero(by_amplitude != by_level)) == 0


def test_analytic_down_crossing_two_independent_routes_agree():
    u = amplitude_threshold_to_gaussian_level(0.6, 0.6)
    rho = rho_from_tau(2.0e-4, 1.0e6)
    a = analytic_down_crossing_probability(u, rho, method="mvn")
    b = analytic_down_crossing_probability(u, rho, method="quad")
    assert abs(a - b) < 1e-12


def test_analytic_down_crossing_independent_case():
    # rho = 0: P(x0 >= u) P(x1 < u) = (1 - Phi(u)) Phi(u).
    from scipy.stats import norm

    u = -1.0
    expected = float((1.0 - norm.cdf(u)) * norm.cdf(u))
    assert analytic_down_crossing_probability(u, 0.0) == pytest.approx(expected, abs=1e-10)


def test_analytic_down_crossing_falls_with_correlation():
    u = -1.0
    low = analytic_down_crossing_probability(u, 0.99)
    high = analytic_down_crossing_probability(u, 0.5)
    assert low < high


def test_analytic_outage_fraction_matches_sample(short_channel):
    expected = analytic_outage_fraction(0.6, 0.6)
    measured = float(np.mean(short_channel.amplitude < 0.6))
    assert measured == pytest.approx(expected, rel=0.08)


def test_analytic_lcr_matches_sample_lcr(short_channel):
    analytic = analytic_level_crossing_rate(0.6, si=0.6, tau_s=2.0e-4, fs_hz=1.0e6)
    sample = level_crossing_rate(short_channel.amplitude, 0.6, 1.0e6)
    # ~1640 crossings in 200 000 samples -> about 2.5 % relative standard error.
    assert sample == pytest.approx(analytic, rel=0.08)


def test_analytic_mean_fade_duration_is_outage_over_rate():
    rate = analytic_level_crossing_rate(0.6, si=0.6, tau_s=2.0e-4, fs_hz=1.0e6)
    frac = analytic_outage_fraction(0.6, 0.6)
    mfd = analytic_mean_fade_duration(0.6, si=0.6, tau_s=2.0e-4, fs_hz=1.0e6)
    assert mfd == pytest.approx(frac / rate, rel=1e-12)


def test_analytic_lcr_rises_with_threshold():
    low = analytic_level_crossing_rate(0.3, si=0.6, tau_s=2.0e-4, fs_hz=1.0e6)
    high = analytic_level_crossing_rate(0.6, si=0.6, tau_s=2.0e-4, fs_hz=1.0e6)
    assert low < high


def test_conditional_onset_horizon_one_closed_form():
    from scipy.stats import norm

    u = -1.5
    rho = 0.9
    s = math.sqrt(1.0 - rho * rho)
    c = np.array([-2.0, -1.0, 0.0, 2.0])
    expected = 1.0 - np.exp(-np.where(c >= u, norm.cdf((u - rho * c) / s), 0.0))
    got = conditional_onset_probability(c, level=u, rho=rho, horizon_samples=1)
    assert np.allclose(got, expected, atol=1e-12)


def test_conditional_onset_is_in_unit_interval():
    p = conditional_onset_probability(
        np.linspace(-6.0, 6.0, 41), level=-2.0, rho=0.99, horizon_samples=50
    )
    assert np.all(p >= 0.0) and np.all(p < 1.0)


def test_conditional_onset_decays_far_above_the_level():
    g = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    p = conditional_onset_probability(g, level=-2.5, rho=0.995, horizon_samples=100)
    assert np.all(np.diff(p) <= 1e-12)


def test_conditional_onset_matches_monte_carlo():
    # Direct simulation of the AR(1) conditional law, same definitions.
    u = -1.0
    rho = 0.9
    horizon = 5
    c0 = 0.5
    rng = np.random.default_rng(11)
    n = 200_000
    s = math.sqrt(1.0 - rho * rho)
    x = np.full(n, c0)
    hit = np.zeros(n, dtype=bool)
    prev = x
    for _ in range(horizon):
        nxt = rho * prev + s * rng.standard_normal(n)
        hit |= (prev >= u) & (nxt < u)
        prev = nxt
    mc = float(hit.mean())
    analytic = float(
        conditional_onset_probability(c0, level=u, rho=rho, horizon_samples=horizon)
    )
    # 1 - exp(-m) overstates P(at least one) when crossings clump, so the
    # analytic value sits slightly above the Monte Carlo estimate.
    assert analytic >= mc - 0.005
    assert analytic - mc < 0.02


def test_conditional_onset_scalar_returns_scalar():
    out = conditional_onset_probability(0.0, level=-1.0, rho=0.9, horizon_samples=3)
    assert np.ndim(out) == 0


@pytest.mark.parametrize("si", [-0.1, float("nan")])
def test_bad_si_raises(si):
    with pytest.raises(ValueError, match="scintillation index"):
        sigma_ln_i_from_si(si)


@pytest.mark.parametrize("tau", [0.0, -1.0, float("inf")])
def test_bad_tau_raises(tau):
    with pytest.raises(ValueError, match="correlation time"):
        rho_from_tau(tau, 1.0e6)


def test_bad_fs_raises():
    with pytest.raises(ValueError, match="sample rate"):
        rho_from_tau(1.0e-4, 0.0)


def test_zero_si_threshold_mapping_raises():
    with pytest.raises(ValueError, match="non-fading"):
        amplitude_threshold_to_gaussian_level(0.6, 0.0)


def test_bad_threshold_raises():
    with pytest.raises(ValueError, match="positive"):
        amplitude_threshold_to_gaussian_level(0.0, 0.6)


def test_bad_method_raises():
    with pytest.raises(ValueError, match="method"):
        analytic_down_crossing_probability(-1.0, 0.5, method="nope")


def test_zero_samples_raises():
    with pytest.raises(ValueError, match="n_samples"):
        lognormal_amplitude_series(0, fs_hz=1.0, tau_s=1.0, si=0.5, seed=1)


def test_bad_horizon_raises():
    with pytest.raises(ValueError, match="horizon_samples"):
        conditional_onset_probability(0.0, level=-1.0, rho=0.5, horizon_samples=0)


def test_mean_irradiance_scaling():
    s = lognormal_amplitude_series(
        100_000, fs_hz=1.0e6, tau_s=1.0e-4, si=0.4, seed=9, mean_irradiance=4.0
    )
    assert float(s.irradiance.mean()) == pytest.approx(4.0, rel=0.05)
