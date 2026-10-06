"""Webb distribution: exact moments, the two limits, normalisation, validation."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from photoncount import webb


@pytest.mark.parametrize(("m", "f"), [(10.0, 2.0), (100.0, 2.0), (50.0, 3.0), (10.0, 5.0)])
def test_closed_form_moments_match_numerical_quadrature(m, f):
    """The closed forms (W2) are checked against the density itself, not assumed."""
    params = webb.WebbParameters(m, f)
    sd = np.sqrt(m * f)
    lo = max(webb.support_lower_bound(params), m - 60 * sd)
    grid = np.linspace(lo, m + 200 * sd, 2_000_001)
    dens = webb.pdf(grid, params)
    norm = np.trapezoid(dens, grid)
    mean = np.trapezoid(grid * dens, grid) / norm
    var = np.trapezoid((grid - mean) ** 2 * dens, grid) / norm
    third = np.trapezoid((grid - mean) ** 3 * dens, grid) / norm
    closed = webb.moments(params)
    assert norm == pytest.approx(1.0, abs=1e-6)
    assert mean == pytest.approx(closed["mean"], rel=1e-5)
    assert var == pytest.approx(closed["variance"], rel=1e-4)
    assert third == pytest.approx(closed["third_central"], rel=2e-3)


def test_skewness_known_answer():
    # m = 100, F = 2: third central = 3*100*2*1 = 600, variance = 200,
    # skewness = 600 / 200^1.5 = 600 / 2828.427 = 0.2121320
    mom = webb.moments(webb.WebbParameters(100.0, 2.0))
    assert mom["third_central"] == pytest.approx(600.0)
    assert mom["skewness"] == pytest.approx(0.2121320, rel=1e-6)


def test_f_equal_one_is_exactly_gaussian():
    params = webb.WebbParameters(100.0, 1.0)
    grid = np.linspace(40.0, 160.0, 501)
    assert np.allclose(webb.pdf(grid, params), webb.gaussian_limit_pdf(grid, params), atol=0)
    assert np.allclose(
        webb.pdf(grid, params),
        stats.norm.pdf(grid, loc=100.0, scale=10.0),
        rtol=1e-12,
    )
    assert webb.shape_parameter(params) == float("inf")
    assert webb.support_lower_bound(params) == float("-inf")
    assert webb.moments(params)["skewness"] == 0.0


def test_gaussian_limit_is_approached_as_excess_noise_falls():
    """Sup-norm distance to N(m, mF) must fall monotonically as F -> 1."""
    m = 200.0
    grid = np.linspace(m - 6 * np.sqrt(m), m + 6 * np.sqrt(m), 2001)
    distances = []
    for f in (2.0, 1.5, 1.2, 1.05, 1.01):
        params = webb.WebbParameters(m, f)
        distances.append(
            float(np.max(np.abs(webb.pdf(grid, params) - webb.gaussian_limit_pdf(grid, params))))
        )
    assert all(b < a for a, b in zip(distances, distances[1:], strict=False))
    assert distances[-1] < 1e-4


def test_poisson_limit_total_variation_falls_as_inverse_sqrt_mean():
    """The honest Poisson limit: TV ~ m^-1/2, no better.

    Webb at F = 1 is exactly N(m, m), and the Gaussian approximation to a
    Poisson has total variation falling as m^-1/2. Going from m = 100 to
    m = 1000 must therefore divide the TV by close to sqrt(10) = 3.162.
    """
    tvs = {}
    for m in (100.0, 1000.0):
        k_max = int(m + 12 * np.sqrt(m))
        probs = webb.binned_pmf(webb.WebbParameters(m, 1.0), k_max)
        poisson = stats.poisson.pmf(np.arange(k_max + 1), m)
        tvs[m] = float(0.5 * np.abs(probs - poisson).sum())
    ratio = tvs[100.0] / tvs[1000.0]
    assert ratio == pytest.approx(np.sqrt(10.0), rel=0.05)
    assert tvs[1000.0] < tvs[100.0]


def test_binned_pmf_is_normalised_and_non_negative():
    params = webb.WebbParameters(60.0, 2.5)
    probs = webb.binned_pmf(params, 300)
    assert probs.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(probs >= 0.0)


def test_sampling_reproduces_closed_form_moments():
    params = webb.WebbParameters(100.0, 2.0)
    draws = webb.sample(params, 200_000, np.random.default_rng(3))
    closed = webb.moments(params)
    # Standard error of the mean is sqrt(mF/n) = sqrt(200/2e5) = 0.0316.
    assert draws.mean() == pytest.approx(closed["mean"], abs=5 * np.sqrt(200.0 / 200_000))
    assert draws.var(ddof=1) == pytest.approx(closed["variance"], rel=0.02)
    skew = float(((draws - draws.mean()) ** 3).mean() / draws.var() ** 1.5)
    assert skew == pytest.approx(closed["skewness"], abs=0.02)


def test_sampling_is_deterministic_for_a_seed():
    params = webb.WebbParameters(30.0, 2.0)
    a = webb.sample(params, 100, np.random.default_rng(7))
    b = webb.sample(params, 100, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_cdf_is_monotone_and_spans_zero_to_one():
    params = webb.WebbParameters(80.0, 2.0)
    grid = np.linspace(webb.support_lower_bound(params), 400.0, 3001)
    c = webb.cdf(grid, params)
    assert np.all(np.diff(c) >= -1e-12)
    assert c[0] == pytest.approx(0.0, abs=1e-6)
    assert c[-1] == pytest.approx(1.0, abs=1e-6)


def test_excess_noise_factor_limits():
    # k = 1 gives F = G exactly; k = 0 gives F = 2 - 1/G.
    assert webb.apd_excess_noise_factor(100.0, 1.0) == pytest.approx(100.0)
    assert webb.apd_excess_noise_factor(100.0, 0.0) == pytest.approx(1.99)
    assert webb.apd_excess_noise_factor(1.0, 0.5) == pytest.approx(1.0)
    # F is monotone increasing in k at fixed gain.
    ks = np.linspace(0.0, 1.0, 11)
    fs = [webb.apd_excess_noise_factor(50.0, float(k)) for k in ks]
    assert all(b > a for a, b in zip(fs, fs[1:], strict=False))


@pytest.mark.parametrize(
    "args",
    [(0.0, 2.0, 1.0), (-1.0, 2.0, 1.0), (10.0, 0.5, 1.0), (10.0, 2.0, 0.0), (10.0, 2.0, -1.0)],
)
def test_invalid_parameters_raise(args):
    with pytest.raises(ValueError):
        webb.WebbParameters(*args)


def test_invalid_excess_noise_inputs_raise():
    with pytest.raises(ValueError):
        webb.apd_excess_noise_factor(0.5, 0.1)
    with pytest.raises(ValueError):
        webb.apd_excess_noise_factor(10.0, 1.5)


def test_pdf_is_zero_below_the_support():
    params = webb.WebbParameters(10.0, 2.0)
    lo = webb.support_lower_bound(params)
    assert webb.pdf([lo - 1.0, lo - 100.0], params).tolist() == [0.0, 0.0]


def test_negative_size_and_bad_k_max_raise():
    params = webb.WebbParameters(10.0, 2.0)
    with pytest.raises(ValueError):
        webb.sample(params, -1, np.random.default_rng(0))
    with pytest.raises(ValueError):
        webb.binned_pmf(params, -1)


@given(
    st.floats(min_value=5.0, max_value=500.0),
    st.floats(min_value=1.0, max_value=6.0),
)
@settings(max_examples=40, deadline=None)
def test_variance_always_at_least_mean(m, f):
    """An APD can only add noise: Var >= mean, with equality only at F = 1."""
    mom = webb.moments(webb.WebbParameters(m, f))
    assert mom["variance"] >= mom["mean"] - 1e-12
    assert mom["third_central"] >= -1e-12
