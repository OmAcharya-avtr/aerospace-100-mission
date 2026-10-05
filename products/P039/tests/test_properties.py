"""Property-based tests for the algebraic identities, with Hypothesis.

These cover the identities that must hold for *any* input, which is exactly
what a fixed known-answer test cannot establish: exact additivity of the
sum-of-stages moments, scale equivariance of the Fenton-Wilkinson quantile,
the n^-1/2 scaling of the order-statistic standard error, and the
reparameterisation invariances of the OLS fit.

Deadlines are disabled and example counts are small: the suite shares a
single CPU core with four other builds.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from latencynet.analytic import (
    fenton_wilkinson_quantile,
    sum_of_stages_mean,
    sum_of_stages_variance,
)
from latencynet.conformal import conformal_radius
from latencynet.linear import ols_fit, ols_predict
from latencynet.pipeline import make_lognormal_pipeline, sample_stage_latencies
from latencynet.tails import lognormal_quantile, lognormal_quantile_se, quantile
from latencynet.units import s_to_us, us_to_s

SETTINGS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

_positive = st.floats(min_value=1e-6, max_value=1e-1, allow_nan=False, allow_infinity=False)
_probability = st.floats(min_value=0.01, max_value=0.9999)


@SETTINGS
@given(st.lists(_positive, min_size=1, max_size=8))
def test_mean_is_permutation_invariant_and_additive(values):
    arr = np.array(values)
    assert sum_of_stages_mean(arr) == pytest.approx(sum_of_stages_mean(arr[::-1]), rel=1e-13)
    assert sum_of_stages_mean(arr) == pytest.approx(float(np.sum(arr)), rel=0.0, abs=0.0)


@SETTINGS
@given(st.lists(_positive, min_size=1, max_size=8))
def test_independent_variance_is_the_sum_of_squares(values):
    arr = np.array(values)
    assert sum_of_stages_variance(arr) == pytest.approx(float(np.sum(arr**2)), rel=1e-14)


@SETTINGS
@given(
    st.lists(_positive, min_size=2, max_size=5),
    st.floats(min_value=-0.2, max_value=0.95),
    st.integers(min_value=0, max_value=2**20),
)
def test_sample_moment_identities_hold_for_any_draw(means, rho, seed):
    stds = [m * 0.3 for m in means]
    spec = make_lognormal_pipeline(means, stds, latent_rho=max(rho, -1.0 / (len(means) - 1)))
    trace = sample_stage_latencies(spec, 256, seed)
    total = trace.sum(axis=1)
    # Linearity of the mean is exact under any dependence.
    assert sum_of_stages_mean(trace.mean(axis=0)) == pytest.approx(total.mean(), rel=1e-12)
    # Variance additivity with the full covariance matrix is exact.
    cov = np.cov(trace, rowvar=False, ddof=1)
    sds = np.sqrt(np.diag(cov))
    assert sum_of_stages_variance(sds, cov) == pytest.approx(total.var(ddof=1), rel=1e-11)


@SETTINGS
@given(_positive, st.floats(min_value=1e-14, max_value=1e-2), _probability, _positive)
def test_fenton_wilkinson_is_scale_equivariant(mean, var, p, c):
    # Scaling a latency by c scales the mean by c and the variance by c^2, so
    # every quantile must scale by exactly c.
    base = fenton_wilkinson_quantile(mean, var, p)
    scaled = fenton_wilkinson_quantile(mean * c, var * c * c, p)
    assert scaled == pytest.approx(base * c, rel=1e-11)


@SETTINGS
@given(_positive, st.floats(min_value=1e-14, max_value=1e-2))
def test_fenton_wilkinson_is_monotone_in_probability(mean, var):
    qs = [fenton_wilkinson_quantile(mean, var, p) for p in (0.1, 0.5, 0.9, 0.99, 0.999)]
    assert qs == sorted(qs)


@SETTINGS
@given(_positive, st.floats(min_value=1e-12, max_value=1e-4))
def test_fenton_wilkinson_median_never_exceeds_the_mean(mean, var):
    # A lognormal is right-skewed, so its median is at or below its mean.
    assert fenton_wilkinson_quantile(mean, var, 0.5) <= mean * (1.0 + 1e-12)


@SETTINGS
@given(
    st.floats(min_value=-12.0, max_value=0.0),
    st.floats(min_value=0.05, max_value=1.5),
    _probability,
    st.integers(min_value=100, max_value=10_000),
)
def test_order_statistic_se_scales_as_inverse_root_n(mu, sigma, p, n):
    a = lognormal_quantile_se(mu, sigma, p, n)
    b = lognormal_quantile_se(mu, sigma, p, 4 * n)
    assert a / b == pytest.approx(2.0, rel=1e-10)
    # The relative standard error is scale-free: it does not depend on mu.
    q = lognormal_quantile(mu, sigma, p)
    q2 = lognormal_quantile(mu + 3.0, sigma, p)
    se2 = lognormal_quantile_se(mu + 3.0, sigma, p, n)
    assert a / q == pytest.approx(se2 / q2, rel=1e-10)


@SETTINGS
@given(
    st.lists(_positive, min_size=4, max_size=60),
    st.floats(min_value=0.02, max_value=0.98),
)
def test_nearest_rank_quantile_is_an_observation_and_monotone(values, p):
    arr = np.array(values)
    q = quantile(arr, p, "nearest_rank")
    assert q in set(arr.tolist())
    assert quantile(arr, min(p + 0.01, 0.99), "nearest_rank") >= q
    # The linear definition is bracketed by the sample range.
    assert arr.min() <= quantile(arr, p, "linear") <= arr.max()


@SETTINGS
@given(st.lists(st.floats(min_value=0.0, max_value=100.0), min_size=20, max_size=60))
def test_conformal_radius_is_monotone_in_level(residuals):
    arr = np.array(residuals)
    radii = [conformal_radius(arr, lvl) for lvl in (0.5, 0.7, 0.9)]
    assert radii == sorted(radii)
    assert all(r == float("inf") or 0.0 <= r <= arr.max() for r in radii)


@SETTINGS
@given(
    st.integers(min_value=0, max_value=2**20),
    st.floats(min_value=-5.0, max_value=5.0),
)
def test_ols_is_equivariant_under_shifting_the_target(seed, shift):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(40, 3))
    y = x @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.2, size=40)
    base = ols_predict(ols_fit(x, y), x)
    shifted = ols_predict(ols_fit(x, y + shift), x)
    assert np.allclose(shifted, base + shift, atol=1e-9)


@SETTINGS
@given(st.floats(min_value=-1e6, max_value=1e6))
def test_unit_round_trip(value):
    assert s_to_us(us_to_s(value)) == pytest.approx(value, rel=1e-12, abs=1e-9)


@SETTINGS
@given(
    st.floats(min_value=-12.0, max_value=0.0),
    st.floats(min_value=0.05, max_value=1.5),
    _probability,
)
def test_lognormal_quantile_round_trips_through_its_own_cdf(mu, sigma, p):
    from scipy import stats

    q = lognormal_quantile(mu, sigma, p)
    assert float(stats.norm.cdf((math.log(q) - mu) / sigma)) == pytest.approx(p, abs=1e-10)
