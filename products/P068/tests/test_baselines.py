"""Analytic baselines: inverse regression, Fieller sets and failure modes."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy import stats

from degradepredict.baselines import (
    DegradationFitError,
    exponential_baseline,
    fit_transformed_ols,
    linear_baseline,
)
from degradepredict.limits import LimitSpec

LIMIT = LimitSpec(value=80.0, direction="below", unit="percent of BOL", atol=1e-9)


def _clean_linear(slope=-0.02, y0=100.0, n=60, dt=7.0, noise=0.0, seed=0):
    t = np.arange(n, dtype=float) * dt
    y = y0 + slope * t
    if noise > 0:
        y = y + np.random.default_rng(seed).normal(0.0, noise, size=n)
    return t, y


def test_linear_baseline_recovers_a_hand_calculated_crossing():
    # y = 100 - 0.02 t reaches 80 at t = 20/0.02 = 1000 d exactly. With a tiny
    # amount of noise the estimate must land close, not land exactly.
    t, y = _clean_linear(noise=1e-6, seed=1)
    fit = linear_baseline(t, y, LIMIT)
    assert fit.crossing_days == pytest.approx(1000.0, rel=1e-5)
    assert fit.slope == pytest.approx(-0.02, rel=1e-5)
    assert fit.metric_link == "identity"
    assert fit.z == 1.0


def test_exponential_baseline_recovers_a_hand_calculated_rate():
    # y = 100 exp(-k t) with k = ln(100/80)/1000 crosses 80 at t = 1000 d.
    k = np.log(100.0 / 80.0) / 1000.0
    t = np.arange(60, dtype=float) * 7.0
    y = 100.0 * np.exp(-k * t) * (1.0 + np.random.default_rng(2).normal(0.0, 1e-7, size=60))
    fit = exponential_baseline(t, y, LIMIT)
    assert fit.crossing_days == pytest.approx(1000.0, rel=1e-4)
    assert -fit.slope == pytest.approx(k, rel=1e-4)
    assert fit.metric_link == "log"


def test_fieller_interval_contains_the_point_estimate():
    t, y = _clean_linear(noise=0.3, seed=3)
    fit = linear_baseline(t, y, LIMIT)
    lo, hi = fit.fieller_interval_days
    assert lo < fit.crossing_days < hi


def test_fieller_and_delta_agree_when_g_is_small():
    """The delta-method interval is the g -> 0 limit of the Fieller set.

    What is guaranteed: the two endpoints agree to a relative tolerance that
    shrinks with g, not that they are equal. Here g is of order 1e-3 and the
    endpoints are required to agree to 1 percent.
    """
    t, y = _clean_linear(noise=0.3, seed=4)
    fit = linear_baseline(t, y, LIMIT)
    assert fit.fieller_g < 0.01
    for a, b in zip(fit.fieller_interval_days, fit.delta_interval_days, strict=True):
        assert a == pytest.approx(b, rel=0.01)


def test_fieller_set_is_unbounded_when_the_slope_is_not_significant():
    """With g >= 1 the exact confidence set is a half line, not an interval."""
    rng = np.random.default_rng(5)
    t = np.arange(20, dtype=float) * 7.0
    y = 100.0 - 1e-4 * t + rng.normal(0.0, 3.0, size=20)
    fit = linear_baseline(t, y, LIMIT)
    assert fit.fieller_g >= 1.0
    assert not np.isfinite(fit.fieller_interval_days[1])
    # The delta-method interval is bounded in exactly the case where the exact
    # set is not, which is the point of reporting both.
    assert np.isfinite(fit.delta_interval_days[1])


def test_fieller_g_matches_its_definition():
    # g = t_crit**2 * sigma**2 / (Sxx * slope**2), recomputed from the reported
    # pieces of the fit.
    t, y = _clean_linear(noise=0.4, seed=6)
    fit = linear_baseline(t, y, LIMIT, level=0.90)
    sxx = float(np.sum((t - t.mean()) ** 2))
    t_crit = stats.t.ppf(0.95, df=fit.n_samples - 2)
    expected = t_crit**2 * fit.sigma**2 / (sxx * fit.slope**2)
    assert fit.fieller_g == pytest.approx(expected, rel=1e-10)


def test_improving_window_raises_because_there_is_no_crossing_to_report():
    # A window whose metric is drifting upward has no crossing of a lower limit
    # to report, so the model must refuse rather than return a negative time.
    t = np.arange(30, dtype=float) * 7.0
    y = 95.0 + 0.01 * t + np.random.default_rng(7).normal(0.0, 0.05, size=30)
    with pytest.raises(DegradationFitError, match="not negative"):
        linear_baseline(t, y, LIMIT)


def test_window_already_below_the_limit_raises():
    t = np.arange(30, dtype=float) * 7.0
    y = 79.0 - 0.01 * t
    with pytest.raises(DegradationFitError, match="non-positive|not after"):
        linear_baseline(t, y, LIMIT)


def test_zero_residual_variance_raises():
    # Four points exactly on a line: the residual sum of squares evaluates to
    # exactly 0.0 here, so no interval is defined and the fit must say so.
    t = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([100.0, 95.0, 90.0, 85.0])
    with pytest.raises(DegradationFitError, match="zero residual variance"):
        linear_baseline(t, y, LIMIT)


def test_coincident_sample_times_raise():
    with pytest.raises(DegradationFitError, match="coincide"):
        linear_baseline(np.zeros(5), np.linspace(100.0, 90.0, 5), LIMIT)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"z": 0.0}, "z must be finite and positive"),
        ({"level": 0.0}, "level must be in"),
        ({"level": 1.0}, "level must be in"),
    ],
)
def test_fit_transformed_ols_validates_its_arguments(kwargs, match):
    t, y = _clean_linear(noise=0.2, seed=8)
    with pytest.raises(ValueError, match=match):
        fit_transformed_ols(t, y, LIMIT, **kwargs)


def test_fit_rejects_short_windows():
    with pytest.raises(ValueError, match="at least three samples"):
        linear_baseline(np.array([0.0, 7.0]), np.array([100.0, 99.0]), LIMIT)


def test_fit_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        linear_baseline(np.arange(5.0), np.arange(4.0), LIMIT)


def test_log_metric_rejects_non_positive_values():
    t = np.arange(10, dtype=float)
    y = np.linspace(5.0, -1.0, 10)
    with pytest.raises(ValueError, match="strictly positive"):
        exponential_baseline(t, y, LIMIT)


def test_upward_limit_is_refused_by_the_shipped_baselines():
    t, y = _clean_linear(noise=0.2, seed=9)
    with pytest.raises(ValueError, match="downward-degrading"):
        linear_baseline(t, y, LimitSpec(value=120.0, direction="above"))


def test_forecast_origin_defaults_to_the_last_sample():
    t, y = _clean_linear(noise=0.2, seed=10)
    fit = linear_baseline(t, y, LIMIT)
    assert fit.forecast.origin_days == pytest.approx(t[-1])
    assert fit.forecast.median() == pytest.approx(fit.crossing_days - t[-1], rel=0.05)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    scale=st.floats(min_value=0.2, max_value=5.0, allow_nan=False),
    offset=st.floats(min_value=-20.0, max_value=20.0, allow_nan=False),
    seed=st.integers(min_value=0, max_value=500),
)
def test_linear_crossing_is_invariant_under_an_affine_metric_rescaling(scale, offset, seed):
    """Rescaling y and the limit together leaves the crossing time unchanged.

    This is exact in exact arithmetic. In floating point the two estimates are
    required to agree to a relative tolerance; no bit-exact claim is made and no
    boolean derived from the crossing is compared.
    """
    t, y = _clean_linear(noise=0.3, seed=seed)
    base = linear_baseline(t, y, LIMIT)
    scaled = linear_baseline(
        t, scale * y + offset, LimitSpec(value=scale * LIMIT.value + offset, atol=1e-9)
    )
    assert scaled.crossing_days == pytest.approx(base.crossing_days, rel=1e-8)
    assert scaled.fieller_g == pytest.approx(base.fieller_g, rel=1e-8)
