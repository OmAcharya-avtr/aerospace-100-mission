"""Predictive distribution objects."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from degradepredict.forecast import QuantileForecast, TransformedNormalForecast

LEVELS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])


def _qf(values=(100.0, 150.0, 300.0, 500.0, 800.0, 1200.0, 1500.0)):
    return QuantileForecast(levels=LEVELS, values_days=np.asarray(values, dtype=float))


def test_quantile_forecast_interval_reads_the_fitted_levels():
    f = _qf()
    lo, hi = f.interval(0.90)
    assert lo == pytest.approx(100.0)
    assert hi == pytest.approx(1500.0)
    assert f.median() == pytest.approx(500.0)


def test_quantile_forecast_rearranges_crossing_quantiles():
    # The fitted 0.75 level is below the 0.50 level; the cumulative maximum
    # repairs it (Chernozhukov, Fernandez-Val and Galichon 2010).
    f = _qf(values=(100.0, 150.0, 300.0, 500.0, 400.0, 1200.0, 1500.0))
    assert np.all(np.diff(f.values_days) >= 0.0)
    assert f.values_days[4] == pytest.approx(500.0)


def test_quantile_forecast_cdf_inverts_the_quantile_function():
    f = _qf()
    for level in (0.10, 0.25, 0.50, 0.75, 0.90):
        q = float(f.quantile(level))
        assert float(f.cdf(q)) == pytest.approx(level, abs=1e-6)


def test_quantile_forecast_extrapolates_flat_outside_the_fitted_levels():
    f = _qf()
    assert float(f.cdf(1.0)) == pytest.approx(0.05)
    assert float(f.cdf(1e6)) == pytest.approx(0.95)


@pytest.mark.parametrize(
    ("levels", "values", "match"),
    [
        (np.array([0.5]), np.array([1.0]), "at least two"),
        (np.array([0.5, 0.4]), np.array([1.0, 2.0]), "strictly increasing"),
        (np.array([0.0, 0.5]), np.array([1.0, 2.0]), "strictly inside"),
        (np.array([0.5, 0.6]), np.array([1.0]), "same shape"),
        (np.array([0.5, 0.6]), np.array([1.0, np.nan]), "finite"),
    ],
)
def test_quantile_forecast_validation(levels, values, match):
    with pytest.raises(ValueError, match=match):
        QuantileForecast(levels=levels, values_days=values)


def test_transformed_normal_with_z_one_is_a_shifted_truncated_normal():
    # Crossing time ~ N(1000, 100) truncated to exceed the origin of 400 d.
    # The truncation point is 6 standard deviations away, so the median
    # remaining life is 1000 - 400 = 600 d to within rounding.
    f = TransformedNormalForecast(mu_w=1000.0, sd_w=100.0, z=1.0, origin_days=400.0)
    assert f.median() == pytest.approx(600.0, abs=1e-3)
    lo, hi = f.interval(0.90)
    # 1.6449 standard deviations each side of the mean.
    assert lo == pytest.approx(600.0 - 1.6449 * 100.0, abs=0.5)
    assert hi == pytest.approx(600.0 + 1.6449 * 100.0, abs=0.5)


def test_transformed_normal_respects_the_origin_floor():
    f = TransformedNormalForecast(mu_w=420.0, sd_w=100.0, z=1.0, origin_days=400.0)
    assert float(f.quantile(1e-6)) > 0.0
    lo, _ = f.interval(0.99)
    assert lo > 0.0


def test_transformed_normal_cdf_inverts_its_quantile_function():
    f = TransformedNormalForecast(mu_w=1200.0**0.55, sd_w=1.5, z=0.55, origin_days=500.0)
    for p in (0.05, 0.25, 0.5, 0.75, 0.95):
        assert float(f.cdf(float(f.quantile(p)))) == pytest.approx(p, abs=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"z": 0.0}, "z must be finite and positive"),
        ({"sd_w": 0.0}, "sd_w must be finite and positive"),
        ({"origin_days": -1.0}, "origin_days"),
    ],
)
def test_transformed_normal_validation(kwargs, match):
    base = {"mu_w": 1000.0, "sd_w": 50.0, "z": 1.0, "origin_days": 100.0}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        TransformedNormalForecast(**base)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5])
def test_interval_rejects_levels_outside_the_unit_interval(level):
    with pytest.raises(ValueError, match="level must be in"):
        _qf().interval(level)


@pytest.mark.parametrize("p", [0.0, 1.0, np.nan])
def test_quantile_rejects_degenerate_probabilities(p):
    with pytest.raises(ValueError, match="strictly inside"):
        _qf().quantile(p)


@settings(max_examples=100, deadline=None)
@given(
    p=st.floats(min_value=0.001, max_value=0.999, allow_nan=False),
    q=st.floats(min_value=0.001, max_value=0.999, allow_nan=False),
)
def test_quantile_function_is_monotone(p, q):
    """A quantile function must be non-decreasing in its probability argument.

    Compared with a tolerance, because both endpoints come from interpolation
    of floating-point values.
    """
    f = _qf()
    lo, hi = (p, q) if p <= q else (q, p)
    assert float(f.quantile(hi)) >= float(f.quantile(lo)) - 1e-9
