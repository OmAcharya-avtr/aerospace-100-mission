"""Pinned regression tests.

Every test here reproduces a defect that was actually present in this
repository during the build, or a boundary case that a property-based search
would otherwise have to rediscover. The inputs are written as literals in the
test body on purpose: ``.hypothesis/`` is gitignored, so a falsifying example
found by one run is lost on a fresh clone.

Hypothesis search record for this build (2026-10-16): twelve properties were
searched with roughly ten thousand generated examples in total -- Fieller
containment, quantile/CDF round trips, interval monotonicity in the level,
Kaplan-Meier monotonicity and range, IPCW finiteness, affine equivariance of the
margin, non-negativity of CRPS and pinball, PIT range, and finiteness of the
transformed-normal quantile over nine decades of scale. **No falsifying example
was found.** The cases below are therefore the defects found by ordinary
development and review rather than by the search, plus the boundary cases the
search covered and that are pinned so that they stay covered.
"""

from __future__ import annotations

import numpy as np
import pytest

from degradepredict.baselines import linear_baseline
from degradepredict.forecast import QuantileForecast, TransformedNormalForecast
from degradepredict.generate import generate_population
from degradepredict.limits import LimitSpec, crossing_status, has_crossed
from degradepredict.metrics import crps

LIMIT = LimitSpec(value=80.0, direction="below", unit="percent of BOL", atol=1e-9)


def test_regression_event_flag_is_computed_against_the_snapped_origin():
    """Defect found 2026-10-16 by ``test_observed_duration_and_event_flag_are_consistent``.

    The generator drew a forecast-origin service age ``t0`` and then snapped the
    last sample down to the housekeeping cadence, but computed ``event`` from
    the unsnapped ``t0`` while computing ``remaining_true_days`` from the snapped
    origin. For a unit whose follow-up fell between the two remaining lives --
    trajectory 17 of ``seed=32`` had remaining 417.301 d against a follow-up of
    415.355 d -- the recorded ``event`` said complete while the recorded
    remaining life said censored. Every censored-coverage number in the
    repository is computed from those two fields, so the whole headline result
    would have been silently wrong.
    """
    for seed in (31, 32, 20261016):
        for tr in generate_population(300, seed=seed):
            assert tr.remaining_true_days == pytest.approx(
                tr.crossing_time_days - tr.origin_days
            )
            assert tr.remaining_observed_days == pytest.approx(
                min(tr.remaining_true_days, tr.followup_days)
            )
            assert tr.event == (tr.remaining_true_days <= tr.followup_days)
            assert tr.origin_days == pytest.approx(float(tr.times_days[-1]))


def test_regression_fieller_set_is_shifted_back_onto_the_time_axis():
    """Defect found during the build: the quadratic solves for ``u = w - xbar``.

    The first implementation returned the roots in ``u`` without adding ``xbar``
    back, which produced an interval that did not contain its own point
    estimate -- for the literal window below the reported interval was
    [299.8, 395.5] d around a point estimate of 429 d. The test is the
    containment property with literal inputs.
    """
    times = np.arange(40, dtype=float) * 7.0
    values = np.array(
        [100.21, 99.74, 99.63, 99.41, 99.08, 98.92, 98.70, 98.33, 98.26, 97.95,
         97.74, 97.43, 97.31, 96.98, 96.75, 96.54, 96.21, 96.09, 95.78, 95.52,
         95.33, 95.01, 94.84, 94.52, 94.33, 94.02, 93.85, 93.51, 93.34, 93.02,
         92.81, 92.55, 92.31, 92.06, 91.79, 91.58, 91.31, 91.07, 90.82, 90.55]
    )
    fit = linear_baseline(times, values, LIMIT)
    lo, hi = fit.fieller_interval_days
    assert lo < fit.crossing_days < hi
    assert fit.fieller_g < 1.0


def test_regression_unbounded_fieller_set_at_literal_inputs():
    """Boundary case: with ``g >= 1`` the exact set is a half line.

    Literal window with a slope that is not significantly different from zero.
    The upper Fieller endpoint must be ``+inf`` while the delta-method endpoint
    stays finite -- that disagreement is the point of computing both.
    """
    times = np.arange(12, dtype=float) * 7.0
    values = np.array(
        [99.4, 103.1, 96.8, 101.9, 97.2, 102.6, 96.1, 100.8, 98.3, 101.2, 97.5, 99.1]
    )
    fit = linear_baseline(times, values, LIMIT)
    assert fit.fieller_g >= 1.0
    assert fit.fieller_interval_days[1] == np.inf
    assert np.isfinite(fit.delta_interval_days[1])


def test_regression_scalar_argument_returns_a_python_float():
    """Defect found during the build on NumPy 2.5.

    ``float()`` of a one-element, one-dimensional array raises ``TypeError`` on
    NumPy 2.x. The forecast API therefore returns a scalar for a scalar
    argument and an array for an array argument.
    """
    f = QuantileForecast(
        levels=np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]),
        values_days=np.array([100.0, 150.0, 300.0, 500.0, 800.0, 1200.0, 1500.0]),
    )
    assert isinstance(f.quantile(0.5), float)
    assert isinstance(f.cdf(500.0), float)
    assert isinstance(f.quantile(np.array([0.25, 0.75])), np.ndarray)
    g = TransformedNormalForecast(mu_w=1000.0, sd_w=100.0, z=1.0, origin_days=400.0)
    assert isinstance(g.quantile(0.5), float)
    assert isinstance(g.cdf(600.0), float)


def test_regression_crps_grid_is_a_midpoint_rule():
    """Defect found during the build: the probability grid was mis-centred.

    ``np.linspace(0.005, 0.995, 199)`` leaves 0.0025 of probability unintegrated
    at each end, which biased the CRPS of a Uniform(100, 300) predictive at
    y = 180 to 18.7588 against the closed-form 18.6667, a relative error of
    5e-3. The midpoint grid now shipped reproduces the closed form to better
    than 1e-5 relative.
    """
    n = 401
    levels = np.linspace(1.0 / (n + 1), n / (n + 1), n)
    f = QuantileForecast(levels=levels, values_days=100.0 + levels * 200.0)
    u = (180.0 - 100.0) / 200.0
    expected = 200.0 * (u**2 - u + 1.0 / 3.0)
    assert float(crps([f], np.array([180.0]))[0]) == pytest.approx(expected, rel=1e-5)


def test_regression_quantile_crossing_is_rearranged_at_literal_inputs():
    """Independently fitted quantile levels can cross; the cumulative maximum repairs it."""
    f = QuantileForecast(
        levels=np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]),
        values_days=np.array([310.0, 290.0, 300.0, 505.0, 400.0, 1180.0, 1200.0]),
    )
    np.testing.assert_allclose(
        f.values_days, [310.0, 310.0, 310.0, 505.0, 505.0, 1180.0, 1200.0]
    )


def test_regression_crossing_boolean_is_not_claimed_inside_the_tolerance_band():
    """P058 class: a boolean from a strict threshold comparison is not portable.

    Two algebraically identical routes to the same margin, at literal inputs
    that sit one part in 1e16 from the limit. The tri-state result must say
    ``within_tolerance`` for a declared band of 1e-9, and the bare boolean is
    allowed to disagree between the two routes; nothing in this repository may
    assert that it does not.
    """
    limit_value = 80.0
    value_a = 80.0 + 1e-13
    value_b = (80.0 * 3.0 + 3e-13) / 3.0  # same number in exact arithmetic
    banded = LimitSpec(value=limit_value, atol=1e-9)
    assert crossing_status(value_a, banded) == "within_tolerance"
    assert crossing_status(value_b, banded) == "within_tolerance"
    bare = LimitSpec(value=limit_value, atol=0.0)
    # The bare booleans may or may not agree. Only the margins are compared, and
    # only to tolerance.
    assert abs((value_a - limit_value) - (value_b - limit_value)) < 1e-9
    assert isinstance(has_crossed(value_a, bare), bool)
