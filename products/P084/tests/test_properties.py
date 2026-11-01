"""Property-based tests, where an algebraic identity exists to test.

Every tolerance here is tied to the CONDITIONING of the expression being
tested rather than to an absolute constant. Where a comparison involves a sum
of ``n`` floating-point terms its relative error grows like ``n * eps``, and
the tolerance says so in a comment.

Any falsifying example Hypothesis finds is pinned, with literal inputs, in
``tests/test_regressions.py``, because ``.hypothesis/`` is gitignored and a
counterexample found by one run is otherwise lost on a fresh clone.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from transfergap.bounds import (
    empirical_bernstein_mean_upper,
    empirical_bernstein_slack,
    hoeffding_mean_upper,
    required_n_hoeffding,
    required_n_wilks,
    tolerance_factor_normal,
)
from transfergap.exchangeability import half_split_statistic, trend_statistic
from transfergap.resample import conformal_order_index, split_conformal_quantile

EPS = float(np.finfo(np.float64).eps)

ranges = st.floats(min_value=1e-3, max_value=1e4, allow_nan=False, allow_infinity=False)
levels = st.floats(min_value=0.50, max_value=0.9999, allow_nan=False, allow_infinity=False)
counts = st.integers(min_value=2, max_value=5000)


@given(bound_range=ranges, confidence=levels, n=counts)
@settings(max_examples=300, deadline=None)
def test_hoeffding_slack_times_root_n_is_invariant(bound_range, confidence, n):
    """slack(n) * sqrt(n) = R sqrt(ln(1/delta)/2), independent of n.

    An exact identity of equation (3). The tolerance is 16 eps relative: the
    expression is three multiplications and a square root, each contributing
    at most half an ulp, and nothing cancels.
    """
    gaps = np.zeros(n)
    slack = hoeffding_mean_upper(gaps, bound_range, confidence).slack
    expected = bound_range * math.sqrt(math.log(1.0 / (1.0 - confidence)) / 2.0)
    assert slack * math.sqrt(n) == pytest.approx(expected, rel=16.0 * EPS, abs=0.0)


@given(
    values=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=2, max_size=60
    ),
    scale=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False),
    confidence=levels,
)
@settings(max_examples=300, deadline=None)
def test_hoeffding_bound_is_scale_equivariant(values, scale, confidence):
    """Scaling the gaps and R by c scales the bound by c.

    Exact in exact arithmetic. The tolerance follows the conditioning of a sum
    of n terms, whose relative error grows like n * eps, with a factor of 8 of
    headroom: ``8 * n * eps``. It is not an absolute constant, because the
    bound's magnitude spans six decades across this strategy.
    """
    gaps = np.asarray(values, dtype=np.float64)
    plain = hoeffding_mean_upper(gaps, 1.0, confidence).upper
    scaled = hoeffding_mean_upper(gaps * scale, scale, confidence).upper
    assert scaled == pytest.approx(
        plain * scale, rel=8.0 * gaps.size * EPS, abs=0.0
    )


@given(bound_range=ranges, confidence=levels, target=ranges)
@settings(max_examples=300, deadline=None)
def test_required_n_hoeffding_is_the_smallest_sufficient_n(bound_range, confidence, target):
    """The inversion of equation (10) is tight: n works and n-1 does not."""
    n = required_n_hoeffding(bound_range, target, confidence)
    factor = bound_range * math.sqrt(math.log(1.0 / (1.0 - confidence)) / 2.0)
    assert factor / math.sqrt(n) <= target * (1.0 + 16.0 * EPS)
    if n > 1:
        assert factor / math.sqrt(n - 1) > target * (1.0 - 16.0 * EPS)


@given(
    variance=st.floats(min_value=0.0, max_value=1e6, allow_nan=False),
    bound_range=ranges,
    confidence=levels,
    n=st.integers(min_value=3, max_value=5000),
)
@settings(max_examples=300, deadline=None)
def test_empirical_bernstein_slack_decreases_in_n(variance, bound_range, confidence, n):
    """Both terms of equation (5) fall with n, so the slack is strictly decreasing."""
    larger = empirical_bernstein_slack(n, variance, bound_range, confidence)
    smaller = empirical_bernstein_slack(n - 1, variance, bound_range, confidence)
    assert larger <= smaller


@given(proportion=levels, confidence=levels)
@settings(max_examples=300, deadline=None)
def test_required_n_wilks_satisfies_its_defining_inequality(proportion, confidence):
    """Equation (7): p^n <= delta and p^(n-1) > delta, checked in log space.

    The comparison is done on ``n ln p`` against ``ln delta`` rather than on
    ``p^n`` against ``delta``, because ``p^n`` underflows for the large n this
    strategy reaches and a comparison against an underflowed zero would pass
    vacuously. The tolerance is 8 eps relative on the logarithm, which is the
    conditioning of a single multiplication.
    """
    delta = 1.0 - confidence
    n = required_n_wilks(proportion, confidence)
    log_p = math.log(proportion)
    log_delta = math.log(delta)
    assert n * log_p <= log_delta * (1.0 - 8.0 * EPS)
    if n > 1:
        assert (n - 1) * log_p > log_delta * (1.0 + 8.0 * EPS)


@given(n=st.integers(min_value=1, max_value=10000), alpha=st.floats(0.001, 0.5))
@settings(max_examples=300, deadline=None)
def test_conformal_order_index_is_monotone_and_bounded(n, alpha):
    """k is non-decreasing in n and never exceeds n+1."""
    k = conformal_order_index(n, alpha)
    assert 1 <= k <= n + 1
    assert conformal_order_index(n + 1, alpha) >= k


@given(
    values=st.lists(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
        min_size=19,
        max_size=200,
        unique=True,
    )
)
@settings(max_examples=200, deadline=None)
def test_split_conformal_quantile_is_the_kth_order_statistic(values):
    """An order statistic, so the identity is exact and needs no tolerance."""
    scores = np.asarray(values, dtype=np.float64)
    k = conformal_order_index(scores.size, 0.05)
    assume(k <= scores.size)
    assert split_conformal_quantile(scores, 0.05) == float(np.sort(scores)[k - 1])


@given(
    values=st.lists(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False), min_size=3, max_size=120
    ),
    shift=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
    scale=st.floats(min_value=1e-6, max_value=1e6, allow_nan=False),
)
@settings(max_examples=300, deadline=None)
def test_trend_statistic_is_invariant_under_an_increasing_affine_map(values, shift, scale):
    """A rank statistic cannot see a monotone reparameterisation of its input.

    The ranks of ``scale * x + shift`` with ``scale > 0`` are the ranks of
    ``x``, so the statistic is computed from an identical rank vector. The
    tolerance is 1e-13 absolute on a quantity bounded in [-1, 1], where a
    relative bound has no meaning near zero.

    **The identity holds only where the map preserves distinctness in floating
    point, and Hypothesis found that.** At ``values = [5e-324, 0.0, 0.0]`` and
    ``scale = 0.5`` the multiplication underflows the denormal minimum to
    zero, the three values become tied, the average ranks go constant and the
    statistic drops from -0.8660254037844387 to 0.0. That is not a defect in
    the statistic: a strictly increasing real map is not strictly increasing
    in float64 once it collapses two representable numbers onto one. The
    assumption below states the precondition, and the counterexample is pinned
    with literal inputs in ``tests/test_regressions.py``.
    """
    array = np.asarray(values, dtype=np.float64)
    transformed = scale * array + shift
    assume(np.all(np.isfinite(transformed)))
    assume(np.unique(transformed).size == np.unique(array).size)
    assert trend_statistic(transformed) == pytest.approx(
        trend_statistic(array), rel=0.0, abs=1e-13
    )


@given(
    values=st.lists(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False), min_size=4, max_size=120
    ),
    shift=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
    scale=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False),
)
@settings(max_examples=300, deadline=None)
def test_half_split_statistic_is_affine_equivariant(values, shift, scale):
    """stat(a x + b) = a stat(x) for a > 0: the shift cancels in the difference.

    The tolerance follows the conditioning of the cancellation: the two half
    means are each of order ``|b| + a max|x|`` and their difference can be far
    smaller, so the bound is relative to the SCALE of the operands rather than
    to the result. That scale is ``a * (max|x| + 1) + |b|``.
    """
    array = np.asarray(values, dtype=np.float64)
    transformed = scale * array + shift
    assume(np.all(np.isfinite(transformed)))
    operand_scale = scale * (float(np.max(np.abs(array))) + 1.0) + abs(shift)
    assert half_split_statistic(transformed) == pytest.approx(
        scale * half_split_statistic(array),
        rel=0.0,
        abs=64.0 * array.size * EPS * operand_scale,
    )


@given(
    values=st.lists(
        st.floats(min_value=0.0, max_value=40.0, allow_nan=False), min_size=2, max_size=120
    ),
    confidence=levels,
)
@settings(max_examples=300, deadline=None)
def test_every_mean_bound_sits_above_the_sample_mean(values, confidence):
    """Both slacks of equations (3) and (5) are non-negative by construction."""
    gaps = np.asarray(values, dtype=np.float64)
    for bound in (
        hoeffding_mean_upper(gaps, 40.0, confidence),
        empirical_bernstein_mean_upper(gaps, 40.0, confidence),
    ):
        assert bound.upper >= bound.estimate
        assert bound.slack >= 0.0


@given(
    n=st.integers(min_value=3, max_value=2000),
    proportion=st.floats(min_value=0.6, max_value=0.999),
)
@settings(max_examples=200, deadline=None)
def test_normal_tolerance_factor_exceeds_the_normal_quantile(n, proportion):
    """k > z_p for every finite n: the factor pays for the unknown variance.

    An inequality from the non-central t construction of equation (9), and a
    bound that does not come from this implementation's own arithmetic.
    """
    from scipy import stats

    assert tolerance_factor_normal(n, proportion, 0.95) > float(stats.norm.ppf(proportion))
