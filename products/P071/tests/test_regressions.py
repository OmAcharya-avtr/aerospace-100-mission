"""Pinned counterexamples. Every one was found by a property-based test here.

``.hypothesis/`` is not committed, so a falsifying example found by one run is
lost on a fresh clone. Each test below reproduces one defect with its literal
inputs, states what went wrong, and would fail again if the fix were reverted.

Every reference value is either a closed form or
:func:`scipy.stats.ncx2.cdf`, the non-central chi-square distribution
function, which is the exact collision probability whenever the
encounter-plane covariance is circular: with ``u = R^2 / sigma^2`` and
``v = d^2 / sigma^2``, ``Pc = P(chi'^2_2(v) <= u)``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import ncx2

from conjprob.analytic import (
    _y_mass,
    chan_recommended_order,
    pc_alfano,
    pc_chan,
    pc_foster,
)
from conjprob.dilution import max_probability
from conjprob.quadrature import pc_quadrature


def test_regression_absolute_quadrature_tolerance_broke_scale_invariance():
    """sigma=(2160, 2134) m, miss=(2, 18061) m, R=200 m, scaled by 19.

    With ``epsabs=1e-14`` in :func:`conjprob.analytic.pc_foster` the adaptive
    quadrature stopped on the first subdivision for any Pc at or below that
    value, and the unscaled and scaled geometries returned 1.2142e-18 and
    1.2791e-18, a 5.3 per cent disagreement in a quantity that is invariant by
    construction. Both were also wrong: the correct value is 1.30582e-18.
    """
    sigma_x, sigma_y, xbar, ybar, radius = 2160.0, 2134.0, 2.0, 18061.0, 200.0
    base = pc_foster(sigma_x, sigma_y, xbar, ybar, radius)
    factor = 19.0
    scaled = pc_foster(
        sigma_x * factor, sigma_y * factor, xbar * factor, ybar * factor, radius * factor
    )
    reference, abserr = pc_quadrature(
        np.diag([sigma_x**2, sigma_y**2]), (xbar, ybar), radius
    )
    assert scaled == pytest.approx(base, rel=1e-9)
    assert base == pytest.approx(reference, rel=1e-9, abs=10.0 * abserr)
    assert base == pytest.approx(1.3058241277669523e-18, rel=1e-9)


def test_regression_error_function_sum_cancelled_to_zero_in_the_far_tail():
    """sigma_y=3 m, ybar=68 m, half-chord 1 m: the published form returns 0.

    ``0.5 [erf((h - ybar) / (sigma sqrt 2)) + erf((h + ybar) / (sigma sqrt 2))]``
    evaluates both error functions at about -15.8 and +16.3, which round to -1
    and +1, so the sum is exactly zero. The correct mass is 8.767e-111. The
    complementary-error-function branch recovers it.
    """
    assert _y_mass(1.0, 68.0, 3.0) == pytest.approx(8.767418031863173e-111, rel=1e-9)
    value = pc_foster(3.0, 3.0, 0.0, 68.0, 1.0)
    reference, abserr = pc_quadrature(np.diag([9.0, 9.0]), (0.0, 68.0), 1.0)
    assert value > 0.0
    assert value == pytest.approx(reference, rel=1e-9, abs=10.0 * abserr)
    assert value == pytest.approx(1.0070601608147203e-111, rel=1e-9)


def test_regression_error_function_cancellation_also_hit_alfano():
    """The same cancellation was present in the Simpson implementation."""
    value = pc_alfano(3.0, 3.0, 0.0, 68.0, 1.0, nodes=2001)
    assert value > 0.0
    assert value == pytest.approx(1.0070601608147203e-111, rel=1e-3)


def test_regression_chan_finite_sum_lost_twelve_digits_to_cancellation():
    """sigma=1 m, ybar=8 m, R=1 m: published finite sum gives 4.27324e-13.

    The bracket ``1 - exp(-u/2) sum_{k<=m} (u/2)^k / k!`` is a difference of two
    numbers that both approach one, and for ``u = 1`` it is of order 1e-12 for
    modest ``m``. The exact value from the non-central chi-square distribution
    function is 4.2710148852895227e-13; the published form gave
    4.27323528473266e-13, a relative error of 5.2e-4. Evaluating the bracket as
    the regularised incomplete gamma function removes it.
    """
    exact = float(ncx2.cdf(1.0, 2, 64.0))
    assert exact == pytest.approx(4.2710148852895227e-13, rel=1e-12)
    assert pc_chan(1.0, 1.0, 0.0, 8.0, 1.0, order=60) == pytest.approx(exact, rel=1e-12)
    assert pc_foster(1.0, 1.0, 0.0, 8.0, 1.0) == pytest.approx(exact, rel=1e-12)
    assert pc_quadrature(np.eye(2), (0.0, 8.0), 1.0)[0] == pytest.approx(exact, rel=1e-10)


def test_regression_chan_fixed_order_sixty_is_not_enough_at_mahalanobis_nine():
    """sigma=1 m, ybar=9 m, R=9 m: order 60 is 2.0e-6 wrong, order 96 is exact.

    The series weights are Poisson with mean ``M^2 / 2``, so a fixed order is
    adequate only up to a fixed Mahalanobis distance.
    :func:`conjprob.analytic.chan_recommended_order` returns 96 here.
    """
    exact = float(ncx2.cdf(81.0, 2, 81.0))
    truncated = pc_chan(1.0, 1.0, 0.0, 9.0, 9.0, order=60)
    assert abs(truncated / exact - 1.0) > 1e-6
    order = chan_recommended_order(9.0)
    assert order == 96
    assert pc_chan(1.0, 1.0, 0.0, 9.0, 9.0, order=order) == pytest.approx(exact, rel=1e-12)


def test_regression_subnormal_miss_component_underflowed_the_mahalanobis_square():
    """xbar=2.2250738585072014e-308 m with sigma=1 m: hypot is positive, the square is not.

    :func:`conjprob.dilution.max_probability` divides by the squared Mahalanobis
    distance. A subnormal miss component leaves the unsquared hypotenuse
    strictly positive while its square underflows to zero, which turned a
    guarded condition into a division by zero. The function now raises
    ``ValueError`` and says which components underflowed.
    """
    tiny = 2.2250738585072014e-308
    assert math.hypot(tiny, 0.0) > 0.0
    assert math.hypot(tiny, 0.0) ** 2 == 0.0
    with pytest.raises(ValueError, match="underflowing miss vector"):
        max_probability(1.0, 1.0, tiny, 0.0, 1.0)


def test_regression_chan_series_overflowed_to_nan_at_large_mahalanobis_distance():
    """sigma=437 m, xbar=16515 m, R=1 m: the direct series returns nan.

    The Chan series accumulates ``(v/2)^m / m!`` and multiplies by ``exp(-v/2)``
    at the end. Here ``v / 2`` is 714, the accumulation overflows to infinity and
    the prefactor underflows to zero, so the product is nan. Evaluating the term
    weights in logarithms as a Poisson distribution removes it. The correct value
    is 1.92739845e-316, a subnormal double, which the Foster-Estes path also
    reaches.
    """
    order = chan_recommended_order(16515.0 / 437.0)
    value = pc_chan(437.0, 437.0, 16515.0, 0.0, 1.0, order=order)
    reference = pc_foster(437.0, 437.0, 16515.0, 0.0, 1.0)
    assert math.isfinite(value)
    assert value > 0.0
    assert value == pytest.approx(reference, rel=1e-6, abs=1e-320)
    assert value == pytest.approx(1.92739845e-316, rel=1e-6, abs=1e-320)
