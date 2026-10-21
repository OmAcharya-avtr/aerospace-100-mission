"""Three independently implemented analytic collision-probability paths.

All three evaluate the same encounter-plane integral

.. math::

    P_c = \\frac{1}{2\\pi\\sigma_x\\sigma_y}
          \\iint_{x^2+y^2 \\le R^2}
          \\exp\\!\\left[-\\tfrac12\\left(
          \\frac{(x-\\bar x)^2}{\\sigma_x^2}+\\frac{(y-\\bar y)^2}{\\sigma_y^2}
          \\right)\\right] \\, \\mathrm{d}x\\,\\mathrm{d}y

over the hard-body circle of radius ``R`` centred on the primary, with the miss
vector at ``(xbar, ybar)`` in the principal axes of the projected covariance.
Units: metres for ``sigma_x``, ``sigma_y``, ``xbar``, ``ybar`` and ``radius``;
the result is a dimensionless probability in ``[0, 1]``.

References
----------
Foster, J. L., and Estes, H. S. (1992). "A parametric analysis of orbital debris
collision probability and maneuver rate for space vehicles." NASA/JSC-25898,
NASA Johnson Space Center. The reduction of the area integral to one dimension
with an error function in the second coordinate.

Chan, F. K. (1997). "Collision probability analyses for Earth-orbiting
satellites." *Advances in the Astronautical Sciences*, 96, 1033-1048; and
Chan, F. K. (2008). *Spacecraft Collision Probability*, The Aerospace Press.
The equivalent-area series used by :func:`pc_chan`.

Alfano, S. (2005). "A numerical implementation of spherical object collision
probability." *Journal of the Astronautical Sciences*, 53(1), 103-109. The
fixed-node Simpson evaluation used by :func:`pc_alfano`.

Honest note on independence, which the validation reports as a measured result:
:func:`pc_foster` and :func:`pc_alfano` are two quadratures of the *same*
one-dimensional reduction, so their agreement is a numerical check and not an
independent derivation. :func:`pc_chan` is independent algebra, and
:func:`conjprob.quadrature.pc_quadrature` is an independent two-dimensional
integral that uses neither the error-function reduction nor the
eigendecomposition.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad
from scipy.special import erf, erfc, gammainc

_SQRT2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _check_inputs(
    sigma_x_m: float, sigma_y_m: float, xbar_m: float, ybar_m: float, radius_m: float
) -> None:
    for name, value in (
        ("sigma_x_m", sigma_x_m),
        ("sigma_y_m", sigma_y_m),
        ("xbar_m", xbar_m),
        ("ybar_m", ybar_m),
        ("radius_m", radius_m),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if sigma_x_m <= 0.0 or sigma_y_m <= 0.0:
        raise ValueError(
            "sigma_x_m and sigma_y_m must be strictly positive metres, got "
            f"({sigma_x_m!r}, {sigma_y_m!r})"
        )
    if radius_m < 0.0:
        raise ValueError(f"radius_m must be non-negative metres, got {radius_m!r}")


def _y_mass(half_chord_m: float, ybar_m: float, sigma_y_m: float) -> float:
    """Gaussian mass between ``-h`` and ``+h`` in the second coordinate.

    Mathematically ``0.5 [erf((h - ybar) / (sigma_y sqrt 2))
    + erf((h + ybar) / (sigma_y sqrt 2))]``, dimensionless, which is the exact
    inner integral and the step that turns the 2-D problem into a 1-D one.

    That expression is **not** what is evaluated. When the whole chord lies many
    standard deviations from the mean, both error functions round to plus and
    minus one and their sum cancels to exactly zero, so the published form
    returns 0 where the answer is small but non-zero. The failure is not
    hypothetical: it was found by a property-based test at ``sigma_y = 3 m``,
    ``ybar = 68 m``, ``h = 1 m``, where the correct mass is 8.77e-111 and the
    error-function form gives 0.0, and it is pinned in
    ``tests/test_regressions.py``. This implementation therefore uses the
    complementary error function on whichever side keeps both arguments
    positive, which has no cancellation. It still underflows to zero once the
    chord is beyond about 27 standard deviations; that floor is documented in
    README.md under Limitations.
    """
    scale = sigma_y_m * _SQRT2
    lower = (-half_chord_m - ybar_m) / scale
    upper = (half_chord_m - ybar_m) / scale
    if upper <= 0.0:
        return 0.5 * (erfc(-upper) - erfc(-lower))
    if lower >= 0.0:
        return 0.5 * (erfc(lower) - erfc(upper))
    return 0.5 * (erf(upper) + erf(-lower))


def _y_mass_array(
    half_chord_m: np.ndarray, ybar_m: float, sigma_y_m: float
) -> np.ndarray:
    """Vectorised :func:`_y_mass`, same stable branching, dimensionless."""
    scale = sigma_y_m * _SQRT2
    lower = (-half_chord_m - ybar_m) / scale
    upper = (half_chord_m - ybar_m) / scale
    both_negative = upper <= 0.0
    both_positive = lower >= 0.0
    straddling = ~(both_negative | both_positive)
    out = np.empty_like(half_chord_m, dtype=float)
    if np.any(both_negative):
        out[both_negative] = 0.5 * (
            erfc(-upper[both_negative]) - erfc(-lower[both_negative])
        )
    if np.any(both_positive):
        out[both_positive] = 0.5 * (
            erfc(lower[both_positive]) - erfc(upper[both_positive])
        )
    if np.any(straddling):
        out[straddling] = 0.5 * (
            erf(upper[straddling]) + erf(-lower[straddling])
        )
    return out


def pc_foster(
    sigma_x_m: float,
    sigma_y_m: float,
    xbar_m: float,
    ybar_m: float,
    radius_m: float,
    *,
    epsabs: float = 0.0,
    epsrel: float = 1e-11,
    limit: int = 400,
) -> float:
    """Foster-Estes short-encounter Pc by adaptive quadrature. Operational default.

    The area integral is reduced to one dimension with an error function in the
    second coordinate (Foster & Estes 1992), then the substitution
    ``x = R sin(theta)`` removes the square-root branch point at ``x = +/- R``
    so that the remaining integrand is smooth and adaptive quadrature converges
    at its full order.

    Parameters
    ----------
    sigma_x_m, sigma_y_m
        Principal standard deviations of the projected covariance, metres.
    xbar_m, ybar_m
        Miss-vector components on the principal axes, metres.
    radius_m
        Combined hard-body radius, metres.
    epsabs, epsrel, limit
        Passed to :func:`scipy.integrate.quad`. ``epsabs`` defaults to zero so
        that the tolerance is purely relative. A non-zero absolute tolerance
        destroys scale invariance: with the previous default of ``1e-14`` the
        quadrature stopped immediately for any Pc near or below that value, and
        a property-based test found a 5.3 per cent disagreement between a
        geometry and the same geometry with every length multiplied by 19. That
        case is pinned in ``tests/test_regressions.py``.

    Returns
    -------
    float
        Collision probability, dimensionless, in ``[0, 1]``.

    Notes
    -----
    Validity: rectilinear relative motion, constant combined covariance,
    negligible velocity uncertainty, encounter short compared with the orbital
    period. The quantity returned is the probability under the supplied
    covariance; it is not a statement about whether that covariance is right.
    """
    _check_inputs(sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m)
    if radius_m == 0.0:
        return 0.0

    def integrand(theta: float) -> float:
        x = radius_m * math.sin(theta)
        half_chord = radius_m * math.cos(theta)
        gauss = math.exp(-0.5 * ((x - xbar_m) / sigma_x_m) ** 2)
        return gauss * _y_mass(half_chord, ybar_m, sigma_y_m) * half_chord

    value, _ = quad(
        integrand,
        -0.5 * math.pi,
        0.5 * math.pi,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
    )
    return float(min(max(value / (sigma_x_m * _SQRT_2PI), 0.0), 1.0))


def pc_chan(
    sigma_x_m: float,
    sigma_y_m: float,
    xbar_m: float,
    ybar_m: float,
    radius_m: float,
    *,
    order: int = 3,
) -> float:
    """Chan's equivalent-area series Pc. Closed form, independent of the quadratures.

    The circle of radius ``R`` is mapped into the coordinates in which the
    covariance is the identity, where it becomes an ellipse, and is replaced by
    the circle of equal area. With

    ``u = R^2 / (sigma_x sigma_y)`` and
    ``v = xbar^2 / sigma_x^2 + ybar^2 / sigma_y^2``

    the probability is the convergent series

    ``Pc = exp(-v/2) sum_m (v/2)^m / m! [1 - exp(-u/2) sum_{k<=m} (u/2)^k / k!]``

    truncated at ``m = order`` (Chan 1997, 2008). The bracket is the cumulative
    distribution function of a chi-square variate with ``2 + 2m`` degrees of
    freedom at ``u``, and this implementation evaluates it as the regularised
    incomplete gamma function ``gammainc(m + 1, u / 2)`` rather than as the
    published finite exponential sum. The two are algebraically identical; the
    published form is a difference of two numbers that both approach one and
    loses up to twelve significant figures to cancellation, which a
    property-based test found as a 5.2e-4 relative error at ``sigma = 1 m``,
    ``ybar = 8 m``, ``R = 1 m`` against the non-central chi-square distribution
    function. That case is pinned in ``tests/test_regressions.py``.

    Parameters
    ----------
    sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m
        As for :func:`pc_foster`, metres.
    order
        Number of series terms beyond the leading one. The default of 3 is the
        small value Chan recommends and is adequate only while the Mahalanobis
        distance is of order one; the order required grows as ``M^2 / 2`` and
        :func:`chan_recommended_order` returns a value that has converged.
        ``validation/validate_analytic.py`` measures the truncation error
        against order and against ``M``.

    Returns
    -------
    float
        Collision probability, dimensionless, in ``[0, 1]``.

    Notes
    -----
    The equivalent-area substitution is exact only for a circular scaled
    cross-section, so the method carries an aspect-ratio error in addition to
    the truncation error. Both are measured as curves in
    ``validation/validate_regimes.py`` rather than asserted here.
    """
    _check_inputs(sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m)
    if order < 0:
        raise ValueError(f"order must be non-negative, got {order!r}")
    if radius_m == 0.0:
        return 0.0
    u = radius_m * radius_m / (sigma_x_m * sigma_y_m)
    v = (xbar_m / sigma_x_m) ** 2 + (ybar_m / sigma_y_m) ** 2
    half_u = 0.5 * u
    half_v = 0.5 * v
    if half_v <= 0.0:
        # Zero miss vector: all the weight is on the first term.
        return float(min(max(float(gammainc(1, half_u)), 0.0), 1.0))
    # The term weights are a Poisson distribution of mean half_v and are
    # evaluated in logarithms. Accumulating (half_v)^m / m! directly and
    # multiplying by exp(-half_v) at the end overflows to infinity while the
    # prefactor underflows to zero, giving nan: a property-based test found that
    # at sigma = 437 m, xbar = 16515 m, R = 1 m, where half_v is 714 and
    # exp(-714) is below the smallest subnormal. The case is pinned in
    # tests/test_regressions.py.
    log_half_v = math.log(half_v)
    total = 0.0
    for m in range(order + 1):
        log_weight = -half_v + m * log_half_v - math.lgamma(m + 1)
        if log_weight < -745.0:
            continue
        total += math.exp(log_weight) * float(gammainc(m + 1, half_u))
    return float(min(max(total, 0.0), 1.0))


def chan_recommended_order(mahalanobis: float) -> int:
    """Series order at which :func:`pc_chan` has converged, for a given Mahalanobis distance.

    The Chan series weights its terms by a Poisson distribution of mean
    ``v / 2 = M^2 / 2``, so the terms that matter sit near ``m = v / 2`` with a
    spread of ``sqrt(v / 2)`` and a fixed order is only adequate up to a fixed
    ``M``. This returns ``ceil(v/2 + 8 sqrt(v/2)) + 4``, eight standard
    deviations past the mode plus a floor for small ``M``, capped at the value
    for ``M = 45`` because beyond about ``M = 38.6`` the probability underflows
    to zero in double precision and more terms cannot help.

    The need for it was found by a property-based test: at ``sigma = 1 m``,
    ``ybar = 9 m``, ``R = 9 m`` (so ``M = 9`` and ``v = 81``) the default order
    of 3, and even an order of 60, leave a relative error of 2.0e-6 against the
    non-central chi-square distribution function, while this recommendation of
    96 terms reaches 2.3e-16. The case is pinned in
    ``tests/test_regressions.py`` and the order requirement is measured as a
    curve in ``validation/validate_analytic.py``.

    Parameters
    ----------
    mahalanobis
        Mahalanobis distance of the miss vector, dimensionless.

    Returns
    -------
    int
        Recommended ``order`` argument for :func:`pc_chan`.
    """
    if not math.isfinite(mahalanobis) or mahalanobis < 0.0:
        raise ValueError(
            f"mahalanobis must be a non-negative finite number, got {mahalanobis!r}"
        )
    half_v = 0.5 * mahalanobis * mahalanobis
    # Beyond a Mahalanobis distance of about 38.6 the probability underflows to
    # zero in double precision, so a larger order buys nothing and the cost is
    # quadratic in M. The recommendation is capped at the order for M = 45.
    half_v = min(half_v, 0.5 * 45.0 * 45.0)
    return int(math.ceil(half_v + 8.0 * math.sqrt(half_v))) + 4


def pc_alfano(
    sigma_x_m: float,
    sigma_y_m: float,
    xbar_m: float,
    ybar_m: float,
    radius_m: float,
    *,
    nodes: int = 201,
) -> float:
    """Alfano's fixed-node Simpson evaluation of the reduced integral.

    Equal steps in ``x`` across ``[-R, R]`` with composite Simpson weights
    (Alfano 2005). The half-chord ``sqrt(R^2 - x^2)`` vanishes at both
    endpoints, so the endpoint terms contribute nothing, but its derivative is
    singular there, which caps the attainable convergence order below Simpson's
    nominal fourth. The measured order is reported in
    ``validation/validate_analytic.py`` rather than claimed here.

    Parameters
    ----------
    sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m
        As for :func:`pc_foster`, metres.
    nodes
        Number of Simpson nodes; must be odd and at least 3. The default of
        201 was chosen from the measured convergence study, not assumed.

    Returns
    -------
    float
        Collision probability, dimensionless, in ``[0, 1]``.
    """
    _check_inputs(sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m)
    if nodes < 3 or nodes % 2 == 0:
        raise ValueError(f"nodes must be an odd integer >= 3, got {nodes!r}")
    if radius_m == 0.0:
        return 0.0
    x = np.linspace(-radius_m, radius_m, nodes)
    half_chord = np.sqrt(np.clip(radius_m * radius_m - x * x, 0.0, None))
    y_mass = _y_mass_array(half_chord, ybar_m, sigma_y_m)
    values = np.exp(-0.5 * ((x - xbar_m) / sigma_x_m) ** 2) * y_mass
    weights = np.ones(nodes)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    step = 2.0 * radius_m / (nodes - 1)
    integral = step / 3.0 * float(values @ weights)
    return float(min(max(integral / (sigma_x_m * _SQRT_2PI), 0.0), 1.0))


def pc_circular_closed_form(radius_m: float, sigma_m: float) -> float:
    """Exact Pc for a circular covariance and zero miss distance, dimensionless.

    With ``sigma_x = sigma_y = sigma`` and ``xbar = ybar = 0`` the area integral
    is a Rayleigh cumulative distribution and evaluates in closed form to
    ``1 - exp(-R^2 / (2 sigma^2))``. This is the known answer the three methods
    are checked against in ``tests/test_analytic.py``; it holds for all
    ``R / sigma``, including the large-radius limit where the series and the
    small-area approximation both fail.
    """
    if sigma_m <= 0.0:
        raise ValueError(f"sigma_m must be strictly positive metres, got {sigma_m!r}")
    if radius_m < 0.0:
        raise ValueError(f"radius_m must be non-negative metres, got {radius_m!r}")
    return float(-math.expm1(-0.5 * (radius_m / sigma_m) ** 2))


def pc_small_body_approximation(
    sigma_x_m: float,
    sigma_y_m: float,
    xbar_m: float,
    ybar_m: float,
    radius_m: float,
) -> float:
    """Leading-order Pc for a hard body small against the covariance.

    ``Pc ~= R^2 / (2 sigma_x sigma_y) * exp(-M^2 / 2)`` where ``M`` is the
    Mahalanobis distance of the miss vector. This is the first term of the
    expansion of the area integral about a point mass: the Gaussian density is
    treated as constant across the hard-body circle and multiplied by its area.

    It is used for two separate purposes in this package, both of them honest
    rather than decorative: it is the closed form whose maximum over covariance
    scale gives Alfano's maximum probability (see :mod:`conjprob.dilution`), and
    it is the strongest *cheap* analytic screen the learned classifier in
    :mod:`conjprob.screening` is scored against, because it uses exactly the
    features the classifier is given.

    Its relative error grows with ``R / sqrt(sigma_x sigma_y)``; the measured
    error curve is in ``validation/validate_regimes.py``.
    """
    _check_inputs(sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m)
    mahalanobis_sq = (xbar_m / sigma_x_m) ** 2 + (ybar_m / sigma_y_m) ** 2
    area_term = radius_m * radius_m / (2.0 * sigma_x_m * sigma_y_m)
    return float(min(max(area_term * math.exp(-0.5 * mahalanobis_sq), 0.0), 1.0))
