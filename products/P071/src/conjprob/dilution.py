"""Probability dilution: where a worse state estimate produces a smaller Pc.

Collision probability is not monotone in the size of the covariance. Starting
from a very tight covariance, growing it *increases* Pc because the hard body
moves into the bulk of the distribution; past a turning point, growing it
*decreases* Pc because the density is spread over a larger area. The far side
of that turning point is the dilution region, and a tool that returns a scalar
there without saying so hands back a reassuring number produced by a worse
state estimate.

The turning point has a closed form. Scaling both principal standard deviations
by ``k`` and holding the miss vector fixed, the small-hard-body approximation
(:func:`conjprob.analytic.pc_small_body_approximation`) gives

``Pc(k) = R^2 / (2 k^2 sigma_x sigma_y) exp(-M^2 / (2 k^2))``

with ``M`` the Mahalanobis distance at ``k = 1``. Writing ``s = k^-2`` this is
``const * s * exp(-M^2 s / 2)``, maximised at ``s = 2 / M^2``, so

* the maximum is at ``k* = M / sqrt(2)``;
* the maximum value is ``Pc_max = R^2 / (e M^2 sigma_x sigma_y)``, which for a
  circular covariance reduces to the familiar ``R^2 / (e d^2)``;
* the present covariance is already on the dilution side when ``k* < 1``, that
  is when ``M < sqrt(2)``.

The ``R^2 / (e d^2)`` form is the maximum-probability result of Alfano, S.
(2005), "Relating position uncertainty to maximum conjunction probability",
*Journal of the Astronautical Sciences* 53(2), 193-205. The inferential problem
that dilution creates -- a large covariance yielding high confidence of safety
from an uninformative measurement -- is analysed in Balch, M. S., Martin, R.,
and Ferson, S. (2019), "Satellite conjunction analysis and the false confidence
theorem", *Proceedings of the Royal Society A* 475(2227), 20180565.

Units: metres for lengths; Mahalanobis distance, scale factors and
probabilities are dimensionless.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .analytic import pc_foster, pc_small_body_approximation

DILUTION_MAHALANOBIS_THRESHOLD = math.sqrt(2.0)
"""Mahalanobis distance below which the covariance is already diluting Pc."""


class DilutionWarning(UserWarning):
    """Raised when the reported Pc sits on the dilution side of its maximum.

    The warning carries the measured sensitivity of Pc to covariance scale so
    that a caller can act on the magnitude and not only on the flag.
    """


@dataclass(frozen=True)
class DilutionAssessment:
    """Measured dilution diagnosis for one conjunction.

    Attributes
    ----------
    mahalanobis
        Mahalanobis distance of the miss vector in the encounter plane,
        dimensionless.
    dilution_ratio
        ``sqrt(2) / M``. Greater than one means the present covariance is
        larger than the one that maximises Pc, so Pc is being diluted.
    on_dilution_branch
        True when the measured ``log_scale_sensitivity`` is negative, that is
        when inflating the covariance would reduce the reported Pc. This is a
        boolean produced by a sign test on a finite-difference derivative; near
        the turning point it is not reliable and ``near_turning_point`` says so.
    near_turning_point
        True when ``|log(dilution_ratio)|`` is within ``1e-3``, where the sign
        of the derivative is not resolvable at the finite-difference step used.
    log_scale_sensitivity
        ``d Pc / d ln k`` at ``k = 1``, by central difference on the
        Foster-Estes Pc with a relative step of ``1e-3``. Dimensionless.
    pc_at_unit_scale
        Foster-Estes Pc at the supplied covariance, dimensionless.
    pc_max_closed_form
        ``R^2 / (e M^2 sigma_x sigma_y)``, the small-hard-body maximum over
        covariance scale, dimensionless.
    scale_at_max_closed_form
        ``M / sqrt(2)``, the covariance scale factor at which Pc is maximised
        under the small-hard-body approximation, dimensionless.
    pc_max_numerical, scale_at_max_numerical
        The maximum of the Foster-Estes Pc over covariance scale and the scale
        at which it occurs, located on a logarithmic grid refined by golden
        section. These are the measured values the closed form is checked
        against.
    """

    mahalanobis: float
    dilution_ratio: float
    on_dilution_branch: bool
    near_turning_point: bool
    log_scale_sensitivity: float
    pc_at_unit_scale: float
    pc_max_closed_form: float
    scale_at_max_closed_form: float
    pc_max_numerical: float
    scale_at_max_numerical: float

    @property
    def dilution_factor(self) -> float:
        """How many times smaller the reported Pc is than its attainable maximum."""
        if self.pc_at_unit_scale <= 0.0:
            return math.inf
        return self.pc_max_numerical / self.pc_at_unit_scale


def max_probability(
    sigma_x_m: float, sigma_y_m: float, xbar_m: float, ybar_m: float, radius_m: float
) -> tuple[float, float]:
    """Closed-form maximum Pc over covariance scale, and the scale attaining it.

    Returns ``(pc_max, scale_at_max)``, both dimensionless. Valid in the
    small-hard-body limit ``R << sqrt(sigma_x sigma_y)``; the measured error
    against the maximum of the Foster-Estes Pc is reported in
    ``validation/validate_dilution_output.txt``.

    Raises
    ------
    ValueError
        If the miss vector is exactly zero, for which Pc is monotonically
        decreasing in covariance scale and no interior maximum exists.
    """
    mahalanobis_sq = math.hypot(xbar_m / sigma_x_m, ybar_m / sigma_y_m) ** 2
    if mahalanobis_sq <= 0.0:
        raise ValueError(
            "maximum probability is undefined for a zero or underflowing miss vector: "
            "Pc decreases monotonically with covariance scale and its supremum is at "
            "zero covariance. The squared Mahalanobis distance evaluated to zero for "
            f"miss components ({xbar_m!r}, {ybar_m!r}) metres"
        )
    pc_max = radius_m * radius_m / (
        math.e * mahalanobis_sq * sigma_x_m * sigma_y_m
    )
    return float(min(pc_max, 1.0)), float(math.sqrt(0.5 * mahalanobis_sq))


def dilution_curve(
    sigma_x_m: float,
    sigma_y_m: float,
    xbar_m: float,
    ybar_m: float,
    radius_m: float,
    scales: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pc against covariance scale factor, exact and small-hard-body.

    Parameters
    ----------
    sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m
        Encounter-plane geometry, metres.
    scales
        Multipliers applied to both principal standard deviations,
        dimensionless, strictly positive.

    Returns
    -------
    tuple of ndarray
        ``(pc_foster, pc_small_body)``, each the same shape as ``scales``,
        dimensionless. The non-monotonic turn in the first array is the
        dilution region.
    """
    grid = np.asarray(scales, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("scales must be a non-empty one-dimensional array")
    if np.any(grid <= 0.0):
        raise ValueError("scales must be strictly positive")
    exact = np.empty_like(grid)
    approx = np.empty_like(grid)
    for index, scale in enumerate(grid):
        exact[index] = pc_foster(
            sigma_x_m * scale, sigma_y_m * scale, xbar_m, ybar_m, radius_m
        )
        approx[index] = pc_small_body_approximation(
            sigma_x_m * scale, sigma_y_m * scale, xbar_m, ybar_m, radius_m
        )
    return exact, approx


def _pc_at_scale(
    scale: float,
    sigma_x_m: float,
    sigma_y_m: float,
    xbar_m: float,
    ybar_m: float,
    radius_m: float,
) -> float:
    return pc_foster(sigma_x_m * scale, sigma_y_m * scale, xbar_m, ybar_m, radius_m)


def _numerical_maximum(
    sigma_x_m: float,
    sigma_y_m: float,
    xbar_m: float,
    ybar_m: float,
    radius_m: float,
    *,
    bracket_decades: float = 4.0,
    coarse_nodes: int = 81,
    golden_iterations: int = 60,
) -> tuple[float, float]:
    """Maximum of the Foster-Estes Pc over covariance scale, by grid then golden section."""
    grid = np.logspace(-bracket_decades, bracket_decades, coarse_nodes)
    values = np.array(
        [_pc_at_scale(s, sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m) for s in grid]
    )
    best = int(np.argmax(values))
    low = math.log(grid[max(best - 1, 0)])
    high = math.log(grid[min(best + 1, coarse_nodes - 1)])
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = low, high
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = _pc_at_scale(math.exp(c), sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m)
    fd = _pc_at_scale(math.exp(d), sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m)
    for _ in range(golden_iterations):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = _pc_at_scale(math.exp(c), sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = _pc_at_scale(math.exp(d), sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m)
        if abs(b - a) < 1e-12:
            break
    scale = math.exp(0.5 * (a + b))
    return (
        float(_pc_at_scale(scale, sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m)),
        float(scale),
    )


def assess_dilution(
    sigma_x_m: float,
    sigma_y_m: float,
    xbar_m: float,
    ybar_m: float,
    radius_m: float,
    *,
    warn: bool = True,
    relative_step: float = 1e-3,
) -> DilutionAssessment:
    """Diagnose dilution for one conjunction and optionally warn.

    Parameters
    ----------
    sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m
        Encounter-plane geometry in the principal frame, metres.
    warn
        If True and the measured sensitivity of Pc to covariance scale is
        negative, issue a :class:`DilutionWarning`. The library raises the
        warning itself rather than leaving the caller to notice.
    relative_step
        Logarithmic step for the central-difference sensitivity, dimensionless.

    Returns
    -------
    DilutionAssessment

    Raises
    ------
    ValueError
        If the miss vector is exactly zero. A zero miss distance has no
        dilution turning point and the caller should know that rather than be
        handed a flag.
    """
    if relative_step <= 0.0:
        raise ValueError(f"relative_step must be positive, got {relative_step!r}")
    mahalanobis = math.hypot(xbar_m / sigma_x_m, ybar_m / sigma_y_m)
    pc_max_closed, scale_closed = max_probability(
        sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m
    )
    ratio = DILUTION_MAHALANOBIS_THRESHOLD / mahalanobis
    up = _pc_at_scale(
        math.exp(relative_step), sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m
    )
    down = _pc_at_scale(
        math.exp(-relative_step), sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m
    )
    sensitivity = (up - down) / (2.0 * relative_step)
    pc_here = _pc_at_scale(1.0, sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m)
    pc_max_numeric, scale_numeric = _numerical_maximum(
        sigma_x_m, sigma_y_m, xbar_m, ybar_m, radius_m
    )
    assessment = DilutionAssessment(
        mahalanobis=float(mahalanobis),
        dilution_ratio=float(ratio),
        on_dilution_branch=bool(sensitivity < 0.0),
        near_turning_point=bool(abs(math.log(ratio)) < 1e-3),
        log_scale_sensitivity=float(sensitivity),
        pc_at_unit_scale=float(pc_here),
        pc_max_closed_form=float(pc_max_closed),
        scale_at_max_closed_form=float(scale_closed),
        pc_max_numerical=float(pc_max_numeric),
        scale_at_max_numerical=float(scale_numeric),
    )
    if warn and assessment.on_dilution_branch:
        warnings.warn(
            "collision probability is on the dilution branch: the Mahalanobis distance "
            f"is {mahalanobis:.4f} (below sqrt(2) = {DILUTION_MAHALANOBIS_THRESHOLD:.4f}), "
            f"so inflating this covariance would reduce the reported Pc of {pc_here:.6e} "
            f"rather than raise it. The attainable maximum over covariance scale is "
            f"{pc_max_numeric:.6e} at scale {scale_numeric:.4f}. A small Pc here is not "
            "evidence of a small risk.",
            DilutionWarning,
            stacklevel=2,
        )
    return assessment
