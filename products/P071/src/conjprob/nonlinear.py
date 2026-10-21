"""Where the two-dimensional formulation stops being valid, measured.

Every analytic method in :mod:`conjprob.analytic` assumes that the relative
motion is rectilinear and uniform and that the encounter is integrated over an
unbounded time interval. Two things break that assumption, and this module
parameterises both so that the breakdown is a measured curve rather than a
statement.

**Finite screening window.** If only ``|t| <= T`` is considered, the probability
of a conjunction inside the window is strictly smaller than the 2-D result. The
natural dimensionless parameter is ``T / t_cross`` with
``t_cross = sigma_eff / |v_rel|`` the time the relative motion takes to cross
one effective standard deviation. The ratio approaches one from below as the
window grows.

**Relative acceleration.** Rectilinear motion neglects the difference in
gravitational acceleration between the two objects. Over a window ``T`` the
accumulated departure from a straight line is ``|a| T^2 / 2``, so the natural
dimensionless curvature parameter is

``kappa = |a| T^2 / (2 sigma_eff)``.

A reference value for a real encounter is obtained from the gravity-gradient
difference: for two objects separated by ``delta_r`` at geocentric radius
``r``, the relative acceleration is of order ``mu |delta_r| / r^3``, which
:func:`gravity_gradient_acceleration_scale` evaluates. At low Earth orbit with a
separation of a few hundred metres this is of order ``1e-4 m/s^2``, which is why
the rectilinear assumption is good for a fast low-Earth conjunction and why the
breakdown has to be exhibited by sweeping ``kappa`` rather than by waiting for a
real case to supply it.

Reference for the short-encounter assumption set and the slow-encounter
exception: Alfriend, K. T., Akella, M. R., Frisbee, J., Foster, J. L., Lee,
D.-J., and Wilkins, M. (1999), "Probability of collision error analysis",
*Space Debris* 1(1), 21-35; and Patera, R. P. (2001), "General method for
calculating satellite collision probability", *Journal of Guidance, Control,
and Dynamics* 24(4), 716-722, for the non-linear-trajectory generalisation this
module measures the need for rather than implements.

Units: metres, metres per second, metres per second squared, seconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .montecarlo import (
    MonteCarloResult,
    pc_monte_carlo,
    pc_monte_carlo_curved,
    pc_monte_carlo_window,
)

EARTH_MU_M3_S2 = 3.986004418e14
"""Earth gravitational parameter, cubic metres per second squared (WGS 84 value)."""


def gravity_gradient_acceleration_scale(
    geocentric_radius_m: float, separation_m: float
) -> float:
    """Order of magnitude of relative acceleration from the gravity gradient.

    ``mu |delta_r| / r^3``, metres per second squared. This is the magnitude of
    the leading term of the difference of the two central-gravity accelerations
    for a separation ``delta_r`` small against the geocentric radius ``r``; the
    tensor form is ``-(mu / r^3)(I - 3 \\hat r \\hat r^T) \\delta r``, whose
    largest eigenvalue is ``2 mu / r^3``, so this expression is the correct
    order and is within a factor of two of the directional extremes.

    Parameters
    ----------
    geocentric_radius_m
        Geocentric radius of the encounter, metres.
    separation_m
        Separation between the two objects, metres.
    """
    if geocentric_radius_m <= 0.0:
        raise ValueError(
            f"geocentric_radius_m must be positive metres, got {geocentric_radius_m!r}"
        )
    if separation_m < 0.0:
        raise ValueError(f"separation_m must be non-negative metres, got {separation_m!r}")
    return float(EARTH_MU_M3_S2 * separation_m / geocentric_radius_m**3)


def curvature_parameter(
    acceleration_mps2: float, window_s: float, sigma_effective_m: float
) -> float:
    """``kappa = |a| T^2 / (2 sigma_eff)``, dimensionless.

    The accumulated departure from a straight line over the window, measured in
    units of the effective position uncertainty. Values far below one mean the
    rectilinear assumption costs nothing; the value at which it starts to cost
    something is measured in ``validation/validate_nonlinear_output.txt``.
    """
    if sigma_effective_m <= 0.0:
        raise ValueError(
            f"sigma_effective_m must be positive metres, got {sigma_effective_m!r}"
        )
    if window_s < 0.0:
        raise ValueError(f"window_s must be non-negative seconds, got {window_s!r}")
    return float(abs(acceleration_mps2) * window_s * window_s / (2.0 * sigma_effective_m))


@dataclass(frozen=True)
class WindowPoint:
    """One point of the finite-window sweep."""

    window_s: float
    window_over_crossing_time: float
    monte_carlo: MonteCarloResult
    ratio_to_unbounded: float
    ratio_ci_low: float
    ratio_ci_high: float


@dataclass(frozen=True)
class CurvaturePoint:
    """One point of the curvature sweep."""

    acceleration_mps2: float
    curvature_parameter: float
    monte_carlo: MonteCarloResult
    pc_linear_2d: float
    relative_error: float
    relative_error_ci_low: float
    relative_error_ci_high: float


def window_sweep(
    covariance_m2: ArrayLike,
    miss_vector_m: ArrayLike,
    relative_velocity_mps: ArrayLike,
    radius_m: float,
    windows_s: ArrayLike,
    *,
    sigma_effective_m: float,
    n_samples: int = 400_000,
    seed: int = 0,
) -> tuple[MonteCarloResult, list[WindowPoint]]:
    """Measure the finite-window shortfall of the unbounded-time formulation.

    Returns the unbounded-time Monte Carlo reference and one
    :class:`WindowPoint` per window. The ratio reported is windowed Pc divided
    by unbounded Pc, with an interval obtained by combining the two
    Clopper-Pearson intervals at their extremes, which is conservative because
    the two estimates share the same seed stream and are positively correlated.
    """
    unbounded = pc_monte_carlo(
        covariance_m2,
        miss_vector_m,
        relative_velocity_mps,
        radius_m,
        n_samples=n_samples,
        seed=seed,
    )
    speed = float(np.linalg.norm(np.asarray(relative_velocity_mps, dtype=float)))
    crossing = sigma_effective_m / speed
    points: list[WindowPoint] = []
    for window in np.asarray(windows_s, dtype=float):
        result = pc_monte_carlo_window(
            covariance_m2,
            miss_vector_m,
            relative_velocity_mps,
            radius_m,
            float(window),
            n_samples=n_samples,
            seed=seed,
        )
        ratio = result.pc / unbounded.pc if unbounded.pc > 0.0 else math.nan
        low = result.ci_low / unbounded.ci_high if unbounded.ci_high > 0.0 else math.nan
        high = result.ci_high / unbounded.ci_low if unbounded.ci_low > 0.0 else math.inf
        points.append(
            WindowPoint(
                window_s=float(window),
                window_over_crossing_time=float(window / crossing),
                monte_carlo=result,
                ratio_to_unbounded=float(ratio),
                ratio_ci_low=float(low),
                ratio_ci_high=float(high),
            )
        )
    return unbounded, points


def curvature_sweep(
    covariance_m2: ArrayLike,
    miss_vector_m: ArrayLike,
    relative_velocity_mps: ArrayLike,
    radius_m: float,
    accelerations_mps2: ArrayLike,
    *,
    window_s: float,
    pc_linear_2d: float,
    sigma_effective_m: float,
    acceleration_direction: ArrayLike | None = None,
    n_samples: int = 120_000,
    seed: int = 0,
    time_nodes: int = 129,
    refinements: int = 4,
) -> list[CurvaturePoint]:
    """Measure the error of the rectilinear 2-D Pc against a curved 3-D trajectory.

    The acceleration is applied along ``acceleration_direction`` (default: the
    encounter-plane direction of the miss vector, which is the direction that
    moves the trajectory towards or away from the primary and therefore the
    worst case for Pc). Each point reports the relative error of
    ``pc_linear_2d`` against the curved Monte Carlo estimate, with the interval
    propagated from the Monte Carlo interval.

    Notes
    -----
    The curved estimator discretises time, which can only overestimate the
    minimum separation and therefore underestimate Pc. The grid bias is
    measured separately in ``validation/validate_nonlinear.py`` by refining
    ``time_nodes``; it is not assumed negligible here.
    """
    miss = np.asarray(miss_vector_m, dtype=float).reshape(-1)
    if acceleration_direction is None:
        norm = float(np.linalg.norm(miss))
        if norm <= 0.0:
            raise ValueError(
                "acceleration_direction must be given when the miss vector is zero"
            )
        direction = miss / norm
    else:
        direction = np.asarray(acceleration_direction, dtype=float).reshape(-1)
        norm = float(np.linalg.norm(direction))
        if direction.shape != (3,) or norm <= 0.0:
            raise ValueError("acceleration_direction must be a non-zero 3-vector")
        direction = direction / norm
    points: list[CurvaturePoint] = []
    for magnitude in np.asarray(accelerations_mps2, dtype=float):
        result = pc_monte_carlo_curved(
            covariance_m2,
            miss,
            relative_velocity_mps,
            radius_m,
            window_s,
            direction * float(magnitude),
            n_samples=n_samples,
            seed=seed,
            time_nodes=time_nodes,
            refinements=refinements,
        )
        error = (
            (pc_linear_2d - result.pc) / result.pc if result.pc > 0.0 else math.nan
        )
        low = (
            (pc_linear_2d - result.ci_high) / result.ci_high
            if result.ci_high > 0.0
            else math.nan
        )
        high = (
            (pc_linear_2d - result.ci_low) / result.ci_low
            if result.ci_low > 0.0
            else math.inf
        )
        points.append(
            CurvaturePoint(
                acceleration_mps2=float(magnitude),
                curvature_parameter=curvature_parameter(
                    float(magnitude), window_s, sigma_effective_m
                ),
                monte_carlo=result,
                pc_linear_2d=float(pc_linear_2d),
                relative_error=float(error),
                relative_error_ci_low=float(low),
                relative_error_ci_high=float(high),
            )
        )
    return points


def first_crossing(
    x_values: ArrayLike, y_values: ArrayLike, threshold: float
) -> float:
    """Linearly interpolated abscissa at which ``|y|`` first exceeds ``threshold``.

    Used to turn a measured error curve into a single measured regime boundary.
    Returns ``nan`` when the threshold is never crossed, which is itself a
    result and is reported as such rather than silently clipped to the end of
    the sweep.

    Parameters
    ----------
    x_values
        Monotonically increasing abscissae.
    y_values
        Values whose magnitude is compared against ``threshold``.
    threshold
        Positive threshold, same units as ``y_values``.
    """
    x = np.asarray(x_values, dtype=float)
    y = np.abs(np.asarray(y_values, dtype=float))
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError("x_values and y_values must be 1-D arrays of equal length >= 2")
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold!r}")
    above = np.nonzero(y > threshold)[0]
    if above.size == 0:
        return float("nan")
    index = int(above[0])
    if index == 0:
        return float(x[0])
    x0, x1 = float(x[index - 1]), float(x[index])
    y0, y1 = float(y[index - 1]), float(y[index])
    if y1 == y0:
        return x1
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))


