"""Baseline 3: the profile likelihood. The ground truth the covariance is scored against.

Why this baseline exists
------------------------
Baseline 1's Cramer-Rao bounds and baseline 2's confidence intervals are both
read off a *linearisation* of the model at one parameter point. They are the
cheap answer. The expensive answer re-optimises the model, nonlinearly, at each
value of the parameter under test, and it is the one that is right when the two
disagree:

    equation (17)   chi2_profile(c) = min over { delta : v^T delta = c }
                                      | y - model(delta) |^2 / sigma^2

evaluated on a declared grid of ``c`` and reported as
``chi2_profile(c) - min_c chi2_profile(c)``. The method, and the use of the
profile's flatness as the operational definition of practical
unidentifiability, is Raue, Kreutz, Maiwald, Bachmann, Schilling,
Klingmueller & Timmer 2009, "Structural and practical identifiability analysis
of partially observed dynamical models by exploiting the profile likelihood",
Bioinformatics 25(15), 1923-1929. The pointwise threshold on the profile is the
chi-square quantile on one degree of freedom,
``chi2.ppf(CONFIDENCE_LEVEL, 1) = 3.841459``, which is the standard likelihood
ratio interval; see Seber & Wild 1989, "Nonlinear Regression", Wiley,
chapter 5, for the likelihood-ratio interval in a nonlinear regression model
and for its asymmetry.

Three outcomes, and the override
--------------------------------
For each direction profiled, the result is one of:

- **curved and crossing.** The profile rises through the threshold on both
  sides inside the grid. A finite interval exists, it is reported, and the
  Wald half-width from the linearisation is scored against it.
- **curved but not crossing inside the grid.** The parameter is determined more
  loosely than the grid half-width. Reported as such, with the grid half-width
  named, because "no interval found" and "no interval exists" are different
  statements and only the first is what was measured.
- **flat.** The profile does not move by more than :data:`FLAT_TOLERANCE` over
  the whole grid. The data contain no information about this direction at all,
  and **the flat profile overrides whatever the covariance claims**: when the
  Wald half-width from the linearisation is finite, that number is wrong and
  :attr:`ProfileResult.overrides_covariance` is set. This is the single most
  important output of this module. For the ``rate_full`` telemetry set the
  profile of every one of the five parameters is flat to within rounding --
  which is what equation (6) predicts, since a change in any one parameter can
  be cancelled exactly by a compensating move along the scale direction -- while
  the pseudo-inverse diagonal of the Fisher information cheerfully reports
  half-widths of around two per cent.

Assumptions, cost and limits
----------------------------
- The noise is Gaussian with the declared known standard deviation, so
  ``chi2`` is a true log-likelihood up to a constant and the threshold applies.
- The inner minimisation is ``scipy.optimize.least_squares`` with the analytic
  Jacobian of :func:`twindiverge.plant.simulate_with_sensitivities`, started
  from the unconstrained optimum, which is the usual continuation strategy and
  is what makes the cost tolerable. It is a local minimiser: a profile that is
  reported as curved is a local statement, and a second minimum elsewhere in
  parameter space would not be found. Bellman & Astrom 1970, "On structural
  identifiability", Mathematical Biosciences 7(3-4), 329-339, is the global
  question and is not answered here.
- Cost is one inner optimisation per grid point per direction, each costing a
  handful of integrations of the joint state-and-sensitivity system. The
  measured cost of the whole declared profile set is reported in
  ``validation/validate_profile_output.txt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy import optimize, stats

from .plant import NOMINAL, PARAMETER_NAMES, WheelParameters, perturb
from .plant import simulate as _simulate
from .telemetry import TelemetrySet, design_matrix, residual_vector

__all__ = [
    "CONFIDENCE_LEVEL",
    "FLAT_TOLERANCE",
    "GRID_HALF_WIDTH",
    "GRID_MIN_HALF_WIDTH",
    "GRID_POINTS",
    "GRID_WALD_MULTIPLE",
    "INNER_BOX",
    "WARM_START_FLOOR",
    "ProfileResult",
    "chi_square",
    "fit",
    "grid_half_width_for",
    "profile_direction",
    "profile_parameters",
    "wheel_from_offset",
]

#: Declared confidence level of the likelihood-ratio interval.
CONFIDENCE_LEVEL: Final[float] = 0.95

#: Declared half-width of the profile grid, in log-parameter units. 0.10 is a
#: 10.5 per cent perturbation, which is the scale of the twin-versus-reality
#: discrepancies this product is built for.
GRID_HALF_WIDTH: Final[float] = 0.10

#: Declared number of grid points per profile, including both ends and the
#: centre. Odd so that the centre point is on the grid.
GRID_POINTS: Final[int] = 15

#: Declared multiple of the linearised (Wald) half-width that a profile grid
#: spans, when that is narrower than :data:`GRID_HALF_WIDTH`. A grid fixed at
#: the declared half-width is useless for a direction the data determine to one
#: part in a thousand: the inner optimisation at the far grid points is then
#: asked to compensate a parameter excursion thousands of standard deviations
#: wide, which it cannot do inside :data:`INNER_BOX`, and the profile it returns
#: is an upper bound on the true one rather than the true one. Four Wald
#: half-widths comfortably brackets a threshold crossing at 1.96 of them.
GRID_WALD_MULTIPLE: Final[float] = 4.0

#: Declared floor on the profile grid half-width, in log-parameter units. A
#: grid narrower than this would be measuring the integrator rather than the
#: likelihood.
GRID_MIN_HALF_WIDTH: Final[float] = 1.0e-3

#: Declared box constraint on the inner minimisation, in log-parameter units,
#: applied to each coefficient in the complement basis. Equation (1) is valid
#: only inside the declared rate window, and an unconstrained Gauss-Newton step
#: on a problem whose condition number is above 1e4 can leave it in one step and
#: send the integrator to a rate of order 1e199, which is what happened before
#: this bound existed. A result whose inner optimum sits ON the bound is
#: reported through ``boundary_points`` rather than silently accepted.
INNER_BOX: Final[float] = 0.5

#: Declared floor below which a warm start is snapped to exactly zero.
#:
#: Measured 2026-11-09 against scipy 1.18.1: ``least_squares(method="trf")``
#: started from a point whose components are of order 1e-17 terminates on its
#: ``xtol`` test after 23 function evaluations WITHOUT MOVING, returning a
#: chi-square 3151 above the minimum it reaches from exactly zero. The failure
#: band is narrow and was mapped by bisection -- a start of 0.0 converges, 1e-20
#: and 1e-17 both fail, and 1e-14, 1e-11, 1e-8, 1e-5 and 1e-3 all converge --
#: and it occurs with and without bounds and under both ``x_scale="jac"`` and
#: ``x_scale=1.0``. It matters here because a start of order 1e-17 is precisely
#: what the centre-out continuation hands the next grid point when the profile
#: is exactly flat, which is the case this product exists to measure, so the
#: band is not a corner case for this code. A start below this floor is snapped
#: to exactly zero. ``tests/test_regressions.py`` pins the measurement.
WARM_START_FLOOR: Final[float] = 1.0e-12

#: Declared tolerance below which a profile is called flat, in chi-square units.
#: The threshold itself is 3.841459, so 1e-3 is three and a half orders of
#: magnitude below the level at which an interval would close. A profile that
#: moves by less than this over a 10 per cent excursion of the parameter carries
#: no information about it on any reading.
FLAT_TOLERANCE: Final[float] = 1.0e-3


def chi_square(
    telemetry: TelemetrySet, observed: np.ndarray, delta_log: np.ndarray
) -> float:
    """``|y - model(delta)|^2 / sigma^2`` at one point. The objective of equation (17)."""
    twin = _simulate(perturb(NOMINAL, delta_log), telemetry.campaign)
    vector = residual_vector(telemetry, observed, twin)
    return float(vector @ vector)


class _Objective:
    """Scaled residual and Jacobian at a logarithmic offset, with a one-entry memo.

    ``scipy.optimize.least_squares`` calls ``fun`` and ``jac`` separately at the
    same iterate, and each call here costs one integration of the joint
    state-and-sensitivity system. Caching the last point halves the cost of
    every profile in this module; the cache key is the exact bytes of the
    offset vector, so it can only ever return the value for the point asked
    for.

    The Jacobian of ``y - model(delta)`` is ``-A(delta)``, with ``A`` the design
    matrix of equation (8) evaluated at the PERTURBED parameters, so the
    analytic sensitivities are re-derived at every iterate rather than held
    fixed at the nominal point.
    """

    def __init__(self, telemetry: TelemetrySet, observed: np.ndarray) -> None:
        self.telemetry = telemetry
        self.observed = np.asarray(observed, dtype=float)
        self._key: bytes | None = None
        self._value: tuple[np.ndarray, np.ndarray] | None = None
        self.evaluations = 0

    def __call__(self, delta_log: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        offset = np.ascontiguousarray(np.asarray(delta_log, dtype=float))
        key = offset.tobytes()
        if key == self._key and self._value is not None:
            return self._value
        matrix, trajectory = design_matrix(self.telemetry, perturb(NOMINAL, offset))
        vector = residual_vector(self.telemetry, self.observed, trajectory)
        self._key = key
        self._value = (vector, -matrix)
        self.evaluations += 1
        return self._value


def _residual_and_jacobian(
    telemetry: TelemetrySet, observed: np.ndarray, delta_log: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Scaled residual and its Jacobian with respect to ``delta_log``. Uncached."""
    return _Objective(telemetry, observed)(delta_log)


@dataclass(frozen=True)
class ProfileResult:
    """One profile of equation (17), and its verdict on the linearised covariance."""

    label: str
    direction: np.ndarray
    grid: np.ndarray
    delta_chi2: np.ndarray
    chi2_min: float
    threshold: float
    flat: bool
    max_delta_chi2: float
    interval: tuple[float, float] | None
    lower_limit: float | None
    upper_limit: float | None
    minimum_at: float
    wald_half_width: float
    inner_iterations: int
    grid_half_width: float
    boundary_points: int = 0
    interval_truncated: bool = False

    @property
    def profile_half_width(self) -> float:
        """Half-width of the profile interval, or ``inf`` when none was found."""
        if self.interval is None:
            return float("inf")
        return 0.5 * (self.interval[1] - self.interval[0])

    @property
    def one_sided_half_width(self) -> float:
        """Distance from the profile minimum to the nearer threshold crossing.

        ``inf`` when neither side crossed inside the grid. On an ill-conditioned
        problem the profile is routinely one-sided over any grid a reader would
        choose, and this is then the only half-width the data support. It is
        what :attr:`one_sided_ratio` scores the linearisation against.
        """
        distances = [
            abs(limit - self.minimum_at)
            for limit in (self.lower_limit, self.upper_limit)
            if limit is not None
        ]
        return min(distances) if distances else float("inf")

    @property
    def one_sided_ratio(self) -> float:
        """One-sided profile half-width divided by the Wald half-width.

        Greater than 1 means the linearisation was optimistic on the side the
        profile closed. ``nan`` when either quantity is not finite.
        """
        width = self.one_sided_half_width
        if not np.isfinite(width) or not np.isfinite(self.wald_half_width):
            return float("nan")
        if self.wald_half_width == 0.0:
            return float("nan")
        return width / self.wald_half_width

    @property
    def sides_crossed(self) -> int:
        """How many of the two sides crossed the threshold inside the grid."""
        return int(self.lower_limit is not None) + int(self.upper_limit is not None)

    @property
    def overrides_covariance(self) -> bool:
        """True when the profile is flat and the linearisation claims a finite width.

        The case the module exists for. A thresholded boolean: tests assert
        ``max_delta_chi2`` and ``wald_half_width`` to tolerance and allow this
        flag to differ only when either sits within tolerance of its threshold.
        """
        return self.flat and np.isfinite(self.wald_half_width)

    @property
    def covariance_ratio(self) -> float:
        """Profile half-width divided by the Wald half-width.

        Greater than 1 means the linearisation was optimistic. ``inf`` when the
        profile found no interval and the Wald width is finite, which is the
        override case; ``nan`` when neither is finite.
        """
        if not np.isfinite(self.wald_half_width):
            return float("nan") if self.interval is None else 0.0
        if self.interval is None:
            return float("inf")
        return self.profile_half_width / self.wald_half_width


def _crossing(
    grid: np.ndarray, delta: np.ndarray, threshold: float, truncated: np.ndarray
) -> tuple[float | None, float | None, bool]:
    """The two threshold crossings by interpolation in ``sqrt(delta_chi2)``.

    Returns the interval and whether a bracketing grid point was truncated by
    the inner box constraint.

    Interpolation is linear in ``sqrt(delta_chi2)`` rather than in
    ``delta_chi2``, because a profile is locally quadratic about its minimum, so
    ``sqrt(delta_chi2)`` is locally LINEAR in the profiled coordinate and the
    interpolation is then exact for the leading behaviour instead of being
    wrong by the curvature. Interpolating the quadratic itself on a grid whose
    spacing is comparable to the interval half-width underestimates the
    crossing by orders of magnitude, which is how this function was first
    written and what the sqrt fixes.

    Both sides must cross for an interval to exist; a one-sided crossing is
    reported as no interval, because a half-open set is not what the threshold
    defines. The two limits are nevertheless returned separately, because "the
    profile closed on the low side at -1.3e-4 and never crossed on the high
    side inside the grid" is a far more useful report than "no interval", and
    the asymmetric case is the ordinary one on this plant: the chi-square
    valley of an ill-conditioned problem is one-sided over any grid a reader
    would choose. A crossing bracketed by a grid point whose inner optimisation
    hit the box is flagged: the profile value there is an upper bound on the
    true profile, so the limit would be an inner bound and is not quoted.
    """
    root = np.sqrt(np.maximum(delta, 0.0))
    level = float(np.sqrt(threshold))
    centre = int(np.argmin(delta))
    hit = False
    lower = None
    for index in range(centre, 0, -1):
        if root[index - 1] >= level > root[index]:
            span = root[index - 1] - root[index]
            weight = (level - root[index]) / span if span > 0.0 else 0.0
            lower = grid[index] + weight * (grid[index - 1] - grid[index])
            hit = hit or bool(truncated[index - 1]) or bool(truncated[index])
            break
    upper = None
    for index in range(centre, len(grid) - 1):
        if root[index + 1] >= level > root[index]:
            span = root[index + 1] - root[index]
            weight = (level - root[index]) / span if span > 0.0 else 0.0
            upper = grid[index] + weight * (grid[index + 1] - grid[index])
            hit = hit or bool(truncated[index + 1]) or bool(truncated[index])
            break
    return (
        None if lower is None else float(lower),
        None if upper is None else float(upper),
        hit,
    )


def profile_direction(
    telemetry: TelemetrySet,
    observed: np.ndarray,
    direction: np.ndarray,
    label: str,
    wald_half_width: float,
    centre: np.ndarray | None = None,
    half_width: float = GRID_HALF_WIDTH,
    points: int = GRID_POINTS,
    max_iterations: int = 300,
) -> ProfileResult:
    """Profile the likelihood along one unit direction in logarithmic parameters.

    ``direction`` need not be a coordinate axis: profiling the identifiable
    combinations of equation (11) is exactly as meaningful as profiling the
    parameters, and for a telemetry set with a null space it is the only
    meaningful thing to profile.

    ``centre`` is the point the grid is centred on, defaulting to the
    unperturbed twin. ``wald_half_width`` is the half-width the linearisation
    claims for this direction; it is carried through so that the comparison
    this module exists to make is in the result rather than left to the caller.
    """
    unit = np.asarray(direction, dtype=float)
    if unit.shape != (5,):
        raise ValueError(f"direction must have 5 components, got shape {unit.shape}")
    norm = float(np.linalg.norm(unit))
    if norm == 0.0:
        raise ValueError("direction must be non-zero")
    unit = unit / norm
    if points < 3 or points % 2 == 0:
        raise ValueError(f"points must be an odd integer of at least 3, got {points!r}")
    if half_width <= 0.0:
        raise ValueError(f"half_width must be strictly positive, got {half_width!r}")

    origin = np.zeros(5) if centre is None else np.asarray(centre, dtype=float)
    # Basis of the complement of `unit`, which is what the inner minimisation
    # is free to move in.
    basis, _ = np.linalg.qr(np.column_stack([unit, np.eye(5)]))
    complement = basis[:, 1:5]

    grid = np.linspace(-half_width, half_width, points)
    offset = float(unit @ origin)
    chi2 = np.empty(points)
    truncated = np.zeros(points, dtype=bool)
    iterations = 0
    boundary = 0

    evaluator = _Objective(telemetry, observed)

    def solve(step: float, start: np.ndarray) -> tuple[float, np.ndarray, int, bool]:
        """One inner minimisation of equation (17) at ``unit @ delta == step``.

        The anchor satisfies the constraint exactly and the free coefficients
        move only in the complement, so the constraint holds at every iterate
        with no penalty term and no projection.
        """
        anchor = origin + unit * step
        # See WARM_START_FLOOR. A warm start of order 1e-17 -- which is exactly
        # what the previous grid point returns when the compensation is exact
        # and the profile is flat -- makes the solver stop without moving.
        warm = np.where(np.abs(start) < WARM_START_FLOOR, 0.0, start)

        def objective(coefficients: np.ndarray) -> np.ndarray:
            return evaluator(anchor + complement @ coefficients)[0]

        def jacobian(coefficients: np.ndarray) -> np.ndarray:
            return evaluator(anchor + complement @ coefficients)[1] @ complement

        solution = optimize.least_squares(
            objective,
            np.clip(warm, -INNER_BOX * 0.999, INNER_BOX * 0.999),
            jac=jacobian,
            bounds=(-INNER_BOX, INNER_BOX),
            method="trf",
            tr_solver="exact",
            x_scale="jac",
            xtol=1.0e-14,
            ftol=1.0e-14,
            gtol=1.0e-14,
            max_nfev=max_iterations,
        )
        on_bound = bool(np.any(np.abs(np.abs(solution.x) - INNER_BOX) < 1.0e-9))
        return 2.0 * float(solution.cost), solution.x, int(solution.nfev), on_bound

    # Centre-out continuation: each grid point is warm-started from its already
    # solved neighbour, walking outwards from the middle in both directions.
    # Starting at one end and sweeping across instead makes the first point the
    # hardest, which is where an unwarmed Gauss-Newton step does its damage.
    middle = points // 2
    chi2[middle], centre_free, used, hit = solve(float(grid[middle]), np.zeros(4))
    iterations += used
    truncated[middle] = hit
    boundary += int(hit)
    free = centre_free
    for index in range(middle + 1, points):
        chi2[index], free, used, hit = solve(float(grid[index]), free)
        iterations += used
        truncated[index] = hit
        boundary += int(hit)
    free = centre_free
    for index in range(middle - 1, -1, -1):
        chi2[index], free, used, hit = solve(float(grid[index]), free)
        iterations += used
        truncated[index] = hit
        boundary += int(hit)

    _ = evaluator.evaluations
    minimum = float(chi2.min())
    delta = chi2 - minimum
    threshold = float(stats.chi2.ppf(CONFIDENCE_LEVEL, 1))
    maximum = float(delta.max())
    lower, upper, hit_bracket = _crossing(grid + offset, delta, threshold, truncated)
    if hit_bracket:
        lower = None
        upper = None
    interval = None if lower is None or upper is None else (lower, upper)
    return ProfileResult(
        label=label,
        direction=unit,
        grid=grid + offset,
        delta_chi2=delta,
        chi2_min=minimum,
        threshold=threshold,
        flat=maximum < FLAT_TOLERANCE,
        max_delta_chi2=maximum,
        interval=interval,
        lower_limit=lower,
        upper_limit=upper,
        minimum_at=float((grid + offset)[int(np.argmin(delta))]),
        wald_half_width=float(wald_half_width),
        inner_iterations=iterations,
        grid_half_width=float(half_width),
        boundary_points=boundary,
        interval_truncated=hit_bracket,
    )


def grid_half_width_for(wald_half_width: float) -> float:
    """The declared grid half-width for a direction with this linearised width.

    ``min(GRID_HALF_WIDTH, max(GRID_MIN_HALF_WIDTH, GRID_WALD_MULTIPLE * wald))``,
    and the declared half-width itself when the linearised width is infinite.
    See :data:`GRID_WALD_MULTIPLE` for why a fixed grid does not work.
    """
    if not np.isfinite(wald_half_width) or wald_half_width <= 0.0:
        return GRID_HALF_WIDTH
    return float(
        min(
            GRID_HALF_WIDTH,
            max(GRID_MIN_HALF_WIDTH, GRID_WALD_MULTIPLE * wald_half_width),
        )
    )


def profile_parameters(
    telemetry: TelemetrySet,
    observed: np.ndarray,
    wald_half_widths: np.ndarray,
    points: int = GRID_POINTS,
    centre: np.ndarray | None = None,
) -> tuple[ProfileResult, ...]:
    """Profile all five parameters, in :data:`twindiverge.plant.PARAMETER_NAMES` order.

    The grid for each parameter is centred on ``centre`` -- the unconstrained
    fit of :func:`fit` when not supplied -- and its half-width is chosen by
    :func:`grid_half_width_for` from the linearised half-width of that
    parameter.
    """
    widths = np.asarray(wald_half_widths, dtype=float)
    if widths.shape != (5,):
        raise ValueError(f"expected 5 Wald half-widths, got shape {widths.shape}")
    origin = fit(telemetry, observed)[0] if centre is None else np.asarray(centre, float)
    results = []
    for index, name in enumerate(PARAMETER_NAMES):
        axis = np.zeros(5)
        axis[index] = 1.0
        results.append(
            profile_direction(
                telemetry,
                observed,
                axis,
                f"log {name}",
                float(widths[index]),
                centre=origin,
                half_width=grid_half_width_for(float(widths[index])),
                points=points,
            )
        )
    return tuple(results)


def fit(
    telemetry: TelemetrySet, observed: np.ndarray, max_iterations: int = 200
) -> tuple[np.ndarray, float]:
    """Unconstrained nonlinear least-squares fit, for the profile's reference minimum.

    Returns the logarithmic-parameter offset and its chi-square. On a telemetry
    set with a null space the minimiser is not unique -- every point of the
    affine null-space family attains the same chi-square -- and ``least_squares``
    returns the one its own regularised step sequence lands on. That
    non-uniqueness is the subject of this product and is not treated as a
    failure; only the chi-square is used downstream.
    """

    def objective(coefficients: np.ndarray) -> np.ndarray:
        return _residual_and_jacobian(telemetry, observed, coefficients)[0]

    def jacobian(coefficients: np.ndarray) -> np.ndarray:
        return _residual_and_jacobian(telemetry, observed, coefficients)[1]

    solution = optimize.least_squares(
        objective,
        np.zeros(5),
        jac=jacobian,
        method="trf",
        tr_solver="exact",
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=max_iterations,
    )
    return solution.x, 2.0 * float(solution.cost)


def wheel_from_offset(delta_log: np.ndarray) -> WheelParameters:
    """The parameter set at a logarithmic offset from nominal. Convenience for reports."""
    return perturb(NOMINAL, delta_log)
