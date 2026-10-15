"""Profile-likelihood confidence intervals.

The quadratic (Hessian) covariance of :mod:`twincalib.fitting` is the curvature
of the likelihood at one point. The profile likelihood walks away from that
point and asks what the likelihood actually does, and it is the tool that
separates the two kinds of failure Raue et al. (2009) name:

* **structurally** non-identifiable -- the profile is exactly flat, because a
  whole direction in parameter space leaves the predictions unchanged;
* **practically** non-identifiable -- the profile rises, but so slowly that it
  never crosses the threshold within any physically sensible range, so the
  interval is open on one or both sides even though the Hessian reported a
  finite standard deviation.

Method
------
For a parameter ``p`` fix it at a value and re-optimise all the others. Let

``D(p) = chi2_min(p) - chi2_min``

be the increase in the weighted sum of squares, which for independent Gaussian
errors of known variance is twice the increase in the negative log-likelihood.
By Wilks' theorem (Wilks 1938) ``D`` is asymptotically chi-squared with one
degree of freedom, so the set

``{p : D(p) <= chi2_{1, alpha}}``

is a confidence interval of level ``alpha`` -- 3.8415 for 95 per cent. The
interval boundaries are found by linear interpolation of ``D`` between the two
grid points that bracket the threshold.

The grid is laid out in ``ln p``, centred on the estimate, with a half width of
``span_sigma`` times the Hessian relative standard deviation where that is
finite, and capped at ``max_log_half_width``. A parameter whose quadratic
standard deviation is infinite -- because it loads on a direction the data does
not constrain -- gets the capped width, so the search is widest exactly where it
needs to be. Each crossing found on the grid is then refined by bisection, since
the grid spacing on a wide range is far coarser than the crossing deserves.
**Where no crossing is found inside the grid the interval boundary is reported
as unbounded rather than extrapolated**, and the range that was covered is
reported with it, so that "unbounded" always means "unbounded within a stated
range".

Cost and its consequences
-------------------------
One profile point is a full ``p-1`` parameter fit. A six-parameter profile at
eight points per side is 96 nested fits, each warm-started from its neighbour.
That is seconds, not minutes, for the designs here, but it is two orders of
magnitude more work than a Hessian, and the coverage study in
``validation/validate_profile.py`` is sized against that.

What the intervals do not account for
-------------------------------------
Model error. Every interval here is computed under the assumption that the
calibration model is the model that generated the data. When it is not -- which
is the normal case for a digital twin, and is deliberately the case for the
shipped truth model -- the profile interval is a statement about the best fit of
the wrong model and can exclude the truth with high confidence. That is measured
rather than asserted: ``validation/validate_profile_output.txt`` reports coverage
both with and without the declared model discrepancy, and the second number is
much worse.

References
----------
Wilks, S. S. (1938), "The large-sample distribution of the likelihood ratio for
testing composite hypotheses", *Annals of Mathematical Statistics* 9(1), 60-62.

Raue, A., Kreutz, C., Maiwald, T., Bachmann, J., Schilling, M., Klingmueller, U.
and Timmer, J. (2009), "Structural and practical identifiability analysis of
partially observed dynamical models by exploiting the profile likelihood",
*Bioinformatics* 25(15), 1923-1929.

Venzon, D. J. and Moolgavkar, S. H. (1988), "A method for computing
profile-likelihood-based confidence intervals", *Journal of the Royal
Statistical Society Series C* 37(1), 87-94. A faster root-finding alternative to
the grid walk used here; the grid is kept because it returns the whole curve,
which is what makes a flat profile visible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

from .fitting import CalibrationProblem, FitResult, GuardedResidual, chi_square

__all__ = [
    "DEFAULT_LEVEL",
    "ProfileCurve",
    "ProfilePoint",
    "chi2_threshold",
    "profile_all",
    "profile_likelihood",
]

DEFAULT_LEVEL: float = 0.95
"""Default confidence level, dimensionless."""


def chi2_threshold(level: float = DEFAULT_LEVEL) -> float:
    """Wilks threshold for a one-parameter profile, dimensionless.

    ``chi2.ppf(level, 1)``: 2.7055 at 0.90, 3.8415 at 0.95, 6.6349 at 0.99.
    """
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    return float(chi2_dist.ppf(level, 1))


@dataclass(frozen=True)
class ProfilePoint:
    """One point of a profile curve.

    Attributes
    ----------
    log_offset:
        ``ln(value) - ln(theta_hat)``, dimensionless. 0 at the estimate.
    value:
        The fixed parameter value, in the parameter's own unit.
    chi2:
        Weighted sum of squares after re-optimising the other parameters.
    delta_chi2:
        ``chi2 - chi2_min``, dimensionless.
    converged:
        Whether the nested optimisation reported convergence.
    """

    log_offset: float
    value: float
    chi2: float
    delta_chi2: float
    converged: bool


@dataclass(frozen=True)
class ProfileCurve:
    """A profile likelihood for one parameter, and the interval it implies.

    Attributes
    ----------
    index, name:
        Which parameter.
    estimate:
        The point estimate, in the parameter's own unit.
    level, threshold:
        Confidence level and the ``chi2`` threshold it maps to.
    points:
        The whole curve including the centre point, ordered by ``log_offset``.
    lower, upper:
        Interval boundaries in the parameter's own unit, or ``None`` when the
        profile did not cross the threshold on that side inside the search
        range.
    search_lower, search_upper:
        The range that was actually searched, in the parameter's own unit. An
        unbounded side means unbounded *within this range*.
    hessian_lower, hessian_upper:
        The quadratic-approximation interval
        ``theta_hat * exp(-+ z * relative_sd)`` at the same level, for
        comparison. ``None`` when the relative standard deviation is not finite.
    n_rejected_evaluations:
        Trial parameter vectors the integrator refused across all nested fits.
    truncated_lower, truncated_upper:
        Whether the walk stopped early on that side because the profile had
        already risen far past the threshold. A truncated side is a *bounded*
        side; it is recorded only so that the curve's extent is not mistaken for
        the search range.
    """

    index: int
    name: str
    estimate: float
    level: float
    threshold: float
    points: tuple[ProfilePoint, ...]
    lower: float | None
    upper: float | None
    search_lower: float
    search_upper: float
    hessian_lower: float | None
    hessian_upper: float | None
    truncated_lower: bool
    truncated_upper: bool
    n_rejected_evaluations: int = 0

    @property
    def bounded(self) -> bool:
        """True when both boundaries were found inside the search range."""
        return self.lower is not None and self.upper is not None

    @property
    def log_width(self) -> float:
        """Interval width in natural log units, dimensionless; ``inf`` if open."""
        if self.lower is None or self.upper is None:
            return float("inf")
        if self.lower <= 0.0:
            return float("inf")
        return float(np.log(self.upper) - np.log(self.lower))

    @property
    def hessian_log_width(self) -> float:
        """Quadratic-approximation width in log units; ``inf`` if unavailable."""
        if self.hessian_lower is None or self.hessian_upper is None:
            return float("inf")
        return float(np.log(self.hessian_upper) - np.log(self.hessian_lower))

    @property
    def width_ratio(self) -> float:
        """Profile width divided by quadratic width, dimensionless.

        Greater than 1 means the Hessian was optimistic -- it reported a
        tighter interval than the likelihood supports. ``inf`` when the profile
        is open on a side the Hessian called finite, which is the signature of
        practical non-identifiability.
        """
        h = self.hessian_log_width
        if not np.isfinite(h) or h <= 0.0:
            return float("nan")
        return self.log_width / h

    def contains(self, value: float) -> bool:
        """Whether a value lies in the interval, treating an open side as
        extending to the search boundary and beyond."""
        v = float(value)
        if self.lower is not None and v < self.lower:
            return False
        if self.upper is not None and v > self.upper:
            return False
        return True

    def curve_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """``(values, delta_chi2)`` as arrays, for plotting."""
        return (
            np.array([p.value for p in self.points], dtype=float),
            np.array([p.delta_chi2 for p in self.points], dtype=float),
        )


def _nested_fit(
    problem: CalibrationProblem,
    log_theta_full: np.ndarray,
    log_theta_centre: np.ndarray,
    index: int,
    fixed_log_value: float,
    xtol: float,
    ftol: float,
    max_nfev: int,
    half_box: float,
) -> tuple[np.ndarray, float, bool, int]:
    """Re-optimise all parameters but ``index``, which is held fixed.

    The free parameters are box constrained to ``half_box`` natural log units
    either side of ``log_theta_centre``, the unconstrained estimate. Without
    that box a structurally non-identifiable direction walks until the
    fixed-step integrator becomes unstable, and the profile of an unrelated
    parameter fails for a reason that has nothing to do with that parameter.

    Returns ``(log_theta, chi2, converged, n_rejected)``.
    """
    free = [j for j in range(log_theta_full.size) if j != index]
    x0 = np.clip(
        log_theta_full[free],
        log_theta_centre[free] - half_box + 1e-9,
        log_theta_centre[free] + half_box - 1e-9,
    )
    guard = GuardedResidual(problem)

    def residuals(x: np.ndarray) -> np.ndarray:
        full = log_theta_full.copy()
        full[index] = fixed_log_value
        full[free] = x
        return guard(full)

    sol = least_squares(
        residuals,
        x0,
        bounds=(log_theta_centre[free] - half_box, log_theta_centre[free] + half_box),
        xtol=xtol,
        ftol=ftol,
        gtol=1e-12,
        max_nfev=max_nfev,
    )
    full = log_theta_full.copy()
    full[index] = fixed_log_value
    full[free] = sol.x
    return full, float(2.0 * sol.cost), bool(sol.success), int(guard.n_rejected)


def profile_likelihood(
    problem: CalibrationProblem,
    result: FitResult,
    index: int,
    level: float = DEFAULT_LEVEL,
    n_points: int = 8,
    span_sigma: float = 4.0,
    max_log_half_width: float = 3.0,
    min_log_half_width: float = 0.02,
    stop_factor: float = 6.0,
    xtol: float = 1e-9,
    ftol: float = 1e-11,
    max_nfev: int = 200,
    nuisance_box: float = 2.0,
    refine_steps: int = 5,
) -> ProfileCurve:
    """Profile the likelihood in one parameter and return its interval.

    Parameters
    ----------
    problem:
        The calibration problem the fit came from.
    result:
        The unconstrained fit. Its ``theta`` is the centre of the grid and its
        ``chi2`` is the reference the increase is measured against.
    index:
        Which parameter to profile.
    level:
        Confidence level, dimensionless.
    n_points:
        Grid points on *each* side of the estimate.
    span_sigma:
        Half width of the grid in units of the Hessian relative standard
        deviation. Four sigma covers a 95 per cent interval with room to spare
        when the likelihood is near quadratic.
    max_log_half_width:
        Cap on the half width in natural log units. The default 3.0 is a factor
        of ``e**3 = 20.1`` either side of the estimate; a parameter whose
        profile has not crossed the threshold within a factor of 20 is not
        determined by the data in any useful sense, and reporting the side as
        open is the honest result.
    min_log_half_width:
        Floor on the half width, so that a very well determined parameter still
        gets a grid wide enough to interpolate on.
    stop_factor:
        Stop walking outwards once ``delta_chi2`` exceeds
        ``stop_factor * threshold``. Saves the far tail, which no interval
        needs.
    xtol, ftol, max_nfev:
        Tolerances of the nested optimisations.
    nuisance_box:
        Half width, in natural log units, of the box the nuisance parameters are
        constrained to around the unconstrained estimate. The default 2.0 is a
        factor of ``e**2 = 7.4`` either side. A nuisance parameter that ends on
        this box makes the nested optimisation a constrained one, which is
        recorded in ``ProfileCurve.points[...].converged``.
    refine_steps:
        Bisection steps spent on each threshold crossing after the grid walk.
        The grid alone brackets the crossing only to the grid spacing, which on
        a wide search range is coarse; five bisections shrink the bracket by a
        factor of 32 at a cost of five extra nested fits per side. Setting this
        to 0 falls back to interpolating within the raw grid interval.

    Returns
    -------
    ProfileCurve

    Raises
    ------
    ValueError
        On an out-of-range index, level, or non-positive grid settings.
    """
    p = result.theta.size
    if not 0 <= index < p:
        raise ValueError(f"index must lie in [0, {p - 1}]")
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    for name, value in (
        ("span_sigma", span_sigma),
        ("max_log_half_width", max_log_half_width),
        ("min_log_half_width", min_log_half_width),
        ("stop_factor", stop_factor),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if min_log_half_width >= max_log_half_width:
        raise ValueError("min_log_half_width must be below max_log_half_width")
    threshold = chi2_threshold(level)
    base_chi2 = chi_square(problem, result.theta)
    log_theta = np.log(result.theta)
    rel_sd = float(result.relative_sd[index])
    if np.isfinite(rel_sd) and rel_sd > 0.0:
        half = min(span_sigma * rel_sd, max_log_half_width)
    else:
        half = max_log_half_width
    half = float(min(max(half, min_log_half_width), max_log_half_width))
    offsets = np.linspace(half / n_points, half, n_points)

    evaluated: dict[int, list[tuple[float, float, np.ndarray]]] = {-1: [], 1: []}
    points: list[ProfilePoint] = [
        ProfilePoint(0.0, float(result.theta[index]), base_chi2, 0.0, True)
    ]
    truncated = {-1: False, 1: False}
    n_rejected = 0

    def evaluate(
        direction: int, off: float, warm: np.ndarray
    ) -> tuple[float, np.ndarray]:
        nonlocal n_rejected
        target = log_theta[index] + direction * float(off)
        solution, chi2_value, ok, rejected = _nested_fit(
            problem,
            warm,
            log_theta,
            index,
            target,
            xtol,
            ftol,
            max_nfev,
            float(nuisance_box),
        )
        n_rejected += rejected
        delta = float(chi2_value - base_chi2)
        points.append(
            ProfilePoint(
                log_offset=direction * float(off),
                value=float(np.exp(target)),
                chi2=chi2_value,
                delta_chi2=delta,
                converged=ok,
            )
        )
        evaluated[direction].append((float(off), delta, solution))
        return delta, solution

    for direction in (-1, 1):
        warm = log_theta.copy()
        for off in offsets:
            delta, warm = evaluate(direction, float(off), warm)
            if delta > stop_factor * threshold:
                truncated[direction] = bool(off < offsets[-1] - 1e-12)
                break

    def bracket(direction: int) -> tuple[float, float, float, float] | None:
        """Innermost (off_lo, delta_lo, off_hi, delta_hi) spanning the threshold."""
        seq = sorted(evaluated[direction], key=lambda item: item[0])
        prev_off, prev_delta = 0.0, 0.0
        for off, delta, _sol in seq:
            if prev_delta <= threshold < delta:
                return (prev_off, prev_delta, off, delta)
            prev_off, prev_delta = off, delta
        return None

    def refine(direction: int) -> float | None:
        found = bracket(direction)
        if found is None:
            return None
        lo_off, lo_delta, hi_off, hi_delta = found
        warm = log_theta.copy()
        for _ in range(int(refine_steps)):
            mid = 0.5 * (lo_off + hi_off)
            delta, warm = evaluate(direction, mid, warm)
            if delta <= threshold:
                lo_off, lo_delta = mid, delta
            else:
                hi_off, hi_delta = mid, delta
        span = hi_delta - lo_delta
        if span <= 0.0:
            off = hi_off
        else:
            off = lo_off + (threshold - lo_delta) / span * (hi_off - lo_off)
        return float(np.exp(log_theta[index] + direction * off))

    lower = refine(-1)
    upper = refine(1)
    points.sort(key=lambda q: q.log_offset)

    z = float(np.sqrt(threshold))
    # A quadratic interval is offered only when it is both finite and
    # representable. A relative standard deviation of 400 gives exp(784), which
    # overflows; an interval that cannot be written down is not an interval, and
    # reporting it as absent is the same statement as reporting it as infinite.
    if np.isfinite(rel_sd) and 0.0 < z * rel_sd < 300.0:
        h_lo: float | None = float(result.theta[index] * np.exp(-z * rel_sd))
        h_hi: float | None = float(result.theta[index] * np.exp(z * rel_sd))
    else:
        h_lo = h_hi = None
    return ProfileCurve(
        index=int(index),
        name=result.parameter_names[index],
        estimate=float(result.theta[index]),
        level=float(level),
        threshold=threshold,
        points=tuple(points),
        lower=lower,
        upper=upper,
        search_lower=float(result.theta[index] * np.exp(-half)),
        search_upper=float(result.theta[index] * np.exp(half)),
        hessian_lower=h_lo,
        hessian_upper=h_hi,
        truncated_lower=truncated[-1],
        truncated_upper=truncated[1],
        n_rejected_evaluations=n_rejected,
    )


def profile_all(
    problem: CalibrationProblem,
    result: FitResult,
    indices: Sequence[int] | None = None,
    **kwargs: object,
) -> tuple[ProfileCurve, ...]:
    """Profile several parameters. ``indices=None`` profiles all of them.

    Keyword arguments are passed through to :func:`profile_likelihood`.
    """
    idx = range(result.theta.size) if indices is None else [int(i) for i in indices]
    return tuple(profile_likelihood(problem, result, i, **kwargs) for i in idx)  # type: ignore[arg-type]
