"""Importance sampling with a declared tilting family.

Declared family
---------------
The sampling family is the **mean-shift (translation) family** on standard
normal space,

    q_theta(x) = phi(x - theta),      theta in R^d

where ``phi`` is the standard multivariate normal density with identity
covariance.  For the Gaussian this family *is* the exponential tilting family:

    exp(theta . x - |theta|^2 / 2) phi(x) = phi(x - theta)

so "mean shift" and "exponential tilt with natural parameter theta" name the
same object here.  The likelihood ratio is therefore available in closed form,

    w(x) = phi(x) / phi(x - theta) = exp(-theta . x + |theta|^2 / 2)

and the estimator is

    p_hat = (1 / n) * sum_i w(x_i) 1[g(x_i) <= 0],     x_i ~ N(theta, I)

which is unbiased for every ``theta`` because ``q_theta`` is positive
everywhere ``phi`` is.  Source: Rubinstein, R. Y. and Kroese, D. P. (2016),
"Simulation and the Monte Carlo Method", 3rd edition, Wiley; and Owen, A. B.
(2013), "Monte Carlo theory, methods and examples", chapter 9, for the
variance, the effective sample size, and the warning that a badly chosen
proposal can have infinite variance.

What the family cannot do
-------------------------
The covariance is fixed at the identity.  A failure region that is strongly
curved or that has several disconnected components is not well served by a
single mean shift, and no amount of tuning ``theta`` repairs that; the rippled
limit state in :mod:`rareverify.limitstates` is the mild version of this and
subset simulation (:mod:`rareverify.subset`) is the alternative this package
offers for it.

Choosing theta
--------------
For a linear limit state ``g = beta - a . x`` the variance-minimising mean
shift within this family is the design point ``theta = beta a``: it moves the
sampling density onto the most probable failure point, which is the standard
first-order reliability result (Hasofer, A. M. and Lind, N. C. (1974), Journal
of the Engineering Mechanics Division, ASCE, volume 100).  This is what
:func:`analytic_mean_shift` returns.  :func:`find_design_point` recovers the
same point numerically from ``g`` alone, which is what the learned surrogate
has to do and what :func:`oracle_mean_shift` does with the true ``g`` to
provide an upper reference on the rippled instance.

**Importance sampling is not automatically better than no importance
sampling.** Under-tilting leaves the failure region rare; over-tilting makes
the weights degenerate; tilting in the wrong direction or orthogonal to the
design direction adds weight variance for no gain.  :func:`scaled_mean_shift`
and :func:`orthogonal_mean_shift` exist to generate those cases on purpose, and
validation/validate_is_worse.py measures where the variance reduction factor
drops below 1.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import optimize

from .estimate import RareEventEstimate
from .limitstates import LimitState

__all__ = [
    "MeanShiftTilt",
    "analytic_mean_shift",
    "find_design_point",
    "find_design_point_radial",
    "importance_sampling",
    "oracle_mean_shift",
    "orthogonal_mean_shift",
    "scaled_mean_shift",
]


@dataclass(frozen=True)
class MeanShiftTilt:
    """A member of the declared mean-shift family ``q(x) = phi(x - theta)``.

    Attributes
    ----------
    theta
        Mean shift, shape ``(d,)``, in standard-normal units.  ``theta = 0``
        recovers plain Monte Carlo exactly, with all weights equal to 1.
    label
        Short name recorded in results so a reported variance reduction factor
        can never be separated from the tilt that produced it.
    """

    theta: np.ndarray
    label: str = "mean-shift"

    def __post_init__(self) -> None:
        theta = np.asarray(self.theta, dtype=float).ravel()
        if theta.size == 0:
            raise ValueError("theta must have at least one component")
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta must be finite in every component")
        object.__setattr__(self, "theta", theta)

    @property
    def dimension(self) -> int:
        """Number of components of ``theta``."""
        return int(self.theta.size)

    @property
    def norm(self) -> float:
        """Euclidean norm ``|theta|`` in standard-normal units."""
        return float(np.linalg.norm(self.theta))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` samples from ``N(theta, I)``; shape ``(n, d)``."""
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        return rng.standard_normal((n, self.dimension)) + self.theta

    def log_weight(self, x: np.ndarray) -> np.ndarray:
        """Log likelihood ratio ``-theta . x + |theta|^2 / 2``; shape ``(n,)``."""
        arr = np.atleast_2d(np.asarray(x, dtype=float))
        if arr.shape[1] != self.dimension:
            raise ValueError(
                f"x must have {self.dimension} columns, got shape {arr.shape}"
            )
        return -(arr @ self.theta) + 0.5 * float(self.theta @ self.theta)

    def weight(self, x: np.ndarray) -> np.ndarray:
        """Likelihood ratio ``exp(log_weight(x))``; shape ``(n,)``, dimensionless."""
        return np.exp(self.log_weight(x))


def analytic_mean_shift(limit_state: LimitState) -> MeanShiftTilt:
    """The tilt at the limit state's analytically known design point.

    For the rippled limit state this is the design point of the *smooth part*,
    which is deliberately not optimal; that is the baseline the surrogate has
    to beat.
    """
    return MeanShiftTilt(limit_state.design_point(), label="analytic-design-point")


def scaled_mean_shift(limit_state: LimitState, scale: float) -> MeanShiftTilt:
    """Tilt at ``scale`` times the analytic design point.

    ``scale = 0`` gives plain Monte Carlo, ``scale = 1`` the design point, and
    negative values tilt away from the failure region.  Used to generate the
    under-tilt, over-tilt and wrong-direction cases on purpose.
    """
    scale = float(scale)
    if not math.isfinite(scale):
        raise ValueError(f"scale must be finite, got {scale}")
    return MeanShiftTilt(
        scale * limit_state.design_point(), label=f"scale={scale:g}"
    )


def orthogonal_mean_shift(limit_state: LimitState, scale: float = 1.0) -> MeanShiftTilt:
    """Tilt of magnitude ``scale * |design point|`` orthogonal to the design direction.

    This shifts the sampling density sideways: it does not make the failure
    region less rare, and it adds weight variance.  Its variance reduction
    factor should therefore be below 1, and validation measures that it is.

    Raises
    ------
    ValueError
        If the limit state has fewer than two input dimensions, where no
        orthogonal direction exists.
    """
    point = np.asarray(limit_state.design_point(), dtype=float)
    if point.size < 2:
        raise ValueError(
            "an orthogonal tilt needs at least two input dimensions, got "
            f"{point.size}"
        )
    norm = float(np.linalg.norm(point))
    if norm == 0.0:
        raise ValueError("design point is the origin; no orthogonal direction defined")
    direction = point / norm
    # Gram-Schmidt against the first coordinate axis that is not parallel to it.
    axis = int(np.argmin(np.abs(direction)))
    candidate = np.zeros_like(direction)
    candidate[axis] = 1.0
    orthogonal = candidate - (candidate @ direction) * direction
    orthogonal /= float(np.linalg.norm(orthogonal))
    return MeanShiftTilt(scale * norm * orthogonal, label=f"orthogonal={scale:g}")


def find_design_point(
    g: Callable[[np.ndarray], np.ndarray],
    dimension: int,
    n_starts: int = 12,
    rng: np.random.Generator | None = None,
    start_radius: float = 4.0,
    max_iter: int = 300,
    start_point: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Numerically minimise ``|x|`` subject to ``g(x) = 0``.

    Parameters
    ----------
    g
        Limit-state callable accepting shape ``(n, dimension)`` and returning
        shape ``(n,)``.  May be a surrogate mean.
    dimension
        Input dimension, ``>= 1``.
    n_starts
        Number of random SLSQP starts, ``>= 1``.  Multi-start is necessary
        because a rippled or learned boundary has several local design points.
    rng
        Random generator for the starts.
    start_radius
        Scale of the random starting points, in standard-normal units.
    max_iter
        SLSQP iteration cap per start.
    start_point
        If given, the first start is this point instead of a random one.  Used
        to polish a ray-search result.

    Returns
    -------
    tuple
        ``(point, converged)``.  ``point`` has shape ``(dimension,)``;
        ``converged`` is True only if at least one start satisfied the
        constraint to within ``1e-6`` in ``g`` units.
    """
    if dimension < 1:
        raise ValueError(f"dimension must be at least 1, got {dimension}")
    if n_starts < 1:
        raise ValueError(f"n_starts must be at least 1, got {n_starts}")
    generator = np.random.default_rng(0) if rng is None else rng

    def scalar_g(x: np.ndarray) -> float:
        return float(np.asarray(g(np.atleast_2d(x)))[0])

    best_point: np.ndarray | None = None
    best_norm = math.inf
    converged = False
    constraint = {"type": "eq", "fun": scalar_g}
    for start_index in range(n_starts):
        if start_point is not None and start_index == 0:
            x0 = np.asarray(start_point, dtype=float).ravel().copy()
        else:
            x0 = generator.standard_normal(dimension) * start_radius
        try:
            result = optimize.minimize(
                lambda x: float(x @ x),
                x0,
                jac=lambda x: 2.0 * x,
                method="SLSQP",
                constraints=[constraint],
                options={"maxiter": max_iter, "ftol": 1e-12},
            )
        except (ValueError, FloatingPointError):
            continue
        point = np.asarray(result.x, dtype=float)
        if not np.all(np.isfinite(point)):
            continue
        residual = abs(scalar_g(point))
        norm = float(np.linalg.norm(point))
        if residual <= 1e-6 and norm < best_norm:
            best_norm = norm
            best_point = point
            converged = True
        elif best_point is None and norm < best_norm:
            best_norm = norm
            best_point = point
    if best_point is None:
        best_point = np.zeros(dimension)
    return best_point, converged



def find_design_point_radial(
    g: Callable[[np.ndarray], np.ndarray],
    dimension: int,
    n_directions: int = 512,
    max_radius: float = 8.0,
    n_grid: int = 48,
    n_bisect: int = 24,
    n_refine: int = 1,
    refine_spread: float = 0.15,
    polish: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, bool]:
    """Design point by ray search: minimise ``|x|`` over directions.

    For each of ``n_directions`` unit directions the first radius at which
    ``g`` changes sign is located by a coarse radial grid followed by
    bisection, and the smallest such radius over all directions is returned.
    Every evaluation is batched across directions, so the whole search costs
    ``n_grid + n_bisect`` vectorised calls to ``g`` rather than thousands of
    single-row calls; that is the reason this routine exists alongside
    :func:`find_design_point`, which is a general constrained optimiser.

    Assumption and validity
    -----------------------
    The failure region is assumed **star-shaped with respect to the origin
    along each sampled ray within** ``max_radius``: the bisection locates the
    first sign change along a ray and ignores any later ones.  A failure region
    that does not contain a ray's first crossing, or one that is disconnected
    in a way the directions miss, will not be found correctly.  The two methods
    are cross-checked against each other and against the closed-form design
    point in validation/validate_surrogate.py, which is how this assumption is
    held to account rather than asserted.

    Parameters
    ----------
    g
        Limit-state callable, shape ``(n, dimension) -> (n,)``.
    dimension
        Input dimension, ``>= 1``.
    n_directions
        Directions sampled uniformly on the unit sphere, ``>= 1``.  The
        positive and negative coordinate axes are always included as well.
    max_radius
        Largest radius searched, in standard-normal units.  A design point
        beyond it is not found, and ``converged`` is then False.
    n_grid
        Coarse radial grid points, ``>= 2``.
    n_bisect
        Bisection steps after bracketing, ``>= 1``.
    n_refine
        Rounds of local direction refinement around the current best, ``>= 0``.
    refine_spread
        Angular spread of the refinement, dimensionless, ``> 0``.
    polish
        Run one SLSQP step from the best ray-search point.  The ray search
        samples directions, so its accuracy degrades with dimension: on the
        rippled limit state in six dimensions, 192 directions recover
        ``beta = 3.1091`` against the true 3.072735, a 1.2 % error, falling to
        3.0763 only at 8192 directions (measured in
        validation/validate_surrogate.py).  The polish step removes that error
        at the cost of a few hundred single-point evaluations and is on by
        default; set it False to see the raw ray-search accuracy.
    rng
        Random generator.

    Returns
    -------
    tuple
        ``(point, converged)``; ``converged`` is False when no ray crossed zero
        inside ``max_radius``.
    """
    if dimension < 1:
        raise ValueError(f"dimension must be at least 1, got {dimension}")
    if n_directions < 1:
        raise ValueError(f"n_directions must be at least 1, got {n_directions}")
    if max_radius <= 0.0:
        raise ValueError(f"max_radius must be positive, got {max_radius}")
    if n_grid < 2:
        raise ValueError(f"n_grid must be at least 2, got {n_grid}")
    if n_bisect < 1:
        raise ValueError(f"n_bisect must be at least 1, got {n_bisect}")
    if n_refine < 0:
        raise ValueError(f"n_refine must be non-negative, got {n_refine}")
    if refine_spread <= 0.0:
        raise ValueError(f"refine_spread must be positive, got {refine_spread}")
    generator = np.random.default_rng(0) if rng is None else rng

    def unit(rows: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return rows / norms

    axes = np.vstack([np.eye(dimension), -np.eye(dimension)])
    directions = np.vstack([axes, unit(generator.standard_normal((n_directions, dimension)))])

    best_point = np.zeros(dimension)
    best_radius = math.inf
    converged = False

    for round_index in range(n_refine + 1):
        grid = np.linspace(0.0, max_radius, n_grid)
        values = np.empty((n_grid, directions.shape[0]))
        for i, r in enumerate(grid):
            values[i] = np.asarray(g(directions * r))
        negative = values <= 0.0
        has_crossing = negative.any(axis=0)
        if np.any(has_crossing):
            first = np.argmax(negative, axis=0)
            usable = has_crossing & (first > 0)
            if np.any(usable):
                idx = np.flatnonzero(usable)
                lo = grid[first[idx] - 1]
                hi = grid[first[idx]]
                dirs = directions[idx]
                for _ in range(n_bisect):
                    mid = 0.5 * (lo + hi)
                    mid_neg = np.asarray(g(dirs * mid[:, None])) <= 0.0
                    hi = np.where(mid_neg, mid, hi)
                    lo = np.where(mid_neg, lo, mid)
                radii = hi
                winner = int(np.argmin(radii))
                if float(radii[winner]) < best_radius:
                    best_radius = float(radii[winner])
                    best_point = dirs[winner] * best_radius
                    converged = True
        if round_index < n_refine and converged:
            centre = best_point / best_radius
            perturbed = centre + refine_spread * generator.standard_normal(
                (n_directions, dimension)
            )
            directions = np.vstack([centre[None, :], unit(perturbed)])
            max_radius = min(max_radius, best_radius * 1.5)

    if polish and converged:
        polished, polished_ok = find_design_point(
            g, dimension, n_starts=1, rng=rng, start_radius=0.0, start_point=best_point
        )
        if polished_ok and float(np.linalg.norm(polished)) < best_radius:
            best_point = polished
    return best_point, converged


def oracle_mean_shift(
    limit_state: LimitState, n_starts: int = 24, rng: np.random.Generator | None = None
) -> MeanShiftTilt:
    """Tilt at the true design point found numerically from the true ``g``.

    This is an upper reference, not a method: it spends unlimited exact
    evaluations of the true limit state inside an optimiser, which a real
    campaign with an expensive simulator cannot do.  It bounds what any
    mean-shift tilt could achieve.
    """
    point, _ = find_design_point(
        limit_state.g, limit_state.dimension, n_starts=n_starts, rng=rng
    )
    return MeanShiftTilt(point, label="oracle-design-point")


def importance_sampling(
    limit_state: LimitState,
    tilt: MeanShiftTilt,
    n_samples: int,
    rng: np.random.Generator | None = None,
    batch_size: int = 250_000,
) -> RareEventEstimate:
    """Estimate ``P(g(X) <= 0)`` by mean-shift importance sampling.

    Parameters
    ----------
    limit_state
        Limit state to evaluate.
    tilt
        Member of the declared mean-shift family.
    n_samples
        Number of samples, ``>= 1``.
    rng
        Random generator; a default-seeded one is created if omitted.
    batch_size
        Samples per batch, to bound peak memory.

    Returns
    -------
    RareEventEstimate
        ``is_binomial`` is True only when ``theta = 0``, where the estimator
        degenerates to the crude count.  ``diagnostics`` carries
        ``max_weight``, ``theta_norm`` and ``weight_cov`` (the coefficient of
        variation of the non-zero contributions), which is the quantity that
        blows up when the tilt is wrong.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if tilt.dimension != limit_state.dimension:
        raise ValueError(
            f"tilt dimension {tilt.dimension} does not match limit state "
            f"dimension {limit_state.dimension}"
        )
    generator = np.random.default_rng(0) if rng is None else rng
    start = time.perf_counter()
    pieces: list[np.ndarray] = []
    failures = 0
    remaining = int(n_samples)
    while remaining > 0:
        take = min(batch_size, remaining)
        x = tilt.sample(take, generator)
        failed = limit_state.g(x) <= 0.0
        count = int(np.count_nonzero(failed))
        failures += count
        if count:
            pieces.append(tilt.weight(x[failed]))
        remaining -= take
    wall = time.perf_counter() - start
    contributions = (
        np.concatenate(pieces) if pieces else np.zeros(0, dtype=float)
    )
    total = float(contributions.sum())
    estimate = total / n_samples
    sum_sq = float((contributions**2).sum())
    # Var of the per-sample contribution, with the n_samples - m zeros included.
    second_moment = sum_sq / n_samples
    variance = max(second_moment - estimate**2, 0.0)
    se = math.sqrt(variance / n_samples)
    ess = (total**2 / sum_sq) if sum_sq > 0.0 else 0.0
    weight_cov = (
        float(contributions.std(ddof=1) / contributions.mean())
        if contributions.size > 1 and contributions.mean() > 0.0
        else 0.0
    )
    return RareEventEstimate(
        method="importance-sampling",
        estimate=estimate,
        standard_error=se,
        n_samples=int(n_samples),
        true_evaluations=int(n_samples),
        n_failures=failures,
        is_binomial=tilt.norm == 0.0,
        effective_sample_size=float(ess),
        wall_seconds=wall,
        contributions=contributions,
        diagnostics={
            "theta_norm": tilt.norm,
            "max_weight": float(contributions.max()) if contributions.size else 0.0,
            "weight_cov": weight_cov,
        },
    )
