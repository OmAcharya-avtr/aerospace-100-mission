"""Ground truth 2: Monte Carlo over the three-dimensional relative state.

Monte Carlo is the reference in this package, not a competitor to the analytic
methods. Three estimators are provided, and the differences between them are
the measurement that matters:

* :func:`pc_monte_carlo` samples the relative position offset from the combined
  3-D covariance, propagates each sample along a rectilinear relative
  trajectory over an unbounded time interval, and counts the samples whose
  perpendicular miss falls inside the hard body. Under the short-encounter
  assumptions this is the same quantity the 2-D analytic methods compute, so it
  validates the encounter-plane projection itself and not merely the integral.
* :func:`pc_monte_carlo_window` restricts the time interval to ``+/- T``, which
  is the probability of a conjunction *within the screening window* and is
  strictly smaller.
* :func:`pc_monte_carlo_curved` adds a declared constant relative acceleration,
  which breaks the rectilinear assumption the 2-D formulation rests on.

Confidence intervals are Clopper-Pearson exact binomial intervals
(Clopper, C. J., and Pearson, E. S., 1934, "The use of confidence or fiducial
limits illustrated in the case of the binomial", *Biometrika* 26(4), 404-413),
not normal approximations, because the hit counts at operationally interesting
probabilities are small enough that the normal approximation misstates the
interval.

Units: metres, metres per second, metres per second squared, seconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import beta

_DEFAULT_CHUNK = 200_000


@dataclass(frozen=True)
class MonteCarloResult:
    """A Monte Carlo collision-probability estimate with its exact interval.

    Attributes
    ----------
    pc
        Point estimate ``hits / n_samples``, dimensionless.
    hits
        Number of samples whose minimum separation was within the hard body.
    n_samples
        Number of samples drawn.
    ci_low, ci_high
        Clopper-Pearson interval at ``confidence``, dimensionless.
    confidence
        Nominal coverage of the interval, dimensionless.
    standard_error
        ``sqrt(p (1 - p) / n)``, reported for scaling arguments only; the
        interval is the quantity to quote.
    seed
        The seed the generator was constructed with, so the run is reproducible.
    """

    pc: float
    hits: int
    n_samples: int
    ci_low: float
    ci_high: float
    confidence: float
    standard_error: float
    seed: int

    @property
    def ci_width(self) -> float:
        """Width of the confidence interval, dimensionless."""
        return self.ci_high - self.ci_low

    @property
    def relative_ci_width(self) -> float:
        """Interval width divided by the point estimate; infinite if no hits."""
        if self.pc <= 0.0:
            return math.inf
        return self.ci_width / self.pc


def clopper_pearson(hits: int, n_samples: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact binomial confidence interval for ``hits`` successes in ``n_samples``.

    Returns ``(low, high)``, dimensionless. The interval is conservative by
    construction: its actual coverage is at least ``confidence``. Zero hits
    give ``(0, 1 - (alpha/2)^(1/n))``, which is the only honest statement
    available when nothing was observed.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples!r}")
    if not 0 <= hits <= n_samples:
        raise ValueError(f"hits must lie in [0, {n_samples}], got {hits!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence!r}")
    alpha = 1.0 - confidence
    low = 0.0 if hits == 0 else float(beta.ppf(alpha / 2.0, hits, n_samples - hits + 1))
    high = (
        1.0
        if hits == n_samples
        else float(beta.ppf(1.0 - alpha / 2.0, hits + 1, n_samples - hits))
    )
    return low, high


def _package(hits: int, n_samples: int, confidence: float, seed: int) -> MonteCarloResult:
    pc = hits / n_samples
    low, high = clopper_pearson(hits, n_samples, confidence)
    return MonteCarloResult(
        pc=pc,
        hits=hits,
        n_samples=n_samples,
        ci_low=low,
        ci_high=high,
        confidence=confidence,
        standard_error=math.sqrt(max(pc * (1.0 - pc), 0.0) / n_samples),
        seed=seed,
    )


def _validate_common(
    covariance_m2: ArrayLike,
    miss_vector_m: ArrayLike,
    relative_velocity_mps: ArrayLike,
    radius_m: float,
    n_samples: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    cov = np.asarray(covariance_m2, dtype=float)
    if cov.shape != (3, 3):
        raise ValueError(f"covariance_m2 must have shape (3, 3), got {cov.shape}")
    sym = 0.5 * (cov + cov.T)
    if np.linalg.eigvalsh(sym).min() < 0.0:
        raise ValueError("covariance_m2 must be positive semi-definite")
    miss = np.asarray(miss_vector_m, dtype=float).reshape(-1)
    vel = np.asarray(relative_velocity_mps, dtype=float).reshape(-1)
    if miss.shape != (3,) or vel.shape != (3,):
        raise ValueError("miss_vector_m and relative_velocity_mps must have 3 components")
    if radius_m < 0.0:
        raise ValueError(f"radius_m must be non-negative metres, got {radius_m!r}")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples!r}")
    if float(vel @ vel) <= 0.0:
        raise ValueError("relative velocity is zero; no encounter is defined")
    return sym, miss, vel


def pc_monte_carlo(
    covariance_m2: ArrayLike,
    miss_vector_m: ArrayLike,
    relative_velocity_mps: ArrayLike,
    radius_m: float,
    *,
    n_samples: int = 1_000_000,
    seed: int = 0,
    confidence: float = 0.95,
    chunk: int = _DEFAULT_CHUNK,
) -> MonteCarloResult:
    """Reference Pc by 3-D Monte Carlo under rectilinear relative motion.

    Each sample draws a relative-position offset from ``covariance_m2``, adds it
    to the miss vector, and computes the minimum separation along the
    rectilinear relative trajectory analytically as the component of the
    offset relative position perpendicular to the relative velocity. No time
    discretisation is involved, so the only error is sampling error.

    Parameters
    ----------
    covariance_m2
        Combined 3x3 position covariance at the time of closest approach,
        metres squared.
    miss_vector_m
        Relative position at closest approach, metres, shape ``(3,)``.
    relative_velocity_mps
        Relative velocity, metres per second, shape ``(3,)``.
    radius_m
        Combined hard-body radius, metres.
    n_samples
        Samples to draw. Interval width scales as ``n^-1/2``; the measured
        scaling is in ``validation/validate_montecarlo_output.txt``.
    seed
        Seed for :func:`numpy.random.default_rng`. The same seed reproduces the
        same hit count exactly.
    confidence
        Nominal coverage of the Clopper-Pearson interval.
    chunk
        Samples drawn per batch, to bound peak memory.

    Returns
    -------
    MonteCarloResult
    """
    cov, miss, vel = _validate_common(
        covariance_m2, miss_vector_m, relative_velocity_mps, radius_m, n_samples
    )
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk!r}")
    unit = vel / np.linalg.norm(vel)
    rng = np.random.default_rng(seed)
    radius_sq = radius_m * radius_m
    hits = 0
    remaining = n_samples
    while remaining > 0:
        batch = min(chunk, remaining)
        offsets = rng.multivariate_normal(miss, cov, size=batch, method="cholesky")
        along = offsets @ unit
        perpendicular_sq = np.einsum("ij,ij->i", offsets, offsets) - along * along
        hits += int(np.count_nonzero(perpendicular_sq <= radius_sq))
        remaining -= batch
    return _package(hits, n_samples, confidence, seed)


def pc_monte_carlo_window(
    covariance_m2: ArrayLike,
    miss_vector_m: ArrayLike,
    relative_velocity_mps: ArrayLike,
    radius_m: float,
    window_s: float,
    *,
    n_samples: int = 1_000_000,
    seed: int = 0,
    confidence: float = 0.95,
    chunk: int = _DEFAULT_CHUNK,
) -> MonteCarloResult:
    """Pc restricted to a finite screening window ``[-window_s, +window_s]``.

    Rectilinear motion is retained, so the per-sample minimum over the window
    is still exact: the unconstrained minimising time is projected onto the
    window and the separation evaluated there. The result is bounded above by
    :func:`pc_monte_carlo` and converges to it as the window grows, which is
    the finite-duration limit of the 2-D formulation and is measured in
    ``validation/validate_nonlinear.py``.
    """
    cov, miss, vel = _validate_common(
        covariance_m2, miss_vector_m, relative_velocity_mps, radius_m, n_samples
    )
    if not math.isfinite(window_s) or window_s <= 0.0:
        raise ValueError(f"window_s must be a positive finite number of seconds, got {window_s!r}")
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk!r}")
    speed_sq = float(vel @ vel)
    rng = np.random.default_rng(seed)
    radius_sq = radius_m * radius_m
    hits = 0
    remaining = n_samples
    while remaining > 0:
        batch = min(chunk, remaining)
        offsets = rng.multivariate_normal(miss, cov, size=batch, method="cholesky")
        t_star = -(offsets @ vel) / speed_sq
        np.clip(t_star, -window_s, window_s, out=t_star)
        closest = offsets + t_star[:, None] * vel
        hits += int(
            np.count_nonzero(np.einsum("ij,ij->i", closest, closest) <= radius_sq)
        )
        remaining -= batch
    return _package(hits, n_samples, confidence, seed)


def pc_monte_carlo_curved(
    covariance_m2: ArrayLike,
    miss_vector_m: ArrayLike,
    relative_velocity_mps: ArrayLike,
    radius_m: float,
    window_s: float,
    relative_acceleration_mps2: ArrayLike,
    *,
    n_samples: int = 200_000,
    seed: int = 0,
    confidence: float = 0.95,
    time_nodes: int = 129,
    refinements: int = 3,
    chunk: int = 20_000,
) -> MonteCarloResult:
    """Pc with a declared constant relative acceleration over a finite window.

    This deliberately violates the rectilinear assumption that every 2-D
    formulation in this package rests on. The relative trajectory is
    ``rho(t) = rho_0 + v t + a t^2 / 2`` and the minimum separation over
    ``[-window_s, window_s]`` is found by evaluating on ``time_nodes`` uniform
    nodes and then bisecting the bracketing interval ``refinements`` times, so
    the time resolution is ``2 window_s / ((time_nodes - 1) * 2^refinements)``.
    The resulting discretisation bias is one-sided — a grid can only
    *overestimate* the minimum separation and therefore *underestimate* Pc —
    and ``validation/validate_nonlinear.py`` measures it by refining the grid.

    Parameters
    ----------
    relative_acceleration_mps2
        Constant relative acceleration, metres per second squared, shape
        ``(3,)``. Zero recovers the windowed rectilinear case.
    window_s
        Half-width of the time window, seconds.
    time_nodes
        Number of uniform time nodes; must be at least 3.
    refinements
        Bisection refinements applied to the bracketing interval.

    Returns
    -------
    MonteCarloResult
    """
    cov, miss, vel = _validate_common(
        covariance_m2, miss_vector_m, relative_velocity_mps, radius_m, n_samples
    )
    accel = np.asarray(relative_acceleration_mps2, dtype=float).reshape(-1)
    if accel.shape != (3,):
        raise ValueError("relative_acceleration_mps2 must have 3 components")
    if not math.isfinite(window_s) or window_s <= 0.0:
        raise ValueError(f"window_s must be a positive finite number of seconds, got {window_s!r}")
    if time_nodes < 3:
        raise ValueError(f"time_nodes must be at least 3, got {time_nodes!r}")
    if refinements < 0:
        raise ValueError(f"refinements must be non-negative, got {refinements!r}")
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk!r}")

    rng = np.random.default_rng(seed)
    radius_sq = radius_m * radius_m
    times = np.linspace(-window_s, window_s, time_nodes)
    step = float(times[1] - times[0])
    hits = 0
    remaining = n_samples
    while remaining > 0:
        batch = min(chunk, remaining)
        offsets = rng.multivariate_normal(miss, cov, size=batch, method="cholesky")
        separations = _separation_sq(offsets, vel, accel, times)
        index = np.argmin(separations, axis=1)
        best = separations[np.arange(batch), index]
        centre = times[index]
        half = step
        for _ in range(refinements):
            half *= 0.5
            candidates = np.clip(
                np.stack((centre - half, centre + half), axis=1), -window_s, window_s
            )
            values = np.stack(
                [
                    _separation_sq_at(offsets, vel, accel, candidates[:, k])
                    for k in range(2)
                ],
                axis=1,
            )
            trial = np.argmin(values, axis=1)
            trial_value = values[np.arange(batch), trial]
            improved = trial_value < best
            best = np.where(improved, trial_value, best)
            centre = np.where(improved, candidates[np.arange(batch), trial], centre)
        hits += int(np.count_nonzero(best <= radius_sq))
        remaining -= batch
    return _package(hits, n_samples, confidence, seed)


def _separation_sq(
    offsets: NDArray[np.float64],
    vel: NDArray[np.float64],
    accel: NDArray[np.float64],
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Squared separation, shape ``(n_samples, n_times)``, metres squared."""
    positions = (
        offsets[:, None, :]
        + times[None, :, None] * vel[None, None, :]
        + 0.5 * (times[None, :, None] ** 2) * accel[None, None, :]
    )
    return np.einsum("ijk,ijk->ij", positions, positions)


def _separation_sq_at(
    offsets: NDArray[np.float64],
    vel: NDArray[np.float64],
    accel: NDArray[np.float64],
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Squared separation at one time per sample, shape ``(n_samples,)``."""
    positions = offsets + times[:, None] * vel + 0.5 * (times[:, None] ** 2) * accel
    return np.einsum("ij,ij->i", positions, positions)
