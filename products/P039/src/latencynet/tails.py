"""Tail-latency estimation and the order-statistic theory that bounds it.

Why the tail and not the mean
-----------------------------
A deadline is violated by a single late pass, so the quantity a real-time
designer needs is a high quantile of the end-to-end latency distribution --
p99 or p99.9 -- not its mean. A mean can be comfortable while the p99.9 is
three times the deadline.

Quantile definitions
--------------------
Two definitions are offered and must be named explicitly; there is no default
that could change between releases.

``"nearest_rank"``
    The smallest order statistic ``x_(k)`` with ``k = ceil(p * n)``, so the
    reported value is always an observed sample. This is the definition in
    ISO 5725-style reporting and the one most latency tooling uses.

``"linear"``
    Linear interpolation between order statistics, ``h = (n - 1) p + 1``,
    i.e. type 7 of Hyndman & Fan (1996), *American Statistician* 50(4):
    361-365, which is ``numpy.quantile``'s default.

Sampling uncertainty
--------------------
For a continuous distribution with density ``f`` positive at the quantile
``q_p``, the sample quantile is asymptotically normal,

    sqrt(n) (q_hat_p - q_p)  ->  N(0, p (1 - p) / f(q_p)^2)

so its standard error is

    SE(q_hat_p) = sqrt(p (1 - p) / n) / f(q_p)

and the root-mean-square error of a tail estimate therefore decays as
``n^(-1/2)``: the log-log slope of RMSE against sample count is ``-1/2``.
Sources: Mosteller (1946), *Annals of Mathematical Statistics* 17(4):
377-408; David & Nagaraja (2003), *Order Statistics*, 3rd ed., Sec. 10.2.

Validity range: the result is asymptotic in ``n`` and requires
``n p (1 - p)`` to be large enough that the quantile is interior to the
sample. For ``p = 0.999`` that needs ``n`` in the thousands; below about
``n = 1/(1 - p)`` the estimator is the sample maximum and the asymptotic
expression does not apply. The helpers below report that condition rather
than hide it.

Units: latencies in seconds; ``p`` dimensionless in ``(0, 1)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

QUANTILE_METHODS = ("nearest_rank", "linear")


def quantile(samples: np.ndarray, p: float, method: str = "linear") -> float:
    """Sample quantile at probability ``p`` under the named definition.

    Parameters
    ----------
    samples:
        One-dimensional sample, seconds. Must be non-empty.
    p:
        Probability in ``(0, 1)``.
    method:
        One of :data:`QUANTILE_METHODS`.
    """
    arr = np.asarray(samples, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("samples must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("samples must be finite")
    if not (0.0 < float(p) < 1.0):
        raise ValueError(f"p must lie strictly in (0, 1), got {p!r}")
    if method not in QUANTILE_METHODS:
        raise ValueError(f"method must be one of {QUANTILE_METHODS}, got {method!r}")
    if method == "linear":
        return float(np.quantile(arr, float(p), method="linear"))
    rank = math.ceil(float(p) * arr.size)
    rank = min(max(rank, 1), arr.size)
    return float(np.partition(arr, rank - 1)[rank - 1])


def quantile_min_samples(p: float) -> int:
    """Smallest ``n`` for which ``p`` is not the sample maximum: ``ceil(1/(1-p))``.

    At ``n`` below this the nearest-rank quantile is the largest observation
    and the asymptotic normal theory above does not apply.
    """
    if not (0.0 < float(p) < 1.0):
        raise ValueError(f"p must lie strictly in (0, 1), got {p!r}")
    return int(math.ceil(1.0 / (1.0 - float(p))))


def lognormal_quantile(mu: float, sigma: float, p: float) -> float:
    """Exact lognormal quantile ``exp(mu + sigma * Phi^-1(p))``, seconds."""
    if not (sigma > 0.0) or not math.isfinite(sigma):
        raise ValueError(f"sigma must be finite and > 0, got {sigma!r}")
    if not (0.0 < float(p) < 1.0):
        raise ValueError(f"p must lie strictly in (0, 1), got {p!r}")
    return float(math.exp(mu + sigma * stats.norm.ppf(float(p))))


def lognormal_quantile_se(mu: float, sigma: float, p: float, n: int) -> float:
    """Analytic order-statistic standard error of a lognormal sample quantile, s.

    Uses ``SE = sqrt(p (1 - p) / n) / f(q_p)`` with the lognormal density at
    its own quantile, ``f(q_p) = phi(z_p) / (sigma q_p)``, giving

        SE = sqrt(p (1 - p) / n) * sigma * q_p / phi(z_p)

    where ``z_p = Phi^-1(p)``. Asymptotic in ``n``; see the module docstring
    for the validity condition.
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"n must be an integer >= 1, got {n!r}")
    p = float(p)
    q = lognormal_quantile(mu, sigma, p)
    z = stats.norm.ppf(p)
    dens_z = float(stats.norm.pdf(z))
    return float(math.sqrt(p * (1.0 - p) / n) * sigma * q / dens_z)


@dataclass(frozen=True)
class ConvergenceResult:
    """Measured convergence of a tail-quantile estimator against theory.

    Attributes
    ----------
    p:
        Tail probability estimated.
    n_grid:
        Sample counts used.
    rmse_s:
        Root-mean-square error against the exact quantile, seconds, one per
        ``n``.
    analytic_se_s:
        Analytic order-statistic standard error, seconds, one per ``n``.
    fitted_slope:
        Least-squares slope of ``log(rmse)`` against ``log(n)``.
    predicted_slope:
        ``-0.5``, from the asymptotic normality of sample quantiles.
    ratio_measured_over_analytic:
        ``rmse_s / analytic_se_s``, which should approach 1 as ``n`` grows.
    n_repeats:
        Independent repeats averaged at each ``n``.
    """

    p: float
    n_grid: tuple[int, ...]
    rmse_s: tuple[float, ...]
    analytic_se_s: tuple[float, ...]
    fitted_slope: float
    predicted_slope: float
    ratio_measured_over_analytic: tuple[float, ...]
    n_repeats: int


def lognormal_tail_convergence(
    mu: float,
    sigma: float,
    p: float,
    n_grid: tuple[int, ...] | list[int],
    n_repeats: int,
    seed: int,
    method: str = "linear",
) -> ConvergenceResult:
    """Measure RMSE of a lognormal tail-quantile estimator against sample count.

    The exact quantile is known in closed form, so the error is an error and
    not a difference between two estimates. Compare ``fitted_slope`` against
    ``predicted_slope = -0.5``.
    """
    n_grid = tuple(int(n) for n in n_grid)
    if not n_grid:
        raise ValueError("n_grid must be non-empty")
    if any(n < 1 for n in n_grid):
        raise ValueError("every entry of n_grid must be >= 1")
    if not isinstance(n_repeats, (int, np.integer)) or n_repeats < 2:
        raise ValueError(f"n_repeats must be an integer >= 2, got {n_repeats!r}")
    truth = lognormal_quantile(mu, sigma, p)
    rng = np.random.default_rng(seed)
    rmse: list[float] = []
    analytic: list[float] = []
    for n in n_grid:
        errs = np.empty(int(n_repeats), dtype=float)
        for r in range(int(n_repeats)):
            draw = np.exp(mu + sigma * rng.standard_normal(n))
            errs[r] = quantile(draw, p, method=method) - truth
        rmse.append(float(math.sqrt(float(np.mean(errs**2)))))
        analytic.append(lognormal_quantile_se(mu, sigma, p, n))
    logs_n = np.log(np.asarray(n_grid, dtype=float))
    logs_e = np.log(np.asarray(rmse, dtype=float))
    slope = float(np.polyfit(logs_n, logs_e, 1)[0])
    ratio = tuple(float(a / b) for a, b in zip(rmse, analytic, strict=True))
    return ConvergenceResult(
        p=float(p),
        n_grid=n_grid,
        rmse_s=tuple(rmse),
        analytic_se_s=tuple(analytic),
        fitted_slope=slope,
        predicted_slope=-0.5,
        ratio_measured_over_analytic=ratio,
        n_repeats=int(n_repeats),
    )
