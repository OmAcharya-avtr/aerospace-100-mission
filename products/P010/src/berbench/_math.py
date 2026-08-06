"""Numerical primitives: Gaussian Q-function and Gauss-Hermite quadrature.

References
----------
- J. G. Proakis and M. Salehi, "Digital Communications", 5th ed., McGraw-Hill,
  2008 (Q-function definition, Sec. 2.3).
- M. Abramowitz and I. A. Stegun, "Handbook of Mathematical Functions", Dover,
  1972, Sec. 25.4.46 (Gauss-Hermite quadrature).

All quantities are dimensionless unless stated otherwise.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import log_ndtr, ndtr, ndtri

__all__ = ["qfunc", "log_qfunc", "log10_qfunc", "gauss_hermite", "wilson_interval"]

_LN10 = float(np.log(10.0))


def qfunc(x: np.ndarray | float) -> np.ndarray | float:
    """Gaussian tail probability Q(x) = P(N(0,1) > x) = 1 - Phi(x).

    Computed as ``ndtr(-x)`` which is numerically stable for large positive x
    (underflows gracefully to 0.0 around x ~ 38 without overflow or NaN).

    Parameters
    ----------
    x : float or ndarray
        Argument (dimensionless).

    Source: Proakis & Salehi 2008, Sec. 2.3-2. Valid for all real x.
    """
    return ndtr(-np.asarray(x, dtype=float)) if np.ndim(x) else float(ndtr(-x))


def log_qfunc(x: np.ndarray | float) -> np.ndarray | float:
    """Natural logarithm of the Gaussian Q-function, ln Q(x).

    Uses ``scipy.special.log_ndtr`` for log-domain stability: finite for
    arguments far beyond the x ~ 38 underflow point of Q(x) itself
    (asymptotically ln Q(x) ~ -x^2/2 - ln(x sqrt(2 pi))).
    """
    out = log_ndtr(-np.asarray(x, dtype=float))
    return out if np.ndim(x) else float(out)


def log10_qfunc(x: np.ndarray | float) -> np.ndarray | float:
    """Base-10 logarithm of the Gaussian Q-function, log10 Q(x)."""
    return log_qfunc(x) / _LN10


@lru_cache(maxsize=8)
def gauss_hermite(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (nodes, weights) of n-point Gauss-Hermite quadrature.

    Approximates integral over R of f(x) exp(-x^2) dx ~ sum_i w_i f(x_i)
    (Abramowitz & Stegun 1972, 25.4.46). Cached because node computation is
    O(n^2) and reused across calls.
    """
    if not isinstance(n_nodes, int) or n_nodes < 2:
        raise ValueError(f"n_nodes must be an integer >= 2, got {n_nodes!r}")
    if n_nodes > 256:
        raise ValueError(
            f"n_nodes must be <= 256 (numpy hermgauss weights overflow beyond ~300 "
            f"nodes), got {n_nodes}"
        )
    x, w = np.polynomial.hermite.hermgauss(n_nodes)
    return x, w


def wilson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion k/n.

    Source: E. B. Wilson, "Probable inference, the law of succession, and
    statistical inference", J. Amer. Statist. Assoc. 22, 209-212 (1927).
    Preferred over the normal (Wald) approximation for small counts and
    proportions near 0, which is exactly the BER-estimation regime.

    Parameters
    ----------
    k : int
        Number of observed errors (successes), 0 <= k <= n.
    n : int
        Number of trials, n >= 1.
    level : float
        Two-sided confidence level in (0, 1), default 0.95.

    Returns
    -------
    (low, high) : tuple of float
        Confidence bounds on the true error probability.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"k must satisfy 0 <= k <= n, got k={k}, n={n}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    z = float(ndtri(0.5 + level / 2.0))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * float(np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)))
    low = 0.0 if k == 0 else max(0.0, centre - half)  # exact endpoint, no fp residue
    high = 1.0 if k == n else min(1.0, centre + half)
    return low, high
