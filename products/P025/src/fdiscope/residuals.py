"""Residual conditioning and filter-consistency checks.

The detectors in this package are only valid if the normalised residual really
is ``N(0, I)`` and white under the fault-free hypothesis.  That is an
assumption about the *filter*, not about the fault, and it is checked here
explicitly rather than assumed.

Two checks are provided:

* :func:`nis_consistency` -- is the mean normalised innovation squared equal to
  the measurement dimension?  For ``N`` independent samples the sample mean of
  a chi-squared with ``m`` dof has mean ``m`` and variance ``2m/N``, giving the
  standard two-sided acceptance region (Bar-Shalom, Rong Li & Kirubarajan
  2001, sec. 5.4.2, the "time-average NIS test").
* :func:`whiteness` -- the lag-``k`` autocorrelation of each residual channel.
  For a white sequence of length ``N`` each sample autocorrelation is
  approximately ``N(0, 1/N)``, so ``|rho| > 1.96/sqrt(N)`` at the 5 % level is
  evidence against whiteness (Box, Jenkins & Reinsel, *Time Series Analysis*,
  4th ed., Wiley, 2008, sec. 8.2).

Both return the measured numbers and the acceptance bounds, not a verdict, so
the caller decides and the number ends up in the report either way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2

__all__ = ["normalise", "nis_from_residual", "NisCheck", "nis_consistency", "whiteness"]


def normalise(innovation: ArrayLike, innovation_cov: ArrayLike) -> NDArray[np.float64]:
    """Whiten an innovation sequence: ``r = L^-1 y`` with ``S = L L^T``.

    Parameters
    ----------
    innovation : array_like, shape (N, m)
        Raw innovations in measurement units.
    innovation_cov : array_like, shape (m, m)
        Innovation covariance ``S``, symmetric positive definite.

    Returns
    -------
    ndarray, shape (N, m)
        Dimensionless normalised residuals.

    Raises
    ------
    numpy.linalg.LinAlgError
        If ``S`` is not positive definite.
    """
    y = np.atleast_2d(np.asarray(innovation, dtype=float))
    s = np.atleast_2d(np.asarray(innovation_cov, dtype=float))
    if y.ndim != 2:
        raise ValueError(f"innovation must be (N, m), got shape {y.shape}")
    if s.shape != (y.shape[1], y.shape[1]):
        raise ValueError(f"innovation_cov must be ({y.shape[1]}, {y.shape[1]}), got {s.shape}")
    chol = np.linalg.cholesky(0.5 * (s + s.T))
    return np.linalg.solve(chol, y.T).T


def nis_from_residual(residual: ArrayLike) -> NDArray[np.float64]:
    """``|r_k|^2`` per sample, shape ``(N,)``, dimensionless."""
    r = np.atleast_2d(np.asarray(residual, dtype=float))
    if r.ndim != 2:
        raise ValueError(f"residual must be (N, m), got shape {r.shape}")
    return np.sum(r * r, axis=1)


@dataclass(frozen=True)
class NisCheck:
    """Result of the time-average NIS consistency test.

    Attributes
    ----------
    mean_nis : float
        Sample mean of ``|r_k|^2``.
    expected : float
        ``m``, the measurement dimension.
    low, high : float
        Two-sided acceptance bounds on the sample mean at ``level``.
    level : float
        Confidence level of the bounds.
    n_samples : int
        Sample count used.
    """

    mean_nis: float
    expected: float
    low: float
    high: float
    level: float
    n_samples: int

    @property
    def consistent(self) -> bool:
        """True when the sample mean lies inside the acceptance region."""
        return bool(self.low <= self.mean_nis <= self.high)


def nis_consistency(residual: ArrayLike, level: float = 0.95) -> NisCheck:
    """Time-average NIS test on a normalised residual sequence.

    The acceptance region on ``N * mean(NIS)`` is the central ``level``
    interval of a chi-squared with ``N m`` degrees of freedom; dividing by
    ``N`` gives the bounds on the sample mean.

    Parameters
    ----------
    residual : array_like, shape (N, m)
        Normalised residuals, ``N >= 2``.
    level : float
        Two-sided confidence, in ``(0, 1)``.

    Returns
    -------
    NisCheck
    """
    r = np.atleast_2d(np.asarray(residual, dtype=float))
    if r.ndim != 2 or r.shape[0] < 2:
        raise ValueError(f"residual must be (N, m) with N >= 2, got shape {r.shape}")
    lv = float(level)
    if not (0.0 < lv < 1.0):
        raise ValueError(f"level must lie in (0, 1), got {level}")
    n, m = r.shape
    dof = n * m
    tail = 0.5 * (1.0 - lv)
    return NisCheck(
        mean_nis=float(np.mean(nis_from_residual(r))),
        expected=float(m),
        low=float(chi2.ppf(tail, dof) / n),
        high=float(chi2.isf(tail, dof) / n),
        level=lv,
        n_samples=int(n),
    )


def whiteness(residual: ArrayLike, max_lag: int = 5) -> tuple[NDArray[np.float64], float]:
    """Sample autocorrelation per channel and the 5 % whiteness bound.

    Parameters
    ----------
    residual : array_like, shape (N, m)
        Normalised residuals.
    max_lag : int
        Largest lag to report, ``>= 1``.

    Returns
    -------
    (rho, bound) : tuple
        ``rho`` has shape ``(max_lag, m)`` with ``rho[k - 1, c]`` the lag-``k``
        autocorrelation of channel ``c``; ``bound = 1.96 / sqrt(N)`` is the
        two-sided 5 % significance level for a white sequence.
    """
    r = np.atleast_2d(np.asarray(residual, dtype=float))
    lag = int(max_lag)
    if lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag}")
    if r.ndim != 2 or r.shape[0] <= lag + 1:
        raise ValueError(f"residual must be (N, m) with N > max_lag + 1, got shape {r.shape}")
    n, m = r.shape
    centred = r - r.mean(axis=0)
    denom = np.sum(centred * centred, axis=0)
    rho = np.zeros((lag, m))
    for k in range(1, lag + 1):
        rho[k - 1] = np.sum(centred[:-k] * centred[k:], axis=0) / denom
    return rho, float(1.96 / np.sqrt(n))
