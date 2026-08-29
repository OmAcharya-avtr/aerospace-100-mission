"""Filter consistency diagnostics: NEES, NIS and chi-squared bounds.

This is the module NavBench exists for.  RMS error alone cannot tell you
whether a filter is *consistent* — whether the covariance it reports is an
honest description of the error it actually makes.  A filter can have small
RMS error and a covariance that is wrong by an order of magnitude, and it
will then fuse badly with anything downstream.

DEFINITIONS (Bar-Shalom, Y., Li, X.-R. & Kirubarajan, T. (2001), *Estimation
with Applications to Tracking and Navigation*, Wiley, §5.4)

**NEES** — normalised estimation error squared, Eq. (5.4.2-1):

    ε_k = x̃_kᵀ P_k⁻¹ x̃_k ,      x̃_k = x_k − x̂_k

For a consistent linear-Gaussian filter ``ε_k ~ χ²_n`` with ``n`` the state
dimension, so ``E[ε_k] = n``.  NEES requires the truth and is therefore a
simulation-only diagnostic.

**NIS** — normalised innovation squared, Eq. (5.4.2-2):

    ε^ν_k = ν_kᵀ S_k⁻¹ ν_k ,     ν_k = z_k − ẑ_{k|k-1}

``ε^ν_k ~ χ²_m`` with ``m`` the measurement dimension.  NIS needs no truth
and is therefore the diagnostic available on real data.

**Ensemble average and its bounds** — Eq. (5.4.2-3).  Averaging ``M``
*independent* runs, ``M ε̄ ~ χ²_{M·d}``, so the two-sided ``1−α`` acceptance
region for ``ε̄`` is

    [ χ²_{M·d}(α/2) / M ,  χ²_{M·d}(1−α/2) / M ]

Averaging over *time* within a single run uses the same formula only if the
samples are independent; they are not exactly, so the time-averaged variant
is reported here as an indicative statistic and labelled as such
(``independent=False`` widens nothing — it only changes the label).  The
Monte Carlo variant in ``validation/v2_nees_nis_consistency.py`` uses
independent runs, which is the statistically defensible form.

**Whiteness** — Eq. (5.4.3-2).  For a consistent filter the normalised
innovations are white; the sample autocorrelation at lag ``l`` satisfies
``sqrt(N) ρ̂(l) → N(0, 1)``, giving a ±1.96/√N acceptance band at 95 %.

INTERPRETATION
* ``ε̄`` above the upper bound → the filter is **optimistic** (covariance too
  small): the classic symptom of under-modelled process noise ``Q``.
* ``ε̄`` below the lower bound → the filter is **pessimistic** (covariance too
  large): ``Q`` or ``R`` inflated, estimates under-weighted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

__all__ = [
    "chi2_bounds",
    "nees",
    "nis",
    "ConsistencyResult",
    "consistency_test",
    "ensemble_consistency",
    "innovation_whiteness",
    "WhitenessResult",
]


def chi2_bounds(dof: int, n_runs: int = 1, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided acceptance region for an ``n_runs``-average of a ``χ²_dof`` statistic.

    Parameters
    ----------
    dof : int
        Degrees of freedom of one sample (state dimension for NEES,
        measurement dimension for NIS), ≥ 1.
    n_runs : int
        Number of independent samples averaged, ≥ 1.
    alpha : float
        Total two-sided significance, in ``(0, 1)``.  0.05 → 95 % region.

    Returns
    -------
    (lower, upper)
        Bounds on the **average** statistic, i.e.
        ``χ²_{M·d}(α/2)/M`` and ``χ²_{M·d}(1−α/2)/M``.

    Notes
    -----
    Bar-Shalom, Li & Kirubarajan 2001, Eq. (5.4.2-3).  For ``n_runs = 1`` this
    reduces to the single-sample ``χ²_d`` quantiles.
    """
    d = int(dof)
    m = int(n_runs)
    a = float(alpha)
    if d < 1:
        raise ValueError(f"dof must be >= 1, got {dof!r}")
    if m < 1:
        raise ValueError(f"n_runs must be >= 1, got {n_runs!r}")
    if not np.isfinite(a) or not (0.0 < a < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    total = m * d
    return (
        float(stats.chi2.ppf(a / 2.0, total) / m),
        float(stats.chi2.ppf(1.0 - a / 2.0, total) / m),
    )


def nees(errors: ArrayLike, covariances: ArrayLike) -> NDArray[np.float64]:
    """Per-sample NEES ``x̃ᵀ P⁻¹ x̃``.

    Parameters
    ----------
    errors : array_like, shape (N, n) or (n,)
        Estimation errors ``x_true − x̂``.
    covariances : array_like, shape (N, n, n) or (n, n)
        The filter's reported covariance at the same instants.

    Returns
    -------
    ndarray, shape (N,)

    Raises
    ------
    ValueError
        On shape mismatch, non-finite input, or a covariance that is not
        positive definite (Cholesky failure) — that is itself a finding, not
        a nuisance, so it is raised rather than silently regularised.
    """
    e = np.atleast_2d(np.asarray(errors, dtype=float))
    p = np.asarray(covariances, dtype=float)
    if p.ndim == 2:
        p = p[None, ...]
    if e.ndim != 2:
        raise ValueError(f"errors must be 1-D or 2-D, got shape {e.shape}")
    n = e.shape[1]
    if p.shape[0] == 1 and e.shape[0] > 1:
        p = np.repeat(p, e.shape[0], axis=0)
    if p.shape != (e.shape[0], n, n):
        raise ValueError(
            f"covariances must have shape ({e.shape[0]}, {n}, {n}), got {p.shape}"
        )
    if not np.all(np.isfinite(e)):
        raise ValueError("errors must be finite")
    if not np.all(np.isfinite(p)):
        raise ValueError("covariances must be finite")
    out = np.zeros(e.shape[0])
    for k in range(e.shape[0]):
        pk = 0.5 * (p[k] + p[k].T)
        try:
            chol = np.linalg.cholesky(pk)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"covariance at index {k} is not positive definite "
                f"(min eigenvalue {float(np.linalg.eigvalsh(pk).min()):.3e}); "
                "NEES is undefined"
            ) from exc
        y = np.linalg.solve(chol, e[k])
        out[k] = float(y @ y)
    return out


def nis(innovations: ArrayLike, innovation_covs: ArrayLike) -> NDArray[np.float64]:
    """Per-sample NIS ``νᵀ S⁻¹ ν``; rows containing NaN yield NaN.

    Parameters
    ----------
    innovations : array_like, shape (N, m) or (m,)
    innovation_covs : array_like, shape (N, m, m) or (m, m)

    Returns
    -------
    ndarray, shape (N,) — NaN where the innovation row was NaN (no measurement).
    """
    v = np.atleast_2d(np.asarray(innovations, dtype=float))
    s = np.asarray(innovation_covs, dtype=float)
    if s.ndim == 2:
        s = s[None, ...]
    if v.ndim != 2:
        raise ValueError(f"innovations must be 1-D or 2-D, got shape {v.shape}")
    m = v.shape[1]
    if s.shape[0] == 1 and v.shape[0] > 1:
        s = np.repeat(s, v.shape[0], axis=0)
    if s.shape != (v.shape[0], m, m):
        raise ValueError(
            f"innovation_covs must have shape ({v.shape[0]}, {m}, {m}), got {s.shape}"
        )
    out = np.full(v.shape[0], np.nan)
    for k in range(v.shape[0]):
        if not np.all(np.isfinite(v[k])):
            continue
        sk = 0.5 * (s[k] + s[k].T)
        if not np.all(np.isfinite(sk)):
            raise ValueError(f"innovation covariance at index {k} is not finite")
        try:
            chol = np.linalg.cholesky(sk)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"innovation covariance at index {k} is not positive definite "
                f"(min eigenvalue {float(np.linalg.eigvalsh(sk).min()):.3e})"
            ) from exc
        y = np.linalg.solve(chol, v[k])
        out[k] = float(y @ y)
    return out


@dataclass(frozen=True)
class ConsistencyResult:
    """Outcome of a chi-squared consistency test.

    Attributes
    ----------
    statistic : str
        ``"NEES"`` or ``"NIS"``.
    mean : float
        Average of the samples used.
    dof : int
        Degrees of freedom per sample.
    n_samples : int
        Number of samples averaged.
    lower, upper : float
        Acceptance bounds for the average.
    alpha : float
        Two-sided significance used.
    fraction_inside : float
        Fraction of *individual* samples inside the single-sample ``χ²_dof``
        acceptance region — a distribution-shape check that complements the
        mean test.
    independent : bool
        Whether the samples averaged were independent runs (True) or
        successive time steps of one run (False, indicative only).
    """

    statistic: str
    mean: float
    dof: int
    n_samples: int
    lower: float
    upper: float
    alpha: float
    fraction_inside: float
    independent: bool

    @property
    def passed(self) -> bool:
        """True when the average lies inside the acceptance region."""
        return bool(self.lower <= self.mean <= self.upper)

    @property
    def verdict(self) -> str:
        """``"consistent"``, ``"optimistic"`` (above) or ``"pessimistic"`` (below)."""
        if self.mean > self.upper:
            return "optimistic"
        if self.mean < self.lower:
            return "pessimistic"
        return "consistent"

    def summary(self) -> str:
        """One-line human-readable summary (no trailing newline)."""
        kind = "independent runs" if self.independent else "time samples (indicative)"
        return (
            f"{self.statistic}: mean {self.mean:.4f} vs dof {self.dof} "
            f"[{self.lower:.4f}, {self.upper:.4f}] over {self.n_samples} {kind} "
            f"-> {self.verdict.upper()} ({'PASS' if self.passed else 'FAIL'}); "
            f"{100.0 * self.fraction_inside:.1f}% of samples inside single-sample bounds"
        )


def consistency_test(
    samples: ArrayLike,
    dof: int,
    alpha: float = 0.05,
    statistic: str = "NEES",
    independent: bool = True,
) -> ConsistencyResult:
    """Chi-squared consistency test on a set of NEES or NIS samples.

    NaN samples (steps with no measurement) are dropped before averaging.

    Parameters
    ----------
    samples : array_like, shape (N,)
        NEES or NIS values.
    dof : int
        Degrees of freedom of one sample.
    alpha : float
        Two-sided significance.  Default 0.05.
    statistic : str
        Label carried into the result, ``"NEES"`` or ``"NIS"``.
    independent : bool
        Whether the samples are independent.  Recorded in the result; the
        bounds themselves assume independence either way, which is why the
        dependent case is labelled indicative.
    """
    s = np.asarray(samples, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("no finite samples to test")
    if statistic not in ("NEES", "NIS"):
        raise ValueError(f"statistic must be 'NEES' or 'NIS', got {statistic!r}")
    lo, hi = chi2_bounds(dof, s.size, alpha)
    single_lo, single_hi = chi2_bounds(dof, 1, alpha)
    inside = float(np.mean((s >= single_lo) & (s <= single_hi)))
    return ConsistencyResult(
        statistic=statistic,
        mean=float(np.mean(s)),
        dof=int(dof),
        n_samples=int(s.size),
        lower=lo,
        upper=hi,
        alpha=float(alpha),
        fraction_inside=inside,
        independent=bool(independent),
    )


def ensemble_consistency(
    per_run_samples: ArrayLike, dof: int, alpha: float = 0.05, statistic: str = "NEES"
) -> tuple[NDArray[np.float64], float, float]:
    """Time series of the ensemble-average statistic with its chi-squared bounds.

    Parameters
    ----------
    per_run_samples : array_like, shape (M, N)
        Statistic for each of ``M`` independent runs at each of ``N`` steps.
    dof : int
        Degrees of freedom of one sample.
    alpha : float
        Two-sided significance.
    statistic : str
        Unused except for input validation symmetry; kept so callers can pass
        the same label they use elsewhere.

    Returns
    -------
    (average, lower, upper)
        ``average`` has shape (N,); the bounds are scalars valid at every step.
    """
    a = np.atleast_2d(np.asarray(per_run_samples, dtype=float))
    if a.ndim != 2:
        raise ValueError(f"per_run_samples must be 2-D (M, N), got shape {a.shape}")
    if statistic not in ("NEES", "NIS"):
        raise ValueError(f"statistic must be 'NEES' or 'NIS', got {statistic!r}")
    m = a.shape[0]
    if m < 1:
        raise ValueError("per_run_samples must contain at least one run")
    with np.errstate(invalid="ignore"):
        avg = np.nanmean(a, axis=0)
    lo, hi = chi2_bounds(dof, m, alpha)
    return avg, lo, hi


@dataclass(frozen=True)
class WhitenessResult:
    """Innovation whiteness test result (Bar-Shalom et al. 2001, Eq. (5.4.3-2))."""

    lags: NDArray[np.float64]
    autocorrelation: NDArray[np.float64]
    band: float
    n_samples: int

    @property
    def passed(self) -> bool:
        """True when every tested lag ≥ 1 lies inside the ±band acceptance region."""
        return bool(np.all(np.abs(self.autocorrelation[1:]) <= self.band))

    def summary(self) -> str:
        """One-line summary of the worst lag."""
        if self.autocorrelation.size < 2:
            return f"whiteness: no lags tested (N = {self.n_samples})"
        worst = int(np.argmax(np.abs(self.autocorrelation[1:]))) + 1
        return (
            f"whiteness: max |rho| = {abs(self.autocorrelation[worst]):.4f} at lag {worst}, "
            f"band +/-{self.band:.4f} over N = {self.n_samples} "
            f"-> {'PASS' if self.passed else 'FAIL'}"
        )


def innovation_whiteness(
    innovations: ArrayLike, max_lag: int = 10, alpha: float = 0.05
) -> WhitenessResult:
    """Sample autocorrelation of scalarised innovations with the ±1.96/√N band.

    Multi-dimensional innovations are reduced by summing across components
    before correlating, which tests the dominant temporal structure.  Rows
    containing NaN (no measurement) are dropped.

    Parameters
    ----------
    innovations : array_like, shape (N,) or (N, m)
    max_lag : int
        Highest lag tested, ≥ 1 and < N.
    alpha : float
        Two-sided significance for the band, in ``(0, 1)``.
    """
    v = np.asarray(innovations, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    if v.ndim != 2:
        raise ValueError(f"innovations must be 1-D or 2-D, got shape {v.shape}")
    keep = np.all(np.isfinite(v), axis=1)
    x = np.sum(v[keep], axis=1)
    n = x.size
    lag = int(max_lag)
    if lag < 1:
        raise ValueError(f"max_lag must be >= 1, got {max_lag!r}")
    if n <= lag + 1:
        raise ValueError(f"need more than max_lag+1 = {lag + 1} finite samples, got {n}")
    if not np.isfinite(float(alpha)) or not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    x = x - np.mean(x)
    denom = float(x @ x)
    if denom <= 0.0:
        raise ValueError("innovations have zero variance; whiteness is undefined")
    rho = np.array([float(x[: n - k] @ x[k:]) / denom for k in range(lag + 1)])
    band = float(stats.norm.ppf(1.0 - float(alpha) / 2.0) / np.sqrt(n))
    return WhitenessResult(np.arange(lag + 1, dtype=float), rho, band, n)
