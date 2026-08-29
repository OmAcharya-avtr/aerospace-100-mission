r"""Filter consistency diagnostics — NEES, NIS and chi-squared acceptance bounds.

This is the module the rest of NavBench exists to serve. RMS error alone cannot
tell you whether a filter is *credible*: a filter can have small error and lie
about its uncertainty, or large error and be perfectly honest about it. The
standard tests are:

**NEES** — normalised estimation error squared, per step ``k`` and run ``i``:

.. math:: \varepsilon^{i}_k = (x_k - \hat x^{i}_k)^{\mathsf T}
          (P^{i}_k)^{-1} (x_k - \hat x^{i}_k)

For a consistent filter with ``n``-dimensional state, ``ε ∼ χ²_n``, so
``E[ε] = n``.

**NIS** — normalised innovation squared:

.. math:: \eta^{i}_k = \nu^{{i}\mathsf T}_k (S^{i}_k)^{-1} \nu^{i}_k
          \ \sim\ \chi^2_m

NIS is computable **without truth**, which is why it is the diagnostic that
survives contact with real flight data; NEES needs truth and therefore lives in
simulation.

**Acceptance region (per step).** Averaging over ``M`` independent Monte Carlo
runs, ``M ε̄_k ∼ χ²_{Mn}``, so the two-sided ``1−α`` acceptance region for
``ε̄_k`` is

.. math:: \left[\frac{\chi^2_{Mn}(\alpha/2)}{M},\ \frac{\chi^2_{Mn}(1-\alpha/2)}{M}\right]

The same construction with ``m`` and ``η̄`` gives the NIS region. This test is
**exact at each time step**, because the runs really are independent, and it is
the test plotted by :mod:`navbench` examples. For a consistent filter the
fraction of steps falling inside should be ≈ ``1 − α``.

**Overall verdict — and why it is not a chi-squared test.** The obvious next
step, pooling all ``M·K`` samples and using a ``χ²_{MKn}`` region, is *wrong*:
NEES values at neighbouring time steps come from the same trajectory and are
strongly correlated, so the pooled chi-squared interval is far too narrow and
rejects consistent filters. NavBench therefore forms the **per-run time
average** ``ε̄^{(i)} = (1/K)Σ_k ε^{(i)}_k``, which *are* i.i.d. across the ``M``
independent runs, and puts a Student-``t`` confidence interval on their mean.
The filter passes when the nominal value (``n`` for NEES, ``m`` for NIS) lies
inside that interval. The pooled chi-squared region is still reported, marked
as optimistic, so the size of the effect is visible.

**Whiteness.** For a consistent filter the innovation sequence is zero-mean and
white. The normalised sample autocorrelation at lag ``j``

.. math:: \hat\rho(j) = \frac{\sum_k \nu_k^{\mathsf T}\nu_{k+j}}
          {\sqrt{\sum_k \nu_k^{\mathsf T}\nu_k \sum_k \nu_{k+j}^{\mathsf T}\nu_{k+j}}}

is asymptotically ``N(0, 1/K)``, giving a ``±1.96/√K`` band.

Source for all of the above: Bar-Shalom, Y., Rong Li, X. and Kirubarajan, T.
(2001), *Estimation with Applications to Tracking and Navigation*, Wiley, §5.4
("Consistency of state estimators"), where the NEES/NIS chi-squared tests and
the whiteness test are set out. See also Bar-Shalom & Li (1993), *Estimation
and Tracking: Principles, Techniques and Software*, Artech House, Ch. 5.

Statistical power — read this before quoting a result
-----------------------------------------------------
The width of the acceptance region shrinks like ``1/√M``. With ``n = 4`` and
``M = 50`` runs the 95 % region for ``ε̄`` is roughly ``[3.3, 4.8]``; with
``M = 500`` it is roughly ``[3.7, 4.3]``. A filter whose true ANEES is 1.1×
nominal will therefore pass at ``M = 50`` and fail at ``M = 500``. NavBench
always reports ``M``, the bounds, and the fraction of steps inside them, so the
power of the test is visible rather than implied.

Units: NEES and NIS are dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2

from .linalg import mahalanobis_sq

__all__ = [
    "nees_series",
    "nis_series",
    "chi2_average_bounds",
    "ConsistencyReport",
    "assess",
    "innovation_autocorrelation",
    "whiteness_band",
]


def nees_series(errors: ArrayLike, covariances: ArrayLike) -> NDArray[np.float64]:
    """Per-step NEES.

    Parameters
    ----------
    errors : array_like, shape (K, n)
        ``x_k − x̂_k``.
    covariances : array_like, shape (K, n, n)
        Filter covariances ``P_k``.

    Returns
    -------
    ndarray, shape (K,)
    """
    e = np.atleast_2d(np.asarray(errors, dtype=float))
    p = np.asarray(covariances, dtype=float)
    if p.ndim != 3 or p.shape[0] != e.shape[0] or p.shape[1] != e.shape[1]:
        raise ValueError(f"covariances shape {p.shape} incompatible with errors shape {e.shape}")
    return np.array([mahalanobis_sq(e[k], p[k]) for k in range(e.shape[0])])


def nis_series(innovations: ArrayLike, innovation_covs: ArrayLike) -> NDArray[np.float64]:
    """Per-step NIS. Same shape contract as :func:`nees_series`."""
    return nees_series(innovations, innovation_covs)


def chi2_average_bounds(dof: int, n_average: int, alpha: float = 0.05) -> tuple[float, float]:
    r"""Two-sided acceptance region for a ``M``-sample average of ``χ²_dof``.

    Returns ``(lo, hi)`` such that ``P(lo <= ε̄ <= hi) = 1 − α`` when the
    individual samples are independent ``χ²_dof``.

    Parameters
    ----------
    dof : int
        Degrees of freedom of one sample (state or measurement dimension).
    n_average : int
        Number of independent samples averaged (Monte Carlo runs, or time steps
        for the time-averaged test).
    alpha : float
        Total tail probability, split evenly between the two tails.
    """
    if dof < 1 or n_average < 1:
        raise ValueError(f"dof and n_average must be >= 1, got {dof}, {n_average}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    total = dof * n_average
    lo = float(chi2.ppf(alpha / 2.0, total) / n_average)
    hi = float(chi2.ppf(1.0 - alpha / 2.0, total) / n_average)
    return lo, hi


@dataclass(frozen=True)
class ConsistencyReport:
    """Result of a NEES or NIS consistency assessment.

    Attributes
    ----------
    label : str
        ``"NEES"`` or ``"NIS"`` plus a free-text tag.
    dof : int
        Degrees of freedom per sample (nominal mean).
    n_runs : int
        Number of independent Monte Carlo runs.
    n_steps : int
        Number of time steps assessed.
    mean : float
        Grand mean over runs and steps.
    normalised_mean : float
        ``mean / dof`` — the "ANEES"/"ANIS". 1.0 is nominal; > 1 means the
        filter is **optimistic** (covariance too small), < 1 **conservative**.
    step_bounds : tuple of float
        Per-step chi-squared acceptance region for the run-average (exact).
    fraction_inside : float
        Fraction of time steps whose run-average lies inside ``step_bounds``.
        Nominally ``1 − α``.
    mean_ci : tuple of float
        Student-``t`` confidence interval on the grand mean, built from the
        ``M`` i.i.d. per-run time averages. This is the interval used for the
        verdict. ``(nan, nan)`` when ``M < 2``.
    pooled_chi2_bounds : tuple of float
        Chi-squared region for ``M·K`` *independent* samples. Reported for
        reference only — it is optimistically narrow because NEES is
        time-correlated (see the module docstring).
    passed : bool
        True when ``dof`` lies inside ``mean_ci`` (``M >= 2``), or inside
        ``pooled_chi2_bounds`` when only one run is available.
    """

    label: str
    dof: int
    n_runs: int
    n_steps: int
    mean: float
    normalised_mean: float
    step_bounds: tuple[float, float]
    fraction_inside: float
    mean_ci: tuple[float, float]
    pooled_chi2_bounds: tuple[float, float]
    passed: bool

    def summary(self) -> str:
        """One-line human-readable summary (no printing; caller decides)."""
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{self.label}: mean={self.mean:.4f} (dof={self.dof}, "
            f"normalised={self.normalised_mean:.4f}); 95% CI on the mean "
            f"[{self.mean_ci[0]:.4f}, {self.mean_ci[1]:.4f}] -> {verdict}; "
            f"per-step chi2 region [{self.step_bounds[0]:.4f}, {self.step_bounds[1]:.4f}] "
            f"contains {self.fraction_inside * 100:.1f}% of {self.n_steps} steps "
            f"(M={self.n_runs})"
        )


def assess(
    samples: ArrayLike, dof: int, alpha: float = 0.05, label: str = "NEES"
) -> ConsistencyReport:
    """Assess a NEES/NIS sample array against its chi-squared bounds.

    Parameters
    ----------
    samples : array_like, shape (M, K) or (K,)
        NEES or NIS values; rows are independent Monte Carlo runs, columns are
        time steps. A 1-D input is treated as a single run.
    dof : int
        Degrees of freedom per sample.
    alpha : float
        Significance level for the two-sided tests.
    label : str
        Tag carried into the report.

    Notes
    -----
    The verdict uses the Student-``t`` interval on the ``M`` i.i.d. per-run time
    averages; see the module docstring for why the pooled chi-squared interval
    is not used for that purpose.
    """
    from scipy.stats import t as student_t

    s = np.atleast_2d(np.asarray(samples, dtype=float))
    if not np.all(np.isfinite(s)):
        raise ValueError("samples contain non-finite values — the filter diverged or P is singular")
    m, k = s.shape
    step_avg = s.mean(axis=0)
    step_lo, step_hi = chi2_average_bounds(dof, m, alpha)
    pooled_lo, pooled_hi = chi2_average_bounds(dof, m * k, alpha)
    grand = float(s.mean())
    inside = float(np.mean((step_avg >= step_lo) & (step_avg <= step_hi)))
    if m >= 2:
        per_run = s.mean(axis=1)
        se = float(np.std(per_run, ddof=1) / np.sqrt(m))
        half = float(student_t.ppf(1.0 - alpha / 2.0, m - 1)) * se
        ci = (grand - half, grand + half)
        passed = bool(ci[0] <= dof <= ci[1])
    else:
        ci = (float("nan"), float("nan"))
        passed = bool(pooled_lo <= grand <= pooled_hi)
    return ConsistencyReport(
        label=label,
        dof=int(dof),
        n_runs=int(m),
        n_steps=int(k),
        mean=grand,
        normalised_mean=grand / float(dof),
        step_bounds=(step_lo, step_hi),
        fraction_inside=inside,
        mean_ci=ci,
        pooled_chi2_bounds=(pooled_lo, pooled_hi),
        passed=passed,
    )


def innovation_autocorrelation(innovations: ArrayLike, max_lag: int = 10) -> NDArray[np.float64]:
    r"""Normalised innovation autocorrelation ``ρ̂(j)`` for ``j = 1 … max_lag``.

    Parameters
    ----------
    innovations : array_like, shape (K, m)
        Innovation sequence from one run.
    max_lag : int
        Largest lag.

    Returns
    -------
    ndarray, shape (max_lag,)
        ``ρ̂(1) … ρ̂(max_lag)``.
    """
    v = np.atleast_2d(np.asarray(innovations, dtype=float))
    if v.shape[0] < max_lag + 2:
        raise ValueError(f"need at least max_lag+2={max_lag + 2} samples, got {v.shape[0]}")
    out = np.zeros(max_lag)
    for j in range(1, max_lag + 1):
        a, b = v[:-j], v[j:]
        num = float(np.sum(a * b))
        den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
        out[j - 1] = num / den if den > 0.0 else 0.0
    return out


def whiteness_band(n_samples: int, alpha: float = 0.05) -> float:
    r"""Half-width of the ``1−α`` whiteness band, ``z_{1−α/2}/√K``.

    Under the null hypothesis of a white innovation sequence, ``ρ̂(j)`` is
    asymptotically ``N(0, 1/K)`` (Bar-Shalom, Li & Kirubarajan 2001, §5.4).
    """
    from scipy.stats import norm

    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")
    return float(norm.ppf(1.0 - alpha / 2.0) / np.sqrt(n_samples))
