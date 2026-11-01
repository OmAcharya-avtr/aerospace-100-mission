"""The top-level decomposition: a budget, a rank verdict, and two robust fits.

:func:`decompose` is the whole product in one call. It returns a
:class:`PointingBudget`, and the order of that object's fields is the order a
reader needs:

1. **What the geometry cannot determine.** The identifiability verdict comes
   first, before any coefficient, because a per-term number read without it is
   the specific failure this product exists to prevent.
2. **The per-term budget**, with a confidence interval and a variance
   inflation factor on every term, and the terms the rank analysis refused
   marked as refused rather than quietly printed.
3. **The identifiable linear combinations**, which are what the run measured
   when the per-term split is not available.
4. **The robust fits and their disagreement with least squares**, as a
   diagnostic, with neither preferred.

A note on what the budget is a budget of. ``contribution_rms_arcsec`` for term
``j`` is the root mean square over the observed rows of that column's own
contribution, ``rms_i( A_ij * theta_j )``. It is the term's share of the
residual field **over the geometry that was observed** and nowhere else: the
same coefficient over a different schedule gives a different contribution, and
for an unidentifiable term the split of the contribution between the terms
involved is as arbitrary as the split of the coefficients. The contributions do
not sum to the total in quadrature unless the columns are orthogonal, which
they are not; the sum of squares of contributions divided by the squared total
is reported as ``quadrature_closure`` so that the departure is visible rather
than assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .identifiability import (
    DEFAULT_MARGINAL_THRESHOLD,
    DEFAULT_NULL_THRESHOLD,
    IdentifiabilityReport,
    analyse_fit,
)
from .robust import Disagreement, RobustFit, disagreement, huber_fit, ransac_fit
from .terms import N_TERMS, TERM_NAMES
from .wls import WlsFit, fit

__all__ = ["PointingBudget", "TermBudget", "decompose"]


@dataclass(frozen=True)
class TermBudget:
    """One line of the error budget.

    Attributes
    ----------
    name
        Term name.
    coefficient
        Least-squares estimate, arcseconds.
    standard_error
        Its standard error, arcseconds.
    interval
        ``(lower, upper)`` at the fit's confidence level, arcseconds.
    vif
        Variance inflation factor, dimensionless.
    contribution_rms_arcsec
        Root mean square of this column's contribution over the observed rows,
        arcseconds. See the module docstring on what this is and is not.
    separable
        False when the term participates in a null direction, in which case
        ``coefficient``, ``standard_error`` and ``interval`` are reported for
        completeness and must not be read as a measurement of that term.
    """

    name: str
    coefficient: float
    standard_error: float
    interval: tuple[float, float]
    vif: float
    contribution_rms_arcsec: float
    separable: bool


@dataclass(frozen=True)
class PointingBudget:
    """The result of :func:`decompose`.

    Attributes
    ----------
    identifiability
        The rank verdict. Read this first.
    least_squares
        The weighted least squares fit.
    terms
        Nine :class:`TermBudget` rows in design-matrix order.
    huber, ransac
        The two robust fits, or ``None`` if they were not requested.
    huber_disagreement, ransac_disagreement
        Each robust fit against least squares, or ``None``.
    total_rms_arcsec
        Root mean square of the observed residual vector, arcseconds.
    fit_rms_arcsec
        Root mean square of the least-squares residuals, arcseconds.
    explained_fraction
        ``1 - (fit_rms / total_rms)^2``, dimensionless. Negative values are
        possible in principle and would be reported.
    quadrature_closure
        ``sum_j contribution_j^2 / total_rms^2``, dimensionless. Equals 1 only
        for an orthogonal design; the measured departure is the correlation
        between terms made visible.
    sigma_arcsec
        The declared per-row sigma used.
    """

    identifiability: IdentifiabilityReport
    least_squares: WlsFit
    terms: tuple[TermBudget, ...]
    huber: RobustFit | None
    ransac: RobustFit | None
    huber_disagreement: Disagreement | None
    ransac_disagreement: Disagreement | None
    total_rms_arcsec: float
    fit_rms_arcsec: float
    explained_fraction: float
    quadrature_closure: float
    sigma_arcsec: float

    @property
    def refused_terms(self) -> tuple[str, ...]:
        """Terms the rank analysis refuses to split, in design-matrix order."""
        refused = set(self.identifiability.unidentifiable_terms)
        return tuple(name for name in TERM_NAMES if name in refused)


def decompose(
    design: np.ndarray,
    observed: np.ndarray,
    sigma_arcsec: float | np.ndarray = 2.0,
    confidence_level: float = 0.95,
    null_threshold: float = DEFAULT_NULL_THRESHOLD,
    marginal_threshold: float = DEFAULT_MARGINAL_THRESHOLD,
    robust: bool = True,
    ransac_seed: int = 20261101,
    ransac_trials: int = 200,
    error_model: str = "scaled",
) -> PointingBudget:
    """Decompose a pointing error budget from measured residuals.

    Parameters
    ----------
    design
        ``(2n, 9)`` stacked design matrix from
        :func:`pointmodel.terms.design_matrix` and
        :func:`pointmodel.terms.stack_residuals`.
    observed
        Length-``2n`` stacked residual vector, arcseconds: cross-elevation
        block above elevation block.
    sigma_arcsec
        Declared per-row standard deviation, arcseconds. Scalar or length-``2n``.
    confidence_level
        Two-sided coverage for every reported interval.
    null_threshold, marginal_threshold
        Declared singular-value ratios; see :mod:`pointmodel.identifiability`.
    robust
        Run the Huber and RANSAC fits and the disagreement diagnostics.
    ransac_seed, ransac_trials
        Passed through to :func:`pointmodel.robust.ransac_fit`.
    error_model
        ``"scaled"`` or ``"declared"``; see :mod:`pointmodel.wls`.

    Returns
    -------
    PointingBudget

    Raises
    ------
    ValueError
        On any invalid input; the messages name the offending quantity and its
        value.
    """
    matrix = np.asarray(design, dtype=float)
    values = np.asarray(observed, dtype=float)
    least_squares = fit(
        matrix,
        values,
        sigma_arcsec,
        confidence_level=confidence_level,
        error_model=error_model,
    )
    report = analyse_fit(matrix, least_squares, null_threshold, marginal_threshold)
    refused = set(report.unidentifiable_terms)

    contributions = np.sqrt(
        np.mean((matrix * least_squares.coefficients) ** 2, axis=0)
    )
    rows: list[TermBudget] = []
    for index, name in enumerate(TERM_NAMES):
        rows.append(
            TermBudget(
                name=name,
                coefficient=float(least_squares.coefficients[index]),
                standard_error=float(least_squares.standard_errors[index]),
                interval=(
                    float(least_squares.confidence_intervals[index, 0]),
                    float(least_squares.confidence_intervals[index, 1]),
                ),
                vif=float(least_squares.vif[index]),
                contribution_rms_arcsec=float(contributions[index]),
                separable=name not in refused,
            )
        )

    total_rms = float(np.sqrt(np.mean(values**2)))
    fit_rms = float(least_squares.rms_arcsec)
    explained = 1.0 - (fit_rms / total_rms) ** 2 if total_rms > 0.0 else float("nan")
    closure = (
        float(np.sum(contributions**2) / total_rms**2) if total_rms > 0.0 else float("nan")
    )

    huber_result: RobustFit | None = None
    ransac_result: RobustFit | None = None
    huber_delta: Disagreement | None = None
    ransac_delta: Disagreement | None = None
    if robust:
        reference = float(np.median(np.broadcast_to(np.asarray(sigma_arcsec, float), values.shape)))
        huber_result = huber_fit(matrix, values, sigma_arcsec)
        huber_delta = disagreement(
            matrix,
            least_squares,
            huber_result,
            sigma_reference=reference,
            unidentifiable=report.unidentifiable_terms,
        )
        ransac_result = ransac_fit(
            matrix, values, sigma_arcsec, seed=ransac_seed, trials=ransac_trials
        )
        ransac_delta = disagreement(
            matrix,
            least_squares,
            ransac_result,
            sigma_reference=reference,
            unidentifiable=report.unidentifiable_terms,
        )

    if len(rows) != N_TERMS:  # pragma: no cover - structural invariant
        raise AssertionError(f"budget has {len(rows)} rows, expected {N_TERMS}")
    return PointingBudget(
        identifiability=report,
        least_squares=least_squares,
        terms=tuple(rows),
        huber=huber_result,
        ransac=ransac_result,
        huber_disagreement=huber_delta,
        ransac_disagreement=ransac_delta,
        total_rms_arcsec=total_rms,
        fit_rms_arcsec=fit_rms,
        explained_fraction=float(explained),
        quadrature_closure=float(closure),
        sigma_arcsec=float(
            np.median(np.broadcast_to(np.asarray(sigma_arcsec, float), values.shape))
        ),
    )
