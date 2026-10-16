"""The declared correlation structure and the exact combination it implies.

Equation
--------
For a sum ``y = sum_i x_i`` of terms whose sensitivity coefficients are all
unity, the law of propagation of uncertainty gives

    u(y) = sqrt( sum_i sum_j u_i * R_ij * u_j )  =  sqrt( w^T R w )

with ``w`` the vector of 1 sigma uncertainties in the budget's unit and ``R``
the correlation matrix (dimensionless, symmetric, unit diagonal).

Source
    JCGM 100:2008, *Evaluation of measurement data - Guide to the expression
    of uncertainty in measurement* (the GUM), section 5.1.2 for the
    uncorrelated case and section 5.2.2 for the correlated case. Issued by
    the Joint Committee for Guides in Metrology and available from BIPM.

Units
    ``w`` carries the budget's unit; ``R`` is dimensionless; the result
    carries the budget's unit.

Assumptions
    Additive combination with unit sensitivity coefficients, and
    second-order terms negligible. Both hold exactly for a plain sum, which
    is the only combination this package implements.

Validity range
    Any real symmetric positive semi-definite ``R``. A declaration whose
    matrix is not positive semi-definite describes no joint distribution and
    is reported by check BR008 rather than silently used.

Two special cases are used throughout as the algebraic reference points:
``R = I`` reduces the formula to root-sum-square, and ``R = J`` (every entry
one) reduces it to the arithmetic sum. Since ``R = J`` is the largest
admissible matrix entrywise for non-negative ``w``, RSS never exceeds the
linear sum; that inequality is the identity property-tested in
``tests/test_properties.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .errors import AggregationError
from .model import Budget, Correlation, leaves

#: Eigenvalues below ``-PSD_TOLERANCE * n`` mark a matrix as not positive
#: semi-definite. Scaled by the matrix order because the smallest eigenvalue
#: of a numerically PSD matrix drifts with size.
PSD_TOLERANCE = 1e-9


def correlation_matrix(leaf_ids: Sequence[str], correlations: Sequence[Correlation]) -> np.ndarray:
    """Build the symmetric correlation matrix for ``leaf_ids``.

    Undeclared pairs are set to zero, which is the declaration that they are
    independent. The diagonal is 1.0.

    Returns
    -------
    numpy.ndarray
        ``(n, n)`` dimensionless float array, symmetric by construction.
    """
    index = {ident: i for i, ident in enumerate(leaf_ids)}
    n = len(leaf_ids)
    matrix = np.eye(n, dtype=float)
    for corr in correlations:
        if corr.a not in index or corr.b not in index:
            continue
        i, j = index[corr.a], index[corr.b]
        matrix[i, j] = corr.rho
        matrix[j, i] = corr.rho
    return matrix


def is_positive_semidefinite(matrix: np.ndarray) -> tuple[bool, float]:
    """Return ``(verdict, smallest_eigenvalue)`` for a symmetric matrix.

    The verdict is taken with a tolerance proportional to the matrix order,
    because the smallest eigenvalue of an exactly singular PSD matrix is
    computed as a small negative number in floating point. A matrix is
    reported as not positive semi-definite only when the smallest eigenvalue
    is below ``-PSD_TOLERANCE * n``; the raw eigenvalue is returned so the
    caller can report how far outside the cone the declaration falls.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise AggregationError(f"correlation matrix must be square, got shape {matrix.shape}")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
        raise AggregationError("correlation matrix must be symmetric")
    if matrix.shape[0] == 0:
        return True, 0.0
    smallest = float(np.linalg.eigvalsh(matrix)[0])
    return smallest >= -PSD_TOLERANCE * matrix.shape[0], smallest


def quadratic_form(weights: Sequence[float], matrix: np.ndarray) -> float:
    """Return ``sqrt(w^T R w)`` in the unit of ``w``.

    A variance that comes out marginally negative from rounding on a singular
    or near-singular matrix is clipped to zero rather than producing a NaN;
    the clip is at most one rounding step and is never used to hide a genuine
    negative, which cannot arise from a positive semi-definite matrix.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise AggregationError(f"weights must be one-dimensional, got shape {w.shape}")
    if matrix.shape != (w.size, w.size):
        raise AggregationError(
            f"correlation matrix shape {matrix.shape} does not match {w.size} weights"
        )
    variance = float(w @ matrix @ w)
    return float(np.sqrt(max(variance, 0.0)))


def budget_correlation_matrix(budget: Budget) -> tuple[list[str], np.ndarray]:
    """Return the ordered leaf identifiers of ``budget`` and their matrix."""
    leaf_ids = [item.id for item in leaves(budget.root)]
    return leaf_ids, correlation_matrix(leaf_ids, budget.correlations)


def declared_pairs(budget: Budget, among: Sequence[str]) -> list[Correlation]:
    """Return the declared non-zero correlations whose both ends lie in ``among``."""
    members = set(among)
    return [
        corr
        for corr in budget.correlations
        if corr.rho != 0.0 and corr.a in members and corr.b in members
    ]
