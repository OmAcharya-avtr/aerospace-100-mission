"""Weighted least squares with a full covariance, and the normal-equation trap.

The fit
-------
For a stacked residual vector ``y`` of length ``2n`` in arcseconds, a design
matrix ``A`` of shape ``(2n, 9)`` from :func:`pointmodel.terms.design_matrix`,
and per-row standard deviations ``sigma``, the weighted least squares estimate
minimises

    chi2(theta) = sum_i ( (y_i - (A theta)_i) / sigma_i )^2.                (18)

Writing ``W = diag(1/sigma_i^2)``, the estimate and its covariance are the
textbook pair

    theta_hat = (A^T W A)^{-1} A^T W y,                                     (19)
    Cov(theta_hat) = (A^T W A)^{-1}.                                        (20)

See Bevington & Robinson (2003), "Data Reduction and Error Analysis for the
Physical Sciences", 3rd edition, chapter 7, or Press et al. (2007),
"Numerical Recipes", 3rd edition, section 15.4, which also states the point
this module is built around.

**Equations (19) and (20) are never evaluated as written here.** They are
evaluated through the singular value decomposition of the whitened design
matrix ``A_w = W^{1/2} A``, because forming ``A^T W A`` squares the condition
number. That is not a style preference: on the single-azimuth geometry of this
product's own validation set, the whitened design matrix has condition number
5.2e15 and the normal matrix has 1.5e16 with a numerically meaningless inverse,
and the inverse's diagonal then reports formal standard errors of 1.1e6 to
4.5e6 arcsec for parameters whose actual scatter over noise realisations is
4 to 13 arcsec. The measured numbers are in
``validation/validate_identifiability_output.txt``.

:func:`fit` uses the SVD. :func:`normal_equation_fit` deliberately uses
equations (19) and (20) literally, is **not** part of the recommended API, and
exists so that the failure above can be demonstrated rather than asserted.

Error model
-----------
``error_model="scaled"`` (the default) multiplies (20) by the reduced
chi-square and takes interval multipliers from Student's t with ``2n - rank``
degrees of freedom. This is the right choice when ``sigma`` describes the
relative weighting of the observations but its absolute scale is uncertain,
which is the normal situation for a pointing run. ``error_model="declared"``
trusts ``sigma`` absolutely and takes multipliers from the standard normal. The
choice changes the intervals and is reported in the result, never implicit.

Variance inflation
------------------
The per-term variance inflation factor reported here is

    VIF_j = Cov(theta_hat)_jj * (A_w^T A_w)_jj,                             (21)

the ratio of a coefficient's actual variance to the variance it would have if
its column were orthogonal to every other column. For unit-norm columns this
is the classical ``1 / (1 - R_j^2)`` and it is scale invariant, computed here
as ``sum_k V_jk^2 / s_k^2`` from the SVD of the column-normalised whitened
design matrix. A VIF of 100 means the geometry has cost that coefficient a
factor of 10 in standard error relative to an orthogonal design; an infinite
VIF means the coefficient is not determined at all, and
:mod:`pointmodel.identifiability` is what says so in terms of the physical
combinations.

What a confidence interval here does and does not mean
-----------------------------------------------------
The interval on a coefficient is conditional on the declared term set being
complete and correct. It says nothing about a term that is missing from the
model, and nothing about whether the geometry can separate the term from its
neighbours — that is a rank statement, and an interval on an unidentifiable
coefficient is wide but still misleading, because the *shape* of the
uncertainty is a near-degenerate ellipsoid and the marginal interval hides the
correlation entirely. This is why every fit in this package is reported beside
an identifiability verdict and never alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .terms import N_TERMS, TERM_NAMES

__all__ = ["WlsFit", "fit", "normal_equation_fit"]


@dataclass(frozen=True)
class WlsFit:
    """A weighted least squares fit of the pointing model.

    Attributes
    ----------
    coefficients
        Length-9 estimate in **arcseconds**, in :data:`pointmodel.terms.TERM_NAMES`
        order.
    covariance
        ``(9, 9)`` covariance in arcsec^2, equation (20), scaled per
        ``error_model``.
    standard_errors
        Square roots of the diagonal of ``covariance``, arcseconds.
    confidence_intervals
        ``(9, 2)`` lower and upper bounds in arcseconds at ``confidence_level``.
    confidence_level
        Two-sided coverage, e.g. 0.95.
    interval_multiplier
        The t or z multiplier actually used.
    residuals
        Length-``2n`` fit residuals ``y - A theta_hat`` in arcseconds, in the
        stacked order of :func:`pointmodel.terms.stack_residuals`.
    weights
        Length-``2n`` weights ``1 / sigma^2`` actually used, arcsec^-2.
    chi_square, degrees_of_freedom, reduced_chi_square
        Equation (18) at the estimate, ``2n - rank``, and their ratio.
    rms_arcsec
        Unweighted root mean square of ``residuals``, arcseconds.
    singular_values
        Singular values of the **column-normalised** whitened design matrix,
        dimensionless, descending.
    condition_number
        Ratio of the largest to the smallest of ``singular_values``.
    rank
        Numerical rank of the whitened design matrix at ``rcond``.
    vif
        Length-9 variance inflation factors, equation (21), dimensionless.
    column_scales
        Length-9 Euclidean norms of the whitened design matrix columns, used to
        normalise it; needed to interpret :mod:`pointmodel.identifiability`.
    error_model
        ``"scaled"`` or ``"declared"``.
    method
        ``"svd"`` or ``"normal-equations"``.
    """

    coefficients: np.ndarray
    covariance: np.ndarray
    standard_errors: np.ndarray
    confidence_intervals: np.ndarray
    confidence_level: float
    interval_multiplier: float
    residuals: np.ndarray
    weights: np.ndarray
    chi_square: float
    degrees_of_freedom: int
    reduced_chi_square: float
    rms_arcsec: float
    singular_values: np.ndarray
    condition_number: float
    rank: int
    vif: np.ndarray
    column_scales: np.ndarray
    error_model: str
    method: str
    term_names: tuple[str, ...] = field(default=TERM_NAMES)

    def as_mapping(self) -> dict[str, float]:
        """Coefficients as a name-to-arcsecond dict."""
        return {name: float(self.coefficients[i]) for i, name in enumerate(self.term_names)}


def _check_inputs(
    design: np.ndarray, observed: np.ndarray, sigma: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(design, dtype=float)
    values = np.asarray(observed, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"design must be 2-D, got shape {matrix.shape}")
    if matrix.shape[1] != N_TERMS:
        raise ValueError(
            f"design must have {N_TERMS} columns for the declared term set, "
            f"got {matrix.shape[1]}"
        )
    if values.ndim != 1 or values.size != matrix.shape[0]:
        raise ValueError(
            f"observed must be 1-D with {matrix.shape[0]} entries, got shape {values.shape}"
        )
    if matrix.shape[0] < N_TERMS:
        raise ValueError(
            f"{matrix.shape[0]} residual rows cannot constrain {N_TERMS} terms; "
            f"at least {N_TERMS} rows are needed and far more are wanted"
        )
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(values)):
        raise ValueError("design or observed contains a non-finite value")
    scale = np.broadcast_to(np.asarray(sigma, dtype=float), values.shape).astype(float)
    if not np.all(np.isfinite(scale)):
        raise ValueError("sigma contains a non-finite value")
    if np.any(scale <= 0.0):
        raise ValueError(
            f"sigma must be strictly positive, got a minimum of {float(np.min(scale)):.6g}"
        )
    return matrix, values, scale


def _intervals(
    coefficients: np.ndarray,
    standard_errors: np.ndarray,
    confidence_level: float,
    degrees_of_freedom: int,
    error_model: str,
) -> tuple[np.ndarray, float]:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must lie strictly in (0, 1), got {confidence_level}"
        )
    tail = 0.5 * (1.0 + confidence_level)
    if error_model == "declared" or degrees_of_freedom <= 0:
        multiplier = float(stats.norm.ppf(tail))
    else:
        multiplier = float(stats.t.ppf(tail, degrees_of_freedom))
    lower = coefficients - multiplier * standard_errors
    upper = coefficients + multiplier * standard_errors
    return np.stack([lower, upper], axis=1), multiplier


def fit(
    design: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray | float = 1.0,
    confidence_level: float = 0.95,
    error_model: str = "scaled",
    rcond: float = 1e-12,
) -> WlsFit:
    """Weighted least squares via the SVD of the whitened design matrix.

    Parameters
    ----------
    design
        ``(2n, 9)`` stacked design matrix, dimensionless.
    observed
        Length-``2n`` stacked residual vector, arcseconds.
    sigma
        Per-row standard deviation in arcseconds, scalar or length-``2n``.
    confidence_level
        Two-sided coverage for the reported intervals.
    error_model
        ``"scaled"`` or ``"declared"``; see the module docstring.
    rcond
        Relative singular-value cut for the pseudo-inverse and for the rank.
        Directions below ``rcond`` times the largest singular value are set to
        zero in the solution, which makes the estimate the minimum-norm one in
        those directions. That choice is reported as a rank, and the
        combinations it suppresses are named by
        :mod:`pointmodel.identifiability`.

    Returns
    -------
    WlsFit

    Raises
    ------
    ValueError
        On a shape mismatch, a non-finite entry, a non-positive sigma, an
        out-of-range confidence level, an unknown error model, or fewer rows
        than terms.
    """
    if error_model not in {"scaled", "declared"}:
        raise ValueError(
            f"error_model must be 'scaled' or 'declared', got {error_model!r}"
        )
    matrix, values, scale = _check_inputs(design, observed, sigma)
    weights = 1.0 / scale**2
    root = 1.0 / scale
    whitened = matrix * root[:, None]
    whitened_y = values * root

    column_scales = np.linalg.norm(whitened, axis=0)
    if np.any(column_scales <= 0.0):
        dead = [TERM_NAMES[i] for i in np.flatnonzero(column_scales <= 0.0)]
        raise ValueError(
            f"design column(s) {dead} are identically zero over these observations; "
            "the geometry carries no information about them at all"
        )
    normalised = whitened / column_scales

    u, singular, vt = np.linalg.svd(normalised, full_matrices=False)
    cut = rcond * singular[0]
    keep = singular > cut
    rank = int(np.count_nonzero(keep))
    inverse = np.zeros_like(singular)
    inverse[keep] = 1.0 / singular[keep]
    scaled_solution = vt.T @ (inverse * (u.T @ whitened_y))
    coefficients = scaled_solution / column_scales

    inverse_sq = np.zeros_like(singular)
    inverse_sq[keep] = 1.0 / singular[keep] ** 2
    covariance_scaled = (vt.T * inverse_sq) @ vt
    covariance = covariance_scaled / np.outer(column_scales, column_scales)
    # The variance inflation factors use the FULL spectrum, not the truncated
    # one. They are a statement about the design, which is what a reader needs:
    # an infinite VIF says the geometry carries no information about that term.
    # The standard errors above are a statement about the estimator actually
    # returned, which sets the truncated directions to zero and so has a
    # smaller variance than the design supports -- at the cost of a bias that
    # no standard error can show. The two disagreeing is the whole point.
    full_inverse_sq = np.where(
        singular > 0.0, 1.0 / np.where(singular > 0.0, singular, 1.0) ** 2, np.inf
    )

    residuals = values - matrix @ coefficients
    chi_square = float(np.sum(weights * residuals**2))
    degrees_of_freedom = int(matrix.shape[0] - rank)
    reduced = chi_square / degrees_of_freedom if degrees_of_freedom > 0 else float("nan")
    if error_model == "scaled" and degrees_of_freedom > 0:
        covariance = covariance * reduced
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    intervals, multiplier = _intervals(
        coefficients, standard_errors, confidence_level, degrees_of_freedom, error_model
    )
    vif = np.einsum("jk,k->j", vt.T**2, full_inverse_sq)

    return WlsFit(
        coefficients=coefficients,
        covariance=covariance,
        standard_errors=standard_errors,
        confidence_intervals=intervals,
        confidence_level=float(confidence_level),
        interval_multiplier=multiplier,
        residuals=residuals,
        weights=weights,
        chi_square=chi_square,
        degrees_of_freedom=degrees_of_freedom,
        reduced_chi_square=float(reduced),
        rms_arcsec=float(np.sqrt(np.mean(residuals**2))),
        singular_values=singular,
        condition_number=float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf"),
        rank=rank,
        vif=vif,
        column_scales=column_scales,
        error_model=error_model,
        method="svd",
    )


def normal_equation_fit(
    design: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray | float = 1.0,
    confidence_level: float = 0.95,
    error_model: str = "scaled",
) -> WlsFit:
    """Equations (19) and (20) evaluated literally. **Not the recommended API.**

    This exists to be wrong in a measurable way. It forms ``A^T W A`` and
    inverts it with :func:`numpy.linalg.inv`, which is what a pointing-model
    fitter written straight from a textbook does, and it reports the diagonal of
    that inverse as the standard errors. On a well-conditioned geometry it
    agrees with :func:`fit` to many digits. On a rank-deficient one it returns a
    specific, repeatable, confidently-quoted split of parameters that no
    geometry can separate, and formal standard errors unrelated to the actual
    scatter in either direction.

    ``rank`` is reported as the full term count regardless, because that is what
    this method believes; ``condition_number`` is the condition number of the
    **normal matrix**, not of the design matrix, for the same reason.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the normal matrix is exactly singular to LAPACK. It usually is not,
        even when the design matrix is rank deficient, which is the whole
        problem.
    """
    if error_model not in {"scaled", "declared"}:
        raise ValueError(
            f"error_model must be 'scaled' or 'declared', got {error_model!r}"
        )
    matrix, values, scale = _check_inputs(design, observed, sigma)
    weights = 1.0 / scale**2
    normal = matrix.T @ (weights[:, None] * matrix)
    right = matrix.T @ (weights * values)
    inverse = np.linalg.inv(normal)
    coefficients = inverse @ right
    residuals = values - matrix @ coefficients
    chi_square = float(np.sum(weights * residuals**2))
    degrees_of_freedom = int(matrix.shape[0] - N_TERMS)
    reduced = chi_square / degrees_of_freedom if degrees_of_freedom > 0 else float("nan")
    covariance = inverse * (reduced if error_model == "scaled" and degrees_of_freedom > 0 else 1.0)
    standard_errors = np.sqrt(np.abs(np.diag(covariance)))
    intervals, multiplier = _intervals(
        coefficients, standard_errors, confidence_level, degrees_of_freedom, error_model
    )
    whitened = matrix * (1.0 / scale)[:, None]
    column_scales = np.linalg.norm(whitened, axis=0)
    singular = np.linalg.svd(whitened / column_scales, compute_uv=False)
    return WlsFit(
        coefficients=coefficients,
        covariance=covariance,
        standard_errors=standard_errors,
        confidence_intervals=intervals,
        confidence_level=float(confidence_level),
        interval_multiplier=multiplier,
        residuals=residuals,
        weights=weights,
        chi_square=chi_square,
        degrees_of_freedom=degrees_of_freedom,
        reduced_chi_square=float(reduced),
        rms_arcsec=float(np.sqrt(np.mean(residuals**2))),
        singular_values=singular,
        condition_number=float(np.linalg.cond(normal)),
        rank=N_TERMS,
        vif=np.diag(inverse) * np.diag(normal),
        column_scales=column_scales,
        error_model=error_model,
        method="normal-equations",
    )
