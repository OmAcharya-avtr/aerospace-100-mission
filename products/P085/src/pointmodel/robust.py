"""Robust fits, and the disagreement between them and least squares.

Why a robust fit is not optional here
-------------------------------------
A pointing run produces residuals from acquisitions, and some acquisitions
fail: the wrong source was centred, the autoguider locked on a neighbour, a
cable wrap reset mid-scan, the operator typed a coordinate wrong. Those
residuals are not Gaussian tails of the pointing error; they are a different
process, and least squares, whose influence function is unbounded, is dragged
by them in proportion to their size. A single 300-arcsec acquisition failure in
a 200-observation run with 2-arcsec noise moves a least-squares coefficient by
more than its entire confidence interval.

Two robust fits are provided, and **neither is preferred over least squares by
this package**. What is reported is their disagreement, as a diagnostic. A
robust fit that agrees with least squares says the outliers did not matter. A
robust fit that disagrees says the data contain a second process and the
least-squares numbers describe a mixture; which fit to believe then depends on
what the outliers are, which is an operational question this package cannot
answer and does not pretend to.

Huber M-estimation
------------------
Minimise ``sum_i rho(u_i)`` with ``u_i = (y_i - (A theta)_i) / (sigma_i * s)``
and Huber's loss

    rho(u) = u^2 / 2                     for |u| <= delta,
    rho(u) = delta |u| - delta^2 / 2     otherwise.                        (25)

Huber, P. J. (1964), "Robust estimation of a location parameter", Annals of
Mathematical Statistics 35(1), 73-101, introduces (25); Huber, P. J. (1981),
"Robust Statistics", Wiley, chapter 7, gives the iteratively reweighted least
squares solution used here and the asymptotic covariance below. The tuning
constant ``delta = 1.345`` is the standard choice giving 95 per cent
asymptotic efficiency relative to least squares under exactly Gaussian errors;
it is a declared argument, not a hidden constant.

The iteration is ordinary IRLS: weights ``w_i = min(1, delta / |u_i|)``,
re-solve, recompute. The residual scale ``s`` is re-estimated each iteration
from the median absolute deviation of the standardised residuals,

    s = 1.4826 * median( |u_i - median(u)| ),                               (26)

with 1.4826 the consistency constant making (26) an unbiased estimator of the
standard deviation for Gaussian data; see Rousseeuw, P. J. & Croux, C. (1993),
"Alternatives to the median absolute deviation", Journal of the American
Statistical Association 88(424), 1273-1283.

Standard errors use Huber's asymptotic sandwich,

    Cov = K * (A^T W A)^{-1},
    K = s^2 * ( mean psi(u)^2 ) / ( mean psi'(u) )^2,                       (27)

with ``psi = rho'`` clipped at ``+-delta`` and ``psi'`` the indicator of
``|u| <= delta``. Equation (27) is asymptotic: it is not a small-sample
interval, and it is conditional on the same term set and the same geometry as
the least-squares intervals, so it inherits the identifiability caveat in full.

RANSAC
------
Fischler, M. A. & Bolles, R. C. (1981), "Random sample consensus: a paradigm
for model fitting with applications to image analysis and automated
cartography", Communications of the ACM 24(6), 381-395. Draw a random subset
of observations, fit, count the observations whose residual is within
``inlier_sigma`` declared sigmas, keep the largest consensus set, and refit on
it.

Two deliberate departures from the paper, both because of this problem's
structure:

- The subset is **not** minimal. A nine-parameter model needs five
  observations to be determined at all, and a five-observation subset of a
  pointing run is almost always near-degenerate, so a minimal-subset RANSAC
  spends its iterations fitting noise. The default subset is
  ``3 * 9 = 27`` observations, which is 54 rows.
- Subset fits go through the same SVD pseudo-inverse as everything else, so a
  degenerate subset yields the minimum-norm solution rather than an exception.
  That is the right behaviour for a consensus count — a degenerate subset still
  predicts the residual field fine, it just cannot attribute it — and it is the
  reason a RANSAC consensus set on a rank-deficient geometry is meaningful
  while the coefficients it produces are not.

Both the subset size and the inlier threshold are declared arguments and are
reported in the result. The sampler is seeded; the same seed gives the same
consensus set.

What is reported as the diagnostic
----------------------------------
:func:`disagreement` compares two fits in the only space that is identifiable
regardless of geometry: the **predicted residual field over the observations
that were actually made**. The RMS of ``A (theta_1 - theta_2)`` in arcseconds
is a statement about the sky and it is well defined even when neither
coefficient vector is. Per-term differences are also reported, in units of the
quadrature sum of the two standard errors, but they are flagged as meaningless
for any term the identifiability report names as unidentifiable, and
:func:`disagreement` takes that term list as an argument so the point cannot be
forgotten.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .terms import N_TERMS, TERM_NAMES
from .wls import WlsFit, fit

__all__ = [
    "Disagreement",
    "HUBER_DELTA",
    "MAD_CONSISTENCY",
    "RobustFit",
    "disagreement",
    "huber_fit",
    "ransac_fit",
]

#: Standard Huber tuning constant, 95 per cent Gaussian efficiency.
HUBER_DELTA = 1.345

#: Consistency constant of equation (26), Rousseeuw & Croux (1993).
MAD_CONSISTENCY = 1.4826


@dataclass(frozen=True)
class RobustFit:
    """A robust fit of the pointing model.

    Attributes
    ----------
    coefficients
        Length-9 estimate in arcseconds.
    standard_errors
        Length-9 asymptotic standard errors in arcseconds. For Huber these come
        from equation (27); for RANSAC they come from an ordinary weighted
        least squares on the consensus set and are therefore **conditional on
        that set** and optimistic, because the set was chosen using the data.
    covariance
        ``(9, 9)`` covariance in arcsec^2, same caveats.
    weights
        Length-``2n`` final robust weights in ``[0, 1]``. For RANSAC these are
        0 or 1 and mark the consensus set.
    residuals
        Length-``2n`` residuals at the estimate, arcseconds.
    scale_arcsec
        Robust residual scale, arcseconds: equation (26) times the declared
        sigma for Huber, the consensus-set weighted RMS for RANSAC.
    iterations
        IRLS iterations used, or RANSAC trials performed.
    downweighted
        Count of rows with a final weight below 1 (Huber) or 0 (RANSAC).
    method
        ``"huber"`` or ``"ransac"``.
    detail
        Method-specific scalars: ``delta`` and ``converged`` for Huber,
        ``inlier_sigma``, ``subset_size``, ``consensus_rows`` and ``seed`` for
        RANSAC.
    """

    coefficients: np.ndarray
    standard_errors: np.ndarray
    covariance: np.ndarray
    weights: np.ndarray
    residuals: np.ndarray
    scale_arcsec: float
    iterations: int
    downweighted: int
    method: str
    detail: dict[str, float]

    def as_mapping(self) -> dict[str, float]:
        """Coefficients as a name-to-arcsecond dict."""
        return {name: float(self.coefficients[i]) for i, name in enumerate(TERM_NAMES)}


def _prepare(
    design: np.ndarray, observed: np.ndarray, sigma: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(design, dtype=float)
    values = np.asarray(observed, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != N_TERMS:
        raise ValueError(f"design must be (rows, {N_TERMS}), got shape {matrix.shape}")
    if values.ndim != 1 or values.size != matrix.shape[0]:
        raise ValueError(
            f"observed must be 1-D with {matrix.shape[0]} entries, got shape {values.shape}"
        )
    scale = np.broadcast_to(np.asarray(sigma, dtype=float), values.shape).astype(float)
    if np.any(scale <= 0.0) or not np.all(np.isfinite(scale)):
        raise ValueError("sigma must be finite and strictly positive")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(values)):
        raise ValueError("design or observed contains a non-finite value")
    return matrix, values, scale


def _weighted_solve(
    matrix: np.ndarray, values: np.ndarray, weights: np.ndarray, rcond: float = 1e-12
) -> np.ndarray:
    """Minimum-norm weighted least squares solution through the SVD.

    Rows with zero weight drop out. A rank-deficient system returns the
    minimum-norm solution rather than raising, which is deliberate: see the
    module docstring on RANSAC subsets.
    """
    root = np.sqrt(weights)
    whitened = matrix * root[:, None]
    active = np.linalg.norm(whitened, axis=0) > 0.0
    if not np.all(active):
        raise ValueError(
            "a design column is identically zero among the weighted rows; "
            f"columns {[TERM_NAMES[i] for i in np.flatnonzero(~active)]} carry no information"
        )
    column_scales = np.linalg.norm(whitened, axis=0)
    normalised = whitened / column_scales
    u, singular, vt = np.linalg.svd(normalised, full_matrices=False)
    keep = singular > rcond * singular[0]
    inverse = np.zeros_like(singular)
    inverse[keep] = 1.0 / singular[keep]
    solution = vt.T @ (inverse * (u.T @ (values * root)))
    return solution / column_scales


def huber_fit(
    design: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray | float = 1.0,
    delta: float = HUBER_DELTA,
    max_iterations: int = 60,
    tolerance_arcsec: float = 1e-9,
) -> RobustFit:
    """Huber M-estimate by iteratively reweighted least squares, equation (25).

    Parameters
    ----------
    design
        ``(2n, 9)`` stacked design matrix, dimensionless.
    observed
        Length-``2n`` stacked residual vector, arcseconds.
    sigma
        Per-row declared standard deviation, arcseconds.
    delta
        Huber tuning constant in units of the robust scale. Must be positive.
    max_iterations
        IRLS cap. Convergence is declared when the largest coefficient change
        falls below ``tolerance_arcsec``; the result records whether it did.
    tolerance_arcsec
        Convergence threshold on the coefficient vector, arcseconds.

    Returns
    -------
    RobustFit

    Notes
    -----
    IRLS on Huber's loss is a convergent fixed-point iteration for a convex
    objective, so the result does not depend on the starting point; the start
    here is the ordinary weighted least squares solution. The returned
    ``converged`` flag is reported rather than asserted, because a scale
    estimate that oscillates between two values at the 1e-10 arcsec level is
    possible and is not a defect worth raising on.
    """
    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be at least 1, got {max_iterations}")
    matrix, values, scale = _prepare(design, observed, sigma)
    base_weights = 1.0 / scale**2
    coefficients = _weighted_solve(matrix, values, base_weights)
    weights = base_weights.copy()
    robust_scale = 1.0
    iterations = 0
    converged = False
    for step in range(1, max_iterations + 1):
        iterations = step
        residuals = values - matrix @ coefficients
        standardised = residuals / scale
        centre = float(np.median(standardised))
        mad = float(np.median(np.abs(standardised - centre)))
        robust_scale = MAD_CONSISTENCY * mad
        if robust_scale <= 0.0:
            robust_scale = float(np.sqrt(np.mean(standardised**2)))
        if robust_scale <= 0.0:
            robust_scale = 1.0
        unit = standardised / robust_scale
        factor = np.minimum(1.0, delta / np.maximum(np.abs(unit), 1e-300))
        weights = base_weights * factor
        updated = _weighted_solve(matrix, values, weights)
        shift = float(np.max(np.abs(updated - coefficients)))
        coefficients = updated
        if shift < tolerance_arcsec:
            converged = True
            break

    residuals = values - matrix @ coefficients
    unit = residuals / (scale * robust_scale)
    psi = np.clip(unit, -delta, delta)
    psi_prime = (np.abs(unit) <= delta).astype(float)
    denominator = float(np.mean(psi_prime))
    if denominator <= 0.0:
        raise ValueError(
            "every residual was downweighted; delta is too small for this data set"
        )
    correction = robust_scale**2 * float(np.mean(psi**2)) / denominator**2
    whitened = matrix * np.sqrt(weights)[:, None]
    column_scales = np.linalg.norm(whitened, axis=0)
    normalised = whitened / column_scales
    _u, singular, vt = np.linalg.svd(normalised, full_matrices=False)
    inverse_sq = np.zeros_like(singular)
    live = singular > 1e-12 * singular[0]
    inverse_sq[live] = 1.0 / singular[live] ** 2
    covariance = ((vt.T * inverse_sq) @ vt) / np.outer(column_scales, column_scales)
    covariance = covariance * correction
    return RobustFit(
        coefficients=coefficients,
        standard_errors=np.sqrt(np.clip(np.diag(covariance), 0.0, None)),
        covariance=covariance,
        weights=weights / base_weights,
        residuals=residuals,
        scale_arcsec=float(robust_scale * np.median(scale)),
        iterations=iterations,
        downweighted=int(np.count_nonzero(weights < base_weights * (1.0 - 1e-12))),
        method="huber",
        detail={"delta": float(delta), "converged": float(converged)},
    )


def ransac_fit(
    design: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray | float = 1.0,
    inlier_sigma: float = 3.0,
    subset_size: int | None = None,
    trials: int = 200,
    seed: int = 20261101,
) -> RobustFit:
    """RANSAC-style consensus fit, Fischler & Bolles (1981).

    Parameters
    ----------
    design
        ``(2n, 9)`` stacked design matrix. The stacking convention of
        :func:`pointmodel.terms.stack_residuals` is assumed: rows ``0..n-1`` are
        cross-elevation and rows ``n..2n-1`` are elevation for the same
        observations, so subsets are drawn over **observations** and both of an
        observation's rows are taken or left together. That matters: an
        acquisition failure corrupts both components of one observation, so
        sampling rows independently would split the very structure the method
        is there to find.
    observed
        Length-``2n`` residual vector, arcseconds.
    sigma
        Per-row declared standard deviation, arcseconds.
    inlier_sigma
        Consensus threshold in declared sigmas. At 3.0 a clean Gaussian run
        leaves about 0.27 per cent of rows outside the consensus set by chance,
        which is reported rather than hidden.
    subset_size
        Observations per trial. Default ``3 * 9 = 27``; must be at least 5, so
        that 10 rows constrain 9 terms, and at most the observation count.
    trials
        Number of random subsets.
    seed
        Seed for :func:`numpy.random.default_rng`. The result is deterministic
        in the seed.

    Returns
    -------
    RobustFit

    Raises
    ------
    ValueError
        If the row count is odd (the stacking convention requires an even
        count), if ``subset_size`` is out of range, or on the usual input
        problems.
    """
    matrix, values, scale = _prepare(design, observed, sigma)
    rows = matrix.shape[0]
    if rows % 2 != 0:
        raise ValueError(
            f"design has {rows} rows; the stacked convention needs an even count "
            "(cross-elevation block above elevation block)"
        )
    observations = rows // 2
    if subset_size is None:
        subset_size = 3 * N_TERMS
    minimum = (N_TERMS + 1) // 2
    if not minimum <= subset_size <= observations:
        raise ValueError(
            f"subset_size must lie in [{minimum}, {observations}], got {subset_size}"
        )
    if trials < 1:
        raise ValueError(f"trials must be at least 1, got {trials}")
    if inlier_sigma <= 0.0:
        raise ValueError(f"inlier_sigma must be positive, got {inlier_sigma}")

    rng = np.random.default_rng(seed)
    base_weights = 1.0 / scale**2
    best_mask = np.ones(rows, dtype=bool)
    best_count = -1
    best_cost = np.inf
    for _trial in range(trials):
        picked = rng.choice(observations, size=subset_size, replace=False)
        mask = np.zeros(rows, dtype=bool)
        mask[picked] = True
        mask[picked + observations] = True
        try:
            candidate = _weighted_solve(matrix, values, base_weights * mask)
        except ValueError:
            continue
        standardised = np.abs(values - matrix @ candidate) / scale
        inliers = standardised <= inlier_sigma
        count = int(np.count_nonzero(inliers))
        cost = float(np.sum(np.minimum(standardised, inlier_sigma) ** 2))
        if count > best_count or (count == best_count and cost < best_cost):
            best_count, best_cost, best_mask = count, cost, inliers

    if best_count < N_TERMS:
        raise ValueError(
            f"no RANSAC trial reached a consensus set of {N_TERMS} rows "
            f"(best was {best_count}); the residuals are not described by this model "
            "at this inlier threshold"
        )
    consensus = fit(matrix[best_mask], values[best_mask], scale[best_mask])
    residuals = values - matrix @ consensus.coefficients
    weighted_rms = float(
        np.sqrt(np.sum(base_weights[best_mask] * residuals[best_mask] ** 2) / best_count)
    )
    return RobustFit(
        coefficients=consensus.coefficients,
        standard_errors=consensus.standard_errors,
        covariance=consensus.covariance,
        weights=best_mask.astype(float),
        residuals=residuals,
        scale_arcsec=weighted_rms * float(np.median(scale)),
        iterations=trials,
        downweighted=int(rows - best_count),
        method="ransac",
        detail={
            "inlier_sigma": float(inlier_sigma),
            "subset_size": float(subset_size),
            "consensus_rows": float(best_count),
            "seed": float(seed),
        },
    )


@dataclass(frozen=True)
class Disagreement:
    """The diagnostic: how far apart two fits are, and where it is meaningful.

    Attributes
    ----------
    prediction_rms_arcsec
        Root mean square of ``A (theta_1 - theta_2)`` over the observed rows,
        arcseconds. This is identifiable whatever the geometry.
    prediction_max_arcsec
        Largest absolute element of the same vector, arcseconds.
    difference
        Length-9 coefficient difference ``theta_1 - theta_2``, arcseconds.
    z_score
        Length-9 ``|difference| / sqrt(se_1^2 + se_2^2)``, dimensionless. NaN
        where both standard errors are zero.
    unidentifiable
        Term names for which ``difference`` and ``z_score`` are meaningless,
        passed in from the identifiability report.
    flagged
        True when ``prediction_rms_arcsec`` exceeds
        ``threshold_sigma * sigma_reference``.
    threshold_sigma, sigma_reference
        The declared threshold and the reference residual scale it multiplies.
    reportable_terms
        Term names not in ``unidentifiable``, in design-matrix order.
    """

    prediction_rms_arcsec: float
    prediction_max_arcsec: float
    difference: np.ndarray
    z_score: np.ndarray
    unidentifiable: tuple[str, ...]
    flagged: bool
    threshold_sigma: float
    sigma_reference: float
    reportable_terms: tuple[str, ...]


def disagreement(
    design: np.ndarray,
    first: WlsFit | RobustFit,
    second: WlsFit | RobustFit,
    sigma_reference: float,
    unidentifiable: tuple[str, ...] = (),
    threshold_sigma: float = 1.0,
) -> Disagreement:
    """Compare two fits in prediction space and in coefficient space.

    ``sigma_reference`` is the residual scale the threshold is measured
    against, in arcseconds; pass the declared observation sigma. The default
    ``threshold_sigma = 1.0`` flags a disagreement whose predicted sky
    difference is as large as one observation's noise, which is the level at
    which the choice between the two fits changes a pointing correction.

    Nothing here decides which fit is right. ``flagged`` means the two fits
    describe measurably different skies and a human has to look at the
    downweighted observations.
    """
    matrix = np.asarray(design, dtype=float)
    if sigma_reference <= 0.0:
        raise ValueError(f"sigma_reference must be positive, got {sigma_reference}")
    if threshold_sigma <= 0.0:
        raise ValueError(f"threshold_sigma must be positive, got {threshold_sigma}")
    unknown = sorted(set(unidentifiable) - set(TERM_NAMES))
    if unknown:
        raise ValueError(f"unknown term name(s) in unidentifiable: {unknown}")
    difference = np.asarray(first.coefficients, dtype=float) - np.asarray(
        second.coefficients, dtype=float
    )
    predicted = matrix @ difference
    combined = np.sqrt(
        np.asarray(first.standard_errors, dtype=float) ** 2
        + np.asarray(second.standard_errors, dtype=float) ** 2
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(combined > 0.0, np.abs(difference) / combined, np.nan)
    rms = float(np.sqrt(np.mean(predicted**2)))
    return Disagreement(
        prediction_rms_arcsec=rms,
        prediction_max_arcsec=float(np.max(np.abs(predicted))),
        difference=difference,
        z_score=z,
        unidentifiable=tuple(unidentifiable),
        flagged=bool(rms > threshold_sigma * sigma_reference),
        threshold_sigma=float(threshold_sigma),
        sigma_reference=float(sigma_reference),
        reportable_terms=tuple(
            name for name in TERM_NAMES if name not in set(unidentifiable)
        ),
    )
