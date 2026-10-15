"""Local identifiability: what the data can and cannot pin down.

This module answers the question the package exists for, and it is meant to be
run *before* any parameter value is reported. Given a sensitivity matrix it
returns the rank of the experiment, its condition number, which individual
parameters are weakly seen, and -- the part that is usually missing -- the
*combinations* of parameters that the data does not constrain at all, written
out as combinations rather than reported as a warning.

Method
------
Let ``J_w`` be the weighted log-parameter sensitivity matrix of
:mod:`twincalib.sensitivity`, shape ``(n, p)``, and let

``J_w = U S V^T``

be its singular value decomposition (Golub and Van Loan 2013, section 2.4).
Then:

* **Rank.** The numerical rank is the number of singular values above
  ``rank_tol * s_max``. A rank below ``p`` means the model is *structurally*
  non-identifiable at this parameter point for this design: there is a direction
  in parameter space along which the predictions do not change at all, to the
  accuracy of the differentiation (Jacquez and Greif 1985).
* **Unconstrained combinations.** The right singular vector ``v_i`` belonging to
  a small ``s_i`` is a direction in ``ln theta`` along which moving the
  parameters changes the weighted predictions by only ``s_i`` per unit step.
  Written out, ``v_i`` *is* the unconstrained combination: for the shipped
  ``full-sun`` design it comes out as a combination of ``ln q_sun`` and
  ``ln q_par`` alone, which is the exact answer.
* **Condition number.** ``s_max / s_min``, the standard measure of how far the
  experiment is from degenerate. The inverse Fisher information has condition
  number ``(s_max/s_min)**2``, so a sensitivity condition number of 1e8 means a
  covariance matrix at the edge of double precision.
* **Collinearity index.** Brun et al. (2001), section 2.3: normalise every
  column of the sensitivity matrix to unit length and take ``1/s_min`` of the
  result. This separates "this parameter has no effect" from "these parameters
  have the same effect", because the normalisation removes the magnitude. Brun
  et al. report that subsets with an index above about 10 to 15 are not
  identifiable in practice.
* **Cramer-Rao standard deviations.** ``sqrt(diag(F^-1))`` in log-parameter
  coordinates, which is directly a *relative* standard deviation. For a
  rank-deficient ``F`` the pseudo-inverse is used and the result is flagged, so
  that a number computed from a degenerate matrix is never returned unlabelled.

What this does not do
---------------------
Local identifiability at one parameter point is not global identifiability. A
full-rank sensitivity matrix at ``theta_hat`` does not rule out a second,
distant optimum, and the condition number says nothing about the shape of the
likelihood away from the point. The profile likelihood in :mod:`twincalib.profile`
is the next step precisely because of that, and it is the one that catches
practical non-identifiability of the ``lumped-nodes`` kind, which this module
reports as full rank.

References
----------
Golub, G. H. and Van Loan, C. F., *Matrix Computations*, 4th ed., Johns Hopkins
University Press, 2013. Sections 2.4 and 5.4: the SVD, numerical rank and the
condition number.

Jacquez, J. A. and Greif, P. (1985), "Numerical parameter identifiability and
estimability: integrating identifiability, estimability, and optimal sampling
design", *Mathematical Biosciences* 77(1-2), 201-227. Rank of the sensitivity
matrix as the test for local identifiability.

Brun, R., Reichert, P. and Kuensch, H. R. (2001), *Water Resources Research*
37(4), 1015-1030. The collinearity index.

Raue, A., Kreutz, C., Maiwald, T., Bachmann, J., Schilling, M., Klingmueller, U.
and Timmer, J. (2009), "Structural and practical identifiability analysis of
partially observed dynamical models by exploiting the profile likelihood",
*Bioinformatics* 25(15), 1923-1929. The structural/practical distinction used
throughout this package.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .sensitivity import SensitivityMatrix

__all__ = [
    "DEFAULT_RANK_TOL",
    "DEFAULT_WEAK_RATIO",
    "Combination",
    "IdentifiabilityReport",
    "analyse",
    "collinearity_index",
    "null_direction_loading",
    "worst_subsets",
]


def null_direction_loading(
    sens: SensitivityMatrix, rank_tol: float = 1e-8
) -> np.ndarray:
    """How strongly each parameter loads on the unconstrained directions.

    Parameters
    ----------
    sens:
        Weighted log-parameter sensitivity matrix.
    rank_tol:
        Relative singular-value cut below which a direction counts as null.

    Returns
    -------
    numpy.ndarray
        Shape ``(p,)``, dimensionless, in ``[0, 1]``: the Euclidean norm of each
        parameter's components in the sub-tolerance right singular vectors. Zero
        everywhere when the matrix has full numerical rank.

    Notes
    -----
    This is what turns a pseudo-inverse covariance from misleading into honest.
    A pseudo-inverse reports a *finite* variance for a parameter the data does
    not constrain at all, because it silently drops the infinite direction. A
    parameter with a non-negligible loading here has infinite variance whatever
    the pseudo-inverse diagonal says, and both
    :func:`analyse` and :func:`~twincalib.fitting.fit` use this function to say
    so.
    """
    if not np.isfinite(rank_tol) or not 0.0 < rank_tol <= 1.0:
        raise ValueError("rank_tol must be finite and lie in (0, 1]")
    _u, sv, vt = np.linalg.svd(sens.weighted, full_matrices=False)
    s_max = float(sv[0])
    if s_max <= 0.0:
        return np.ones(sens.n_parameters)
    mask = (sv / s_max) < rank_tol
    if not mask.any():
        return np.zeros(sens.n_parameters)
    return np.sqrt((vt[mask] ** 2).sum(axis=0))

DEFAULT_RANK_TOL: float = 1e-8
"""Relative singular-value cut for the numerical rank, dimensionless.

A singular value below ``DEFAULT_RANK_TOL * s_max`` counts as zero. The value is
set well above double-precision epsilon (2.2e-16) because the sensitivity
matrix is built from central differences of a Runge-Kutta solution, so its own
accuracy is nearer 1e-9 relative than 1e-16. The measured noise floor of the
differentiation is reported in ``validation/validate_sensitivity_output.txt``;
``validation/validate_identifiability_output.txt`` shows that the shipped
structurally degenerate designs fall below this cut by several decades, and that
the full-rank designs sit several decades above it, so the verdict does not
balance on the tolerance.
"""

DEFAULT_WEAK_RATIO: float = 1e-3
"""Relative singular-value cut below which a direction is called weak.

Directions with ``s_i / s_max`` below this are reported as near-null: the data
constrains them a thousand times less well than the best-determined direction.
Unlike the rank tolerance this is a reporting threshold, not a numerical one,
and both the ratio and the raw singular values are always printed so that a
reader can apply their own.
"""


@dataclass(frozen=True)
class Combination:
    """One parameter combination from the right singular basis.

    Attributes
    ----------
    index:
        Position in the spectrum, 0 is the best-determined direction.
    singular_value:
        ``s_i``, dimensionless: the change in the weighted predictions, in units
        of the measurement noise, caused by moving one unit along this direction
        in log-parameter space.
    ratio:
        ``s_i / s_max``, dimensionless.
    vector:
        The right singular vector, shape ``(p,)``, unit norm, in log-parameter
        coordinates.
    parameter_names:
        Names in the same order as ``vector``.
    """

    index: int
    singular_value: float
    ratio: float
    vector: np.ndarray
    parameter_names: tuple[str, ...]

    def terms(self, min_weight: float = 0.05) -> list[tuple[str, float]]:
        """Components with magnitude at least ``min_weight``, largest first."""
        pairs = [
            (n, float(v))
            for n, v in zip(self.parameter_names, self.vector, strict=True)
            if abs(float(v)) >= min_weight
        ]
        return sorted(pairs, key=lambda p: -abs(p[1]))

    def as_text(self, min_weight: float = 0.05) -> str:
        """The combination as a signed sum of ``d ln`` terms, e.g.
        ``+0.31 d ln q_sun -0.95 d ln q_par``."""
        kept = self.terms(min_weight)
        if not kept:
            return "(no component above the weight threshold)"
        parts = [f"{w:+.3f} d ln {n}" for n, w in kept]
        text = " ".join(parts)
        dropped = 1.0 - float(np.sqrt(sum(w * w for _n, w in kept)))
        if dropped > 1e-3:
            text += f"  (+{dropped:.3f} of unit norm in smaller terms)"
        return text


def collinearity_index(sens: SensitivityMatrix, subset: Sequence[int] | None = None) -> float:
    """Brun et al. (2001) collinearity index of a parameter subset.

    Parameters
    ----------
    sens:
        The sensitivity matrix.
    subset:
        Column indices. ``None`` means all columns.

    Returns
    -------
    float
        ``1 / s_min`` of the column-normalised weighted sensitivity submatrix,
        dimensionless, at least 1. ``inf`` when the submatrix is exactly
        singular, and ``inf`` as well when any selected column is identically
        zero, since a parameter with no effect cannot be identified at all.

    Notes
    -----
    Column normalisation is what makes this different from the condition number:
    it removes "this parameter barely matters" and leaves only "these parameters
    do the same thing". Brun et al. report subsets with an index above 10 to 15
    as not identifiable in practice; that is a rule of thumb from environmental
    models, not a theorem, and it is quoted here as theirs.
    """
    cols = sens.weighted if subset is None else sens.weighted[:, list(subset)]
    if cols.ndim != 2 or cols.shape[1] == 0:
        raise ValueError("subset must select at least one column")
    norms = np.linalg.norm(cols, axis=0)
    if np.any(norms <= 0.0):
        return float("inf")
    sv = np.linalg.svd(cols / norms, compute_uv=False)
    smin = float(sv[-1])
    return float("inf") if smin <= 0.0 else 1.0 / smin


def worst_subsets(
    sens: SensitivityMatrix, size: int = 2, top: int = 5
) -> list[tuple[tuple[str, ...], float]]:
    """The most collinear parameter subsets of a given size.

    Returns
    -------
    list
        ``(names, index)`` pairs sorted by decreasing collinearity index, at
        most ``top`` of them.
    """
    p = sens.n_parameters
    if not 2 <= size <= p:
        raise ValueError(f"size must lie in [2, {p}]")
    if top < 1:
        raise ValueError("top must be >= 1")
    scored = [
        (tuple(sens.parameter_names[i] for i in combo), collinearity_index(sens, combo))
        for combo in combinations(range(p), size)
    ]
    scored.sort(key=lambda item: -item[1])
    return scored[:top]


@dataclass(frozen=True)
class IdentifiabilityReport:
    """The result of a local identifiability analysis.

    Attributes
    ----------
    parameter_names:
        Names in parameter order.
    theta:
        The point the analysis was done at.
    singular_values:
        Singular values of the weighted log-sensitivity matrix, descending,
        dimensionless, shape ``(p,)``.
    rank:
        Numerical rank at ``rank_tol``, dimensionless.
    condition_number:
        ``s_max / s_min``, dimensionless; ``inf`` if ``s_min`` is zero.
    rank_tol, weak_ratio:
        The thresholds used.
    combinations:
        All ``p`` right singular directions, best-determined first.
    delta_msqr:
        Per-parameter sensitivity measure, dimensionless, shape ``(p,)``.
    collinearity:
        Collinearity index of the full parameter set, dimensionless.
    relative_sd:
        Cramer-Rao lower bound on the relative standard deviation of each
        parameter, dimensionless, shape ``(p,)``. ``inf`` in the rank-deficient
        directions.
    correlation:
        Parameter correlation matrix implied by the inverse Fisher information,
        dimensionless, shape ``(p, p)``.
    pseudo_inverse_used:
        True when ``F`` was rank deficient and a pseudo-inverse was taken, in
        which case ``relative_sd`` and ``correlation`` are not meaningful for
        the deficient directions and the report says so.
    """

    parameter_names: tuple[str, ...]
    theta: np.ndarray
    singular_values: np.ndarray
    rank: int
    condition_number: float
    rank_tol: float
    weak_ratio: float
    combinations: tuple[Combination, ...]
    delta_msqr: np.ndarray
    collinearity: float
    relative_sd: np.ndarray
    correlation: np.ndarray
    pseudo_inverse_used: bool

    @property
    def n_parameters(self) -> int:
        """Number of parameters, dimensionless."""
        return len(self.parameter_names)

    @property
    def structurally_identifiable(self) -> bool:
        """True when the numerical rank equals the number of parameters.

        This is a statement about the numerical rank at one parameter point and
        one tolerance, not a proof of structural identifiability. Read it as
        "no exactly flat direction was found here".
        """
        return self.rank == self.n_parameters

    @property
    def null_combinations(self) -> tuple[Combination, ...]:
        """Directions below the rank tolerance: unconstrained to differentiation
        accuracy."""
        return tuple(c for c in self.combinations if c.ratio < self.rank_tol)

    @property
    def weak_combinations(self) -> tuple[Combination, ...]:
        """Directions below the weak-reporting ratio, including the null ones."""
        return tuple(c for c in self.combinations if c.ratio < self.weak_ratio)

    def verdict(self) -> str:
        """A one-line verdict, deliberately refusing to say 'identifiable'
        without qualification."""
        if not self.structurally_identifiable:
            n = self.n_parameters - self.rank
            return (
                f"NOT IDENTIFIABLE: rank {self.rank} of {self.n_parameters}, "
                f"{n} unconstrained parameter combination"
                f"{'s' if n != 1 else ''} at this design"
            )
        if self.weak_combinations:
            return (
                f"full numerical rank, but {len(self.weak_combinations)} near-null "
                f"direction(s) below s/s_max = {self.weak_ratio:g}; "
                "treat the covariance as a lower bound and read the profiles"
            )
        return (
            f"full numerical rank {self.rank}, condition number "
            f"{self.condition_number:.3g}; no near-null direction found at this point"
        )


def analyse(
    sens: SensitivityMatrix,
    rank_tol: float = DEFAULT_RANK_TOL,
    weak_ratio: float = DEFAULT_WEAK_RATIO,
) -> IdentifiabilityReport:
    """Local identifiability analysis of a sensitivity matrix.

    Parameters
    ----------
    sens:
        Weighted log-parameter sensitivity matrix.
    rank_tol:
        Relative singular-value cut for the numerical rank.
    weak_ratio:
        Relative singular-value cut for reporting a direction as near-null.

    Returns
    -------
    IdentifiabilityReport

    Raises
    ------
    ValueError
        If either threshold is outside ``(0, 1]``, or if the sensitivity matrix
        is identically zero, in which case nothing at all is identifiable and
        the caller is told so rather than handed a report full of ``inf``.
    """
    for name, value in (("rank_tol", rank_tol), ("weak_ratio", weak_ratio)):
        if not np.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be finite and lie in (0, 1]")
    jw = sens.weighted
    u_s, sv, vt = np.linalg.svd(jw, full_matrices=False)
    del u_s
    s_max = float(sv[0]) if sv.size else 0.0
    if s_max <= 0.0:
        raise ValueError(
            "the sensitivity matrix is identically zero: no parameter affects "
            "the observations at this point, so no analysis is possible"
        )
    ratios = sv / s_max
    rank = int(np.count_nonzero(ratios >= rank_tol))
    s_min = float(sv[-1])
    cond = float("inf") if s_min <= 0.0 else s_max / s_min
    combos = tuple(
        Combination(
            index=i,
            singular_value=float(sv[i]),
            ratio=float(ratios[i]),
            vector=np.ascontiguousarray(vt[i]),
            parameter_names=sens.parameter_names,
        )
        for i in range(sv.size)
    )
    fisher = sens.fisher()
    deficient = rank < sens.n_parameters
    if deficient:
        cov = np.linalg.pinv(fisher, rcond=rank_tol)
    else:
        cov = np.linalg.inv(fisher)
    diag = np.diag(cov).astype(float)
    rel_sd = np.where(diag > 0.0, np.sqrt(np.maximum(diag, 0.0)), np.inf)
    if deficient:
        # Directions the data does not constrain have infinite variance, not the
        # finite value the pseudo-inverse reports. Mark every parameter with a
        # non-negligible component in a null direction.
        null_mask = ratios < rank_tol
        if null_mask.any():
            loading = np.sqrt((vt[null_mask] ** 2).sum(axis=0))
            rel_sd = np.where(loading > 1e-6, np.inf, rel_sd)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.sqrt(np.outer(diag, diag))
        corr = np.where(scale > 0.0, cov / scale, np.nan)
    np.fill_diagonal(corr, 1.0)
    return IdentifiabilityReport(
        parameter_names=sens.parameter_names,
        theta=sens.theta,
        singular_values=sv,
        rank=rank,
        condition_number=cond,
        rank_tol=float(rank_tol),
        weak_ratio=float(weak_ratio),
        combinations=combos,
        delta_msqr=sens.delta_msqr(),
        collinearity=collinearity_index(sens),
        relative_sd=rel_sd,
        correlation=corr,
        pseudo_inverse_used=bool(deficient),
    )
