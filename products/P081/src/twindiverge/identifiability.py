"""Baseline 1: local structural and practical identifiability. The gate.

This module answers, and is the only thing in this package allowed to answer,
the question that has to be settled before any residual is attributed to
anything: **which parameters can the supplied telemetry set distinguish at
all?** Everything downstream -- the generalised-least-squares attribution of
:mod:`twindiverge.attribution`, the profile likelihoods of
:mod:`twindiverge.profile`, the learned classifier of
:mod:`twindiverge.learned` -- is either gated by this answer or is measured
against it.

Refusal is a first-class outcome here, not an error path. A parameter that the
telemetry cannot distinguish gets the verdict ``REFUSED`` and no estimate, and
the identifiable *combinations* are reported in its place.

What is computed, and from what
-------------------------------
Let ``A`` be the noise-scaled sensitivity matrix of equation (8) and
``A = U diag(sigma) V^T`` its singular value decomposition.

**Numerical rank.** The rank is the number of singular values with
``sigma_i / sigma_1 > RANK_RTOL``. The treatment of rank as a numerical rather
than algebraic property, and the use of a relative cutoff on the singular
values to decide it, is Golub & Van Loan 2013, "Matrix Computations", 4th ed.,
Johns Hopkins University Press, chapter 2 (the singular value decomposition
and numerical rank) and chapter 5 (least squares and conditioning). The cutoff
is declared, is the same for every telemetry set, and is never tuned per
result. Because a rank is a thresholded quantity, every report carries
``rank_margin_decades``: the base-10 distance from the nearest singular-value
ratio to the cutoff. A margin of 6 decades means the verdict would survive a
million-fold change in the cutoff; a margin near zero means the verdict is an
artefact of the cutoff and must be read as such.

**Fisher information.** ``F = A^T A``, equation (9), for a Gaussian likelihood
with independent errors of known standard deviation (Bard 1974, chapters 4 and
7). ``F`` is formed and reported, but **nothing here is computed from it**: its
condition number is the square of ``A``'s, and in the measured environment
``numpy.linalg.eigvalsh(F)`` returns a *negative* eigenvalue for the
``rate_full`` set -- a Fisher information matrix is positive semi-definite, so
that value is pure rounding from the squaring. Every quantity below is
computed from the singular values of ``A`` directly, which is the standard
reason for preferring the SVD to the normal equations (Golub & Van Loan 2013,
chapter 5). The negative eigenvalue is measured and reported in
``validation/validate_identifiability_output.txt`` rather than hidden.

**Cramer-Rao bounds.** Rao 1945, "Information and the accuracy attainable in
the estimation of statistical parameters", Bulletin of the Calcutta
Mathematical Society 37, 81-91; Cramer 1946, "Mathematical Methods of
Statistics", Princeton University Press. For a linear functional ``c^T p`` of
the logarithmic parameters, the bound on the variance of any unbiased
estimator is

    equation (10)   var(c^T p) >= c^T F^+ c,    **finite only if c lies in the
                    range of F**, i.e. only if ``c`` has no component in the
                    null space of ``A``.

That second clause is the whole product. It is exactly what a naive
implementation drops: ``numpy.linalg.pinv(F)`` has a finite diagonal for every
parameter, including the structurally unidentifiable ones, because the
pseudo-inverse silently returns the minimum-norm solution's variance instead of
the bound. This module computes the per-parameter bound as ``+inf`` whenever
``|P_null e_j|`` exceeds :data:`NULL_COMPONENT_TOL`, and reports the misleading
pinv diagonal beside it under the name ``pinv_diagonal_not_a_bound`` so the two
can be compared. ``validation/validate_negative_controls.py`` declares the
difference as a control and asserts that it fires.

**Identifiable combinations.** Each identifiable right singular vector ``v_i``
defines the scalar combination

    equation (11)   q_i = prod_j theta_j ** v_ij

(a weighted geometric mean of the parameters, because the parameterisation is
logarithmic) whose relative standard error is bounded below by ``1 / sigma_i``.
These are what is reported in place of the refused parameters: for the
``rate_full`` set the four identifiable combinations span exactly the
four-dimensional space of ratios ``k_t/J``, ``T_c/J``, ``b/J``, ``c_d/J``,
which is what equation (6) says is all the rate channel can see.

**Practical identifiability, separated from structural.** Raue, Kreutz,
Maiwald, Bachmann, Schilling, Klingmueller & Timmer 2009, "Structural and
practical identifiability analysis of partially observed dynamical models by
exploiting the profile likelihood", Bioinformatics 25(15), 1923-1929, draws
the distinction this module implements: structural unidentifiability is a
property of the model and the observation functions and is independent of the
data quality; practical unidentifiability is a property of the particular
record and its noise. Here a parameter is

- ``REFUSED`` when it has a component in the null space of ``A``: structurally
  unidentifiable from this telemetry set, no amount of data of this kind helps;
- ``WEAK`` when its bound is finite but its relative standard error exceeds
  :data:`PRACTICAL_LIMIT` (1.0, i.e. the record cannot pin the parameter to
  within its own magnitude): practically unidentifiable, and a longer or
  quieter record would fix it;
- ``IDENTIFIABLE`` otherwise.

``validation/validate_identifiability.py`` sweeps the record length and shows
the structural verdict constant while the practical verdict moves across four
orders of magnitude, which is the operational content of the distinction.

Validity and limits of this analysis
------------------------------------
Everything here is **local**: a first-order statement at the parameter point
the sensitivities were evaluated at. It is not global structural
identifiability in the sense of Bellman & Astrom 1970, "On structural
identifiability", Mathematical Biosciences 7(3-4), 329-339, which asks whether
two *distinct* parameter vectors can give identical output for every input;
nor is it a global sensitivity analysis in the sense of Saltelli, Ratto,
Andres, Campolongo, Cariboni, Gatelli, Saisana & Tarantola 2008, "Global
Sensitivity Analysis: The Primer", Wiley, which apportions output variance over
a whole parameter distribution. Neither is implemented here and neither is
claimed. For this plant the local result happens to coincide with the global
one along the scale direction, because equation (6) holds at every parameter
point and for every excitation -- that is a property of this plant, proved in
:mod:`twindiverge.plant`, and not a general licence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from .plant import PARAMETER_NAMES, SCALE_DIRECTION, WheelParameters
from .telemetry import TelemetrySet, design_matrix

__all__ = [
    "NULL_COMPONENT_TOL",
    "PRACTICAL_LIMIT",
    "RANK_RTOL",
    "VERDICTS",
    "IdentifiabilityReport",
    "analyse",
    "analytic_null_basis",
    "combination_label",
    "subspace_angles_deg",
]

#: Declared relative singular-value cutoff for the numerical rank. Golub & Van
#: Loan 2013, chapter 2. One value for every telemetry set, never tuned per
#: result. 1e-10 sits roughly midway on a logarithmic scale between the
#: measured machine-precision null ratios (around 1e-16) and the smallest
#: genuinely non-zero ratio measured on any declared set (2.6e-6), so every
#: declared verdict has several decades of margin; the margin is reported.
RANK_RTOL: Final[float] = 1.0e-10

#: Declared tolerance on the norm of a parameter's component in the null space,
#: above which its Cramer-Rao bound is infinite by equation (10). The null-space
#: basis is orthonormal to within rounding, so a parameter genuinely outside the
#: null space has a component at the 1e-16 level and one inside it has a
#: component of order 1/sqrt(5) = 0.447; there is no population between.
NULL_COMPONENT_TOL: Final[float] = 1.0e-6

#: Declared relative standard error above which a structurally identifiable
#: parameter is called practically unidentifiable. 1.0 means "the record cannot
#: pin the parameter to within its own magnitude", which is the weakest claim
#: worth making about a parameter at all.
PRACTICAL_LIMIT: Final[float] = 1.0

#: The three verdicts, in increasing order of what the telemetry supports.
VERDICTS: Final[tuple[str, str, str]] = ("REFUSED", "WEAK", "IDENTIFIABLE")


def combination_label(weights: np.ndarray, digits: int = 3) -> str:
    """Render equation (11) as a readable product of powers.

    ``combination_label([0.5, -0.5, 0, 0, 0])`` gives ``J^+0.5 k_t^-0.5``.
    Components below 1e-3 in magnitude are dropped, and the dropped total is
    not reported here: the caller has the full weight vector.
    """
    array = np.asarray(weights, dtype=float)
    if array.shape != (5,):
        raise ValueError(f"expected 5 weights, got shape {array.shape}")
    parts = [
        f"{name}^{value:+.{digits}f}"
        for name, value in zip(PARAMETER_NAMES, array, strict=True)
        if abs(value) >= 1.0e-3
    ]
    return " ".join(parts) if parts else "(empty combination)"


def analytic_null_basis(telemetry: TelemetrySet) -> np.ndarray:
    """The null space derived on paper in :mod:`twindiverge.plant`, shape ``(k, 5)``.

    This is the **independently derived reference** for the numerical null
    space of :func:`analyse`. It is algebra, from identities A and B and their
    corollary; it never touches a sensitivity matrix, and the agreement of the
    two is one of this product's validation checks
    (``validation/validate_identifiability.py``, part 2).

    - A rate-only set has the exact scale direction of equation (6) in its null
      space whatever the excitation.
    - A set that includes the reaction-torque channel does not, by equation (7),
      wherever the net torque is non-zero somewhere in the record.
    - Any set recorded under zero current has ``e_{k_t}`` in its null space, by
      the corollary: a parameter the excitation never excites.

    The rows returned are orthonormalised, so they are a basis and not the
    particular vectors the derivation produced.
    """
    rows: list[np.ndarray] = []
    undriven = telemetry.campaign.drive_current_a == 0.0
    if undriven:
        unexcited = np.zeros(5)
        unexcited[PARAMETER_NAMES.index("k_t")] = 1.0
        rows.append(unexcited)
    if "torque" not in telemetry.channels:
        rows.append(SCALE_DIRECTION.copy())
    if not rows:
        return np.zeros((0, 5))
    stacked = np.vstack(rows)
    # Orthonormalise. The two vectors above are not orthogonal when both are
    # present, and a basis is what the caller is promised.
    basis, _ = np.linalg.qr(stacked.T)
    return basis.T[: stacked.shape[0]]


def subspace_angles_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Principal angles between two row-spanned subspaces, in degrees, ascending.

    Returns an empty array when either subspace is empty. Both arguments are
    orthonormalised internally, so they need only span the subspace.
    ``numpy.linalg.svd`` of the inner-product matrix gives the cosines; the
    arccos is clipped to ``[-1, 1]`` because a cosine computed as a dot product
    of floating-point vectors can exceed 1 by one unit in the last place.
    """
    left = np.atleast_2d(np.asarray(first, dtype=float))
    right = np.atleast_2d(np.asarray(second, dtype=float))
    if left.size == 0 or right.size == 0:
        return np.zeros(0)

    def orthonormal(rows: np.ndarray) -> np.ndarray:
        basis, _ = np.linalg.qr(rows.T)
        return basis[:, : rows.shape[0]]

    cosines = np.linalg.svd(orthonormal(left).T @ orthonormal(right), compute_uv=False)
    return np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))[::-1]


@dataclass(frozen=True)
class IdentifiabilityReport:
    """Everything baseline 1 has to say about one telemetry set.

    Attributes carrying a bound are relative: a ``crb_relative`` of 0.05 means
    the Cramer-Rao bound on the standard error of ``log(theta_j)`` is 0.05, so
    the parameter cannot be determined better than about 5 per cent whatever
    the estimator.
    """

    telemetry_name: str
    singular_values: np.ndarray
    rank: int
    rank_rtol: float
    rank_margin_decades: float
    null_basis: np.ndarray
    identifiable_basis: np.ndarray
    direction_crb_relative: np.ndarray
    crb_relative: np.ndarray
    pinv_diagonal_not_a_bound: np.ndarray
    null_component: np.ndarray
    verdict: tuple[str, ...]
    condition_number: float
    fisher_condition_number: float
    fisher_min_eigenvalue: float

    @property
    def refused(self) -> tuple[str, ...]:
        """Names of the structurally unidentifiable parameters."""
        return tuple(
            name
            for name, verdict in zip(PARAMETER_NAMES, self.verdict, strict=True)
            if verdict == "REFUSED"
        )

    @property
    def weak(self) -> tuple[str, ...]:
        """Names of the practically unidentifiable but structurally sound parameters."""
        return tuple(
            name
            for name, verdict in zip(PARAMETER_NAMES, self.verdict, strict=True)
            if verdict == "WEAK"
        )

    @property
    def identifiable(self) -> tuple[str, ...]:
        """Names of the parameters this telemetry set supports an estimate for."""
        return tuple(
            name
            for name, verdict in zip(PARAMETER_NAMES, self.verdict, strict=True)
            if verdict == "IDENTIFIABLE"
        )

    def combinations(self) -> tuple[tuple[str, float], ...]:
        """The identifiable combinations of equation (11) with their relative bounds.

        One entry per identifiable direction, best determined first: the
        rendered combination and the Cramer-Rao bound on its relative standard
        error.
        """
        return tuple(
            (combination_label(self.identifiable_basis[index]),
             float(self.direction_crb_relative[index]))
            for index in range(self.rank)
        )

    def projector_null(self) -> np.ndarray:
        """Orthogonal projector onto the null space, shape ``(5, 5)``."""
        if self.null_basis.size == 0:
            return np.zeros((5, 5))
        return self.null_basis.T @ self.null_basis

    def equivalent_perturbation(self, delta_log: np.ndarray) -> np.ndarray:
        """The part of ``delta_log`` this telemetry set cannot see.

        The projection of a perturbation onto the null space. Two perturbations
        differing only by this part produce identical noiseless telemetry, which
        is what makes attribution ill-posed and is verified directly in
        ``validation/validate_identifiability.py``, part 3.
        """
        array = np.asarray(delta_log, dtype=float)
        if array.shape != (5,):
            raise ValueError(f"expected 5 components, got shape {array.shape}")
        return self.projector_null() @ array


def analyse(
    telemetry: TelemetrySet,
    parameters: WheelParameters,
    rank_rtol: float = RANK_RTOL,
    practical_limit: float = PRACTICAL_LIMIT,
) -> IdentifiabilityReport:
    """Run baseline 1 on one telemetry set at one parameter point.

    ``rank_rtol`` and ``practical_limit`` are exposed so that
    ``validation/validate_identifiability.py`` can show the verdicts' sensitivity
    to them, and are defaulted to the declared values everywhere else.
    """
    if not 0.0 < rank_rtol < 1.0:
        raise ValueError(f"rank_rtol must lie in (0, 1), got {rank_rtol!r}")
    if practical_limit <= 0.0:
        raise ValueError(f"practical_limit must be strictly positive, got {practical_limit!r}")
    matrix, _ = design_matrix(telemetry, parameters)
    _, singular_values, right = np.linalg.svd(matrix, full_matrices=True)
    ratios = singular_values / singular_values[0]
    rank = int(np.count_nonzero(ratios > rank_rtol))
    if rank == 0:  # pragma: no cover - unreachable: ratios[0] is exactly 1.0
        raise ValueError(
            f"telemetry set {telemetry.name!r} has numerical rank 0 at this parameter "
            "point: no parameter has any measurable effect, which means the campaign "
            "or the noise levels are misconfigured rather than that the model is "
            "unidentifiable"
        )
    # Distance, in decades, from the cutoff to the nearest ratio on either side.
    # A thresholded verdict is only as good as this number.
    margins = np.abs(np.log10(np.maximum(ratios, np.finfo(float).tiny)) - np.log10(rank_rtol))
    rank_margin_decades = float(margins.min())

    identifiable_basis = right[:rank]
    null_basis = right[rank:]
    direction_crb = 1.0 / singular_values[:rank]

    # Equation (10), with its finiteness clause. The pseudo-inverse diagonal is
    # computed too, under a name that says what it is not.
    pinv_fisher = (identifiable_basis.T * direction_crb**2) @ identifiable_basis
    pinv_diagonal = np.sqrt(np.maximum(np.diag(pinv_fisher), 0.0))
    if null_basis.size:
        null_component = np.linalg.norm(null_basis, axis=0)
    else:
        null_component = np.zeros(5)
    crb = np.where(null_component > NULL_COMPONENT_TOL, np.inf, pinv_diagonal)

    verdict = tuple(
        "REFUSED"
        if null_component[index] > NULL_COMPONENT_TOL
        else ("WEAK" if crb[index] > practical_limit else "IDENTIFIABLE")
        for index in range(5)
    )

    fisher = matrix.T @ matrix
    eigenvalues = np.linalg.eigvalsh(fisher)
    return IdentifiabilityReport(
        telemetry_name=telemetry.name,
        singular_values=singular_values,
        rank=rank,
        rank_rtol=float(rank_rtol),
        rank_margin_decades=rank_margin_decades,
        null_basis=null_basis,
        identifiable_basis=identifiable_basis,
        direction_crb_relative=direction_crb,
        crb_relative=crb,
        pinv_diagonal_not_a_bound=pinv_diagonal,
        null_component=null_component,
        verdict=verdict,
        condition_number=float(singular_values[0] / singular_values[rank - 1]),
        fisher_condition_number=float(np.linalg.cond(fisher)),
        fisher_min_eigenvalue=float(eigenvalues.min()),
    )
