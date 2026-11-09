"""Baseline 2: attribute the residual, but only inside the identifiable subspace.

Order of operations
-------------------
:mod:`twindiverge.identifiability` runs first and its verdict is binding. This
module will not return a per-parameter estimate for a parameter that baseline 1
refused, whatever the residual looks like. What it returns instead is the
estimate for the identifiable *combinations* of equation (11), and -- for a
caller who has an external reason to believe only one parameter moved -- a
separate, explicitly assumption-bearing single-parameter attribution with the
equivalence family it cannot distinguish printed beside it.

The three estimators here
-------------------------
Let ``A`` be the design matrix of equation (8), ``r`` the noise-scaled residual
of :func:`twindiverge.telemetry.residual_vector`, and
``A = U diag(sigma) V^T``.

**(a) Unconstrained generalised least squares on the identifiable subspace.**
The minimum-norm solution of ``A delta = r`` is

    equation (12)   delta_hat = V_r diag(1/sigma_r) U_r^T r

summing over the identifiable directions only. Its components along the null
space are set to zero by construction, and the honest way to read it is one
scalar per identifiable direction:

    equation (13)   q_hat_i = v_i^T delta_hat = (u_i^T r) / sigma_i,
                    sd(q_hat_i) = 1 / sigma_i

which is exact for a linear model with unit-variance scaled noise: ``u_i`` is a
unit vector and the scaled noise is isotropic, so ``u_i^T r`` has unit variance.
Each ``q_hat_i`` is reported with a Wald interval at the declared level and with
its z-score. This is generalised least squares rather than ordinary least
squares because the noise scaling of equation (8) is exactly the whitening
transform of a diagonal covariance; see Seber & Wild 1989, "Nonlinear
Regression", Wiley, chapter 2, for the weighted-least-squares form and its
covariance.

**(b) First-order sensitivity attribution.** The estimator the specification
names: assign to parameter ``j`` the share of the residual its own sensitivity
column explains,

    equation (14)   share_j = (a_j^T r)^2 / (|a_j|^2 |r|^2)

and attribute to the largest. It ignores the correlation between columns
entirely, which is why it is reported here beside (c) and not instead of it:
on this plant the sensitivity columns have pairwise correlations above 0.99 and
equation (14) is measured to be the weaker of the two. The measurement is in
``validation/validate_attribution_output.txt`` and the weaker figure is
reported beside the stronger one rather than dropped.

**(c) Constrained single-parameter maximum likelihood.** Under the assumption
that exactly one parameter moved, hypothesis ``j`` has the closed-form estimate
and residual

    equation (15)   delta_j = (a_j^T r) / (a_j^T a_j),
                    chi2_j = |r|^2 - (a_j^T r)^2 / (a_j^T a_j)

and the attribution is ``argmin_j chi2_j``. This is the strongest honest
analytic attributor on this population and it is the one the learned classifier
is benchmarked against. A confidence is attached as the normalised
profile-likelihood weight

    equation (16)   w_j = exp(-(chi2_j - min_k chi2_k) / 2) / sum_m exp(...)

which is a likelihood ratio between the five point hypotheses after each has
been maximised over its own amplitude. **It is not a posterior probability**:
the amplitudes were profiled out rather than integrated, so there is no prior
and no Occam factor. It is reported as a relative weight and named as such.

**The sparsity assumption is doing real work, and it is not testable from the
telemetry.** Under the ``rate_full`` set, the five single-parameter hypotheses
are distinguishable from each other -- their identifiable components differ --
and (c) therefore attains high accuracy. That accuracy is entirely conditional
on exactly one parameter having moved. Drop the assumption and the same data
cannot separate "J rose by 8 per cent" from "the other four each fell by 8 per
cent": the two differ by a pure scale shift, which equation (6) says the rate
channel cannot see. :meth:`Attribution.equivalence_family` constructs that set
explicitly, and ``validation/validate_attribution.py`` verifies the two
produce identical noiseless telemetry to within rounding.

Detection before attribution
----------------------------
A residual that is consistent with noise should be attributed to nothing. The
declared test is the chi-square goodness of fit of the null hypothesis
``delta = 0``: ``|r|^2`` has a chi-square distribution on ``rows`` degrees of
freedom under the null, and ``detected`` is true when it exceeds the
``DETECTION_LEVEL`` quantile. On the unidentifiable-subspace cases of
``validation/validate_learned_model.py`` this is what makes the analytic path
refuse rather than guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy import stats

from .identifiability import IdentifiabilityReport, combination_label
from .plant import PARAMETER_NAMES, Trajectory
from .telemetry import TelemetrySet, design_matrix

__all__ = [
    "CONFIDENCE_LEVEL",
    "DETECTION_LEVEL",
    "Attribution",
    "DirectionEstimate",
    "attribute",
]

#: Declared two-sided confidence level for every interval reported here.
CONFIDENCE_LEVEL: Final[float] = 0.95

#: Declared level of the chi-square detection test on ``delta = 0``.
DETECTION_LEVEL: Final[float] = 0.99


@dataclass(frozen=True)
class DirectionEstimate:
    """One identifiable direction of equation (13): what moved, and by how much."""

    label: str
    weights: np.ndarray
    estimate: float
    standard_deviation: float
    z_score: float
    half_width: float

    @property
    def interval(self) -> tuple[float, float]:
        """Two-sided Wald interval at :data:`CONFIDENCE_LEVEL`, in log-parameter units."""
        return (self.estimate - self.half_width, self.estimate + self.half_width)

    @property
    def significant(self) -> bool:
        """Whether the interval excludes zero.

        A thresholded boolean. Tests in this repository assert the z-score to
        tolerance and allow this flag to differ only for a direction whose
        z-score lies within tolerance of the critical value.
        """
        return abs(self.z_score) > stats.norm.ppf(0.5 + CONFIDENCE_LEVEL / 2.0)


@dataclass(frozen=True)
class Attribution:
    """The output of baseline 2 on one residual.

    ``parameter_estimate`` and ``parameter_half_width`` carry ``nan`` for every
    parameter baseline 1 refused. That is deliberate: a number there would be
    read as an answer.
    """

    telemetry_name: str
    identifiability: IdentifiabilityReport
    residual_chi2: float
    residual_rows: int
    detected: bool
    detection_threshold: float
    directions: tuple[DirectionEstimate, ...]
    fit_chi2: float
    parameter_estimate: np.ndarray
    parameter_half_width: np.ndarray
    refused: tuple[str, ...]
    sensitivity_share: np.ndarray
    sensitivity_choice: str
    constrained_chi2: np.ndarray
    constrained_estimate: np.ndarray
    constrained_choice: str
    constrained_weight: np.ndarray

    @property
    def constrained_confidence(self) -> float:
        """The relative likelihood weight of the chosen single-parameter hypothesis."""
        return float(self.constrained_weight.max())

    @property
    def combination_answer(self) -> tuple[str, ...]:
        """The identifiable directions that moved significantly, best determined first."""
        return tuple(
            f"{direction.label} moved by {direction.estimate:+.6f} "
            f"+/- {direction.half_width:.6f} (z = {direction.z_score:+.3f})"
            for direction in self.directions
            if direction.significant
        )

    def equivalence_family(self, parameter: str | None = None) -> np.ndarray:
        """Perturbations with identical noiseless telemetry to the chosen attribution.

        Returns an array of shape ``(k + 1, 5)`` whose first row is the chosen
        single-parameter perturbation and whose remaining rows are, for each
        null direction, the member of the family in which the chosen parameter
        is instead left at its nominal value. Every row produces the same
        noiseless telemetry as every other to within rounding, which is the
        content of equation (6) and is verified in
        ``validation/validate_attribution.py``.

        An empty extra set (shape ``(1, 5)``) means this telemetry set has no
        structural degeneracy and the attribution is unique.
        """
        name = self.constrained_choice if parameter is None else parameter
        if name not in PARAMETER_NAMES:
            raise ValueError(
                f"unknown parameter {name!r}; expected one of {PARAMETER_NAMES}"
            )
        index = PARAMETER_NAMES.index(name)
        chosen = np.zeros(5)
        chosen[index] = self.constrained_estimate[index]
        rows = [chosen]
        for direction in self.identifiability.null_basis:
            if abs(direction[index]) < 1.0e-12:
                continue
            # Subtract the multiple of this null direction that cancels the
            # chosen parameter's own component: a different physical story,
            # identical telemetry.
            rows.append(chosen - direction * (chosen[index] / direction[index]))
        return np.vstack(rows)


def attribute(
    telemetry: TelemetrySet,
    identifiability: IdentifiabilityReport,
    residual: np.ndarray,
    confidence_level: float = CONFIDENCE_LEVEL,
    detection_level: float = DETECTION_LEVEL,
    twin: Trajectory | None = None,
) -> Attribution:
    """Run baseline 2 on one noise-scaled residual.

    ``identifiability`` must be the report for the same telemetry set, which is
    checked. ``residual`` is in the row order of
    :func:`twindiverge.telemetry.design_matrix`.

    ``twin`` may be supplied to reuse an already integrated nominal trajectory
    carrying sensitivities. It does not change any number: the design matrix is
    a function of the nominal parameters and the campaign alone, and
    :func:`twindiverge.telemetry.design_matrix` checks that the trajectory
    belongs to the right campaign. Passing it is the difference between one
    integration per study and one per case.
    """
    if identifiability.telemetry_name != telemetry.name:
        raise ValueError(
            f"identifiability report is for {identifiability.telemetry_name!r} but the "
            f"telemetry set is {telemetry.name!r}"
        )
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must lie in (0, 1), got {confidence_level!r}")
    if not 0.0 < detection_level < 1.0:
        raise ValueError(f"detection_level must lie in (0, 1), got {detection_level!r}")
    vector = np.asarray(residual, dtype=float)
    if vector.shape != (telemetry.rows,):
        raise ValueError(
            f"telemetry set {telemetry.name!r} has {telemetry.rows} rows but the "
            f"residual has shape {vector.shape}"
        )

    matrix, _ = design_matrix(
        telemetry, identifiability_parameters(identifiability), trajectory=twin
    )
    left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    rank = identifiability.rank
    projections = left[:, :rank].T @ vector

    residual_chi2 = float(vector @ vector)
    threshold = float(stats.chi2.ppf(detection_level, telemetry.rows))
    critical = float(stats.norm.ppf(0.5 + confidence_level / 2.0))

    directions = []
    estimate_log = np.zeros(5)
    for index in range(rank):
        amplitude = float(projections[index] / singular_values[index])
        deviation = float(1.0 / singular_values[index])
        weights = right[index]
        estimate_log += amplitude * weights
        directions.append(
            DirectionEstimate(
                label=combination_label(weights),
                weights=weights,
                estimate=amplitude,
                standard_deviation=deviation,
                z_score=float(projections[index]),
                half_width=critical * deviation,
            )
        )
    fit_chi2 = float(residual_chi2 - projections @ projections)

    refused = identifiability.refused
    parameter_estimate = np.where(
        np.isfinite(identifiability.crb_relative), estimate_log, np.nan
    )
    parameter_half_width = np.where(
        np.isfinite(identifiability.crb_relative),
        critical * identifiability.crb_relative,
        np.nan,
    )

    # Equation (14): the specified first-order sensitivity attribution.
    column_norms = np.einsum("ij,ij->j", matrix, matrix)
    inner = matrix.T @ vector
    # A parameter the excitation never excites has a column of exact zeros --
    # k_t on the coast campaign, by the corollary to identity B -- so the
    # denominators below are made safe rather than guarded afterwards: a
    # numpy.where evaluates BOTH branches and would warn on the division it is
    # about to discard.
    safe_norms = np.where(column_norms > 0.0, column_norms, 1.0)
    denominator = safe_norms * max(residual_chi2, np.finfo(float).tiny)
    share = np.where(column_norms > 0.0, inner**2 / denominator, 0.0)
    sensitivity_choice = PARAMETER_NAMES[int(np.argmax(share))]

    # Equation (15): the constrained single-parameter maximum likelihood.
    amplitude = np.where(column_norms > 0.0, inner / safe_norms, 0.0)
    reduction = np.where(column_norms > 0.0, inner**2 / safe_norms, 0.0)
    constrained_chi2 = residual_chi2 - reduction
    # A parameter the excitation never excites explains nothing and must not win
    # by default; its chi2 equals the null chi2, which is the largest value.
    best = int(np.argmin(constrained_chi2))
    weight = np.exp(-0.5 * (constrained_chi2 - constrained_chi2[best]))
    weight = weight / weight.sum()

    return Attribution(
        telemetry_name=telemetry.name,
        identifiability=identifiability,
        residual_chi2=residual_chi2,
        residual_rows=telemetry.rows,
        detected=residual_chi2 > threshold,
        detection_threshold=threshold,
        directions=tuple(directions),
        fit_chi2=fit_chi2,
        parameter_estimate=parameter_estimate,
        parameter_half_width=parameter_half_width,
        refused=refused,
        sensitivity_share=share,
        sensitivity_choice=sensitivity_choice,
        constrained_chi2=constrained_chi2,
        constrained_estimate=amplitude,
        constrained_choice=PARAMETER_NAMES[best],
        constrained_weight=weight,
    )


def identifiability_parameters(report: IdentifiabilityReport):
    """The parameter point a report was built at, recovered for re-use.

    The report does not store the parameter object, because storing it would
    invite a caller to treat the report as self-contained when it is a local
    statement at one point. The nominal set is the only point this package
    linearises at, and :func:`attribute` re-derives the design matrix from it so
    that the matrix it uses is provably the one the report describes.
    """
    from .plant import NOMINAL

    _ = report
    return NOMINAL
