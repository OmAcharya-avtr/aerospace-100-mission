"""Baseline 1: the batch solution over the whole record, on the observable subspace only.

This is the analytic estimator and it is implemented and validated before the
learned one. It takes an :class:`aligncal.observability.ObservabilityReport` as
a **required** argument so that the verdict cannot be produced after the
estimate it gates.

The weighted least-squares solution
-----------------------------------
Minimising the weighted residual of equation (7) over the record,

    (14)  J(theta) = sum_t | r_t - [w_t]x theta |^2 / s_j^2

gives the normal equations ``G_j theta = b_j`` with ``G_j`` the Gramian (8) and

    (15)  b_j = (1 / s_j^2) sum_t [w_t]x^T r_t = (1 / s_j^2) sum_t ( r_t x w_t )

using ``[a]x^T = -[a]x``. The solution is taken on the retained subspace only:

    (16)  theta_hat = sum_{i in observable} ( v_i^T b_j / lambda_i ) v_i

so a degree of freedom the record cannot support is returned as **exactly
zero and flagged as withheld**, not as a large number with a large covariance.
Its covariance entry is ``inf``.

(14) is Wahba's problem, linearised. Wahba, G. (1965), *SIAM Review* 7(3), 409,
poses the minimisation of ``sum w_i |b_i - R r_i|^2`` over rotations; expanding
``R = exp(-[theta]x) ~ I - [theta]x`` about the identity turns it into (14),
and the matrix of the resulting normal equations is ``sum w_i (|r_i|^2 I - r_i
r_i^T)``, the same Gramian. The iteration of :func:`solve` is Gauss-Newton on
the unlinearised problem and therefore does not carry the ``O(|theta|^2)``
truncation of a single linear pass.

Independent cross-check: the SVD solution
-----------------------------------------
:func:`wahba_svd` solves the same problem by a route that shares no arithmetic
with the 3x3 normal equations: form the attitude profile matrix ``B = sum_t
y_t w_t^T``, take ``B = U S V^T``, and set

    (17)  D = U diag(1, 1, det(U) det(V)) V^T

which is the optimal rotation. Source: Markley, F. L. (1988), Attitude
determination using vector observations and the singular value decomposition,
*Journal of the Astronautical Sciences* 36(3), 245-258. The ``det`` factor
excludes the reflection that minimises (14) over O(3) rather than SO(3).

(17) is an unconstrained solution over all of SO(3) and therefore has **no
mechanism to withhold an unobservable degree of freedom**. On a rank-deficient
record its answer in the unobservable direction is whatever the gauge of a
singular-value decomposition with a repeated or zero singular value happens to
return. That is measured in ``validation/validate_unobservable.py`` and it is
the analytic illustration of the same failure the learned estimator shows.

A regularised variant, reported beside the specified baseline
-------------------------------------------------------------
:func:`solve_map` adds a zero-mean Gaussian prior of standard deviation
``prior_sigma`` on every component and solves

    (18)  theta_hat = ( G_j + I / prior_sigma^2 )^-1 b_j

over the full space. This is the maximum-a-posteriori estimator and it is a
*biased* estimator, so it can and does fall below the Cramer-Rao bound of (12),
which applies to unbiased estimators only. It is included because the mission's
build guide requires the strongest honest baseline to be used and the specified
one reported beside it: a learned estimator trained on records whose
misalignments are drawn from a known distribution is performing exactly this
shrinkage, and comparing it only against the unbiased solution would overstate
its contribution. (18) still has no mechanism to refuse: in the unobservable
directions it returns the prior mean, which is zero, with the prior variance
unchanged, and it reports that it has done so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import rotation_from_vector, rotation_to_vector, skew
from .observability import ObservabilityReport, SensorObservability
from .record import AlignmentRecord, residuals

__all__ = [
    "DEFAULT_MAX_STEP_RAD",
    "BatchSolution",
    "normal_equation_rhs",
    "solve",
    "solve_map",
    "wahba_svd",
]


#: Trust region on one Gauss-Newton step, radians. The linearisation (20) of
#: :mod:`aligncal.recursive` and the one behind (14) are both ``O(|step|^2)``
#: accurate, so an unlimited Newton step on a low-information record can leave
#: the region where the model holds and oscillate. 0.2 rad is a declared policy
#: choice; with it the iteration converged on every record measured in
#: ``validation/validate_recovery_grid.py`` and without it on 92.2 per cent of a
#: 600-record short-record population. Both figures are in VALIDATION.md.
DEFAULT_MAX_STEP_RAD: float = 0.2


def _limit(step: np.ndarray, max_step_rad: float) -> np.ndarray:
    """Scale ``step`` down to ``max_step_rad`` in norm, leaving its direction."""
    norm = float(np.linalg.norm(step))
    if norm <= max_step_rad or norm == 0.0:
        return step
    return step * (max_step_rad / norm)


def normal_equation_rhs(record: AlignmentRecord, name: str) -> np.ndarray:
    """``b_j`` of equation (15) for secondary ``name``. Units 1/(s rad), shape ``(3,)``."""
    rates = record.reference_rates
    residual = residuals(record, name)
    weight = 1.0 / (record.pair_sigma(name) ** 2)
    return weight * np.sum(np.cross(residual, rates), axis=0)


@dataclass(frozen=True)
class BatchSolution:
    """The batch estimate for one secondary, with its observability verdict first.

    Attributes
    ----------
    observability:
        The verdict that gated this estimate. First field, deliberately.
    name:
        Secondary sensor name.
    theta_rad:
        Estimated misalignment vector in the reference frame, radians. Exactly
        zero in every withheld direction.
    withheld:
        Degree-of-freedom indices that were not estimated, with their
        directions available from ``observability.degrees_of_freedom``.
    covariance:
        ``(3, 3)`` estimate covariance in rad^2, equal to ``G^+`` on the
        retained subspace. ``inf`` on the diagonal of a withheld direction when
        expressed in the degree-of-freedom basis; in the body frame the matrix
        is the retained-subspace pseudo-inverse and does **not** claim a finite
        variance for a withheld direction, which is what
        :meth:`sigma_rad_in_direction` is for.
    iterations:
        Gauss-Newton iterations actually taken.
    converged:
        Whether the update fell below ``tol`` before ``max_iterations``.
    residual_rms_before, residual_rms_after:
        Weighted residual RMS of (14) in rad/s, before and after the fit.
    """

    observability: SensorObservability
    name: str
    theta_rad: np.ndarray
    withheld: tuple[int, ...]
    covariance: np.ndarray
    iterations: int
    converged: bool
    residual_rms_before: float
    residual_rms_after: float

    @property
    def theta_mrad(self) -> np.ndarray:
        """The estimate in milliradians."""
        return self.theta_rad * 1.0e3

    @property
    def magnitude_arcsec(self) -> float:
        """Magnitude of the estimated misalignment in arcseconds."""
        return float(np.linalg.norm(self.theta_rad)) * 206264.806247

    @property
    def rotation(self) -> np.ndarray:
        """``exp(-[theta]x)``: the correction that takes the declared mounting to the fitted one."""
        return rotation_from_vector(-self.theta_rad)

    def sigma_rad_in_direction(self, direction: np.ndarray) -> float:
        """One-sigma of the estimate along ``direction``, radians.

        ``inf`` when ``direction`` has any projection onto a withheld degree of
        freedom, because the estimate carries no information there at all.
        """
        unit = np.asarray(direction, dtype=float).reshape(3)
        norm = float(np.linalg.norm(unit))
        if norm == 0.0:
            raise ValueError("sigma_rad_in_direction() needs a non-zero direction")
        unit = unit / norm
        basis = self.observability.basis
        mask = self.observability.observable_mask
        projection = basis.T @ unit
        if np.any(np.abs(projection[~mask]) > 1.0e-12):
            return float("inf")
        values = np.array([d.eigenvalue for d in self.observability.degrees_of_freedom])
        if not np.any(mask):
            return float("inf")
        return float(np.sqrt(np.sum(projection[mask] ** 2 / values[mask])))

    def summary(self) -> list[str]:
        """Verdict first, then the estimate, then what was withheld."""
        lines = list(self.observability.summary())
        lines.append(f"BATCH ESTIMATE for {self.name} (equations 14-16)")
        lines.append(
            "  theta = ["
            + ", ".join(f"{c:+.6f}" for c in self.theta_mrad)
            + f"] mrad, magnitude {self.magnitude_arcsec:.3f} arcsec"
        )
        for dof in self.observability.degrees_of_freedom:
            component = float(np.dot(self.theta_rad, dof.direction))
            if dof.index in self.withheld:
                lines.append(
                    f"    dof {dof.index} ({dof.verdict}): WITHHELD, not estimated -- "
                    f"the record does not support it; returned exactly 0, "
                    f"reported component {component:+.3e} rad"
                )
            else:
                sigma = self.sigma_rad_in_direction(dof.direction)
                lines.append(
                    f"    dof {dof.index} ({dof.verdict}): {component * 206264.806247:+.3f} "
                    f"+/- {sigma * 206264.806247:.3f} arcsec (1 sigma, CRB)"
                )
        lines.append(
            f"  Gauss-Newton iterations {self.iterations}, converged {self.converged}; "
            f"weighted residual RMS {self.residual_rms_before:.6g} -> "
            f"{self.residual_rms_after:.6g} rad/s"
        )
        return lines


def _retained(observability: SensorObservability) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = observability.basis
    mask = observability.observable_mask
    values = np.array([d.eigenvalue for d in observability.degrees_of_freedom], dtype=float)
    return basis, mask, values


def _weighted_rms(record: AlignmentRecord, name: str, correction: np.ndarray) -> float:
    rates = record.reference_rates
    predicted = rates @ correction.T
    difference = record.derotated(name) - predicted
    return float(np.sqrt(np.mean(np.sum(difference * difference, axis=1)))) / 1.0


def solve(
    record: AlignmentRecord,
    observability: ObservabilityReport,
    name: str,
    *,
    max_iterations: int = 40,
    tol: float = 1.0e-12,
    max_step_rad: float = DEFAULT_MAX_STEP_RAD,
) -> BatchSolution:
    """Batch least-squares misalignment of secondary ``name``, equations (14)-(16).

    Parameters
    ----------
    record:
        The measurements.
    observability:
        The verdict from :func:`aligncal.observability.assess`, computed first.
        Required, not optional: the subspace it names is what this function
        solves on.
    name:
        Secondary sensor name.
    max_iterations, tol:
        Gauss-Newton stopping rule. ``tol`` is an absolute bound in radians on
        the update norm.
    max_step_rad:
        Trust region on one step, radians. See :data:`DEFAULT_MAX_STEP_RAD`.

    The default of 40 iterations is measured, not guessed. At 24 iterations one
    record in 864 of the grid in ``validation/validate_recovery_grid.py`` -- an
    ill-conditioned ``near_single_axis`` record at 150 samples, Gramian
    condition number 39266 -- reached the 1e-12 rad step bound on its 25th
    iteration. Its answer at 24 iterations already differed from the converged
    one by 4.0e-10 mrad, so the bound rather than the answer was the problem;
    the stopping tolerance was left alone and the iteration budget raised.

    Returns
    -------
    BatchSolution
        With ``observability`` as its first field and exact zeros in every
        withheld direction.
    """
    if not np.isfinite(max_step_rad) or max_step_rad <= 0.0:
        raise ValueError(f"max_step_rad must be finite and > 0 rad, got {max_step_rad!r}")
    if int(max_iterations) < 1:
        raise ValueError(f"max_iterations must be at least 1, got {max_iterations!r}")
    sensor = observability[name]
    basis, mask, values = _retained(sensor)
    withheld = tuple(int(i) for i in np.flatnonzero(~mask))
    weight = 1.0 / (record.pair_sigma(name) ** 2)
    rates = record.reference_rates
    measured = record.derotated(name)

    before = _weighted_rms(record, name, np.eye(3))
    correction = np.eye(3)
    iterations = 0
    converged = False
    for step_index in range(1, int(max_iterations) + 1):
        iterations = step_index
        predicted = rates @ correction.T
        innovation = measured - predicted
        rhs = weight * np.sum(np.cross(innovation, predicted), axis=0)
        projected = basis.T @ rhs
        step = np.zeros(3)
        if np.any(mask):
            step = _limit(basis[:, mask] @ (projected[mask] / values[mask]), max_step_rad)
        correction = rotation_from_vector(-step) @ correction
        if float(np.linalg.norm(step)) <= tol:
            converged = True
            break

    theta = -rotation_to_vector(correction)
    # Enforce the exact zero in the withheld directions. Gauss-Newton updates
    # are built from the retained basis only, but the matrix logarithm of a
    # product of such updates can leak a rounding-level component.
    if withheld:
        theta = basis[:, mask] @ (basis[:, mask].T @ theta) if np.any(mask) else np.zeros(3)
        correction = rotation_from_vector(-theta)
    after = _weighted_rms(record, name, correction)

    covariance = np.zeros((3, 3))
    if np.any(mask):
        retained = basis[:, mask]
        covariance = retained @ np.diag(1.0 / values[mask]) @ retained.T
    return BatchSolution(
        observability=sensor,
        name=name,
        theta_rad=theta,
        withheld=withheld,
        covariance=covariance,
        iterations=int(iterations),
        converged=bool(converged),
        residual_rms_before=before,
        residual_rms_after=after,
    )


def solve_map(
    record: AlignmentRecord,
    observability: ObservabilityReport,
    name: str,
    *,
    prior_sigma_rad: float,
    max_iterations: int = 40,
    tol: float = 1.0e-12,
    max_step_rad: float = DEFAULT_MAX_STEP_RAD,
) -> BatchSolution:
    """Maximum-a-posteriori misalignment with a zero-mean prior, equation (18).

    A biased estimator. It solves over the **full** space, so nothing is
    withheld and ``withheld`` is empty; in the unobservable directions it
    returns the prior mean, which is zero, and its covariance there is the
    prior variance. ``prior_sigma_rad`` is the prior one-sigma per component in
    radians and must be finite and positive.

    This estimator can fall below the Cramer-Rao bound of (12), which bounds
    unbiased estimators only. Where it does, its bias is reported alongside, in
    ``validation/validate_recovery_grid.py``.
    """
    if not np.isfinite(prior_sigma_rad) or prior_sigma_rad <= 0.0:
        raise ValueError(f"prior_sigma_rad must be finite and > 0 rad, got {prior_sigma_rad!r}")
    if not np.isfinite(max_step_rad) or max_step_rad <= 0.0:
        raise ValueError(f"max_step_rad must be finite and > 0 rad, got {max_step_rad!r}")
    if int(max_iterations) < 1:
        raise ValueError(f"max_iterations must be at least 1, got {max_iterations!r}")
    sensor = observability[name]
    weight = 1.0 / (record.pair_sigma(name) ** 2)
    rates = record.reference_rates
    measured = record.derotated(name)
    precision = sensor.gramian + np.eye(3) / (prior_sigma_rad * prior_sigma_rad)

    before = _weighted_rms(record, name, np.eye(3))
    correction = np.eye(3)
    theta = np.zeros(3)
    iterations = 0
    converged = False
    for step_index in range(1, int(max_iterations) + 1):
        iterations = step_index
        predicted = rates @ correction.T
        innovation = measured - predicted
        rhs = weight * np.sum(np.cross(innovation, predicted), axis=0)
        # The prior pulls on the total estimate, not on the increment.
        rhs = rhs - theta / (prior_sigma_rad * prior_sigma_rad)
        step = _limit(np.linalg.solve(precision, rhs), max_step_rad)
        correction = rotation_from_vector(-step) @ correction
        theta = -rotation_to_vector(correction)
        if float(np.linalg.norm(step)) <= tol:
            converged = True
            break
    after = _weighted_rms(record, name, correction)
    return BatchSolution(
        observability=sensor,
        name=name,
        theta_rad=theta,
        withheld=(),
        covariance=np.linalg.inv(precision),
        iterations=int(iterations),
        converged=bool(converged),
        residual_rms_before=before,
        residual_rms_after=after,
    )


def wahba_svd(record: AlignmentRecord, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Markley's SVD solution of equation (17): an independent cross-check.

    Returns ``(theta_rad, singular_values)``. ``theta_rad`` is the misalignment
    vector in the reference frame; ``singular_values`` are those of the
    attitude profile matrix ``B``, descending, in rad^2/s^2. A zero or
    near-zero smallest singular value is the rank deficiency, and in that case
    the returned ``theta_rad`` component along the corresponding direction is
    determined by the decomposition's gauge rather than by the data. This
    function deliberately has no mechanism to withhold it, and that is the
    point of the comparison.
    """
    rates = record.reference_rates
    measured = record.derotated(name)
    profile = measured.T @ rates
    left, singular, right_t = np.linalg.svd(profile)
    sign = float(np.linalg.det(left) * np.linalg.det(right_t))
    rotation = left @ np.diag([1.0, 1.0, sign]) @ right_t
    return -rotation_to_vector(rotation), singular


def predicted_residual(record: AlignmentRecord, theta_rad: np.ndarray) -> np.ndarray:
    """``[w_A]x theta`` of (7) for a given misalignment, rad/s, shape ``(n, 3)``.

    Used by the examples and by the residual checks in ``validation/``.
    """
    vector = np.asarray(theta_rad, dtype=float).reshape(3)
    return skew(record.reference_rates) @ vector
