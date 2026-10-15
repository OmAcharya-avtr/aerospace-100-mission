"""How much of a residual model's improvement is real, and how much is it
absorbing an identifiability failure.

This module exists because the honest risk of a learned discrepancy term is not
that it fails. It is that it succeeds, on the metric it is scored on, by
compensating for parameter error in directions the data cannot constrain --
leaving the predictions better and the physical parameters worse. Brynjarsdottir
and O'Hagan (2014) state the problem; this module measures it, three ways, on
held-out data.

Measurement 1: subspace decomposition of the learned residual
-------------------------------------------------------------
At a parameter point ``theta_hat`` the weighted log-sensitivity matrix of the
*held-out* design has the singular value decomposition ``J_w = U S V^T``. Its
left singular vectors split the observation space into three orthogonal parts:

* ``strong`` -- the span of the left singular vectors whose ``s_i/s_max`` is at
  or above a stated cut. A change in the data that lies here could have been
  produced by changing the parameters, *and* the data constrains that change
  well. If the learned residual lives here, the physics fit has not converged or
  the design has shifted; it is not new structure.
* ``weak`` -- the span of the left singular vectors below that cut but inside
  the matrix's column space. A change here could also have been produced by
  moving the parameters, but along a direction the data barely sees. **This is
  the absorption term.** A residual model with a large weight here is silently
  standing in for parameters that are not identifiable.
* ``orthogonal`` -- everything outside the column space of ``J_w``. No parameter
  value can produce a change here. This is the only part that is unambiguously
  model-form error, and it is the part a discrepancy model is legitimately for.

The report gives the three energy fractions of the learned residual field,
``||P r||**2 / ||r||**2``, and they sum to 1 by construction -- a property the
test suite checks.

Measurement 2: parameter drift under a discrepancy term
-------------------------------------------------------
Refit the physics with the learned discrepancy subtracted from the data, and
compare the parameters with the physics-only fit and with the truth. If the
discrepancy model is absorbing, the parameters move; whether they move *towards*
or *away from* the truth is the question, and on synthetic data it can be
answered. The report gives the root-mean-square relative parameter error before
and after, and the per-parameter movement.

Measurement 3: improvement where there is nothing to learn
----------------------------------------------------------
Run the same benchmark on a design whose model-form discrepancy is far below the
measurement noise. Any improvement there cannot be real structure, so it is a
direct read on how much the method flatters itself. The shipped
``lumped-nodes`` design has a discrepancy of about 0.06 K against 0.3 K noise
and is used for exactly this.

References
----------
Brynjarsdottir, J. and O'Hagan, A. (2014), "Learning about physical parameters:
the importance of model discrepancy", *Inverse Problems* 30(11), 114007.

Kennedy, M. C. and O'Hagan, A. (2001), *Journal of the Royal Statistical Society
Series B* 63(3), 425-464.

Golub, G. H. and Van Loan, C. F., *Matrix Computations*, 4th ed., 2013, section
2.4: the four fundamental subspaces of a matrix from its SVD, which is what the
decomposition above is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .fitting import CalibrationProblem, FitResult, fit
from .sensitivity import SensitivityMatrix

__all__ = [
    "DEFAULT_STRONG_CUT",
    "DriftReport",
    "SubspaceSplit",
    "decompose_residual",
    "parameter_drift",
]

DEFAULT_STRONG_CUT: float = 1e-2
"""Relative singular-value cut separating the strong and weak subspaces.

A direction with ``s_i/s_max`` below 1e-2 is seen by the data a hundred times
less well than the best-determined one. The cut is a reporting choice and the
whole spectrum is always printed beside the decomposition, so a reader can move
it; ``validation/validate_residual_output.txt`` repeats the decomposition at
three cuts (1e-1, 1e-2, 1e-3) so that the sensitivity of the conclusion to the
cut is visible rather than assumed away.
"""


@dataclass(frozen=True)
class SubspaceSplit:
    """Energy of a residual field in the three sensitivity subspaces.

    Attributes
    ----------
    total_energy:
        ``||r_w||**2`` of the weighted residual field, dimensionless.
    strong_energy, weak_energy, orthogonal_energy:
        Its squared projections, dimensionless. They sum to ``total_energy`` up
        to rounding.
    strong_fraction, weak_fraction, orthogonal_fraction:
        The same, divided by ``total_energy``. These sum to 1.
    strong_cut:
        The relative singular-value cut that was used.
    n_strong, n_weak:
        Dimensions of the two in-column-space subspaces.
    singular_ratios:
        ``s_i/s_max`` of the sensitivity matrix, for reference.
    """

    total_energy: float
    strong_energy: float
    weak_energy: float
    orthogonal_energy: float
    strong_fraction: float
    weak_fraction: float
    orthogonal_fraction: float
    strong_cut: float
    n_strong: int
    n_weak: int
    singular_ratios: np.ndarray

    def summary(self) -> str:
        """One block of text, including what each fraction means."""
        return "\n".join(
            [
                f"subspace decomposition of the learned residual field "
                f"(cut s/s_max = {self.strong_cut:g})",
                f"  total weighted energy        : {self.total_energy:.6g}",
                f"  strong-sensitivity subspace  : {self.strong_fraction:.6f}  "
                f"({self.n_strong} directions) -- the parameters could have done "
                "this and the data would have seen it",
                f"  weak-sensitivity subspace    : {self.weak_fraction:.6f}  "
                f"({self.n_weak} directions) -- ABSORPTION: the parameters could "
                "have done this and the data cannot tell",
                f"  orthogonal to all parameters : {self.orthogonal_fraction:.6f}"
                "  -- genuine model-form error, which no parameter value can "
                "reproduce",
            ]
        )


def decompose_residual(
    sens: SensitivityMatrix,
    residual: Sequence[float],
    strong_cut: float = DEFAULT_STRONG_CUT,
) -> SubspaceSplit:
    """Split a residual field into the sensitivity subspaces.

    Parameters
    ----------
    sens:
        Weighted log-sensitivity matrix of the design the residual belongs to,
        evaluated at the fitted parameters.
    residual:
        Residual field in K, shape ``(n_obs,)``. It is divided by the same
        ``sigma`` the sensitivity matrix was weighted with before projection, so
        that both live in the same (dimensionless) space.
    strong_cut:
        Relative singular-value cut.

    Returns
    -------
    SubspaceSplit

    Raises
    ------
    ValueError
        On a shape mismatch, a non-finite residual, an out-of-range cut, or a
        residual field that is identically zero, which has no decomposition.
    """
    if not np.isfinite(strong_cut) or not 0.0 < strong_cut < 1.0:
        raise ValueError("strong_cut must lie in (0, 1)")
    r = np.asarray(residual, dtype=float).ravel()
    if r.shape != (sens.n_observations,):
        raise ValueError(
            f"residual must have shape ({sens.n_observations},), got {r.shape}"
        )
    if not np.all(np.isfinite(r)):
        raise ValueError("residual must be finite")
    rw = r / sens.sigma
    total = float(rw @ rw)
    if total <= 0.0:
        raise ValueError("the residual field is identically zero")
    u, sv, _vt = np.linalg.svd(sens.weighted, full_matrices=False)
    ratios = sv / float(sv[0])
    strong_mask = ratios >= strong_cut
    coeff = u.T @ rw
    strong = float(coeff[strong_mask] @ coeff[strong_mask])
    weak = float(coeff[~strong_mask] @ coeff[~strong_mask])
    orthogonal = float(max(total - strong - weak, 0.0))
    return SubspaceSplit(
        total_energy=total,
        strong_energy=strong,
        weak_energy=weak,
        orthogonal_energy=orthogonal,
        strong_fraction=strong / total,
        weak_fraction=weak / total,
        orthogonal_fraction=orthogonal / total,
        strong_cut=float(strong_cut),
        n_strong=int(np.count_nonzero(strong_mask)),
        n_weak=int(np.count_nonzero(~strong_mask)),
        singular_ratios=ratios,
    )


@dataclass(frozen=True)
class DriftReport:
    """Parameter movement caused by introducing a discrepancy term.

    Attributes
    ----------
    parameter_names:
        Names in parameter order.
    theta_physics_only, theta_with_residual:
        The two estimates, in model units.
    theta_true:
        Truth, if known. ``None`` for a real calibration.
    log_shift:
        ``ln(theta_with/theta_without)``, dimensionless: the signed relative
        movement of each parameter.
    rms_relative_error_before, rms_relative_error_after:
        Root mean square of ``ln(theta_hat/theta_true)`` over the parameters,
        dimensionless. ``nan`` when truth is unknown.
    chi2_before, chi2_after:
        Weighted sum of squares of each fit on the training data.
    """

    parameter_names: tuple[str, ...]
    theta_physics_only: np.ndarray
    theta_with_residual: np.ndarray
    theta_true: np.ndarray | None
    log_shift: np.ndarray
    rms_relative_error_before: float
    rms_relative_error_after: float
    chi2_before: float
    chi2_after: float

    @property
    def parameters_got_worse(self) -> bool | None:
        """Whether the physical parameters moved away from the truth.

        ``None`` when the truth is unknown, which is the normal case outside a
        synthetic experiment -- and the reason the question is usually
        unanswerable in practice.
        """
        if self.theta_true is None:
            return None
        return bool(self.rms_relative_error_after > self.rms_relative_error_before)

    def summary(self) -> str:
        """One block of text."""
        lines = [
            "parameter drift caused by the learned discrepancy term",
            f"  {'parameter':<10}{'physics only':>15}{'with residual':>15}"
            f"{'log shift':>12}",
        ]
        for i, name in enumerate(self.parameter_names):
            lines.append(
                f"  {name:<10}{self.theta_physics_only[i]:>15.6g}"
                f"{self.theta_with_residual[i]:>15.6g}{self.log_shift[i]:>+12.5f}"
            )
        lines.append(f"  training chi2: {self.chi2_before:.6g} -> {self.chi2_after:.6g}")
        if self.theta_true is not None:
            lines.append(
                f"  RMS log error against truth: "
                f"{self.rms_relative_error_before:.6f} -> "
                f"{self.rms_relative_error_after:.6f}"
            )
            verdict = (
                "the physical parameters got WORSE"
                if self.parameters_got_worse
                else "the physical parameters got better"
            )
            lines.append(f"  {verdict}")
        else:
            lines.append("  truth unknown, so the direction of the drift cannot be judged")
        return "\n".join(lines)


def parameter_drift(
    problem: CalibrationProblem,
    physics_fit: FitResult,
    learned_offset: Sequence[float],
    theta_true: Sequence[float] | None = None,
    max_nfev: int | None = 400,
) -> DriftReport:
    """Refit the physics with a learned discrepancy present, and report the move.

    Parameters
    ----------
    problem:
        The training problem the physics-only fit was made on. It must not
        already carry an offset; passing one would double-count the discrepancy
        and that is rejected.
    physics_fit:
        The physics-only fit.
    learned_offset:
        The residual model's prediction on the training observations, K, shape
        ``(n_obs,)``. It is *added* to the model prediction, which is the same
        convention as :class:`~twincalib.fitting.CalibrationProblem.offset`.
    theta_true:
        Truth, when known.
    max_nfev:
        Passed to the refit.

    Returns
    -------
    DriftReport
    """
    if problem.offset is not None:
        raise ValueError(
            "problem already carries an offset; pass the physics-only problem "
            "so that the discrepancy is not counted twice"
        )
    offset = np.asarray(learned_offset, dtype=float).ravel()
    if offset.shape != (problem.n_observations,):
        raise ValueError(
            f"learned_offset must have shape ({problem.n_observations},), "
            f"got {offset.shape}"
        )
    corrected = CalibrationProblem(
        model=problem.model,
        inputs=problem.inputs,
        initial_state=problem.initial_state,
        channels=problem.channels,
        sample_stride=problem.sample_stride,
        y=problem.y,
        sigma=problem.sigma,
        offset=offset,
    )
    refit = fit(corrected, physics_fit.theta, max_nfev=max_nfev)
    log_shift = np.log(refit.theta / physics_fit.theta)
    if theta_true is not None:
        truth = np.asarray(theta_true, dtype=float).ravel()
        if truth.shape != physics_fit.theta.shape:
            raise ValueError("theta_true must have one entry per parameter")
        before = float(np.sqrt(np.mean(np.log(physics_fit.theta / truth) ** 2)))
        after = float(np.sqrt(np.mean(np.log(refit.theta / truth) ** 2)))
    else:
        truth = None
        before = after = float("nan")
    return DriftReport(
        parameter_names=physics_fit.parameter_names,
        theta_physics_only=physics_fit.theta,
        theta_with_residual=refit.theta,
        theta_true=truth,
        log_shift=log_shift,
        rms_relative_error_before=before,
        rms_relative_error_after=after,
        chi2_before=physics_fit.chi2,
        chi2_after=refit.chi2,
    )
