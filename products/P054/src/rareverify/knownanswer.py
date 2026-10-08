"""Known-answer checks against reference failure probabilities.

A known-answer check compares an estimate to a reference probability that is
known independently of the estimator.  The question it has to answer is what
"agrees" means, and this module answers it in one way only: **the tolerance is
stated as a multiple of the counting-noise floor of the run that produced the
estimate.**

The counting-noise floor is

    sigma_0 = sqrt(p (1 - p) / n_eff)

with ``p`` the reference probability and ``n_eff`` the number of true
limit-state evaluations the estimate spent.  It is the standard deviation a
crude Monte-Carlo estimate of that probability would have at that budget, and
no estimator can be required to agree with the reference more tightly than
that unless it has lower variance, in which case
:func:`check_against_reference` uses the estimator's own reported standard
error instead -- whichever is larger, so the tolerance is never tighter than
what the run can support.

A tolerance of ``k`` noise floors corresponds to a two-sided normal test at
level ``2 (1 - Phi(k))``: ``k = 3`` is a 0.27 % false-alarm rate per check.
With a dozen checks in a suite that is about a 3 % chance of one spurious
failure, which is why the default is ``k = 4`` (6.3e-5 per check) for the
suite and why no tolerance anywhere in this package is tuned after seeing a
result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .estimate import RareEventEstimate, counting_noise_floor

__all__ = ["KnownAnswerCheck", "check_against_reference"]


@dataclass(frozen=True)
class KnownAnswerCheck:
    """One known-answer comparison.

    Attributes
    ----------
    name
        Check name, e.g. ``"linear-gaussian beta=3.719 crude"``.
    reference
        Reference failure probability, dimensionless.
    reference_kind
        ``"closed-form"`` or ``"quadrature"``, carried from the limit state so
        a quadrature reference is never quoted as exact.
    estimate
        The estimate under test, dimensionless.
    reported_standard_error
        The estimator's own standard error.
    noise_floor
        ``sqrt(p (1 - p) / n_eff)``, dimensionless.
    tolerance_basis
        The larger of :attr:`noise_floor` and
        :attr:`reported_standard_error`, which is what the tolerance multiplies.
    absolute_error
        ``abs(estimate - reference)``.
    error_in_tolerance_units
        ``absolute_error / tolerance_basis``: how many noise floors out the
        estimate is.
    tolerance_multiples
        The allowed number of noise floors.
    true_evaluations
        Evaluations the estimate spent.
    passed
        ``error_in_tolerance_units <= tolerance_multiples``.
    """

    name: str
    reference: float
    reference_kind: str
    estimate: float
    reported_standard_error: float
    noise_floor: float
    tolerance_basis: float
    absolute_error: float
    error_in_tolerance_units: float
    tolerance_multiples: float
    true_evaluations: int
    passed: bool

    def describe(self) -> str:
        """One-line summary. Contains no I/O."""
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"[{verdict}] {self.name}: ref={self.reference:.6e} "
            f"({self.reference_kind}) est={self.estimate:.6e} "
            f"err={self.absolute_error:.3e} = {self.error_in_tolerance_units:.3f} "
            f"x tolerance basis {self.tolerance_basis:.3e} "
            f"(limit {self.tolerance_multiples:.1f}, evals={self.true_evaluations})"
        )


def check_against_reference(
    name: str,
    estimate: RareEventEstimate,
    reference: float,
    reference_kind: str = "closed-form",
    tolerance_multiples: float = 4.0,
) -> KnownAnswerCheck:
    """Compare an estimate to a reference probability.

    Parameters
    ----------
    name
        Check name recorded in the result.
    estimate
        The estimate under test.
    reference
        Reference failure probability in ``(0, 1)``.
    reference_kind
        ``"closed-form"`` or ``"quadrature"``.
    tolerance_multiples
        Allowed error in units of the tolerance basis, ``> 0``.  Default 4.

    Returns
    -------
    KnownAnswerCheck
    """
    reference = float(reference)
    if not 0.0 < reference < 1.0:
        raise ValueError(f"reference must lie strictly in (0, 1), got {reference}")
    if tolerance_multiples <= 0.0:
        raise ValueError(
            f"tolerance_multiples must be positive, got {tolerance_multiples}"
        )
    floor = counting_noise_floor(reference, max(1, estimate.true_evaluations))
    reported = (
        estimate.standard_error
        if math.isfinite(estimate.standard_error)
        else 0.0
    )
    basis = max(floor, reported)
    error = abs(estimate.estimate - reference)
    units = error / basis if basis > 0.0 else math.inf
    return KnownAnswerCheck(
        name=name,
        reference=reference,
        reference_kind=reference_kind,
        estimate=estimate.estimate,
        reported_standard_error=reported,
        noise_floor=floor,
        tolerance_basis=basis,
        absolute_error=error,
        error_in_tolerance_units=units,
        tolerance_multiples=float(tolerance_multiples),
        true_evaluations=int(estimate.true_evaluations),
        passed=bool(units <= tolerance_multiples),
    )
