"""Invert the bounds: how many MORE field measurements close the gap to a target.

The question this module answers
--------------------------------
A programme has fifteen paired measurements and a distribution-free bound that
says the mean sim-to-field discrepancy is at most 12 kN.m with 95 per cent
confidence. The programme wants to be able to say 8 kN.m. The useful answer is
not "collect more data"; it is a number of additional flights, with the
assumption behind that number on the same line.

Three different inversions, with three different standings
-----------------------------------------------------------
**Hoeffding, equation (10).** Exact and data-free. The slack
``R sqrt(ln(1/delta)/(2n))`` does not depend on the sample at all, so the
required ``n`` is a deterministic function of ``R``, ``delta`` and the target.
If the plan says 231 flights, 231 flights reach that slack, always. What it
does not promise is that the resulting BOUND is below any particular value:
the sample mean moves too.

**Empirical Bernstein, equation (11).** A **prediction**, not a guarantee. The
slack depends on the sample variance, which is estimated from the measurements
in hand, so the required ``n`` inherits that estimate's error. A programme
that plans to this number and draws an unlucky high-variance sample will miss
the target. The rate at which that happens is measured, over many draws, in
``validation/validate_inversion_output.txt`` rather than assumed.

**Wilks, equation (7), and split conformal, equation (12).** Both are
requirements on ``n`` for the STATEMENT TO EXIST at all, independent of any
width target and of any data. Wilks' limit at proportion ``p`` and confidence
``1 - delta`` needs ``n >= ln(delta)/ln(p)``; split conformal at miscoverage
``alpha`` needs ``n >= ceil(1/alpha) - 1``. Below those sample sizes there is
no limit to report, and this package raises rather than returning one.

Every plan returned here says which of the three it is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bounds import (
    empirical_bernstein_slack,
    required_n_empirical_bernstein,
    required_n_hoeffding,
    required_n_wilks,
)
from .discrepancy import GapSample, validate_confidence
from .resample import conformal_required_n

__all__ = ["SamplePlan", "plan_existence", "plan_for_slack"]


@dataclass(frozen=True)
class SamplePlan:
    """How many more measurements a method needs, and what that number means.

    Attributes
    ----------
    method:
        The bound being inverted.
    basis:
        ``"exact"`` when the inversion is a deterministic function of declared
        quantities only; ``"plug-in"`` when it rests on a quantity estimated
        from the sample in hand and is therefore a prediction; ``"existence"``
        when the requirement is for the statement to exist at all and has
        nothing to do with a width target.
    current_n:
        Measurements already in hand.
    required_n:
        Total measurements required.
    additional_n:
        ``max(0, required_n - current_n)``.
    target_slack:
        The slack targeted, in ``unit``; ``None`` for an existence plan.
    achieved_slack:
        The slack the method's forward expression gives at ``required_n``, in
        ``unit``; ``None`` for an existence plan. Reported so that the
        inversion can be checked against the forward expression by eye.
    confidence:
        ``1 - delta``.
    unit:
        Unit of ``target_slack`` and ``achieved_slack``.
    note:
        One sentence on what the number does and does not promise.
    """

    method: str
    basis: str
    current_n: int
    required_n: int
    additional_n: int
    target_slack: float | None
    achieved_slack: float | None
    confidence: float
    unit: str
    note: str


def plan_for_slack(
    sample: GapSample, target_slack: float, confidence: float = 0.95
) -> list[SamplePlan]:
    """Plans for the two mean bounds at a declared target slack.

    Parameters
    ----------
    sample:
        The measurements in hand. Only its size, range, unit and sample
        variance are used.
    target_slack:
        The slack to reach, in the sample's unit, strictly positive.
    confidence:
        ``1 - delta``.

    Returns
    -------
    list of SamplePlan
        One for Hoeffding (``basis="exact"``) and one for empirical Bernstein
        (``basis="plug-in"``).
    """
    level = validate_confidence(confidence)
    if not np.isfinite(target_slack) or target_slack <= 0.0:
        raise ValueError(
            f"target_slack must be finite and positive, in {sample.unit or 'the sample unit'}; "
            f"got {target_slack!r}"
        )
    n = sample.size
    plans = []

    hoeffding_n = required_n_hoeffding(sample.bound_range, target_slack, level)
    hoeffding_slack = sample.bound_range * np.sqrt(
        np.log(1.0 / (1.0 - level)) / (2.0 * hoeffding_n)
    )
    plans.append(
        SamplePlan(
            method="hoeffding",
            basis="exact",
            current_n=n,
            required_n=hoeffding_n,
            additional_n=max(0, hoeffding_n - n),
            target_slack=float(target_slack),
            achieved_slack=float(hoeffding_slack),
            confidence=level,
            unit=sample.unit,
            note=(
                "exact: the Hoeffding slack of equation (3) depends only on R, delta "
                "and n, so this count is a guarantee about the SLACK. It is not a "
                "guarantee about the value of the bound, which also moves with the "
                "sample mean."
            ),
        )
    )

    variance = float(np.var(sample.gaps, ddof=1)) if n >= 2 else 0.0
    bernstein_n = required_n_empirical_bernstein(
        variance, sample.bound_range, target_slack, level
    )
    plans.append(
        SamplePlan(
            method="empirical_bernstein",
            basis="plug-in",
            current_n=n,
            required_n=bernstein_n,
            additional_n=max(0, bernstein_n - n),
            target_slack=float(target_slack),
            achieved_slack=float(
                empirical_bernstein_slack(bernstein_n, variance, sample.bound_range, level)
            ),
            confidence=level,
            unit=sample.unit,
            note=(
                f"plug-in: equation (11) uses the sample variance of the {n} "
                "measurements in hand as a stand-in for the population variance, so "
                "this count is a PREDICTION. Its measured hit rate at this sample "
                "size is in validation/validate_inversion_output.txt."
            ),
        )
    )
    return plans


def plan_existence(
    sample: GapSample, proportion: float = 0.95, confidence: float = 0.95
) -> list[SamplePlan]:
    """Plans for the two prediction limits: the ``n`` at which they exist at all.

    Neither Wilks' limit nor the split-conformal quantile has a width target to
    invert. What they have is a sample size below which the statement cannot be
    made, and that is what is returned.
    """
    level = validate_confidence(confidence)
    p = validate_confidence(proportion, "proportion")
    n = sample.size
    wilks_n = required_n_wilks(p, level)
    conformal_n = conformal_required_n(1.0 - p)
    return [
        SamplePlan(
            method="wilks_tolerance",
            basis="existence",
            current_n=n,
            required_n=wilks_n,
            additional_n=max(0, wilks_n - n),
            target_slack=None,
            achieved_slack=None,
            confidence=level,
            unit=sample.unit,
            note=(
                f"existence: equation (7), n >= ln(delta)/ln(p). Below {wilks_n} "
                f"measurements the largest observation does not cover {p:g} of the "
                f"population at confidence {level:g}, whatever its value."
            ),
        ),
        SamplePlan(
            method="split_conformal",
            basis="existence",
            current_n=n,
            required_n=conformal_n,
            additional_n=max(0, conformal_n - n),
            target_slack=None,
            achieved_slack=None,
            confidence=p,
            unit=sample.unit,
            note=(
                f"existence: equation (12) asks for the ceil((n+1)*{p:g})-th smallest "
                f"of n calibration scores, which is above n until n = {conformal_n}. "
                f"This is a MARGINAL guarantee, weaker than the Wilks row above it."
            ),
        ),
    ]
