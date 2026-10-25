"""Independently derived bounds on system reliability.

Why a bound is a different kind of check from a second implementation
---------------------------------------------------------------------

Two implementations that agree tell you they agree. If both evaluate the same
algebraic expression -- and a property test asserting ``R = 1 - prod(1 - R_i)``
against an implementation that computes exactly that is the clearest case --
then a catastrophic cancellation in the expression is present in both, and the
check passes while the answer is wrong. A bound derived from a *different*
argument does not share the expression, so it catches that class.

The bounds here are derived from combinatorics, not from the structural
recursion the quantifier uses, and they are evaluated from the minimal cut and
path sets. Every one of them is checked against the exact figure in
``validation/validate_bounds.py``, and a violation is reported as a failure of
the implementation rather than widened away.

Esary-Proschan bounds
---------------------

For a coherent system of independent components with minimal path sets
``P_1..P_p`` and minimal cut sets ``K_1..K_k``:

    ``prod_j (1 - prod_{i in K_j} q_i)  <=  R_sys  <=
      1 - prod_j (1 - prod_{i in P_j} p_i)``

Esary, J. D. and Proschan, F. (1963), "Coherent structures of non-identical
components", Technometrics 5(2), 191-209.

The lower bound treats the cut sets as if they were independent, which they are
not when they share components; sharing makes their joint failure more likely
than the product, so the product of the per-cut survival terms understates the
system's reliability. The upper bound does the same with the paths, in the other
direction. Both are exact when no two cut sets (respectively no two paths) share
a component, which is the series-parallel case. The bridge network is the
smallest interesting case where they are not exact, and the gap is measured.

Bonferroni bounds on series unreliability
-----------------------------------------

For a series chain, ``Q_sys = 1 - prod(1 - q_i)`` and inclusion-exclusion
truncated at successive orders brackets it:

    ``sum_i q_i - sum_{i<j} q_i q_j  <=  Q_sys  <=  sum_i q_i``

Bonferroni, C. E. (1936), "Teoria statistica delle classi e calcolo delle
probabilita", Pubblicazioni del R. Istituto Superiore di Scienze Economiche e
Commerciali di Firenze 8, 3-62. These are the first- and second-order
Bonferroni inequalities applied to the union of the per-block failure events.

This is the check that matters most for this package, because the first-order
bound ``Q_sys <= sum q_i`` is a *sum of positive terms* and the quantity it
bounds is the one the naive expression ``1 - prod(1 - q_i)`` destroys. For ten
blocks at ``q_i = 1e-12`` the bounds bracket ``Q_sys`` in
``[9.99999999995e-12, 1e-11]``; the naive expression returns a figure outside
that bracket, and the bound check -- not a property test -- is what detects it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from .diagram import Diagram
from .mission import MissionProfile
from .structure import minimal_cut_sets, minimal_path_sets

__all__ = [
    "Bracket",
    "bonferroni_series_bounds",
    "esary_proschan_bounds",
]


@dataclass(frozen=True)
class Bracket:
    """A lower and upper bound with the value they are supposed to bracket."""

    lower: float
    upper: float
    value: float
    name: str = ""
    #: Absolute rounding allowance on the bracket, in the units of ``value``,
    #: **propagated from the actual conditioning of the bound expressions** by
    #: whichever function built this bracket. Not a constant, and not a
    #: multiple of a term count.
    slack: float = 0.0

    @property
    def contains(self) -> bool:
        """True if ``value`` lies within the bracket, to within :attr:`slack`.

        Getting this allowance right took three attempts and two of the wrong
        answers are worth recording, because both are tempting:

        **A slack floored at an absolute constant.** The Bonferroni bounds on a
        ten-block chain of blocks at ``q = 1e-12`` have a width of 4.5e-23
        around 1e-11. A slack of ``8 eps * max(scale, 1.0)`` is 1.8e-15 there,
        forty million times the bracket's own width, and calls every value in
        the neighbourhood inside. That is not a tolerance, it is the absence of
        a check, and it is the same defect class as a relative ``approx``
        tolerance silently replaced by an absolute default.

        **A slack proportional to the number of arithmetic operations.** The
        Esary-Proschan bounds are *exact* when no two cut sets share a
        component, so for a series-parallel structure the bracket has zero
        width in exact arithmetic and only rounding separates the value from
        the bounds. Hypothesis falsified ``8 eps`` and then ``2 m eps`` on the
        same two-cut-set structure, at
        ``rates = [0.0078125, 5.960464477539063e-08]`` and ``hours = 762213``,
        where the exact reliability sits 2.7e-17 below the lower bound while an
        operation-count slack allowed only 3.4e-18. The reason is conditioning,
        not operation count: the min-cut bound evaluates ``1 - prod q_i`` with
        every ``q_i`` close to 1, and that subtraction has an absolute error of
        order ``eps`` whatever its result, so its *relative* error is
        ``eps / (1 - prod q_i)``, which here is 9e-13. Only a slack propagated
        from the expression's own conditioning is right, and that is what
        :func:`esary_proschan_bounds` now computes.
        """
        return self.lower - self.slack <= self.value <= self.upper + self.slack

    @property
    def width(self) -> float:
        """``upper - lower``."""
        return self.upper - self.lower

    def __str__(self) -> str:
        verdict = "inside" if self.contains else "OUTSIDE"
        return (
            f"{self.name or 'bracket'}: [{self.lower:.12e}, {self.upper:.12e}] "
            f"value {self.value:.12e} ({verdict}, width {self.width:.6e}, "
            f"slack {self.slack:.3e})"
        )


def esary_proschan_bounds(
    diagram: Diagram,
    mission: MissionProfile | float,
    reliability: float,
    *,
    unreliabilities: dict[str, float] | None = None,
) -> Bracket:
    """Esary-Proschan min-cut lower and min-path upper bounds on ``R_sys``.

    Parameters
    ----------
    diagram
        The declared diagram. Its minimal cut and path sets are enumerated
        structurally; the bound arithmetic shares nothing else with the
        quantifier.
    mission
        Mission profile or duration in hours, used to get per-block
        unreliabilities when ``unreliabilities`` is not given.
    reliability
        The exact system reliability the bounds are to bracket.
    unreliabilities
        Optional per-event unreliability map, overriding the mission.

    Returns
    -------
    Bracket
        Lower bound, upper bound and the supplied value.
    """
    q = (
        dict(unreliabilities)
        if unreliabilities is not None
        else _mission_unreliabilities(diagram, mission)
    )
    cuts = minimal_cut_sets(diagram)
    paths = minimal_path_sets(diagram)
    eps = 2.220446049250313e-16
    lower = 1.0
    lower_relative_error = 0.0
    for cut in cuts:
        product = 1.0
        for event in sorted(cut):
            product *= q[event]
        factor = 1.0 - product
        # The product's absolute error is about len(cut) * eps * product, and
        # the subtraction from 1 contributes one more eps at the scale of the
        # larger of the two operands. The factor's RELATIVE error is therefore
        # that absolute error divided by the factor, which blows up exactly
        # when the cut is nearly certain to occur.
        absolute = (len(cut) + 1) * eps * max(product, factor)
        lower_relative_error += eps + (absolute / factor if factor > 0.0 else 0.0)
        lower *= factor
    upper_complement = 1.0
    upper_relative_error = 0.0
    for path in paths:
        product = 1.0
        for event in sorted(path):
            product *= 1.0 - q[event]
        factor = 1.0 - product
        absolute = (len(path) + 1) * eps * max(product, factor)
        upper_relative_error += eps + (absolute / factor if factor > 0.0 else 0.0)
        upper_complement *= factor
    slack = max(
        lower * lower_relative_error,
        upper_complement * upper_relative_error,
        8.0 * eps * abs(float(reliability)),
    )
    return Bracket(
        lower=lower,
        upper=1.0 - upper_complement,
        value=float(reliability),
        name=f"Esary-Proschan ({len(cuts)} cut sets, {len(paths)} path sets)",
        slack=slack,
    )


def bonferroni_series_bounds(unreliabilities: Sequence[float], value: float) -> Bracket:
    """First- and second-order Bonferroni bounds on a series chain's ``Q_sys``.

    ``sum q_i - sum_{i<j} q_i q_j <= Q_sys <= sum q_i``.

    Both sides are sums of non-negative terms, so neither inherits the
    cancellation of ``1 - prod(1 - q_i)``. This is the independently derived
    bound the package's precision claim rests on.
    """
    probabilities = [float(q) for q in unreliabilities]
    for index, q in enumerate(probabilities):
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"unreliabilities[{index}] must lie in [0, 1], got {q}")
    first = sum(probabilities)
    second = sum(a * b for a, b in combinations(probabilities, 2))
    n = len(probabilities)
    # Both sides are sums of non-negative terms, so there is no cancellation
    # and the error is the ordinary accumulation of one rounding per term.
    eps = 2.220446049250313e-16
    terms = n + n * (n - 1) // 2
    return Bracket(
        lower=max(first - second, 0.0),
        upper=min(first, 1.0),
        value=float(value),
        name=f"Bonferroni order 1 and 2 over {n} blocks",
        slack=max(terms, 2) * eps * max(first, 0.0),
    )


def _mission_unreliabilities(
    diagram: Diagram, mission: MissionProfile | float
) -> dict[str, float]:
    """Per-block unreliability at the mission point."""
    profile = mission if isinstance(mission, MissionProfile) else MissionProfile.from_hours(mission)
    ages = profile.effective_ages(diagram)
    return {b.name: b.distribution.unreliability(ages[b.name]) for b in diagram.blocks()}
