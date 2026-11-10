"""Bounds on a joint unavailability, derived from mathematics nothing here uses.

Why this module exists
----------------------
"Two implementations agree" and "the answer lies inside a bound derived from
different mathematics" are different checks, and the second catches what the
first cannot. Every number this package reports for a joint unavailability comes
from one of three routes -- a product of marginals, a Gaussian-copula orthant, or
a count of epochs -- and all three can be wrong together if the marginals are
wrong. The bounds below depend on the marginals and on NOTHING ELSE: no copula,
no correlation, no latent field, no record. They are distribution-free.

**(17) Frechet-Hoeffding bounds.** For any joint distribution with marginal
probabilities ``q_1 ... q_n``, the probability that all ``n`` events occur
satisfies

    max( 0, sum_i q_i - (n - 1) )  <=  P_joint  <=  min_i q_i

The upper bound is immediate -- the intersection is contained in each event. The
lower bound is the complement of the union bound: ``P(union of complements) <=
sum (1 - q_i) = n - sum q_i``, so ``P_joint = 1 - P(union of complements) >=
1 - n + sum q_i``. Hoeffding 1940 and Frechet 1951; the statement for copulas is
Nelsen 2006 theorem 2.10.12, where the bounds are the copulas ``W`` and ``M``.
Both bounds are ATTAINED, by comonotone and countermonotone arrangements
respectively, so neither can be tightened without an assumption.

These are the right bounds for this product because the whole question is how
far from independence a real field sits, and ``prod q_i`` lies strictly inside
them for ``n >= 2``: the independence assumption is one point in an interval
whose width is the entire content of the site-diversity problem.

**(18) A Bonferroni bracket.** A second-order inclusion-exclusion bracket on the
union of the complements gives, with ``u_i = 1 - q_i``,

    sum_i u_i - sum_{i<j} u_ij  <=  P(union)  <=  sum_i u_i

where ``u_ij = P(both clear)``. Taking complements,

    1 - sum_i u_i  <=  P_joint  <=  1 - sum_i u_i + sum_{i<j} u_ij

The lower bound here is the same as equation (17)'s; the UPPER bound is new and
uses pairwise information, so it is sharper than ``min q_i`` whenever the
pairwise clear probabilities are known. This is the Kounias-Hunter family; see
Kounias 1968 and the standard Bonferroni treatment in Galambos & Simonelli 1996.
It is included because it is the one bound here that uses pair statistics, so it
tests an n-station Monte Carlo result against pair quantities computed by a
different route.

**(19) The slack, and the conditioning it follows.** A computed value is
compared against a bound with a slack, and that slack must follow the
CONDITIONING of the bound expression. The lower bound of equation (17) is a sum
of ``n`` terms of order one minus a constant of order ``n``, and its value can
be arbitrarily close to zero: as ``sum q_i`` approaches ``n - 1`` the
subtraction loses every significant digit it has. The relative condition number
of ``f(q) = sum q_i - (n - 1)`` with respect to perturbations in ``q`` is

    kappa = sum_i |q_i| / | sum_i q_i - (n - 1) |

and the ABSOLUTE rounding scale of the same expression is

    slack = c eps ( sum_i |q_i| + |f(q)| )

with ``c = SLACK_FACTOR``. These are two statements of one fact: the absolute
slack is set by the magnitude of the SUMMANDS and not by the magnitude of the
result, so the relative slack is ``kappa`` times machine epsilon and diverges
exactly where the expression cancels. Writing the slack as ``eps kappa |f|``
instead would be algebraically identical to ``eps sum|q_i|`` and would therefore
look conditioned while being the same number -- an error this product made
during its own build and caught with the test
``test_the_slack_is_not_a_relative_slack_in_disguise``.

The consequence the implementation must carry is that where ``|f(q)|`` falls
below its own slack the lower bound is NOT INFORMATIVE: the check on that side
admits anything and must say so rather than report a pass.
``FrechetBounds.lower_informative`` is that flag, it is printed by
``validation/validate_bounds.py`` for every case, and no pass is claimed on an
uninformative side. A slack floored at an absolute constant, by contrast, makes
a bracket of width 1e-23 vacuous while still reporting a pass, and admits
exactly the error the check exists to catch -- which a predecessor in this
mission measured directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Double-precision unit roundoff, used by equation (19).
MACHINE_EPSILON: float = float(np.finfo(float).eps)

#: Multiplier on the conditioned slack of equation (19). DECLARED. It covers the
#: handful of roundings between the marginals and the compared value, and it is
#: a MULTIPLIER on a conditioned quantity, not an additive floor.
SLACK_FACTOR: float = 64.0


def bound_condition_number(rates: np.ndarray, bound_value: float) -> float:
    """``kappa`` of equation (19): the relative conditioning of the bound.

    ``inf`` where the bound expression has cancelled completely, because no
    finite relative conditioning describes that case.
    """
    values = np.asarray(rates, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            f"rates must be a non-empty one-dimensional array, got {values.shape}"
        )
    value = abs(float(bound_value))
    if value == 0.0:
        return float("inf")
    return float(np.sum(np.abs(values))) / value


def bound_slack(rates: np.ndarray, bound_value: float) -> float:
    """Equation (19): the absolute rounding scale of the bound expression.

    Set by the magnitude of the SUMMANDS, never by the magnitude of the result.
    That is what makes it a conditioned slack: the relative slack
    ``slack / |bound_value|`` is ``kappa`` times machine epsilon and diverges
    as the expression cancels, while the absolute slack stays at the level the
    summation's own rounding sets.

    Parameters
    ----------
    rates:
        ``(n,)`` marginal probabilities the bound was computed from.
    bound_value:
        The computed bound, included so that its own representation error is
        covered.

    Returns
    -------
    float
        A strictly positive slack.
    """
    values = np.asarray(rates, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            f"rates must be a non-empty one-dimensional array, got {values.shape}"
        )
    magnitude = float(np.sum(np.abs(values))) + abs(float(bound_value))
    return SLACK_FACTOR * MACHINE_EPSILON * magnitude


@dataclass(frozen=True)
class FrechetBounds:
    """Equation (17) for one station set, with the slack of equation (19).

    Attributes
    ----------
    lower, upper:
        The distribution-free bounds.
    rates:
        The marginals they were computed from.
    slack:
        The comparison slack of equation (19), evaluated at the tighter of the
        two bounds in the sense of conditioning: the lower bound, which is the
        one that can cancel.
    condition_number:
        ``kappa`` of equation (19) for the lower bound, or ``inf`` where that
        bound is identically zero.
    """

    lower: float
    upper: float
    rates: tuple[float, ...]
    slack: float
    condition_number: float

    @property
    def width(self) -> float:
        """``upper - lower``: the whole range a dependence assumption chooses in."""
        return self.upper - self.lower

    @property
    def lower_informative(self) -> bool:
        """Whether the lower bound carries more information than its own slack.

        ``False`` means the check on the lower side admits anything, either
        because the union bound is vacuous and the bound clipped to zero, or
        because ``sum q_i - (n - 1)`` cancelled to within its own rounding. A
        report must not claim a pass on an uninformative side, and
        ``validation/validate_bounds.py`` prints this flag for every case.
        """
        return self.lower > self.slack

    @property
    def upper_informative(self) -> bool:
        """Whether the upper bound excludes anything. ``min q_i < 1`` suffices."""
        return self.upper < 1.0 - self.slack

    def contains(self, value: float) -> bool:
        """Whether ``value`` lies inside the bounds, allowing the slack."""
        return self.lower - self.slack <= float(value) <= self.upper + self.slack

    def violation(self, value: float) -> float:
        """How far outside the slackened bounds ``value`` lies; 0.0 if inside."""
        probe = float(value)
        if probe < self.lower - self.slack:
            return self.lower - self.slack - probe
        if probe > self.upper + self.slack:
            return probe - self.upper - self.slack
        return 0.0


def frechet_hoeffding(rates: np.ndarray) -> FrechetBounds:
    """Equation (17): the distribution-free bounds on an ``n``-fold intersection.

    Parameters
    ----------
    rates:
        ``(n,)`` marginal probabilities in ``[0, 1]``, ``n >= 2``.

    Returns
    -------
    FrechetBounds
    """
    values = np.asarray(rates, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"rates must be one-dimensional, got {values.ndim}D")
    if values.size < 2:
        raise ValueError(
            f"at least 2 marginals are needed for a joint bound, got {values.size}"
        )
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("every marginal rate must lie in [0, 1]")
    count = values.size
    raw_lower = float(np.sum(values) - (count - 1))
    lower = max(0.0, raw_lower)
    upper = float(np.min(values))
    condition = bound_condition_number(values, raw_lower)
    return FrechetBounds(
        lower=lower,
        upper=upper,
        rates=tuple(float(rate) for rate in values),
        slack=bound_slack(values, raw_lower),
        condition_number=condition,
    )


@dataclass(frozen=True)
class BonferroniBracket:
    """Equation (18) for one station set.

    Attributes
    ----------
    lower, upper:
        The bracket. ``lower`` coincides with the Frechet lower bound before
        clipping at zero; ``upper`` uses the pairwise clear probabilities and is
        sharper than ``min q_i`` whenever they are informative.
    clear_rates:
        ``(n,)`` marginal CLEAR probabilities ``u_i = 1 - q_i``.
    pair_clear_sum:
        ``sum_{i<j} u_ij``, the second-order term.
    slack:
        Equation (19), evaluated on the clear marginals.
    """

    lower: float
    upper: float
    clear_rates: tuple[float, ...]
    pair_clear_sum: float
    slack: float

    def contains(self, value: float) -> bool:
        """Whether ``value`` lies inside the bracket, allowing the slack."""
        return self.lower - self.slack <= float(value) <= self.upper + self.slack

    def violation(self, value: float) -> float:
        """How far outside the slackened bracket ``value`` lies; 0.0 if inside."""
        probe = float(value)
        if probe < self.lower - self.slack:
            return self.lower - self.slack - probe
        if probe > self.upper + self.slack:
            return probe - self.upper - self.slack
        return 0.0


def bonferroni_bracket(
    rates: np.ndarray, pair_clear: np.ndarray
) -> BonferroniBracket:
    """Equation (18): a second-order bracket using pairwise clear probabilities.

    Parameters
    ----------
    rates:
        ``(n,)`` marginal cloud probabilities.
    pair_clear:
        ``(n, n)`` symmetric matrix of ``P(both clear)``. The diagonal is
        ignored.

    Returns
    -------
    BonferroniBracket
    """
    values = np.asarray(rates, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(
            f"at least 2 marginals are needed, got shape {values.shape}"
        )
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("every marginal rate must lie in [0, 1]")
    matrix = np.asarray(pair_clear, dtype=float)
    count = values.size
    if matrix.shape != (count, count):
        raise ValueError(
            f"pair_clear must be ({count}, {count}) to match rates, got "
            f"{matrix.shape}"
        )
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError("every pairwise clear probability must lie in [0, 1]")
    clear = 1.0 - values
    rows, cols = np.triu_indices(count, 1)
    pair_sum = float(np.sum(matrix[rows, cols]))
    union_low = float(np.sum(clear)) - pair_sum
    union_high = float(np.sum(clear))
    return BonferroniBracket(
        lower=1.0 - union_high,
        upper=1.0 - union_low,
        clear_rates=tuple(float(rate) for rate in clear),
        pair_clear_sum=pair_sum,
        slack=bound_slack(clear, 1.0 - union_high),
    )
