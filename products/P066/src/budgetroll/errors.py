"""Exception types raised by budgetroll.

Every error carries an actionable message naming the offending identifier or
YAML key path, so that a malformed declaration can be fixed without reading
the parser.
"""

from __future__ import annotations


class BudgetRollError(Exception):
    """Base class for every error raised by this package."""


class DeclarationError(BudgetRollError):
    """A budget declaration is malformed, incomplete or self-contradictory.

    Raised at parse or construction time: unknown keys, duplicate identifiers,
    dangling correlation references, a correlation coefficient outside
    [-1, 1], an unknown maturity level, a negative uncertainty.
    """


class AggregationError(BudgetRollError):
    """An aggregation could not be performed as declared.

    Raised when a combination rule is not one of the three implemented rules,
    when a node has no children, or when a correlation matrix supplied for an
    exact combination is not square or not symmetric.
    """
