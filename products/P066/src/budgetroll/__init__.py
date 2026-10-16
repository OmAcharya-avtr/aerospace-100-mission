"""budgetroll: hierarchical margin and budget rollup with explicit combination rules.

Educational-grade (validation level 1). Not flight-qualified, not certified,
not approved for operational aerospace use.

The package keeps deterministic contingency and random uncertainty apart,
makes the linear against root-sum-square choice explicit at every node, and
checks each node's stated independence assumption against the budget's
declared correlation structure.
"""

from __future__ import annotations

from .aggregate import (
    NodeResult,
    Rollup,
    combine,
    leaf_contingency,
    linear_rollup_uncertainty,
    rollup,
    rss_rollup_uncertainty,
)
from .checks import CODES, Finding, has_blocking, run_checks, severity_counts
from .correlation import (
    correlation_matrix,
    is_positive_semidefinite,
    quadratic_form,
)
from .errors import AggregationError, BudgetRollError, DeclarationError
from .model import Budget, Correlation, Item, MarginPolicy, Node, leaves, walk
from .parse import load, loads
from .report import json_report, markdown_report, text_report, to_dict
from .sensitivity import (
    LeafSensitivity,
    margin_elasticity,
    sensitivity,
    value_shares,
    variance_shares,
)

__version__ = "0.1.0"

__all__ = [
    "CODES",
    "AggregationError",
    "Budget",
    "BudgetRollError",
    "Correlation",
    "DeclarationError",
    "Finding",
    "Item",
    "LeafSensitivity",
    "MarginPolicy",
    "Node",
    "NodeResult",
    "Rollup",
    "__version__",
    "combine",
    "correlation_matrix",
    "has_blocking",
    "is_positive_semidefinite",
    "json_report",
    "leaf_contingency",
    "leaves",
    "linear_rollup_uncertainty",
    "load",
    "loads",
    "margin_elasticity",
    "markdown_report",
    "quadratic_form",
    "rollup",
    "rss_rollup_uncertainty",
    "run_checks",
    "sensitivity",
    "severity_counts",
    "text_report",
    "to_dict",
    "value_shares",
    "variance_shares",
    "walk",
]
