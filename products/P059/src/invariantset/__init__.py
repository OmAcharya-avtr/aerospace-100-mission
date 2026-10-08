"""invariantset: robust invariant and reachable sets for discrete-time linear systems.

Research-grade. Not flight-qualified, not certified, not approved for
operational aerospace use.

This package computes the **maximal** robust invariant set inside a given
constraint set, by the one-step-set recursion.  It does **not** compute the
minimal robust positively invariant set of Rakovic et al. 2005; the two are
different objects and `invariant` documents the distinction.
"""

from __future__ import annotations

from .diagnostics import (
    GrowthRow,
    ToleranceRow,
    growth_table,
    tolerance_sweep,
)
from .invariant import (
    CONVERGED,
    EMPTY,
    ITERATION_CAP,
    InvariantSetResult,
    IterationRecord,
    maximal_robust_invariant_set,
    pre_set,
    verify_robust_invariance,
)
from .polytope import (
    DEFAULT_REDUNDANCY_TOL,
    EmptyPolytopeError,
    Polytope,
    UnboundedDirectionError,
    VertexEnumerationError,
)
from .setalgebra import (
    intersect,
    is_subset,
    minkowski_sum,
    pontryagin_difference,
    support_gap,
)
from .systems import SYSTEMS, LinearSystem, get_system, system_names

__version__ = "0.1.0"

__all__ = [
    "CONVERGED",
    "DEFAULT_REDUNDANCY_TOL",
    "EMPTY",
    "EmptyPolytopeError",
    "GrowthRow",
    "ITERATION_CAP",
    "InvariantSetResult",
    "IterationRecord",
    "LinearSystem",
    "Polytope",
    "SYSTEMS",
    "ToleranceRow",
    "UnboundedDirectionError",
    "VertexEnumerationError",
    "__version__",
    "get_system",
    "growth_table",
    "intersect",
    "is_subset",
    "maximal_robust_invariant_set",
    "minkowski_sum",
    "pontryagin_difference",
    "pre_set",
    "support_gap",
    "system_names",
    "tolerance_sweep",
    "verify_robust_invariance",
]
