"""AllocLab: control allocation for over-actuated spacecraft effector sets."""

from .allocation import (
    METHODS,
    AllocationResult,
    InfeasibleAllocationError,
    allocate,
    is_attainable,
    lp_allocate,
    pseudo_inverse_allocate,
    qp_allocate,
    redistributed_pseudo_inverse_allocate,
    weighted_pseudo_inverse_allocate,
)
from .ams import (
    AttainableMomentSet,
    attainable_moment_set,
    expected_vertex_count,
    zonotope_volume,
)
from .effectors import (
    EffectorSet,
    general_effector_set,
    orthogonal_effectors,
    pyramid_reaction_wheels,
    reaction_wheel_array,
    thruster_cluster,
)
from .failure import FailureReport, failure_margin, reallocate_after_failure

__version__ = "0.1.0"

__all__ = [
    "METHODS",
    "AllocationResult",
    "AttainableMomentSet",
    "EffectorSet",
    "FailureReport",
    "InfeasibleAllocationError",
    "__version__",
    "allocate",
    "attainable_moment_set",
    "expected_vertex_count",
    "failure_margin",
    "general_effector_set",
    "is_attainable",
    "lp_allocate",
    "orthogonal_effectors",
    "pseudo_inverse_allocate",
    "pyramid_reaction_wheels",
    "qp_allocate",
    "reaction_wheel_array",
    "reallocate_after_failure",
    "redistributed_pseudo_inverse_allocate",
    "thruster_cluster",
    "weighted_pseudo_inverse_allocate",
    "zonotope_volume",
]
