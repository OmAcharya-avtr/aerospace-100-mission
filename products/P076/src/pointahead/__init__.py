"""pointahead: the point-ahead angle for an optical crosslink, with every error
term attributed to its own checkable line.

The total is not the deliverable. A 5 microradian point-ahead error is 10 per
cent of a 50 microradian beam and the whole of a 5 microradian one, and the
dominant term is what tells an engineer what to fix. This package computes the
angle two ways -- a closed-form first-order solution and an iterated
light-time reference the closed form is scored against -- and returns a budget
in which relative velocity knowledge, ephemeris staleness, light-time
iteration residual, stellar aberration, velocity aberration and the model error
of the closed form are separate lines, each with its equation, its validity
statement and a numerical check.

Research grade. Not flight-qualified, not certified, not approved for
operational aerospace use. No output of this package is suitable for an
operational pointing decision.
"""

from __future__ import annotations

from .aberration import (
    ABERRATION_ROUTES,
    EXACT_ROUTES,
    aberrate,
    aberrate_angle_relation,
    aberrate_event_worldline,
    aberrate_first_order,
    aberrate_momentum_boost,
    aberration_angle,
    beta_from_velocity,
    lorentz_gamma,
    route_disagreement,
)
from .angle import (
    BaselineScore,
    PointAheadDecomposition,
    PointAheadResult,
    closed_form_point_ahead,
    decompose_point_ahead,
    light_time_pair,
    offset_difference,
    reference_point_ahead,
    score_closed_form,
)
from .budget import (
    BudgetInputs,
    BudgetTerm,
    ErrorBudget,
    SensitivityCheck,
    build_budget,
    relative_acceleration,
    relative_jerk,
    staleness_position_error_m,
    staleness_velocity_error_mps,
    two_body_jerk,
    verify_sensitivities,
)
from .cases import Case, dump_case, load_case
from .divergence import (
    beam_fraction,
    error_for_loss_db,
    gaussian_divergence_half_angle,
    pointing_loss_db,
    pointing_loss_fraction,
    rayleigh_range,
)
from .geometry import (
    C_LIGHT,
    angle_between,
    orthonormal_basis,
    rotation_matrix,
    transverse,
    unit,
)
from .kepler import (
    MU_EARTH,
    R_EARTH,
    KeplerTrajectory,
    LinearTrajectory,
    propagate_two_body,
)
from .lighttime import (
    LightTimeSolution,
    light_time_residual,
    linear_light_time_closed_form,
    solve_light_time,
)
from .scenario import (
    DEFAULT_BEAM,
    NAMED_SCENARIOS,
    NARROW_BEAM,
    CircularOrbit,
    crosslink_from_orbits,
    geo_to_leo_pair,
    leo_crossing_pair,
    leo_trailing_pair,
    scenario_family,
)
from .terminal import BeamContext, Crosslink, TerminalState

__version__ = "0.1.0"

__all__ = [
    "ABERRATION_ROUTES",
    "C_LIGHT",
    "DEFAULT_BEAM",
    "EXACT_ROUTES",
    "MU_EARTH",
    "NAMED_SCENARIOS",
    "NARROW_BEAM",
    "R_EARTH",
    "BaselineScore",
    "BeamContext",
    "BudgetInputs",
    "BudgetTerm",
    "Case",
    "CircularOrbit",
    "Crosslink",
    "ErrorBudget",
    "KeplerTrajectory",
    "LightTimeSolution",
    "LinearTrajectory",
    "PointAheadDecomposition",
    "PointAheadResult",
    "SensitivityCheck",
    "TerminalState",
    "__version__",
    "aberrate",
    "aberrate_angle_relation",
    "aberrate_event_worldline",
    "aberrate_first_order",
    "aberrate_momentum_boost",
    "aberration_angle",
    "angle_between",
    "beam_fraction",
    "beta_from_velocity",
    "build_budget",
    "closed_form_point_ahead",
    "crosslink_from_orbits",
    "decompose_point_ahead",
    "dump_case",
    "error_for_loss_db",
    "gaussian_divergence_half_angle",
    "geo_to_leo_pair",
    "leo_crossing_pair",
    "leo_trailing_pair",
    "light_time_pair",
    "light_time_residual",
    "linear_light_time_closed_form",
    "load_case",
    "lorentz_gamma",
    "offset_difference",
    "orthonormal_basis",
    "pointing_loss_db",
    "pointing_loss_fraction",
    "propagate_two_body",
    "rayleigh_range",
    "reference_point_ahead",
    "relative_acceleration",
    "relative_jerk",
    "rotation_matrix",
    "route_disagreement",
    "scenario_family",
    "score_closed_form",
    "solve_light_time",
    "staleness_position_error_m",
    "staleness_velocity_error_mps",
    "transverse",
    "two_body_jerk",
    "unit",
    "verify_sensitivities",
]
