"""coderateopt: availability-constrained code-rate selection on a fading optical link.

Research-grade. Not flight-qualified, not certified, not approved for
operational aerospace use.

The problem, the assumptions behind it and two structural results that bound
what an optimiser can usefully add are written out in
:mod:`coderateopt.problem`. Read that module docstring before the numbers.
"""

from __future__ import annotations

from .exhaustive import (
    count_subsets,
    enumerate_subsets,
    solve_closed_form_k2,
    solve_exhaustive,
)
from .fade import (
    AvailabilityModel,
    EmpiricalFade,
    GammaGammaFade,
    LognormalFade,
    db_to_linear,
    gamma_gamma_pdf,
    linear_to_db,
    scintillation_from_log_amplitude,
    sigma_ln_i_from_scintillation,
)
from .heuristics import (
    HeuristicComparison,
    compare_to_optimum,
    highest_feasible_rate,
    highest_rate_within_reserve,
)
from .lp import restricted_lp
from .milp import MilpEncoding, assert_feasible, build_milp, solve_milp
from .modcod import Modcod, ModcodSet, illustrative_modcod_table
from .problem import InfeasibleProblem, RateProblem, Solution
from .select import select_rate
from .sensitivity import (
    SensitivityReport,
    margin_sensitivity,
    scintillation_sensitivity,
    stability_interval,
    target_sensitivity,
)

__version__ = "0.1.0"

__all__ = [
    "AvailabilityModel",
    "EmpiricalFade",
    "GammaGammaFade",
    "HeuristicComparison",
    "InfeasibleProblem",
    "LognormalFade",
    "MilpEncoding",
    "Modcod",
    "ModcodSet",
    "RateProblem",
    "SensitivityReport",
    "Solution",
    "__version__",
    "assert_feasible",
    "build_milp",
    "compare_to_optimum",
    "count_subsets",
    "db_to_linear",
    "enumerate_subsets",
    "gamma_gamma_pdf",
    "highest_feasible_rate",
    "highest_rate_within_reserve",
    "illustrative_modcod_table",
    "linear_to_db",
    "margin_sensitivity",
    "restricted_lp",
    "scintillation_from_log_amplitude",
    "scintillation_sensitivity",
    "select_rate",
    "sigma_ln_i_from_scintillation",
    "solve_closed_form_k2",
    "solve_exhaustive",
    "solve_milp",
    "stability_interval",
    "target_sensitivity",
]
