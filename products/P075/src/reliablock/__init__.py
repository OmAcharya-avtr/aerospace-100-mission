"""reliablock: redundancy arithmetic that is explicit about the assumption doing the work.

Takes a **declared** reliability block diagram -- series, parallel, k-of-n,
standby with a declared switching reliability, shared load, and common cause as
a declared beta factor -- with per-block failure distributions, and computes:

- system reliability over a mission profile, in closed form;
- the same figure **with and without the declared common-cause beta, side by
  side**, which is the headline, because independence is the assumption that
  fails in flight hardware and the one most often left implicit;
- minimal cut sets, with the single points of failure called out;
- Birnbaum, criticality and Fussell-Vesely importance measures, computed
  exactly rather than by the rare-event approximation.

There is **no learned component in this package, deliberately.** For a declared
diagram with declared distributions the reliability has a correct answer, so a
model fitted to approximate it could only be worse than the arithmetic, and
there is nothing in the problem for it to learn. Monte Carlo
(:mod:`reliablock.montecarlo`) exists to be scored against the closed forms,
not to replace them.

The `reliability` package on PyPI is the better tool for distribution fitting,
accelerated life testing and repairable-system analysis, and the README says so
and says when to reach for it instead. What this package adds is the declared
diagram as a checkable artifact and the explicit accounting of what rests on the
independence assumption.

Research-grade software. Not flight-qualified, not certified, not approved for
operational aerospace use. A reliability number is not a safety case.
"""

from __future__ import annotations

from .bounds import Bracket, bonferroni_series_bounds, esary_proschan_bounds
from .bridge import (
    bridge_diagram,
    bridge_reliability_equal,
    bridge_reliability_inclusion_exclusion,
    bridge_reliability_pivotal,
)
from .cases import CASES, MISSIONS, case, mission
from .commoncause import (
    GroupBetaEffect,
    ScaledHazard,
    apply_common_cause,
    group_beta_effects,
    strip_common_cause,
)
from .diagram import (
    Block,
    Diagram,
    KOfN,
    Network,
    Node,
    Parallel,
    Series,
    SharedLoad,
    Standby,
    load_diagram,
    node_from_mapping,
)
from .distributions import (
    Exponential,
    FailureDistribution,
    FixedProbability,
    Lognormal,
    Weibull,
    distribution_from_mapping,
)
from .importance import (
    Importance,
    boolean_system_unreliability,
    importance_table,
    pivot_unreliability,
    system_unreliability_at,
)
from .mission import MissionPhase, MissionProfile, load_mission
from .montecarlo import (
    MonteCarloResult,
    simulate,
    trials_for_half_width,
    wilson_interval,
)
from .quantify import (
    Quantity,
    quantify,
    shared_load_reliability,
    standby_reliability_cold_identical,
    standby_reliability_general,
)
from .report import (
    common_cause_comparison,
    cut_set_report,
    diagram_report,
    importance_report,
    mission_report,
    quantity_tree,
)
from .structure import (
    basic_events,
    enumerate_states,
    minimal_cut_sets,
    minimal_path_sets,
    structure_function,
)

__version__ = "0.1.0"

#: Stated once, here, and repeated in the README: this package has no learned
#: component and will not be given one.
AI_ENABLED = False

__all__ = [
    "AI_ENABLED",
    "Block",
    "Bracket",
    "CASES",
    "Diagram",
    "Exponential",
    "FailureDistribution",
    "FixedProbability",
    "GroupBetaEffect",
    "Importance",
    "KOfN",
    "Lognormal",
    "MISSIONS",
    "MissionPhase",
    "MissionProfile",
    "MonteCarloResult",
    "Network",
    "Node",
    "Parallel",
    "Quantity",
    "ScaledHazard",
    "Series",
    "SharedLoad",
    "Standby",
    "Weibull",
    "__version__",
    "apply_common_cause",
    "basic_events",
    "bonferroni_series_bounds",
    "boolean_system_unreliability",
    "bridge_diagram",
    "bridge_reliability_equal",
    "bridge_reliability_inclusion_exclusion",
    "bridge_reliability_pivotal",
    "case",
    "common_cause_comparison",
    "cut_set_report",
    "diagram_report",
    "distribution_from_mapping",
    "enumerate_states",
    "esary_proschan_bounds",
    "group_beta_effects",
    "importance_report",
    "importance_table",
    "load_diagram",
    "load_mission",
    "minimal_cut_sets",
    "minimal_path_sets",
    "mission",
    "mission_report",
    "node_from_mapping",
    "pivot_unreliability",
    "quantify",
    "quantity_tree",
    "shared_load_reliability",
    "simulate",
    "standby_reliability_cold_identical",
    "standby_reliability_general",
    "strip_common_cause",
    "structure_function",
    "system_unreliability_at",
    "trials_for_half_width",
    "wilson_interval",
]
