"""constellink -- time-varying constellation contact graphs, routing and link capacity.

Research-grade software.  Not flight-qualified, not certified, not approved for
operational aerospace use.

Orbit propagation is delegated to the ``sgp4`` package; this package depends on
it and does not replace it.  See the README for the alternatives table.

Public surface
--------------
Geometry and propagation
    :mod:`constellink.frames`, :mod:`constellink.geometry`,
    :mod:`constellink.constellation`
Contacts and graphs
    :mod:`constellink.contacts`, :mod:`constellink.graph`
Routing and scheduling
    :mod:`constellink.routing`, :mod:`constellink.flow`,
    :mod:`constellink.queueing`
Link capacity
    :mod:`constellink.capacity`
Link-availability prediction
    :mod:`constellink.synthdata`, :mod:`constellink.availability`,
    :mod:`constellink.metrics`
Level 3 extras
    :mod:`constellink.uncertainty`, :mod:`constellink.benchmark`
"""

from __future__ import annotations

__version__ = "0.1.0"
__license__ = "AGPL-3.0"

from .availability import (
    ClimatologyBaseline,
    LinkAvailabilityModel,
    LogisticBaseline,
    PredictorScores,
    grouped_split,
    score_predictor,
)
from .capacity import (
    OpticalTerminal,
    RfTerminal,
    free_space_path_loss_db,
    hybrid_capacity_bps,
    optical_link,
    rf_link,
)
from .constellation import (
    TLE,
    Constellation,
    GroundStation,
    PropagationError,
    Satellite,
    TleEpochError,
    walker_delta,
)
from .contacts import (
    ContactWindow,
    all_contact_windows,
    contact_windows_ground,
    contact_windows_isl,
)
from .flow import FlowResult, NoSolverError, brute_force_max_flow, ilp_max_flow
from .geometry import ground_max_central_angle, isl_clear, max_isl_central_angle
from .graph import ContactGraph, TimeExpandedGraph
from .metrics import brier_decomposition, brier_score, reliability_curve
from .queueing import md1_mean_delay_s, mm1_mean_delay_s, route_delay
from .routing import Route, enumerate_routes, shortest_route
from .synthdata import DatasetConfig, LinkDataset, generate_dataset

__all__ = [
    "__version__",
    "__license__",
    "TLE",
    "Satellite",
    "GroundStation",
    "Constellation",
    "walker_delta",
    "TleEpochError",
    "PropagationError",
    "ContactWindow",
    "contact_windows_isl",
    "contact_windows_ground",
    "all_contact_windows",
    "ContactGraph",
    "TimeExpandedGraph",
    "Route",
    "shortest_route",
    "enumerate_routes",
    "FlowResult",
    "ilp_max_flow",
    "brute_force_max_flow",
    "NoSolverError",
    "RfTerminal",
    "OpticalTerminal",
    "rf_link",
    "optical_link",
    "free_space_path_loss_db",
    "hybrid_capacity_bps",
    "isl_clear",
    "max_isl_central_angle",
    "ground_max_central_angle",
    "mm1_mean_delay_s",
    "md1_mean_delay_s",
    "route_delay",
    "DatasetConfig",
    "LinkDataset",
    "generate_dataset",
    "ClimatologyBaseline",
    "LogisticBaseline",
    "LinkAvailabilityModel",
    "PredictorScores",
    "score_predictor",
    "grouped_split",
    "brier_score",
    "brier_decomposition",
    "reliability_curve",
]
