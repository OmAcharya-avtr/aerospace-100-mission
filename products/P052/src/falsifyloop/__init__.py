"""falsifyloop -- requirement falsification for autonomous control loops.

Find the input or parameter setting that violates a stated requirement, and
measure how many simulations it took.

**Falsification is one-sided: finding no violation is not evidence of
correctness.** Every search in this package can only ever return a
counterexample or the statement that it found none within its budget. Nothing
here verifies anything.

Research-grade. Not flight-qualified, not certified, not approved for
operational aerospace use.

Layout
------
``traces``
    The uniformly sampled multi-signal trace the semantics is defined over.
``requirements``
    The requirement language, its robustness semantics, and an independently
    implemented Boolean semantics the robustness sign is property-tested
    against.
``systems``
    The simulator under test: a single-axis attitude loop with a rate-limited
    actuator and a sinusoidal gust. A synthetic benchmark, not a vehicle model.
``instances``
    Eight seeded benchmark instances of stated difficulty, sharing one box.
``search``
    Five strategies: uniform random (the baseline), Latin hypercube, simulated
    annealing, cross-entropy, and a random-forest surrogate-guided search.
``surrogate``
    The learned component and its ensemble-spread uncertainty output.
``curves``
    Sample-efficiency curves and their bootstrap bands.
``benchmark``
    The instance-by-strategy grid and its per-instance accounting.
``report``
    String rendering. The CLI is the only thing that writes to a stream.
"""

from __future__ import annotations

from .benchmark import BenchmarkReport, CellSummary, run_benchmark, run_cell
from .curves import (
    aggregate_curve,
    area_under_curve,
    bootstrap_aggregate_band,
    bootstrap_band,
    clopper_pearson,
    efficiency_curve,
    median_first_violation,
    success_rate,
)
from .instances import SEARCH_BOX, SUITE_ORDER, TIERS, Instance, instance, suite
from .requirements import (
    Abs,
    Always,
    And,
    Difference,
    Eventually,
    Formula,
    Or,
    Predicate,
    Signal,
    Term,
    check_horizon,
    robustness,
    satisfies,
    violated,
)
from .search import (
    BASELINE,
    STRATEGIES,
    SearchResult,
    analytic_random_curve,
    cross_entropy,
    latin_hypercube,
    simulated_annealing,
    strategy,
    surrogate_guided,
    uniform_random,
)
from .surrogate import ForestSurrogate
from .systems import (
    DEFAULT_DT,
    DEFAULT_HORIZON,
    LoopInput,
    LoopParameters,
    simulate,
    simulate_linear_zoh,
)
from .traces import Trace

__version__ = "0.1.0"

__all__ = [
    "BASELINE",
    "DEFAULT_DT",
    "DEFAULT_HORIZON",
    "SEARCH_BOX",
    "STRATEGIES",
    "SUITE_ORDER",
    "TIERS",
    "Abs",
    "Always",
    "And",
    "BenchmarkReport",
    "CellSummary",
    "Difference",
    "Eventually",
    "ForestSurrogate",
    "Formula",
    "Instance",
    "LoopInput",
    "LoopParameters",
    "Or",
    "Predicate",
    "SearchResult",
    "Signal",
    "Term",
    "Trace",
    "__version__",
    "aggregate_curve",
    "analytic_random_curve",
    "area_under_curve",
    "bootstrap_aggregate_band",
    "bootstrap_band",
    "check_horizon",
    "clopper_pearson",
    "cross_entropy",
    "efficiency_curve",
    "instance",
    "latin_hypercube",
    "median_first_violation",
    "robustness",
    "run_benchmark",
    "run_cell",
    "satisfies",
    "simulate",
    "simulate_linear_zoh",
    "simulated_annealing",
    "strategy",
    "success_rate",
    "suite",
    "surrogate_guided",
    "uniform_random",
    "violated",
]
