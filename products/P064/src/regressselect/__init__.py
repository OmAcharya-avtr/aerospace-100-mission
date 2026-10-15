"""regressselect: regression test selection with measured safety loss.

Three strategies over the same change, scored on the same held-out revisions:

1. ``importgraph`` -- Baseline 1, dependency-based selection. Safe with
   respect to the declared import graph, and the README says plainly how much
   of the suite that costs.
2. ``prioritise`` -- Baseline 2, history-based co-failure prioritisation,
   measured with APFD.
3. ``selector`` -- the learned selector, a ranked order with a probability per
   test, benchmarked against both baselines on APFD *and* safety loss.

The quantity this package exists to report is the safety loss: the fraction of
real failures a selection would have missed. It is reported whether or not it
flatters the learned component. On the bundled fixtures it does not; see
``README.md`` and ``validation/VALIDATION.md``.

Research-grade software. Not flight-qualified, not certified, and not approved
for operational aerospace use.
"""

from .benchmark import (
    BudgetPoint,
    MutationPoint,
    StrategyScore,
    dependency_order,
    evaluate_on_fault_matrix,
    evaluate_strategies,
    fault_matrix_apfd,
)
from .features import FEATURE_NAMES, build_design_matrix, feature_row
from .history import History, Revision, load_history, save_history
from .importgraph import ImportGraph, build_import_graph
from .inventory import TestInventory, collect_tests, module_of_node
from .metrics import (
    apfd,
    apfd_single_fault,
    detection_curve,
    first_detection_positions,
    safety_loss,
    selection_cost,
)
from .mutation import (
    FaultMatrix,
    SeededFault,
    SuiteResult,
    load_faults,
    run_mutation_evaluation,
    run_suite,
)
from .prioritise import CoFailurePrioritiser, RankedTest
from .selector import LearnedSelector, SelectorConfig

__version__ = "0.1.0"

__all__ = [
    "FEATURE_NAMES",
    "BudgetPoint",
    "CoFailurePrioritiser",
    "FaultMatrix",
    "History",
    "ImportGraph",
    "LearnedSelector",
    "MutationPoint",
    "RankedTest",
    "Revision",
    "SeededFault",
    "SelectorConfig",
    "StrategyScore",
    "SuiteResult",
    "TestInventory",
    "__version__",
    "apfd",
    "apfd_single_fault",
    "build_design_matrix",
    "build_import_graph",
    "collect_tests",
    "dependency_order",
    "detection_curve",
    "evaluate_on_fault_matrix",
    "evaluate_strategies",
    "fault_matrix_apfd",
    "feature_row",
    "first_detection_positions",
    "load_faults",
    "load_history",
    "module_of_node",
    "run_mutation_evaluation",
    "run_suite",
    "safety_loss",
    "save_history",
    "selection_cost",
]
