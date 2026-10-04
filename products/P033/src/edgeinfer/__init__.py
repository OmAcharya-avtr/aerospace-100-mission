"""edgeinfer --- declared budgets for aerospace edge inference.

What it does
------------
Treats a deployment envelope --- worst-case latency, peak memory, a power
ceiling, a duty cycle --- as a declared object, then puts two numbers next to
it for a candidate model: an **analytic estimate** computed from the model
graph without running anything, and a **measured profile** from a real
``onnxruntime`` session, each with its measurement method, repeat count and
uncertainty. Worst-case and median latency are reported separately throughout,
because a control loop is sized by its tail.

Validation level
----------------
**Level 3, hardware-pending.** Everything in this package has been exercised
on a shared single-CPU-core cloud container, and every output file says so in
its own text. No measurement from an edge target exists, so the target-device
column of every results table is empty. What is missing is stated in
``README.md``: measured timing and resource use from a Jetson Orin Nano
itself.

Research-grade software. Not flight-qualified, not certified, not approved for
operational aerospace use.

Module map
----------
======================  =================================================
:mod:`~edgeinfer.budget`      the declared envelope, pass/fail, feasibility
:mod:`~edgeinfer.graph`       static model-graph IR
:mod:`~edgeinfer.ops`         per-operator op counts and memory traffic
:mod:`~edgeinfer.roofline`    roofline bound (Williams et al. 2009) and
                              device model calibration
:mod:`~edgeinfer.analytic`    the analytic baseline: latency and peak memory
:mod:`~edgeinfer.onnx_io`     build and parse ONNX bytes without ``onnx``
:mod:`~edgeinfer.backends`    simulated, ``onnxruntime`` and sklearn backends
:mod:`~edgeinfer.harness`     repeated timing, tail reported separately
:mod:`~edgeinfer.memtrace`    peak-memory measurement
:mod:`~edgeinfer.uncertainty` the uncertainty budget (GUM, bootstrap)
:mod:`~edgeinfer.thermal`     declared throttle derating
:mod:`~edgeinfer.dataset`     reproducible synthetic graph population
:mod:`~edgeinfer.features`    graph features for the learned predictor
:mod:`~edgeinfer.predictor`   the learned predictor and the comparison
:mod:`~edgeinfer.report`      result files with the empty device column
======================  =================================================
"""

from __future__ import annotations

from edgeinfer.analytic import AnalyticEstimate, analytic_estimate, peak_activation_bytes
from edgeinfer.backends import (
    InferenceBackend,
    OnnxRuntimeBackend,
    PipelineStage,
    SimulatedBackend,
    SklearnBackend,
)
from edgeinfer.budget import Budget, BudgetReport, CheckRow, Verdict, build_report
from edgeinfer.graph import GraphError, ModelGraph, Node, TensorSpec
from edgeinfer.harness import LatencyProfile, benchmark
from edgeinfer.memtrace import PeakMemory, measure_peak_python_bytes
from edgeinfer.onnx_io import OnnxModel, build_model, parse_model
from edgeinfer.ops import OpCost, UnsupportedOperatorError, node_cost
from edgeinfer.predictor import LatencyPredictor, compare_predictors, evaluate_predictions
from edgeinfer.report import ResultRow, write_results_file
from edgeinfer.roofline import DeviceModel, calibrate_device, roofline_time_s
from edgeinfer.thermal import ThrottleState, throttled_device
from edgeinfer.uncertainty import UncertaintyBudget, uncertainty_budget

__version__ = "0.1.0"

#: Validation level of this package, verbatim. Level 4 would require measured
#: timing and resource use from the target device itself, which does not exist
#: for this repository.
VALIDATION_LEVEL = "3, hardware-pending"

__all__ = [
    "VALIDATION_LEVEL",
    "AnalyticEstimate",
    "Budget",
    "BudgetReport",
    "CheckRow",
    "DeviceModel",
    "GraphError",
    "InferenceBackend",
    "LatencyPredictor",
    "LatencyProfile",
    "ModelGraph",
    "Node",
    "OnnxModel",
    "OnnxRuntimeBackend",
    "OpCost",
    "PeakMemory",
    "PipelineStage",
    "ResultRow",
    "SimulatedBackend",
    "SklearnBackend",
    "TensorSpec",
    "ThrottleState",
    "UncertaintyBudget",
    "UnsupportedOperatorError",
    "Verdict",
    "__version__",
    "analytic_estimate",
    "benchmark",
    "build_model",
    "build_report",
    "calibrate_device",
    "compare_predictors",
    "evaluate_predictions",
    "measure_peak_python_bytes",
    "node_cost",
    "parse_model",
    "peak_activation_bytes",
    "roofline_time_s",
    "throttled_device",
    "uncertainty_budget",
    "write_results_file",
]
