"""rtclock -- real-time timing arithmetic and schedulability analysis for Python loops.

What this package is: the *arithmetic* of real-time timing, held to a stated
level of rigor -- unit conversions, period and deadline algebra, exact
percentiles with the definition named, utilization bounds and response-time
analysis with their references and validity ranges, priority-ceiling blocking,
and timing-budget composition with the clock's measured resolution carried as
an error term.

What it is not: a real-time kernel, a scheduler, or a profiler. CPython on a
general-purpose operating system is not a real-time platform -- there is no
bound on garbage-collection pauses, on the GIL, or on OS scheduling latency.
This package analyses and reports timing; it does not deliver it.

Research-grade, educational-to-research tier. Not flight-qualified, not
certified, not approved for operational aerospace use.
"""

from __future__ import annotations

from .budget import BudgetResult, Stage, compose_budget
from .histogram import (
    LatencyHistogram,
    OverrunReport,
    PercentileMethod,
    overrun_report,
    percentile,
)
from .loop import (
    FixedRateLoop,
    IterationRecord,
    LoopReport,
    ScheduleMode,
    absolute_mode_drift,
    relative_mode_drift,
)
from .schedulability import (
    ResponseTime,
    SchedulabilityResult,
    edf_test,
    hyperbolic_bound_test,
    priority_ceiling_blocking,
    response_time,
    response_time_analysis,
    rm_utilization_bound,
    rm_utilization_test,
)
from .taskset import PeriodicTask, TaskSet, hyperperiod_s
from .timebase import (
    ClockResolution,
    MonotonicTimebase,
    SimulatedTimebase,
    SkewedSimulatedTimebase,
    Timebase,
    duration_uncertainty_s,
    measure_clock_resolution,
)
from .units import (
    frequency_to_period,
    ms_to_s,
    ns_to_s,
    period_to_frequency,
    s_to_ms,
    s_to_ns,
    s_to_us,
    us_to_s,
)

__version__ = "0.1.0"

__all__ = [
    "BudgetResult",
    "ClockResolution",
    "FixedRateLoop",
    "IterationRecord",
    "LatencyHistogram",
    "LoopReport",
    "MonotonicTimebase",
    "OverrunReport",
    "PercentileMethod",
    "PeriodicTask",
    "ResponseTime",
    "ScheduleMode",
    "SchedulabilityResult",
    "SimulatedTimebase",
    "SkewedSimulatedTimebase",
    "Stage",
    "TaskSet",
    "Timebase",
    "__version__",
    "absolute_mode_drift",
    "compose_budget",
    "duration_uncertainty_s",
    "edf_test",
    "frequency_to_period",
    "hyperbolic_bound_test",
    "hyperperiod_s",
    "measure_clock_resolution",
    "ms_to_s",
    "ns_to_s",
    "overrun_report",
    "percentile",
    "period_to_frequency",
    "priority_ceiling_blocking",
    "relative_mode_drift",
    "response_time",
    "response_time_analysis",
    "rm_utilization_bound",
    "rm_utilization_test",
    "s_to_ms",
    "s_to_ns",
    "s_to_us",
    "us_to_s",
]
