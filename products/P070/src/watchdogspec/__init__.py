"""watchdogspec: watchdog and liveness supervision as a checkable specification.

Declare heartbeat periods, missed-beat tolerances, liveness deadlines and
escalation ladders in YAML; this package computes the worst-case detection
latency, the false-trip rate implied by the declared jitter statistics, and
which rung of each escalation ladder can actually be reached before the
deadline, and reports the places where the declaration contradicts itself.

There is no learned component in this package and none is planned. See the
README section "No model in this one, deliberately".

Research-grade. Not flight-qualified, not certified, not approved for
operational aerospace use. Every number here is arithmetic over a declaration;
nothing here measures a running system.
"""

from __future__ import annotations

from .checks import CHECKS, CheckSpec, Finding, Report, Severity, check_spec, run_checks
from .diagram import render_escalation_png, render_mermaid
from .errors import SpecFormatError, WatchdogSpecError
from .falsetrip import (
    FalseTripRate,
    false_trip_probability,
    false_trip_rate,
    normal_upper_tail,
    spec_false_trip_rates,
)
from .latency import (
    DEFAULT_TOL_MS,
    REL_GUARD_ULP,
    Margin,
    RungTiming,
    TaskTiming,
    beat_window_ms,
    comparison_band_ms,
    max_tolerance_for_deadline,
    spec_timing,
    task_timing,
    verdict,
    worst_case_detection_latency_ms,
)
from .model import Action, Ladder, Rung, Spec, Supervisor, Task
from .parse import SPEC_VERSION, load_spec, parse_spec
from .report import (
    SCOPE_NOTE,
    format_codes,
    format_false_trip_table,
    format_findings,
    format_json,
    format_markdown,
    format_report,
    format_timing_table,
    report_to_dict,
)
from .simulate import (
    DetectionResult,
    FalseTripResult,
    simulate_detection,
    simulate_false_trips,
    wilson_interval,
)

__version__ = "0.1.0"

__all__ = [
    "CHECKS",
    "DEFAULT_TOL_MS",
    "SCOPE_NOTE",
    "SPEC_VERSION",
    "Action",
    "CheckSpec",
    "DetectionResult",
    "FalseTripRate",
    "FalseTripResult",
    "Finding",
    "Ladder",
    "Margin",
    "REL_GUARD_ULP",
    "Report",
    "Rung",
    "RungTiming",
    "Severity",
    "Spec",
    "SpecFormatError",
    "Supervisor",
    "Task",
    "TaskTiming",
    "WatchdogSpecError",
    "__version__",
    "beat_window_ms",
    "check_spec",
    "comparison_band_ms",
    "false_trip_probability",
    "false_trip_rate",
    "format_codes",
    "format_false_trip_table",
    "format_findings",
    "format_json",
    "format_markdown",
    "format_report",
    "format_timing_table",
    "load_spec",
    "max_tolerance_for_deadline",
    "normal_upper_tail",
    "parse_spec",
    "render_escalation_png",
    "render_mermaid",
    "report_to_dict",
    "run_checks",
    "simulate_detection",
    "simulate_false_trips",
    "spec_false_trip_rates",
    "spec_timing",
    "task_timing",
    "verdict",
    "wilson_interval",
    "worst_case_detection_latency_ms",
]
