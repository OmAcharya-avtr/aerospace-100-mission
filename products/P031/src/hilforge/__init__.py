"""HilForge — a hardware-in-the-loop harness for a GNC or comms control loop.

What this package is
--------------------
A hardware abstraction layer with two interchangeable backends, a fixed-rate
loop with deadline and overrun accounting, dry-run rehearsal, deterministic
seeded replay, executable pre-run and recovery checks, a benchmark harness
that records its own measurement method, and a deadline-overrun predictor
benchmarked against two deterministic baselines.

Validation level
----------------
**Level 3, hardware-pending.** Everything in this package has been run against
a simulated backend and against a device backend pointed at a loopback stub.
Nothing in it has been run on a board. No number produced by this package is a
hardware measurement unless it came from a run whose
:attr:`hilforge.hal.BackendInfo.is_hardware` was ``True``, and no such run has
happened. What is still missing, and what would be required to close it, is
stated in ``README.md`` and ``validation/VALIDATION.md``.

This software is research-grade. It is not flight-qualified, not certified,
and not approved for operational aerospace use.

Units
-----
Seconds for every time and duration, radians and radians per second for
attitude and rate, newton-metres for torque. Units are in every public
docstring; nothing in this package infers units from a variable name.
"""

from __future__ import annotations

__version__ = "0.1.0"
__license__ = "AGPL-3.0-or-later"
__copyright__ = "© 2026 OPTIMA Organisation"

from .backends import (
    AbsentDriver,
    DeviceBackend,
    DriverShim,
    LoopbackDriver,
    SimulatedBackend,
    make_backend_pair,
)
from .bench import BenchmarkRecord, run_benchmark, write_record
from .deploy import CheckReport, CheckResult, RunGuard, preflight
from .dryrun import DryRunActuator
from .errors import (
    BackendError,
    ConfigurationError,
    DeviceAbsentError,
    DeviceDisconnectedError,
    DryRunViolationError,
    HilForgeError,
    OverrunCascadeError,
    PreflightFailure,
    RecoveryError,
    SampleDroppedError,
    TimebaseRegressionError,
    WriteRejectedError,
)
from .hal import ActuatorChannel, Backend, BackendInfo, ChannelSpec, SensorChannel, WriteAck
from .loop import STAGES, HilLoop, IterationRecord, LoopConfig, RunRecord
from .plant import AttitudePlant, PDController, PDGains, PlantConfig
from .replay import ReplayBackend, load_samples, replay_run, save_run
from .timebase import (
    ClockReport,
    MonotonicTimebase,
    Timebase,
    VirtualTimebase,
    clock_report,
    duration_resolution_uncertainty,
)
from .timing import (
    LatencyHistogram,
    MonotonicGuard,
    OverrunAccount,
    PeriodSpec,
    TimingUncertainty,
    overrun_report,
    quantile_standard_error,
    timing_uncertainty,
)

__all__ = [
    "STAGES",
    "AbsentDriver",
    "ActuatorChannel",
    "AttitudePlant",
    "Backend",
    "BackendError",
    "BackendInfo",
    "BenchmarkRecord",
    "ChannelSpec",
    "CheckReport",
    "CheckResult",
    "ClockReport",
    "ConfigurationError",
    "DeviceAbsentError",
    "DeviceBackend",
    "DeviceDisconnectedError",
    "DriverShim",
    "DryRunActuator",
    "DryRunViolationError",
    "HilForgeError",
    "HilLoop",
    "IterationRecord",
    "LatencyHistogram",
    "LoopConfig",
    "LoopbackDriver",
    "MonotonicGuard",
    "MonotonicTimebase",
    "OverrunAccount",
    "OverrunCascadeError",
    "PDController",
    "PDGains",
    "PeriodSpec",
    "PlantConfig",
    "PreflightFailure",
    "RecoveryError",
    "ReplayBackend",
    "RunGuard",
    "RunRecord",
    "SampleDroppedError",
    "SensorChannel",
    "SimulatedBackend",
    "Timebase",
    "TimebaseRegressionError",
    "TimingUncertainty",
    "VirtualTimebase",
    "WriteAck",
    "WriteRejectedError",
    "__copyright__",
    "__license__",
    "__version__",
    "clock_report",
    "duration_resolution_uncertainty",
    "load_samples",
    "make_backend_pair",
    "overrun_report",
    "preflight",
    "quantile_standard_error",
    "replay_run",
    "run_benchmark",
    "save_run",
    "timing_uncertainty",
    "write_record",
]
