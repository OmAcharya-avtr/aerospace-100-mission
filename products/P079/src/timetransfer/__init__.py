"""timetransfer -- clock offset and skew across a link, with the uncertainty measured.

Two-way time transfer: offset, skew and drift from four-timestamp exchanges
under asymmetric and variable delay; stability as an Allan deviation over tau
with a confidence interval at each tau; and a **measured** uncertainty on the
offset estimate itself, not a nominal figure from a datasheet.

This is a measurement and characterisation tool, not a time-synchronisation
daemon. It does not steer a clock, it does not implement NTP or PTP, and it
should not be used where ``linuxptp`` or ``chrony`` is the answer. See
README.md, "Alternatives, honestly".

Where to start
--------------
* :mod:`timetransfer.twoway` -- the symmetric-delay estimator and its exact
  asymmetry bias. Read this first; everything else is downstream of equation
  (5) in it.
* :mod:`timetransfer.skew` -- least-squares and Kalman offset/skew/drift.
* :mod:`timetransfer.uncertainty` -- the measured uncertainty budget and its
  measured coverage.
* :mod:`timetransfer.allan` -- Allan deviation over tau with intervals.
* :mod:`timetransfer.hal` -- the hardware abstraction layer; the simulated and
  device backends behind it are in :mod:`timetransfer.backends`.
* :mod:`timetransfer.asym` -- the analytic asymmetry estimators and the learned
  corrector, scored against each other on the same held-out split.

Research grade. Not flight-qualified, not certified, not approved for
operational aerospace use.
"""

from __future__ import annotations

from .acquire import AcquisitionReport, acquire
from .allan import AllanTable, allan_table, fit_log_slope, identify_noise
from .backends import (
    DeviceBackend,
    LoopbackDriver,
    RecordedDriver,
    SimulatedBackend,
    SimulationSpec,
    make_backend_pair,
)
from .bench import HARDWARE_STAGES, BenchmarkRecord, hardware_rows, run_benchmark
from .clockmodel import ClockModel, ClockSpec
from .deploy import CheckReport, RunGuard, preflight, recover_truncated_estimate
from .dryrun import dry_run, dry_run_negative_control, record_exchanges
from .errors import (
    ConfigurationError,
    DeviceAbsentError,
    ExchangeLostError,
    HardwareClaimError,
    InsufficientDataError,
    PreflightFailure,
    RecoveryError,
    TimebaseRegressionError,
    TimeTransferError,
)
from .exchange import Exchange, ExchangeSet
from .hal import Backend, BackendInfo, ExchangeOutcome, ExchangeTruth
from .linkmodel import LinkModel, LinkSpec
from .skew import KalmanConfig, KalmanResult, LeastSquaresFit, fit_least_squares, kalman_skew
from .timebase import MonotonicTimebase, VirtualTimebase, clock_report
from .twoway import TwoWayEstimate, asymmetry_bias_s, two_way_estimate
from .uncertainty import AsymmetrySource, CoverageResult, measure_coverage, offset_uncertainty

__version__ = "0.1.0"

VALIDATION_LABEL = "Level 3, hardware-pending"
"""The validation label. Never Level 4: see ``bench.HARDWARE_STAGES``."""

__all__ = [
    "HARDWARE_STAGES",
    "VALIDATION_LABEL",
    "AcquisitionReport",
    "AllanTable",
    "AsymmetrySource",
    "Backend",
    "BackendInfo",
    "BenchmarkRecord",
    "CheckReport",
    "ClockModel",
    "ClockSpec",
    "ConfigurationError",
    "CoverageResult",
    "DeviceAbsentError",
    "DeviceBackend",
    "Exchange",
    "ExchangeLostError",
    "ExchangeOutcome",
    "ExchangeSet",
    "ExchangeTruth",
    "HardwareClaimError",
    "InsufficientDataError",
    "KalmanConfig",
    "KalmanResult",
    "LeastSquaresFit",
    "LinkModel",
    "LinkSpec",
    "LoopbackDriver",
    "MonotonicTimebase",
    "PreflightFailure",
    "RecordedDriver",
    "RecoveryError",
    "RunGuard",
    "SimulatedBackend",
    "SimulationSpec",
    "TimeTransferError",
    "TimebaseRegressionError",
    "TwoWayEstimate",
    "VirtualTimebase",
    "__version__",
    "acquire",
    "allan_table",
    "asymmetry_bias_s",
    "clock_report",
    "dry_run",
    "dry_run_negative_control",
    "fit_least_squares",
    "fit_log_slope",
    "hardware_rows",
    "identify_noise",
    "kalman_skew",
    "make_backend_pair",
    "measure_coverage",
    "offset_uncertainty",
    "preflight",
    "recover_truncated_estimate",
    "record_exchanges",
    "run_benchmark",
    "two_way_estimate",
]
