"""telemgap -- separate missing telemetry from corrupt telemetry from a wrong reader.

Takes a decoded telemetry record with timestamps and produces an integrity
report: gaps with their durations, duplicate and out-of-order frames, clock
resets and backward jumps, and a classification of each anomaly into
*missing*, *corrupt*, or *correctly-received-and-misparsed*. Emits the
gap-handling policy explicitly -- which gaps may be interpolated across at a
declared tolerance, which must be left as gaps, and why.

The first output of every entry point here is a determinability verdict: what
the supplied record **cannot** tell you. It is computed before any
classification and it is returned, not documented.

Research grade. Not flight-qualified, not certified, not approved for
operational aerospace use.

Typical use
-----------
>>> from telemgap import dataset, check, policy
>>> generated = dataset.test_record(samples=600)
>>> report = check(generated.record)
>>> report.determinability["in_range_corruption"].determinable
False
>>> plan = policy.plan(
...     generated.record,
...     report,
...     tolerances={name: 0.05 for name in generated.record.schema.channel_names},
... )
>>> bool(len(plan.refusals) >= 0)
True
"""

from __future__ import annotations

from . import (
    benchmark,
    classify,
    dataset,
    determinability,
    features,
    interpolate,
    learned,
    policy,
    reparse,
    report,
    screens,
)
from .classify import Anomaly, AnomalyClass, IntegrityReport
from .classify import classify as classify_record
from .determinability import DeterminabilityVerdict, Limit
from .interpolate import Reconstruction, reconstruct
from .policy import GapDecision, PolicyPlan
from .record import ChannelSpec, RecordSchema, TelemetryRecord
from .screens import Gap, ScreenFindings
from .screens import screen as screen_record

__version__ = "0.1.0"

__all__ = [
    "Anomaly",
    "AnomalyClass",
    "ChannelSpec",
    "DeterminabilityVerdict",
    "Gap",
    "GapDecision",
    "IntegrityReport",
    "Limit",
    "PolicyPlan",
    "Reconstruction",
    "RecordSchema",
    "ScreenFindings",
    "TelemetryRecord",
    "__version__",
    "benchmark",
    "check",
    "classify",
    "classify_record",
    "dataset",
    "determinability",
    "features",
    "interpolate",
    "learned",
    "policy",
    "reconstruct",
    "reparse",
    "report",
    "screen_record",
    "screens",
]


def check(
    record: TelemetryRecord,
    mode: str = "strengthened",
    residual_window: int = 8,
    residual_z: float = 8.0,
) -> IntegrityReport:
    """Screen, assess determinability, and classify ``record`` in one call.

    This is the package's front door. See
    :func:`telemgap.classify.classify` for the parameters and
    :class:`telemgap.classify.IntegrityReport` for the result, whose first
    field is the determinability verdict.
    """
    return classify.classify(
        record, mode=mode, residual_window=residual_window, residual_z=residual_z
    )
