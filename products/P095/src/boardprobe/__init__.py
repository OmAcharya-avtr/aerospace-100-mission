"""boardprobe -- a hardware abstraction layer whose readings carry their provenance.

A uniform interface over an NVIDIA Jetson Orin Nano's clock, die-temperature,
rail-power and performance-counter sources, behind three backends -- live,
recorded and simulated -- with the provenance of every reading carried in the
value and propagated through every derivation, so that a figure derived from
simulated data cannot be labelled a measurement.

Validation level 3, hardware-pending. The board was not reachable from the
environment this package was built in, so the live reading path is structurally
complete and unexercised against hardware, every declared sysfs path and
counter name is taken from documentation and is UNVERIFIED, and **no figure in
this repository was measured on a Jetson Orin Nano.** Level 4 requires measured
timing and resource figures from the board itself.

This software is research-grade. It is not flight-qualified, not certified and
not approved for operational aerospace use.

The one rule
------------
Provenance is the type system here, not a logging field:

>>> from boardprobe import ORIN_NANO, Label, SimulatedSource, attest_platform
>>> from boardprobe import attribute, certify
>>> source = SimulatedSource(ORIN_NANO, seed=7)
>>> readings = [source.read("die_temp_cpu") for _ in range(4)]
>>> attribution = attribute(readings)
>>> attribution.provenance
<Provenance.SIMULATED: 'simulated'>
>>> certify("die temperature mean", attribution, Label.MEASURED, attest_platform())
Traceback (most recent call last):
boardprobe.certify.LabelViolation: ...
"""

from __future__ import annotations

from .certify import Certificate, Label, LabelViolation, certify, label_for
from .channels import Access, BoardDeclaration, Channel, ChannelKind
from .dryrun import DryRunReport, dry_run
from .harness import (
    BenchmarkResult,
    HarnessError,
    OverheadReport,
    benchmark,
    characterise_overhead,
    reference_payload,
    timing_resolution_s,
)
from .orin_nano import ORIN_NANO, RAIL_POWER_PAIRS, TARGET_MODEL_PREFIX
from .platform_attest import PlatformAttestation, attest_platform
from .provenance import (
    ABSORBING,
    IDENTITY,
    ORDER,
    Provenance,
    ProvenanceError,
    join,
    leq,
    meet,
    meet_all,
    rank,
)
from .readings import (
    Attribution,
    Reading,
    TamperError,
    attribute,
    combine,
    from_mapping,
    mint,
    verify,
)
from .report import (
    format_table,
    render_benchmark,
    render_certificate,
    render_channels,
    render_label_rules,
    render_lattice,
    render_overhead,
    render_readings,
    render_statistics,
)
from .sources import (
    LiveSource,
    RecordedSource,
    SimulatedSource,
    Source,
    backend_names,
    make_source,
    read_available,
    sealed_backends,
)
from .statistics import Statistic, derive, rail_power, summarise
from .sysio import DeviceUnreachableError, IoLedger, ledger_window
from .trace import Trace, TraceError, TraceSample

__version__ = "0.1.0"

#: The exact validation-level phrase this product may use. The registry records
#: level 4 as the SLOT's target; this package is level 3 until a figure in it
#: comes from the board.
VALIDATION_LEVEL = "Level 3, hardware-pending"

__all__ = [
    "ABSORBING",
    "IDENTITY",
    "ORDER",
    "ORIN_NANO",
    "RAIL_POWER_PAIRS",
    "TARGET_MODEL_PREFIX",
    "VALIDATION_LEVEL",
    "Access",
    "Attribution",
    "BenchmarkResult",
    "BoardDeclaration",
    "Certificate",
    "Channel",
    "ChannelKind",
    "DeviceUnreachableError",
    "DryRunReport",
    "HarnessError",
    "IoLedger",
    "Label",
    "LabelViolation",
    "LiveSource",
    "OverheadReport",
    "PlatformAttestation",
    "Provenance",
    "ProvenanceError",
    "Reading",
    "RecordedSource",
    "SimulatedSource",
    "Source",
    "Statistic",
    "TamperError",
    "Trace",
    "TraceError",
    "TraceSample",
    "__version__",
    "attest_platform",
    "attribute",
    "backend_names",
    "benchmark",
    "certify",
    "characterise_overhead",
    "combine",
    "derive",
    "dry_run",
    "format_table",
    "from_mapping",
    "join",
    "label_for",
    "ledger_window",
    "leq",
    "make_source",
    "meet",
    "meet_all",
    "mint",
    "rail_power",
    "rank",
    "read_available",
    "reference_payload",
    "render_benchmark",
    "render_certificate",
    "render_channels",
    "render_label_rules",
    "render_lattice",
    "render_overhead",
    "render_readings",
    "render_statistics",
    "sealed_backends",
    "summarise",
    "timing_resolution_s",
    "verify",
]
