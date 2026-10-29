"""tlecheck: screen a two-line element set before propagating it.

Screens an element set for format and checksum validity, epoch staleness
against a declared requirement, element plausibility against bounds from
orbital mechanics and against the object's own element history, ephemeris
continuity against its own predecessor, and cross-source disagreement; and
classifies each problem as one the propagator will report loudly, one that will
produce a confidently wrong ephemeris, or one that is benign.

The classification is a **measured consequence**, not an assumed label: the
class boundary comes from propagating the element set with ``sgp4`` and
measuring the position error in kilometres against an uncorrupted propagation.

The first output of every screen is a report of what the screen could not
determine. A passing screen does not mean a propagated ephemeris is accurate;
nothing in this package can establish that, and the report says so as computed
output rather than as a caveat.

This package does not propagate orbits and does not replace ``sgp4``. It
screens *for* it. It also ships no element catalogue: the authoritative source
of element sets is an operator's catalogue, which is not redistributable, so
every element set here is constructed from declared element values by
:mod:`tlecheck.dataset`.

Research grade. Not flight-qualified, not certified, not approved for
operational aerospace use.
"""

from __future__ import annotations

from . import (
    benchmark,
    bounds,
    consequence,
    continuity,
    dataset,
    determinability,
    divergence,
    elements,
    epoch,
    features,
    history,
    learned,
    propagate,
    report,
    screens,
    verdict,
)
from .consequence import BENIGN, CLASSES, SILENTLY_WRONG, WILL_FAIL_LOUDLY
from .elements import ElementSet, checksum, format_tle, parse_tle
from .verdict import Verdict, screen

__version__ = "0.1.0"

__all__ = [
    "BENIGN",
    "CLASSES",
    "SILENTLY_WRONG",
    "WILL_FAIL_LOUDLY",
    "ElementSet",
    "Verdict",
    "__version__",
    "benchmark",
    "bounds",
    "checksum",
    "consequence",
    "continuity",
    "dataset",
    "determinability",
    "divergence",
    "elements",
    "epoch",
    "features",
    "format_tle",
    "history",
    "learned",
    "parse_tle",
    "propagate",
    "report",
    "screen",
    "screens",
    "verdict",
]
