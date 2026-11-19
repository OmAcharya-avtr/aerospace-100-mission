"""evidencestale -- measure how much of an assurance case rests on stale evidence.

A traceability tool checks that a link EXISTS between a claim and its evidence.
This package checks whether that link is still TRUE: whether the artifact the
evidence is about, or anything that artifact was derived from, has changed since
the evidence was produced. The headline output is the stale fraction of the
assurance case.

Three methods are implemented and all three are reported:

1. ``baselines.baseline_timestamp`` -- the modification revision of the subject
   artifact against the evidence date. Implemented at full strength as the
   method being refuted: it misses a change that reaches the artifact through a
   dependency, and it raises a false alarm on a touched but unchanged file.
2. ``baselines.baseline_closure_hash`` -- SHA-256 comparison over the
   dependency closure of the subject, which under stated conditions agrees with
   the ground truth on every item.
3. ``baselines.ground_truth`` -- the replayed change history, in which each
   item's staleness is known by construction.

``learned.StalenessRiskModel`` is a staleness-risk model over graph and churn
features with a confidence output. It is scored against the methods above on
held-out artifacts and held-out time separately. The measured finding is in
``MODEL_CARD.md`` and ``validation/VALIDATION.md``.

This software is research-grade. It is not flight-qualified, not certified and
not approved for operational aerospace use.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "AssuranceCase",
    "ChangeEvent",
    "ChangeHistory",
    "Corpus",
    "DependencyGraph",
    "EvidenceItem",
    "StalenessRiskModel",
    "__version__",
    "baseline_closure_hash",
    "baseline_timestamp",
    "baseline_timestamp_depth1",
    "bootstrap_stale_fraction",
    "build_report",
    "ground_truth",
    "load_corpus",
    "render_report",
    "stale_fraction_bracket",
]

from .baselines import (
    baseline_closure_hash,
    baseline_timestamp,
    baseline_timestamp_depth1,
    ground_truth,
)
from .bootstrap import bootstrap_stale_fraction
from .bounds import stale_fraction_bracket
from .corpus import Corpus, load_corpus
from .graph import DependencyGraph
from .learned import StalenessRiskModel
from .model import AssuranceCase, ChangeEvent, EvidenceItem
from .replay import ChangeHistory
from .report import build_report, render_report
