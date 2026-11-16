"""seedaudit: audit a repository's own reproducibility claims, with stated power.

The product's premise, in one paragraph. A reproducibility checker that runs a
script twice, sees the same answer and reports "reproducible" has claimed an
implied zero it has not earned. At an intermittency rate of 1 in 44 -- the rate
this mission actually shipped -- two runs detect the defect with probability
0.0444, so agreement is the *expected* outcome and carries almost no
information. The honest statement after two agreeing runs is that the rate could
be as high as 0.776 at 95 per cent confidence. `seedaudit` computes that bound
in closed form, attaches it to every clean verdict, and measures its own
detection power against a corpus with non-determinism injected at known rates.

Modules
-------
:mod:`seedaudit.power`
    The closed-form detection power, its two designs, its inversions and three
    independently derived brackets. The core of the product.
:mod:`seedaudit.numeric`
    Numeric-claim extraction, skeleton alignment and both-bounds-explicit
    comparison of a committed text against a re-measured one.
:mod:`seedaudit.seeds`
    Finding and rewriting the seed literals a script declares, from the syntax
    tree rather than by regular expression.
:mod:`seedaudit.static`
    The static screen for the self-polluting-check class: a script that writes
    under a path it also reads or globs.
:mod:`seedaudit.dynamic`
    Pristine-copy execution and the two-arrangement classifier that separates
    intrinsic non-determinism from self-pollution.
:mod:`seedaudit.corpus`
    Baseline 3: the injected corpus of known rate, and the disjoint-block
    detection-power measurement over it.
:mod:`seedaudit.audit`
    Auditing a whole repository, with coverage reported as a number.
:mod:`seedaudit.report`
    Declared constants and text rendering, in one place.

This software is research-grade. It is not flight-qualified, not certified and
not approved for operational aerospace use.
"""

from __future__ import annotations

from .audit import (
    Candidate,
    ClaimResult,
    RepositoryAudit,
    audit_claim,
    audit_repository,
    find_candidates,
    is_instrument_footprint,
    static_scan,
)
from .corpus import (
    INJECTED_RATES,
    CorpusManifest,
    block_detection,
    generate_corpus,
    manifestation_indicators,
)
from .dynamic import (
    INTRINSIC,
    INTRINSIC_MASKED,
    NO_EVIDENCE,
    SELF_POLLUTING,
    ArrangementResult,
    RunResult,
    classify,
    repeat_runs,
    run_once,
    two_arrangements,
)
from .numeric import Comparison, MovedNumber, compare, numeric_tokens, skeleton
from .power import (
    MISSION_RATE,
    PowerRow,
    agreement_probability,
    bonferroni_bracket_reference,
    bootstrap_interval,
    detection_probability_pairwise,
    detection_probability_reference,
    escaped_rate_upper_bound,
    exponential_bracket_reference,
    monte_carlo_detection,
    power_table,
    repeats_for_detection,
    repeats_for_escaped_rate_bound,
    seed_axis_detection,
    self_pollution_misclassification,
    smallest_detectable_rate,
    union_bound_reference,
    worst_case_misclassification,
    worst_case_misclassification_rate,
)
from .report import (
    AUDIT_MODEL,
    DURATION_MARKERS,
    SEED,
    format_audit,
    format_power_table,
    looks_like_a_duration,
)
from .seeds import PERTURBATION, SeedLiteral, SeedScan, find_seed_literals, rewrite_seeds
from .static import (
    PollutionFinding,
    find_self_pollution,
    is_informative,
    is_under,
    path_use_coverage,
    path_uses,
)

__version__ = "0.1.0"

__all__ = [
    "AUDIT_MODEL",
    "INJECTED_RATES",
    "INTRINSIC",
    "INTRINSIC_MASKED",
    "MISSION_RATE",
    "NO_EVIDENCE",
    "PERTURBATION",
    "SEED",
    "SELF_POLLUTING",
    "ArrangementResult",
    "Candidate",
    "ClaimResult",
    "Comparison",
    "DURATION_MARKERS",
    "CorpusManifest",
    "MovedNumber",
    "PollutionFinding",
    "PowerRow",
    "RepositoryAudit",
    "RunResult",
    "SeedLiteral",
    "SeedScan",
    "__version__",
    "agreement_probability",
    "audit_claim",
    "audit_repository",
    "block_detection",
    "bonferroni_bracket_reference",
    "bootstrap_interval",
    "classify",
    "compare",
    "detection_probability_pairwise",
    "detection_probability_reference",
    "escaped_rate_upper_bound",
    "exponential_bracket_reference",
    "find_candidates",
    "find_seed_literals",
    "find_self_pollution",
    "format_audit",
    "format_power_table",
    "generate_corpus",
    "is_informative",
    "is_instrument_footprint",
    "is_under",
    "looks_like_a_duration",
    "manifestation_indicators",
    "monte_carlo_detection",
    "numeric_tokens",
    "path_use_coverage",
    "path_uses",
    "power_table",
    "repeat_runs",
    "repeats_for_detection",
    "repeats_for_escaped_rate_bound",
    "rewrite_seeds",
    "run_once",
    "seed_axis_detection",
    "self_pollution_misclassification",
    "skeleton",
    "smallest_detectable_rate",
    "static_scan",
    "two_arrangements",
    "union_bound_reference",
    "worst_case_misclassification",
    "worst_case_misclassification_rate",
]
