"""tmtcdiff: semantic diff of two telemetry and command dictionaries, with impact.

What this package is for
------------------------

A telemetry and command dictionary is upgraded. The question that matters is
not what the text diff shows; it is which of the procedures, scripts and
configurations already written against the old version now behave differently.
Two kinds of answer exist and they are not equally urgent:

- A mnemonic was removed. Everything referencing it fails at load or at first
  use, with the name in the error. This is **loud**.
- A mnemonic kept its name and changed its scale factor, its unit, its bit
  offset, its byte order, its data type or its enumeration. Everything
  referencing it keeps working and quietly means something else. This is
  **silent**, and it is reported first by every function in this package.

The public surface
------------------

- :mod:`tmtcdiff.model` — ``Parameter``, ``CommandArgument``, ``Command``,
  ``Dictionary``.
- :mod:`tmtcdiff.parse` — read and write dictionary JSON, carrying unmodelled
  fields rather than dropping them.
- :mod:`tmtcdiff.changes` — the change-class taxonomy and the severity tiers.
- :mod:`tmtcdiff.rename` — rename detection by structural similarity, with a
  measurable false-match rate.
- :mod:`tmtcdiff.diff` — ``diff_dictionaries``, returning a ranked
  ``DiffResult`` and a ``DiffLimits`` saying what it could not determine.
- :mod:`tmtcdiff.corpus` — reference extraction, including the references it
  cannot resolve statically.
- :mod:`tmtcdiff.impact` — ``assess_impact``, returning ranked findings and a
  ``CoverageVerdict``.
- :mod:`tmtcdiff.synth` — the seeded generator every measured number here
  comes from.
- :mod:`tmtcdiff.metrics`, :mod:`tmtcdiff.report` — counting and rendering.

What it does not do
-------------------

It does not say an upgrade is safe. An empty impact report means no finding
was produced for the references this tool could resolve in the sources it was
given; :attr:`tmtcdiff.impact.ImpactReport.interpretation` renders that
statement and no stronger one, and
:class:`tmtcdiff.impact.CoverageVerdict` carries the counts behind it.

Research grade. Not flight-qualified, not certified, not approved for
operational aerospace use.
"""

from __future__ import annotations

from .changes import (
    ADVISORY_CLASSES,
    CLASS_NOTES,
    CLASS_ORDER,
    CLASS_SEVERITY,
    LOUD_CLASSES,
    SILENT_CLASSES,
    Change,
    ChangeClass,
    Severity,
    rank_changes,
)
from .corpus import (
    Corpus,
    CorpusSource,
    Reference,
    extract_from_source,
    extract_references,
    load_corpus,
)
from .diff import (
    DEFAULT_NUMERIC_ABS_TOL,
    DEFAULT_NUMERIC_REL_TOL,
    DEFAULT_RENAME_THRESHOLD,
    DiffConfig,
    DiffLimits,
    DiffResult,
    IncomparableField,
    compare_commands,
    compare_parameters,
    diff_dictionaries,
)
from .impact import (
    IMPACT_NOTES,
    CoverageVerdict,
    Finding,
    ImpactClass,
    ImpactReport,
    assess_impact,
)
from .metrics import ClassMetrics, aggregate, clopper_pearson, per_class_metrics
from .model import Command, CommandArgument, Dictionary, Parameter
from .parse import (
    DictionaryFormatError,
    dictionary_from_mapping,
    dictionary_to_mapping,
    dumps,
    load_dictionary,
    loads,
)
from .rename import (
    DEFAULT_WEIGHTS,
    FIELD_WEIGHTS,
    STRATEGIES,
    RenameCandidate,
    match_renames,
    pair_score,
    signature_collision_bound,
    structural_similarity,
    text_similarity,
)
from .report import format_table, impact_class_table, render_diff, render_impact, taxonomy_table
from .synth import (
    INJECTABLE,
    INJECTABLE_SILENT,
    InjectedChange,
    SynthPair,
    baseline_dictionary,
    change_pair,
    inject_changes,
    rename_pair,
    reordering_control,
    synthetic_corpus,
)

__version__ = "0.1.0"

__all__ = [
    "ADVISORY_CLASSES",
    "CLASS_NOTES",
    "CLASS_ORDER",
    "CLASS_SEVERITY",
    "DEFAULT_NUMERIC_ABS_TOL",
    "DEFAULT_NUMERIC_REL_TOL",
    "DEFAULT_RENAME_THRESHOLD",
    "DEFAULT_WEIGHTS",
    "FIELD_WEIGHTS",
    "IMPACT_NOTES",
    "INJECTABLE",
    "INJECTABLE_SILENT",
    "LOUD_CLASSES",
    "SILENT_CLASSES",
    "STRATEGIES",
    "Change",
    "ChangeClass",
    "ClassMetrics",
    "Command",
    "CommandArgument",
    "Corpus",
    "CorpusSource",
    "CoverageVerdict",
    "Dictionary",
    "DictionaryFormatError",
    "DiffConfig",
    "DiffLimits",
    "DiffResult",
    "Finding",
    "ImpactClass",
    "ImpactReport",
    "IncomparableField",
    "InjectedChange",
    "Parameter",
    "Reference",
    "RenameCandidate",
    "Severity",
    "SynthPair",
    "__version__",
    "aggregate",
    "assess_impact",
    "baseline_dictionary",
    "change_pair",
    "clopper_pearson",
    "compare_commands",
    "compare_parameters",
    "dictionary_from_mapping",
    "dictionary_to_mapping",
    "diff_dictionaries",
    "dumps",
    "extract_from_source",
    "extract_references",
    "format_table",
    "impact_class_table",
    "inject_changes",
    "load_corpus",
    "load_dictionary",
    "loads",
    "match_renames",
    "pair_score",
    "per_class_metrics",
    "rank_changes",
    "rename_pair",
    "render_diff",
    "render_impact",
    "reordering_control",
    "signature_collision_bound",
    "structural_similarity",
    "synthetic_corpus",
    "taxonomy_table",
    "text_similarity",
]
