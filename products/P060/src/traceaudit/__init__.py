"""traceaudit — requirements-to-test traceability computed from a repository.

Parses numbered requirements out of markdown, parses test identifiers,
outcomes and requirement claims out of a junit XML report or a pytest
collection report, computes the bidirectional mapping, and reports the
bookkeeping defects with a non-zero exit status.

**What a clean run means.** It means the bookkeeping closes: every declared
requirement is claimed by at least one test that actually ran, no test claims
an id that does not exist, and no id is declared twice.  It does **not** mean
the requirements are met, that the tests are adequate, or that anything has
been verified.  Traceability is a necessary condition on the paperwork and
nothing more; a requirement traced to a test that asserts nothing is still
traced.

This software is research-grade.  It is not flight-qualified, not certified,
and not approved for operational aerospace use.  It is not a DO-178C or
ARP4754A compliance tool and produces no certification artifact.

Validation level 1: every check is verified against hand-constructed text and
XML fixtures.  There is no physical reference to validate a text linter
against, which is why the level is 1 and not 2.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .assertions import (
    AssertionVerdict,
    index_test_files,
    inspect_file,
    inspect_function,
    inspect_source,
)
from .config import TraceConfig
from .findings import (
    CODE_DESCRIPTIONS,
    HEURISTIC_CODES,
    AuditResult,
    Finding,
    audit,
    outcome_breakdown,
)
from .markers import claims_from_marker_args, record_claims
from .matrix import CoverageFigure, TraceMatrix, build_matrix
from .report import (
    BOOKKEEPING_NOTICE,
    render_json,
    render_markdown_matrix,
    render_text,
    to_dict,
)
from .requirements import (
    Requirement,
    RequirementDocument,
    merge_documents,
    parse_requirements_file,
    parse_requirements_text,
)
from .testreports import (
    Outcome,
    TestCase,
    TestReport,
    load_report,
    parse_collection_report,
    parse_junit_xml,
)

__all__ = [
    "BOOKKEEPING_NOTICE",
    "CODE_DESCRIPTIONS",
    "HEURISTIC_CODES",
    "AssertionVerdict",
    "AuditResult",
    "CoverageFigure",
    "Finding",
    "Outcome",
    "Requirement",
    "RequirementDocument",
    "TestCase",
    "TestReport",
    "TraceConfig",
    "TraceMatrix",
    "__version__",
    "audit",
    "build_matrix",
    "claims_from_marker_args",
    "index_test_files",
    "inspect_file",
    "inspect_function",
    "inspect_source",
    "load_report",
    "merge_documents",
    "outcome_breakdown",
    "parse_collection_report",
    "parse_junit_xml",
    "parse_requirements_file",
    "parse_requirements_text",
    "record_claims",
    "render_json",
    "render_markdown_matrix",
    "render_text",
    "to_dict",
]
