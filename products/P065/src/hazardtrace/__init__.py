"""hazardtrace: STPA control-structure hazard analysis as a machine-checkable artifact.

Reads a declared System-Theoretic Process Analysis (STPA) from YAML -- losses,
hazards, system-level constraints, the control structure, unsafe control
actions, loss scenarios, mitigations and evidence -- and computes what is
missing from it: enumeration gaps over the four UCA classes, hazards with no
constraint, constraints with no requirement, UCAs with no scenario, scenarios
with no mitigation, mitigations with no evidence, and cited requirement and
test identifiers that do not resolve against the repository.

Method references
-----------------
Leveson, N. G. and Thomas, J. P., *STPA Handbook*, March 2018. The four UCA
    classes, the control-structure model and the four analysis steps.
Leveson, N. G., *Engineering a Safer World: Systems Thinking Applied to
    Safety*, MIT Press, 2011. ISBN 978-0-262-01662-9. STAMP and STPA in full.
Leveson, N. G., "A new accident model for engineering safer systems", *Safety
    Science*, vol. 42, no. 4, 2004, pp. 237-270. doi:10.1016/S0925-7535(03)00047-X
Thomas, J., "Extending and automating a systems-theoretic hazard analysis for
    requirements generation and analysis", PhD thesis, Massachusetts Institute
    of Technology, 2013. Formal UCA context tables.

This software is research-grade. It is not flight-qualified, not certified, and
not approved for operational aerospace use. It checks a declared argument for
completeness; it does not evaluate, validate or endorse that argument, and a
clean run discharges no safety obligation.
"""

from __future__ import annotations

from .checks import (
    CHECKS,
    MIN_JUSTIFICATION_CHARS,
    CheckSpec,
    Coverage,
    Finding,
    Ratio,
    Report,
    Severity,
    check_spec,
    run_checks,
)
from .crosslinks import (
    DEFAULT_REQUIREMENT_PATTERN,
    RequirementIndex,
    TestCase,
    TestIndex,
    TestOutcome,
    load_junit,
    load_requirements,
    parse_junit,
    parse_requirements,
)
from .diagram import render_control_structure_png, render_mermaid
from .enumeration import Cell, CellState, EnumerationGrid, build_grid
from .errors import AnalysisFormatError, HazardTraceError
from .model import (
    UCA_CLASSES,
    Analysis,
    Component,
    ComponentKind,
    ControlAction,
    ControlStructure,
    Evidence,
    EvidenceKind,
    Exclusion,
    FeedbackPath,
    Hazard,
    Loss,
    LossScenario,
    Mitigation,
    SystemConstraint,
    Uca,
    UcaClass,
)
from .parse import load_analysis, parse_analysis
from .report import (
    SCOPE_NOTE,
    format_codes,
    format_coverage_table,
    format_findings,
    format_grid,
    format_json,
    format_markdown,
    format_report,
    report_to_dict,
)

__version__ = "0.1.0"

__all__ = [
    "CHECKS",
    "DEFAULT_REQUIREMENT_PATTERN",
    "MIN_JUSTIFICATION_CHARS",
    "SCOPE_NOTE",
    "UCA_CLASSES",
    "Analysis",
    "AnalysisFormatError",
    "Cell",
    "CellState",
    "CheckSpec",
    "Component",
    "ComponentKind",
    "ControlAction",
    "ControlStructure",
    "Coverage",
    "EnumerationGrid",
    "Evidence",
    "EvidenceKind",
    "Exclusion",
    "FeedbackPath",
    "Finding",
    "Hazard",
    "HazardTraceError",
    "Loss",
    "LossScenario",
    "Mitigation",
    "Ratio",
    "Report",
    "RequirementIndex",
    "Severity",
    "SystemConstraint",
    "TestCase",
    "TestIndex",
    "TestOutcome",
    "Uca",
    "UcaClass",
    "__version__",
    "build_grid",
    "check_spec",
    "format_codes",
    "format_coverage_table",
    "format_findings",
    "format_grid",
    "format_json",
    "format_markdown",
    "format_report",
    "load_analysis",
    "load_junit",
    "load_requirements",
    "parse_analysis",
    "parse_junit",
    "parse_requirements",
    "render_control_structure_png",
    "render_mermaid",
    "report_to_dict",
    "run_checks",
]
