"""The computed checks. This module is the product.

Sixteen checks over a parsed analysis. Every one is a count or a set
difference over declarations the author made, plus two that resolve cited
identifiers against files in the repository. None of them reads the prose, and
none of them has an opinion about whether a declared hazard is the right
hazard.

What a clean run means, precisely: the analysis is **fully drawn** and the
identifiers it cites resolve. It is not a statement that the system is safe, it
is not a statement that the hazard analysis is correct or adequate, and it does
not discharge any safety obligation of any kind.

Units: counts and dimensionless ratios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .crosslinks import RequirementIndex, TestIndex
from .enumeration import EnumerationGrid, build_grid
from .model import Analysis, ComponentKind

#: Shortest justification accepted for an enumeration exclusion, in characters.
#: Chosen so that "n/a" and "not applicable" are rejected and a one-sentence
#: reason is accepted. It is a declared policy, not a measurement.
MIN_JUSTIFICATION_CHARS: int = 20


class Severity(StrEnum):
    """Finding severity. Only ``ERROR`` affects the exit status by default."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Finding:
    """One check result naming the objects it is about."""

    code: str
    severity: Severity
    message: str
    subjects: tuple[str, ...] = ()
    line: int = 0

    def __str__(self) -> str:
        where = f" ({','.join(self.subjects)})" if self.subjects else ""
        return f"{self.code} {self.severity.value}: {self.message}{where}"


@dataclass(frozen=True)
class CheckSpec:
    """Static description of one check."""

    code: str
    severity: Severity
    title: str


#: Every check, in report order. ``hazardtrace codes`` prints this table.
CHECKS: tuple[CheckSpec, ...] = (
    CheckSpec("HT001", Severity.ERROR, "UCA enumeration gap: cell has no UCA and no exclusion"),
    CheckSpec("HT002", Severity.ERROR, "exclusion justification missing or too short"),
    CheckSpec("HT003", Severity.ERROR, "exclusion contradicted by a UCA in the same cell"),
    CheckSpec("HT004", Severity.ERROR, "hazard with no system-level constraint"),
    CheckSpec("HT005", Severity.ERROR, "constraint with no requirement identifier"),
    CheckSpec("HT006", Severity.ERROR, "UCA with no loss scenario"),
    CheckSpec("HT007", Severity.ERROR, "loss scenario with no mitigation"),
    CheckSpec("HT008", Severity.ERROR, "mitigation with no evidence artifact"),
    CheckSpec("HT009", Severity.ERROR, "hazard linked to no loss"),
    CheckSpec("HT010", Severity.ERROR, "UCA linked to no hazard"),
    CheckSpec("HT011", Severity.WARNING, "loss reached by no hazard"),
    CheckSpec("HT012", Severity.ERROR, "requirement identifier not declared in the document"),
    CheckSpec("HT013", Severity.ERROR, "test identifier not present in the test report"),
    CheckSpec("HT014", Severity.ERROR, "cited test did not pass"),
    CheckSpec("HT015", Severity.ERROR, "controller with no feedback path"),
    CheckSpec("HT016", Severity.WARNING, "control-structure component referenced by nothing"),
)

_CHECK_BY_CODE = {spec.code: spec for spec in CHECKS}


@dataclass(frozen=True)
class Ratio:
    """A coverage figure that always carries its denominator and definition."""

    numerator: int
    denominator: int
    definition: str

    @property
    def defined(self) -> bool:
        """False when the denominator is zero, in which case there is no ratio."""
        return self.denominator > 0

    @property
    def fraction(self) -> float:
        """Numerator over denominator, dimensionless. NaN when undefined."""
        if not self.defined:
            return float("nan")
        return self.numerator / self.denominator

    @property
    def percent(self) -> float:
        """The same figure in percent. NaN when undefined."""
        return 100.0 * self.fraction

    def __str__(self) -> str:
        if not self.defined:
            return f"{self.numerator}/0 = undefined"
        return f"{self.numerator}/{self.denominator} = {self.percent:.4f} %"


@dataclass(frozen=True)
class Coverage:
    """Every coverage figure the report prints, each with its denominator."""

    enumeration: Ratio
    hazard_constraint: Ratio
    constraint_requirement: Ratio
    uca_scenario: Ratio
    scenario_mitigation: Ratio
    mitigation_evidence: Ratio
    requirement_resolution: Ratio
    test_resolution: Ratio
    test_passing: Ratio

    def as_items(self) -> tuple[tuple[str, Ratio], ...]:
        """Named figures in report order."""
        return (
            ("enumeration completeness", self.enumeration),
            ("hazards with a constraint", self.hazard_constraint),
            ("constraints with a requirement", self.constraint_requirement),
            ("UCAs with a loss scenario", self.uca_scenario),
            ("scenarios with a mitigation", self.scenario_mitigation),
            ("mitigations with evidence", self.mitigation_evidence),
            ("cited requirements resolved", self.requirement_resolution),
            ("cited tests resolved", self.test_resolution),
            ("cited tests passing", self.test_passing),
        )


@dataclass(frozen=True)
class Report:
    """Result of running every check over one analysis."""

    analysis: Analysis
    grid: EnumerationGrid
    findings: tuple[Finding, ...]
    coverage: Coverage
    requirements_source: str = ""
    tests_source: str = ""
    blocking_severities: frozenset[Severity] = field(
        default_factory=lambda: frozenset({Severity.ERROR})
    )

    @property
    def blocking(self) -> tuple[Finding, ...]:
        """Findings whose severity affects the exit status."""
        return tuple(f for f in self.findings if f.severity in self.blocking_severities)

    @property
    def exit_code(self) -> int:
        """0 with no blocking finding, 1 with at least one."""
        return 1 if self.blocking else 0

    @property
    def is_complete(self) -> bool:
        """True when no blocking finding was raised.

        Complete means fully drawn and resolvable. It is not a safety claim.
        """
        return not self.blocking

    def count_by_severity(self) -> dict[Severity, int]:
        """Number of findings per severity."""
        out = dict.fromkeys(Severity, 0)
        for finding in self.findings:
            out[finding.severity] += 1
        return out

    def count_by_code(self) -> dict[str, int]:
        """Number of findings per check code, in ``CHECKS`` order."""
        out = {spec.code: 0 for spec in CHECKS}
        for finding in self.findings:
            out[finding.code] += 1
        return out


def check_spec(code: str) -> CheckSpec:
    """The static description of one check code.

    Raises
    ------
    KeyError
        If ``code`` is not a check this package implements.
    """
    return _CHECK_BY_CODE[code]


def _emit(code: str, message: str, subjects: tuple[str, ...], line: int) -> Finding:
    spec = _CHECK_BY_CODE[code]
    return Finding(
        code=code, severity=spec.severity, message=message, subjects=subjects, line=line
    )


def run_checks(
    analysis: Analysis,
    requirements: RequirementIndex | None = None,
    tests: TestIndex | None = None,
) -> Report:
    """Run every check over a parsed analysis.

    Parameters
    ----------
    analysis : Analysis
        Parsed analysis.
    requirements : RequirementIndex, optional
        Requirement declarations read from the repository. When omitted, HT012
        is not run and ``requirement_resolution`` is undefined; the report says
        so rather than reporting 100 %.
    tests : TestIndex, optional
        Test cases read from a junit report. When omitted, HT013 and HT014 are
        not run and the two test ratios are undefined.

    Returns
    -------
    Report
        Findings in check order, then by subject, so two runs over the same
        inputs produce byte-identical output.
    """
    grid = build_grid(analysis)
    findings: list[Finding] = []

    # HT001 enumeration gaps.
    for cell in grid.gaps:
        action = analysis.control_action(cell.control_action)
        name = action.name if action is not None else cell.control_action
        findings.append(
            _emit(
                "HT001",
                f"control action {cell.control_action} ({name}) has no UCA and no "
                f"justified exclusion for class '{cell.uca_class.label}'",
                (cell.control_action, cell.uca_class.value),
                action.line if action is not None else 0,
            )
        )

    # HT002 short or blank justification.
    for exclusion in analysis.exclusions:
        text = exclusion.justification.strip()
        if len(text) < MIN_JUSTIFICATION_CHARS:
            findings.append(
                _emit(
                    "HT002",
                    f"exclusion of ({exclusion.control_action}, "
                    f"{exclusion.uca_class.value}) is justified by {len(text)} "
                    f"characters, below the {MIN_JUSTIFICATION_CHARS} required",
                    (exclusion.control_action, exclusion.uca_class.value),
                    exclusion.line,
                )
            )

    # HT003 exclusion contradicted by a UCA.
    for cell in grid.conflicts:
        findings.append(
            _emit(
                "HT003",
                f"cell ({cell.control_action}, {cell.uca_class.value}) is excluded "
                f"as not applicable and also holds UCA(s) "
                f"{', '.join(cell.uca_ids)}",
                (cell.control_action, cell.uca_class.value),
                0,
            )
        )

    # HT004 hazard with no constraint, HT009 hazard with no loss.
    constrained: set[str] = set()
    for constraint in analysis.constraints:
        constrained.update(constraint.hazards)
    for hazard in analysis.hazards:
        if hazard.id not in constrained:
            findings.append(
                _emit(
                    "HT004",
                    f"hazard {hazard.id} has no system-level constraint",
                    (hazard.id,),
                    hazard.line,
                )
            )
        if not hazard.losses:
            findings.append(
                _emit(
                    "HT009",
                    f"hazard {hazard.id} is linked to no loss",
                    (hazard.id,),
                    hazard.line,
                )
            )

    # HT005 constraint with no requirement identifier.
    for constraint in analysis.constraints:
        if not constraint.requirements:
            findings.append(
                _emit(
                    "HT005",
                    f"constraint {constraint.id} cites no requirement identifier",
                    (constraint.id,),
                    constraint.line,
                )
            )

    # HT006 UCA with no loss scenario, HT010 UCA with no hazard.
    scenario_ucas: set[str] = set()
    for scenario in analysis.loss_scenarios:
        scenario_ucas.update(scenario.ucas)
    for uca in analysis.ucas:
        if uca.id not in scenario_ucas:
            findings.append(
                _emit(
                    "HT006",
                    f"UCA {uca.id} has no loss scenario",
                    (uca.id,),
                    uca.line,
                )
            )
        if not uca.hazards:
            findings.append(
                _emit(
                    "HT010",
                    f"UCA {uca.id} is linked to no hazard",
                    (uca.id,),
                    uca.line,
                )
            )

    # HT007 scenario with no mitigation.
    for scenario in analysis.loss_scenarios:
        if not scenario.mitigations:
            findings.append(
                _emit(
                    "HT007",
                    f"loss scenario {scenario.id} has no mitigation",
                    (scenario.id,),
                    scenario.line,
                )
            )

    # HT008 mitigation with no evidence.
    for mitigation in analysis.mitigations:
        if not mitigation.evidence:
            findings.append(
                _emit(
                    "HT008",
                    f"mitigation {mitigation.id} cites no evidence artifact",
                    (mitigation.id,),
                    mitigation.line,
                )
            )

    # HT011 loss reached by no hazard.
    reached: set[str] = set()
    for hazard in analysis.hazards:
        reached.update(hazard.losses)
    for loss in analysis.losses:
        if loss.id not in reached:
            findings.append(
                _emit(
                    "HT011",
                    f"loss {loss.id} is reached by no declared hazard",
                    (loss.id,),
                    loss.line,
                )
            )

    # HT012 unresolved requirement identifiers.
    cited_requirements = analysis.cited_requirements()
    resolved_requirements = 0
    if requirements is not None:
        for identifier in cited_requirements:
            if identifier in requirements:
                resolved_requirements += 1
            else:
                findings.append(
                    _emit(
                        "HT012",
                        f"requirement {identifier} is cited by this analysis but is "
                        f"not declared in {requirements.source}",
                        (identifier,),
                        0,
                    )
                )

    # HT013 / HT014 test identifiers.
    cited_tests = analysis.cited_tests()
    resolved_tests = 0
    passing_tests = 0
    if tests is not None:
        for identifier in cited_tests:
            matches = tests.lookup(identifier)
            if not matches:
                findings.append(
                    _emit(
                        "HT013",
                        f"test {identifier} is cited as evidence but is not present "
                        f"in {tests.source}",
                        (identifier,),
                        0,
                    )
                )
                continue
            resolved_tests += 1
            if all(case.outcome.is_pass for case in matches):
                passing_tests += 1
            else:
                outcomes = ", ".join(sorted({c.outcome.value for c in matches}))
                findings.append(
                    _emit(
                        "HT014",
                        f"test {identifier} is cited as evidence and did not pass "
                        f"({outcomes})",
                        (identifier,),
                        0,
                    )
                )

    # HT015 controller with no feedback path.
    controllers = analysis.structure.by_kind(ComponentKind.CONTROLLER)
    fed: set[str] = {path.controller for path in analysis.structure.feedback}
    for controller in controllers:
        if controller.id not in fed:
            findings.append(
                _emit(
                    "HT015",
                    f"controller {controller.id} ({controller.name}) has no feedback "
                    f"path: it is declared open loop",
                    (controller.id,),
                    controller.line,
                )
            )

    # HT016 unreferenced control-structure component.
    referenced: set[str] = set()
    for action in analysis.structure.control_actions:
        referenced.update({action.controller, action.actuator, action.process})
    for path in analysis.structure.feedback:
        referenced.update({path.sensor, path.controller, path.process})
    for component in analysis.structure.components:
        if component.id not in referenced:
            findings.append(
                _emit(
                    "HT016",
                    f"{component.kind.value} {component.id} ({component.name}) is "
                    f"referenced by no control action and no feedback path",
                    (component.id,),
                    component.line,
                )
            )

    order = {spec.code: index for index, spec in enumerate(CHECKS)}
    findings.sort(key=lambda f: (order[f.code], f.subjects, f.message))

    covered, total = grid.completeness()
    coverage = Coverage(
        enumeration=Ratio(
            covered, total, "cells with a UCA or a justified exclusion, of 4 per action"
        ),
        hazard_constraint=Ratio(
            sum(1 for h in analysis.hazards if h.id in constrained),
            len(analysis.hazards),
            "hazards named by >=1 system-level constraint",
        ),
        constraint_requirement=Ratio(
            sum(1 for c in analysis.constraints if c.requirements),
            len(analysis.constraints),
            "constraints citing >=1 requirement identifier",
        ),
        uca_scenario=Ratio(
            sum(1 for u in analysis.ucas if u.id in scenario_ucas),
            len(analysis.ucas),
            "UCAs named by >=1 loss scenario",
        ),
        scenario_mitigation=Ratio(
            sum(1 for s in analysis.loss_scenarios if s.mitigations),
            len(analysis.loss_scenarios),
            "loss scenarios citing >=1 mitigation",
        ),
        mitigation_evidence=Ratio(
            sum(1 for m in analysis.mitigations if m.evidence),
            len(analysis.mitigations),
            "mitigations citing >=1 evidence artifact",
        ),
        requirement_resolution=Ratio(
            resolved_requirements if requirements is not None else 0,
            len(cited_requirements) if requirements is not None else 0,
            "cited requirement identifiers found in the requirements document",
        ),
        test_resolution=Ratio(
            resolved_tests if tests is not None else 0,
            len(cited_tests) if tests is not None else 0,
            "cited test identifiers found in the junit report",
        ),
        test_passing=Ratio(
            passing_tests if tests is not None else 0,
            len(cited_tests) if tests is not None else 0,
            "cited test identifiers whose every matching case passed",
        ),
    )

    return Report(
        analysis=analysis,
        grid=grid,
        findings=tuple(findings),
        coverage=coverage,
        requirements_source=requirements.source if requirements is not None else "",
        tests_source=tests.source if tests is not None else "",
    )
