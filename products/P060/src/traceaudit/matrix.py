"""The bidirectional requirement-to-test mapping and its coverage arithmetic.

Three coverage figures are computed and **all three are reported with their
denominator spelled out**, because a single "coverage %" is the number that
makes a traceability report misleading:

``nominal``
    requirements with at least one claiming test, of any outcome.
``executed``
    requirements with at least one claiming test whose body actually ran
    (passed, failed or errored).  A requirement traced only by a skipped or
    xfailed test counts in ``nominal`` and not in ``executed``.
``passing``
    requirements with at least one claiming test that passed.

The denominator of all three is the number of **unique declared requirement
ids**; a duplicated declaration is one requirement, and is also reported as a
finding.  When the test report is a collection report every outcome is
``unknown``, so ``executed`` and ``passing`` are undefined and reported as
``None`` rather than as zero.
"""

from __future__ import annotations

from dataclasses import dataclass

from .requirements import RequirementDocument
from .testreports import Outcome, TestReport


@dataclass(frozen=True)
class CoverageFigure:
    """A coverage ratio with its numerator and denominator kept visible."""

    label: str
    numerator: int
    denominator: int
    definition: str

    @property
    def percent(self) -> float | None:
        """Percentage, or ``None`` when the denominator is zero."""
        if self.denominator == 0:
            return None
        return 100.0 * self.numerator / self.denominator

    def render(self) -> str:
        """``label: 4/7 = 57.1429 % (definition)``."""
        pct = self.percent
        shown = "undefined" if pct is None else f"{pct:.4f} %"
        return f"{self.label}: {self.numerator}/{self.denominator} = {shown}"


@dataclass(frozen=True)
class TraceMatrix:
    """Both directions of the mapping, plus the coverage figures.

    Attributes
    ----------
    requirement_to_tests:
        Declared id -> node ids claiming it, in report order.
    test_to_requirements:
        Node id -> claimed ids, including ids that are not declared.
    unknown_claims:
        Claimed id -> node ids, for ids with no declaration.
    outcome_by_test:
        Node id -> outcome, for the outcome lookups the findings need.
    """

    requirement_to_tests: dict[str, tuple[str, ...]]
    test_to_requirements: dict[str, tuple[str, ...]]
    unknown_claims: dict[str, tuple[str, ...]]
    outcome_by_test: dict[str, Outcome]
    outcomes_known: bool

    @property
    def n_requirements(self) -> int:
        """Unique declared requirement ids: the denominator."""
        return len(self.requirement_to_tests)

    def traced(self) -> tuple[str, ...]:
        """Requirements with at least one claiming test."""
        return tuple(r for r, t in self.requirement_to_tests.items() if t)

    def untraced(self) -> tuple[str, ...]:
        """Requirements with no claiming test at all."""
        return tuple(r for r, t in self.requirement_to_tests.items() if not t)

    def _with_outcome(self, predicate) -> tuple[str, ...]:
        out = []
        for req, tests in self.requirement_to_tests.items():
            if any(predicate(self.outcome_by_test[t]) for t in tests):
                out.append(req)
        return tuple(out)

    def executed_traced(self) -> tuple[str, ...]:
        """Requirements with at least one claiming test whose body ran."""
        return self._with_outcome(lambda o: o.executed)

    def passing_traced(self) -> tuple[str, ...]:
        """Requirements with at least one passing claiming test."""
        return self._with_outcome(lambda o: o is Outcome.PASSED)

    def illusory_traced(self) -> tuple[str, ...]:
        """Requirements traced, but only by skipped or xfailed tests."""
        out = []
        for req, tests in self.requirement_to_tests.items():
            if not tests:
                continue
            outcomes = [self.outcome_by_test[t] for t in tests]
            if all(o.illusory for o in outcomes):
                out.append(req)
        return tuple(out)

    def failing_only_traced(self) -> tuple[str, ...]:
        """Requirements traced, but every claiming test failed or errored."""
        out = []
        for req, tests in self.requirement_to_tests.items():
            if not tests:
                continue
            outcomes = [self.outcome_by_test[t] for t in tests]
            if all(o in (Outcome.FAILED, Outcome.ERRORED) for o in outcomes):
                out.append(req)
        return tuple(out)

    def coverage(self) -> tuple[CoverageFigure, ...]:
        """The three coverage figures, denominators included."""
        n = self.n_requirements
        figures = [
            CoverageFigure(
                "nominal coverage",
                len(self.traced()),
                n,
                "requirements with >=1 claiming test of any outcome",
            )
        ]
        if self.outcomes_known:
            figures.append(
                CoverageFigure(
                    "executed coverage",
                    len(self.executed_traced()),
                    n,
                    "requirements with >=1 claiming test that actually ran",
                )
            )
            figures.append(
                CoverageFigure(
                    "passing coverage",
                    len(self.passing_traced()),
                    n,
                    "requirements with >=1 claiming test that passed",
                )
            )
        return tuple(figures)


def build_matrix(doc: RequirementDocument, report: TestReport) -> TraceMatrix:
    """Compute the bidirectional mapping between declarations and test cases.

    Parameters
    ----------
    doc:
        Parsed requirement declarations.
    report:
        Parsed test report.

    Returns
    -------
    TraceMatrix
        Both directions, with duplicate declarations collapsed to one key.
    """
    declared = list(doc.ids)
    req_to_tests: dict[str, list[str]] = {r: [] for r in declared}
    test_to_reqs: dict[str, tuple[str, ...]] = {}
    unknown: dict[str, list[str]] = {}
    outcome_by_test: dict[str, Outcome] = {}
    for case in report.cases:
        outcome_by_test[case.node_id] = case.outcome
        test_to_reqs[case.node_id] = case.claims
        for claim in case.claims:
            if claim in req_to_tests:
                if case.node_id not in req_to_tests[claim]:
                    req_to_tests[claim].append(case.node_id)
            else:
                unknown.setdefault(claim, []).append(case.node_id)
    outcomes_known = any(c.outcome is not Outcome.UNKNOWN for c in report.cases) or not report.cases
    if report.kind == "collect-only":
        outcomes_known = False
    return TraceMatrix(
        requirement_to_tests={k: tuple(v) for k, v in req_to_tests.items()},
        test_to_requirements=test_to_reqs,
        unknown_claims={k: tuple(v) for k, v in unknown.items()},
        outcome_by_test=outcome_by_test,
        outcomes_known=outcomes_known,
    )
