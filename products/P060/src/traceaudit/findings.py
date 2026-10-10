"""The checks, their codes, and the audit that runs them.

Finding codes
-------------

=======  =====================================================================
``TA001``  a declared requirement that no test claims
``TA002``  a test claiming a requirement id that is not declared anywhere
``TA003``  a requirement id declared more than once
``TA004``  a requirement traced only by skipped or xfailed tests -- the one
           that looks covered and is not
``TA005``  a requirement traced only by tests that failed or errored
``TA006``  a claiming test with no detectable assertion (**heuristic**, see
           :mod:`traceaudit.assertions`)
=======  =====================================================================

Every code can be moved out of the exit-status calculation with
``--ignore CODE``; it is still reported, marked ``ignored``.  ``TA006`` is the
one most likely to be ignored deliberately, because it is the only check that
can be wrong about correct input.

What the audit does not check
-----------------------------

It does not read the requirement text, does not judge whether a test is an
adequate verification of anything, and has no notion of a verification method,
a review status, a baseline or an approval.  A clean run means the bookkeeping
closes.  It does not mean the requirements are met.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .assertions import AssertionVerdict, index_test_files
from .config import TraceConfig
from .matrix import CoverageFigure, TraceMatrix, build_matrix
from .requirements import RequirementDocument
from .testreports import Outcome, TestReport

CODE_DESCRIPTIONS = {
    "TA001": "requirement has no test",
    "TA002": "test claims an undeclared requirement",
    "TA003": "requirement id declared more than once",
    "TA004": "requirement traced only by skipped or xfailed tests",
    "TA005": "requirement traced only by failing or erroring tests",
    "TA006": "claiming test has no detectable assertion (heuristic)",
}
HEURISTIC_CODES = frozenset({"TA006"})


@dataclass(frozen=True)
class Finding:
    """One reported problem.

    Attributes
    ----------
    code:
        One of :data:`CODE_DESCRIPTIONS`.
    subject:
        The requirement id or test node id the finding is about.
    message:
        One line, specific enough to act on.
    location:
        ``path:line`` when known, else empty.
    heuristic:
        True for findings a correct input can trigger.
    ignored:
        True when the code was passed to ``--ignore``; still printed.
    """

    code: str
    subject: str
    message: str
    location: str = ""
    heuristic: bool = False
    ignored: bool = False

    def render(self) -> str:
        """``TA001 REQ-004  message  (docs/REQUIREMENTS.md:21)``."""
        tags = []
        if self.heuristic:
            tags.append("heuristic")
        if self.ignored:
            tags.append("ignored")
        suffix = f"  [{', '.join(tags)}]" if tags else ""
        where = f"  ({self.location})" if self.location else ""
        return f"{self.code} {self.subject:<28s} {self.message}{where}{suffix}"


@dataclass(frozen=True)
class AuditResult:
    """Everything one audit produced."""

    findings: tuple[Finding, ...]
    matrix: TraceMatrix
    document: RequirementDocument
    report: TestReport
    coverage: tuple[CoverageFigure, ...]
    notes: tuple[str, ...]
    assertion_verdicts: tuple[AssertionVerdict, ...] = ()

    @property
    def blocking(self) -> tuple[Finding, ...]:
        """Findings that set the exit status."""
        return tuple(f for f in self.findings if not f.ignored)

    @property
    def exit_code(self) -> int:
        """0 when nothing blocking was found, 1 otherwise."""
        return 1 if self.blocking else 0

    def counts_by_code(self) -> dict[str, int]:
        """Number of findings per code, zero-filled over every known code."""
        counts = dict.fromkeys(CODE_DESCRIPTIONS, 0)
        for finding in self.findings:
            counts[finding.code] = counts.get(finding.code, 0) + 1
        return counts


def _assertion_lookup(
    test_root: Path | None,
) -> tuple[dict[tuple[str, str], AssertionVerdict], tuple[str, ...]]:
    if test_root is None:
        return {}, ("assertion heuristic not run: no test source root given",)
    try:
        return index_test_files(test_root)
    except (NotADirectoryError, OSError) as exc:
        return {}, (f"assertion heuristic not run: {exc}",)


def _verdict_for(case, index: dict[tuple[str, str], AssertionVerdict]) -> AssertionVerdict | None:
    stem = Path(case.file).stem if case.file else ""
    key = (stem, case.function_name)
    if key in index:
        return index[key]
    classname_stem = case.classname.split(".")[-1]
    for candidate in (classname_stem, case.classname.split(".")[0], ""):
        if (candidate, case.function_name) in index:
            return index[(candidate, case.function_name)]
    return None


def _display_root(test_root: Path | None, relative_to: str | Path | None) -> str:
    """How the test-source root is printed in a TA006 location.

    Never an absolute path: relative to ``relative_to`` when that works, else
    the directory's own name, so committed output does not record the layout
    of the machine the audit ran on.
    """
    if test_root is None:
        return ""
    if relative_to is not None:
        try:
            return Path(test_root).resolve().relative_to(Path(relative_to).resolve()).as_posix()
        except ValueError:
            pass
    return Path(test_root).name


def audit(
    document: RequirementDocument,
    report: TestReport,
    *,
    config: TraceConfig | None = None,
    test_root: str | Path | None = None,
    relative_to: str | Path | None = None,
) -> AuditResult:
    """Run every check and return the findings.

    Parameters
    ----------
    document:
        Parsed requirement declarations.
    report:
        Parsed junit XML or collection report.
    config:
        Options; ``ignored_codes`` and ``heuristic_assertions`` are used here.
    test_root:
        Directory of test sources, for the assertion heuristic.  Without it
        TA006 is not computed and a note says so.
    relative_to:
        Directory that a TA006 location is printed relative to.  When it does
        not contain ``test_root``, the root's own directory name is used.

    Returns
    -------
    AuditResult
        Findings in code order, then subject order.
    """
    cfg = config or TraceConfig.default()
    matrix = build_matrix(document, report)
    findings: list[Finding] = []
    notes: list[str] = []
    ignored = set(cfg.ignored_codes)

    def add(code: str, subject: str, message: str, location: str = "") -> None:
        findings.append(
            Finding(
                code=code,
                subject=subject,
                message=message,
                location=location,
                heuristic=code in HEURISTIC_CODES,
                ignored=code in ignored,
            )
        )

    for req_id in matrix.untraced():
        decl = document.first(req_id)
        add("TA001", req_id, "no test claims this requirement",
            decl.location if decl else "")

    for claim, tests in sorted(matrix.unknown_claims.items()):
        for node in tests:
            add("TA002", claim, f"claimed by {node}, but no declaration exists")

    for req_id, decls in sorted(document.duplicates.items()):
        places = ", ".join(d.location for d in decls)
        add("TA003", req_id, f"declared {len(decls)} times: {places}",
            decls[0].location)

    if matrix.outcomes_known:
        for req_id in matrix.illusory_traced():
            tests = matrix.requirement_to_tests[req_id]
            detail = ", ".join(
                f"{n} [{matrix.outcome_by_test[n].value}]" for n in tests
            )
            decl = document.first(req_id)
            add("TA004", req_id,
                f"every claiming test was skipped or xfailed: {detail}",
                decl.location if decl else "")
        for req_id in matrix.failing_only_traced():
            tests = matrix.requirement_to_tests[req_id]
            detail = ", ".join(
                f"{n} [{matrix.outcome_by_test[n].value}]" for n in tests
            )
            decl = document.first(req_id)
            add("TA005", req_id,
                f"every claiming test failed or errored: {detail}",
                decl.location if decl else "")
    else:
        notes.append(
            f"{report.source} is a {report.kind} report: no outcomes, so TA004 and "
            "TA005 were not computed and executed/passing coverage are undefined"
        )

    verdicts: list[AssertionVerdict] = []
    if cfg.heuristic_assertions:
        index, index_notes = _assertion_lookup(Path(test_root) if test_root else None)
        notes.extend(index_notes)
        shown_root = _display_root(Path(test_root) if test_root else None, relative_to)
        unresolved = 0
        for case in report.cases:
            if not case.claims:
                continue
            verdict = _verdict_for(case, index)
            if verdict is None:
                unresolved += 1
                continue
            verdicts.append(verdict)
            if not verdict.has_assertion:
                where = f"{shown_root}/{verdict.file}" if shown_root else verdict.file
                add("TA006", case.node_id,
                    f"no assertion detected in {verdict.function}: {verdict.detail}",
                    f"{where}:{verdict.line}")
        if index and unresolved:
            notes.append(
                f"assertion heuristic: {unresolved} claiming test(s) could not be "
                "matched to a source function and were not inspected"
            )
    else:
        notes.append("assertion heuristic disabled by configuration")

    if report.declared_totals:
        declared = report.declared_totals
        counted = report.outcome_counts
        if declared.get("tests") not in (None, 0) and declared["tests"] != len(report.cases):
            notes.append(
                f"{report.source}: <testsuite tests={declared['tests']}> but "
                f"{len(report.cases)} <testcase> elements were parsed"
            )
        if declared.get("skipped", 0) != counted["skipped"] + counted["xfailed"]:
            notes.append(
                f"{report.source}: <testsuite skipped={declared.get('skipped', 0)}> "
                f"against {counted['skipped']} skipped + {counted['xfailed']} xfailed "
                "counted from the cases"
            )

    order = {code: i for i, code in enumerate(CODE_DESCRIPTIONS)}
    findings.sort(key=lambda f: (order.get(f.code, 99), f.subject, f.message))
    return AuditResult(
        findings=tuple(findings),
        matrix=matrix,
        document=document,
        report=report,
        coverage=matrix.coverage(),
        notes=tuple(notes),
        assertion_verdicts=tuple(verdicts),
    )


def outcome_breakdown(report: TestReport) -> dict[str, int]:
    """Outcome counts for a report, counted from the cases, not the header."""
    counts = report.outcome_counts
    return {k: v for k, v in counts.items() if k != Outcome.UNKNOWN.value or v}
