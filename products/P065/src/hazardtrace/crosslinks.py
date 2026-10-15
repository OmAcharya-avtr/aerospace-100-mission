"""Resolve requirement and test identifiers against the repository.

The point of this module is that the coverage figures are **computed from the
repository**, not asserted in the hazard analysis. A mitigation may cite
``REQ-014`` and ``tests/test_bdot.py::test_first_step_commands_zero``; this
module reads the requirements document and the junit XML and says whether those
two things exist and, for the test, whether it passed.

Identifier conventions are the ones P060 TraceAudit uses, so that a repository
already carrying numbered requirements and a junit report needs no new
bookkeeping. Nothing is imported from P060: every product in this portfolio is
self-contained, and the two parsers here are independent implementations of the
same two file formats.

Units: counts only.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .errors import AnalysisFormatError

#: Default requirement-identifier pattern, matching ``REQ-001``.
DEFAULT_REQUIREMENT_PATTERN = r"REQ-\d{3,}"

#: A declaration is an identifier at the start of a line, optionally behind a
#: markdown heading marker, a list bullet or bold markers. A mention of an
#: identifier inside a sentence is deliberately not a declaration.
_DECLARATION_PREFIX = r"^\s{0,3}(?:#{1,6}\s+|[-*+]\s+|\*\*)?"

_FENCE = re.compile(r"^\s{0,3}(```|~~~)")


class TestOutcome(StrEnum):
    """Outcome of one test case as recorded in junit XML."""

    # Not a pytest test class, despite the name: tell the collector so.
    __test__ = False

    PASSED = "passed"
    FAILED = "failed"
    ERRORED = "errored"
    SKIPPED = "skipped"
    XFAILED = "xfailed"

    @property
    def is_pass(self) -> bool:
        """True only for a test that ran and passed."""
        return self is TestOutcome.PASSED


@dataclass(frozen=True)
class TestCase:
    """One ``<testcase>`` element, with every identifier form it answers to."""

    __test__ = False

    name: str
    classname: str
    file: str
    outcome: TestOutcome
    aliases: frozenset[str]


@dataclass(frozen=True)
class RequirementIndex:
    """Requirement identifiers declared in a markdown document."""

    source: str
    declared: tuple[str, ...]
    lines: dict[str, int]

    def __contains__(self, identifier: object) -> bool:
        return identifier in self.lines

    @property
    def unique(self) -> frozenset[str]:
        """Set of declared identifiers."""
        return frozenset(self.lines)


@dataclass(frozen=True)
class TestIndex:
    """Test cases read from a junit XML report."""

    __test__ = False

    source: str
    cases: tuple[TestCase, ...]

    def lookup(self, identifier: str) -> tuple[TestCase, ...]:
        """Every test case answering to ``identifier``, in report order."""
        return tuple(c for c in self.cases if identifier in c.aliases)

    @property
    def counts(self) -> dict[TestOutcome, int]:
        """Number of cases per outcome."""
        out = dict.fromkeys(TestOutcome, 0)
        for case in self.cases:
            out[case.outcome] += 1
        return out


def parse_requirements(
    text: str, source: str = "<string>", pattern: str = DEFAULT_REQUIREMENT_PATTERN
) -> RequirementIndex:
    """Index requirement declarations in markdown text.

    A declaration is an identifier matching ``pattern`` at the start of a line,
    optionally behind ``#``, a list bullet or ``**``. Lines inside fenced code
    blocks are skipped. The first declaration of an identifier wins; a repeat
    is recorded in ``declared`` so a caller can count duplicates.
    """
    try:
        matcher = re.compile(_DECLARATION_PREFIX + "(" + pattern + ")")
    except re.error as exc:
        raise AnalysisFormatError(f"invalid requirement pattern: {exc}") from exc
    declared: list[str] = []
    lines: dict[str, int] = {}
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        found = matcher.match(line)
        if found is None:
            continue
        identifier = found.group(1)
        declared.append(identifier)
        lines.setdefault(identifier, number)
    return RequirementIndex(source=source, declared=tuple(declared), lines=lines)


def load_requirements(
    path: str | os.PathLike[str], pattern: str = DEFAULT_REQUIREMENT_PATTERN
) -> RequirementIndex:
    """Index requirement declarations in a markdown file."""
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AnalysisFormatError(f"cannot read requirements document: {exc}") from exc
    return parse_requirements(text, source=str(file_path), pattern=pattern)


def _classify(case: ET.Element) -> TestOutcome:
    """Classify one testcase element, following pytest's junit writer.

    pytest writes an expected failure as ``<skipped type="pytest.xfail">``, so
    an xfail is distinguishable from a plain skip but an **xpass is not
    distinguishable from a pass**. That limitation is pytest's and it is
    documented in the README rather than guessed at here.
    """
    if case.find("error") is not None:
        return TestOutcome.ERRORED
    if case.find("failure") is not None:
        return TestOutcome.FAILED
    skipped = case.find("skipped")
    if skipped is not None:
        if (skipped.get("type") or "") == "pytest.xfail":
            return TestOutcome.XFAILED
        return TestOutcome.SKIPPED
    return TestOutcome.PASSED


def _aliases(name: str, classname: str, file_attr: str) -> frozenset[str]:
    """Every identifier form one test case answers to.

    pytest node ids look like ``tests/test_x.py::test_y`` or
    ``tests/test_x.py::TestCls::test_y``; junit XML carries the dotted
    ``classname`` and, with ``junit_family = xunit1``, the ``file`` attribute
    as well. All forms derivable from what is present are accepted, so a
    hazard analysis can cite whichever the author has to hand.
    """
    out: set[str] = {name}
    if classname:
        out.add(f"{classname}::{name}")
        out.add(f"{classname}.{name}")
        parts = classname.split(".")
        if parts:
            # tests.test_x -> tests/test_x.py ; tests.test_x.TestCls -> nested
            for split in range(len(parts), 0, -1):
                module = "/".join(parts[:split]) + ".py"
                tail = "::".join(parts[split:] + [name])
                out.add(f"{module}::{tail}")
    if file_attr:
        out.add(f"{file_attr}::{name}")
        if classname:
            parts = classname.split(".")
            for split in range(len(parts), 0, -1):
                tail = "::".join(parts[split:] + [name])
                out.add(f"{file_attr}::{tail}")
    return frozenset(out)


def parse_junit(text: str, source: str = "<string>") -> TestIndex:
    """Index test cases in a junit XML report.

    Accepts either a ``<testsuites>`` root or a single ``<testsuite>`` root,
    which are both shapes pytest emits.
    """
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise AnalysisFormatError(f"junit XML is not well formed: {exc}") from exc
    if root.tag == "testsuite":
        suites = [root]
    elif root.tag == "testsuites":
        suites = list(root.iter("testsuite"))
    else:
        raise AnalysisFormatError(
            f"junit root element is <{root.tag}>, expected <testsuites> or <testsuite>"
        )
    cases: list[TestCase] = []
    for suite in suites:
        for element in suite.iter("testcase"):
            name = element.get("name") or ""
            classname = element.get("classname") or ""
            file_attr = element.get("file") or ""
            cases.append(
                TestCase(
                    name=name,
                    classname=classname,
                    file=file_attr,
                    outcome=_classify(element),
                    aliases=_aliases(name, classname, file_attr),
                )
            )
    return TestIndex(source=source, cases=tuple(cases))


def load_junit(path: str | os.PathLike[str]) -> TestIndex:
    """Index test cases in a junit XML file."""
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AnalysisFormatError(f"cannot read junit report: {exc}") from exc
    return parse_junit(text, source=str(file_path))
