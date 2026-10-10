"""Parsing test identifiers, outcomes and requirement claims from a report.

Two input shapes are supported and they are not equivalent:

**junit XML** (``pytest --junitxml=FILE``) carries an outcome per test case, so
the skipped-and-xfailed findings are computable from it.

**A pytest collection report** (``pytest --collect-only -q``) carries node ids
only.  Every test parsed from it has outcome :data:`Outcome.UNKNOWN`, and the
audit therefore cannot tell a skipped test from a passing one.  This is a
property of the input, not a defect of the parser, and the report says so.

Outcome extraction follows pytest's own junit writer, verified against
pytest 9.1.1 output in this container:

* ``<skipped type="pytest.skip">``  -> skipped
* ``<skipped type="pytest.xfail">`` -> xfailed
* ``<failure>``                     -> failed
* ``<error>``                       -> errored
* no child element                  -> passed

An **xpassed** test (one marked ``xfail`` that unexpectedly passed) is written
by pytest as a bare ``<testcase>`` and is therefore indistinguishable from a
plain pass in junit XML.  See the Limitations section of README.md.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from .config import TraceConfig


class Outcome(StrEnum):
    """Outcome of one test case."""

    PASSED = "passed"
    FAILED = "failed"
    ERRORED = "errored"
    SKIPPED = "skipped"
    XFAILED = "xfailed"
    UNKNOWN = "unknown"

    @property
    def executed(self) -> bool:
        """True when the test body ran (passed, failed or errored)."""
        return self in (Outcome.PASSED, Outcome.FAILED, Outcome.ERRORED)

    @property
    def illusory(self) -> bool:
        """True when the test was reported but its body never asserted anything."""
        return self in (Outcome.SKIPPED, Outcome.XFAILED)


@dataclass(frozen=True)
class TestCase:
    """One test case from a report.

    Attributes
    ----------
    node_id:
        ``file::Class::name`` when derivable, else ``classname::name``.
    classname:
        Dotted module (and class) path as the report gives it.
    name:
        Test function name, including any parametrisation suffix.
    outcome:
        See :class:`Outcome`.
    file:
        Source file path when the report supplies one (``junit_family =
        xunit1`` does, the default ``xunit2`` does not), else ``None``.
    line:
        One-based line of the test function when supplied, else ``None``.
    message:
        Skip reason or failure message when present.
    claims:
        Requirement ids this test claims, in order of first appearance.
    properties:
        Every ``<property>`` on the test case.
    """

    __test__ = False  # not a pytest test class, despite the name

    node_id: str
    classname: str
    name: str
    outcome: Outcome
    file: str | None = None
    line: int | None = None
    message: str = ""
    claims: tuple[str, ...] = ()
    properties: tuple[tuple[str, str], ...] = ()

    @property
    def function_name(self) -> str:
        """Test name with any ``[...]`` parametrisation suffix removed."""
        return self.name.split("[", 1)[0]


@dataclass(frozen=True)
class TestReport:
    """Every test case from one report, plus the report's own totals."""

    __test__ = False  # not a pytest test class, despite the name

    cases: tuple[TestCase, ...]
    source: str
    kind: str
    declared_totals: dict[str, int] = field(default_factory=dict)

    @property
    def outcome_counts(self) -> dict[str, int]:
        """Number of cases per outcome, counted from the cases themselves."""
        counts = {o.value: 0 for o in Outcome}
        for case in self.cases:
            counts[case.outcome.value] += 1
        return counts

    @property
    def claimed_ids(self) -> tuple[str, ...]:
        """Unique requirement ids claimed by any case, in first-seen order."""
        seen: dict[str, None] = {}
        for case in self.cases:
            for claim in case.claims:
                seen.setdefault(claim, None)
        return tuple(seen)


def _split_values(value: str, separators: str) -> list[str]:
    pattern = "[" + re.escape(separators) + "]+"
    return [part for part in re.split(pattern, value.strip()) if part]


def _claims_for(
    node_id: str, properties: tuple[tuple[str, str], ...], cfg: TraceConfig
) -> tuple[str, ...]:
    id_re = re.compile(rf"^{cfg.id_pattern}$")
    seen: dict[str, None] = {}
    for name, value in properties:
        if name not in cfg.claim_property_names:
            continue
        for candidate in _split_values(value, cfg.claim_value_separators):
            if id_re.match(candidate):
                seen.setdefault(candidate, None)
    if cfg.claim_from_node_id:
        node_re = re.compile(cfg.node_id_claim_pattern)
        for match in node_re.finditer(node_id):
            built = cfg.node_id_claim_template.format(*match.groups())
            if id_re.match(built):
                seen.setdefault(built, None)
    return tuple(seen)


def _outcome_of(element: ET.Element) -> tuple[Outcome, str]:
    skipped = element.find("skipped")
    if skipped is not None:
        kind = (skipped.get("type") or "").strip()
        message = skipped.get("message") or (skipped.text or "")
        if kind.endswith("xfail"):
            return Outcome.XFAILED, message.strip()
        return Outcome.SKIPPED, message.strip()
    failure = element.find("failure")
    if failure is not None:
        return Outcome.FAILED, (failure.get("message") or "").strip()
    error = element.find("error")
    if error is not None:
        return Outcome.ERRORED, (error.get("message") or "").strip()
    return Outcome.PASSED, ""


def _node_id(classname: str, name: str, file: str | None) -> str:
    if file:
        parts = classname.split(".")
        stem = Path(file).stem
        trailing = parts[parts.index(stem) + 1 :] if stem in parts else []
        return "::".join([file, *trailing, name])
    return f"{classname}::{name}" if classname else name


def parse_junit_xml(
    path: str | Path, *, config: TraceConfig | None = None, relative_to: Path | None = None
) -> TestReport:
    """Parse a junit XML report written by pytest.

    Parameters
    ----------
    path:
        The XML file.
    config:
        Parser options.
    relative_to:
        If given, the recorded source label is made relative to this
        directory when possible.

    Raises
    ------
    FileNotFoundError
        The file does not exist.
    ValueError
        The file is not parseable XML, or contains no ``<testsuite>``.
    """
    cfg = config or TraceConfig.default()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"junit XML report not found: {p}")
    try:
        root = ET.parse(p).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"{p} is not parseable XML: {exc}") from exc
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if not suites:
        raise ValueError(f"{p} contains no <testsuite> element; is it a junit report?")

    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    cases: list[TestCase] = []
    for suite in suites:
        for key in totals:
            try:
                totals[key] += int(suite.get(key, 0) or 0)
            except ValueError:
                pass
        for element in suite.findall("testcase"):
            classname = element.get("classname", "") or ""
            name = element.get("name", "") or ""
            file_attr = element.get("file")
            line_attr = element.get("line")
            props = tuple(
                (prop.get("name", ""), prop.get("value", ""))
                for container in element.findall("properties")
                for prop in container.findall("property")
            )
            outcome, message = _outcome_of(element)
            node_id = _node_id(classname, name, file_attr)
            cases.append(
                TestCase(
                    node_id=node_id,
                    classname=classname,
                    name=name,
                    outcome=outcome,
                    file=file_attr,
                    line=int(line_attr) + 1 if line_attr and line_attr.isdigit() else None,
                    message=message,
                    claims=_claims_for(node_id, props, cfg),
                    properties=props,
                )
            )
    label = str(p)
    if relative_to is not None:
        try:
            label = str(p.resolve().relative_to(Path(relative_to).resolve()))
        except ValueError:
            label = p.name
    return TestReport(
        cases=tuple(cases), source=label, kind="junit-xml", declared_totals=totals
    )


_COLLECT_LINE = re.compile(r"^(?P<node>[^\s<>]+\.py::[^\s]+)\s*$")


def parse_collection_report(
    path: str | Path, *, config: TraceConfig | None = None, relative_to: Path | None = None
) -> TestReport:
    """Parse the output of ``pytest --collect-only -q``.

    Every case gets :data:`Outcome.UNKNOWN`: a collection report records no
    outcomes, so findings TA004 and TA005 are not computable from it.
    """
    cfg = config or TraceConfig.default()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"collection report not found: {p}")
    cases: list[TestCase] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        match = _COLLECT_LINE.match(raw.strip())
        if match is None:
            continue
        node_id = match.group("node")
        file_part, _, rest = node_id.partition("::")
        name = rest.split("::")[-1]
        classname = ".".join([Path(file_part).with_suffix("").as_posix().replace("/", ".")]
                             + rest.split("::")[:-1])
        cases.append(
            TestCase(
                node_id=node_id,
                classname=classname,
                name=name,
                outcome=Outcome.UNKNOWN,
                file=file_part,
                claims=_claims_for(node_id, (), cfg),
            )
        )
    label = str(p)
    if relative_to is not None:
        try:
            label = str(p.resolve().relative_to(Path(relative_to).resolve()))
        except ValueError:
            label = p.name
    return TestReport(cases=tuple(cases), source=label, kind="collect-only")


def load_report(
    path: str | Path, *, config: TraceConfig | None = None, relative_to: Path | None = None
) -> TestReport:
    """Dispatch on the file suffix: ``.xml`` is junit, anything else collection."""
    p = Path(path)
    if p.suffix.lower() == ".xml":
        return parse_junit_xml(p, config=config, relative_to=relative_to)
    return parse_collection_report(p, config=config, relative_to=relative_to)
