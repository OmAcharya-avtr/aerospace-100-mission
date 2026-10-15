"""Requirement and test identifier resolution, measured against the fixtures.

The claim this script checks is the one the README makes: coverage against
requirements and tests is **computed from the repository**, not asserted in the
hazard analysis. The two parsers are exercised against the two committed
fixtures, and the resolution numbers are reported with their denominators.

The P024 junit fixture is a real pytest run of P024 DetumbleSim's own suite,
normalised by ``fixtures/regenerate_fixtures.py``. The requirements document
for it is **hand-written for this worked example**: P024 ships none, and that
is itself one of the findings.
"""

from __future__ import annotations

from _support import CASES, FIXTURES, Emitter

from hazardtrace import (
    TestOutcome,
    load_analysis,
    load_junit,
    load_requirements,
    parse_junit,
    parse_requirements,
    run_checks,
)

DECLARATION_FORMS = """
### REQ-001 Heading form

Prose mentioning REQ-004 mid-sentence, which is not a declaration.

- REQ-002 Bullet form

**REQ-003** Bold form

```
### REQ-900 inside a fenced block
```

### REQ-001 Declared a second time
"""

ALL_OUTCOMES = (
    '<testsuites><testsuite name="pytest" tests="5">'
    '<testcase classname="tests.test_x" name="t_pass" file="tests/test_x.py"/>'
    '<testcase classname="tests.test_x" name="t_fail"><failure message="m"/></testcase>'
    '<testcase classname="tests.test_x" name="t_err"><error message="m"/></testcase>'
    '<testcase classname="tests.test_x" name="t_skip">'
    '<skipped type="pytest.skip" message="m"/></testcase>'
    '<testcase classname="tests.test_x" name="t_xfail">'
    '<skipped type="pytest.xfail" message="m"/></testcase>'
    "</testsuite></testsuites>"
)


def main() -> int:
    """Measure the two parsers and the resolution ratios. Returns an exit code."""
    emit = Emitter(__file__)
    emit("hazardtrace validation: requirement and test identifier resolution")
    emit()

    # -- A. requirement declaration forms ---------------------------------
    emit("A. requirement declaration forms, hand-counted")
    index = parse_requirements(DECLARATION_FORMS, source="inline")
    emit(f"   declarations found      : {len(index.declared)}")
    emit(f"   unique identifiers      : {len(index.unique)}  {sorted(index.unique)}")
    emit("   expected: 4 declarations over 3 unique ids. The heading, bullet and")
    emit("   bold forms declare; the mid-sentence mention and the fenced line do")
    emit("   not; REQ-001 is declared twice.")
    emit.check("declaration count is 4", len(index.declared) == 4, str(index.declared))
    emit.check(
        "unique identifiers are REQ-001..003",
        index.unique == {"REQ-001", "REQ-002", "REQ-003"},
    )
    emit.check("a mid-sentence mention is not a declaration", "REQ-004" not in index)
    emit.check("a fenced line is not a declaration", "REQ-900" not in index)
    emit()

    # -- B. junit outcome classification ----------------------------------
    emit("B. junit outcome classification, all five pytest outcomes")
    outcomes = parse_junit(ALL_OUTCOMES, source="inline").counts
    for outcome in TestOutcome:
        emit(f"   {outcome.value:<22}{outcomes[outcome]}")
    emit.check(
        "each of the five outcomes is classified once",
        all(outcomes[o] == 1 for o in TestOutcome),
        str({o.value: outcomes[o] for o in TestOutcome}),
    )
    emit.check(
        "only a passed case counts as a pass",
        [o.is_pass for o in TestOutcome] == [True, False, False, False, False],
    )
    emit()

    # -- C. identifier alias forms ----------------------------------------
    emit("C. identifier alias forms accepted for one test case")
    index_x = parse_junit(ALL_OUTCOMES)
    forms = [
        "t_pass",
        "tests.test_x::t_pass",
        "tests.test_x.t_pass",
        "tests/test_x.py::t_pass",
    ]
    resolved = [len(index_x.lookup(form)) for form in forms]
    for form, count in zip(forms, resolved, strict=True):
        emit(f"   {form:<42}{count} match(es)")
    emit(f"   {'tests/test_y.py::t_pass':<42}{len(index_x.lookup('tests/test_y.py::t_pass'))}"
         " match(es)  (correctly none)")
    emit.check("all four alias forms resolve", resolved == [1, 1, 1, 1])
    emit.check(
        "an identifier naming the wrong file does not resolve",
        index_x.lookup("tests/test_y.py::t_pass") == (),
    )
    emit()

    # -- D. the committed P024 fixtures -----------------------------------
    emit("D. committed fixtures for the P024 worked example")
    p024_tests = load_junit(FIXTURES / "p024_bdot" / "junit.xml")
    p024_reqs = load_requirements(FIXTURES / "p024_bdot" / "docs" / "REQUIREMENTS.md")
    emit(f"   junit test cases        : {len(p024_tests.cases)}")
    emit(
        "   outcomes                : "
        + ", ".join(f"{o.value} {p024_tests.counts[o]}" for o in TestOutcome)
    )
    emit(f"   requirements declared   : {len(p024_reqs.unique)}")
    emit.check(
        "the P024 junit fixture holds 305 passing cases",
        len(p024_tests.cases) == 305 and p024_tests.counts[TestOutcome.PASSED] == 305,
    )
    emit.check("the P024 requirements fixture declares 13", len(p024_reqs.unique) == 13)
    emit.check(
        "REQ-021 and REQ-022 are deliberately absent",
        "REQ-021" not in p024_reqs and "REQ-022" not in p024_reqs,
    )
    emit()

    # -- E. resolution of the worked example ------------------------------
    emit("E. resolution of the worked example against those fixtures")
    analysis = load_analysis(CASES / "p024_bdot_detumble.yaml")
    report = run_checks(analysis, requirements=p024_reqs, tests=p024_tests)
    emit(f"   requirement ids cited   : {len(analysis.cited_requirements())}")
    emit(f"   test ids cited          : {len(analysis.cited_tests())}")
    emit(f"   {'figure':<32}value")
    for label in ("cited requirements resolved", "cited tests resolved", "cited tests passing"):
        ratio = dict(report.coverage.as_items())[label]
        emit(f"   {label:<32}{ratio}")
    emit()
    emit("   unresolved identifiers, named:")
    for finding in report.findings:
        if finding.code in {"HT012", "HT013", "HT014"}:
            emit(f"     {finding.code}  {finding.subjects[0]}")
    emit()
    emit.check(
        "8 of 10 cited requirements resolve",
        (report.coverage.requirement_resolution.numerator,
         report.coverage.requirement_resolution.denominator) == (8, 10),
    )
    emit.check(
        "7 of 8 cited tests resolve and all 7 passed",
        (report.coverage.test_resolution.numerator,
         report.coverage.test_resolution.denominator,
         report.coverage.test_passing.numerator) == (7, 8, 7),
    )
    emit()

    # -- F. the planted fixture, which carries a failing test -------------
    emit("F. planted fixture: a cited test that did not pass")
    planted_tests = load_junit(FIXTURES / "planted" / "junit.xml")
    planted = run_checks(
        load_analysis(CASES / "planted_gaps.yaml"),
        requirements=load_requirements(FIXTURES / "planted" / "docs" / "REQUIREMENTS.md"),
        tests=planted_tests,
    )
    emit(f"   junit test cases        : {len(planted_tests.cases)}")
    emit(
        "   outcomes                : "
        + ", ".join(f"{o.value} {planted_tests.counts[o]}" for o in TestOutcome)
    )
    for label in ("cited requirements resolved", "cited tests resolved", "cited tests passing"):
        ratio = dict(planted.coverage.as_items())[label]
        emit(f"   {label:<32}{ratio}")
    emit.check(
        "the failing cited test raises HT014",
        any(f.code == "HT014" for f in planted.findings),
    )
    emit.check(
        "cited test resolution is 2/3 and passing is 1/3",
        (planted.coverage.test_resolution.numerator,
         planted.coverage.test_resolution.denominator,
         planted.coverage.test_passing.numerator) == (2, 3, 1),
    )
    emit()

    # -- G. withholding the sources leaves the ratios undefined -----------
    emit("G. with no requirements document and no test report")
    bare = run_checks(analysis)
    for label in ("cited requirements resolved", "cited tests resolved", "cited tests passing"):
        ratio = dict(bare.coverage.as_items())[label]
        emit(f"   {label:<32}{ratio}")
    emit(f"   findings                : {len(bare.findings)} (against "
         f"{len(report.findings)} with the sources supplied)")
    emit.check(
        "withheld sources give undefined ratios, not 100 %",
        all(
            not dict(bare.coverage.as_items())[label].defined
            for label in (
                "cited requirements resolved", "cited tests resolved", "cited tests passing"
            )
        ),
    )
    emit.check(
        "supplying the sources can only add findings",
        set(bare.findings).issubset(set(report.findings)),
    )
    return emit.finish()


if __name__ == "__main__":
    raise SystemExit(main())
