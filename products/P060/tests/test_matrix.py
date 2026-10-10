"""Mapping and coverage arithmetic, with the expected ratios hand-computed."""

from __future__ import annotations

import pytest

from traceaudit import Outcome, TestCase, TestReport, build_matrix, parse_requirements_text

DOC = "## REQ-001 A\n## REQ-002 B\n## REQ-003 C\n## REQ-004 D\n"


def _case(name, outcome, claims):
    return TestCase(
        node_id=f"tests/test_x.py::{name}",
        classname="tests.test_x",
        name=name,
        outcome=outcome,
        file="tests/test_x.py",
        claims=tuple(claims),
    )


@pytest.fixture
def matrix():
    doc = parse_requirements_text(DOC, source="s.md")
    report = TestReport(
        cases=(
            _case("test_a", Outcome.PASSED, ["REQ-001"]),
            _case("test_b", Outcome.FAILED, ["REQ-002"]),
            _case("test_c", Outcome.SKIPPED, ["REQ-003"]),
            _case("test_d", Outcome.PASSED, ["REQ-009"]),
        ),
        source="j.xml",
        kind="junit-xml",
    )
    return build_matrix(doc, report)


@pytest.mark.verifies("REQ-010")
def test_requirement_to_tests_covers_every_declared_id(matrix):
    assert list(matrix.requirement_to_tests) == ["REQ-001", "REQ-002", "REQ-003", "REQ-004"]
    assert matrix.requirement_to_tests["REQ-001"] == ("tests/test_x.py::test_a",)
    assert matrix.requirement_to_tests["REQ-004"] == ()


@pytest.mark.verifies("REQ-010")
def test_test_to_requirements_is_the_other_direction(matrix):
    assert matrix.test_to_requirements["tests/test_x.py::test_b"] == ("REQ-002",)
    assert matrix.test_to_requirements["tests/test_x.py::test_d"] == ("REQ-009",)


@pytest.mark.verifies("REQ-010")
def test_unknown_claims_are_separated_from_the_mapping(matrix):
    assert matrix.unknown_claims == {"REQ-009": ("tests/test_x.py::test_d",)}
    assert "REQ-009" not in matrix.requirement_to_tests


@pytest.mark.verifies("REQ-010")
def test_traced_and_untraced_partition_the_requirements(matrix):
    assert set(matrix.traced()) == {"REQ-001", "REQ-002", "REQ-003"}
    assert matrix.untraced() == ("REQ-004",)
    assert set(matrix.traced()) | set(matrix.untraced()) == set(matrix.requirement_to_tests)
    assert not set(matrix.traced()) & set(matrix.untraced())


@pytest.mark.verifies("REQ-011")
def test_three_coverage_figures_with_hand_computed_values(matrix):
    # Hand arithmetic, denominator 4 unique declared ids:
    #   nominal  : REQ-001, REQ-002, REQ-003 traced -> 3/4 = 75 %
    #   executed : REQ-001 passed, REQ-002 failed   -> 2/4 = 50 %
    #   passing  : REQ-001 only                     -> 1/4 = 25 %
    figures = {f.label: f for f in matrix.coverage()}
    assert figures["nominal coverage"].numerator == 3
    assert figures["nominal coverage"].denominator == 4
    assert figures["nominal coverage"].percent == pytest.approx(75.0)
    assert figures["executed coverage"].numerator == 2
    assert figures["executed coverage"].percent == pytest.approx(50.0)
    assert figures["passing coverage"].numerator == 1
    assert figures["passing coverage"].percent == pytest.approx(25.0)


@pytest.mark.verifies("REQ-011")
def test_every_coverage_figure_carries_its_definition(matrix):
    for figure in matrix.coverage():
        assert figure.definition
        assert f"{figure.numerator}/{figure.denominator}" in figure.render()


@pytest.mark.verifies("REQ-011")
def test_zero_denominator_gives_an_undefined_percentage():
    doc = parse_requirements_text("", source="s.md")
    report = TestReport(cases=(), source="j.xml", kind="junit-xml")
    matrix = build_matrix(doc, report)
    figure = matrix.coverage()[0]
    assert figure.denominator == 0
    assert figure.percent is None
    assert "undefined" in figure.render()


@pytest.mark.verifies("REQ-012")
def test_coverage_ordering_holds_on_the_fixture(matrix):
    figures = {f.label: f.numerator for f in matrix.coverage()}
    assert figures["nominal coverage"] >= figures["executed coverage"]
    assert figures["executed coverage"] >= figures["passing coverage"]


@pytest.mark.verifies("REQ-016")
def test_illusory_traced_lists_only_skip_and_xfail_traces(matrix):
    assert matrix.illusory_traced() == ("REQ-003",)


@pytest.mark.verifies("REQ-016")
def test_a_requirement_with_one_real_and_one_skipped_test_is_not_illusory():
    doc = parse_requirements_text("## REQ-001 A\n", source="s.md")
    report = TestReport(
        cases=(
            _case("test_a", Outcome.SKIPPED, ["REQ-001"]),
            _case("test_b", Outcome.PASSED, ["REQ-001"]),
        ),
        source="j.xml",
        kind="junit-xml",
    )
    matrix = build_matrix(doc, report)
    assert matrix.illusory_traced() == ()
    assert matrix.passing_traced() == ("REQ-001",)


@pytest.mark.verifies("REQ-017")
def test_failing_only_traced_lists_failures_and_errors(matrix):
    assert matrix.failing_only_traced() == ("REQ-002",)


@pytest.mark.verifies("REQ-017")
def test_an_erroring_test_counts_as_failing_only():
    doc = parse_requirements_text("## REQ-001 A\n", source="s.md")
    report = TestReport(
        cases=(_case("test_a", Outcome.ERRORED, ["REQ-001"]),),
        source="j.xml",
        kind="junit-xml",
    )
    matrix = build_matrix(doc, report)
    assert matrix.failing_only_traced() == ("REQ-001",)
    assert matrix.executed_traced() == ("REQ-001",)
    assert matrix.passing_traced() == ()


@pytest.mark.verifies("REQ-008")
def test_collection_report_marks_outcomes_as_unknown():
    doc = parse_requirements_text(DOC, source="s.md")
    report = TestReport(
        cases=(_case("test_a", Outcome.UNKNOWN, ["REQ-001"]),),
        source="c.txt",
        kind="collect-only",
    )
    matrix = build_matrix(doc, report)
    assert matrix.outcomes_known is False
    assert [f.label for f in matrix.coverage()] == ["nominal coverage"]


@pytest.mark.verifies("REQ-010")
def test_a_duplicate_declaration_collapses_to_one_mapping_key():
    doc = parse_requirements_text("## REQ-001 A\n## REQ-001 A again\n", source="s.md")
    report = TestReport(
        cases=(_case("test_a", Outcome.PASSED, ["REQ-001"]),),
        source="j.xml",
        kind="junit-xml",
    )
    matrix = build_matrix(doc, report)
    assert matrix.n_requirements == 1
    assert matrix.coverage()[0].denominator == 1


@pytest.mark.verifies("REQ-010")
def test_the_same_test_claiming_an_id_twice_is_listed_once():
    doc = parse_requirements_text("## REQ-001 A\n", source="s.md")
    report = TestReport(
        cases=(_case("test_a", Outcome.PASSED, ["REQ-001", "REQ-001"]),),
        source="j.xml",
        kind="junit-xml",
    )
    matrix = build_matrix(doc, report)
    assert matrix.requirement_to_tests["REQ-001"] == ("tests/test_x.py::test_a",)
