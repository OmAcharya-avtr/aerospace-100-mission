"""Property-based tests for the invariants that hold for every input.

The identities exercised here are the ones a reviewer would actually check:
the mapping partitions the requirement set, the three coverage figures form a
monotone chain, and the parser recovers exactly the ids it was given.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from traceaudit import (
    Outcome,
    TestCase,
    TestReport,
    TraceConfig,
    audit,
    build_matrix,
    parse_requirements_text,
    render_json,
    render_text,
)

NO_HEURISTIC = TraceConfig(heuristic_assertions=False)

req_numbers = st.integers(min_value=0, max_value=999)
outcomes = st.sampled_from(
    [Outcome.PASSED, Outcome.FAILED, Outcome.ERRORED, Outcome.SKIPPED, Outcome.XFAILED]
)


def _doc_from(numbers):
    text = "".join(f"## REQ-{n:03d} Requirement {n}\n" for n in numbers)
    return parse_requirements_text(text, source="s.md")


def _report_from(claims_per_test):
    cases = []
    for i, (outcome, claims) in enumerate(claims_per_test):
        cases.append(
            TestCase(
                node_id=f"tests/test_g.py::test_{i}",
                classname="tests.test_g",
                name=f"test_{i}",
                outcome=outcome,
                file="tests/test_g.py",
                claims=tuple(f"REQ-{n:03d}" for n in claims),
            )
        )
    return TestReport(cases=tuple(cases), source="j.xml", kind="junit-xml")


@pytest.mark.verifies("REQ-001")
@settings(max_examples=120, deadline=None)
@given(st.lists(req_numbers, min_size=0, max_size=12, unique=True))
def test_parser_recovers_exactly_the_declared_ids(numbers):
    doc = _doc_from(numbers)
    assert list(doc.ids) == [f"REQ-{n:03d}" for n in numbers]
    assert len(doc.declarations) == len(numbers)


@pytest.mark.verifies("REQ-004")
@settings(max_examples=80, deadline=None)
@given(st.lists(req_numbers, min_size=1, max_size=8, unique=True),
       st.integers(min_value=2, max_value=4))
def test_repeating_every_declaration_k_times_gives_k_grouped_declarations(numbers, k):
    doc = _doc_from(list(numbers) * k)
    assert len(doc.declarations) == k * len(numbers)
    assert list(doc.ids) == [f"REQ-{n:03d}" for n in numbers]
    assert all(len(group) == k for group in doc.duplicates.values())
    assert len(doc.duplicates) == len(numbers)


@pytest.mark.verifies("REQ-010")
@settings(max_examples=120, deadline=None)
@given(
    st.lists(req_numbers, min_size=0, max_size=10, unique=True),
    st.lists(st.tuples(outcomes, st.lists(req_numbers, max_size=4, unique=True)),
             max_size=8),
)
def test_traced_and_untraced_partition_the_declared_set(numbers, tests):
    matrix = build_matrix(_doc_from(numbers), _report_from(tests))
    traced, untraced = set(matrix.traced()), set(matrix.untraced())
    assert traced | untraced == set(matrix.requirement_to_tests)
    assert traced & untraced == set()
    assert len(traced) + len(untraced) == matrix.n_requirements


@pytest.mark.verifies("REQ-012")
@settings(max_examples=200, deadline=None)
@given(
    st.lists(req_numbers, min_size=1, max_size=10, unique=True),
    st.lists(st.tuples(outcomes, st.lists(req_numbers, max_size=4, unique=True)),
             max_size=8),
)
def test_nominal_at_least_executed_at_least_passing(numbers, tests):
    matrix = build_matrix(_doc_from(numbers), _report_from(tests))
    figures = {f.label: f.numerator for f in matrix.coverage()}
    assert figures["nominal coverage"] >= figures["executed coverage"]
    assert figures["executed coverage"] >= figures["passing coverage"]


@pytest.mark.verifies("REQ-011")
@settings(max_examples=150, deadline=None)
@given(
    st.lists(req_numbers, min_size=0, max_size=10, unique=True),
    st.lists(st.tuples(outcomes, st.lists(req_numbers, max_size=4, unique=True)),
             max_size=8),
)
def test_every_coverage_figure_is_a_fraction_between_zero_and_one_hundred(numbers, tests):
    matrix = build_matrix(_doc_from(numbers), _report_from(tests))
    for figure in matrix.coverage():
        assert 0 <= figure.numerator <= figure.denominator == matrix.n_requirements
        if figure.denominator:
            assert 0.0 <= figure.percent <= 100.0
        else:
            assert figure.percent is None


@pytest.mark.verifies("REQ-010")
@settings(max_examples=120, deadline=None)
@given(
    st.lists(req_numbers, min_size=0, max_size=10, unique=True),
    st.lists(st.tuples(outcomes, st.lists(req_numbers, max_size=4, unique=True)),
             max_size=8),
)
def test_every_claim_is_either_mapped_or_reported_as_unknown(numbers, tests):
    doc, report = _doc_from(numbers), _report_from(tests)
    matrix = build_matrix(doc, report)
    declared = set(matrix.requirement_to_tests)
    for case in report.cases:
        for claim in case.claims:
            if claim in declared:
                assert case.node_id in matrix.requirement_to_tests[claim]
            else:
                assert case.node_id in matrix.unknown_claims[claim]


@pytest.mark.verifies("REQ-013")
@settings(max_examples=120, deadline=None)
@given(
    st.lists(req_numbers, min_size=0, max_size=10, unique=True),
    st.lists(st.tuples(outcomes, st.lists(req_numbers, max_size=4, unique=True)),
             max_size=8),
)
def test_exit_code_is_one_exactly_when_a_blocking_finding_exists(numbers, tests):
    result = audit(_doc_from(numbers), _report_from(tests), config=NO_HEURISTIC)
    assert result.exit_code == (1 if result.blocking else 0)
    assert len(result.blocking) <= len(result.findings)


@pytest.mark.verifies("REQ-013")
@settings(max_examples=100, deadline=None)
@given(
    st.lists(req_numbers, min_size=0, max_size=8, unique=True),
    st.lists(st.tuples(outcomes, st.lists(req_numbers, max_size=3, unique=True)),
             max_size=6),
)
def test_counts_by_code_sums_to_the_finding_total(numbers, tests):
    result = audit(_doc_from(numbers), _report_from(tests), config=NO_HEURISTIC)
    assert sum(result.counts_by_code().values()) == len(result.findings)


@pytest.mark.verifies("REQ-023")
@settings(max_examples=60, deadline=None)
@given(
    st.lists(req_numbers, min_size=0, max_size=8, unique=True),
    st.lists(st.tuples(outcomes, st.lists(req_numbers, max_size=3, unique=True)),
             max_size=6),
)
def test_json_and_text_rendering_never_raise(numbers, tests):
    import json

    result = audit(_doc_from(numbers), _report_from(tests), config=NO_HEURISTIC)
    data = json.loads(render_json(result))
    assert data["exit_code"] == result.exit_code
    assert "findings" in render_text(result, show_matrix=True)


@pytest.mark.verifies("REQ-016")
@settings(max_examples=150, deadline=None)
@given(
    st.lists(req_numbers, min_size=1, max_size=8, unique=True),
    st.lists(st.tuples(outcomes, st.lists(req_numbers, max_size=3, unique=True)),
             min_size=1, max_size=6),
)
def test_illusory_and_failing_only_are_disjoint_subsets_of_traced(numbers, tests):
    matrix = build_matrix(_doc_from(numbers), _report_from(tests))
    illusory, failing = set(matrix.illusory_traced()), set(matrix.failing_only_traced())
    traced = set(matrix.traced())
    assert illusory <= traced
    assert failing <= traced
    assert illusory & failing == set()
    assert illusory & set(matrix.executed_traced()) == set()
