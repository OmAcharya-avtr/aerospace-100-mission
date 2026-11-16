"""Numeric-claim extraction, skeleton alignment and both-bounds comparison."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from seedaudit import numeric


def test_tokens_are_found_with_their_text_value_and_position() -> None:
    tokens = numeric.numeric_tokens("mean 1.234567 over 10 cells", 7)
    assert [t.text for t in tokens] == ["1.234567", "10"]
    assert [t.value for t in tokens] == [1.234567, 10.0]
    assert [(t.line, t.index) for t in tokens] == [(7, 0), (7, 1)]
    assert numeric.numeric_tokens("") == []


def test_exponential_and_signed_and_bare_point_forms_are_single_tokens() -> None:
    assert [t.text for t in numeric.numeric_tokens("worst 8.114e-05 relative")] == ["8.114e-05"]
    assert [t.text for t in numeric.numeric_tokens("x 2.5E3")] == ["2.5E3"]
    assert [t.value for t in numeric.numeric_tokens("delta -0.5 and +2")] == [-0.5, 2.0]
    assert [t.text for t in numeric.numeric_tokens("x .25")] == [".25"]
    assert [t.text for t in numeric.numeric_tokens("x 25. y")] == ["25."]


def test_digits_inside_an_identifier_or_a_version_are_not_claims() -> None:
    # w64_d050 is a cell name and P100 is a product id. Decomposing either
    # would manufacture claims nobody made; so would reading 2.5.3 as numbers.
    assert numeric.numeric_tokens("cell w64_d050 in P100") == []
    assert [t.text for t in numeric.numeric_tokens("numpy 2.5.3 installed")] == []
    assert [t.text for t in numeric.numeric_tokens("pytest 9.1 here")] == ["9.1"]


def test_the_skeleton_is_the_line_with_its_quantities_removed() -> None:
    assert numeric.skeleton("a 1 b 2.5") == f"a {numeric.PLACEHOLDER} b {numeric.PLACEHOLDER}"
    assert numeric.skeleton("value 1.000000") == numeric.skeleton("value 2.000000")
    assert numeric.skeleton("order a b") != numeric.skeleton("order b a")


def test_extraction_rejects_non_strings() -> None:
    with pytest.raises(TypeError, match="must be a str"):
        numeric.numeric_tokens(42)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a str"):
        numeric.skeleton(None)  # type: ignore[arg-type]


def test_both_tolerance_bounds_default_to_zero() -> None:
    # A committed number that changes in its last printed digit HAS moved, and
    # a default tolerance would be the product softening its own findings.
    assert numeric.changed(1.000000, 1.000001)
    assert not numeric.changed(1.0, 1.0)


def test_the_looser_of_the_two_tolerance_bounds_wins() -> None:
    assert not numeric.changed(1.0, 1.0005, abs_tol=1e-3)
    assert numeric.changed(1.0, 1.002, abs_tol=1e-3)
    assert not numeric.changed(1000.0, 1000.5, rel_tol=1e-3)
    assert numeric.changed(1000.0, 1002.0, rel_tol=1e-3)
    assert not numeric.changed(1000.0, 1000.5, rel_tol=1e-3, abs_tol=1e-9)


def test_nan_and_infinity_are_handled_explicitly() -> None:
    # Every comparison with nan is false, so without this a nan pair would
    # silently report "unchanged".
    assert not numeric.changed(math.nan, math.nan)
    assert numeric.changed(math.nan, 1.0)
    assert numeric.changed(1.0, math.nan)
    assert not numeric.changed(math.inf, math.inf)
    assert numeric.changed(math.inf, -math.inf)
    with pytest.raises(ValueError, match="non-negative"):
        numeric.changed(1.0, 1.0, rel_tol=-1e-9)


def test_comparing_identical_texts_reports_nothing_moved() -> None:
    text = "value 1.000000\nconstant 42\n"
    result = numeric.compare(text, text)
    assert result.identical
    assert result.moved_count == 0
    assert result.reference_tokens == 2
    assert result.compared_tokens == 2
    assert not result.structural_change
    empty = numeric.compare("", "")
    assert empty.identical and empty.reference_tokens == 0


def test_a_moved_number_is_reported_with_both_values_and_both_changes() -> None:
    result = numeric.compare("value 1.000000\n", "value 2.000000\n")
    assert result.moved_count == len(result.moved) == 1
    moved = result.moved[0]
    assert (moved.before, moved.after) == ("1.000000", "2.000000")
    assert moved.absolute_change == pytest.approx(1.0, rel=1e-15, abs=0.0)
    assert moved.relative_change == pytest.approx(1.0, rel=1e-15, abs=0.0)
    assert numeric.PLACEHOLDER in moved.skeleton


def test_a_relative_change_from_zero_is_infinite_rather_than_a_division_error() -> None:
    assert numeric.compare("residual 0.0\n", "residual 1e-9\n").moved[0].relative_change == math.inf


def test_structural_change_is_counted_separately_from_moved_numbers() -> None:
    # The iteration-order case: no number moves, and a numeric-only comparison
    # would report agreement. The structural count is what catches it.
    result = numeric.compare("order a b\nvalue 1.0\n", "order b a\nvalue 1.0\n")
    assert result.moved_count == 0
    assert result.structural_change
    assert result.structural_lines_reference == 1
    assert result.structural_lines_candidate == 1


def test_unreachable_tokens_are_reported_rather_than_counted_as_agreeing() -> None:
    result = numeric.compare("alpha 1.0\n", "bravo 1.0\n")
    assert result.unreachable_tokens == 1
    assert result.compared_tokens == 0


def test_added_and_removed_lines_are_counted_on_the_right_side() -> None:
    added = numeric.compare("a 1\n", "a 1\nb 2\n")
    assert (added.structural_lines_reference, added.structural_lines_candidate) == (0, 1)
    removed = numeric.compare("a 1\nb 2\n", "a 1\n")
    assert (removed.structural_lines_reference, removed.structural_lines_candidate) == (1, 0)


def test_alignment_survives_an_inserted_line() -> None:
    result = numeric.compare("a 1.0\nc 3.0\n", "a 1.0\nb 2.0\nc 4.0\n")
    assert result.moved_count == 1
    assert (result.moved[0].before, result.moved[0].after) == ("3.0", "4.0")


def test_a_sub_precision_change_is_caught_at_full_precision_and_missed_when_rounded() -> None:
    # Published as a limitation, not hidden: a change below printed precision
    # is invisible at any repeat count, which is why the corpus carries one.
    caught = numeric.compare(
        "rounded 0.123457\nfull 0.12345678901230\n", "rounded 0.123457\nfull 0.12345678901240\n"
    )
    assert caught.moved_count == 1
    assert caught.moved[0].line == 1
    assert numeric.compare("rounded 0.123457\n", "rounded 0.123457\n").moved_count == 0


def test_a_stated_tolerance_can_suppress_a_small_move() -> None:
    result = numeric.compare("value 1.000000\n", "value 1.000001\n", rel_tol=1e-5, abs_tol=0.0)
    assert result.moved_count == 0


def test_compare_rejects_non_string_arguments() -> None:
    with pytest.raises(TypeError, match="must both be str"):
        numeric.compare("a", 1)  # type: ignore[arg-type]


@given(value=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
@settings(max_examples=200, deadline=None)
def test_property_a_token_round_trips_through_its_own_text(value: float) -> None:
    tokens = numeric.numeric_tokens(f"value {value:.17g}")
    assert len(tokens) == 1
    assert tokens[0].value == pytest.approx(value, rel=1e-15, abs=1e-300)


@given(
    left=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    right=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200, deadline=None)
def test_property_changed_is_symmetric_at_zero_tolerance(left: float, right: float) -> None:
    assert numeric.changed(left, right) == numeric.changed(right, left)


@given(text=st.text(max_size=120))
@settings(max_examples=200, deadline=None)
def test_property_comparing_a_text_against_itself_finds_nothing(text: str) -> None:
    result = numeric.compare(text, text)
    assert result.moved_count == 0
    assert not result.structural_change


def test_regression_a_one_ulp_difference_is_not_rounded_away() -> None:
    # Pinned literal: an earlier product of this mission hand-rendered
    # 0.500000000000000 for a value that is in fact 0.49999999999999983.
    result = numeric.compare("x 0.500000000000000\n", "x 0.49999999999999983\n")
    assert result.moved_count == 1
    assert result.moved[0].absolute_change == pytest.approx(1.7e-16, rel=0.2, abs=0.0)


def test_the_skeleton_placeholder_is_printable_ascii() -> None:
    # The first version used a NUL-delimited placeholder, which put NUL bytes
    # into every committed validation output that quoted a skeleton and made
    # the evidence a binary file. Committed evidence has to be readable text.
    assert numeric.PLACEHOLDER.isprintable()
    assert numeric.PLACEHOLDER.isascii()
    assert "\x00" not in numeric.PLACEHOLDER
    rendered = numeric.skeleton("mean 1.5 of 10")
    assert rendered.isprintable()
    assert rendered == f"mean {numeric.PLACEHOLDER} of {numeric.PLACEHOLDER}"


def test_a_line_containing_the_placeholder_literally_is_a_declared_collision() -> None:
    # Declared rather than hidden. The collision affects line alignment only;
    # it never changes whether a numeric token is read as a number.
    collision = f"status {numeric.PLACEHOLDER} ok"
    assert numeric.skeleton(collision) == collision
    assert numeric.skeleton("status 7 ok") == collision
    assert numeric.numeric_tokens(collision) == []
